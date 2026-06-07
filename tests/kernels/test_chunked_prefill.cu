// test_chunked_prefill.cu
// ---------------------------------------------------------------------------
// Stage-0 spike for the multi-turn agentic rollout (docs/AGENTIC_ROLLOUT_SPEC.md).
//
// QUESTION: can we "continue" a prefill — feed only the NEW tokens of a turn and
// have them attend over the already-cached prefix — with today's kernels?
//
// A sequence of N tokens is split into a cached prefix [0, n0) and a new chunk
// [n0, N).  Correct continued prefill means: for each new-chunk query at absolute
// position p, attention runs causally over keys [0, p]  (prefix + new-so-far).
//
// This test needs NO model weights — it is a pure attention-kernel property.
//
//   A. SANITY  — full FA2 prefill over all N tokens matches the CPU reference.
//   B. GAP     — feeding only the new chunk to launch_flash_attention_prefill
//                (what qwen3_prefill does today for a continued chunk) attends
//                ONLY within the chunk → diverges hard from the true answer.
//                This is the capability that does not exist yet.
//   C. PATH    — the paged decode kernel, walked one query at a time over the
//                full cache, DOES reproduce the true causal answer.  So the fix
//                is its S_q>1 sibling: a prefill that reads the paged KV cache.
//
// Exit 0 iff reality matches that diagnosis (A pass, B diverges, C pass).
// When a cache-reading prefill kernel lands, swap B's call for it and it flips
// to PASS — this file becomes its regression test.
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/attention.cuh"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

static unsigned int lcg_state = 1234u;
static float lcg_randf() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(lcg_state >> 1) / (float)0x7fffffffu) * 2.0f - 1.0f;
}

// CPU reference: naive causal GQA attention.  Q,K,V,O layout [S, H, D] (B=1).
static void ref_causal_attention(
    float* O, const float* Q, const float* K, const float* V,
    int S, int H_q, int H_kv, int D)
{
    const float scale = 1.0f / sqrtf((float)D);
    std::vector<float> scores(S);
    for (int h = 0; h < H_q; h++) {
        const int hkv = h * H_kv / H_q;
        for (int i = 0; i < S; i++) {
            const float* qi = Q + (i * H_q + h) * D;
            float max_s = -1e30f;
            for (int j = 0; j <= i; j++) {
                const float* kj = K + (j * H_kv + hkv) * D;
                float dot = 0.0f;
                for (int d = 0; d < D; d++) dot += qi[d] * kj[d];
                scores[j] = dot * scale;
                if (scores[j] > max_s) max_s = scores[j];
            }
            float sum_e = 0.0f;
            for (int j = 0; j <= i; j++) { scores[j] = expf(scores[j] - max_s); sum_e += scores[j]; }
            float* oi = O + (i * H_q + h) * D;
            for (int d = 0; d < D; d++) {
                float acc = 0.0f;
                for (int j = 0; j <= i; j++)
                    acc += scores[j] * V[(j * H_kv + hkv) * D + d];
                oi[d] = acc / sum_e;
            }
        }
    }
}

static float max_abs_diff(const half* got, const float* ref, long n) {
    float m = 0.0f;
    for (long i = 0; i < n; i++) {
        float d = fabsf(__half2float(got[i]) - ref[i]);
        if (d > m) m = d;
    }
    return m;
}

// ---------------------------------------------------------------------------
// One scenario: N tokens, split n0 (cached) + n1 (new chunk).
// ---------------------------------------------------------------------------
static bool run_scenario(const char* name, int n0, int n1,
                         int H_q, int H_kv, int D = 128,
                         int BLOCK_SIZE = 16, float tol = 5e-3f)
{
    const int N = n0 + n1;
    const long NQ = (long)N * H_q  * D;
    const long NK = (long)N * H_kv * D;

    // Random FP16 inputs (round-tripped through FP16 for the reference).
    std::vector<half> hQ(NQ), hK(NK), hV(NK);
    for (long i = 0; i < NQ; i++) hQ[i] = __float2half(lcg_randf() * 0.5f);
    for (long i = 0; i < NK; i++) hK[i] = __float2half(lcg_randf() * 0.5f);
    for (long i = 0; i < NK; i++) hV[i] = __float2half(lcg_randf() * 0.5f);

    std::vector<float> fQ(NQ), fK(NK), fV(NK);
    for (long i = 0; i < NQ; i++) fQ[i] = __half2float(hQ[i]);
    for (long i = 0; i < NK; i++) fK[i] = __half2float(hK[i]);
    for (long i = 0; i < NK; i++) fV[i] = __half2float(hV[i]);

    // Golden: full causal attention over all N tokens.
    std::vector<float> gold(NQ);
    ref_causal_attention(gold.data(), fQ.data(), fK.data(), fV.data(), N, H_q, H_kv, D);
    // Rows for the NEW chunk only: [n0, N) → [n1, H_q, D].
    const float* gold_chunk = gold.data() + (long)n0 * H_q * D;
    const long   NQ_chunk   = (long)n1 * H_q * D;

    printf("\n== %s  (n0=%d cached, n1=%d new, N=%d, H_q=%d H_kv=%d) ==\n",
           name, n0, n1, N, H_q, H_kv);

    bool ok = true;

    // ---- A. SANITY: full prefill over all N matches golden ------------------
    {
        half *dQ, *dK, *dV, *dO;
        CUDA_CHECK(cudaMalloc(&dQ, NQ * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dK, NK * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dV, NK * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dO, NQ * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), NQ * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK, hK.data(), NK * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV, hV.data(), NK * sizeof(half), cudaMemcpyHostToDevice));
        launch_flash_attention_prefill(dQ, dK, dV, dO, 1, N, H_q, H_kv, D);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<half> hO(NQ);
        CUDA_CHECK(cudaMemcpy(hO.data(), dO, NQ * sizeof(half), cudaMemcpyDeviceToHost));
        float err = max_abs_diff(hO.data(), gold.data(), NQ);
        bool pass = err < tol;
        printf("  A. full prefill vs golden                  max_err=%.5f  [%s]\n",
               err, pass ? "PASS" : "FAIL");
        ok &= pass;
        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    }

    // ---- B. GAP: chunk-only prefill (today's continued-prefill path) --------
    // Feed ONLY the new chunk's Q/K/V as a fresh S=n1 prefill. The chunk attends
    // within itself (positions 0..n1-1), blind to the n0 cached tokens.
    {
        const long cq = (long)n1 * H_q  * D;
        const long ck = (long)n1 * H_kv * D;
        half *dQ, *dK, *dV, *dO;
        CUDA_CHECK(cudaMalloc(&dQ, cq * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dK, ck * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dV, ck * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dO, cq * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(dQ, hQ.data() + (long)n0 * H_q  * D, cq * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK, hK.data() + (long)n0 * H_kv * D, ck * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV, hV.data() + (long)n0 * H_kv * D, ck * sizeof(half), cudaMemcpyHostToDevice));
        launch_flash_attention_prefill(dQ, dK, dV, dO, 1, n1, H_q, H_kv, D);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<half> hO(cq);
        CUDA_CHECK(cudaMemcpy(hO.data(), dO, cq * sizeof(half), cudaMemcpyDeviceToHost));
        float err = max_abs_diff(hO.data(), gold_chunk, NQ_chunk);
        // We EXPECT this to diverge (cached prefix ignored). n0>0 ⇒ must diverge.
        bool diverges = err > 10.0f * tol;
        bool as_expected = (n0 == 0) ? (err < tol) : diverges;
        printf("  B. chunk-only prefill vs golden            max_err=%.5f  [%s]\n",
               err, as_expected
                   ? (n0 == 0 ? "PASS (no prefix)" : "GAP CONFIRMED")
                   : "UNEXPECTED");
        ok &= as_expected;
        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    }

    // ---- C. PATH: paged decode, one query at a time, reproduces golden ------
    // Build a paged cache holding all N tokens; for each new-chunk position p,
    // set seq_len = p+1 and decode q=Q[p] → attends causally over [0, p].
    {
        const int blocks_per_seq = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const long cache_elems = (long)blocks_per_seq * H_kv * BLOCK_SIZE * D;
        std::vector<half> hKc(cache_elems, __float2half(0.0f));
        std::vector<half> hVc(cache_elems, __float2half(0.0f));
        for (int t = 0; t < N; t++) {
            int lb = t / BLOCK_SIZE, off = t % BLOCK_SIZE;
            for (int h = 0; h < H_kv; h++) {
                long base = ((long)lb * H_kv + h) * BLOCK_SIZE * D + (long)off * D;
                for (int d = 0; d < D; d++) {
                    hKc[base + d] = hK[(t * H_kv + h) * D + d];
                    hVc[base + d] = hV[(t * H_kv + h) * D + d];
                }
            }
        }
        std::vector<int> h_bt(blocks_per_seq);
        for (int b = 0; b < blocks_per_seq; b++) h_bt[b] = b;  // identity

        half *dKc, *dVc, *dq, *dout;
        int  *dbt, *dsl;
        CUDA_CHECK(cudaMalloc(&dKc, cache_elems * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dVc, cache_elems * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dq,  (long)H_q * D * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dout,(long)H_q * D * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dbt, blocks_per_seq * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dsl, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(dKc, hKc.data(), cache_elems * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dVc, hVc.data(), cache_elems * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dbt, h_bt.data(), blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));

        std::vector<half> O_paged(NQ_chunk);
        for (int p = n0; p < N; p++) {
            int sl = p + 1;
            CUDA_CHECK(cudaMemcpy(dq, hQ.data() + (long)p * H_q * D,
                                  (long)H_q * D * sizeof(half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dsl, &sl, sizeof(int), cudaMemcpyHostToDevice));
            launch_paged_attention_decode(dq, dKc, dVc, dout, dbt, dsl,
                                          1, H_q, H_kv, D, blocks_per_seq, BLOCK_SIZE);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(O_paged.data() + (long)(p - n0) * H_q * D, dout,
                                  (long)H_q * D * sizeof(half), cudaMemcpyDeviceToHost));
        }
        float err = max_abs_diff(O_paged.data(), gold_chunk, NQ_chunk);
        bool pass = err < tol;
        printf("  C. paged decode (per-token) vs golden      max_err=%.5f  [%s]\n",
               err, pass ? "PASS" : "FAIL");
        ok &= pass;
        cudaFree(dKc); cudaFree(dVc); cudaFree(dq); cudaFree(dout); cudaFree(dbt); cudaFree(dsl);
    }

    // ---- D. FIX: launch_flash_attention_prefill_paged in ONE call -----------
    // The new cache-reading prefill kernel: feed only the n1 new-chunk queries
    // (q_abs_pos = n0..N-1) and have them attend over the full paged cache.
    {
        const int blocks_per_seq = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const long cache_elems = (long)blocks_per_seq * H_kv * BLOCK_SIZE * D;
        std::vector<half> hKc(cache_elems, __float2half(0.0f));
        std::vector<half> hVc(cache_elems, __float2half(0.0f));
        for (int t = 0; t < N; t++) {
            int lb = t / BLOCK_SIZE, off = t % BLOCK_SIZE;
            for (int h = 0; h < H_kv; h++) {
                long base = ((long)lb * H_kv + h) * BLOCK_SIZE * D + (long)off * D;
                for (int d = 0; d < D; d++) {
                    hKc[base + d] = hK[(t * H_kv + h) * D + d];
                    hVc[base + d] = hV[(t * H_kv + h) * D + d];
                }
            }
        }
        std::vector<int> h_bt(blocks_per_seq);
        for (int b = 0; b < blocks_per_seq; b++) h_bt[b] = b;        // identity
        std::vector<int> h_qseq(n1, 0);                              // all seq 0
        std::vector<int> h_qpos(n1);
        for (int i = 0; i < n1; i++) h_qpos[i] = n0 + i;             // abs positions

        // New-chunk queries laid out densely [n1, H_q, D].
        std::vector<half> hQc(NQ_chunk);
        for (long i = 0; i < NQ_chunk; i++) hQc[i] = hQ[(long)n0 * H_q * D + i];

        half *dKc, *dVc, *dQc, *dO;
        int  *dbt, *dqs, *dqp;
        CUDA_CHECK(cudaMalloc(&dKc, cache_elems * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dVc, cache_elems * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dQc, NQ_chunk * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dO,  NQ_chunk * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dbt, blocks_per_seq * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dqs, n1 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dqp, n1 * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(dKc, hKc.data(), cache_elems * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dVc, hVc.data(), cache_elems * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dQc, hQc.data(), NQ_chunk * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dbt, h_bt.data(), blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dqs, h_qseq.data(), n1 * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dqp, h_qpos.data(), n1 * sizeof(int), cudaMemcpyHostToDevice));

        launch_flash_attention_prefill_paged(dQc, dKc, dVc, dO, dbt, dqs, dqp,
                                             n1, H_q, H_kv, D, blocks_per_seq, BLOCK_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<half> hO(NQ_chunk);
        CUDA_CHECK(cudaMemcpy(hO.data(), dO, NQ_chunk * sizeof(half), cudaMemcpyDeviceToHost));
        float err = max_abs_diff(hO.data(), gold_chunk, NQ_chunk);
        bool pass = err < tol;
        printf("  D. paged PREFILL (new kernel) vs golden    max_err=%.5f  [%s]\n",
               err, pass ? "PASS <- the fix" : "FAIL");
        ok &= pass;
        cudaFree(dKc); cudaFree(dVc); cudaFree(dQc); cudaFree(dO);
        cudaFree(dbt); cudaFree(dqs); cudaFree(dqp);
    }

    return ok;
}

// ---------------------------------------------------------------------------
// Multi-sequence batch with padding rows: exercises q_seq_idx routing and the
// q_abs_pos < 0 skip path that the qwen3 continued-prefill integration relies on.
// Two sequences of different lengths share one paged cache (distinct blocks);
// the query tile interleaves their new-chunk rows plus a padding row.
// ---------------------------------------------------------------------------
static bool run_multiseq_test(int H_q = 16, int H_kv = 8, int D = 128,
                              int BLOCK_SIZE = 16, float tol = 5e-3f)
{
    const int len[2]   = {50, 30};   // total tokens per sequence
    const int start[2] = {34, 16};   // new-chunk start (cached prefix = start[s])
    const int bps      = (50 + BLOCK_SIZE - 1) / BLOCK_SIZE;  // blocks per seq (max)

    printf("\n== multi-seq + padding  (seqA len=%d from %d, seqB len=%d from %d) ==\n",
           len[0], start[0], len[1], start[1]);

    // Per-sequence random K/V/Q and golden causal attention.
    std::vector<std::vector<half>> sK(2), sV(2), sQ(2);
    std::vector<std::vector<float>> gold(2);
    for (int s = 0; s < 2; s++) {
        long nq = (long)len[s] * H_q * D, nk = (long)len[s] * H_kv * D;
        sQ[s].resize(nq); sK[s].resize(nk); sV[s].resize(nk);
        for (long i = 0; i < nq; i++) sQ[s][i] = __float2half(lcg_randf() * 0.5f);
        for (long i = 0; i < nk; i++) sK[s][i] = __float2half(lcg_randf() * 0.5f);
        for (long i = 0; i < nk; i++) sV[s][i] = __float2half(lcg_randf() * 0.5f);
        std::vector<float> fQ(nq), fK(nk), fV(nk);
        for (long i = 0; i < nq; i++) fQ[i] = __half2float(sQ[s][i]);
        for (long i = 0; i < nk; i++) fK[i] = __half2float(sK[s][i]);
        for (long i = 0; i < nk; i++) fV[i] = __half2float(sV[s][i]);
        gold[s].resize(nq);
        ref_causal_attention(gold[s].data(), fQ.data(), fK.data(), fV.data(),
                             len[s], H_q, H_kv, D);
    }

    // Shared paged cache: seq s uses physical blocks [s*bps, s*bps + bps).
    const long cache_elems = (long)2 * bps * H_kv * BLOCK_SIZE * D;
    std::vector<half> hKc(cache_elems, __float2half(0.0f));
    std::vector<half> hVc(cache_elems, __float2half(0.0f));
    for (int s = 0; s < 2; s++)
        for (int t = 0; t < len[s]; t++) {
            int phys = s * bps + t / BLOCK_SIZE, off = t % BLOCK_SIZE;
            for (int h = 0; h < H_kv; h++) {
                long base = ((long)phys * H_kv + h) * BLOCK_SIZE * D + (long)off * D;
                for (int d = 0; d < D; d++) {
                    hKc[base + d] = sK[s][(t * H_kv + h) * D + d];
                    hVc[base + d] = sV[s][(t * H_kv + h) * D + d];
                }
            }
        }
    std::vector<int> h_bt(2 * bps, -1);
    for (int s = 0; s < 2; s++)
        for (int b = 0; b < bps; b++) h_bt[s * bps + b] = s * bps + b;

    // Build interleaved query tile: A-rows, one padding row, then B-rows.
    std::vector<half> hQ;
    std::vector<int>  qseq, qpos;
    std::vector<std::pair<int,int>> ref_of;  // (seq, abs_pos) per query row; (-1,*)=pad
    auto push_q = [&](int s, int p) {
        for (int hd = 0; hd < H_q * D; hd++) hQ.push_back(sQ[s][(long)p * H_q * D + hd]);
        qseq.push_back(s); qpos.push_back(p); ref_of.push_back({s, p});
    };
    for (int p = start[0]; p < len[0]; p++) push_q(0, p);
    // padding row (abs_pos < 0): seq idx irrelevant, output must be zero
    for (int hd = 0; hd < H_q * D; hd++) hQ.push_back(__float2half(0.0f));
    qseq.push_back(0); qpos.push_back(-1); ref_of.push_back({-1, -1});
    for (int p = start[1]; p < len[1]; p++) push_q(1, p);

    const int num_q = (int)qseq.size();

    half *dKc, *dVc, *dQ, *dO;
    int  *dbt, *dqs, *dqp;
    CUDA_CHECK(cudaMalloc(&dKc, cache_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dVc, cache_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dQ,  (long)num_q * H_q * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dO,  (long)num_q * H_q * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dbt, 2 * bps * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dqs, num_q * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dqp, num_q * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dKc, hKc.data(), cache_elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVc, hVc.data(), cache_elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dQ,  hQ.data(), (long)num_q * H_q * D * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dbt, h_bt.data(), 2 * bps * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dqs, qseq.data(), num_q * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dqp, qpos.data(), num_q * sizeof(int), cudaMemcpyHostToDevice));

    launch_flash_attention_prefill_paged(dQ, dKc, dVc, dO, dbt, dqs, dqp,
                                         num_q, H_q, H_kv, D, bps, BLOCK_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<half> hO((long)num_q * H_q * D);
    CUDA_CHECK(cudaMemcpy(hO.data(), dO, (long)num_q * H_q * D * sizeof(half), cudaMemcpyDeviceToHost));

    float err = 0.0f, pad_err = 0.0f;
    for (int t = 0; t < num_q; t++) {
        auto [s, p] = ref_of[t];
        const half* got = hO.data() + (long)t * H_q * D;
        if (s < 0) {  // padding row → expect all zeros
            for (int i = 0; i < H_q * D; i++) pad_err = fmaxf(pad_err, fabsf(__half2float(got[i])));
        } else {
            const float* ref = gold[s].data() + (long)p * H_q * D;
            err = fmaxf(err, max_abs_diff(got, ref, H_q * D));
        }
    }
    bool pass = (err < tol) && (pad_err == 0.0f);
    printf("  D. paged PREFILL multi-seq vs golden       max_err=%.5f  pad=%.5f  [%s]\n",
           err, pad_err, pass ? "PASS" : "FAIL");

    cudaFree(dKc); cudaFree(dVc); cudaFree(dQ); cudaFree(dO);
    cudaFree(dbt); cudaFree(dqs); cudaFree(dqp);
    return pass;
}

int main() {
    printf("=== Chunked / continued prefill spike (no weights) ===\n");
    printf("A=full prefill correct  B=chunk-only is the gap  C=cache path works  D=new kernel\n");

    bool ok = true;
    ok &= run_scenario("fresh prefill n0=0",   0, 48, 16, 8);  // empty prefix degenerate case
    ok &= run_scenario("small GQA 2:1",       40, 24, 2,  1);
    ok &= run_scenario("Qwen3 heads",         48, 16, 16, 8);
    ok &= run_scenario("cross block bndry",   30, 20, 16, 8);
    ok &= run_scenario("long prefix",        200, 56, 16, 8);
    ok &= run_multiseq_test();

    printf("\n%s\n", ok
        ? "Diagnosis CONFIRMED: continued prefill needs a cache-reading prefill kernel."
        : "Unexpected result — investigate before building on this.");
    return ok ? 0 : 1;
}
