// test_attention.cu
// Validates Flash Attention 2 (prefill) and Paged Attention (decode) kernels
// against a CPU FP32 naive reference.
//
// Pass criterion (from DESIGN.md): max absolute error < 5e-3 vs naive attention.
//
// Prefill test cases:
//   1. B=1, S=16,  H_q=2, H_kv=1 — small, GQA ratio 2:1
//   2. B=1, S=64,  H_q=4, H_kv=2 — single KV tile (Bc=64)
//   3. B=1, S=128, H_q=16, H_kv=8 — Qwen3 head config, two KV tiles
//   4. B=2, S=96,  H_q=4, H_kv=2 — batched
//
// Decode test cases:
//   5. num_seqs=1, context_len=32,  H_q=2, H_kv=1
//   6. num_seqs=2, context_len=128, H_q=16, H_kv=8

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/attention.cuh"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Deterministic LCG (no srand dependency)
static unsigned int lcg_state = 42u;
static float lcg_randf() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(lcg_state >> 1) / (float)0x7fffffffu) * 2.0f - 1.0f;
}

// ---------------------------------------------------------------------------
// CPU reference: naive causal attention
// Q, K, V layout: [B, S, H, D]   (H = num_heads)
// O        layout: [B, S, H_q, D]
// FP32 throughout.
// ---------------------------------------------------------------------------
static void ref_attention(
    float*       O,
    const float* Q,
    const float* K,
    const float* V,
    int B, int S, int H_q, int H_kv, int D
) {
    const float scale = 1.0f / sqrtf((float)D);

    float* scores = new float[S];

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H_q; h++) {
            const int hkv = h * H_kv / H_q;

            for (int i = 0; i < S; i++) {
                const float* qi = Q + ((b * S + i) * H_q + h) * D;

                // Compute scores[0..i]
                float max_s = -1e30f;
                for (int j = 0; j <= i; j++) {
                    const float* kj = K + ((b * S + j) * H_kv + hkv) * D;
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++) dot += qi[d] * kj[d];
                    scores[j] = dot * scale;
                    if (scores[j] > max_s) max_s = scores[j];
                }

                // Softmax over [0..i]
                float sum_e = 0.0f;
                for (int j = 0; j <= i; j++) {
                    scores[j] = expf(scores[j] - max_s);
                    sum_e += scores[j];
                }
                for (int j = 0; j <= i; j++) scores[j] /= sum_e;

                // O[i] = softmax(S) · V[0..i]
                float* oi = O + ((b * S + i) * H_q + h) * D;
                for (int d = 0; d < D; d++) {
                    float acc = 0.0f;
                    for (int j = 0; j <= i; j++) {
                        const float* vj = V + ((b * S + j) * H_kv + hkv) * D;
                        acc += scores[j] * vj[d];
                    }
                    oi[d] = acc;
                }
            }
        }
    }

    delete[] scores;
}

// ---------------------------------------------------------------------------
// CPU reference: non-causal attention (for decode, S_q=1)
// q layout: [num_seqs, H_q, D]
// K, V layout: [num_seqs, S_kv, H_kv, D]  — contiguous KV context
// O layout: [num_seqs, H_q, D]
// ---------------------------------------------------------------------------
static void ref_decode_attention(
    float*       O,
    const float* q,
    const float* K,
    const float* V,
    int num_seqs, int S_kv, int H_q, int H_kv, int D
) {
    const float scale = 1.0f / sqrtf((float)D);
    float* scores = new float[S_kv];

    for (int s = 0; s < num_seqs; s++) {
        for (int h = 0; h < H_q; h++) {
            const int hkv = h * H_kv / H_q;
            const float* qi = q + (s * H_q + h) * D;

            float max_s = -1e30f;
            for (int j = 0; j < S_kv; j++) {
                const float* kj = K + ((s * S_kv + j) * H_kv + hkv) * D;
                float dot = 0.0f;
                for (int d = 0; d < D; d++) dot += qi[d] * kj[d];
                scores[j] = dot * scale;
                if (scores[j] > max_s) max_s = scores[j];
            }

            float sum_e = 0.0f;
            for (int j = 0; j < S_kv; j++) {
                scores[j] = expf(scores[j] - max_s);
                sum_e += scores[j];
            }
            for (int j = 0; j < S_kv; j++) scores[j] /= sum_e;

            float* oi = O + (s * H_q + h) * D;
            for (int d = 0; d < D; d++) {
                float acc = 0.0f;
                for (int j = 0; j < S_kv; j++) {
                    const float* vj = V + ((s * S_kv + j) * H_kv + hkv) * D;
                    acc += scores[j] * vj[d];
                }
                oi[d] = acc;
            }
        }
    }

    delete[] scores;
}

// ---------------------------------------------------------------------------
// Prefill test runner
// ---------------------------------------------------------------------------
static bool run_prefill_test(
    const char* name, int B, int S, int H_q, int H_kv,
    int D = 128, float tol = 5e-3f
) {
    const long N_q  = (long)B * S * H_q  * D;
    const long N_kv = (long)B * S * H_kv * D;

    // Host buffers (FP32 for reference, FP16 for kernel)
    float* h_Q_f32 = new float[N_q];
    float* h_K_f32 = new float[N_kv];
    float* h_V_f32 = new float[N_kv];
    float* h_ref   = new float[N_q];

    half*  h_Q   = new half[N_q];
    half*  h_K   = new half[N_kv];
    half*  h_V   = new half[N_kv];
    half*  h_out = new half[N_q];

    // Fill with random values, scale to prevent softmax saturation
    for (long i = 0; i < N_q;  i++) h_Q[i] = __float2half(lcg_randf() * 0.5f);
    for (long i = 0; i < N_kv; i++) h_K[i] = __float2half(lcg_randf() * 0.5f);
    for (long i = 0; i < N_kv; i++) h_V[i] = __float2half(lcg_randf() * 0.5f);

    // FP32 reference uses the FP16-quantised values (round-trip)
    for (long i = 0; i < N_q;  i++) h_Q_f32[i] = __half2float(h_Q[i]);
    for (long i = 0; i < N_kv; i++) h_K_f32[i] = __half2float(h_K[i]);
    for (long i = 0; i < N_kv; i++) h_V_f32[i] = __half2float(h_V[i]);

    ref_attention(h_ref, h_Q_f32, h_K_f32, h_V_f32, B, S, H_q, H_kv, D);

    // Device buffers
    half *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, N_q  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K, N_kv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V, N_kv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_O, N_q  * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, N_q  * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, N_kv * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, N_kv * sizeof(half), cudaMemcpyHostToDevice));

    launch_flash_attention_prefill(d_Q, d_K, d_V, d_O, B, S, H_q, H_kv, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_O, N_q * sizeof(half), cudaMemcpyDeviceToHost));

    // Compare
    float max_err = 0.0f;
    for (long i = 0; i < N_q; i++) {
        float diff = fabsf(__half2float(h_out[i]) - h_ref[i]);
        if (diff > max_err) max_err = diff;
    }

    bool passed = (max_err < tol);
    printf("[%s] %-50s max_err=%.6f  %s\n",
           passed ? "PASS" : "FAIL", name, max_err,
           passed ? "" : "<-- EXCEEDS 5e-3");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    delete[] h_Q_f32; delete[] h_K_f32; delete[] h_V_f32; delete[] h_ref;
    delete[] h_Q;    delete[] h_K;    delete[] h_V;    delete[] h_out;

    return passed;
}

// ---------------------------------------------------------------------------
// Paged attention decode test runner
// Uses a contiguous block mapping (block i → physical block i) so the paged
// result is equivalent to standard attention on contiguous K/V.
// ---------------------------------------------------------------------------
static bool run_decode_test(
    const char* name,
    int num_seqs, int context_len, int H_q, int H_kv,
    int D = 128, int BLOCK_SIZE = 16, float tol = 5e-3f
) {
    // Total KV tokens per sequence (all seqs same length for simplicity)
    const long Q_total  = (long)num_seqs * H_q * D;
    const long O_total  = Q_total;

    // Host buffers (FP32 reference)
    float* h_q_f32 = new float[Q_total];
    float* h_K_f32 = new float[(long)num_seqs * context_len * H_kv * D];
    float* h_V_f32 = new float[(long)num_seqs * context_len * H_kv * D];
    float* h_ref   = new float[O_total];

    half*  h_q   = new half[Q_total];
    half*  h_out = new half[O_total];

    for (long i = 0; i < Q_total;  i++) h_q[i] = __float2half(lcg_randf() * 0.5f);
    for (long i = 0; i < (long)num_seqs * context_len * H_kv * D; i++) {
        h_K_f32[i] = lcg_randf() * 0.5f;
        h_V_f32[i] = lcg_randf() * 0.5f;
    }

    for (long i = 0; i < Q_total; i++) h_q_f32[i] = __half2float(h_q[i]);
    // Quantise K/V round-trip
    for (long i = 0; i < (long)num_seqs * context_len * H_kv * D; i++) {
        h_K_f32[i] = __half2float(__float2half(h_K_f32[i]));
        h_V_f32[i] = __half2float(__float2half(h_V_f32[i]));
    }

    ref_decode_attention(h_ref, h_q_f32, h_K_f32, h_V_f32,
                         num_seqs, context_len, H_q, H_kv, D);

    // Build the paged KV cache with an identity block table
    // Each sequence uses context_len/BLOCK_SIZE (rounded up) blocks.
    const int blocks_per_seq  = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int total_phys_blocks = num_seqs * blocks_per_seq;

    // k_cache layout: [num_blocks, H_kv, BLOCK_SIZE, D]
    const long cache_elems = (long)total_phys_blocks * H_kv * BLOCK_SIZE * D;
    half* h_k_cache = new half[cache_elems]();
    half* h_v_cache = new half[cache_elems]();

    // Fill paged cache from contiguous K/V
    // seq s, token t, kv head h → logical_block = t/BLOCK_SIZE, offset = t%BLOCK_SIZE
    // physical_block = s * blocks_per_seq + logical_block
    for (int s = 0; s < num_seqs; s++) {
        for (int t = 0; t < context_len; t++) {
            for (int h = 0; h < H_kv; h++) {
                int  logical_block = t / BLOCK_SIZE;
                int  tok_offset    = t % BLOCK_SIZE;
                int  phys_block    = s * blocks_per_seq + logical_block;

                const float* ksrc = h_K_f32 + ((s * context_len + t) * H_kv + h) * D;
                const float* vsrc = h_V_f32 + ((s * context_len + t) * H_kv + h) * D;

                half* kdst = h_k_cache
                    + ((long)phys_block * H_kv + h) * BLOCK_SIZE * D
                    + tok_offset * D;
                half* vdst = h_v_cache
                    + ((long)phys_block * H_kv + h) * BLOCK_SIZE * D
                    + tok_offset * D;

                for (int d = 0; d < D; d++) {
                    kdst[d] = __float2half(ksrc[d]);
                    vdst[d] = __float2half(vsrc[d]);
                }
            }
        }
    }

    // Block tables: identity mapping (seq s, logical block b → physical block s*bps+b)
    int* h_block_tables = new int[num_seqs * blocks_per_seq];
    int* h_seq_lens     = new int[num_seqs];
    for (int s = 0; s < num_seqs; s++) {
        h_seq_lens[s] = context_len;
        for (int b = 0; b < blocks_per_seq; b++)
            h_block_tables[s * blocks_per_seq + b] = s * blocks_per_seq + b;
    }

    // Device allocations
    half *d_q, *d_k_cache, *d_v_cache, *d_out;
    int  *d_block_tables, *d_seq_lens;

    CUDA_CHECK(cudaMalloc(&d_q,           Q_total * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k_cache,     cache_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v_cache,     cache_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out,         O_total * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_block_tables, num_seqs * blocks_per_seq * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seq_lens,    num_seqs * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_q,           h_q,           Q_total * sizeof(half),                cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_cache,     h_k_cache,     cache_elems * sizeof(half),            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_cache,     h_v_cache,     cache_elems * sizeof(half),            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_tables, h_block_tables, num_seqs * blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_lens,    h_seq_lens,    num_seqs * sizeof(int),                cudaMemcpyHostToDevice));

    launch_paged_attention_decode(
        d_q, d_k_cache, d_v_cache, d_out,
        d_block_tables, d_seq_lens,
        num_seqs, H_q, H_kv, D,
        blocks_per_seq, BLOCK_SIZE
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, O_total * sizeof(half), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (long i = 0; i < O_total; i++) {
        float diff = fabsf(__half2float(h_out[i]) - h_ref[i]);
        if (diff > max_err) max_err = diff;
    }

    bool passed = (max_err < tol);
    printf("[%s] %-50s max_err=%.6f  %s\n",
           passed ? "PASS" : "FAIL", name, max_err,
           passed ? "" : "<-- EXCEEDS 5e-3");

    cudaFree(d_q); cudaFree(d_k_cache); cudaFree(d_v_cache);
    cudaFree(d_out); cudaFree(d_block_tables); cudaFree(d_seq_lens);

    delete[] h_q_f32; delete[] h_K_f32; delete[] h_V_f32; delete[] h_ref;
    delete[] h_q; delete[] h_out;
    delete[] h_k_cache; delete[] h_v_cache;
    delete[] h_block_tables; delete[] h_seq_lens;

    return passed;
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------
static void run_decode_benchmark(
    const char* name,
    int num_seqs, int context_len, int H_q, int H_kv,
    int D = 128, int BLOCK_SIZE = 16,
    int warmup = 20, int iters = 500
) {
    const int blocks_per_seq    = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int total_phys_blocks = num_seqs * blocks_per_seq;

    const long Q_total    = (long)num_seqs * H_q * D;
    const long cache_elem = (long)total_phys_blocks * H_kv * BLOCK_SIZE * D;

    half *d_q, *d_k, *d_v, *d_out;
    int  *d_block_tables, *d_seq_lens;

    CUDA_CHECK(cudaMalloc(&d_q,           Q_total    * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k,           cache_elem * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v,           cache_elem * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out,         Q_total    * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_block_tables, num_seqs * blocks_per_seq * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seq_lens,    num_seqs * sizeof(int)));

    // Identity block table + uniform context length
    {
        int* h_bt = new int[num_seqs * blocks_per_seq];
        int* h_sl = new int[num_seqs];
        for (int s = 0; s < num_seqs; s++) {
            h_sl[s] = context_len;
            for (int b = 0; b < blocks_per_seq; b++)
                h_bt[s * blocks_per_seq + b] = s * blocks_per_seq + b;
        }
        CUDA_CHECK(cudaMemcpy(d_block_tables, h_bt,
            num_seqs * blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seq_lens, h_sl,
            num_seqs * sizeof(int), cudaMemcpyHostToDevice));
        delete[] h_bt;
        delete[] h_sl;
    }

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    for (int i = 0; i < warmup; i++)
        launch_paged_attention_decode(d_q, d_k, d_v, d_out,
            d_block_tables, d_seq_lens,
            num_seqs, H_q, H_kv, D, blocks_per_seq, BLOCK_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < iters; i++)
        launch_paged_attention_decode(d_q, d_k, d_v, d_out,
            d_block_tables, d_seq_lens,
            num_seqs, H_q, H_kv, D, blocks_per_seq, BLOCK_SIZE);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    const float us = ms * 1000.0f / iters;

    // Memory traffic: K read + V read + Q read + O write
    const double bytes_kv = 2.0 * (long)num_seqs * H_kv * context_len * D * sizeof(half);
    const double bytes_qo = 2.0 * (long)num_seqs * H_q  * D * sizeof(half);
    const double gbs = (bytes_kv + bytes_qo) / (us * 1e-6) / 1e9;

    printf("[BENCH] %-55s  %7.2f us  %6.1f GB/s\n", name, us, gbs);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    cudaFree(d_block_tables); cudaFree(d_seq_lens);
}

static void run_prefill_benchmark(
    const char* name, int B, int S, int H_q, int H_kv,
    int D = 128, int warmup = 10, int iters = 200
) {
    const long N_q  = (long)B * S * H_q  * D;
    const long N_kv = (long)B * S * H_kv * D;

    half *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, N_q  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K, N_kv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V, N_kv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_O, N_q  * sizeof(half)));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    for (int i = 0; i < warmup; i++)
        launch_flash_attention_prefill(d_Q, d_K, d_V, d_O, B, S, H_q, H_kv, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < iters; i++)
        launch_flash_attention_prefill(d_Q, d_K, d_V, d_O, B, S, H_q, H_kv, D);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    const float us = ms * 1000.0f / iters;

    // FLOPs: 2 * B * H_q * S^2 * D  (Q@K^T and softmax(S)@V, approximately)
    const double flops = 2.0 * B * H_q * (double)S * S * D * 2;
    const double tflops = flops / (us * 1e-6) / 1e12;

    printf("[BENCH] %-50s  %7.2f us  %5.2f TFLOPS\n", name, us, tflops);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("=== Flash Attention kernel tests ===\n\n");

    // ── Prefill tests ────────────────────────────────────────────────────────
    printf("--- Prefill (FA2) ---\n");
    bool all_pass = true;

    all_pass &= run_prefill_test(
        "Prefill B=1 S=16  H_q=2  H_kv=1 (small GQA 2:1)",
        1, 16, 2, 1);

    all_pass &= run_prefill_test(
        "Prefill B=1 S=64  H_q=4  H_kv=2 (single KV tile)",
        1, 64, 4, 2);

    all_pass &= run_prefill_test(
        "Prefill B=1 S=128 H_q=16 H_kv=8 (Qwen3 heads, 2 KV tiles)",
        1, 128, 16, 8);

    all_pass &= run_prefill_test(
        "Prefill B=2 S=96  H_q=4  H_kv=2 (batched)",
        2, 96, 4, 2);

    all_pass &= run_prefill_test(
        "Prefill B=1 S=256 H_q=16 H_kv=8 (4 KV tiles)",
        1, 256, 16, 8);

    // ── Decode tests ─────────────────────────────────────────────────────────
    printf("\n--- Decode (Paged Attention) ---\n");

    all_pass &= run_decode_test(
        "Decode num_seqs=1 ctx=32  H_q=2  H_kv=1",
        1, 32, 2, 1);

    all_pass &= run_decode_test(
        "Decode num_seqs=1 ctx=128 H_q=16 H_kv=8 (Qwen3)",
        1, 128, 16, 8);

    all_pass &= run_decode_test(
        "Decode num_seqs=2 ctx=64  H_q=4  H_kv=2 (batched seqs)",
        2, 64, 4, 2);

    all_pass &= run_decode_test(
        "Decode num_seqs=4 ctx=128 H_q=16 H_kv=8 (GRPO-style batch)",
        4, 128, 16, 8);

    // ── Summary ──────────────────────────────────────────────────────────────
    printf("\n%s\n", all_pass ? "All tests PASSED." : "Some tests FAILED.");

    // ── Benchmarks ───────────────────────────────────────────────────────────
    printf("\n=== Prefill benchmarks (warmup=10, iters=200) ===\n");
    // run_prefill_benchmark("Prefill B=1 S=128  H_q=16 H_kv=8",  1, 128,  16, 8);
    // run_prefill_benchmark("Prefill B=1 S=512  H_q=16 H_kv=8",  1, 512,  16, 8);
    run_prefill_benchmark("Prefill B=8 S=2048 H_q=16 H_kv=8",  8, 2048, 16, 8);

    printf("\n=== Decode benchmarks — warp-parallel (warmup=20, iters=500) ===\n");
    run_decode_benchmark("Decode  B=1   ctx=512   H_q=16 H_kv=8",  1,   512, 16, 8);
    run_decode_benchmark("Decode  B=1   ctx=2048  H_q=16 H_kv=8",  1,  2048, 16, 8);
    run_decode_benchmark("Decode  B=16  ctx=512   H_q=16 H_kv=8", 16,   512, 16, 8);
    run_decode_benchmark("Decode  B=16  ctx=2048  H_q=16 H_kv=8", 16,  2048, 16, 8);
    run_decode_benchmark("Decode  B=64  ctx=512   H_q=16 H_kv=8", 64,   512, 16, 8);
    run_decode_benchmark("Decode  B=64  ctx=2048  H_q=16 H_kv=8", 64,  2048, 16, 8);
    run_decode_benchmark("Decode  B=128 ctx=2048  H_q=16 H_kv=8",128,  2048, 16, 8);

    return all_pass ? 0 : 1;
}
