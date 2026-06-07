// test_continued_prefill_model.cu
// ---------------------------------------------------------------------------
// Model-level validation of the continued (chunked) prefill path wired into
// qwen3_prefill (docs/AGENTIC_ROLLOUT_SPEC.md §2.4).
//
// Property: prefilling a sequence in K chunks — where each chunk after the first
// resumes with num_cached_tokens>0 and attends over the cached prefix via
// launch_flash_attention_prefill_paged — must yield the SAME next-token logits as
// a single full prefill of the whole sequence.
//
// Requires real weights at model_weights/Qwen3-0.6B (pass dir as argv[1]).
// Uses arbitrary in-vocab token ids (no tokenizer needed) — this is a numerics
// property of the forward pass, independent of token meaning.
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <filesystem>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "model/qwen3.h"

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); exit(1);} } while(0)
#endif

static int g_pass = 0, g_fail = 0;

// Copy row 0 of d_logits [B, vocab] to a host FP32 vector.
static std::vector<float> fetch_logits_row0(half* d_logits, int vocab) {
    std::vector<half> h(vocab);
    CUDA_CHECK(cudaMemcpy(h.data(), d_logits, vocab * sizeof(half), cudaMemcpyDeviceToHost));
    std::vector<float> f(vocab);
    for (int i = 0; i < vocab; i++) f[i] = __half2float(h[i]);
    return f;
}

static int argmax(const std::vector<float>& v) {
    int a = 0; for (int i = 1; i < (int)v.size(); i++) if (v[i] > v[a]) a = i; return a;
}

// Build a deterministic pseudo-random prompt of in-vocab token ids.
static std::vector<int64_t> make_prompt(int n, uint32_t seed, int vocab_cap) {
    std::vector<int64_t> ids(n);
    uint32_t s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        ids[i] = (int64_t)((s >> 9) % (uint32_t)vocab_cap);
    }
    return ids;
}

// Run a single-sequence prefill at the given slot with `cached` tokens already in
// the KV cache; returns row-0 logits. `tokens` is the FULL token list of the seq.
static std::vector<float> prefill_seq(Qwen3Model* m, int slot, int64_t sid,
                                      const std::vector<int64_t>& tokens, int cached) {
    SamplingParams sp;
    Sequence seq(sid, tokens, sp);
    seq.batch_slot        = slot;
    seq.num_cached_tokens = cached;
    std::vector<Sequence*> batch{&seq};
    half* dlog = qwen3_prefill(m, batch, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    return fetch_logits_row0(dlog, m->config.vocab_size);
}

// Compare two logit vectors: top-1 match (the generation-relevant check) + stats.
static void compare(const char* name, const std::vector<float>& full,
                    const std::vector<float>& chunked) {
    int a_full = argmax(full), a_chunk = argmax(chunked);
    float max_abs = 0.0f, dot = 0.0f, nf = 0.0f, nc = 0.0f;
    for (size_t i = 0; i < full.size(); i++) {
        max_abs = fmaxf(max_abs, fabsf(full[i] - chunked[i]));
        dot += full[i] * chunked[i]; nf += full[i] * full[i]; nc += chunked[i] * chunked[i];
    }
    float cos = dot / (sqrtf(nf) * sqrtf(nc) + 1e-9f);
    bool ok = (a_full == a_chunk) && (cos > 0.999f);
    printf("  [%s] %-28s top1 full=%d chunk=%d  max|Δ|=%.4f  cos=%.6f\n",
           ok ? "PASS" : "FAIL", name, a_full, a_chunk, max_abs, cos);
    ok ? g_pass++ : g_fail++;
}

int main(int argc, char** argv) {
    const char* model_dir = (argc > 1) ? argv[1] : "model_weights/Qwen3-0.6B";
    printf("=== Continued (chunked) prefill — model-level test ===\nModel: %s\n", model_dir);
    if (!std::filesystem::exists(model_dir)) {
        printf("[SKIP] %s not found — pass model_dir as argv[1]\n", model_dir);
        return 0;
    }

    Qwen3Model* m = qwen3_load(model_dir, /*max_batch=*/4, /*max_seq=*/512);
    const int vocab = m->config.vocab_size;

    // ---- Scenario 1: 2-chunk split (40 = 24 + 16) -------------------------
    {
        auto ids = make_prompt(40, 7u, 10000);
        // Full prefill on slot 0.
        auto full = prefill_seq(m, /*slot=*/0, 100, ids, /*cached=*/0);
        qwen3_free_seq_slot(m, 0);
        // Chunked on slot 1: chunk A [0,24) fresh, then chunk B resumes at 24.
        std::vector<int64_t> a(ids.begin(), ids.begin() + 24);
        prefill_seq(m, /*slot=*/1, 101, a, /*cached=*/0);     // writes KV [0,24)
        auto chunk = prefill_seq(m, /*slot=*/1, 102, ids, /*cached=*/24);
        qwen3_free_seq_slot(m, 1);
        compare("2-chunk (24+16)", full, chunk);
    }

    // ---- Scenario 2: 3-chunk split (60 = 20 + 20 + 20), crosses blocks ----
    {
        auto ids = make_prompt(60, 13u, 10000);
        auto full = prefill_seq(m, 0, 200, ids, 0);
        qwen3_free_seq_slot(m, 0);
        std::vector<int64_t> c1(ids.begin(),      ids.begin() + 20);
        std::vector<int64_t> c2(ids.begin(),      ids.begin() + 40);
        prefill_seq(m, 1, 201, c1, 0);    // KV [0,20)
        prefill_seq(m, 1, 202, c2, 20);   // KV [20,40)
        auto chunk = prefill_seq(m, 1, 203, ids, 40);  // KV [40,60)
        qwen3_free_seq_slot(m, 1);
        compare("3-chunk (20+20+20)", full, chunk);
    }

    // ---- Scenario 3: single-token resume (decode-shaped chunk) ------------
    {
        auto ids = make_prompt(33, 29u, 10000);
        auto full = prefill_seq(m, 0, 300, ids, 0);
        qwen3_free_seq_slot(m, 0);
        std::vector<int64_t> head(ids.begin(), ids.begin() + 32);
        prefill_seq(m, 1, 301, head, 0);            // KV [0,32)
        auto chunk = prefill_seq(m, 1, 302, ids, 32); // resume with 1 new token
        qwen3_free_seq_slot(m, 1);
        compare("resume 1 token (32+1)", full, chunk);
    }

    qwen3_free(m);
    printf("\n%s  (%d passed, %d failed)\n",
           g_fail == 0 ? "ALL PASSED" : "SOME FAILED", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
