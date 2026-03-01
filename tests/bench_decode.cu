// tests/bench_decode.cu
//
// Decode throughput benchmark for Qwen3-0.6B.
//
// Protocol:
//   1. Load model + tokenizer.
//   2. Prefill the same chat prompt used in test_qwen3 (S≈40 tokens).
//   3. Warm-up: WARMUP decode steps with growing KV context.
//   4. Benchmark: BENCH decode steps, timed with cudaEvent on stream 0.
//      - A single pre-allocated GPU token buffer is reused across all steps
//        (no cudaFree / cudaMalloc inside the timed loop).
//      - KV cache grows by one slot per step (realistic decode behaviour).
//   5. Print: "BENCH_RESULT: tok/s=<float>" for the Python script to parse,
//      plus a human-readable summary.
//
// Usage:
//   ./build/bench_decode
//
// Build:
//   make bench_decode

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "model/qwen3.h"
#include "model/tokenizer.h"

static const char* MODEL_DIR = "model_weights/Qwen3-0.6B";
static constexpr int WARMUP = 10;
static constexpr int BENCH  = 100;
static const int VOCAB_SIZE = 151936;

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t e = (call);                                            \
        if (e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                    cudaGetErrorString(e), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Synchronous device→host argmax over the logit buffer.
static int argmax_from_gpu(const half* d_logits, int V) {
    std::vector<half> buf(V);
    CUDA_CHECK(cudaMemcpy(buf.data(), d_logits, V * sizeof(half),
                          cudaMemcpyDeviceToHost));
    int best = 0;
    for (int i = 1; i < V; i++)
        if (__half2float(buf[i]) > __half2float(buf[best])) best = i;
    return best;
}

int main() {
    if (!std::filesystem::exists(MODEL_DIR)) {
        printf("[SKIP] %s not found\n", MODEL_DIR);
        return 0;
    }

    // ── Tokenizer ────────────────────────────────────────────────────────────
    Tokenizer tok;
    if (!tok.load(std::string(MODEL_DIR) + "/tokenizer.json")) {
        fprintf(stderr, "Tokenizer load failed\n");
        return 1;
    }

    // ── Model ────────────────────────────────────────────────────────────────
    // max_seq=256 gives headroom for WARMUP+BENCH extra tokens beyond the
    // prefill length (40 + 110 = 150 ≤ 256).
    Qwen3Model* m = qwen3_load(MODEL_DIR, /*max_batch=*/1, /*max_seq=*/256);

    // ── Prefill ──────────────────────────────────────────────────────────────
    const std::string prompt   = tok.chat_prompt("What is the capital of France?");
    const std::vector<int> ids = tok.encode(prompt);
    const int S                = (int)ids.size();

    printf("Prefilling %d tokens...\n", S);
    {
        int* d_prompt;
        CUDA_CHECK(cudaMalloc(&d_prompt, S * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_prompt, ids.data(), S * sizeof(int),
                              cudaMemcpyHostToDevice));
        half* dlog = qwen3_prefill(m, d_prompt, 1, S);
        CUDA_CHECK(cudaDeviceSynchronize());
        int next = argmax_from_gpu(dlog, VOCAB_SIZE);
        cudaFree(d_prompt);

        // ── Warm-up ──────────────────────────────────────────────────────────
        printf("Warm-up  (%d steps)...\n", WARMUP);
        int* d_tok_tmp;
        CUDA_CHECK(cudaMalloc(&d_tok_tmp, sizeof(int)));
        for (int i = 0; i < WARMUP; i++) {
            CUDA_CHECK(cudaMemcpy(d_tok_tmp, &next, sizeof(int),
                                  cudaMemcpyHostToDevice));
            dlog = qwen3_decode(m, d_tok_tmp, 1);
            CUDA_CHECK(cudaDeviceSynchronize());
            next = argmax_from_gpu(dlog, VOCAB_SIZE);
        }
        cudaFree(d_tok_tmp);

        // ── Benchmark ────────────────────────────────────────────────────────
        // Allocate one persistent token buffer.  Reuse it for all BENCH steps
        // so there are no allocations inside the timed region.
        int* d_tok;
        CUDA_CHECK(cudaMalloc(&d_tok, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_tok, &next, sizeof(int),
                              cudaMemcpyHostToDevice));

        printf("Benchmark(%d steps)...\n", BENCH);
        cudaEvent_t ev_start, ev_end;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_end));

        // Record start, fire all BENCH steps without any inter-step sync,
        // then record end.  cudaEventSynchronize(ev_end) waits for the GPU
        // to finish all enqueued work.
        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int i = 0; i < BENCH; i++)
            qwen3_decode(m, d_tok, 1);   // KV cache grows; d_tok value stays fixed
        CUDA_CHECK(cudaEventRecord(ev_end));
        CUDA_CHECK(cudaEventSynchronize(ev_end));

        float ms_total = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_total, ev_start, ev_end));
        const float toks_per_sec = BENCH / (ms_total * 1e-3f);
        const float ms_per_step  = ms_total / BENCH;

        printf("\n");
        printf("  Context: %d prefill + %d warmup + %d bench tokens\n",
               S, WARMUP, BENCH);
        printf("  Decode:  %7.1f tok/s  (%d steps, %.2f ms/step)\n",
               toks_per_sec, BENCH, ms_per_step);
        printf("BENCH_RESULT: tok/s=%.2f\n", toks_per_sec);

        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_end);
        cudaFree(d_tok);
    }

    qwen3_free(m);
    return 0;
}
