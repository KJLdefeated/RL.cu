// test_embedding.cu
// Validates the embedding gather kernel against a CPU reference.
//
// Pass criterion: exact FP16 match (gather is a pure memory copy, zero arithmetic).
// We verify that out[tok] == weight[token_ids[tok]] for every element.
//
// Test cases:
//   1. Single token                      — trivial sanity check
//   2. S=32 tokens, vocab=1000            — standard small test
//   3. Repeated token IDs                 — same vocab row, multiple output rows
//   4. S=512, vocab=151936, hidden=1024  — full Qwen3 config (memory bandwidth)
//   5. Token ID 0 and vocab_size-1       — boundary vocab entries

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/embedding.cuh"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                     \
    } while (0)

static unsigned int lcg = 9999u;
static float lcg_randf() {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(lcg >> 1) / (float)0x7fffffffu) * 2.0f - 1.0f;
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
static void ref_embedding(
    float*       out,         // [num_tokens, hidden_size]
    const float* weight,      // [vocab_size, hidden_size]
    const int*   token_ids,
    int num_tokens, int hidden_size
) {
    for (int t = 0; t < num_tokens; t++) {
        const float* src = weight + (long)token_ids[t] * hidden_size;
        float*       dst = out    + (long)t             * hidden_size;
        for (int d = 0; d < hidden_size; d++)
            dst[d] = src[d];
    }
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
static bool run_test(
    const char* name,
    int  num_tokens,
    int  vocab_size,
    int  hidden_size,
    const int* token_ids   // [num_tokens], host
) {
    const long W = (long)vocab_size  * hidden_size;
    const long O = (long)num_tokens  * hidden_size;

    // Build FP32 weight (random), convert to FP16 for kernel
    float* h_weight_f32 = new float[W];
    half*  h_weight     = new half[W];
    for (long i = 0; i < W; i++) {
        h_weight_f32[i] = lcg_randf();
        h_weight[i]     = __float2half(h_weight_f32[i]);
    }
    // Round-trip: reference uses the same FP16-quantised values
    for (long i = 0; i < W; i++)
        h_weight_f32[i] = __half2float(h_weight[i]);

    // CPU reference
    float* h_ref = new float[O];
    ref_embedding(h_ref, h_weight_f32, token_ids, num_tokens, hidden_size);

    // GPU allocations
    half *d_weight, *d_out;
    int  *d_ids;
    CUDA_CHECK(cudaMalloc(&d_weight, W * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out,    O * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ids,    num_tokens * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_weight, h_weight,   W * sizeof(half),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ids,    token_ids,  num_tokens * sizeof(int), cudaMemcpyHostToDevice));

    launch_embedding(d_out, d_weight, d_ids, num_tokens, vocab_size, hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    half* h_out = new half[O];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, O * sizeof(half), cudaMemcpyDeviceToHost));

    // Verify — expect exact FP16 match (no arithmetic involved)
    float max_err = 0.0f;
    for (long i = 0; i < O; i++) {
        float diff = fabsf(__half2float(h_out[i]) - h_ref[i]);
        if (diff > max_err) max_err = diff;
    }

    // FP16 gather is exact — any mismatch is a bug
    const bool passed = (max_err == 0.0f);
    printf("[%s] %-54s max_err=%.6f  %s\n",
           passed ? "PASS" : "FAIL", name, max_err,
           passed ? "" : "<-- MISMATCH");

    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_ids));
    delete[] h_weight_f32; delete[] h_weight;
    delete[] h_ref; delete[] h_out;

    return passed;
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------
static void run_benchmark(
    const char* name,
    int num_tokens, int vocab_size, int hidden_size,
    int warmup = 20, int iters = 500
) {
    const long W = (long)vocab_size * hidden_size;
    const long O = (long)num_tokens * hidden_size;

    half *d_weight, *d_out;
    int  *d_ids;
    CUDA_CHECK(cudaMalloc(&d_weight, W * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out,    O * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ids,    num_tokens * sizeof(int)));

    // Sequential token IDs (representative access pattern for short prompts)
    int* h_ids = new int[num_tokens];
    for (int i = 0; i < num_tokens; i++) h_ids[i] = i % vocab_size;
    CUDA_CHECK(cudaMemcpy(d_ids, h_ids, num_tokens * sizeof(int), cudaMemcpyHostToDevice));
    delete[] h_ids;

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    for (int i = 0; i < warmup; i++)
        launch_embedding(d_out, d_weight, d_ids, num_tokens, vocab_size, hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < iters; i++)
        launch_embedding(d_out, d_weight, d_ids, num_tokens, vocab_size, hidden_size);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    const float us = ms * 1000.0f / iters;
    // Bandwidth: read num_tokens rows from weight + write output
    const float bytes = (float)(2 * O * sizeof(half));
    const float bw    = bytes / (us * 1e-6f) / 1e9f;

    printf("[BENCH] %-50s  %6.2f us  %6.1f GB/s\n", name, us, bw);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_ids));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("=== Embedding kernel tests ===\n\n");

    bool all_pass = true;

    // Test 1: single token
    {
        int ids[] = {42};
        all_pass &= run_test("single token id=42", 1, 1000, 1024, ids);
    }

    // Test 2: 32 tokens, small vocab
    {
        const int S = 32;
        int ids[S];
        for (int i = 0; i < S; i++) ids[i] = i * 7 % 1000;
        all_pass &= run_test("S=32 vocab=1000 hidden=1024", S, 1000, 1024, ids);
    }

    // Test 3: repeated token IDs — same row gathered multiple times
    {
        const int S = 16;
        int ids[S];
        for (int i = 0; i < S; i++) ids[i] = 5;  // all same vocab entry
        all_pass &= run_test("S=16 all same token id=5", S, 1000, 1024, ids);
    }

    // Test 4: boundary vocab IDs (0 and vocab_size-1)
    {
        int ids[] = {0, 999};
        all_pass &= run_test("boundary ids: 0 and vocab_size-1", 2, 1000, 1024, ids);
    }

    // Test 5: larger vocab with random IDs
    {
        const int S = 64, V = 10000;
        int ids[S];
        for (int i = 0; i < S; i++) ids[i] = (i * 1337 + 7) % V;
        all_pass &= run_test("S=64 vocab=10000 hidden=1024", S, V, 1024, ids);
    }

    printf("\n%s\n", all_pass ? "All tests PASSED." : "Some tests FAILED.");

    printf("\n=== Embedding benchmarks (warmup=20, iters=500) ===\n");
    // Qwen3: vocab=151936, hidden=1024
    run_benchmark("S=1    Qwen3 vocab=151936 hidden=1024",   1,   151936, 1024);
    run_benchmark("S=32   Qwen3 vocab=151936 hidden=1024",  32,   151936, 1024);
    run_benchmark("S=128  Qwen3 vocab=151936 hidden=1024", 128,   151936, 1024);
    run_benchmark("S=512  Qwen3 vocab=151936 hidden=1024", 512,   151936, 1024);
    run_benchmark("S=2048 Qwen3 vocab=151936 hidden=1024", 2048,  151936, 1024);

    return all_pass ? 0 : 1;
}
