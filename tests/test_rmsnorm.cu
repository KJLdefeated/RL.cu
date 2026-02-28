// test_rmsnorm.cu
// Validates RMSNorm kernel against a CPU FP32 reference.
// Pass criterion: max absolute error < 1e-3  (FP16 precision budget).
//
// Test cases:
//   1. rows=4,  cols=128  -- head_dim (q_norm / k_norm)
//   2. rows=4,  cols=1024 -- hidden_size (input_layernorm)
//   3. rows=16, cols=1024 -- batched hidden

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/rmsnorm.cuh"

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

// Simple LCG PRNG for deterministic data (no stdlib rand() seed dependency)
static unsigned int lcg_state = 12345u;
static float lcg_randf() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    // Map to [-1, 1]
    return ((float)(lcg_state >> 1) / (float)0x7fffffff) - 1.0f;
}

// CPU reference RMSNorm (FP32 throughout)
static void ref_rmsnorm(
    float*       out,
    const float* x,
    const float* w,
    int          rows,
    int          cols,
    float        eps
) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float*       or_ = out + r * cols;
        float sum_sq = 0.0f;
        for (int c = 0; c < cols; ++c) sum_sq += xr[c] * xr[c];
        float rms_inv = 1.0f / sqrtf(sum_sq / (float)cols + eps);
        for (int c = 0; c < cols; ++c)
            or_[c] = xr[c] * rms_inv * w[c];
    }
}

// ---------------------------------------------------------------------------
// Single test runner
// ---------------------------------------------------------------------------
static bool run_test(const char* name, int rows, int cols, float eps = 1e-5f) {
    const int N = rows * cols;

    // --- host buffers (FP32 for reference, FP16 for kernel) ---
    float* h_x_f32 = new float[N];
    float* h_w_f32 = new float[cols];
    float* h_ref   = new float[N];

    half*  h_x   = new half[N];
    half*  h_out = new half[N];

    for (int i = 0; i < N;    ++i) h_x_f32[i] = lcg_randf();
    for (int i = 0; i < cols; ++i) h_w_f32[i] = lcg_randf() * 0.5f + 1.0f; // near 1

    for (int i = 0; i < N; ++i) h_x[i] = __float2half(h_x_f32[i]);

    // Reference: x round-trips through FP16 (kernel input), weight stays FP32
    for (int i = 0; i < N; ++i) h_x_f32[i] = __half2float(h_x[i]);

    ref_rmsnorm(h_ref, h_x_f32, h_w_f32, rows, cols, eps);

    // --- device buffers ---
    half  *d_x, *d_out;
    float *d_w;
    CUDA_CHECK(cudaMalloc(&d_x,   N    * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w,   cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N    * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x,     N    * sizeof(half),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w_f32, cols * sizeof(float), cudaMemcpyHostToDevice));

    launch_rmsnorm(d_out, d_x, d_w, rows, cols, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(half), cudaMemcpyDeviceToHost));

    // --- compare ---
    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = fabsf(__half2float(h_out[i]) - h_ref[i]);
        if (diff > max_err) max_err = diff;
    }

    const float tol = 1e-3f;
    bool passed = (max_err < tol);
    printf("[%s] %-42s max_err=%.6f  %s\n",
           passed ? "PASS" : "FAIL", name, max_err, passed ? "" : "<-- EXCEEDS 1e-3");

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_out);
    delete[] h_x_f32; delete[] h_w_f32; delete[] h_ref;
    delete[] h_x; delete[] h_out;

    return passed;
}

// ---------------------------------------------------------------------------
// Benchmark  (CUDA events, no host alloc — pure kernel timing)
// Reports: latency in µs and effective memory bandwidth in GB/s.
// Bytes = read x [rows*cols] + read weight [cols] + write out [rows*cols], all half.
// ---------------------------------------------------------------------------
static void run_benchmark(const char* name, int rows, int cols,
                          float eps = 1e-5f,
                          int warmup = 10, int iters = 200) {
    const int N = rows * cols;
    half  *d_x, *d_out;
    float *d_w;
    CUDA_CHECK(cudaMalloc(&d_x,   N    * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w,   cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N    * sizeof(half)));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    for (int i = 0; i < warmup; ++i)
        launch_rmsnorm(d_out, d_x, d_w, rows, cols, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < iters; ++i)
        launch_rmsnorm(d_out, d_x, d_w, rows, cols, eps);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    const float us    = ms * 1000.0f / iters;
    // x (half) + out (half) + weight (float)
    const float bytes = (float)(2 * N) * sizeof(half) + (float)cols * sizeof(float);
    const float bw_gb = bytes / (us * 1e-6f) / 1e9f;

    printf("[BENCH] %-44s  %7.2f us  %6.1f GB/s\n", name, us, bw_gb);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("=== RMSNorm kernel tests ===\n");

    bool all_pass = true;
    all_pass &= run_test("RMSNorm rows=4  cols=128  (head_dim)",     4,  128);
    all_pass &= run_test("RMSNorm rows=4  cols=1024 (hidden_size)",  4,  1024);
    all_pass &= run_test("RMSNorm rows=16 cols=1024 (batched)",      16, 1024);
    all_pass &= run_test("RMSNorm rows=1  cols=128  (single head)",  1,  128);

    printf("\n%s\n", all_pass ? "All tests PASSED." : "Some tests FAILED.");

    printf("\n=== RMSNorm benchmarks (warmup=10, iters=200) ===\n");
    run_benchmark("RMSNorm rows=4    cols=128  (head_dim)",      4,    128);
    run_benchmark("RMSNorm rows=4    cols=1024 (hidden_size)",   4,    1024);
    run_benchmark("RMSNorm rows=16   cols=1024 (batched x16)",   16,   1024);
    run_benchmark("RMSNorm rows=2048 cols=128  (seq=2048 heads)", 2048, 128);
    run_benchmark("RMSNorm rows=2048 cols=1024 (seq=2048 hidden)", 2048, 1024);

    return all_pass ? 0 : 1;
}
