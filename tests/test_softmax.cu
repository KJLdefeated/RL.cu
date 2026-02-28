// test_softmax.cu
// Validates Softmax kernel against a CPU FP32 reference.
// Pass criterion: max absolute error < 1e-3  (FP16 precision budget).
//
// Test cases:
//   1. rows=4,   cols=1024   -- small (hidden dim)
//   2. rows=2,   cols=32768  -- medium (attention scores, long context)
//   3. rows=1,   cols=151936 -- large (vocab logits, Qwen3)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/softmax.cuh"

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

static unsigned int lcg_state = 99991u;
static float lcg_randf() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(lcg_state >> 1) / (float)0x7fffffff) - 1.0f;
}

// CPU reference softmax (FP32, two-pass numerically stable)
static void ref_softmax(
    float*       out,
    const float* x,
    int          rows,
    int          cols
) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float*       or_ = out + r * cols;
        float max_val = xr[0];
        for (int c = 1; c < cols; ++c)
            if (xr[c] > max_val) max_val = xr[c];
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) sum += expf(xr[c] - max_val);
        for (int c = 0; c < cols; ++c) or_[c] = expf(xr[c] - max_val) / sum;
    }
}

// ---------------------------------------------------------------------------
// Single test runner
// ---------------------------------------------------------------------------
static bool run_test(const char* name, int rows, int cols) {
    const long N = (long)rows * cols;

    float* h_x_f32 = new float[N];
    float* h_ref   = new float[N];
    half*  h_x     = new half[N];
    half*  h_out   = new half[N];

    // Scale logits to a realistic range (avoid FP16 overflow: max ~65504)
    for (long i = 0; i < N; ++i) {
        h_x_f32[i] = lcg_randf() * 3.0f;   // [-3, 3] — safe in FP16
        h_x[i]     = __float2half(h_x_f32[i]);
    }

    // Reference uses FP16-quantised values (round-trip) for a fair comparison.
    for (long i = 0; i < N; ++i) h_x_f32[i] = __half2float(h_x[i]);

    ref_softmax(h_ref, h_x_f32, rows, cols);

    half *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x,   N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(half), cudaMemcpyHostToDevice));

    launch_softmax(d_out, d_x, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(half), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (long i = 0; i < N; ++i) {
        float diff = fabsf(__half2float(h_out[i]) - h_ref[i]);
        if (diff > max_err) max_err = diff;
    }

    const float tol = 1e-3f;
    bool passed = (max_err < tol);
    printf("[%s] %-50s max_err=%.6f  %s\n",
           passed ? "PASS" : "FAIL", name, max_err, passed ? "" : "<-- EXCEEDS 1e-3");

    cudaFree(d_x); cudaFree(d_out);
    delete[] h_x_f32; delete[] h_ref;
    delete[] h_x;     delete[] h_out;

    return passed;
}

// ---------------------------------------------------------------------------
// Benchmark  (CUDA events, no host alloc — pure kernel timing)
// Bytes = read x [rows*cols] + write out [rows*cols], all half.
// ---------------------------------------------------------------------------
static void run_benchmark(const char* name, int rows, int cols,
                          int warmup = 10, int iters = 200) {
    const long N = (long)rows * cols;
    half *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x,   N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(half)));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    for (int i = 0; i < warmup; ++i)
        launch_softmax(d_out, d_x, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < iters; ++i)
        launch_softmax(d_out, d_x, rows, cols);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    const float us    = ms * 1000.0f / iters;
    const float bytes = (float)(2 * N) * sizeof(half);
    const float bw_gb = bytes / (us * 1e-6f) / 1e9f;

    printf("[BENCH] %-52s  %7.2f us  %6.1f GB/s\n", name, us, bw_gb);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    cudaFree(d_x); cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("=== Softmax kernel tests ===\n");

    bool all_pass = true;
    all_pass &= run_test("Softmax rows=4   cols=1024   (hidden)",       4,   1024);
    all_pass &= run_test("Softmax rows=2   cols=32768  (attn, seq32k)", 2,   32768);
    all_pass &= run_test("Softmax rows=1   cols=151936 (vocab, Qwen3)", 1,   151936);
    all_pass &= run_test("Softmax rows=8   cols=2048   (attn batch)",   8,   2048);

    printf("\n%s\n", all_pass ? "All tests PASSED." : "Some tests FAILED.");

    printf("\n=== Softmax benchmarks (warmup=10, iters=200) ===\n");
    run_benchmark("Softmax rows=4    cols=1024   (hidden)",          4,    1024);
    run_benchmark("Softmax rows=8    cols=2048   (attn, batch=8)",   8,    2048);
    run_benchmark("Softmax rows=2048 cols=2048   (attn, seq=2048)",  2048, 2048);
    run_benchmark("Softmax rows=2    cols=32768  (attn, seq=32k)",   2,    32768);
    run_benchmark("Softmax rows=1    cols=151936 (vocab, Qwen3)",    1,    151936);

    return all_pass ? 0 : 1;
}
