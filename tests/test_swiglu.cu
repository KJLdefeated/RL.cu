// test_swiglu.cu
// Validates SwiGLU kernel against a CPU FP32 reference.
// Pass criterion: max absolute error < 1e-3  (FP16 precision budget).
//
// Test cases:
//   1. n=3072         -- intermediate_size for Qwen3-0.6B (single token)
//   2. n=3072 * 16    -- batched (seq_len=16)
//   3. n=9728         -- intermediate_size for Qwen3-4B
//   4. Boundary: n not divisible by block_size (256)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/swiglu.cuh"

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

static unsigned int lcg_state = 54321u;
static float lcg_randf() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(lcg_state >> 1) / (float)0x7fffffff) - 1.0f;
}

// CPU reference: silu(gate) * up, FP32
static void ref_swiglu(
    float*       out,
    const float* gate,
    const float* up,
    int          n
) {
    for (int i = 0; i < n; ++i) {
        float g    = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i]     = silu * up[i];
    }
}

// ---------------------------------------------------------------------------
// Single test runner
// ---------------------------------------------------------------------------
static bool run_test(const char* name, int n) {
    float* h_gate_f32 = new float[n];
    float* h_up_f32   = new float[n];
    float* h_ref      = new float[n];
    half*  h_gate     = new half[n];
    half*  h_up       = new half[n];
    half*  h_out      = new half[n];

    for (int i = 0; i < n; ++i) {
        h_gate_f32[i] = lcg_randf() * 2.0f;  // [-2, 2]
        h_up_f32[i]   = lcg_randf() * 2.0f;
        h_gate[i]     = __float2half(h_gate_f32[i]);
        h_up[i]       = __float2half(h_up_f32[i]);
    }

    // Reference uses FP16-quantised values (round-trip) for a fair comparison.
    for (int i = 0; i < n; ++i) h_gate_f32[i] = __half2float(h_gate[i]);
    for (int i = 0; i < n; ++i) h_up_f32[i]   = __half2float(h_up[i]);

    ref_swiglu(h_ref, h_gate_f32, h_up_f32, n);

    half *d_gate, *d_up, *d_out;
    CUDA_CHECK(cudaMalloc(&d_gate, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_up,   n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out,  n * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_gate, h_gate, n * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up,   h_up,   n * sizeof(half), cudaMemcpyHostToDevice));

    launch_swiglu(d_out, d_gate, d_up, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(half), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(__half2float(h_out[i]) - h_ref[i]);
        if (diff > max_err) max_err = diff;
    }

    const float tol = 1e-3f;
    bool passed = (max_err < tol);
    printf("[%s] %-48s max_err=%.6f  %s\n",
           passed ? "PASS" : "FAIL", name, max_err, passed ? "" : "<-- EXCEEDS 1e-3");

    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_out);
    delete[] h_gate_f32; delete[] h_up_f32; delete[] h_ref;
    delete[] h_gate;     delete[] h_up;     delete[] h_out;

    return passed;
}

// ---------------------------------------------------------------------------
// Benchmark  (CUDA events, no host alloc — pure kernel timing)
// Bytes = read gate [n] + read up [n] + write out [n], all half.
// ---------------------------------------------------------------------------
static void run_benchmark(const char* name, int n,
                          int warmup = 10, int iters = 200) {
    half *d_gate, *d_up, *d_out;
    CUDA_CHECK(cudaMalloc(&d_gate, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_up,   n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out,  n * sizeof(half)));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    for (int i = 0; i < warmup; ++i)
        launch_swiglu(d_out, d_gate, d_up, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < iters; ++i)
        launch_swiglu(d_out, d_gate, d_up, n);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    const float us    = ms * 1000.0f / iters;
    const float bytes = (float)(3 * n) * sizeof(half);
    const float bw_gb = bytes / (us * 1e-6f) / 1e9f;

    printf("[BENCH] %-48s  %7.2f us  %6.1f GB/s\n", name, us, bw_gb);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("=== SwiGLU kernel tests ===\n");

    bool all_pass = true;
    all_pass &= run_test("SwiGLU n=3072         (Qwen3-0.6B, 1 token)",  3072);
    all_pass &= run_test("SwiGLU n=3072*16      (Qwen3-0.6B, seq=16)",   3072 * 16);
    all_pass &= run_test("SwiGLU n=9728         (Qwen3-4B, 1 token)",    9728);
    all_pass &= run_test("SwiGLU n=777          (non-multiple of 256)",  777);

    printf("\n%s\n", all_pass ? "All tests PASSED." : "Some tests FAILED.");

    printf("\n=== SwiGLU benchmarks (warmup=10, iters=200) ===\n");
    run_benchmark("SwiGLU n=3072          (Qwen3-0.6B, 1 token)",   3072);
    run_benchmark("SwiGLU n=9728          (Qwen3-4B,   1 token)",   9728);
    run_benchmark("SwiGLU n=3072*2048     (Qwen3-0.6B, seq=2048)",  3072 * 2048);
    run_benchmark("SwiGLU n=9728*2048     (Qwen3-4B,   seq=2048)",  9728 * 2048);

    return all_pass ? 0 : 1;
}
