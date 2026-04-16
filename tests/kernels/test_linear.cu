// test_linear.cu
// Validates linear_half (cuBLAS FP16 GEMM wrapper) against a CPU FP32 reference.
//
// Pass criterion: max_err < 1e-2
//   cuBLAS uses FP32 accumulation (CUDA_R_32F compute type).  The only error
//   source is FP16 quantisation of inputs/weights and FP32 rounding in the
//   partial sums, which is well within 1e-2 for K ≤ 3072.
//
// Test cases:
//   1. Tiny 4×4×4 sanity check
//   2. Square 64×64×64
//   3. Qwen3 Q-proj    : [32, 1024] × [2048, 1024]^T → [32, 2048]
//   4. Qwen3 K/V-proj  : [32, 1024] × [1024, 1024]^T → [32, 1024]
//   5. Qwen3 O-proj    : [32, 2048] × [1024, 2048]^T → [32, 1024]
//   6. Qwen3 gate/up   : [32, 1024] × [3072, 1024]^T → [32, 3072]
//   7. Qwen3 down-proj : [32, 3072] × [1024, 3072]^T → [32, 1024]
//   8. Prefill Q-proj  : [512, 1024] × [2048, 1024]^T → [512, 2048]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/linear.cuh"

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

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t s = (call);                                             \
        if (s != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error at %s:%d — %d\n",                   \
                    __FILE__, __LINE__, (int)s);                               \
            exit(EXIT_FAILURE);                                                \
        }                                                                     \
    } while (0)

static unsigned int lcg_state = 12345u;
static float lcg_randf() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(lcg_state >> 1) / (float)0x7fffffffu) * 2.0f - 1.0f;
}

// ---------------------------------------------------------------------------
// CPU reference: out [M, N] = input [M, K] @ weight^T [K, N]
// All arithmetic in FP32; inputs already rounded to FP16 range.
// ---------------------------------------------------------------------------
static void ref_linear(
    float*       out,     // [M, N]
    const float* input,   // [M, K]
    const float* weight,  // [N, K]
    int M, int N, int K
) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += input[m * K + k] * weight[n * K + k];
            out[m * N + n] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
static bool run_test(
    cublasHandle_t handle,
    const char* name,
    int M, int N, int K,
    float tol = 1e-2f
) {
    const long inp_sz = (long)M * K;
    const long wgt_sz = (long)N * K;
    const long out_sz = (long)M * N;

    // Generate FP32 random data, quantise to FP16, round-trip back to FP32
    // so the CPU reference uses identical values to the GPU kernel.
    float* h_inp_f32 = new float[inp_sz];
    float* h_wgt_f32 = new float[wgt_sz];
    half*  h_inp_h16 = new half[inp_sz];
    half*  h_wgt_h16 = new half[wgt_sz];

    for (long i = 0; i < inp_sz; i++) {
        h_inp_h16[i] = __float2half(lcg_randf());
        h_inp_f32[i] = __half2float(h_inp_h16[i]);
    }
    for (long i = 0; i < wgt_sz; i++) {
        h_wgt_h16[i] = __float2half(lcg_randf());
        h_wgt_f32[i] = __half2float(h_wgt_h16[i]);
    }

    // CPU reference
    float* h_ref = new float[out_sz];
    ref_linear(h_ref, h_inp_f32, h_wgt_f32, M, N, K);

    // GPU
    half *d_inp, *d_wgt, *d_out;
    CUDA_CHECK(cudaMalloc(&d_inp, inp_sz * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_wgt, wgt_sz * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, out_sz * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_inp, h_inp_h16, inp_sz * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wgt, h_wgt_h16, wgt_sz * sizeof(half), cudaMemcpyHostToDevice));

    linear_half(handle, d_inp, d_wgt, d_out, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    half* h_out = new half[out_sz];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_sz * sizeof(half), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (long i = 0; i < out_sz; i++) {
        float diff = fabsf(__half2float(h_out[i]) - h_ref[i]);
        if (diff > max_err) max_err = diff;
    }

    const bool passed = (max_err < tol);
    printf("[%s] %-56s max_err=%.6f  (tol=%.0e)  %s\n",
           passed ? "PASS" : "FAIL", name, max_err, tol,
           passed ? "" : "<-- FAIL");

    CUDA_CHECK(cudaFree(d_inp));
    CUDA_CHECK(cudaFree(d_wgt));
    CUDA_CHECK(cudaFree(d_out));
    delete[] h_inp_f32; delete[] h_inp_h16;
    delete[] h_wgt_f32; delete[] h_wgt_h16;
    delete[] h_ref; delete[] h_out;

    return passed;
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------
static void run_benchmark(
    cublasHandle_t handle,
    const char* name,
    int M, int N, int K,
    int warmup = 20, int iters = 500
) {
    const long inp_sz = (long)M * K;
    const long wgt_sz = (long)N * K;
    const long out_sz = (long)M * N;

    half *d_inp, *d_wgt, *d_out;
    CUDA_CHECK(cudaMalloc(&d_inp, inp_sz * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_wgt, wgt_sz * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, out_sz * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_inp, 0, inp_sz * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_wgt, 0, wgt_sz * sizeof(half)));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    for (int i = 0; i < warmup; i++)
        linear_half(handle, d_inp, d_wgt, d_out, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < iters; i++)
        linear_half(handle, d_inp, d_wgt, d_out, M, N, K);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    float us = ms * 1000.0f / iters;

    // FLOPS: 2 * M * N * K (mul + add per element)
    double flops  = 2.0 * M * N * K;
    double tflops = flops / (us * 1e-6) / 1e12;

    printf("[BENCH] %-54s  %6.2f us  %5.2f TFLOP/s\n", name, us, tflops);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaFree(d_inp));
    CUDA_CHECK(cudaFree(d_wgt));
    CUDA_CHECK(cudaFree(d_out));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("=== linear_half (cuBLAS FP16 GEMM) tests ===\n\n");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    bool all_pass = true;

    // 1. Tiny sanity
    all_pass &= run_test(handle, "tiny 4x4x4",                4,    4,    4);
    // 2. Square
    all_pass &= run_test(handle, "square 64x64x64",          64,   64,   64);
    // 3. Qwen3 Q-proj  [32, 1024] @ [2048, 1024]^T
    all_pass &= run_test(handle, "Qwen3 Q-proj   M=32  K=1024 N=2048",  32, 2048, 1024);
    // 4. Qwen3 K-proj  [32, 1024] @ [1024, 1024]^T
    all_pass &= run_test(handle, "Qwen3 K-proj   M=32  K=1024 N=1024",  32, 1024, 1024);
    // 5. Qwen3 O-proj  [32, 2048] @ [1024, 2048]^T
    all_pass &= run_test(handle, "Qwen3 O-proj   M=32  K=2048 N=1024",  32, 1024, 2048);
    // 6. Qwen3 gate/up [32, 1024] @ [3072, 1024]^T
    all_pass &= run_test(handle, "Qwen3 gate/up  M=32  K=1024 N=3072",  32, 3072, 1024);
    // 7. Qwen3 down    [32, 3072] @ [1024, 3072]^T
    all_pass &= run_test(handle, "Qwen3 down     M=32  K=3072 N=1024",  32, 1024, 3072);
    // 8. Prefill Q-proj [512, 1024] @ [2048, 1024]^T
    all_pass &= run_test(handle, "Qwen3 Q-proj   M=512 K=1024 N=2048", 512, 2048, 1024);

    printf("\n%s\n", all_pass ? "All tests PASSED." : "Some tests FAILED.");

    printf("\n=== linear_half benchmarks (warmup=20, iters=500) ===\n");
    // Decode batch sizes (M = num tokens in flight)
    run_benchmark(handle, "decode Q-proj   M=1   K=1024 N=2048",    1, 2048, 1024);
    run_benchmark(handle, "decode Q-proj   M=32  K=1024 N=2048",   32, 2048, 1024);
    run_benchmark(handle, "prefill Q-proj  M=512 K=1024 N=2048",  512, 2048, 1024);
    run_benchmark(handle, "prefill down    M=512 K=3072 N=1024",  512, 1024, 3072);

    CUBLAS_CHECK(cublasDestroy(handle));
    return all_pass ? 0 : 1;
}
