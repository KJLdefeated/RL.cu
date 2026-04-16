// test_linear_backward.cu
// Validates linear_backward_half against CPU FP32 reference.
//
// Forward: Y[M,N] = X[M,K] @ W[N,K]^T
// Backward:
//   dX[M,K] = dY[M,N] @ W[N,K]
//   dW[N,K] = dY[M,N]^T @ X[M,K]  (accumulated: beta=1)
//
// Pass criterion: max_err < tol, where tol scales with reduction dimension
//   dX reduces over N, dW reduces over M — larger reductions accumulate more FP16 rounding
//
// Test cases:
//   1. Tiny 4×4×4
//   2. Square 64×64×64
//   3. Qwen3 Q-proj backward  : M=32, N=2048, K=1024
//   4. Qwen3 K/V-proj backward: M=32, N=1024, K=1024
//   5. Qwen3 O-proj backward  : M=32, N=1024, K=2048
//   6. Qwen3 gate/up backward : M=32, N=3072, K=1024
//   7. Qwen3 down backward    : M=32, N=1024, K=3072
//   8. Prefill Q-proj backward: M=512, N=2048, K=1024
//   9. dX=nullptr (skip dX computation)
//  10. dW accumulation (call backward twice, verify dW = sum)

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

static unsigned int lcg_state = 54321u;
static float lcg_randf() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(lcg_state >> 1) / (float)0x7fffffffu) * 2.0f - 1.0f;
}

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------

// dX[M,K] = dY[M,N] @ W[N,K]
static void ref_linear_dx(
    float*       dX,     // [M, K]
    const float* dY,     // [M, N]
    const float* W,      // [N, K]
    int M, int N, int K
) {
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            float acc = 0.0f;
            for (int n = 0; n < N; n++)
                acc += dY[m * N + n] * W[n * K + k];
            dX[m * K + k] = acc;
        }
    }
}

// dW[N,K] = dY[M,N]^T @ X[M,K]  (accumulated onto existing dW)
static void ref_linear_dw(
    float*       dW,     // [N, K]  (accumulated)
    const float* dY,     // [M, N]
    const float* X,      // [M, K]
    int M, int N, int K
) {
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            float acc = 0.0f;
            for (int m = 0; m < M; m++)
                acc += dY[m * N + n] * X[m * K + k];
            dW[n * K + k] += acc;
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
    bool test_dx = true
) {
    const long dy_sz = (long)M * N;
    const long x_sz  = (long)M * K;
    const long w_sz  = (long)N * K;

    // Generate random data in FP32, round-trip through FP16
    float* h_dY_f32 = new float[dy_sz];
    float* h_X_f32  = new float[x_sz];
    float* h_W_f32  = new float[w_sz];
    half*  h_dY_h16 = new half[dy_sz];
    half*  h_X_h16  = new half[x_sz];
    half*  h_W_h16  = new half[w_sz];

    for (long i = 0; i < dy_sz; i++) {
        h_dY_h16[i] = __float2half(lcg_randf());
        h_dY_f32[i] = __half2float(h_dY_h16[i]);
    }
    for (long i = 0; i < x_sz; i++) {
        h_X_h16[i] = __float2half(lcg_randf());
        h_X_f32[i] = __half2float(h_X_h16[i]);
    }
    for (long i = 0; i < w_sz; i++) {
        h_W_h16[i] = __float2half(lcg_randf());
        h_W_f32[i] = __half2float(h_W_h16[i]);
    }

    // CPU reference
    float* h_ref_dX = new float[x_sz]();
    float* h_ref_dW = new float[w_sz]();
    if (test_dx)
        ref_linear_dx(h_ref_dX, h_dY_f32, h_W_f32, M, N, K);
    ref_linear_dw(h_ref_dW, h_dY_f32, h_X_f32, M, N, K);

    // GPU allocations
    half *d_dY, *d_X, *d_W, *d_dX, *d_dW;
    CUDA_CHECK(cudaMalloc(&d_dY, dy_sz * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_X,  x_sz  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_W,  w_sz  * sizeof(half)));
    if (test_dx)
        CUDA_CHECK(cudaMalloc(&d_dX, x_sz * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dW, w_sz * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_dY, h_dY_h16, dy_sz * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X,  h_X_h16,  x_sz  * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W,  h_W_h16,  w_sz  * sizeof(half), cudaMemcpyHostToDevice));
    // Zero dW before backward (since beta=1 accumulates)
    CUDA_CHECK(cudaMemset(d_dW, 0, w_sz * sizeof(half)));

    linear_backward_half(handle, d_dY, d_X, d_W,
                         test_dx ? d_dX : nullptr, d_dW,
                         M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Tolerance scales with reduction dimension:
    //   dX reduces over N, dW reduces over M
    //   FP16 GEMM error ~ sqrt(reduce_dim) * eps_fp16
    float tol_dx = 1e-2f * sqrtf((float)N / 64.0f);
    float tol_dw = 1e-2f * sqrtf((float)M / 64.0f);

    // Verify dX
    float max_err_dx = 0.0f;
    if (test_dx) {
        half* h_dX = new half[x_sz];
        CUDA_CHECK(cudaMemcpy(h_dX, d_dX, x_sz * sizeof(half), cudaMemcpyDeviceToHost));
        for (long i = 0; i < x_sz; i++) {
            float diff = fabsf(__half2float(h_dX[i]) - h_ref_dX[i]);
            if (diff > max_err_dx) max_err_dx = diff;
        }
        delete[] h_dX;
    }

    // Verify dW
    half* h_dW = new half[w_sz];
    CUDA_CHECK(cudaMemcpy(h_dW, d_dW, w_sz * sizeof(half), cudaMemcpyDeviceToHost));
    float max_err_dw = 0.0f;
    for (long i = 0; i < w_sz; i++) {
        float diff = fabsf(__half2float(h_dW[i]) - h_ref_dW[i]);
        if (diff > max_err_dw) max_err_dw = diff;
    }

    bool passed = (max_err_dw < tol_dw) && (!test_dx || max_err_dx < tol_dx);
    if (test_dx) {
        printf("[%s] %-50s dX_err=%.6f(tol=%.3f)  dW_err=%.6f(tol=%.3f)\n",
               passed ? "PASS" : "FAIL", name,
               max_err_dx, tol_dx, max_err_dw, tol_dw);
    } else {
        printf("[%s] %-50s dX=skip              dW_err=%.6f(tol=%.3f)\n",
               passed ? "PASS" : "FAIL", name, max_err_dw, tol_dw);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_dY));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_W));
    if (test_dx) CUDA_CHECK(cudaFree(d_dX));
    CUDA_CHECK(cudaFree(d_dW));
    delete[] h_dY_f32; delete[] h_dY_h16;
    delete[] h_X_f32;  delete[] h_X_h16;
    delete[] h_W_f32;  delete[] h_W_h16;
    delete[] h_ref_dX; delete[] h_ref_dW;
    delete[] h_dW;

    return passed;
}

// ---------------------------------------------------------------------------
// Test dW accumulation: call backward twice with different dY, verify dW = sum
// ---------------------------------------------------------------------------
static bool run_accum_test(
    cublasHandle_t handle,
    int M, int N, int K
) {
    const long dy_sz = (long)M * N;
    const long x_sz  = (long)M * K;
    const long w_sz  = (long)N * K;

    // Two different dY batches, same X and W
    float* h_dY1_f32 = new float[dy_sz];
    float* h_dY2_f32 = new float[dy_sz];
    float* h_X_f32   = new float[x_sz];
    float* h_W_f32   = new float[w_sz];
    half*  h_dY1_h16 = new half[dy_sz];
    half*  h_dY2_h16 = new half[dy_sz];
    half*  h_X_h16   = new half[x_sz];
    half*  h_W_h16   = new half[w_sz];

    for (long i = 0; i < dy_sz; i++) {
        h_dY1_h16[i] = __float2half(lcg_randf());
        h_dY1_f32[i] = __half2float(h_dY1_h16[i]);
    }
    for (long i = 0; i < dy_sz; i++) {
        h_dY2_h16[i] = __float2half(lcg_randf());
        h_dY2_f32[i] = __half2float(h_dY2_h16[i]);
    }
    for (long i = 0; i < x_sz; i++) {
        h_X_h16[i] = __float2half(lcg_randf());
        h_X_f32[i] = __half2float(h_X_h16[i]);
    }
    for (long i = 0; i < w_sz; i++) {
        h_W_h16[i] = __float2half(lcg_randf());
        h_W_f32[i] = __half2float(h_W_h16[i]);
    }

    // CPU reference: dW = dY1^T @ X + dY2^T @ X
    float* h_ref_dW = new float[w_sz]();
    ref_linear_dw(h_ref_dW, h_dY1_f32, h_X_f32, M, N, K);
    ref_linear_dw(h_ref_dW, h_dY2_f32, h_X_f32, M, N, K);

    // GPU
    half *d_dY1, *d_dY2, *d_X, *d_W, *d_dW;
    CUDA_CHECK(cudaMalloc(&d_dY1, dy_sz * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dY2, dy_sz * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_X,   x_sz  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_W,   w_sz  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dW,  w_sz  * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_dY1, h_dY1_h16, dy_sz * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dY2, h_dY2_h16, dy_sz * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X,   h_X_h16,   x_sz  * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W,   h_W_h16,   w_sz  * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dW, 0, w_sz * sizeof(half)));

    // Two backward calls — dW should accumulate
    linear_backward_half(handle, d_dY1, d_X, d_W, nullptr, d_dW, M, N, K);
    linear_backward_half(handle, d_dY2, d_X, d_W, nullptr, d_dW, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    half* h_dW = new half[w_sz];
    CUDA_CHECK(cudaMemcpy(h_dW, d_dW, w_sz * sizeof(half), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (long i = 0; i < w_sz; i++) {
        float diff = fabsf(__half2float(h_dW[i]) - h_ref_dW[i]);
        if (diff > max_err) max_err = diff;
    }

    // Larger tolerance: two accumulated GEMMs + FP16 storage truncation
    float accum_tol = 1e-2f * sqrtf((float)M / 64.0f) * 3.0f;
    bool passed = (max_err < accum_tol);
    printf("[%s] %-50s dW_accum_err=%.6f  (tol=%.3f)\n",
           passed ? "PASS" : "FAIL",
           "dW accumulation (2 backward calls)", max_err, accum_tol);

    CUDA_CHECK(cudaFree(d_dY1));
    CUDA_CHECK(cudaFree(d_dY2));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_dW));
    delete[] h_dY1_f32; delete[] h_dY1_h16;
    delete[] h_dY2_f32; delete[] h_dY2_h16;
    delete[] h_X_f32;   delete[] h_X_h16;
    delete[] h_W_f32;   delete[] h_W_h16;
    delete[] h_ref_dW;  delete[] h_dW;

    return passed;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("=== linear_backward_half tests ===\n\n");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    bool all_pass = true;

    // Basic correctness (dX + dW)
    all_pass &= run_test(handle, "tiny 4x4x4",                          4,    4,    4);
    all_pass &= run_test(handle, "square 64x64x64",                    64,   64,   64);
    all_pass &= run_test(handle, "Qwen3 Q-proj   M=32  N=2048 K=1024", 32, 2048, 1024);
    all_pass &= run_test(handle, "Qwen3 K-proj   M=32  N=1024 K=1024", 32, 1024, 1024);
    all_pass &= run_test(handle, "Qwen3 O-proj   M=32  N=1024 K=2048", 32, 1024, 2048);
    all_pass &= run_test(handle, "Qwen3 gate/up  M=32  N=3072 K=1024", 32, 3072, 1024);
    all_pass &= run_test(handle, "Qwen3 down     M=32  N=1024 K=3072", 32, 1024, 3072);
    all_pass &= run_test(handle, "Prefill Q-proj M=512 N=2048 K=1024",512, 2048, 1024);

    // dX=nullptr (skip input gradient)
    all_pass &= run_test(handle, "dX=nullptr M=32 N=2048 K=1024",      32, 2048, 1024, false);

    // dW accumulation across two backward calls
    all_pass &= run_accum_test(handle, 32, 2048, 1024);

    printf("\n%s\n", all_pass ? "All tests PASSED." : "Some tests FAILED.");

    CUBLAS_CHECK(cublasDestroy(handle));
    return all_pass ? 0 : 1;
}
