#include "kernels/linear.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

void linear_half(
    cublasHandle_t handle,
    const half* input,
    const half* weight,
    half*       output,
    int M, int N, int K
) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,   // op(A)=weight^T, op(B)=input
        N, M, K,                     // m, n, k
        &alpha,
        weight, CUDA_R_16F, K,       // A, type, lda
        input,  CUDA_R_16F, K,       // B, type, ldb
        &beta,
        output, CUDA_R_16F, N,       // C, type, ldc
        CUDA_R_32F,                  // compute type (FP32 accumulation)
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasGemmEx failed: status=%d M=%d N=%d K=%d\n",
                (int)status, M, N, K);
        throw std::runtime_error("cublasGemmEx failed in linear_half");
    }
}
