#include "kernels/linear.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

// =============================================================================
// linear_half — FP16 GEMM via cuBLAS
// =============================================================================
// Computes: output [M, N] = input [M, K] @ weight^T [K, N]
//
// cuBLAS is column-major; we exploit the row/col-major duality:
//
//   row-major [M, K]  stored identically as  col-major [K, M]
//   row-major [N, K]  stored identically as  col-major [K, N]
//
// So we ask cuBLAS to compute (in its col-major view):
//
//   C [N, M] = op(A) * op(B)
//     A = weight  col-major [K, N], lda = K  →  op(A) = A^T = [N, K]
//     B = input   col-major [K, M], ldb = K  →  op(B) = B    = [K, M]
//     C = output  col-major [N, M], ldc = N  →  row-major [M, N]  ✓
//
// cublasGemmEx args: m=N, n=M, k=K
// =============================================================================

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

    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasGemmEx failed in linear_half");
}
