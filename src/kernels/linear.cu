#include "kernels/linear.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>

// =============================================================================
// linear_half — forward projection: output[M,N] = input[M,K] × weight^T[K,N]
// Uses cuBLAS GemmEx with FP32 accumulation and tensor cores.
// =============================================================================

void linear_half(
    cublasHandle_t handle,
    const half* input,
    const half* weight,
    half*       output,
    int M, int N, int K,
    cudaStream_t stream
) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // cuBLAS is column-major; to compute C[M,N] = A[M,K] × B^T[K,N] in row-major
    // we compute C^T[N,M] = B[N,K] × A^T[K,M]:
    //   op(B) = N (no transpose), dims (N × K)
    //   op(A) = T (transpose),   dims (K × M)
    //   result: C^T[N, M]
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,   // op(B)=N, op(A)=T (col-major view)
        N, M, K,
        &alpha,
        weight, CUDA_R_16F, K,      // B[N,K], ldb=K
        input,  CUDA_R_16F, K,      // A[M,K], lda=K
        &beta,
        output, CUDA_R_16F, N,      // C[M,N], ldc=N
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "linear_half cublasGemmEx failed: status=%d  M=%d N=%d K=%d\n",
                (int)status, M, N, K);
        throw std::runtime_error("linear_half: cublasGemmEx failed");
    }
}

// =============================================================================
// linear_backward_half — weight + input gradients (cuBLAS; large M in training)
// =============================================================================

void linear_backward_half(
    cublasHandle_t handle,
    const half* dY,      // [M, N]
    const half* X,       // [M, K]
    const half* W,       // [N, K]
    half*       dX,      // [M, K]  (may be nullptr)
    half*       dW,      // [N, K]  (accumulated, beta=1)
    int M, int N, int K
) {
    const float alpha = 1.0f;
    cublasStatus_t status;

    // dX = dY @ W  →  col-major: dX^T[K,M] = W^T[K,N] × dY^T[N,M]
    if (dX) {
        const float beta_dx = 0.0f;
        status = cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            K, M, N,
            &alpha,
            W,  CUDA_R_16F, K,
            dY, CUDA_R_16F, N,
            &beta_dx,
            dX, CUDA_R_16F, K,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "linear_backward dX failed: status=%d\n", (int)status);
            throw std::runtime_error("cublasGemmEx failed in linear_backward (dX)");
        }
    }

    // dW = dY^T @ X  →  col-major: dW^T[K,N] = X^T[K,M] × dY[M,N]
    const float beta_dw = 1.0f;
    status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        K, N, M,
        &alpha,
        X,  CUDA_R_16F, K,
        dY, CUDA_R_16F, N,
        &beta_dw,
        dW, CUDA_R_16F, K,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "linear_backward dW failed: status=%d\n", (int)status);
        throw std::runtime_error("cublasGemmEx failed in linear_backward (dW)");
    }
}
