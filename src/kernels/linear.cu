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

    // ---- dX = dY @ W  [M,N] × [N,K] → [M,K] ----
    // Row-major: dX = dY × W
    // cuBLAS col-major: dX^T = W^T × dY^T
    //   op(A)=W^T  → CUBLAS_OP_T on W[N,K], lda=K
    //   BUT we want W not transposed this time:
    //   dX[M,K] = dY[M,N] × W[N,K]
    //   Col-major: dX^T[K,M] = W^T[K,N] × dY^T[N,M]
    //   → op(A)=CUBLAS_OP_N on W[N,K] (col-major sees [K,N]), m=K, n=M, k=N
    if (dX) {
        const float beta_dx = 0.0f;
        status = cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,   // W not transposed, dY not transposed
            K, M, N,                     // m, n, k
            &alpha,
            W,  CUDA_R_16F, K,          // A=[N,K] col-major sees [K,N], lda=K
            dY, CUDA_R_16F, N,          // B=[M,N] col-major sees [N,M], ldb=N
            &beta_dx,
            dX, CUDA_R_16F, K,          // C=[M,K] col-major sees [K,M], ldc=K
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "linear_backward dX failed: status=%d M=%d N=%d K=%d\n",
                    (int)status, M, N, K);
            throw std::runtime_error("cublasGemmEx failed in linear_backward (dX)");
        }
    }

    // ---- dW = dY^T @ X  [N,M] × [M,K] → [N,K] ----
    // Row-major: dW = dY^T × X
    // Col-major: dW^T[K,N] = X^T[K,M] × dY[M,N]
    //   op(A)=CUBLAS_OP_N on X[M,K] (col-major [K,M]), m=K, n=N, k=M
    //   op(B)=CUBLAS_OP_N on dY[M,N] (col-major [N,M])
    //   Wait — that gives [K,N] = [K,M]×[M,N] ✓
    // beta=1 to accumulate gradients across micro-batches
    const float beta_dw = 1.0f;
    status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,   // X not transposed, dY transposed
        K, N, M,                     // m, n, k
        &alpha,
        X,  CUDA_R_16F, K,          // A=[M,K] col-major sees [K,M], lda=K
        dY, CUDA_R_16F, N,          // B=[M,N] col-major sees [N,M], ldb=N → op_T sees [M,N]
        &beta_dw,
        dW, CUDA_R_16F, K,          // C=[N,K] col-major sees [K,N], ldc=K
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "linear_backward dW failed: status=%d M=%d N=%d K=%d\n",
                (int)status, M, N, K);
        throw std::runtime_error("cublasGemmEx failed in linear_backward (dW)");
    }
}
