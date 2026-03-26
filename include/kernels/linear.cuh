#pragma once
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void linear_half(
    cublasHandle_t handle,
    const half* input,   // [M, K]
    const half* weight,  // [N, K] (row-major, transposed for GEMM)
    half*       output,  // [M, N]
    int M, int N, int K
);

// Backward pass for linear layer: Y = X @ W^T
//   dX = dY @ W          [M,N] × [N,K] → [M,K]
//   dW = dY^T @ X        [N,M] × [M,K] → [N,K]
// If dX is nullptr, skip dX computation (e.g. first layer, no need to backprop further).
// dW is accumulated (beta=1) so caller must zero it before the first backward call
// if not accumulating across micro-batches.
void linear_backward_half(
    cublasHandle_t handle,
    const half* dY,      // [M, N] upstream gradient
    const half* X,       // [M, K] saved input from forward
    const half* W,       // [N, K] weight
    half*       dX,      // [M, K] gradient for input  (may be nullptr)
    half*       dW,      // [N, K] gradient for weight  (accumulated)
    int M, int N, int K
);
