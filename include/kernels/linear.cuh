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
