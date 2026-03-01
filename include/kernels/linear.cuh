#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Linear projection — FP16 input × FP16 weight → FP16 output
// ---------------------------------------------------------------------------
// Computes: output = input @ weight^T
//
//   input  : [M, K]  (FP16, row-major)  — M tokens, K in-features
//   weight : [N, K]  (FP16, row-major)  — N out-features, K in-features
//   output : [M, N]  (FP16, row-major)
//
// Accumulation in FP32 (CUDA_R_32F compute type) via cublasGemmEx with
// CUBLAS_GEMM_DEFAULT_TENSOR_OP (uses Tensor Cores on sm_80+).
//
// Bias is intentionally omitted — all Qwen3 projections are bias-free.
// ---------------------------------------------------------------------------
void linear_half(
    cublasHandle_t handle,
    const half* input,   // [M, K]
    const half* weight,  // [N, K] (row-major, transposed for GEMM)
    half*       output,  // [M, N]
    int M, int N, int K
);
