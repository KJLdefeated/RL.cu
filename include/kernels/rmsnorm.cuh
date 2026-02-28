#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// RMSNorm: out[i] = (x[i] / rms(x_row)) * weight[i]
// rms(x) = sqrt(mean(x^2) + eps)
//
// Shape: x, out -> [rows, cols]  (half)
//        weight -> [cols]        (half)
// FP16 I/O, FP32 accumulation internally.
//
// Supported configs (Qwen3):
//   cols=128  -> head_dim (q_norm / k_norm)  -> 1 warp  (32 threads)
//   cols=1024 -> hidden_size (layernorm)     -> 4 warps (128 threads)
void launch_rmsnorm(
    half*        out,
    const half*  x,
    const half*  weight,
    int          rows,
    int          cols,
    float        eps,
    cudaStream_t stream = 0
);
