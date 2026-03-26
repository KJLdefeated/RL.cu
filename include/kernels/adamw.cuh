#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ============================================================
// Fused AdamW optimizer kernels
//
// Two variants:
//   1. FP16 weights: reads FP32 master copy + FP16 gradient,
//      updates m/v/master in FP32, writes back FP16 model weight.
//   2. FP32 weights: in-place update with FP32 gradient.
//
// AdamW update (decoupled weight decay):
//   m = beta1 * m + (1 - beta1) * grad
//   v = beta2 * v + (1 - beta2) * grad^2
//   m_hat = m / (1 - beta1^t)
//   v_hat = v / (1 - beta2^t)
//   w = w * (1 - lr * wd) - lr * m_hat / (sqrt(v_hat) + eps)
// ============================================================

// Update FP16 weight parameter using FP32 master weight.
// master_w, m, v are FP32; model_w is FP16 (written after update);
// grad is FP16.
void launch_adamw_fp16(
    float*       master_w,    // [n] FP32 master weights (read + write)
    half*        model_w,     // [n] FP16 model weights (write only)
    const half*  grad,        // [n] FP16 gradient
    float*       m,           // [n] first moment (read + write)
    float*       v,           // [n] second moment (read + write)
    int          n,
    float        lr,
    float        beta1,
    float        beta2,
    float        eps,
    float        weight_decay,
    float        bias_correction1,  // 1 - beta1^t
    float        bias_correction2,  // 1 - beta2^t
    cudaStream_t stream = 0
);

// Update FP32 weight parameter in-place.
// All buffers are FP32.
void launch_adamw_fp32(
    float*       w,           // [n] FP32 weights (read + write)
    const float* grad,        // [n] FP32 gradient
    float*       m,           // [n] first moment (read + write)
    float*       v,           // [n] second moment (read + write)
    int          n,
    float        lr,
    float        beta1,
    float        beta2,
    float        eps,
    float        weight_decay,
    float        bias_correction1,
    float        bias_correction2,
    cudaStream_t stream = 0
);
