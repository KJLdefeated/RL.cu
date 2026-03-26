#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void launch_rmsnorm(
    half*         out,
    const half*   x,
    const float*  weight,
    int          rows,
    int          cols,
    float        eps,
    cudaStream_t stream = 0
);

// Backward pass for RMSNorm.
// Forward: y_i = w_i * x_i / sqrt(mean(x^2) + eps)
//
// dX_j = rms_inv * (dY_j * w_j  -  x_j * rms_inv^2 * c / N)
//   where c = sum_i(dY_i * w_i * x_i)
//
// dW_i += sum_over_rows(dY_i * x_i * rms_inv)   (accumulated)
//
// dW is FP32 (same type as weight). Caller must zero dW before first call.
void launch_rmsnorm_backward(
    half*         dX,      // [rows, cols] gradient for input
    float*        dW,      // [cols]       gradient for weight (accumulated across rows)
    const half*   dY,      // [rows, cols] upstream gradient
    const half*   x,       // [rows, cols] saved input from forward
    const float*  weight,  // [cols]       RMSNorm weight
    int           rows,
    int           cols,
    float         eps,
    cudaStream_t  stream = 0
);
