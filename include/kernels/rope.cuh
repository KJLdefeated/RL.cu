#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void launch_rope_precompute(
    float*       cos_table,
    float*       sin_table,
    int          max_seq_len,
    int          head_dim,
    float        rope_theta,
    cudaStream_t stream = 0
);

void launch_rope(
    half*        Q,
    half*        K,
    const float* cos_table,
    const float* sin_table,
    const int*   position_ids,
    int          num_tokens,
    int          num_q_heads,
    int          num_kv_heads,
    int          head_dim,
    cudaStream_t stream = 0
);

// Backward pass for RoPE: inverse rotation (transpose = negate sin).
//   dX[i]      = dOut[i]*cos + dOut[i+D/2]*sin
//   dX[i+D/2]  = dOut[i+D/2]*cos - dOut[i]*sin
// Computes dQ and dK in-place (overwrites dQ, dK with rotated-back gradients).
void launch_rope_backward(
    half*        dQ,            // [num_tokens, H_q, D]  in/out
    half*        dK,            // [num_tokens, H_kv, D] in/out
    const float* cos_table,     // [max_seq_len, D/2]
    const float* sin_table,     // [max_seq_len, D/2]
    const int*   position_ids,  // [num_tokens]
    int          num_tokens,
    int          num_q_heads,
    int          num_kv_heads,
    int          head_dim,
    cudaStream_t stream = 0
);
