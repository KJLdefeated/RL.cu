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
