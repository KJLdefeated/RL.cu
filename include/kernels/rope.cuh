#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// RoPE — Rotary Position Embedding, NeoX split-half variant
// ---------------------------------------------------------------------------
// Qwen3 uses the "split-half" rotation (same as Llama / HuggingFace default):
//
//   For token at position p, head h, dimension index i ∈ [0, D):
//     freq_i  = 1 / (rope_theta ^ (2 * (i % half_dim) / D))
//     angle   = p * freq_i
//     c = cos(angle),  s = sin(angle)
//
//     if i < half_dim:  out[i]         = x[i]*c - x[i+half_dim]*s
//     else:             out[i]         = x[i]*c + x[i-half_dim]*s
//
// This matches Python:  x * cos + rotate_half(x) * sin
//   where rotate_half(x) = cat([-x[..., D//2:], x[..., :D//2]], dim=-1)
//
// Applied AFTER QK-Norm (critical for Qwen3 correctness).
// FP16 I/O; sin/cos table is FP32 for precision.
// ---------------------------------------------------------------------------

// Precompute sin/cos tables — call once at model initialisation.
//
//   cos_table, sin_table : [max_seq_len, head_dim/2]  (FP32, GPU memory)
//   rope_theta           : Qwen3 = 1,000,000
void launch_rope_precompute(
    float*       cos_table,
    float*       sin_table,
    int          max_seq_len,
    int          head_dim,
    float        rope_theta,
    cudaStream_t stream = 0
);

// Apply RoPE in-place to Q and K.
//
//   Q : [num_tokens, num_q_heads,  head_dim]  (FP16, in-place)
//   K : [num_tokens, num_kv_heads, head_dim]  (FP16, in-place)
//   cos_table, sin_table : [max_seq_len, head_dim/2]  (FP32)
//   position_ids         : [num_tokens]  — global token positions (int32)
//
// num_tokens = B * S (flattened batch × sequence).
// GQA: H_q >= H_kv; RoPE applied independently to each head in both tensors.
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
