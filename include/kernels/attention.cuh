#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void launch_flash_attention_prefill(
    const half*  Q,
    const half*  K,
    const half*  V,
    half*        O,
    int B, int S, int H_q, int H_kv, int head_dim,
    cudaStream_t stream = 0,
    float*       lse = nullptr   // [B, S, H_q] logsumexp — optional, for backward
);

// Flash Attention 2 backward pass (prefill/training only).
// Requires LSE saved from forward pass.
// D_buf is a workspace of size [B * S * H_q] (FP32).
// dQ, dK, dV must be pre-allocated (same shapes as Q, K, V).
void launch_flash_attention_backward(
    const half*  Q,       // [B, S, H_q,  D]
    const half*  K,       // [B, S, H_kv, D]
    const half*  V,       // [B, S, H_kv, D]
    const half*  O,       // [B, S, H_q,  D]  (saved output from forward)
    const half*  dO,      // [B, S, H_q,  D]  (upstream gradient)
    const float* lse,     // [B, S, H_q]      (logsumexp from forward)
    float*       D_buf,   // [B, S, H_q]      (workspace for D = dot(O, dO))
    half*        dQ,      // [B, S, H_q,  D]
    half*        dK,      // [B, S, H_kv, D]
    half*        dV,      // [B, S, H_kv, D]
    int B, int S, int H_q, int H_kv, int head_dim,
    cudaStream_t stream = 0
);

void launch_paged_attention_decode(
    const half*  q,
    const half*  k_cache,
    const half*  v_cache,
    half*        out,
    const int*   block_tables,
    const int*   seq_lens,
    int num_seqs, int H_q, int H_kv, int head_dim,
    int max_blocks_per_seq, int block_size,
    cudaStream_t stream = 0
);
