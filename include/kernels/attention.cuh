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

// Cache-reading causal prefill: new-chunk queries attend over the paged KV cache
// (cached prefix + new chunk), causally by absolute position. Enables continued
// (chunked) prefill for multi-turn rollouts. See src/kernels/attention.cu.
//   q          : [num_q, H_q, head_dim]   dense new-chunk queries
//   out        : [num_q, H_q, head_dim]
//   q_seq_idx  : [num_q]  sequence index per query row
//   q_abs_pos  : [num_q]  absolute position per query row (<0 = padding → zeroed)
void launch_flash_attention_prefill_paged(
    const half*  q,
    const half*  k_cache,
    const half*  v_cache,
    half*        out,
    const int*   block_tables,
    const int*   q_seq_idx,
    const int*   q_abs_pos,
    int num_q, int H_q, int H_kv, int head_dim,
    int max_blocks_per_seq, int block_size,
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

// Flash Decoding with Split-K: splits KV context across multiple thread blocks
// for higher GPU occupancy on long sequences.
// Workspace: partial_out [num_seqs * H_q * num_splits * head_dim] floats
//            partial_max [num_seqs * H_q * num_splits] floats
//            partial_sum [num_seqs * H_q * num_splits] floats
void launch_flash_decode_splitk(
    const half*  q,
    const half*  k_cache,
    const half*  v_cache,
    half*        out,
    float*       partial_out,    // workspace
    float*       partial_max,    // workspace
    float*       partial_sum,    // workspace
    const int*   block_tables,
    const int*   seq_lens,
    int num_seqs, int H_q, int H_kv, int head_dim,
    int max_blocks_per_seq, int block_size,
    int num_splits,
    cudaStream_t stream = 0
);
