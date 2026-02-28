#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Flash Attention 2 — Prefill
// ---------------------------------------------------------------------------
// Q, K, V layout: [B, S, H, D]  (H = num_heads, D = head_dim)
// O       layout: [B, S, H_q, D]
//
// GQA supported: H_q % H_kv == 0.  kv_head = q_head * H_kv / H_q.
// Causal mask applied (token i attends only to positions <= i).
// FP16 I/O, FP32 accumulation.  Currently specialised for head_dim = 128.
//
// Tile sizes: Br=16 (Q rows per block), Bc=64 (KV cols per tile).
// Shared memory per block: 2 × Bc × D × sizeof(half) = 32 KB.
// ---------------------------------------------------------------------------
void launch_flash_attention_prefill(
    const half*  Q,
    const half*  K,
    const half*  V,
    half*        O,
    int B, int S, int H_q, int H_kv, int head_dim,
    cudaStream_t stream = 0
);

// ---------------------------------------------------------------------------
// Paged Attention — Decode
// ---------------------------------------------------------------------------
// Single new token (S_q=1) attends to full context stored in a paged KV cache.
//
// q layout:              [num_seqs, H_q, D]
// k_cache / v_cache:     [num_blocks, H_kv, block_size, D]
// block_tables:          [num_seqs, max_blocks_per_seq]   (logical→physical)
// seq_lens:              [num_seqs]  — context length including the new token
//
// Specialised for head_dim=128, block_size=16.
// ---------------------------------------------------------------------------
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
