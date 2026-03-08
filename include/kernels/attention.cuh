#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void launch_flash_attention_prefill(
    const half*  Q,
    const half*  K,
    const half*  V,
    half*        O,
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
