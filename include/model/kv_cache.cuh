#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <assert.h>

// ---------------------------------------------------------------------------
// KV_BLOCK_SIZE: tokens per physical KV block.
// Must match the value used to build slot_mapping and block_tables.
// ---------------------------------------------------------------------------
static constexpr int KV_BLOCK_SIZE = 16;

// ---------------------------------------------------------------------------
// PagedKVCache
// ---------------------------------------------------------------------------
// Manages a fixed pool of physical KV blocks shared across all sequences and
// layers.  Sequences grow by claiming blocks from the free-list on demand.
// Blocks are released when a sequence finishes.
//
// Pool layout (single flat allocation for all layers):
//   k_pool / v_pool: [num_layers, total_blocks, num_kv_heads, KV_BLOCK_SIZE, head_dim]
//
// block_tables (GPU):  [max_batch_size, max_blocks_per_seq]
//   block_tables[s][lb] = physical block for sequence s, logical block lb.
//   Unused entries are -1.
//
// seq_lens (GPU): [max_batch_size]
//   Current context length for each active sequence (0 = empty slot).
//
// free_stack (CPU):
//   Stack of available physical block indices.  Not GPU-thread-safe.
// ---------------------------------------------------------------------------
struct PagedKVCache {
    // ── GPU ────────────────────────────────────────────────────────────────
    half* k_pool;          // [num_layers, total_blocks, num_kv_heads, KV_BLOCK_SIZE, head_dim]
    half* v_pool;
    int*  block_tables;    // [max_batch_size, max_blocks_per_seq]  (GPU int32)
    int*  seq_lens;        // [max_batch_size]                       (GPU int32)

    // ── Shape ───────────────────────────────────────────────────────────────
    int num_layers;
    int num_kv_heads;
    int head_dim;
    int total_blocks;
    int max_batch_size;
    int max_blocks_per_seq;

    // ── CPU free-block stack ─────────────────────────────────────────────────
    int* free_stack;       // [total_blocks]  (CPU malloc)
    int  num_free;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Allocate GPU pool + tables; initialise free-block stack.
PagedKVCache paged_kv_cache_init(
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int total_blocks,
    int max_batch_size,
    int max_blocks_per_seq
);

void paged_kv_cache_free(PagedKVCache& cache);

// ---------------------------------------------------------------------------
// Block management (host-side, single-threaded)
// ---------------------------------------------------------------------------

inline int paged_kv_cache_alloc_block(PagedKVCache& cache) {
    assert(cache.num_free > 0 && "KV block pool exhausted");
    return cache.free_stack[--cache.num_free];
}

inline void paged_kv_cache_free_block(PagedKVCache& cache, int block_idx) {
    assert(cache.num_free < cache.total_blocks);
    cache.free_stack[cache.num_free++] = block_idx;
}

// Advance one token for sequence seq_idx:
//   - allocates a new physical block if we're at a block boundary
//   - updates block_tables and seq_lens on the host shadow copies
//   - returns the flat slot = physical_block * KV_BLOCK_SIZE + in_block_offset
//     suitable for placing in slot_mapping before the kernel call
int paged_kv_cache_append_slot(PagedKVCache& cache,
                                int* h_block_tables,  // host mirror of block_tables
                                int* h_seq_lens,      // host mirror of seq_lens
                                int  seq_idx);

// ---------------------------------------------------------------------------
// GRPO: share prompt KV blocks across G rollout sequences
// ---------------------------------------------------------------------------
// Copy src_seq's block table into each dst_seq (block-pointer copy, no KV copy).
// All G sequences share the same prompt physical blocks — zero extra KV memory.
void paged_kv_cache_fork(
    int* h_block_tables,           // host mirror of block_tables
    int* h_seq_lens,               // host mirror of seq_lens
    int  src_seq,
    const int* dst_seqs, int G,
    int  max_blocks_per_seq
);

// ---------------------------------------------------------------------------
// Kernel + launch wrapper (defined in src/model/kv_cache.cu)
// ---------------------------------------------------------------------------

__global__ void reshape_and_cache_half_kernel(
    const half*    key,
    const half*    value,
    half*          k_pool,
    half*          v_pool,
    const int64_t* slot_mapping,
    int num_tokens, int layer,
    int total_blocks, int num_kv_heads, int head_dim
);

// Scatter freshly-computed K/V for num_tokens tokens into the paged pool.
//
//   key / value:   [num_tokens, num_kv_heads, head_dim]  (GPU FP16)
//   k_pool/v_pool: cache.k_pool / cache.v_pool
//   slot_mapping:  [num_tokens]  (GPU int64) — one slot per token
//   layer:         transformer layer index (0..num_layers-1)
void launch_reshape_and_cache_half(
    const half*    key,
    const half*    value,
    half*          k_pool,
    half*          v_pool,
    const int64_t* slot_mapping,
    int num_tokens, int layer,
    int total_blocks, int num_kv_heads, int head_dim,
    cudaStream_t   stream = 0
);
