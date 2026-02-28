#include "model/kv_cache.cuh"

#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

PagedKVCache paged_kv_cache_init(
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int total_blocks,
    int max_batch_size,
    int max_blocks_per_seq
) {
    PagedKVCache cache{};
    cache.num_layers         = num_layers;
    cache.num_kv_heads       = num_kv_heads;
    cache.head_dim           = head_dim;
    cache.total_blocks       = total_blocks;
    cache.max_batch_size     = max_batch_size;
    cache.max_blocks_per_seq = max_blocks_per_seq;

    // GPU pool — one flat allocation for all layers
    const size_t pool_elems =
        (size_t)num_layers * total_blocks * num_kv_heads * KV_BLOCK_SIZE * head_dim;
    cudaMalloc(&cache.k_pool, pool_elems * sizeof(half));
    cudaMalloc(&cache.v_pool, pool_elems * sizeof(half));
    cudaMemset(cache.k_pool, 0, pool_elems * sizeof(half));
    cudaMemset(cache.v_pool, 0, pool_elems * sizeof(half));

    // Block tables and seq lens on GPU
    cudaMalloc(&cache.block_tables,
               (size_t)max_batch_size * max_blocks_per_seq * sizeof(int));
    cudaMalloc(&cache.seq_lens, max_batch_size * sizeof(int));

    // block_tables = -1 (unallocated), seq_lens = 0
    cudaMemset(cache.block_tables, -1,
               (size_t)max_batch_size * max_blocks_per_seq * sizeof(int));
    cudaMemset(cache.seq_lens, 0, max_batch_size * sizeof(int));

    // CPU free-block stack: blocks 0..total_blocks-1 all free
    cache.free_stack = (int*)malloc(total_blocks * sizeof(int));
    cache.num_free   = total_blocks;
    for (int i = 0; i < total_blocks; i++)
        cache.free_stack[i] = total_blocks - 1 - i;  // top of stack = block 0

    return cache;
}

void paged_kv_cache_free(PagedKVCache& cache) {
    cudaFree(cache.k_pool);
    cudaFree(cache.v_pool);
    cudaFree(cache.block_tables);
    cudaFree(cache.seq_lens);
    free(cache.free_stack);
    cache = PagedKVCache{};
}

// ---------------------------------------------------------------------------
// Block management
// ---------------------------------------------------------------------------

int paged_kv_cache_append_slot(
    PagedKVCache& cache,
    int* h_block_tables,   // host mirror [max_batch_size, max_blocks_per_seq]
    int* h_seq_lens,       // host mirror [max_batch_size]
    int  seq_idx
) {
    const int seq_len     = h_seq_lens[seq_idx];
    const int logical_blk = seq_len / KV_BLOCK_SIZE;
    const int block_off   = seq_len % KV_BLOCK_SIZE;

    if (block_off == 0) {
        // First token in a new block: claim one from the free-stack
        const int phys = paged_kv_cache_alloc_block(cache);
        h_block_tables[seq_idx * cache.max_blocks_per_seq + logical_blk] = phys;
    }

    const int phys_blk =
        h_block_tables[seq_idx * cache.max_blocks_per_seq + logical_blk];

    h_seq_lens[seq_idx]++;
    return phys_blk * KV_BLOCK_SIZE + block_off;
}

void paged_kv_cache_fork(
    int*       h_block_tables,
    int*       h_seq_lens,
    int        src_seq,
    const int* dst_seqs, int G,
    int        max_blocks_per_seq
) {
    const int src_len    = h_seq_lens[src_seq];
    const int num_blocks = (src_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;

    for (int g = 0; g < G; g++) {
        const int dst = dst_seqs[g];
        memcpy(h_block_tables + dst * max_blocks_per_seq,
               h_block_tables + src_seq * max_blocks_per_seq,
               num_blocks * sizeof(int));
        h_seq_lens[dst] = src_len;
    }
}

// ---------------------------------------------------------------------------
// Kernel: reshape_and_cache_half
// ---------------------------------------------------------------------------
// Scatter newly-computed K and V into the paged KV pool for one layer.
//
// slot_mapping[token_idx] = physical_block * KV_BLOCK_SIZE + in_block_offset
// Negative slots → padding token, skip.
//
// Pool layout: [num_layers, total_blocks, num_kv_heads, KV_BLOCK_SIZE, head_dim]
//
// Grid:  (num_tokens, 1)
// Block: (128, 1)
// ---------------------------------------------------------------------------

__global__ void reshape_and_cache_half_kernel(
    const half*    __restrict__ key,          // [num_tokens, num_kv_heads, head_dim]
    const half*    __restrict__ value,        // [num_tokens, num_kv_heads, head_dim]
    half*          __restrict__ k_pool,
    half*          __restrict__ v_pool,
    const int64_t* __restrict__ slot_mapping, // [num_tokens]
    int num_tokens, int layer,
    int total_blocks, int num_kv_heads, int head_dim
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return;  // padding token — skip

    const int block_idx = (int)(slot / KV_BLOCK_SIZE);
    const int block_off = (int)(slot % KV_BLOCK_SIZE);
    const int elems_per_token = num_kv_heads * head_dim;

    // Stride loop: each thread handles multiple (head, dim) pairs
    const int tid    = threadIdx.x + blockIdx.y * blockDim.x;
    const int stride = blockDim.x * gridDim.y;

    for (int i = tid; i < elems_per_token; i += stride) {
        const int h = i / head_dim;
        const int d = i % head_dim;

        // source: contiguous [num_tokens, num_kv_heads, head_dim]
        const half k_val = key  [(token_idx * num_kv_heads + h) * head_dim + d];
        const half v_val = value[(token_idx * num_kv_heads + h) * head_dim + d];

        // destination: [layer, block_idx, h, block_off, d]
        const int64_t base = ((int64_t)layer * total_blocks + block_idx) * num_kv_heads + h;
        const int64_t dst  = (base * KV_BLOCK_SIZE + block_off) * head_dim + d;

        k_pool[dst] = k_val;
        v_pool[dst] = v_val;
    }
}

// ---------------------------------------------------------------------------
// Launch wrapper
// ---------------------------------------------------------------------------

void launch_reshape_and_cache_half(
    const half*    key,
    const half*    value,
    half*          k_pool,
    half*          v_pool,
    const int64_t* slot_mapping,
    int num_tokens, int layer,
    int total_blocks, int num_kv_heads, int head_dim,
    cudaStream_t   stream
) {
    if (num_tokens == 0) return;

    // 128 threads per token; stride loop covers all num_kv_heads * head_dim elements
    dim3 grid(num_tokens, 1);
    dim3 block(128, 1);

    reshape_and_cache_half_kernel<<<grid, block, 0, stream>>>(
        key, value, k_pool, v_pool,
        slot_mapping, num_tokens, layer,
        total_blocks, num_kv_heads, head_dim
    );
}
