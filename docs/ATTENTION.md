# FA2 + Paged KV Cache: Implementation Design

**What the code actually looks like in C++/CUDA for grpo-cuda**

---

## Why This Matters for GRPO

In GRPO training, ~80% of wall-clock time is **generation** (rollout). Each step samples G completions per prompt. This means:

- **Prefill** (prompt): FA2 with tiled Q/K/V, compute-bound → need fast FA2
- **Decode** (token-by-token): memory-bound, reads entire KV cache → need efficient KV cache access
- **Batched rollouts** (G sequences share prompt): Paged KV = share prompt KV blocks across G sequences via copy-on-write, saving G× memory

---

## 1. Paged KV Cache

### Concept

Instead of pre-allocating `[batch, max_seq_len, heads, head_dim]` contiguously per sequence, we:

1. Allocate a **pool** of fixed-size KV blocks on GPU
2. Each block holds `BLOCK_SIZE` tokens (e.g., 16) of K and V for one layer + one KV head
3. Each sequence has a **block table** mapping logical position → physical block index
4. Blocks are allocated on-demand as sequences grow

```
┌─────────────────────────────────────────────┐
│             Physical Block Pool              │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐       │
│  │Blk 0 │ │Blk 1 │ │Blk 2 │ │Blk 3 │ ...  │
│  │16 tok│ │16 tok│ │16 tok│ │16 tok│       │
│  └──────┘ └──────┘ └──────┘ └──────┘       │
└─────────────────────────────────────────────┘

Sequence 0 block_table: [0, 2, 5, _]  → tokens 0-15 in blk0, 16-31 in blk2, 32-47 in blk5
Sequence 1 block_table: [0, 1, 3, _]  → shares blk0 (same prompt prefix!), then diverges
                         ↑ shared!
```

### C++ Data Structures

```cpp
constexpr int BLOCK_SIZE = 16;  // tokens per block

struct PagedKVCache {
    // The physical block pool — all blocks for all layers, all heads
    // Shape: [num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
    half* k_pool;  // Key pool
    half* v_pool;  // Value pool

    // Block table: maps (sequence, logical_block_idx) → physical_block_idx
    // Shape: [max_batch_size, max_blocks_per_seq]
    int* block_tables;

    // Sequence metadata
    int* seq_lens;           // [max_batch_size] current length of each sequence
    int  num_layers;
    int  num_kv_heads;
    int  head_dim;

    // Free block management
    int* free_blocks;        // stack of free physical block indices
    int  num_free;
    int  total_blocks;
};
```

### Block Allocation

```cpp
// Allocate a new physical block for a sequence
int allocate_block(PagedKVCache* cache) {
    assert(cache->num_free > 0);
    return cache->free_blocks[--cache->num_free];
}

// Free a block (when sequence finishes)
void free_block(PagedKVCache* cache, int block_idx) {
    cache->free_blocks[cache->num_free++] = block_idx;
}

// Append a token's KV to the cache for one layer
// Called after computing K, V for the new token
void append_kv(PagedKVCache* cache, int layer, int seq_idx,
               const half* k_new,   // [num_kv_heads, head_dim]
               const half* v_new) { // [num_kv_heads, head_dim]
    int seq_len = cache->seq_lens[seq_idx];
    int logical_block = seq_len / BLOCK_SIZE;
    int block_offset  = seq_len % BLOCK_SIZE;

    // Allocate new block if we're at the start of a block
    if (block_offset == 0) {
        int phys_block = allocate_block(cache);
        cache->block_tables[seq_idx * max_blocks + logical_block] = phys_block;
    }

    int phys_block = cache->block_tables[seq_idx * max_blocks + logical_block];

    // Copy K, V into the correct slot
    // k_pool layout: [num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
    for (int h = 0; h < cache->num_kv_heads; h++) {
        half* k_dst = cache->k_pool
            + ((phys_block * cache->num_kv_heads + h) * BLOCK_SIZE + block_offset)
            * cache->head_dim;
        cudaMemcpy(k_dst, k_new + h * cache->head_dim,
                   cache->head_dim * sizeof(half), cudaMemcpyDeviceToDevice);
        // same for v_pool
    }
    cache->seq_lens[seq_idx]++;
}
```

### GRPO Benefit: Prompt Sharing

When generating G completions for the same prompt, all G sequences share the prompt's KV blocks:

```cpp
// Fork: create G sequences sharing the same prompt KV
void fork_for_grpo(PagedKVCache* cache, int src_seq, int* dst_seqs, int G) {
    int prompt_len = cache->seq_lens[src_seq];
    int num_blocks = (prompt_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int g = 0; g < G; g++) {
        // Copy block table (just pointers, not actual KV data!)
        memcpy(&cache->block_tables[dst_seqs[g] * max_blocks],
               &cache->block_tables[src_seq * max_blocks],
               num_blocks * sizeof(int));
        cache->seq_lens[dst_seqs[g]] = prompt_len;
        // Increment reference count on shared blocks (for CoW)
    }
    // All G sequences now share the prompt KV cache — zero extra memory
}
```

---

## 2. Flash Attention 2

### Two Modes

| Mode | When | Q shape | K/V source | Bottleneck |
|------|------|---------|------------|------------|
| **Prefill** | Processing prompt | `[B, S_prompt, H, D]` | Contiguous (just computed) | Compute-bound |
| **Decode** | Token-by-token gen | `[B, 1, H, D]` | Paged KV cache (scattered) | Memory-bound |

Prefill uses standard FA2 tiling. Decode uses **Paged Attention** (FA2 variant that walks the block table).

### FA2 Prefill Kernel (Simplified)

Standard FA2: tile Q in outer loop, K/V in inner loop, online softmax.

```cuda
// FA2 forward kernel — one thread block computes one tile of output
// Template: head_dim=128, Br=64 (Q tile rows), Bc=64 (KV tile cols)
template<int HEAD_DIM, int Br, int Bc>
__global__ void flash_attention_prefill(
    const half* __restrict__ Q,     // [B, S, H_q, D]
    const half* __restrict__ K,     // [B, S, H_kv, D]
    const half* __restrict__ V,     // [B, S, H_kv, D]
    half* __restrict__ O,           // [B, S, H_q, D]
    int seq_len, int num_q_heads, int num_kv_heads, float scale
) {
    // Each thread block handles one (batch, head, q_tile) combination
    int batch_idx = blockIdx.z;
    int head_idx  = blockIdx.y;
    int q_tile    = blockIdx.x;  // which Br-sized chunk of Q

    int kv_head = head_idx / (num_q_heads / num_kv_heads);  // GQA mapping

    // Shared memory for tiles
    __shared__ half Q_tile[Br][HEAD_DIM];     // 64 * 128 * 2 = 16 KB
    __shared__ half K_tile[Bc][HEAD_DIM];     // 16 KB
    __shared__ half V_tile[Bc][HEAD_DIM];     // 16 KB
    __shared__ float S_tile[Br][Bc];          // 64 * 64 * 4 = 16 KB

    // Per-row online softmax state (in registers)
    float row_max[Br];    // m_i: running max
    float row_sum[Br];    // l_i: running sum of exp
    float O_acc[Br][HEAD_DIM];  // FP32 accumulator for output

    // Initialize
    for (int r = 0; r < Br; r++) {
        row_max[r] = -INFINITY;
        row_sum[r] = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) O_acc[r][d] = 0.0f;
    }

    // Load Q tile into shared memory (once)
    load_tile(Q, Q_tile, batch_idx, head_idx, q_tile * Br, seq_len);

    // Iterate over K/V tiles (inner loop)
    int num_kv_tiles = (seq_len + Bc - 1) / Bc;
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {

        // Load K, V tiles for this chunk
        load_tile(K, K_tile, batch_idx, kv_head, kv_tile * Bc, seq_len);
        load_tile(V, V_tile, batch_idx, kv_head, kv_tile * Bc, seq_len);
        __syncthreads();

        // S = Q @ K^T * scale  (Br x Bc matmul in shared memory)
        for (int r = 0; r < Br; r++)
            for (int c = 0; c < Bc; c++) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++)
                    dot += __half2float(Q_tile[r][d]) * __half2float(K_tile[c][d]);
                S_tile[r][c] = dot * scale;

                // Causal mask
                int q_pos = q_tile * Br + r;
                int k_pos = kv_tile * Bc + c;
                if (k_pos > q_pos) S_tile[r][c] = -INFINITY;
            }

        // Online softmax + accumulate O
        for (int r = 0; r < Br; r++) {
            // Find new max
            float new_max = row_max[r];
            for (int c = 0; c < Bc; c++)
                new_max = fmaxf(new_max, S_tile[r][c]);

            // Rescale previous accumulator
            float rescale = expf(row_max[r] - new_max);
            row_sum[r] *= rescale;
            for (int d = 0; d < HEAD_DIM; d++)
                O_acc[r][d] *= rescale;

            // Compute exp(S - new_max) and accumulate
            for (int c = 0; c < Bc; c++) {
                float p = expf(S_tile[r][c] - new_max);
                row_sum[r] += p;
                for (int d = 0; d < HEAD_DIM; d++)
                    O_acc[r][d] += p * __half2float(V_tile[c][d]);
            }
            row_max[r] = new_max;
        }
        __syncthreads();
    }

    // Normalize: O = O_acc / row_sum
    for (int r = 0; r < Br; r++)
        for (int d = 0; d < HEAD_DIM; d++)
            O[output_offset(batch_idx, q_tile*Br+r, head_idx, d)]
                = __float2half(O_acc[r][d] / row_sum[r]);
}
```

> **Note**: This is the conceptual structure. Real implementation uses warp-level matmul (wmma/mma), vectorized loads (`float4`), double-buffered shared memory, and swizzled layouts. But the algorithm is exactly this.

### Paged Attention Decode Kernel

For decode (S_q=1), the kernel walks the **block table** to read scattered KV blocks:

```cuda
// Paged attention for decode — Q is a single token, KV is in paged cache
template<int HEAD_DIM, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_decode(
    const half* __restrict__ q,           // [num_seqs, num_heads, head_dim]
    const half* __restrict__ k_cache,     // [num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
    const half* __restrict__ v_cache,     // [num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
    half* __restrict__ out,               // [num_seqs, num_heads, head_dim]
    const int* __restrict__ block_tables, // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,     // [num_seqs]
    float scale,
    int max_blocks_per_seq,
    int num_q_heads,
    int num_kv_heads
) {
    int seq_idx  = blockIdx.x;
    int head_idx = blockIdx.y;
    int kv_head  = head_idx / (num_q_heads / num_kv_heads);  // GQA

    int context_len = seq_lens[seq_idx];
    int num_blocks  = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Load Q for this (seq, head) into registers
    float q_reg[HEAD_DIM];
    load_q_to_registers(q, seq_idx, head_idx, q_reg);

    // Online softmax state
    float max_score = -INFINITY;
    float sum_exp   = 0.0f;
    float acc[HEAD_DIM] = {0};

    // Walk the block table — this is the key paging logic
    const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        // *** INDIRECTION: logical block → physical block ***
        int physical_block = seq_block_table[block_idx];

        int tokens_in_block = min(BLOCK_SIZE,
            context_len - block_idx * BLOCK_SIZE);

        for (int tok = 0; tok < tokens_in_block; tok++) {
            // Read K from physical block location
            const half* k_ptr = k_cache
                + physical_block * (num_kv_heads * BLOCK_SIZE * HEAD_DIM)
                + kv_head * (BLOCK_SIZE * HEAD_DIM)
                + tok * HEAD_DIM;

            // Compute Q·K dot product
            float score = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++)
                score += q_reg[d] * __half2float(k_ptr[d]);
            score *= scale;

            // Online softmax update
            float new_max = fmaxf(max_score, score);
            float rescale = expf(max_score - new_max);
            float p = expf(score - new_max);

            sum_exp = sum_exp * rescale + p;
            for (int d = 0; d < HEAD_DIM; d++)
                acc[d] = acc[d] * rescale
                    + p * __half2float(v_cache[/* same indexing as k */]);

            max_score = new_max;
        }
    }

    // Write normalized output
    for (int d = 0; d < HEAD_DIM; d++)
        out[seq_idx * num_q_heads * HEAD_DIM + head_idx * HEAD_DIM + d]
            = __float2half(acc[d] / sum_exp);
}
```

### Key Difference: Prefill vs Decode

```
PREFILL (compute-bound):
  Q = [B, S, H, 128]    ← many query tokens
  K, V = contiguous      ← just computed, in order
  → Use standard FA2 with SMEM tiling, tensor cores
  → Bottleneck: FLOPS (Q @ K^T matmul)

DECODE (memory-bound):
  Q = [B, 1, H, 128]    ← single query token
  K, V = paged cache     ← scattered across physical blocks
  → Use PagedAttention: walk block_table, stream K/V from HBM
  → Bottleneck: memory bandwidth (reading entire KV cache)
  → Warp-per-head parallelism, no SMEM tiling needed for Q
```

---

## 3. Putting It Together

### Full Decode Step

```
Input: token_id for each sequence in batch

1. Embedding lookup                          // trivial gather
2. For layer in 0..27:
   a. RMSNorm(hidden)
   b. Q = cuBLAS_Hgemm(hidden, q_proj)       // [B, 1, 2048]
   c. K = cuBLAS_Hgemm(hidden, k_proj)       // [B, 1, 1024]
   d. V = cuBLAS_Hgemm(hidden, v_proj)       // [B, 1, 1024]
   e. QK-Norm on Q, K (per-head RMSNorm)
   f. RoPE on Q, K
   g. append_kv(cache, layer, K, V)           // → paged pool
   h. paged_attention_decode(Q, cache)        // → walks block table
   i. O = cuBLAS_Hgemm(attn_out, o_proj)
   j. hidden = residual + O
   k. FFN block (RMSNorm → SwiGLU → residual)
3. Final RMSNorm → logits via cuBLAS → sample on GPU
```

### Memory Layout Summary

```
GPU Memory:
┌────────────────────────────────────────┐
│ Model Weights (FP16)         ~1.2 GB   │
│ KV Block Pool (FP16)         ~X GB     │  ← sized at init
│   k_pool: [num_blocks, 8, 16, 128]    │     based on available
│   v_pool: [num_blocks, 8, 16, 128]    │     GPU memory
│ Block Tables (int32)         ~small    │
│ Activations (FP16)           ~small    │  ← only for current step
│ Free Block Stack             ~small    │
└────────────────────────────────────────┘

Per block: 8 heads × 16 tokens × 128 dim × 2 bytes × 2 (K+V) = 64 KB
With 1 GB for KV cache: ~16,000 blocks = ~256K tokens total
```

---

## 4. Implementation Order

| Step | What | Why First |
|------|------|-----------|
| 1 | `PagedKVCache` struct + allocator | Foundation — all attention depends on this |
| 2 | FA2 prefill (contiguous KV) | Easier — standard tiling, validate correctness |
| 3 | `append_kv` kernel | Fused scatter into paged pool |
| 4 | Paged attention decode | The performance-critical path for GRPO |
| 5 | `fork_for_grpo` (block sharing) | GRPO optimization — share prompt KV across G rollouts |
| 6 | Block reference counting + CoW | Correctness for shared blocks during generation |

### Files

```
include/kernels/attention.cuh     # FA2 prefill + paged decode declarations
include/grpo/kv_cache.cuh         # PagedKVCache struct + block allocator
src/kernels/attention_prefill.cu   # FA2 prefill kernel
src/kernels/attention_decode.cu    # Paged attention decode kernel
src/model/kv_cache.cu             # Block alloc, append, fork, free
```