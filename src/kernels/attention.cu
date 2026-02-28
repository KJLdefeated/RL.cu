#include "kernels/attention.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

// =============================================================================
// FA2 Prefill Kernel
// =============================================================================
//
// Grid:  (ceil(S/Br), H_q, B)
// Block: (Br)  — one thread per Q row in the current tile
//
// Algorithm (Flash Attention 2 with online softmax):
//   Outer loop: iterate over KV tiles (K and V loaded into shared memory)
//   Inner loop: each thread computes scores Q[q_row] · K[kv_pos] for all
//               positions in the tile, updating (row_max, row_sum, O_acc)
//               with the standard online softmax rescaling step.
//
// Shared memory layout:
//   K_smem[Bc][HEAD_DIM]  — current K tile, FP16, 16 KB
//   V_smem[Bc][HEAD_DIM]  — current V tile, FP16, 16 KB
//   Total: 32 KB per block
//
// Registers per thread (FP32):
//   q_reg[HEAD_DIM]   — query row (loaded once)
//   o_acc[HEAD_DIM]   — output accumulator
//   row_max, row_sum  — online softmax state
// =============================================================================

template<int HEAD_DIM, int Br, int Bc>
__global__ void flash_attention_prefill_kernel(
    const half* __restrict__ Q,   // [B, S, H_q,  D]
    const half* __restrict__ K,   // [B, S, H_kv, D]
    const half* __restrict__ V,   // [B, S, H_kv, D]
    half*       __restrict__ O,   // [B, S, H_q,  D]
    int B, int S, int H_q, int H_kv, float scale
) {
    const int r      = threadIdx.x;          // Row within Q tile (0..Br-1)
    const int q_tile = blockIdx.x;
    const int h_q    = blockIdx.y;
    const int b      = blockIdx.z;
    const int h_kv   = h_q * H_kv / H_q;    // GQA: integer division

    const int q_row = q_tile * Br + r;       // Global row in Q
    if (q_row >= S) return;

    // Shared memory — K and V tiles, FP16
    __shared__ half K_smem[Bc][HEAD_DIM];
    __shared__ half V_smem[Bc][HEAD_DIM];

    // Load Q row into registers (FP32, converted once)
    float q_reg[HEAD_DIM];
    {
        const half* q_ptr = Q + ((long)(b * S + q_row) * H_q + h_q) * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++)
            q_reg[d] = __half2float(q_ptr[d]);
    }

    // Online softmax state (per Q row, in registers)
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float o_acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) o_acc[d] = 0.0f;

    const int num_kv_tiles = (S + Bc - 1) / Bc;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int tile_start = kv_tile * Bc;

        // ------------------------------------------------------------------
        // Cooperative load: Br threads fill K_smem and V_smem (Bc rows each)
        // Each thread is responsible for rows: r, r+Br, r+2*Br, r+3*Br
        // ------------------------------------------------------------------
        for (int row = r; row < Bc; row += Br) {
            const int kv_row = tile_start + row;
            if (kv_row < S) {
                const half* k_ptr = K + ((long)(b * S + kv_row) * H_kv + h_kv) * HEAD_DIM;
                const half* v_ptr = V + ((long)(b * S + kv_row) * H_kv + h_kv) * HEAD_DIM;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) {
                    K_smem[row][d] = k_ptr[d];
                    V_smem[row][d] = v_ptr[d];
                }
            }
            // Out-of-bounds rows are never accessed (tile_end guards the inner loop)
        }
        __syncthreads();

        // ------------------------------------------------------------------
        // Compute attention scores and update online softmax
        // ------------------------------------------------------------------
        const int tile_end = min(tile_start + Bc, S);
        for (int c = 0; c < tile_end - tile_start; c++) {
            const int kv_pos = tile_start + c;
            // Causal mask: later positions in this row are also masked (break ok)
            if (kv_pos > q_row) break;

            // Dot product: Q[q_row] · K[kv_pos]
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++)
                dot += q_reg[d] * __half2float(K_smem[c][d]);
            dot *= scale;

            // Online softmax update (numerically stable)
            const float new_max = fmaxf(row_max, dot);
            const float alpha   = expf(row_max - new_max);  // rescale old accum
            const float p       = expf(dot - new_max);       // weight for new token

            row_sum = row_sum * alpha + p;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++)
                o_acc[d] = o_acc[d] * alpha + p * __half2float(V_smem[c][d]);

            row_max = new_max;
        }
        __syncthreads();
    }

    // Normalise and write output (FP32 → FP16)
    half* o_ptr = O + ((long)(b * S + q_row) * H_q + h_q) * HEAD_DIM;
    const float inv_sum = 1.0f / row_sum;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++)
        o_ptr[d] = __float2half(o_acc[d] * inv_sum);
}

// =============================================================================
// Paged Attention Decode Kernel
// =============================================================================
//
// Grid:  (num_seqs, H_q)
// Block: (1) — one thread per (sequence, q_head)
//
// The single query token attends to all context_len tokens stored in the
// paged KV cache.  Block tables map logical KV blocks to physical locations.
//
// Algorithm: same online softmax as prefill, but without shared memory tiling
// (Q has only one row; no tiling over Q needed).  Walks block_table to handle
// non-contiguous physical storage.
// =============================================================================

template<int HEAD_DIM, int BLOCK_SIZE>
__global__ void paged_attention_decode_kernel(
    const half* __restrict__ q,            // [num_seqs, H_q, D]
    const half* __restrict__ k_cache,      // [num_blocks, H_kv, BLOCK_SIZE, D]
    const half* __restrict__ v_cache,      // [num_blocks, H_kv, BLOCK_SIZE, D]
    half*       __restrict__ out,          // [num_seqs, H_q, D]
    const int*  __restrict__ block_tables, // [num_seqs, max_blocks_per_seq]
    const int*  __restrict__ seq_lens,     // [num_seqs]
    float scale,
    int max_blocks_per_seq,
    int H_q, int H_kv
) {
    const int seq_idx = blockIdx.x;
    const int h_q     = blockIdx.y;
    const int h_kv    = h_q * H_kv / H_q;   // GQA

    const int context_len = seq_lens[seq_idx];
    const int num_blocks  = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Load Q into registers (FP32)
    float q_reg[HEAD_DIM];
    {
        const half* q_ptr = q + (seq_idx * H_q + h_q) * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++)
            q_reg[d] = __half2float(q_ptr[d]);
    }

    float max_score = -INFINITY;
    float sum_exp   = 0.0f;
    float acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) acc[d] = 0.0f;

    const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    for (int blk = 0; blk < num_blocks; blk++) {
        // Indirection: logical block index → physical block in the pool
        const int physical_block  = seq_block_table[blk];
        const int tokens_in_block = min(BLOCK_SIZE, context_len - blk * BLOCK_SIZE);

        // Base offset for this physical block + KV head
        // k_cache layout: [num_blocks, H_kv, BLOCK_SIZE, D]
        const long block_base =
            ((long)physical_block * H_kv + h_kv) * BLOCK_SIZE * HEAD_DIM;

        for (int tok = 0; tok < tokens_in_block; tok++) {
            const half* k_ptr = k_cache + block_base + (long)tok * HEAD_DIM;
            const half* v_ptr = v_cache + block_base + (long)tok * HEAD_DIM;

            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++)
                score += q_reg[d] * __half2float(k_ptr[d]);
            score *= scale;

            // Online softmax update
            const float new_max = fmaxf(max_score, score);
            const float alpha   = expf(max_score - new_max);
            const float p       = expf(score - new_max);

            sum_exp = sum_exp * alpha + p;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++)
                acc[d] = acc[d] * alpha + p * __half2float(v_ptr[d]);

            max_score = new_max;
        }
    }

    // Write normalised output
    half* out_ptr = out + (seq_idx * H_q + h_q) * HEAD_DIM;
    const float inv = 1.0f / sum_exp;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++)
        out_ptr[d] = __float2half(acc[d] * inv);
}

// =============================================================================
// Launch wrappers
// =============================================================================

void launch_flash_attention_prefill(
    const half*  Q,
    const half*  K,
    const half*  V,
    half*        O,
    int B, int S, int H_q, int H_kv, int head_dim,
    cudaStream_t stream
) {
    constexpr int Br = 16;
    constexpr int Bc = 64;
    const float scale = 1.0f / sqrtf((float)head_dim);

    dim3 grid((S + Br - 1) / Br, H_q, B);
    dim3 block(Br);

    flash_attention_prefill_kernel<128, Br, Bc><<<grid, block, 0, stream>>>(
        Q, K, V, O, B, S, H_q, H_kv, scale
    );
}

void launch_paged_attention_decode(
    const half*  q,
    const half*  k_cache,
    const half*  v_cache,
    half*        out,
    const int*   block_tables,
    const int*   seq_lens,
    int num_seqs, int H_q, int H_kv, int head_dim,
    int max_blocks_per_seq, int block_size,
    cudaStream_t stream
) {
    const float scale = 1.0f / sqrtf((float)head_dim);

    dim3 grid(num_seqs, H_q);
    dim3 block(1);

    paged_attention_decode_kernel<128, 16><<<grid, block, 0, stream>>>(
        q, k_cache, v_cache, out,
        block_tables, seq_lens,
        scale, max_blocks_per_seq, H_q, H_kv
    );
}
