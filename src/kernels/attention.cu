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
    float*      __restrict__ LSE, // [B, S, H_q] or nullptr
    int B, int S, int H_q, int H_kv, float scale
) {
    const int r      = threadIdx.x;          // Row within Q tile (0..Br-1)
    const int q_tile = blockIdx.x;
    const int h_q    = blockIdx.y;
    const int b      = blockIdx.z;
    const int h_kv   = h_q * H_kv / H_q;    // GQA: integer division

    const int q_row = q_tile * Br + r;       // Global row in Q
    // NOTE: do NOT return early here — all Br threads must participate in the
    // cooperative K/V tile load and reach every __syncthreads().  Use a flag
    // instead and skip computation/writes for out-of-bounds rows.
    const bool valid_q = (q_row < S);

    // Shared memory — K and V tiles, FP16
    __shared__ half K_smem[Bc][HEAD_DIM];
    __shared__ half V_smem[Bc][HEAD_DIM];

    // Load Q row into registers (FP32, converted once) — only for valid rows.
    float q_reg[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) q_reg[d] = 0.0f;
    if (valid_q) {
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
        // Cooperative load: ALL Br threads fill K_smem and V_smem (Bc rows).
        // Each thread is responsible for rows: r, r+Br, r+2*Br, r+3*Br.
        // All threads must participate — including those with q_row >= S —
        // so that every K/V row needed by valid threads is populated before
        // the __syncthreads() below.
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
        }
        __syncthreads();

        // ------------------------------------------------------------------
        // Compute attention scores and update online softmax (valid rows only)
        // ------------------------------------------------------------------
        if (valid_q) {
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
        }
        __syncthreads();
    }

    // Normalise and write output (FP32 → FP16) — valid rows only
    if (valid_q) {
        half* o_ptr = O + ((long)(b * S + q_row) * H_q + h_q) * HEAD_DIM;
        const float inv_sum = 1.0f / row_sum;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++)
            o_ptr[d] = __float2half(o_acc[d] * inv_sum);

        // Save logsumexp for backward pass (if requested)
        if (LSE) {
            LSE[(long)(b * S + q_row) * H_q + h_q] = row_max + logf(row_sum);
        }
    }
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
// FA2 Backward — Kernel 1: Precompute D
// =============================================================================
//
// D[i] = dot(O[i], dO[i]) for each attention row.
// Used in softmax backward: dS = P * (dP - D).
//
// Grid:  (ceil(total_rows / 256))
// Block: (256)
// =============================================================================

template<int HEAD_DIM>
__global__ void flash_attention_bwd_compute_D_kernel(
    const half* __restrict__ O,    // [B*S*H_q, D]  (contiguous as [B,S,H_q,D])
    const half* __restrict__ dO,   // [B*S*H_q, D]
    float*      __restrict__ D,    // [B*S*H_q]
    int total_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;

    const half* o_ptr  = O  + (long)row * HEAD_DIM;
    const half* do_ptr = dO + (long)row * HEAD_DIM;

    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++)
        sum += __half2float(o_ptr[d]) * __half2float(do_ptr[d]);

    D[row] = sum;
}

// =============================================================================
// FA2 Backward — Kernel 2: dQ  (Q-outer loop)
// =============================================================================
//
// Same loop structure as the forward kernel: each thread owns one Q row
// and iterates over KV tiles. Recomputes P using saved LSE, then
// accumulates dQ[i] = scale * Σ_j dS[i,j] * K[j].
//
// Grid:  (ceil(S/Br), H_q, B)
// Block: (Br)
//
// Shared memory: K_smem[Bc][D] + V_smem[Bc][D] = 32 KB  (same as forward)
// Registers:     q_reg[D], do_reg[D], dq_acc[D]
// =============================================================================

template<int HEAD_DIM, int Br, int Bc>
__global__ void flash_attention_bwd_dq_kernel(
    const half*  __restrict__ Q,     // [B, S, H_q,  D]
    const half*  __restrict__ K,     // [B, S, H_kv, D]
    const half*  __restrict__ V,     // [B, S, H_kv, D]
    const half*  __restrict__ dO,    // [B, S, H_q,  D]
    const float* __restrict__ LSE,   // [B, S, H_q]
    const float* __restrict__ D,     // [B, S, H_q]
    half*        __restrict__ dQ,    // [B, S, H_q,  D]
    int B, int S, int H_q, int H_kv, float scale
) {
    const int r      = threadIdx.x;
    const int q_tile = blockIdx.x;
    const int h_q    = blockIdx.y;
    const int b      = blockIdx.z;
    const int h_kv   = h_q * H_kv / H_q;

    const int q_row  = q_tile * Br + r;
    const bool valid_q = (q_row < S);

    __shared__ half K_smem[Bc][HEAD_DIM];
    __shared__ half V_smem[Bc][HEAD_DIM];

    // Load Q row, dO row into registers
    float q_reg[HEAD_DIM], do_reg[HEAD_DIM], dq_acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        q_reg[d] = 0.0f;  do_reg[d] = 0.0f;  dq_acc[d] = 0.0f;
    }

    float lse_val = 0.0f, D_val = 0.0f;
    if (valid_q) {
        const long row_idx = (long)(b * S + q_row) * H_q + h_q;
        const half* q_ptr  = Q  + row_idx * HEAD_DIM;
        const half* do_ptr = dO + row_idx * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            q_reg[d]  = __half2float(q_ptr[d]);
            do_reg[d] = __half2float(do_ptr[d]);
        }
        lse_val = LSE[row_idx];
        D_val   = D[row_idx];
    }

    const int num_kv_tiles = (S + Bc - 1) / Bc;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int tile_start = kv_tile * Bc;

        // Cooperative K/V tile load (same as forward)
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
        }
        __syncthreads();

        if (valid_q) {
            const int tile_end = min(tile_start + Bc, S);
            for (int c = 0; c < tile_end - tile_start; c++) {
                const int kv_pos = tile_start + c;
                if (kv_pos > q_row) break;  // causal

                // Recompute score and attention weight
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    score += q_reg[d] * __half2float(K_smem[c][d]);
                score *= scale;

                float p = expf(score - lse_val);

                // dP[i,j] = dot(dO[i], V[j])
                float dp = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    dp += do_reg[d] * __half2float(V_smem[c][d]);

                // dS[i,j] = P * (dP - D[i])
                float ds = p * (dp - D_val);

                // Accumulate dQ (scale applied at write time)
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    dq_acc[d] += ds * __half2float(K_smem[c][d]);
            }
        }
        __syncthreads();
    }

    // Write dQ with scale factor
    if (valid_q) {
        half* dq_ptr = dQ + ((long)(b * S + q_row) * H_q + h_q) * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++)
            dq_ptr[d] = __float2half(dq_acc[d] * scale);
    }
}

// =============================================================================
// FA2 Backward — Kernel 3: dK, dV  (KV-outer loop)
// =============================================================================
//
// Each block owns a tile of Bkv KV positions. Inner loop iterates over Q tiles
// (and over Q heads in the GQA group) to accumulate:
//   dK[j] = scale * Σ_{h_q} Σ_i dS[i,j] * Q[i]
//   dV[j] = Σ_{h_q} Σ_i P[i,j] * dO[i]
//
// Grid:  (ceil(S/Bkv), H_kv, B)
// Block: (Bkv)
//
// Shared memory: Q_smem[Bq][D] + dO_smem[Bq][D] + LSE_smem[Bq] + D_smem[Bq]
// Registers:     k_reg[D], v_reg[D], dk_acc[D], dv_acc[D]
// =============================================================================

template<int HEAD_DIM, int Bkv, int Bq>
__global__ void flash_attention_bwd_dkdv_kernel(
    const half*  __restrict__ Q,     // [B, S, H_q,  D]
    const half*  __restrict__ K,     // [B, S, H_kv, D]
    const half*  __restrict__ V,     // [B, S, H_kv, D]
    const half*  __restrict__ dO,    // [B, S, H_q,  D]
    const float* __restrict__ LSE,   // [B, S, H_q]
    const float* __restrict__ D,     // [B, S, H_q]
    half*        __restrict__ dK,    // [B, S, H_kv, D]
    half*        __restrict__ dV,    // [B, S, H_kv, D]
    int B, int S, int H_q, int H_kv, float scale
) {
    const int r       = threadIdx.x;          // KV position within tile
    const int kv_tile = blockIdx.x;
    const int h_kv    = blockIdx.y;
    const int b       = blockIdx.z;

    const int kv_row  = kv_tile * Bkv + r;
    const bool valid_kv = (kv_row < S);

    // GQA: Q heads that map to this KV head
    const int n_rep    = H_q / H_kv;
    const int h_q_base = h_kv * n_rep;

    __shared__ half  Q_smem[Bq][HEAD_DIM];
    __shared__ half  dO_smem[Bq][HEAD_DIM];
    __shared__ float LSE_smem[Bq];
    __shared__ float D_smem[Bq];

    // Load K, V rows into registers (one row per thread, loaded once)
    float k_reg[HEAD_DIM], v_reg[HEAD_DIM];
    float dk_acc[HEAD_DIM], dv_acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        k_reg[d] = 0.0f;  v_reg[d] = 0.0f;
        dk_acc[d] = 0.0f;  dv_acc[d] = 0.0f;
    }
    if (valid_kv) {
        const half* k_ptr = K + ((long)(b * S + kv_row) * H_kv + h_kv) * HEAD_DIM;
        const half* v_ptr = V + ((long)(b * S + kv_row) * H_kv + h_kv) * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            k_reg[d] = __half2float(k_ptr[d]);
            v_reg[d] = __half2float(v_ptr[d]);
        }
    }

    const int num_q_tiles = (S + Bq - 1) / Bq;

    for (int h_rep = 0; h_rep < n_rep; h_rep++) {
        const int h_q = h_q_base + h_rep;

        for (int q_tile = 0; q_tile < num_q_tiles; q_tile++) {
            const int tile_start = q_tile * Bq;

            // Cooperative load: Q, dO, LSE, D for this Q tile + Q head
            for (int row = r; row < Bq; row += Bkv) {
                const int q_row = tile_start + row;
                if (q_row < S) {
                    const long row_idx = (long)(b * S + q_row) * H_q + h_q;
                    const half* q_ptr  = Q  + row_idx * HEAD_DIM;
                    const half* do_ptr = dO + row_idx * HEAD_DIM;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++) {
                        Q_smem[row][d]  = q_ptr[d];
                        dO_smem[row][d] = do_ptr[d];
                    }
                    LSE_smem[row] = LSE[row_idx];
                    D_smem[row]   = D[row_idx];
                }
            }
            __syncthreads();

            if (valid_kv) {
                const int tile_end = min(tile_start + Bq, S);
                for (int c = 0; c < tile_end - tile_start; c++) {
                    const int q_row = tile_start + c;
                    // Causal: position q_row attends to kv_row only if kv_row <= q_row
                    if (q_row < kv_row) continue;

                    // Recompute score = Q[q_row] · K[kv_row] * scale
                    float score = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++)
                        score += __half2float(Q_smem[c][d]) * k_reg[d];
                    score *= scale;

                    float p = expf(score - LSE_smem[c]);

                    // dP = dot(dO[q_row], V[kv_row])
                    float dp = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++)
                        dp += __half2float(dO_smem[c][d]) * v_reg[d];

                    float ds = p * (dp - D_smem[c]);

                    // Accumulate dK, dV
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++) {
                        dk_acc[d] += ds * __half2float(Q_smem[c][d]);
                        dv_acc[d] += p  * __half2float(dO_smem[c][d]);
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write dK (with scale) and dV
    if (valid_kv) {
        half* dk_ptr = dK + ((long)(b * S + kv_row) * H_kv + h_kv) * HEAD_DIM;
        half* dv_ptr = dV + ((long)(b * S + kv_row) * H_kv + h_kv) * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            dk_ptr[d] = __float2half(dk_acc[d] * scale);
            dv_ptr[d] = __float2half(dv_acc[d]);
        }
    }
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
    cudaStream_t stream,
    float*       lse
) {
    constexpr int Br = 16;
    constexpr int Bc = 64;
    const float scale = 1.0f / sqrtf((float)head_dim);

    dim3 grid((S + Br - 1) / Br, H_q, B);
    dim3 block(Br);

    flash_attention_prefill_kernel<128, Br, Bc><<<grid, block, 0, stream>>>(
        Q, K, V, O, lse, B, S, H_q, H_kv, scale
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

void launch_flash_attention_backward(
    const half*  Q,
    const half*  K,
    const half*  V,
    const half*  O,
    const half*  dO,
    const float* lse,
    float*       D_buf,
    half*        dQ,
    half*        dK,
    half*        dV,
    int B, int S, int H_q, int H_kv, int head_dim,
    cudaStream_t stream
) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    const int total_rows = B * S * H_q;

    // Step 1: Compute D[i] = dot(O[i], dO[i])
    {
        int block_sz = 256;
        int grid_sz  = (total_rows + block_sz - 1) / block_sz;
        flash_attention_bwd_compute_D_kernel<128><<<grid_sz, block_sz, 0, stream>>>(
            O, dO, D_buf, total_rows
        );
    }

    // Step 2: Compute dQ (Q-outer loop, same tiling as forward)
    {
        constexpr int Br = 16;
        constexpr int Bc = 64;
        dim3 grid((S + Br - 1) / Br, H_q, B);
        dim3 block(Br);
        flash_attention_bwd_dq_kernel<128, Br, Bc><<<grid, block, 0, stream>>>(
            Q, K, V, dO, lse, D_buf, dQ, B, S, H_q, H_kv, scale
        );
    }

    // Step 3: Compute dK, dV (KV-outer loop, handles GQA internally)
    {
        constexpr int Bkv = 16;
        constexpr int Bq  = 64;  // Q tile size for inner loop
        dim3 grid((S + Bkv - 1) / Bkv, H_kv, B);
        dim3 block(Bkv);
        flash_attention_bwd_dkdv_kernel<128, Bkv, Bq><<<grid, block, 0, stream>>>(
            Q, K, V, dO, lse, D_buf, dK, dV, B, S, H_q, H_kv, scale
        );
    }
}
