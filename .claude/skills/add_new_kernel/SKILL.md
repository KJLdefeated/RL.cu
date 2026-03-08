---
name: add-cuda-kernel
description: Rules and steps for creating a compute kernel in RL.cu project.
---

# Tutorial: Adding a New Kernel to RL.cu

## Goal

Implement high performance cuda kernel with target format:
- include files: include/kernels/[kernel_name].cuh
- source files: src/kernels/[kernel_name].cu
- testing: tests/test_[kernel_name].cu
- Makefile: Compile the testing implementation
Same format for non-cuda function (e.g. cpp)
After finsihed implementation, you should explain how the kernel works.

---

## Example

### `include/kernels/attention.cuh` — Flash Attention 2 Kernel

```cpp
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
```

### `src/kernels/attention.cu` — Flash Attention 2 Kernel Implementation
```cpp
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
```

### `tests/test_attention.cu` — Flash Attention 2 Kernel Testing

```cpp
// test_attention.cu
// Validates Flash Attention 2 (prefill) and Paged Attention (decode) kernels
// against a CPU FP32 naive reference.
//
// Pass criterion (from DESIGN.md): max absolute error < 5e-3 vs naive attention.
//
// Prefill test cases:
//   1. B=1, S=16,  H_q=2, H_kv=1 — small, GQA ratio 2:1
//   2. B=1, S=64,  H_q=4, H_kv=2 — single KV tile (Bc=64)
//   3. B=1, S=128, H_q=16, H_kv=8 — Qwen3 head config, two KV tiles
//   4. B=2, S=96,  H_q=4, H_kv=2 — batched
//
// Decode test cases:
//   5. num_seqs=1, context_len=32,  H_q=2, H_kv=1
//   6. num_seqs=2, context_len=128, H_q=16, H_kv=8

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/attention.cuh"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Deterministic LCG (no srand dependency)
static unsigned int lcg_state = 42u;
static float lcg_randf() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(lcg_state >> 1) / (float)0x7fffffffu) * 2.0f - 1.0f;
}

// ---------------------------------------------------------------------------
// CPU reference: naive causal attention
// Q, K, V layout: [B, S, H, D]   (H = num_heads)
// O        layout: [B, S, H_q, D]
// FP32 throughout.
// ---------------------------------------------------------------------------
static void ref_attention(
    float*       O,
    const float* Q,
    const float* K,
    const float* V,
    int B, int S, int H_q, int H_kv, int D
) {
    const float scale = 1.0f / sqrtf((float)D);

    float* scores = new float[S];

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H_q; h++) {
            const int hkv = h * H_kv / H_q;

            for (int i = 0; i < S; i++) {
                const float* qi = Q + ((b * S + i) * H_q + h) * D;

                // Compute scores[0..i]
                float max_s = -1e30f;
                for (int j = 0; j <= i; j++) {
                    const float* kj = K + ((b * S + j) * H_kv + hkv) * D;
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++) dot += qi[d] * kj[d];
                    scores[j] = dot * scale;
                    if (scores[j] > max_s) max_s = scores[j];
                }

                // Softmax over [0..i]
                float sum_e = 0.0f;
                for (int j = 0; j <= i; j++) {
                    scores[j] = expf(scores[j] - max_s);
                    sum_e += scores[j];
                }
                for (int j = 0; j <= i; j++) scores[j] /= sum_e;

                // O[i] = softmax(S) · V[0..i]
                float* oi = O + ((b * S + i) * H_q + h) * D;
                for (int d = 0; d < D; d++) {
                    float acc = 0.0f;
                    for (int j = 0; j <= i; j++) {
                        const float* vj = V + ((b * S + j) * H_kv + hkv) * D;
                        acc += scores[j] * vj[d];
                    }
                    oi[d] = acc;
                }
            }
        }
    }

    delete[] scores;
}

// ---------------------------------------------------------------------------
// CPU reference: non-causal attention (for decode, S_q=1)
// q layout: [num_seqs, H_q, D]
// K, V layout: [num_seqs, S_kv, H_kv, D]  — contiguous KV context
// O layout: [num_seqs, H_q, D]
// ---------------------------------------------------------------------------
static void ref_decode_attention(
    float*       O,
    const float* q,
    const float* K,
    const float* V,
    int num_seqs, int S_kv, int H_q, int H_kv, int D
) {
    const float scale = 1.0f / sqrtf((float)D);
    float* scores = new float[S_kv];

    for (int s = 0; s < num_seqs; s++) {
        for (int h = 0; h < H_q; h++) {
            const int hkv = h * H_kv / H_q;
            const float* qi = q + (s * H_q + h) * D;

            float max_s = -1e30f;
            for (int j = 0; j < S_kv; j++) {
                const float* kj = K + ((s * S_kv + j) * H_kv + hkv) * D;
                float dot = 0.0f;
                for (int d = 0; d < D; d++) dot += qi[d] * kj[d];
                scores[j] = dot * scale;
                if (scores[j] > max_s) max_s = scores[j];
            }

            float sum_e = 0.0f;
            for (int j = 0; j < S_kv; j++) {
                scores[j] = expf(scores[j] - max_s);
                sum_e += scores[j];
            }
            for (int j = 0; j < S_kv; j++) scores[j] /= sum_e;

            float* oi = O + (s * H_q + h) * D;
            for (int d = 0; d < D; d++) {
                float acc = 0.0f;
                for (int j = 0; j < S_kv; j++) {
                    const float* vj = V + ((s * S_kv + j) * H_kv + hkv) * D;
                    acc += scores[j] * vj[d];
                }
                oi[d] = acc;
            }
        }
    }

    delete[] scores;
}

// ---------------------------------------------------------------------------
// Prefill test runner
// ---------------------------------------------------------------------------
static bool run_prefill_test(
    const char* name, int B, int S, int H_q, int H_kv,
    int D = 128, float tol = 5e-3f
) {
    const long N_q  = (long)B * S * H_q  * D;
    const long N_kv = (long)B * S * H_kv * D;

    // Host buffers (FP32 for reference, FP16 for kernel)
    float* h_Q_f32 = new float[N_q];
    float* h_K_f32 = new float[N_kv];
    float* h_V_f32 = new float[N_kv];
    float* h_ref   = new float[N_q];

    half*  h_Q   = new half[N_q];
    half*  h_K   = new half[N_kv];
    half*  h_V   = new half[N_kv];
    half*  h_out = new half[N_q];

    // Fill with random values, scale to prevent softmax saturation
    for (long i = 0; i < N_q;  i++) h_Q[i] = __float2half(lcg_randf() * 0.5f);
    for (long i = 0; i < N_kv; i++) h_K[i] = __float2half(lcg_randf() * 0.5f);
    for (long i = 0; i < N_kv; i++) h_V[i] = __float2half(lcg_randf() * 0.5f);

    // FP32 reference uses the FP16-quantised values (round-trip)
    for (long i = 0; i < N_q;  i++) h_Q_f32[i] = __half2float(h_Q[i]);
    for (long i = 0; i < N_kv; i++) h_K_f32[i] = __half2float(h_K[i]);
    for (long i = 0; i < N_kv; i++) h_V_f32[i] = __half2float(h_V[i]);

    ref_attention(h_ref, h_Q_f32, h_K_f32, h_V_f32, B, S, H_q, H_kv, D);

    // Device buffers
    half *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, N_q  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K, N_kv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V, N_kv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_O, N_q  * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, N_q  * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, N_kv * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, N_kv * sizeof(half), cudaMemcpyHostToDevice));

    launch_flash_attention_prefill(d_Q, d_K, d_V, d_O, B, S, H_q, H_kv, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_O, N_q * sizeof(half), cudaMemcpyDeviceToHost));

    // Compare
    float max_err = 0.0f;
    for (long i = 0; i < N_q; i++) {
        float diff = fabsf(__half2float(h_out[i]) - h_ref[i]);
        if (diff > max_err) max_err = diff;
    }

    bool passed = (max_err < tol);
    printf("[%s] %-50s max_err=%.6f  %s\n",
           passed ? "PASS" : "FAIL", name, max_err,
           passed ? "" : "<-- EXCEEDS 5e-3");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    delete[] h_Q_f32; delete[] h_K_f32; delete[] h_V_f32; delete[] h_ref;
    delete[] h_Q;    delete[] h_K;    delete[] h_V;    delete[] h_out;

    return passed;
}

// ---------------------------------------------------------------------------
// Paged attention decode test runner
// Uses a contiguous block mapping (block i → physical block i) so the paged
// result is equivalent to standard attention on contiguous K/V.
// ---------------------------------------------------------------------------
static bool run_decode_test(
    const char* name,
    int num_seqs, int context_len, int H_q, int H_kv,
    int D = 128, int BLOCK_SIZE = 16, float tol = 5e-3f
) {
    // Total KV tokens per sequence (all seqs same length for simplicity)
    const long Q_total  = (long)num_seqs * H_q * D;
    const long O_total  = Q_total;

    // Host buffers (FP32 reference)
    float* h_q_f32 = new float[Q_total];
    float* h_K_f32 = new float[(long)num_seqs * context_len * H_kv * D];
    float* h_V_f32 = new float[(long)num_seqs * context_len * H_kv * D];
    float* h_ref   = new float[O_total];

    half*  h_q   = new half[Q_total];
    half*  h_out = new half[O_total];

    for (long i = 0; i < Q_total;  i++) h_q[i] = __float2half(lcg_randf() * 0.5f);
    for (long i = 0; i < (long)num_seqs * context_len * H_kv * D; i++) {
        h_K_f32[i] = lcg_randf() * 0.5f;
        h_V_f32[i] = lcg_randf() * 0.5f;
    }

    for (long i = 0; i < Q_total; i++) h_q_f32[i] = __half2float(h_q[i]);
    // Quantise K/V round-trip
    for (long i = 0; i < (long)num_seqs * context_len * H_kv * D; i++) {
        h_K_f32[i] = __half2float(__float2half(h_K_f32[i]));
        h_V_f32[i] = __half2float(__float2half(h_V_f32[i]));
    }

    ref_decode_attention(h_ref, h_q_f32, h_K_f32, h_V_f32,
                         num_seqs, context_len, H_q, H_kv, D);

    // Build the paged KV cache with an identity block table
    // Each sequence uses context_len/BLOCK_SIZE (rounded up) blocks.
    const int blocks_per_seq  = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int total_phys_blocks = num_seqs * blocks_per_seq;

    // k_cache layout: [num_blocks, H_kv, BLOCK_SIZE, D]
    const long cache_elems = (long)total_phys_blocks * H_kv * BLOCK_SIZE * D;
    half* h_k_cache = new half[cache_elems]();
    half* h_v_cache = new half[cache_elems]();

    // Fill paged cache from contiguous K/V
    // seq s, token t, kv head h → logical_block = t/BLOCK_SIZE, offset = t%BLOCK_SIZE
    // physical_block = s * blocks_per_seq + logical_block
    for (int s = 0; s < num_seqs; s++) {
        for (int t = 0; t < context_len; t++) {
            for (int h = 0; h < H_kv; h++) {
                int  logical_block = t / BLOCK_SIZE;
                int  tok_offset    = t % BLOCK_SIZE;
                int  phys_block    = s * blocks_per_seq + logical_block;

                const float* ksrc = h_K_f32 + ((s * context_len + t) * H_kv + h) * D;
                const float* vsrc = h_V_f32 + ((s * context_len + t) * H_kv + h) * D;

                half* kdst = h_k_cache
                    + ((long)phys_block * H_kv + h) * BLOCK_SIZE * D
                    + tok_offset * D;
                half* vdst = h_v_cache
                    + ((long)phys_block * H_kv + h) * BLOCK_SIZE * D
                    + tok_offset * D;

                for (int d = 0; d < D; d++) {
                    kdst[d] = __float2half(ksrc[d]);
                    vdst[d] = __float2half(vsrc[d]);
                }
            }
        }
    }

    // Block tables: identity mapping (seq s, logical block b → physical block s*bps+b)
    int* h_block_tables = new int[num_seqs * blocks_per_seq];
    int* h_seq_lens     = new int[num_seqs];
    for (int s = 0; s < num_seqs; s++) {
        h_seq_lens[s] = context_len;
        for (int b = 0; b < blocks_per_seq; b++)
            h_block_tables[s * blocks_per_seq + b] = s * blocks_per_seq + b;
    }

    // Device allocations
    half *d_q, *d_k_cache, *d_v_cache, *d_out;
    int  *d_block_tables, *d_seq_lens;

    CUDA_CHECK(cudaMalloc(&d_q,           Q_total * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k_cache,     cache_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v_cache,     cache_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out,         O_total * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_block_tables, num_seqs * blocks_per_seq * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seq_lens,    num_seqs * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_q,           h_q,           Q_total * sizeof(half),                cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_cache,     h_k_cache,     cache_elems * sizeof(half),            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_cache,     h_v_cache,     cache_elems * sizeof(half),            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_tables, h_block_tables, num_seqs * blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_lens,    h_seq_lens,    num_seqs * sizeof(int),                cudaMemcpyHostToDevice));

    launch_paged_attention_decode(
        d_q, d_k_cache, d_v_cache, d_out,
        d_block_tables, d_seq_lens,
        num_seqs, H_q, H_kv, D,
        blocks_per_seq, BLOCK_SIZE
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, O_total * sizeof(half), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (long i = 0; i < O_total; i++) {
        float diff = fabsf(__half2float(h_out[i]) - h_ref[i]);
        if (diff > max_err) max_err = diff;
    }

    bool passed = (max_err < tol);
    printf("[%s] %-50s max_err=%.6f  %s\n",
           passed ? "PASS" : "FAIL", name, max_err,
           passed ? "" : "<-- EXCEEDS 5e-3");

    cudaFree(d_q); cudaFree(d_k_cache); cudaFree(d_v_cache);
    cudaFree(d_out); cudaFree(d_block_tables); cudaFree(d_seq_lens);

    delete[] h_q_f32; delete[] h_K_f32; delete[] h_V_f32; delete[] h_ref;
    delete[] h_q; delete[] h_out;
    delete[] h_k_cache; delete[] h_v_cache;
    delete[] h_block_tables; delete[] h_seq_lens;

    return passed;
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------
static void run_prefill_benchmark(
    const char* name, int B, int S, int H_q, int H_kv,
    int D = 128, int warmup = 10, int iters = 200
) {
    const long N_q  = (long)B * S * H_q  * D;
    const long N_kv = (long)B * S * H_kv * D;

    half *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, N_q  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K, N_kv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V, N_kv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_O, N_q  * sizeof(half)));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    for (int i = 0; i < warmup; i++)
        launch_flash_attention_prefill(d_Q, d_K, d_V, d_O, B, S, H_q, H_kv, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < iters; i++)
        launch_flash_attention_prefill(d_Q, d_K, d_V, d_O, B, S, H_q, H_kv, D);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    const float us = ms * 1000.0f / iters;

    // FLOPs: 2 * B * H_q * S^2 * D  (Q@K^T and softmax(S)@V, approximately)
    const double flops = 2.0 * B * H_q * (double)S * S * D * 2;
    const double tflops = flops / (us * 1e-6) / 1e12;

    printf("[BENCH] %-50s  %7.2f us  %5.2f TFLOPS\n", name, us, tflops);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("=== Flash Attention kernel tests ===\n\n");

    // ── Prefill tests ────────────────────────────────────────────────────────
    printf("--- Prefill (FA2) ---\n");
    bool all_pass = true;

    all_pass &= run_prefill_test(
        "Prefill B=1 S=16  H_q=2  H_kv=1 (small GQA 2:1)",
        1, 16, 2, 1);

    all_pass &= run_prefill_test(
        "Prefill B=1 S=64  H_q=4  H_kv=2 (single KV tile)",
        1, 64, 4, 2);

    all_pass &= run_prefill_test(
        "Prefill B=1 S=128 H_q=16 H_kv=8 (Qwen3 heads, 2 KV tiles)",
        1, 128, 16, 8);

    all_pass &= run_prefill_test(
        "Prefill B=2 S=96  H_q=4  H_kv=2 (batched)",
        2, 96, 4, 2);

    all_pass &= run_prefill_test(
        "Prefill B=1 S=256 H_q=16 H_kv=8 (4 KV tiles)",
        1, 256, 16, 8);

    // ── Decode tests ─────────────────────────────────────────────────────────
    printf("\n--- Decode (Paged Attention) ---\n");

    all_pass &= run_decode_test(
        "Decode num_seqs=1 ctx=32  H_q=2  H_kv=1",
        1, 32, 2, 1);

    all_pass &= run_decode_test(
        "Decode num_seqs=1 ctx=128 H_q=16 H_kv=8 (Qwen3)",
        1, 128, 16, 8);

    all_pass &= run_decode_test(
        "Decode num_seqs=2 ctx=64  H_q=4  H_kv=2 (batched seqs)",
        2, 64, 4, 2);

    all_pass &= run_decode_test(
        "Decode num_seqs=4 ctx=128 H_q=16 H_kv=8 (GRPO-style batch)",
        4, 128, 16, 8);

    // ── Summary ──────────────────────────────────────────────────────────────
    printf("\n%s\n", all_pass ? "All tests PASSED." : "Some tests FAILED.");

    // ── Benchmarks ───────────────────────────────────────────────────────────
    printf("\n=== Prefill benchmarks (warmup=10, iters=200) ===\n");
    run_prefill_benchmark("Prefill B=1 S=128  H_q=16 H_kv=8",  1, 128,  16, 8);
    run_prefill_benchmark("Prefill B=1 S=512  H_q=16 H_kv=8",  1, 512,  16, 8);
    run_prefill_benchmark("Prefill B=1 S=2048 H_q=16 H_kv=8",  1, 2048, 16, 8);

    return all_pass ? 0 : 1;
}
```
---

```Makefile
# ──────────────────────────────────────────────────────────────────────────────
# Makefile — GRPO-CUDA kernel tests
#
# Quick-start:
#   make tests              # build + run all three kernel tests
#   make test_rmsnorm       # build + run rmsnorm only
#   make generate_refs      # generate PyTorch reference data (needs torch)
#   make clean              # remove build/
#
# To target a specific GPU architecture instead of auto-detect:
#   make tests ARCH=sm_120  # Blackwell (RTX PRO 6000 / RTX 50xx)  ← this machine
#   make tests ARCH=sm_89   # Ada       (RTX 40xx)
#   make tests ARCH=sm_86   # Ampere    (RTX 30xx)
#   make tests ARCH=sm_80   # A100
# ──────────────────────────────────────────────────────────────────────────────

CUDA_HOME := /usr/local/cuda-12.8
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := sm_120
INCLUDES  := -I include
NVCCFLAGS := -O2 -std=c++17 $(INCLUDES) --gpu-architecture=$(ARCH)

BUILDDIR  := build
PYTHON    := python3

# ── Kernel test binaries ───────────────────────────────────────────────────────

$(BUILDDIR)/test_rmsnorm: src/kernels/rmsnorm.cu tests/test_rmsnorm.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILDDIR)/test_softmax: src/kernels/softmax.cu tests/test_softmax.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILDDIR)/test_swiglu: src/kernels/swiglu.cu tests/test_swiglu.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILDDIR)/test_attention: src/kernels/attention.cu tests/test_attention.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILDDIR)/test_kv_cache: src/model/kv_cache.cu tests/test_kv_cache.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILDDIR)/test_rope: src/kernels/rope.cu tests/test_rope.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILDDIR)/test_embedding: src/kernels/embedding.cu tests/test_embedding.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILDDIR)/test_linear: src/kernels/linear.cu tests/test_linear.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcublas

QWEN3_SRCS := src/model/qwen3.cu src/model/kv_cache.cu \
              src/kernels/rmsnorm.cu src/kernels/rope.cu src/kernels/attention.cu \
              src/kernels/swiglu.cu src/kernels/embedding.cu src/kernels/linear.cu \
              src/kernels/config.cpp src/kernels/weights.cpp

$(BUILDDIR)/test_qwen3: $(QWEN3_SRCS) tests/test_qwen3.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcublas

$(BUILDDIR)/bench_decode: $(QWEN3_SRCS) tests/bench_decode.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcublas

$(BUILDDIR)/test_loading_weights: src/kernels/config.cpp src/kernels/weights.cpp tests/test_loading_weights.cpp | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# ── Run targets ────────────────────────────────────────────────────────────────

.PHONY: test_rmsnorm test_softmax test_swiglu test_attention test_kv_cache test_rope test_embedding test_linear test_qwen3 test_loading_weights bench_decode tests generate_refs clean

test_rmsnorm: $(BUILDDIR)/test_rmsnorm
	./$(BUILDDIR)/test_rmsnorm

test_softmax: $(BUILDDIR)/test_softmax
	./$(BUILDDIR)/test_softmax

test_swiglu: $(BUILDDIR)/test_swiglu
	./$(BUILDDIR)/test_swiglu

test_attention: $(BUILDDIR)/test_attention
	./$(BUILDDIR)/test_attention

test_kv_cache: $(BUILDDIR)/test_kv_cache
	./$(BUILDDIR)/test_kv_cache

test_rope: $(BUILDDIR)/test_rope
	./$(BUILDDIR)/test_rope

test_embedding: $(BUILDDIR)/test_embedding
	./$(BUILDDIR)/test_embedding

test_linear: $(BUILDDIR)/test_linear
	./$(BUILDDIR)/test_linear

test_qwen3: $(BUILDDIR)/test_qwen3
	./$(BUILDDIR)/test_qwen3

test_loading_weights: $(BUILDDIR)/test_loading_weights
	./$(BUILDDIR)/test_loading_weights

bench_decode: $(BUILDDIR)/bench_decode
	./$(BUILDDIR)/bench_decode

tests: test_rmsnorm test_softmax test_swiglu test_attention test_kv_cache test_rope test_embedding test_linear test_qwen3 test_loading_weights

# ── PyTorch reference generator ───────────────────────────────────────────────

generate_refs:
	$(PYTHON) tests/generate_references.py --outdir tests/reference_data

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	rm -rf $(BUILDDIR)
```

**Key points:**

- This fully implementation with CUDA/C++ and cublas, do not contain torch api.
- The model will load in fp16, the kernel should optimization for fp16 computation.
- The kernel implementation should consider for LLM RL training.

---

## References

- `docs/DESIGN.md`
- `include/kernels/attention.cuh`
- `src/kernels/attention.cu`
- `include/kernels/rmsnorm.cuh`
- `src/kernels/rmsnorm.cu`
- `tests/test_attention.cu`
- `tests/test_rmsnorm.cu`