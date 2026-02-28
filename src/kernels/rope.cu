#include "kernels/rope.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

// =============================================================================
// Precompute kernel
// =============================================================================
// Grid:  (max_seq_len, 1)
// Block: (head_dim/2)       — one thread per frequency index
//
// cos_table[pos * half_dim + i] = cos(pos / rope_theta^(2i/head_dim))
// sin_table[pos * half_dim + i] = sin(pos / rope_theta^(2i/head_dim))
// =============================================================================

__global__ void rope_precompute_kernel(
    float* cos_table,   // [max_seq_len, half_dim]
    float* sin_table,   // [max_seq_len, half_dim]
    int half_dim,
    float log_rope_theta  // logf(rope_theta) — avoids recomputing per thread
) {
    const int pos  = blockIdx.x;
    const int freq = threadIdx.x;   // 0..half_dim-1

    // inv_freq = rope_theta^(-2*freq/head_dim) = exp(-2*freq/head_dim * log_theta)
    const float inv_freq = expf(-2.0f * freq * log_rope_theta / (float)(2 * half_dim));
    const float angle    = (float)pos * inv_freq;

    cos_table[pos * half_dim + freq] = cosf(angle);
    sin_table[pos * half_dim + freq] = sinf(angle);
}

void launch_rope_precompute(
    float*       cos_table,
    float*       sin_table,
    int          max_seq_len,
    int          head_dim,
    float        rope_theta,
    cudaStream_t stream
) {
    const int half_dim = head_dim / 2;

    // One block per sequence position; head_dim/2 threads per block
    rope_precompute_kernel<<<max_seq_len, half_dim, 0, stream>>>(
        cos_table, sin_table,
        half_dim,
        logf(rope_theta)
    );
}

// =============================================================================
// RoPE apply kernel
// =============================================================================
// Grid:  (num_tokens, num_q_heads)
// Block: (head_dim/2)     — one thread per rotation pair
//
// Thread (tok, h, i):
//   c = cos_table[pos][i],  s = sin_table[pos][i]
//   Q[tok][h][i]         = Q[tok][h][i]          * c - Q[tok][h][i+D/2] * s
//   Q[tok][h][i + D/2]   = Q[tok][h][i + D/2]   * c + Q[tok][h][i]     * s
//
// For h < num_kv_heads: same rotation applied to K[tok][h].
//
// The two writes per thread are to the SAME head but different offsets, so
// we must read both q0 and q1 before writing either (no RAW hazard since
// both operands come from registers).
// =============================================================================

__global__ void rope_kernel(
    half*        __restrict__ Q,          // [num_tokens, H_q,  D]
    half*        __restrict__ K,          // [num_tokens, H_kv, D]
    const float* __restrict__ cos_table,  // [max_seq_len, D/2]
    const float* __restrict__ sin_table,  // [max_seq_len, D/2]
    const int*   __restrict__ position_ids, // [num_tokens]
    int num_tokens, int H_q, int H_kv, int head_dim
) {
    const int tok  = blockIdx.x;   // token index (0..num_tokens-1)
    const int h    = blockIdx.y;   // Q head index (0..H_q-1)
    const int i    = threadIdx.x;  // rotation pair index (0..D/2-1)
    const int hd2 = head_dim / 2;  // = D/2; "half" reserved as CUDA type name

    const int pos = position_ids[tok];

    const float c = cos_table[pos * hd2 + i];
    const float s = sin_table[pos * hd2 + i];

    // ── Apply to Q ──────────────────────────────────────────────────────────
    {
        half* q = Q + (tok * H_q + h) * head_dim;
        const float q0 = __half2float(q[i]);         // first-half element
        const float q1 = __half2float(q[i + hd2]);   // second-half element
        q[i]      = __float2half(q0 * c - q1 * s);
        q[i + hd2] = __float2half(q1 * c + q0 * s);
    }

    // ── Apply to K (only for heads that exist in K) ──────────────────────────
    if (h < H_kv) {
        half* k = K + (tok * H_kv + h) * head_dim;
        const float k0 = __half2float(k[i]);
        const float k1 = __half2float(k[i + hd2]);
        k[i]       = __float2half(k0 * c - k1 * s);
        k[i + hd2] = __float2half(k1 * c + k0 * s);
    }
}

void launch_rope(
    half*        Q,
    half*        K,
    const float* cos_table,
    const float* sin_table,
    const int*   position_ids,
    int          num_tokens,
    int          num_q_heads,
    int          num_kv_heads,
    int          head_dim,
    cudaStream_t stream
) {
    // Grid: one block per (token, Q head)
    // Block: head_dim/2 threads (one per rotation pair)
    dim3 grid(num_tokens, num_q_heads);
    dim3 block(head_dim / 2);

    rope_kernel<<<grid, block, 0, stream>>>(
        Q, K, cos_table, sin_table, position_ids,
        num_tokens, num_q_heads, num_kv_heads, head_dim
    );
}
