#include "kernels/sampler.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// =============================================================================
// Sampler Kernel v3 — Gumbel-max (nano-vllm style)
// =============================================================================
//
// Sampling method: Gumbel-max trick
//
//   argmax_i ( logit_i / T  +  Gumbel(0,1)_i )
//
// This is mathematically equivalent to sampling from softmax(logits / T):
//   P(token = i) = exp(logit_i / T) / sum_j exp(logit_j / T)
// but implemented as a single argmax over perturbed logits — no softmax,
// no CDF walk, no top-k selection.  Just one pass + one block reduction.
//
// Gumbel noise: gumbel_i = -log(-log(u_i)), u_i ~ Uniform(0,1)
// Generated via a counter-based hash of (seed, tok_idx, vocab_idx).
// Each call to __logf is the GPU intrinsic (~4–5 cycles), not the precise
// libm logf (~20 cycles), so the two logf calls add ~10 cycles per element.
//
// Passes over vocab: 1 (v1: 53 passes; v2: 2 passes)
// Shared memory: 64 bytes (v1/v2: up to 12 KB)
// top_k / top_p: not used (nano-vllm also omits them)
//
// Grid:  (num_tokens,)       — one block per token
// Block: (SAMPLER_BLOCK=256)
//
// Shared memory: 8 floats + 8 ints = 64 bytes for inter-warp reduction scratch
// =============================================================================

static constexpr int SAMPLER_BLOCK = 256;

// Counter-based per-element Gumbel(0,1) noise.
// Mixes (seed, tok_idx, vocab_idx) via splitmix64-style finalizer → uniform
// [0,1) → Gumbel sample.  Completely stateless and independent per element.
__device__ __forceinline__
float gumbel_noise(unsigned long long seed, int tok, int idx)
{
    unsigned long long s = seed
        ^ ((unsigned long long)tok  * 0x9e3779b97f4a7c15ULL)
        ^ ((unsigned long long)(unsigned int)idx * 0x6c62272e07bb0142ULL);
    s ^= s >> 30; s *= 0xbf58476d1ce4e5b9ULL;
    s ^= s >> 27; s *= 0x94d049bb133111ebULL;
    s ^= s >> 31;
    // Map to (0,1) — clamp away from 0 to avoid log(0)
    float u = fmaxf((float)(s >> 11) * (1.0f / (float)(1ULL << 53)), 1e-37f);
    // Gumbel(0,1): -log(-log(u))  [fast GPU intrinsic __logf ≈ 4–5 cycles]
    return -__logf(-__logf(u));
}

__global__ void sampler_kernel(
    const half* __restrict__ logits,    // [num_tokens, vocab_size] FP16
    float*      __restrict__ /*probs*/, // unused — kept for API compat
    int         vocab_size,
    int         /*top_k*/,              // unused — Gumbel-max samples full vocab
    float       /*top_p*/,              // unused
    float       temperature,            // 0 → greedy argmax
    unsigned long long seed,
    int64_t*    __restrict__ output_ids
) {
    // Warp-reduction scratch: 8 floats + 8 ints (= 64 bytes, static allocation)
    __shared__ float s_val[SAMPLER_BLOCK / 32];
    __shared__ int   s_idx[SAMPLER_BLOCK / 32];

    const int tok_idx = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp    = tid >> 5;

    const half* src = logits + (long)tok_idx * vocab_size;

    // -------------------------------------------------------------------------
    // Single pass: find argmax of (scaled_logit + noise)
    //   temperature == 0 → greedy: noise = 0
    //   temperature >  0 → Gumbel-max: noise = Gumbel(0,1) per element
    // -------------------------------------------------------------------------
    float best_val = -FLT_MAX;
    int   best_idx = 0;

    if (temperature == 0.0f) {
        // Greedy: pure argmax over logits (no random noise)
        for (int i = tid; i < vocab_size; i += SAMPLER_BLOCK) {
            float v = __half2float(src[i]);
            if (v > best_val) { best_val = v; best_idx = i; }
        }
    } else {
        const float inv_temp = 1.0f / temperature;
        for (int i = tid; i < vocab_size; i += SAMPLER_BLOCK) {
            float score = __half2float(src[i]) * inv_temp
                        + gumbel_noise(seed, tok_idx, i);
            if (score > best_val) { best_val = score; best_idx = i; }
        }
    }

    // -------------------------------------------------------------------------
    // Block-level argmax reduction: warp shuffles → shared memory → warp 0
    // -------------------------------------------------------------------------
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best_val, off);
        int   oi = __shfl_xor_sync(0xffffffffu, best_idx, off);
        if (ov > best_val) { best_val = ov; best_idx = oi; }
    }
    if (lane == 0) { s_val[warp] = best_val; s_idx[warp] = best_idx; }
    __syncthreads();

    if (warp == 0) {
        best_val = (lane < SAMPLER_BLOCK / 32) ? s_val[lane] : -FLT_MAX;
        best_idx = (lane < SAMPLER_BLOCK / 32) ? s_idx[lane] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, best_val, off);
            int   oi = __shfl_xor_sync(0xffffffffu, best_idx, off);
            if (ov > best_val) { best_val = ov; best_idx = oi; }
        }
        if (lane == 0) output_ids[tok_idx] = (int64_t)best_idx;
    }
}

// =============================================================================
// Launch wrapper
// =============================================================================

void launch_sampler(
    const half*        logits,
    float*             temp_probs,   // unused in v3; kept for API compatibility
    int                num_tokens,
    int                vocab_size,
    int                top_k,        // ignored
    float              top_p,        // ignored
    float              temperature,
    unsigned long long seed,
    int64_t*           output_ids,
    cudaStream_t       stream
) {
    (void)temp_probs; (void)top_k; (void)top_p; (void)vocab_size;
    if (num_tokens == 0) return;

    // 64 bytes static smem declared inside kernel; dynamic smem = 0
    sampler_kernel<<<num_tokens, SAMPLER_BLOCK, 0, stream>>>(
        logits, temp_probs, vocab_size,
        top_k, top_p, temperature, seed, output_ids
    );
}
