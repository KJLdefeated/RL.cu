#include "kernels/sampler.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// =============================================================================
// Sampler Kernel
// =============================================================================
//
// Grid:  (num_tokens,)        — one block per token
// Block: (SAMPLER_BLOCK=256)  — each thread handles vocab_size/256 elements
//
// Dynamic shared memory layout (allocated by launch wrapper):
//   s_topk_probs[top_k]:  float  top-k probabilities in descending order
//   s_topk_ids  [top_k]:  int    corresponding vocab indices
//   s_val       [256]:    float  per-thread scratch for block reductions (reused)
//   s_idx       [256]:    int    per-thread scratch for argmax reduction
//
// Total smem: top_k*8 + 256*8 bytes
//   (top_k=50 → 2448 B; top_k=1024 → 10240 B — well within Blackwell's 164 KB)
//
// Steps:
//   1. temperature == 0  →  one-pass parallel argmax, write output, return.
//   2. Softmax: scale logits by 1/T → find row max (block reduce) →
//               compute exp(x-max) + sum (block reduce) → normalize.
//              Result written to temp_probs workspace in global memory.
//   3. Top-k: k rounds of cooperative argmax.  Each round all 256 threads
//             scan their chunk, do a block-level max+idx reduction, thread 0
//             records the winner in s_topk_{probs,ids} then zeros it in
//             temp_probs so the next round finds the next-best value.
//   4. Thread 0 (serial, short loop over top_k ≤ 1024):
//              re-normalize top-k probs, walk cumulative sum for top-p nucleus,
//              re-normalize nucleus, splitmix64 hash → uniform float → sample.
// =============================================================================

static constexpr int SAMPLER_BLOCK = 256;

// ---------------------------------------------------------------------------
// Block-level max+argmax reduction using dynamic shared memory arrays.
// All threads write (val, idx) to s_val[tid]/s_idx[tid], then reduce in-place
// to s_val[0]/s_idx[0].  Requires __syncthreads() before use of s_val[0].
// ---------------------------------------------------------------------------
__device__ __forceinline__
void block_argmax(float& val, int& idx, float* s_val, int* s_idx, int tid)
{
    s_val[tid] = val;
    s_idx[tid] = idx;
    __syncthreads();
    for (int s = SAMPLER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s && s_val[tid + s] > s_val[tid]) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }
    val = s_val[0];
    idx = s_idx[0];
}

// ---------------------------------------------------------------------------
// Block-level max reduction using warp shuffles + shared memory.
// s_val must have at least SAMPLER_BLOCK/32 = 8 elements.
// Result broadcast to all threads via s_val[0].
// ---------------------------------------------------------------------------
__device__ __forceinline__
void block_max(float& val, float* s_val, int tid)
{
    const int lane = tid & 31, warp = tid >> 5;
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, off));
    if (lane == 0) s_val[warp] = val;
    __syncthreads();
    if (warp == 0) {
        val = (lane < SAMPLER_BLOCK / 32) ? s_val[lane] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1)
            val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, off));
        if (lane == 0) s_val[0] = val;
    }
    __syncthreads();
    val = s_val[0];
}

// ---------------------------------------------------------------------------
// Block-level sum reduction using warp shuffles + shared memory.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void block_sum(float& val, float* s_val, int tid)
{
    const int lane = tid & 31, warp = tid >> 5;
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xffffffffu, val, off);
    if (lane == 0) s_val[warp] = val;
    __syncthreads();
    if (warp == 0) {
        val = (lane < SAMPLER_BLOCK / 32) ? s_val[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            val += __shfl_xor_sync(0xffffffffu, val, off);
        if (lane == 0) s_val[0] = val;
    }
    __syncthreads();
    val = s_val[0];
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------
__global__ void sampler_kernel(
    const half* __restrict__ logits,    // [num_tokens, vocab_size] FP16
    float*      __restrict__ probs,     // [num_tokens, vocab_size] FP32 workspace
    int         vocab_size,
    int         top_k,                  // already clamped: [1, MAX_SAMPLER_TOP_K]
    float       top_p,
    float       temperature,            // 0 → greedy
    unsigned long long seed,
    int64_t*    __restrict__ output_ids
) {
    extern __shared__ char smem_raw[];

    const int tok_idx = blockIdx.x;
    const int tid     = threadIdx.x;

    // Shared memory pointers from dynamic allocation
    float* s_topk_probs = (float*)smem_raw;
    int*   s_topk_ids   = (int*)  (smem_raw + (size_t)top_k * sizeof(float));
    float* s_val        = (float*)(smem_raw + (size_t)top_k * (sizeof(float) + sizeof(int)));
    int*   s_idx        = (int*)  (smem_raw + (size_t)top_k * (sizeof(float) + sizeof(int))
                                            + (size_t)SAMPLER_BLOCK * sizeof(float));

    const half* src = logits + (long)tok_idx * vocab_size;
    float*      P   = probs  + (long)tok_idx * vocab_size;

    // -------------------------------------------------------------------------
    // Fast path: temperature == 0 → greedy argmax (no softmax, no sampling)
    // -------------------------------------------------------------------------
    if (temperature == 0.0f) {
        float lv = -FLT_MAX;
        int   li = 0;
        for (int i = tid; i < vocab_size; i += SAMPLER_BLOCK) {
            float v = __half2float(src[i]);
            if (v > lv) { lv = v; li = i; }
        }
        block_argmax(lv, li, s_val, s_idx, tid);
        if (tid == 0) output_ids[tok_idx] = (int64_t)li;
        return;
    }

    // -------------------------------------------------------------------------
    // Step 1: Temperature scaling + numerically stable softmax
    //         Write float probabilities to probs workspace.
    // -------------------------------------------------------------------------
    const float inv_temp = 1.0f / temperature;

    // Pass 1: scale by temperature and find row max
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += SAMPLER_BLOCK) {
        float v = __half2float(src[i]) * inv_temp;
        P[i] = v;
        if (v > local_max) local_max = v;
    }
    block_max(local_max, s_val, tid);
    const float row_max = local_max;

    // Pass 2: exp(x - max) and accumulate sum
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += SAMPLER_BLOCK) {
        float p = expf(P[i] - row_max);
        P[i] = p;
        local_sum += p;
    }
    block_sum(local_sum, s_val, tid);
    const float inv_sum = 1.0f / local_sum;

    // Pass 3: normalize
    for (int i = tid; i < vocab_size; i += SAMPLER_BLOCK)
        P[i] *= inv_sum;
    __syncthreads();

    // -------------------------------------------------------------------------
    // Step 2: Top-k selection via k rounds of cooperative parallel argmax.
    //
    // Each round: all threads scan their chunk → block_argmax → thread 0
    // records winner in s_topk_{probs,ids} and zeros P[winner] so the next
    // round finds the next-best token.  After k rounds, s_topk_probs[0..k-1]
    // holds the top-k probabilities in strictly descending order.
    // -------------------------------------------------------------------------
    for (int r = 0; r < top_k; r++) {
        float lv = -FLT_MAX;
        int   li = 0;
        for (int i = tid; i < vocab_size; i += SAMPLER_BLOCK) {
            if (P[i] > lv) { lv = P[i]; li = i; }
        }
        block_argmax(lv, li, s_val, s_idx, tid);
        if (tid == 0) {
            s_topk_probs[r] = lv;
            s_topk_ids  [r] = li;
            P[li] = 0.0f;   // remove this token so next round finds next-best
        }
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // Step 3: Top-p nucleus truncation + weighted sample (thread 0, serial)
    //
    // The top-k candidates are already in descending probability order.
    // Re-normalize within top-k, walk cumulative sum to find the nucleus,
    // then sample proportionally from the nucleus using splitmix64 PRNG.
    // -------------------------------------------------------------------------
    if (tid == 0) {
        // Re-normalize top-k probs (sum < 1 since remaining mass was zeroed)
        float topk_sum = 0.0f;
        for (int i = 0; i < top_k; i++) topk_sum += s_topk_probs[i];
        const float inv_tk = 1.0f / topk_sum;

        // Find nucleus size for top-p (already in descending order)
        int nucleus = top_k;
        if (top_p < 1.0f) {
            float cum = 0.0f;
            for (int i = 0; i < top_k; i++) {
                cum += s_topk_probs[i] * inv_tk;
                if (cum >= top_p) { nucleus = i + 1; break; }
            }
        }

        // Re-normalize within nucleus
        float nuc_sum = 0.0f;
        for (int i = 0; i < nucleus; i++) nuc_sum += s_topk_probs[i];
        const float inv_nuc = 1.0f / nuc_sum;

        // splitmix64: high-quality integer hash → uniform float in [0, 1)
        unsigned long long s = seed + (unsigned long long)tok_idx * 0x9e3779b97f4a7c15ULL;
        s ^= s >> 30; s *= 0xbf58476d1ce4e5b9ULL;
        s ^= s >> 27; s *= 0x94d049bb133111ebULL;
        s ^= s >> 31;
        const float r = (float)(s >> 11) * (1.0f / (float)(1ULL << 53));

        // Walk cumulative distribution and sample
        float cum = 0.0f;
        int sampled = s_topk_ids[0];
        for (int i = 0; i < nucleus; i++) {
            cum += s_topk_probs[i] * inv_nuc;
            if (r <= cum) { sampled = s_topk_ids[i]; break; }
        }
        output_ids[tok_idx] = (int64_t)sampled;
    }
}

// =============================================================================
// Launch wrapper
// =============================================================================

void launch_sampler(
    const half*        logits,
    float*             temp_probs,
    int                num_tokens,
    int                vocab_size,
    int                top_k,
    float              top_p,
    float              temperature,
    unsigned long long seed,
    int64_t*           output_ids,
    cudaStream_t       stream
) {
    // Clamp top_k to valid range
    if (top_k <= 0 || top_k > MAX_SAMPLER_TOP_K) top_k = MAX_SAMPLER_TOP_K;
    if (top_k > vocab_size)                       top_k = vocab_size;

    // Dynamic shared memory: top_k*(float+int) + BLOCK*(float+int)
    const size_t smem = (size_t)top_k   * (sizeof(float) + sizeof(int))
                      + (size_t)SAMPLER_BLOCK * (sizeof(float) + sizeof(int));

    sampler_kernel<<<num_tokens, SAMPLER_BLOCK, smem, stream>>>(
        logits, temp_probs, vocab_size,
        top_k, top_p, temperature, seed, output_ids
    );
}
