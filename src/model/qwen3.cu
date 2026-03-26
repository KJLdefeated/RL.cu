#include "model/kv_cache.cuh"
#include "model/weights.h"
#include "model/config.h"
#include "model/qwen3.h"
#include "model/sampling_parmas.h"
#include "kernels/attention.cuh"
#include "kernels/embedding.cuh"
#include "kernels/rmsnorm.cuh"
#include "kernels/rope.cuh"
#include "kernels/swiglu.cuh"
#include "kernels/linear.cuh"
#include "kernels/adamw.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

// Helpers
#include "cuda_utils.h"

// In-place elementwise add: x[i] += r[i]
__global__ static void add_residual_kernel(
    half* __restrict__       x,
    const half* __restrict__ r,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __hadd(x[i], r[i]);
}

static void add_residual(half* x, const half* r, int n, cudaStream_t s) {
    add_residual_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, r, n);
}

// Gather last token of each sequence: out[b] = in[b*S + (S-1)]
// Extracts [B, hidden_size] from [B*S, hidden_size] for the lm_head step.
__global__ static void gather_last_tokens_kernel(
    half*       __restrict__ out,   // [B, hidden_size]
    const half* __restrict__ in,    // [B*S, hidden_size]
    int S, int hidden_size
) {
    int b = blockIdx.x;
    const half* src = in  + ((long)b * S + (S - 1)) * hidden_size;
    half*       dst = out + (long)b * hidden_size;
    for (int d = threadIdx.x; d < hidden_size; d += blockDim.x)
        dst[d] = src[d];
}

static void gather_last_tokens(half* out, const half* in,
                                int B, int S, int hidden_size,
                                cudaStream_t stream) {
    gather_last_tokens_kernel<<<B, 256, 0, stream>>>(out, in, S, hidden_size);
}

// Gather at variable per-sequence offsets: out[b] = in[offsets[b]]
// offsets[b] is a row index into in (units of hidden_size).
__global__ static void gather_at_offsets_kernel(
    half*       __restrict__ out,      // [B, hidden_size]
    const half* __restrict__ in,       // [T, hidden_size]
    const int*  __restrict__ offsets,  // [B]
    int hidden_size
) {
    int b = blockIdx.x;
    const half* src = in  + (long)offsets[b] * hidden_size;
    half*       dst = out + (long)b * hidden_size;
    for (int d = threadIdx.x; d < hidden_size; d += blockDim.x)
        dst[d] = src[d];
}

// =============================================================================
// Log-softmax + gather kernel (for training forward)
//
// For each token t, computes:
//   log_probs[t] = logits[t, target[t]] - max - log(sum(exp(logits[t] - max)))
//
// Grid:  (num_tokens,)
// Block: (256)
// =============================================================================

__global__ static void log_softmax_gather_kernel(
    float*       __restrict__ log_probs,    // [T]
    const half*  __restrict__ logits,       // [T, V]
    const int*   __restrict__ targets,      // [T]
    int V
) {
    const int t = blockIdx.x;
    const half* row = logits + (long)t * V;
    const int target = targets[t];

    // Step 1: Find row max (warp shuffle + shared memory reduction)
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float v = __half2float(row[i]);
        if (v > local_max) local_max = v;
    }

    // Warp-level max reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));

    // Inter-warp reduction via shared memory
    __shared__ float smem[32];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    if (lane == 0) smem[warp] = local_max;
    __syncthreads();
    // All threads in warp 0 must participate in shuffle (avoid UB with partial warp)
    if (warp == 0) {
        float val = (lane < num_warps) ? smem[lane] : -INFINITY;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    const float global_max = smem[0];

    // Step 2: Sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < V; i += blockDim.x)
        local_sum += expf(__half2float(row[i]) - global_max);

    // Warp-level sum reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);

    if (lane == 0) smem[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float val = (lane < num_warps) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    const float global_sum = smem[0];

    // Step 3: log_prob = logit[target] - max - log(sum)
    if (threadIdx.x == 0) {
        log_probs[t] = __half2float(row[target]) - global_max - logf(global_sum);
    }
}

// =============================================================================
// Log-softmax + gather BACKWARD kernel
//
// Overwrites d_logits in-place: reads logits, writes gradients.
//   d_logits[t, v] = d_loss[t] * (1_{v == target[t]} - softmax(logits[t])[v])
//
// Grid:  (num_tokens,)
// Block: (256)
// =============================================================================

__global__ static void log_softmax_gather_backward_kernel(
    half*        __restrict__ d_logits,    // [n, V] in: logits, out: gradients
    const float* __restrict__ d_loss,      // [n] upstream gradient
    const int*   __restrict__ targets,     // [n]
    int V
) {
    const int t = blockIdx.x;
    half* row = d_logits + (long)t * V;
    const float dl = d_loss[t];
    const int target = targets[t];

    // Step 1: Row max
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < V; i += blockDim.x)
        local_max = fmaxf(local_max, __half2float(row[i]));

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, off));

    __shared__ float smem[32];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    if (lane == 0) smem[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float val = (lane < num_warps) ? smem[lane] : -INFINITY;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, off));
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    const float global_max = smem[0];

    // Step 2: Sum of exp(logit - max)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < V; i += blockDim.x)
        local_sum += expf(__half2float(row[i]) - global_max);

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, off);

    if (lane == 0) smem[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float val = (lane < num_warps) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            val += __shfl_xor_sync(0xFFFFFFFF, val, off);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    const float inv_sum = 1.0f / smem[0];

    // Step 3: Write gradients (all reads of logits complete before any write)
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float softmax_i = expf(__half2float(row[i]) - global_max) * inv_sum;
        float grad = dl * ((i == target ? 1.0f : 0.0f) - softmax_i);
        row[i] = __float2half(grad);
    }
}

// =============================================================================
// Layer forward
// =============================================================================

void qwen3_layer_forward(
    Qwen3Model*  m,
    int          layer_idx,
    int          T,           // total tokens: B*S (prefill) or B (decode)
    int          B,
    int          S,           // seq len per batch entry (>1 prefill, 1 decode)
    bool         is_prefill,
    cudaStream_t stream
) {
    const Qwen3Config&       c = m->config;
    const Qwen3LayerWeights& L = m->weights.layers[layer_idx];

    // Attention block
    CUDA_CHECK(cudaMemcpyAsync(m->d_residual, m->d_hidden,
                               (long)T * c.hidden_size * sizeof(half),
                               cudaMemcpyDeviceToDevice, stream));
    // LayerNorm  (x=d_residual, out=d_hidden — separate buffers)
    launch_rmsnorm(m->d_hidden, m->d_residual, L.input_layernorm, T, c.hidden_size, c.rms_norm_eps, stream);
    // QKV projections
    linear_half(m->cublas, m->d_hidden, L.q_proj, m->d_Q, T, c.q_dim,  c.hidden_size);
    linear_half(m->cublas, m->d_hidden, L.k_proj, m->d_K, T, c.kv_dim, c.hidden_size);
    linear_half(m->cublas, m->d_hidden, L.v_proj, m->d_V, T, c.kv_dim, c.hidden_size);

    // QK-Norm per head — BEFORE RoPE (critical Qwen3 ordering)
    launch_rmsnorm(m->d_Q, m->d_Q, L.q_norm, T * c.num_attention_heads, c.head_dim, c.rms_norm_eps, stream);
    launch_rmsnorm(m->d_K, m->d_K, L.k_norm, T * c.num_key_value_heads, c.head_dim, c.rms_norm_eps, stream);

    // RoPE in-place on Q and K
    launch_rope(m->d_Q, m->d_K, m->cos_table, m->sin_table, m->d_pos_ids, T, c.num_attention_heads, c.num_key_value_heads, c.head_dim, stream);

    // Append K/V to paged KV cache
    launch_reshape_and_cache_half(m->d_K, m->d_V,m->kv_cache.k_pool, m->kv_cache.v_pool,m->d_slot_map,T, layer_idx,m->kv_cache.total_blocks, c.num_key_value_heads, c.head_dim,stream);

    if (is_prefill) {
        // FA2 prefill — uses live K, V just computed above.
        launch_flash_attention_prefill(
            m->d_Q, m->d_K, m->d_V, m->d_attn_out,
            B, S, c.num_attention_heads, c.num_key_value_heads, c.head_dim,
            stream);
    } else {
        // Paged decode — reads this layer's K/V slice from the cache.
        // Pool layout: [num_layers, total_blocks, H_kv, KV_BLOCK_SIZE, D]
        const size_t layer_stride = (size_t)m->kv_cache.total_blocks
                                  * c.num_key_value_heads * KV_BLOCK_SIZE * c.head_dim;
        const half* k_layer = m->kv_cache.k_pool + layer_idx * layer_stride;
        const half* v_layer = m->kv_cache.v_pool + layer_idx * layer_stride;

        launch_paged_attention_decode(
            m->d_Q,
            k_layer, v_layer,
            m->d_attn_out,
            m->kv_cache.block_tables, m->kv_cache.seq_lens,
            B, c.num_attention_heads, c.num_key_value_heads, c.head_dim,
            m->kv_cache.max_blocks_per_seq, KV_BLOCK_SIZE,
            stream);
    }

    // O projection + residual
    linear_half(m->cublas, m->d_attn_out, L.o_proj, m->d_hidden, T, c.hidden_size, c.q_dim);
    add_residual(m->d_hidden, m->d_residual, T * c.hidden_size, stream);

    // Save post-attention hidden as residual
    CUDA_CHECK(cudaMemcpyAsync(m->d_residual, m->d_hidden, (long)T * c.hidden_size * sizeof(half), cudaMemcpyDeviceToDevice, stream));

    // Post-attention LayerNorm
    launch_rmsnorm(m->d_hidden, m->d_residual,
                   L.post_attn_layernorm, T, c.hidden_size, c.rms_norm_eps, stream);

    // Gate + Up projections
    linear_half(m->cublas, m->d_hidden, L.gate_proj,
                m->d_gate, T, c.intermediate_size, c.hidden_size);
    linear_half(m->cublas, m->d_hidden, L.up_proj,
                m->d_up,   T, c.intermediate_size, c.hidden_size);

    // SwiGLU: mlp_mid[i] = silu(gate[i]) * up[i]
    launch_swiglu(m->d_mlp_mid, m->d_gate, m->d_up, T * c.intermediate_size, stream);

    // Down projection + residual
    linear_half(m->cublas, m->d_mlp_mid, L.down_proj, m->d_hidden, T, c.hidden_size, c.intermediate_size);
    add_residual(m->d_hidden, m->d_residual, T * c.hidden_size, stream);
}

// Final norm + lm_head → logits
// Gathers the last S-th token of each sequence, applies final_norm, then projects to vocab via lm_head.  Result in m->d_logits [B, vocab_size].
void compute_logits(Qwen3Model* m, int B, int S, cudaStream_t stream) {
    const Qwen3Config& c = m->config;
    gather_last_tokens(m->d_residual, m->d_hidden, B, S, c.hidden_size, stream);
    launch_rmsnorm(m->d_residual, m->d_residual,
                   m->weights.final_norm, B, c.hidden_size, c.rms_norm_eps, stream);
    linear_half(m->cublas, m->d_residual, m->weights.lm_head,
                m->d_logits, B, c.vocab_size, c.hidden_size);
}

// Load model from dist
Qwen3Model* qwen3_load(const std::string& model_dir, int max_batch, int max_seq) {
    Qwen3Model* m = new Qwen3Model{};
    m->max_batch = max_batch;
    m->max_seq   = max_seq;

    m->config = load_config(model_dir);
    m->config.compute_derived();
    m->weights = load_weights(model_dir, m->config);

    const Qwen3Config& c   = m->config;
    const int          max_T = max_batch * max_seq;

    // KV cache (+2 blocks per seq for block-boundary headroom)
    int max_blocks_per_seq = (max_seq + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE + 2;
    int total_blocks       = max_batch * max_blocks_per_seq;
    m->kv_cache = paged_kv_cache_init(
        c.num_hidden_layers, c.num_key_value_heads, c.head_dim,
        total_blocks, max_batch, max_blocks_per_seq);

    CUBLAS_CHECK(cublasCreate(&m->cublas));

    // RoPE tables
    long rope_elems = (long)c.max_position_embeddings * (c.head_dim / 2);
    CUDA_CHECK(cudaMalloc(&m->cos_table, rope_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m->sin_table, rope_elems * sizeof(float)));
    launch_rope_precompute(m->cos_table, m->sin_table,
                           c.max_position_embeddings, c.head_dim, c.rope_theta);

    // Scratch buffers
    CUDA_CHECK(cudaMalloc(&m->d_hidden,   (long)max_T * c.hidden_size       * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_residual, (long)max_T * c.hidden_size       * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_Q,        (long)max_T * c.q_dim             * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_K,        (long)max_T * c.kv_dim            * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_V,        (long)max_T * c.kv_dim            * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_attn_out, (long)max_T * c.q_dim             * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_gate,     (long)max_T * c.intermediate_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_up,       (long)max_T * c.intermediate_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_mlp_mid,  (long)max_T * c.intermediate_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_logits,   (long)max_batch * c.vocab_size    * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_tokens,   (long)max_T * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&m->d_pos_ids,  (long)max_T * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&m->d_slot_map, (long)max_T * sizeof(int64_t)));

    // Host mirrors
    int mbps = m->kv_cache.max_blocks_per_seq;
    m->h_block_tables = new int[(long)max_batch * mbps];
    m->h_seq_lens     = new int[max_batch];
    m->h_slot_map     = new int64_t[max_T];
    m->h_pos_ids      = new int[max_T];
    memset(m->h_block_tables, -1, (long)max_batch * mbps * sizeof(int));
    memset(m->h_seq_lens,      0, max_batch * sizeof(int));

    // Initialise GPU block tables to -1 (unused)
    CUDA_CHECK(cudaMemcpy(m->kv_cache.block_tables, m->h_block_tables,
                          (long)max_batch * mbps * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m->kv_cache.seq_lens, m->h_seq_lens,
                          max_batch * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[Qwen3] Loaded: %d layers  H_q=%d H_kv=%d  hidden=%d  vocab=%d\n",
           c.num_hidden_layers, c.num_attention_heads,
           c.num_key_value_heads, c.hidden_size, c.vocab_size);
    printf("[Qwen3] Scratch: max_batch=%d  max_seq=%d  (max_T=%d)\n",
           max_batch, max_seq, max_T);

    return m;
}

// Prepare model for inference: pre-allocate KV cache and scratch buffers, and compute RoPE tables.
void qwen3_init(Qwen3Model* m, int max_batch, int max_seq,
                int max_batched_tokens, int num_kv_blocks) {
    m->max_batch = max_batch;
    m->max_seq   = max_seq;

    const Qwen3Config& c = m->config;
    const int max_T = max_batched_tokens;  // ← scratch sizing

    // Scratch buffers (using max_T, not max_batch × max_seq)
    CUDA_CHECK(cudaMalloc(&m->d_hidden,   (long)max_T * c.hidden_size       * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_residual, (long)max_T * c.hidden_size       * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_Q,        (long)max_T * c.q_dim             * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_K,        (long)max_T * c.kv_dim            * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_V,        (long)max_T * c.kv_dim            * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_attn_out, (long)max_T * c.q_dim             * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_gate,     (long)max_T * c.intermediate_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_up,       (long)max_T * c.intermediate_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_mlp_mid,  (long)max_T * c.intermediate_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_logits,   (long)max_batch * c.vocab_size    * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&m->d_tokens,   (long)max_T * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&m->d_pos_ids,  (long)max_T * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&m->d_slot_map, (long)max_T * sizeof(int64_t)));

    int total_blocks = num_kv_blocks;
    int max_blocks_per_seq = (max_seq + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE + 2;

    m->kv_cache = paged_kv_cache_init(
        c.num_hidden_layers, c.num_key_value_heads, c.head_dim,
        total_blocks,            // ← shared pool from memory budget
        max_batch, max_blocks_per_seq);

    // Expose total_blocks so ModelRunner can create matching BlockManager
    m->total_kv_blocks = total_blocks;

    // cuBLAS + RoPE
    CUBLAS_CHECK(cublasCreate(&m->cublas));
    long rope_elems = (long)c.max_position_embeddings * (c.head_dim / 2);
    CUDA_CHECK(cudaMalloc(&m->cos_table, rope_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m->sin_table, rope_elems * sizeof(float)));
    launch_rope_precompute(m->cos_table, m->sin_table,
                           c.max_position_embeddings, c.head_dim, c.rope_theta);

    // Host mirrors
    int mbps = m->kv_cache.max_blocks_per_seq;
    m->h_block_tables = new int[(long)max_batch * mbps];
    m->h_seq_lens     = new int[max_batch];
    m->h_slot_map     = new int64_t[max_T];
    m->h_pos_ids      = new int[max_T];
    memset(m->h_block_tables, -1, (long)max_batch * mbps * sizeof(int));
    memset(m->h_seq_lens,      0, max_batch * sizeof(int));

    CUDA_CHECK(cudaDeviceSynchronize());

    size_t block_bytes = 2ULL * c.num_hidden_layers * KV_BLOCK_SIZE
                       * c.num_key_value_heads * c.head_dim * sizeof(half);
    printf("[INIT] scratch=%d tokens, KV pool=%d blocks (%.1f GB)\n",
           max_T, total_blocks,
           (total_blocks * block_bytes) / (1024.0*1024*1024));
}

void qwen3_free(Qwen3Model* m) {
    if (!m) return;
    free_weights(m->weights);
    paged_kv_cache_free(m->kv_cache);
    if (m->cublas) cublasDestroy(m->cublas);
    cudaFree(m->cos_table);   cudaFree(m->sin_table);
    cudaFree(m->d_hidden);    cudaFree(m->d_residual);
    cudaFree(m->d_Q);         cudaFree(m->d_K);
    cudaFree(m->d_V);         cudaFree(m->d_attn_out);
    cudaFree(m->d_gate);      cudaFree(m->d_up);
    cudaFree(m->d_mlp_mid);   cudaFree(m->d_logits);
    cudaFree(m->d_pos_ids);   cudaFree(m->d_slot_map);
    cudaFree(m->d_tokens);
    delete[] m->h_block_tables;
    delete[] m->h_seq_lens;
    delete[] m->h_slot_map;
    delete[] m->h_pos_ids;
    delete m;
}

void qwen3_reset(Qwen3Model* m) {
    int mbps = m->kv_cache.max_blocks_per_seq;
    m->kv_cache.num_free = m->kv_cache.total_blocks;
    for (int i = 0; i < m->kv_cache.total_blocks; i++)
        m->kv_cache.free_stack[i] = i;
    memset(m->h_block_tables, -1, (long)m->max_batch * mbps * sizeof(int));
    memset(m->h_seq_lens,      0, (long)m->max_batch * sizeof(int));
    CUDA_CHECK(cudaMemcpy(m->kv_cache.block_tables, m->h_block_tables, (long)m->max_batch * mbps * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m->kv_cache.seq_lens, m->h_seq_lens, (long)m->max_batch * sizeof(int), cudaMemcpyHostToDevice));
}

// Free KV blocks for a finished sequence at batch slot b; reset its state.
void qwen3_free_seq_slot(Qwen3Model* m, int b) {
    int mbps = m->kv_cache.max_blocks_per_seq;
    int* row = m->h_block_tables + (long)b * mbps;
    for (int j = 0; j < mbps; j++) {
        if (row[j] >= 0) {
            m->kv_cache.free_stack[m->kv_cache.num_free++] = row[j];
            row[j] = -1;
        }
    }
    m->h_seq_lens[b] = 0;
}

// prefill with variable lengths
half* qwen3_prefill(Qwen3Model* m, const std::vector<Sequence*>& batch,
                    cudaStream_t stream) {
    const Qwen3Config& c = m->config;
    int B = batch.size();

    std::vector<int> h_tokens, h_pos_ids;
    std::vector<int64_t> h_slot_map;
    std::vector<int> h_cu_seqlens = {0}; //cumulative sequence lengths for launch_embedding

    // Collect per-sequence tokens; pad to max length so the batch is uniform.
    int mbps = m->kv_cache.max_blocks_per_seq;
    std::vector<int> actual_lens(B);
    int S = 0;  // max sequence length (for padding)
    for (int b = 0; b < B; b++) {
        const Sequence* seq = batch[b];
        int start = seq->num_cached_tokens;
        int end   = seq->num_tokens;
        actual_lens[b] = end - start;
        if (actual_lens[b] > S) S = actual_lens[b];
    }

    for (int b = 0; b < B; b++) {
        const Sequence* seq = batch[b];
        int slot  = seq->batch_slot;
        int start = seq->num_cached_tokens;
        int end   = seq->num_tokens;
        // New sequence starting from scratch — reset any stale slot state.
        if (start == 0) {
            m->h_seq_lens[slot] = 0;
            for (int j = 0; j < mbps; j++)
                m->h_block_tables[(long)slot * mbps + j] = -1;
        }
        for (int i = start; i < end; i++){
            h_tokens.push_back(seq->token_ids[i]);
            h_pos_ids.push_back(i);
            h_slot_map.push_back(
                paged_kv_cache_append_slot(m->kv_cache, m->h_block_tables, m->h_seq_lens, slot)
            );
        }
        // Pad shorter sequences to S; padding slots = -1 → KV write is skipped.
        for (int p = actual_lens[b]; p < S; p++) {
            h_tokens.push_back(0);
            h_pos_ids.push_back(0);
            h_slot_map.push_back(-1LL);
        }
        h_cu_seqlens.push_back(h_cu_seqlens.back() + S);
    }

    int T = B * S;

    CUBLAS_CHECK(cublasSetStream(m->cublas, stream));
    CUDA_CHECK(cudaMemcpyAsync(m->d_tokens, h_tokens.data(), T * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m->d_pos_ids, h_pos_ids.data(), T * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m->d_slot_map, h_slot_map.data(), T * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m->kv_cache.seq_lens, m->h_seq_lens, m->max_batch * sizeof(int), cudaMemcpyHostToDevice, stream));

    launch_embedding(m->d_hidden, m->weights.embed_tokens, m->d_tokens, T, c.vocab_size, c.hidden_size, stream);

    for (int l = 0; l < c.num_hidden_layers; l++)
        qwen3_layer_forward(m, l, T, B, S, true, stream);

    // Gather the last *real* token of each sequence (padding may make sequences unequal in len).
    // Reuse d_tokens as a temporary int[B] buffer for offsets.
    {
        std::vector<int> h_offsets(B);
        for (int b = 0; b < B; b++)
            h_offsets[b] = b * S + actual_lens[b] - 1;
        CUDA_CHECK(cudaMemcpyAsync(m->d_tokens, h_offsets.data(), B * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        gather_at_offsets_kernel<<<B, 256, 0, stream>>>(
            m->d_residual, m->d_hidden, m->d_tokens, c.hidden_size);
    }
    launch_rmsnorm(m->d_residual, m->d_residual,
                   m->weights.final_norm, B, c.hidden_size, c.rms_norm_eps, stream);
    linear_half(m->cublas, m->d_residual, m->weights.lm_head,
                m->d_logits, B, c.vocab_size, c.hidden_size);

    return m->d_logits;
}


half* qwen3_decode(Qwen3Model* m, const std::vector<Sequence*>& batch,
                    cudaStream_t stream) {
    const Qwen3Config& c = m->config;
    int B = batch.size();
    // One token per sequence
    std::vector<int> h_tokens;
    for (auto* seq : batch)
        h_tokens.push_back(seq->last_token_id);

    CUDA_CHECK(cudaMemcpyAsync(m->d_tokens, h_tokens.data(),
                               B * sizeof(int), cudaMemcpyHostToDevice, stream));

    CUBLAS_CHECK(cublasSetStream(m->cublas, stream));

    // Build slot_mapping, position_ids, and compact block_tables/seq_lens for decode.
    // Each sequence has a stable batch_slot that indexes h_block_tables / h_seq_lens.
    // The GPU decode kernel uses contiguous rows 0..B-1, so we remap here.
    int mbps = m->kv_cache.max_blocks_per_seq;
    std::vector<int>  h_block_table_compact(B * mbps, -1);
    std::vector<int>  h_seq_lens_compact(B, 0);
    for (int b = 0; b < B; b++) {
        int slot = batch[b]->batch_slot;
        m->h_slot_map[b] = paged_kv_cache_append_slot(m->kv_cache, m->h_block_tables, m->h_seq_lens, slot);
        m->h_pos_ids[b]  = m->h_seq_lens[slot] - 1;
        h_seq_lens_compact[b] = m->h_seq_lens[slot];
        memcpy(&h_block_table_compact[b * mbps],
               m->h_block_tables + (long)slot * mbps,
               mbps * sizeof(int));
    }

    CUDA_CHECK(cudaMemcpyAsync(m->d_pos_ids,  m->h_pos_ids,
                               B * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m->d_slot_map, m->h_slot_map,
                               B * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m->kv_cache.block_tables, h_block_table_compact.data(),
                               (long)B * mbps * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m->kv_cache.seq_lens, h_seq_lens_compact.data(),
                               B * sizeof(int), cudaMemcpyHostToDevice, stream));

    // Token embedding → d_hidden [B, hidden_size]
    launch_embedding(m->d_hidden, m->weights.embed_tokens,
                     m->d_tokens, B, c.vocab_size, c.hidden_size, stream);

    for (int l = 0; l < c.num_hidden_layers; l++)
        qwen3_layer_forward(m, l, B, B, /*S=*/1, /*is_prefill=*/false, stream);

    // S=1 → gather_last_tokens is a trivial copy of d_hidden → d_residual
    compute_logits(m, B, /*S=*/1, stream);

    return m->d_logits;
}

// =============================================================================
// Training: state allocation / free
// =============================================================================

Qwen3TrainState* qwen3_train_state_alloc(const Qwen3Config& c, int B, int S) {
    auto* s = new Qwen3TrainState{};
    s->B = B;  s->S = S;  s->T = B * S;
    s->num_layers = c.num_hidden_layers;
    const int T = s->T;
    const int L = s->num_layers;

    // Per-layer element counts (half)
    const long input_sz = (long)T * c.hidden_size;
    const long qraw_sz  = (long)T * c.q_dim;
    const long kraw_sz  = (long)T * c.kv_dim;
    const long q_sz     = (long)T * c.q_dim;
    const long k_sz     = (long)T * c.kv_dim;
    const long v_sz     = (long)T * c.kv_dim;
    const long o_sz     = (long)T * c.q_dim;
    const long pattn_sz = (long)T * c.hidden_size;
    const long gate_sz  = (long)T * c.intermediate_size;
    const long up_sz    = (long)T * c.intermediate_size;
    const long final_sz = (long)T * c.hidden_size;

    const long per_layer_halfs = input_sz + qraw_sz + kraw_sz + q_sz + k_sz
                               + v_sz + o_sz + pattn_sz + gate_sz + up_sz;
    const long total_halfs = per_layer_halfs * L + final_sz;

    // Per-layer element counts (float) — LSE: [B, S, H_q] = [T, H_q]
    const long lse_per_layer = (long)T * c.num_attention_heads;
    const long total_floats  = lse_per_layer * L;

    // Allocate GPU pools
    CUDA_CHECK(cudaMalloc(&s->activation_pool, total_halfs * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->lse_pool, total_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_token_ids,  T * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s->d_target_ids, T * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s->d_pos_ids,    T * sizeof(int)));

    // Allocate host pointer arrays
    s->layer_input     = new half*[L];
    s->layer_Q_raw     = new half*[L];
    s->layer_K_raw     = new half*[L];
    s->layer_Q         = new half*[L];
    s->layer_K         = new half*[L];
    s->layer_V         = new half*[L];
    s->layer_O         = new half*[L];
    s->layer_post_attn = new half*[L];
    s->layer_gate      = new half*[L];
    s->layer_up        = new half*[L];
    s->layer_lse       = new float*[L];

    // Partition half pool into per-layer buffers
    half* p = s->activation_pool;
    for (int l = 0; l < L; l++) {
        s->layer_input[l]     = p;  p += input_sz;
        s->layer_Q_raw[l]     = p;  p += qraw_sz;
        s->layer_K_raw[l]     = p;  p += kraw_sz;
        s->layer_Q[l]         = p;  p += q_sz;
        s->layer_K[l]         = p;  p += k_sz;
        s->layer_V[l]         = p;  p += v_sz;
        s->layer_O[l]         = p;  p += o_sz;
        s->layer_post_attn[l] = p;  p += pattn_sz;
        s->layer_gate[l]      = p;  p += gate_sz;
        s->layer_up[l]        = p;  p += up_sz;
    }
    s->final_hidden = p;  // last chunk: [T, hidden]

    // Partition float pool into per-layer LSE
    float* fp = s->lse_pool;
    for (int l = 0; l < L; l++) {
        s->layer_lse[l] = fp;  fp += lse_per_layer;
    }

    printf("[TrainState] Allocated: B=%d S=%d T=%d  activations=%.1f MB  LSE=%.2f MB\n",
           B, S, T,
           total_halfs * sizeof(half) / (1024.0 * 1024.0),
           total_floats * sizeof(float) / (1024.0 * 1024.0));

    return s;
}

void qwen3_train_state_free(Qwen3TrainState* s) {
    if (!s) return;
    cudaFree(s->activation_pool);
    cudaFree(s->lse_pool);
    cudaFree(s->d_token_ids);
    cudaFree(s->d_target_ids);
    cudaFree(s->d_pos_ids);
    delete[] s->layer_input;
    delete[] s->layer_Q_raw;
    delete[] s->layer_K_raw;
    delete[] s->layer_Q;
    delete[] s->layer_K;
    delete[] s->layer_V;
    delete[] s->layer_O;
    delete[] s->layer_post_attn;
    delete[] s->layer_gate;
    delete[] s->layer_up;
    delete[] s->layer_lse;
    delete s;
}

// =============================================================================
// Training forward pass
// =============================================================================

void qwen3_forward(
    Qwen3Model*      m,
    Qwen3TrainState* state,
    const int*       h_token_ids,   // [B*S] host
    const int*       h_target_ids,  // [B*S] host
    float*           d_log_probs,   // [B*S] device output
    int B, int S,
    cudaStream_t     stream
) {
    const Qwen3Config& c = m->config;
    const int T = B * S;
    state->B = B;  state->S = S;  state->T = T;

    // --- Upload token IDs, target IDs, position IDs ---
    CUDA_CHECK(cudaMemcpyAsync(state->d_token_ids, h_token_ids,
                               T * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(state->d_target_ids, h_target_ids,
                               T * sizeof(int), cudaMemcpyHostToDevice, stream));

    // Position IDs: 0..S-1 repeated for each batch element
    {
        std::vector<int> h_pos(T);
        for (int b = 0; b < B; b++)
            for (int s = 0; s < S; s++)
                h_pos[b * S + s] = s;
        CUDA_CHECK(cudaMemcpyAsync(state->d_pos_ids, h_pos.data(),
                                   T * sizeof(int), cudaMemcpyHostToDevice, stream));
    }

    CUBLAS_CHECK(cublasSetStream(m->cublas, stream));

    // --- Embedding ---
    launch_embedding(m->d_hidden, m->weights.embed_tokens,
                     state->d_token_ids, T, c.vocab_size, c.hidden_size, stream);

    // --- Transformer layers ---
    for (int l = 0; l < c.num_hidden_layers; l++) {
        const Qwen3LayerWeights& L = m->weights.layers[l];

        // Save input hidden state (= residual for this layer)
        CUDA_CHECK(cudaMemcpyAsync(state->layer_input[l], m->d_hidden,
                                   (long)T * c.hidden_size * sizeof(half),
                                   cudaMemcpyDeviceToDevice, stream));

        // Input RMSNorm: d_hidden = norm(layer_input)
        launch_rmsnorm(m->d_hidden, state->layer_input[l],
                       L.input_layernorm, T, c.hidden_size, c.rms_norm_eps, stream);

        // QKV projections → save raw Q, K into state; V directly into state
        linear_half(m->cublas, m->d_hidden, L.q_proj,
                    state->layer_Q_raw[l], T, c.q_dim, c.hidden_size);
        linear_half(m->cublas, m->d_hidden, L.k_proj,
                    state->layer_K_raw[l], T, c.kv_dim, c.hidden_size);
        linear_half(m->cublas, m->d_hidden, L.v_proj,
                    state->layer_V[l],     T, c.kv_dim, c.hidden_size);

        // QK-Norm: Q_raw → Q, K_raw → K  (separate in/out, no copy needed)
        launch_rmsnorm(state->layer_Q[l], state->layer_Q_raw[l],
                       L.q_norm, T * c.num_attention_heads, c.head_dim, c.rms_norm_eps, stream);
        launch_rmsnorm(state->layer_K[l], state->layer_K_raw[l],
                       L.k_norm, T * c.num_key_value_heads, c.head_dim, c.rms_norm_eps, stream);

        // RoPE in-place on Q, K (state buffers)
        launch_rope(state->layer_Q[l], state->layer_K[l],
                    m->cos_table, m->sin_table, state->d_pos_ids,
                    T, c.num_attention_heads, c.num_key_value_heads, c.head_dim, stream);

        // Flash Attention 2 prefill (no KV cache) — saves LSE for backward
        launch_flash_attention_prefill(
            state->layer_Q[l], state->layer_K[l], state->layer_V[l],
            state->layer_O[l],
            B, S, c.num_attention_heads, c.num_key_value_heads, c.head_dim,
            stream, state->layer_lse[l]);

        // O projection + residual
        linear_half(m->cublas, state->layer_O[l], L.o_proj,
                    m->d_hidden, T, c.hidden_size, c.q_dim);
        add_residual(m->d_hidden, state->layer_input[l], T * c.hidden_size, stream);

        // Save post-attention hidden (residual for MLP block)
        CUDA_CHECK(cudaMemcpyAsync(state->layer_post_attn[l], m->d_hidden,
                                   (long)T * c.hidden_size * sizeof(half),
                                   cudaMemcpyDeviceToDevice, stream));

        // Post-attention RMSNorm
        launch_rmsnorm(m->d_hidden, state->layer_post_attn[l],
                       L.post_attn_layernorm, T, c.hidden_size, c.rms_norm_eps, stream);

        // Gate/Up projections → save into state
        linear_half(m->cublas, m->d_hidden, L.gate_proj,
                    state->layer_gate[l], T, c.intermediate_size, c.hidden_size);
        linear_half(m->cublas, m->d_hidden, L.up_proj,
                    state->layer_up[l],   T, c.intermediate_size, c.hidden_size);

        // SwiGLU → scratch d_mlp_mid (not saved; recomputable from gate+up)
        launch_swiglu(m->d_mlp_mid, state->layer_gate[l], state->layer_up[l],
                      T * c.intermediate_size, stream);

        // Down projection + residual
        linear_half(m->cublas, m->d_mlp_mid, L.down_proj,
                    m->d_hidden, T, c.hidden_size, c.intermediate_size);
        add_residual(m->d_hidden, state->layer_post_attn[l], T * c.hidden_size, stream);
    }

    // --- Save final hidden state (before final_norm) ---
    CUDA_CHECK(cudaMemcpyAsync(state->final_hidden, m->d_hidden,
                               (long)T * c.hidden_size * sizeof(half),
                               cudaMemcpyDeviceToDevice, stream));

    // --- Final RMSNorm → d_residual (scratch) ---
    launch_rmsnorm(m->d_residual, state->final_hidden,
                   m->weights.final_norm, T, c.hidden_size, c.rms_norm_eps, stream);

    // --- LM head + log-softmax in chunks ---
    // d_logits is [max_batch, vocab_size] — process at most max_batch tokens per chunk.
    const int chunk = m->max_batch;
    for (int start = 0; start < T; start += chunk) {
        const int n = (start + chunk <= T) ? chunk : (T - start);

        // Project [n, hidden] → [n, vocab]
        linear_half(m->cublas,
                    m->d_residual + (long)start * c.hidden_size,
                    m->weights.lm_head,
                    m->d_logits, n, c.vocab_size, c.hidden_size);

        // Log-softmax + gather target log-prob
        log_softmax_gather_kernel<<<n, 256, 0, stream>>>(
            d_log_probs + start,
            m->d_logits,
            state->d_target_ids + start,
            c.vocab_size);
    }
}

// =============================================================================
// Gradient allocation / free / zero
// =============================================================================

Qwen3Gradients* qwen3_gradients_alloc(const Qwen3Config& c, int max_T) {
    auto* g = new Qwen3Gradients{};
    g->num_layers = c.num_hidden_layers;
    const int L = g->num_layers;
    const int H = c.hidden_size;

    // Per-layer half sizes
    const long q_proj_sz    = (long)c.q_dim * H;
    const long k_proj_sz    = (long)c.kv_dim * H;
    const long v_proj_sz    = (long)c.kv_dim * H;
    const long o_proj_sz    = (long)H * c.q_dim;
    const long gate_proj_sz = (long)c.intermediate_size * H;
    const long up_proj_sz   = (long)c.intermediate_size * H;
    const long down_proj_sz = (long)H * c.intermediate_size;
    const long per_layer_halfs = q_proj_sz + k_proj_sz + v_proj_sz + o_proj_sz
                                + gate_proj_sz + up_proj_sz + down_proj_sz;
    const long embed_sz     = (long)c.vocab_size * H;
    const long total_halfs  = per_layer_halfs * L + embed_sz;

    // Per-layer float sizes
    const long per_layer_floats = (long)H + c.head_dim + c.head_dim + H;
    const long final_norm_sz    = H;
    const long d_buf_sz         = (long)max_T * c.num_attention_heads;
    const long total_floats     = per_layer_floats * L + final_norm_sz + d_buf_sz;

    g->half_pool_bytes  = total_halfs * sizeof(half);
    g->float_pool_bytes = total_floats * sizeof(float);

    CUDA_CHECK(cudaMalloc(&g->half_pool,  g->half_pool_bytes));
    CUDA_CHECK(cudaMalloc(&g->float_pool, g->float_pool_bytes));

    // Allocate host pointer arrays
    g->dW_q_proj          = new half*[L];
    g->dW_k_proj          = new half*[L];
    g->dW_v_proj          = new half*[L];
    g->dW_o_proj          = new half*[L];
    g->dW_gate_proj       = new half*[L];
    g->dW_up_proj         = new half*[L];
    g->dW_down_proj       = new half*[L];
    g->dW_input_norm      = new float*[L];
    g->dW_q_norm          = new float*[L];
    g->dW_k_norm          = new float*[L];
    g->dW_post_attn_norm  = new float*[L];

    // Partition half pool
    half* hp = g->half_pool;
    for (int l = 0; l < L; l++) {
        g->dW_q_proj[l]    = hp; hp += q_proj_sz;
        g->dW_k_proj[l]    = hp; hp += k_proj_sz;
        g->dW_v_proj[l]    = hp; hp += v_proj_sz;
        g->dW_o_proj[l]    = hp; hp += o_proj_sz;
        g->dW_gate_proj[l] = hp; hp += gate_proj_sz;
        g->dW_up_proj[l]   = hp; hp += up_proj_sz;
        g->dW_down_proj[l] = hp; hp += down_proj_sz;
    }
    g->dW_embed = hp;

    // Partition float pool
    float* fp = g->float_pool;
    for (int l = 0; l < L; l++) {
        g->dW_input_norm[l]     = fp; fp += H;
        g->dW_q_norm[l]         = fp; fp += c.head_dim;
        g->dW_k_norm[l]         = fp; fp += c.head_dim;
        g->dW_post_attn_norm[l] = fp; fp += H;
    }
    g->dW_final_norm = fp; fp += final_norm_sz;
    g->D_buf         = fp;

    printf("[Gradients] Allocated: half=%.1f MB  float=%.2f MB\n",
           g->half_pool_bytes / (1024.0 * 1024.0),
           g->float_pool_bytes / (1024.0 * 1024.0));

    return g;
}

void qwen3_gradients_free(Qwen3Gradients* g) {
    if (!g) return;
    cudaFree(g->half_pool);
    cudaFree(g->float_pool);
    delete[] g->dW_q_proj;
    delete[] g->dW_k_proj;
    delete[] g->dW_v_proj;
    delete[] g->dW_o_proj;
    delete[] g->dW_gate_proj;
    delete[] g->dW_up_proj;
    delete[] g->dW_down_proj;
    delete[] g->dW_input_norm;
    delete[] g->dW_q_norm;
    delete[] g->dW_k_norm;
    delete[] g->dW_post_attn_norm;
    delete g;
}

void qwen3_gradients_zero(Qwen3Gradients* g, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(g->half_pool,  0, g->half_pool_bytes,  stream));
    CUDA_CHECK(cudaMemsetAsync(g->float_pool, 0, g->float_pool_bytes, stream));
}

// =============================================================================
// Training backward pass
// =============================================================================

void qwen3_backward(
    Qwen3Model*      m,
    Qwen3TrainState* state,
    Qwen3Gradients*  grads,
    const float*     d_log_probs,   // [B*S] device, FP32
    cudaStream_t     stream
) {
    const Qwen3Config& c = m->config;
    const int T = state->T, B = state->B, S = state->S;
    const int H = c.hidden_size;
    const int V = c.vocab_size;

    CUBLAS_CHECK(cublasSetStream(m->cublas, stream));

    // ================================================================
    // Phase 1: lm_head + log_softmax backward (chunked)
    // ================================================================

    // Recompute normed final hidden → d_residual
    launch_rmsnorm(m->d_residual, state->final_hidden,
                   m->weights.final_norm, T, H, c.rms_norm_eps, stream);

    // Zero d_hidden (accumulate gradient for normed final hidden)
    CUDA_CHECK(cudaMemsetAsync(m->d_hidden, 0, (long)T * H * sizeof(half), stream));

    const int chunk = m->max_batch;
    for (int start = 0; start < T; start += chunk) {
        const int n = (start + chunk <= T) ? chunk : (T - start);

        // Recompute logits for this chunk
        linear_half(m->cublas,
                    m->d_residual + (long)start * H,
                    m->weights.lm_head,
                    m->d_logits, n, V, H);

        // Backward log_softmax_gather: overwrite d_logits with gradients
        log_softmax_gather_backward_kernel<<<n, 256, 0, stream>>>(
            m->d_logits,
            d_log_probs + start,
            state->d_target_ids + start,
            V);

        // lm_head backward: dX → d_gate (scratch), dW → dW_embed (accumulated)
        linear_backward_half(m->cublas,
            m->d_logits,                           // dY [n, V]
            m->d_residual + (long)start * H,       // X  [n, H]
            m->weights.lm_head,                    // W  [V, H]
            m->d_gate,                             // dX [n, H] scratch
            grads->dW_embed,                       // dW [V, H] accumulated
            n, V, H);

        // Accumulate dX into d_hidden
        add_residual(m->d_hidden + (long)start * H, m->d_gate, n * H, stream);
    }

    // ================================================================
    // Phase 2: final_norm backward
    // ================================================================

    launch_rmsnorm_backward(
        m->d_residual,                // dX [T, H]
        grads->dW_final_norm,         // dW [H]
        m->d_hidden,                  // dY [T, H]
        state->final_hidden,          // x  [T, H] (saved input)
        m->weights.final_norm,        // w  [H]
        T, H, c.rms_norm_eps, stream);

    // d_residual = gradient w.r.t. final_hidden → move to d_hidden
    CUDA_CHECK(cudaMemcpyAsync(m->d_hidden, m->d_residual,
                               (long)T * H * sizeof(half),
                               cudaMemcpyDeviceToDevice, stream));

    // ================================================================
    // Phase 3: Transformer layers backward (reverse order)
    // ================================================================

    for (int l = c.num_hidden_layers - 1; l >= 0; l--) {
        const Qwen3LayerWeights& LW = m->weights.layers[l];

        // Save d_output for residual path
        CUDA_CHECK(cudaMemcpyAsync(m->d_residual, m->d_hidden,
                                   (long)T * H * sizeof(half),
                                   cudaMemcpyDeviceToDevice, stream));

        // ---- MLP backward ----

        // Recompute mlp_mid = swiglu(gate[l], up[l])
        launch_swiglu(m->d_mlp_mid, state->layer_gate[l], state->layer_up[l],
                      T * c.intermediate_size, stream);

        // down_proj backward
        linear_backward_half(m->cublas,
            m->d_hidden,                        // dY [T, H]
            m->d_mlp_mid,                       // X  [T, inter] (recomputed)
            LW.down_proj,                       // W  [H, inter]
            m->d_up,                            // dX [T, inter]
            grads->dW_down_proj[l],             // dW [H, inter]
            T, H, c.intermediate_size);

        // SwiGLU backward: dOut=d_up → dGate=d_gate, dUp=d_mlp_mid
        launch_swiglu_backward(
            m->d_gate,                          // dGate [T, inter]
            m->d_mlp_mid,                       // dUp   [T, inter]
            m->d_up,                            // dOut  [T, inter]
            state->layer_gate[l],               // saved gate
            state->layer_up[l],                 // saved up
            T * c.intermediate_size, stream);

        // Recompute normed post_attn → d_K (scratch [T, H]; kv_dim=H)
        launch_rmsnorm(m->d_K, state->layer_post_attn[l],
                       LW.post_attn_layernorm, T, H, c.rms_norm_eps, stream);

        // gate_proj backward: dX → d_Q
        linear_backward_half(m->cublas,
            m->d_gate,                          // dY [T, inter]
            m->d_K,                             // X  [T, H] (normed post_attn)
            LW.gate_proj,                       // W  [inter, H]
            m->d_Q,                             // dX [T, H]
            grads->dW_gate_proj[l],             // dW [inter, H]
            T, c.intermediate_size, H);

        // up_proj backward: dX → d_V
        linear_backward_half(m->cublas,
            m->d_mlp_mid,                       // dY [T, inter] (=dUp)
            m->d_K,                             // X  [T, H]
            LW.up_proj,                         // W  [inter, H]
            m->d_V,                             // dX [T, H]
            grads->dW_up_proj[l],               // dW [inter, H]
            T, c.intermediate_size, H);

        // Sum gate + up dX → d_hidden
        CUDA_CHECK(cudaMemcpyAsync(m->d_hidden, m->d_Q,
                                   (long)T * H * sizeof(half),
                                   cudaMemcpyDeviceToDevice, stream));
        add_residual(m->d_hidden, m->d_V, T * H, stream);

        // post_attn_norm backward
        launch_rmsnorm_backward(
            m->d_Q,                             // dX [T, H]
            grads->dW_post_attn_norm[l],        // dW [H]
            m->d_hidden,                        // dY [T, H]
            state->layer_post_attn[l],          // x  [T, H]
            LW.post_attn_layernorm,             // w  [H]
            T, H, c.rms_norm_eps, stream);

        // Add MLP residual
        add_residual(m->d_Q, m->d_residual, T * H, stream);

        // ---- Attention backward ----

        // Save d_post_attn_total for attention residual
        CUDA_CHECK(cudaMemcpyAsync(m->d_residual, m->d_Q,
                                   (long)T * H * sizeof(half),
                                   cudaMemcpyDeviceToDevice, stream));

        // o_proj backward: dX → d_attn_out (= dO for flash attention)
        linear_backward_half(m->cublas,
            m->d_Q,                             // dY [T, H]
            state->layer_O[l],                  // X  [T, q_dim]
            LW.o_proj,                          // W  [H, q_dim]
            m->d_attn_out,                      // dX [T, q_dim] = dO
            grads->dW_o_proj[l],                // dW [H, q_dim]
            T, H, c.q_dim);

        // Flash attention backward
        launch_flash_attention_backward(
            state->layer_Q[l], state->layer_K[l], state->layer_V[l],
            state->layer_O[l],
            m->d_attn_out,                      // dO [T, q_dim]
            state->layer_lse[l],
            grads->D_buf,
            m->d_Q,                             // dQ [T, q_dim]
            m->d_K,                             // dK [T, kv_dim]
            m->d_V,                             // dV [T, kv_dim]
            B, S, c.num_attention_heads, c.num_key_value_heads, c.head_dim,
            stream);

        // RoPE backward (in-place on dQ, dK)
        launch_rope_backward(m->d_Q, m->d_K,
                            m->cos_table, m->sin_table, state->d_pos_ids,
                            T, c.num_attention_heads, c.num_key_value_heads,
                            c.head_dim, stream);

        // QK-norm backward (Q): dX → d_attn_out
        launch_rmsnorm_backward(
            m->d_attn_out,                      // dX [T, q_dim]
            grads->dW_q_norm[l],                // dW [head_dim]
            m->d_Q,                             // dY [T, q_dim]
            state->layer_Q_raw[l],              // x
            LW.q_norm,                          // w  [head_dim]
            T * c.num_attention_heads, c.head_dim, c.rms_norm_eps, stream);

        // QK-norm backward (K): dX → d_mlp_mid (scratch)
        launch_rmsnorm_backward(
            m->d_mlp_mid,                       // dX [T, kv_dim]
            grads->dW_k_norm[l],                // dW [head_dim]
            m->d_K,                             // dY [T, kv_dim]
            state->layer_K_raw[l],              // x
            LW.k_norm,                          // w  [head_dim]
            T * c.num_key_value_heads, c.head_dim, c.rms_norm_eps, stream);

        // Recompute normed input → d_gate (scratch [T, H])
        launch_rmsnorm(m->d_gate, state->layer_input[l],
                       LW.input_layernorm, T, H, c.rms_norm_eps, stream);

        // Q proj backward: dX → d_hidden
        linear_backward_half(m->cublas,
            m->d_attn_out,                      // dY [T, q_dim] (=dQ_raw)
            m->d_gate,                          // X  [T, H] (normed input)
            LW.q_proj,                          // W  [q_dim, H]
            m->d_hidden,                        // dX [T, H]
            grads->dW_q_proj[l],                // dW [q_dim, H]
            T, c.q_dim, H);

        // K proj backward: dX → d_Q (scratch), then accumulate
        linear_backward_half(m->cublas,
            m->d_mlp_mid,                       // dY [T, kv_dim] (=dK_raw)
            m->d_gate,                          // X  [T, H]
            LW.k_proj,                          // W  [kv_dim, H]
            m->d_Q,                             // dX [T, H] scratch
            grads->dW_k_proj[l],                // dW [kv_dim, H]
            T, c.kv_dim, H);
        add_residual(m->d_hidden, m->d_Q, T * H, stream);

        // V proj backward: dX → d_Q (scratch), then accumulate
        linear_backward_half(m->cublas,
            m->d_V,                             // dY [T, kv_dim]
            m->d_gate,                          // X  [T, H]
            LW.v_proj,                          // W  [kv_dim, H]
            m->d_Q,                             // dX [T, H] scratch
            grads->dW_v_proj[l],                // dW [kv_dim, H]
            T, c.kv_dim, H);
        add_residual(m->d_hidden, m->d_Q, T * H, stream);

        // input_norm backward
        launch_rmsnorm_backward(
            m->d_Q,                             // dX [T, H]
            grads->dW_input_norm[l],            // dW [H]
            m->d_hidden,                        // dY [T, H]
            state->layer_input[l],              // x  [T, H]
            LW.input_layernorm,                 // w  [H]
            T, H, c.rms_norm_eps, stream);

        // Add attention residual
        add_residual(m->d_Q, m->d_residual, T * H, stream);

        // Move to d_hidden for next layer
        CUDA_CHECK(cudaMemcpyAsync(m->d_hidden, m->d_Q,
                                   (long)T * H * sizeof(half),
                                   cudaMemcpyDeviceToDevice, stream));
    }

    // ================================================================
    // Phase 4: Embedding backward
    // ================================================================

    launch_embedding_backward(
        grads->dW_embed,               // dW [V, H] (accumulated on top of lm_head grad)
        m->d_hidden,                   // dOut [T, H]
        state->d_token_ids,            // token_ids [T]
        T, H, stream);
}

// =============================================================================
// AdamW Optimizer
// =============================================================================

// Helper: copy FP16 weights to FP32 master copy
__global__ static void fp16_to_fp32_kernel(
    float*      __restrict__ dst,
    const half* __restrict__ src,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

static void fp16_to_fp32(float* dst, const half* src, int n, cudaStream_t stream) {
    fp16_to_fp32_kernel<<<(n + 255) / 256, 256, 0, stream>>>(dst, src, n);
}

Qwen3AdamW* qwen3_adamw_alloc(const Qwen3Config& c, const Qwen3Weights& w,
                                const AdamWConfig& config) {
    auto* opt = new Qwen3AdamW{};
    opt->config = config;
    opt->step = 0;
    opt->num_layers = c.num_hidden_layers;
    const int L = opt->num_layers;
    const int H = c.hidden_size;

    // FP16 param sizes
    const long q_proj_sz    = (long)c.q_dim * H;
    const long k_proj_sz    = (long)c.kv_dim * H;
    const long v_proj_sz    = (long)c.kv_dim * H;
    const long o_proj_sz    = (long)H * c.q_dim;
    const long gate_proj_sz = (long)c.intermediate_size * H;
    const long up_proj_sz   = (long)c.intermediate_size * H;
    const long down_proj_sz = (long)H * c.intermediate_size;
    const long per_layer_fp16 = q_proj_sz + k_proj_sz + v_proj_sz + o_proj_sz
                               + gate_proj_sz + up_proj_sz + down_proj_sz;
    const long embed_sz = (long)c.vocab_size * H;
    const long total_fp16_params = per_layer_fp16 * L + embed_sz;

    // FP32 param sizes
    const long per_layer_fp32 = (long)H + c.head_dim + c.head_dim + H;
    const long final_norm_sz = H;
    const long total_fp32_params = per_layer_fp32 * L + final_norm_sz;

    // Allocate: 3x for FP16 (master_w + m + v), 2x for FP32 (m + v)
    opt->fp16_pool_bytes = (size_t)total_fp16_params * 3 * sizeof(float);
    opt->fp32_pool_bytes = (size_t)total_fp32_params * 2 * sizeof(float);

    CUDA_CHECK(cudaMalloc(&opt->fp16_pool, opt->fp16_pool_bytes));
    CUDA_CHECK(cudaMalloc(&opt->fp32_pool, opt->fp32_pool_bytes));

    // Zero m and v (master_w will be initialized from model weights below)
    CUDA_CHECK(cudaMemset(opt->fp16_pool, 0, opt->fp16_pool_bytes));
    CUDA_CHECK(cudaMemset(opt->fp32_pool, 0, opt->fp32_pool_bytes));

    // Allocate host pointer arrays
    opt->q_proj    = new Qwen3AdamW::FP16ParamState[L];
    opt->k_proj    = new Qwen3AdamW::FP16ParamState[L];
    opt->v_proj    = new Qwen3AdamW::FP16ParamState[L];
    opt->o_proj    = new Qwen3AdamW::FP16ParamState[L];
    opt->gate_proj = new Qwen3AdamW::FP16ParamState[L];
    opt->up_proj   = new Qwen3AdamW::FP16ParamState[L];
    opt->down_proj = new Qwen3AdamW::FP16ParamState[L];

    opt->input_norm     = new Qwen3AdamW::FP32ParamState[L];
    opt->q_norm         = new Qwen3AdamW::FP32ParamState[L];
    opt->k_norm         = new Qwen3AdamW::FP32ParamState[L];
    opt->post_attn_norm = new Qwen3AdamW::FP32ParamState[L];

    // Partition FP16 pool: [master_w_all | m_all | v_all]
    float* fp16_master = opt->fp16_pool;
    float* fp16_m      = fp16_master + total_fp16_params;
    float* fp16_v      = fp16_m + total_fp16_params;

    // Helper: assign (master, m, v) for a FP16 param and advance pointers
    auto assign_fp16 = [&](Qwen3AdamW::FP16ParamState& ps, long sz) {
        ps.master_w = fp16_master;  fp16_master += sz;
        ps.m        = fp16_m;       fp16_m      += sz;
        ps.v        = fp16_v;       fp16_v      += sz;
    };

    for (int l = 0; l < L; l++) {
        assign_fp16(opt->q_proj[l],    q_proj_sz);
        assign_fp16(opt->k_proj[l],    k_proj_sz);
        assign_fp16(opt->v_proj[l],    v_proj_sz);
        assign_fp16(opt->o_proj[l],    o_proj_sz);
        assign_fp16(opt->gate_proj[l], gate_proj_sz);
        assign_fp16(opt->up_proj[l],   up_proj_sz);
        assign_fp16(opt->down_proj[l], down_proj_sz);
    }
    assign_fp16(opt->embed, embed_sz);

    // Partition FP32 pool: [m_all | v_all]
    float* fp32_m = opt->fp32_pool;
    float* fp32_v = fp32_m + total_fp32_params;

    auto assign_fp32 = [&](Qwen3AdamW::FP32ParamState& ps, long sz) {
        ps.m = fp32_m;  fp32_m += sz;
        ps.v = fp32_v;  fp32_v += sz;
    };

    for (int l = 0; l < L; l++) {
        assign_fp32(opt->input_norm[l],     H);
        assign_fp32(opt->q_norm[l],         c.head_dim);
        assign_fp32(opt->k_norm[l],         c.head_dim);
        assign_fp32(opt->post_attn_norm[l], H);
    }
    assign_fp32(opt->final_norm, final_norm_sz);

    // Initialize master weights from model FP16 weights
    for (int l = 0; l < L; l++) {
        const auto& lw = w.layers[l];
        fp16_to_fp32(opt->q_proj[l].master_w,    lw.q_proj,    q_proj_sz,    0);
        fp16_to_fp32(opt->k_proj[l].master_w,    lw.k_proj,    k_proj_sz,    0);
        fp16_to_fp32(opt->v_proj[l].master_w,    lw.v_proj,    v_proj_sz,    0);
        fp16_to_fp32(opt->o_proj[l].master_w,    lw.o_proj,    o_proj_sz,    0);
        fp16_to_fp32(opt->gate_proj[l].master_w, lw.gate_proj, gate_proj_sz, 0);
        fp16_to_fp32(opt->up_proj[l].master_w,   lw.up_proj,   up_proj_sz,   0);
        fp16_to_fp32(opt->down_proj[l].master_w, lw.down_proj, down_proj_sz, 0);
    }
    fp16_to_fp32(opt->embed.master_w, w.embed_tokens, embed_sz, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[AdamW] Allocated: fp16_state=%.1f MB  fp32_state=%.2f MB  (lr=%.1e wd=%.2f)\n",
           opt->fp16_pool_bytes / (1024.0 * 1024.0),
           opt->fp32_pool_bytes / (1024.0 * 1024.0),
           config.lr, config.weight_decay);

    return opt;
}

void qwen3_adamw_free(Qwen3AdamW* opt) {
    if (!opt) return;
    cudaFree(opt->fp16_pool);
    cudaFree(opt->fp32_pool);
    delete[] opt->q_proj;
    delete[] opt->k_proj;
    delete[] opt->v_proj;
    delete[] opt->o_proj;
    delete[] opt->gate_proj;
    delete[] opt->up_proj;
    delete[] opt->down_proj;
    delete[] opt->input_norm;
    delete[] opt->q_norm;
    delete[] opt->k_norm;
    delete[] opt->post_attn_norm;
    delete opt;
}

void qwen3_adamw_step(
    Qwen3Model* model,
    Qwen3Gradients* grads,
    Qwen3AdamW* opt,
    cudaStream_t stream
) {
    opt->step++;
    const Qwen3Config& c = model->config;
    const int L = c.num_hidden_layers;
    const int H = c.hidden_size;
    const auto& cfg = opt->config;

    float bc1 = 1.0f - powf(cfg.beta1, (float)opt->step);
    float bc2 = 1.0f - powf(cfg.beta2, (float)opt->step);

    // FP16 param sizes
    const int q_proj_sz    = c.q_dim * H;
    const int k_proj_sz    = c.kv_dim * H;
    const int v_proj_sz    = c.kv_dim * H;
    const int o_proj_sz    = H * c.q_dim;
    const int gate_proj_sz = c.intermediate_size * H;
    const int up_proj_sz   = c.intermediate_size * H;
    const int down_proj_sz = H * c.intermediate_size;
    const int embed_sz     = c.vocab_size * H;

    // Update all per-layer FP16 weights
    for (int l = 0; l < L; l++) {
        auto& lw = model->weights.layers[l];

        launch_adamw_fp16(opt->q_proj[l].master_w, lw.q_proj, grads->dW_q_proj[l],
                          opt->q_proj[l].m, opt->q_proj[l].v, q_proj_sz,
                          cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
                          bc1, bc2, stream);

        launch_adamw_fp16(opt->k_proj[l].master_w, lw.k_proj, grads->dW_k_proj[l],
                          opt->k_proj[l].m, opt->k_proj[l].v, k_proj_sz,
                          cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
                          bc1, bc2, stream);

        launch_adamw_fp16(opt->v_proj[l].master_w, lw.v_proj, grads->dW_v_proj[l],
                          opt->v_proj[l].m, opt->v_proj[l].v, v_proj_sz,
                          cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
                          bc1, bc2, stream);

        launch_adamw_fp16(opt->o_proj[l].master_w, lw.o_proj, grads->dW_o_proj[l],
                          opt->o_proj[l].m, opt->o_proj[l].v, o_proj_sz,
                          cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
                          bc1, bc2, stream);

        launch_adamw_fp16(opt->gate_proj[l].master_w, lw.gate_proj, grads->dW_gate_proj[l],
                          opt->gate_proj[l].m, opt->gate_proj[l].v, gate_proj_sz,
                          cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
                          bc1, bc2, stream);

        launch_adamw_fp16(opt->up_proj[l].master_w, lw.up_proj, grads->dW_up_proj[l],
                          opt->up_proj[l].m, opt->up_proj[l].v, up_proj_sz,
                          cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
                          bc1, bc2, stream);

        launch_adamw_fp16(opt->down_proj[l].master_w, lw.down_proj, grads->dW_down_proj[l],
                          opt->down_proj[l].m, opt->down_proj[l].v, down_proj_sz,
                          cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
                          bc1, bc2, stream);
    }

    // Embedding (tied = lm_head)
    launch_adamw_fp16(opt->embed.master_w, model->weights.embed_tokens, grads->dW_embed,
                      opt->embed.m, opt->embed.v, embed_sz,
                      cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
                      bc1, bc2, stream);

    // Update all per-layer FP32 weights (norms — typically no weight decay)
    float norm_wd = 0.0f;  // standard practice: no weight decay on norm params

    for (int l = 0; l < L; l++) {
        auto& lw = model->weights.layers[l];

        launch_adamw_fp32(lw.input_layernorm, grads->dW_input_norm[l],
                          opt->input_norm[l].m, opt->input_norm[l].v, H,
                          cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, norm_wd,
                          bc1, bc2, stream);

        launch_adamw_fp32(lw.q_norm, grads->dW_q_norm[l],
                          opt->q_norm[l].m, opt->q_norm[l].v, c.head_dim,
                          cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, norm_wd,
                          bc1, bc2, stream);

        launch_adamw_fp32(lw.k_norm, grads->dW_k_norm[l],
                          opt->k_norm[l].m, opt->k_norm[l].v, c.head_dim,
                          cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, norm_wd,
                          bc1, bc2, stream);

        launch_adamw_fp32(lw.post_attn_layernorm, grads->dW_post_attn_norm[l],
                          opt->post_attn_norm[l].m, opt->post_attn_norm[l].v, H,
                          cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, norm_wd,
                          bc1, bc2, stream);
    }

    // Final norm
    launch_adamw_fp32(model->weights.final_norm, grads->dW_final_norm,
                      opt->final_norm.m, opt->final_norm.v, H,
                      cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, norm_wd,
                      bc1, bc2, stream);
}
