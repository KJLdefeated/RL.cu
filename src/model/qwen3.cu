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
