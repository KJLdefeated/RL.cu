// src/model/qwen3.cu
//
// Qwen3 transformer forward pass.
//
// Per-layer computation (docs/DESIGN.md §1):
//
//   residual = hidden
//   hidden   = RMSNorm(hidden, input_layernorm)
//   Q = hidden @ q_proj^T                      [T, q_dim]
//   K = hidden @ k_proj^T                      [T, kv_dim]
//   V = hidden @ v_proj^T                      [T, kv_dim]
//   Q = RMSNorm_per_head(Q, q_norm)  rows=T*H_q,  cols=head_dim  ← BEFORE RoPE
//   K = RMSNorm_per_head(K, k_norm)  rows=T*H_kv, cols=head_dim  ← BEFORE RoPE
//   Q, K = RoPE(Q, K, position_ids)
//   KV_cache.append(K, V)
//   attn_out = FA2(Q, K, V)  or  PagedAttn(Q, K_cache, V_cache)
//   hidden   = attn_out @ o_proj^T + residual
//
//   residual = hidden
//   hidden   = RMSNorm(hidden, post_attn_layernorm)
//   gate     = hidden @ gate_proj^T            [T, intermediate_size]
//   up       = hidden @ up_proj^T              [T, intermediate_size]
//   hidden   = SwiGLU(gate, up) @ down_proj^T + residual

#include "model/qwen3.h"
#include "model/kv_cache.cuh"
#include "model/weights.h"
#include "model/config.h"
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

// =============================================================================
// Internal model struct (opaque to callers via include/model/qwen3.h)
// =============================================================================

struct Qwen3Model {
    Qwen3Config    config;
    Qwen3Weights   weights;
    PagedKVCache   kv_cache;
    cublasHandle_t cublas;

    // RoPE tables: [max_position_embeddings, head_dim/2]  (FP32, GPU)
    float* cos_table;
    float* sin_table;

    // Scratch buffers — allocated once for max_T = max_batch * max_seq
    half*    d_hidden;    // [max_T, hidden_size]
    half*    d_residual;  // [max_T, hidden_size]  (also gather scratch post-layers)
    half*    d_Q;         // [max_T, q_dim]
    half*    d_K;         // [max_T, kv_dim]
    half*    d_V;         // [max_T, kv_dim]
    half*    d_attn_out;  // [max_T, q_dim]
    half*    d_gate;      // [max_T, intermediate_size]
    half*    d_up;        // [max_T, intermediate_size]
    half*    d_mlp_mid;   // [max_T, intermediate_size]
    half*    d_logits;    // [max_batch, vocab_size]
    int*     d_pos_ids;   // [max_T]
    int64_t* d_slot_map;  // [max_T]

    // Host-side KV cache mirrors (updated each forward pass on CPU, then H2D copied)
    int*     h_block_tables;   // [max_batch, max_blocks_per_seq]
    int*     h_seq_lens;       // [max_batch]
    int64_t* h_slot_map;       // [max_T]
    int*     h_pos_ids;        // [max_T]

    int max_batch;
    int max_seq;
};

// =============================================================================
// Helpers
// =============================================================================

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t e = (call);                                            \
        if (e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                    cudaGetErrorString(e), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t s = (call);                                         \
        if (s != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n",                 \
                    (int)s, __FILE__, __LINE__);                           \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

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

// =============================================================================
// Layer forward
// =============================================================================

static void qwen3_layer_forward(
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

    // ── Attention block ──────────────────────────────────────────────────────

    // 1. Save pre-norm hidden as residual
    CUDA_CHECK(cudaMemcpyAsync(m->d_residual, m->d_hidden,
                               (long)T * c.hidden_size * sizeof(half),
                               cudaMemcpyDeviceToDevice, stream));

    // 2. Input LayerNorm  (x=d_residual, out=d_hidden — separate buffers)
    launch_rmsnorm(m->d_hidden, m->d_residual,
                   L.input_layernorm, T, c.hidden_size, c.rms_norm_eps, stream);

    // 3. QKV projections
    linear_half(m->cublas, m->d_hidden, L.q_proj, m->d_Q, T, c.q_dim,  c.hidden_size);
    linear_half(m->cublas, m->d_hidden, L.k_proj, m->d_K, T, c.kv_dim, c.hidden_size);
    linear_half(m->cublas, m->d_hidden, L.v_proj, m->d_V, T, c.kv_dim, c.hidden_size);

    // 4. QK-Norm per head — BEFORE RoPE (critical Qwen3 ordering)
    //    [T, H, D] reinterpreted as [T*H, D] for the row-wise norm.
    //    In-place (out==x) is safe: kernel reads full row before writing back.
    launch_rmsnorm(m->d_Q, m->d_Q, L.q_norm,
                   T * c.num_attention_heads, c.head_dim, c.rms_norm_eps, stream);
    launch_rmsnorm(m->d_K, m->d_K, L.k_norm,
                   T * c.num_key_value_heads, c.head_dim, c.rms_norm_eps, stream);

    // 5. RoPE in-place on Q and K
    launch_rope(m->d_Q, m->d_K,
                m->cos_table, m->sin_table, m->d_pos_ids,
                T, c.num_attention_heads, c.num_key_value_heads, c.head_dim,
                stream);

    // 6. Append K/V to paged KV cache
    launch_reshape_and_cache_half(
        m->d_K, m->d_V,
        m->kv_cache.k_pool, m->kv_cache.v_pool,
        m->d_slot_map,
        T, layer_idx,
        m->kv_cache.total_blocks, c.num_key_value_heads, c.head_dim,
        stream);

    // 7. Attention
    if (is_prefill) {
        // FA2 prefill — uses live K, V just computed above.
        // [T, H, D] == [B, S, H, D] (same contiguous layout) ✓
        launch_flash_attention_prefill(
            m->d_Q, m->d_K, m->d_V, m->d_attn_out,
            B, S, c.num_attention_heads, c.num_key_value_heads, c.head_dim,
            stream);
    } else {
        // Paged decode — reads this layer's K/V slice from the cache.
        // Pool layout: [num_layers, total_blocks, H_kv, KV_BLOCK_SIZE, D]
        // The decode kernel's block_base formula assumes the pointer is at the
        // start of the current layer's slice, so compute that offset here.
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

    // 8. O projection + residual
    linear_half(m->cublas, m->d_attn_out, L.o_proj,
                m->d_hidden, T, c.hidden_size, c.q_dim);
    add_residual(m->d_hidden, m->d_residual, T * c.hidden_size, stream);

    // ── MLP block ────────────────────────────────────────────────────────────

    // 9. Save post-attention hidden as residual
    CUDA_CHECK(cudaMemcpyAsync(m->d_residual, m->d_hidden,
                               (long)T * c.hidden_size * sizeof(half),
                               cudaMemcpyDeviceToDevice, stream));

    // 10. Post-attention LayerNorm
    launch_rmsnorm(m->d_hidden, m->d_residual,
                   L.post_attn_layernorm, T, c.hidden_size, c.rms_norm_eps, stream);

    // 11. Gate + Up projections
    linear_half(m->cublas, m->d_hidden, L.gate_proj,
                m->d_gate, T, c.intermediate_size, c.hidden_size);
    linear_half(m->cublas, m->d_hidden, L.up_proj,
                m->d_up,   T, c.intermediate_size, c.hidden_size);

    // 12. SwiGLU: mlp_mid[i] = silu(gate[i]) * up[i]
    launch_swiglu(m->d_mlp_mid, m->d_gate, m->d_up,
                  T * c.intermediate_size, stream);

    // 13. Down projection + residual
    linear_half(m->cublas, m->d_mlp_mid, L.down_proj,
                m->d_hidden, T, c.hidden_size, c.intermediate_size);
    add_residual(m->d_hidden, m->d_residual, T * c.hidden_size, stream);
}

// =============================================================================
// Final norm + lm_head → logits
// =============================================================================
// Gathers the last S-th token of each sequence, applies final_norm, then
// projects to vocab via lm_head.  Result in m->d_logits [B, vocab_size].

static void compute_logits(Qwen3Model* m, int B, int S, cudaStream_t stream) {
    const Qwen3Config& c = m->config;

    // Gather last token of each sequence into d_residual [B, hidden_size].
    // d_residual is safe to reuse here — layer loop has finished.
    gather_last_tokens(m->d_residual, m->d_hidden, B, S, c.hidden_size, stream);

    // Final RMSNorm in-place on [B, hidden_size]
    launch_rmsnorm(m->d_residual, m->d_residual,
                   m->weights.final_norm, B, c.hidden_size, c.rms_norm_eps, stream);

    // LM head: [B, vocab_size] = [B, hidden_size] @ lm_head^T
    linear_half(m->cublas, m->d_residual, m->weights.lm_head,
                m->d_logits, B, c.vocab_size, c.hidden_size);
}

// =============================================================================
// qwen3_load
// =============================================================================

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

// =============================================================================
// qwen3_free / qwen3_reset
// =============================================================================

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
    delete[] m->h_block_tables;
    delete[] m->h_seq_lens;
    delete[] m->h_slot_map;
    delete[] m->h_pos_ids;
    delete m;
}

void qwen3_reset(Qwen3Model* m) {
    int mbps = m->kv_cache.max_blocks_per_seq;

    // Return all blocks to the free stack
    m->kv_cache.num_free = m->kv_cache.total_blocks;
    for (int i = 0; i < m->kv_cache.total_blocks; i++)
        m->kv_cache.free_stack[i] = i;

    memset(m->h_block_tables, -1, (long)m->max_batch * mbps * sizeof(int));
    memset(m->h_seq_lens,      0, m->max_batch * sizeof(int));

    CUDA_CHECK(cudaMemcpy(m->kv_cache.block_tables, m->h_block_tables,
                          (long)m->max_batch * mbps * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m->kv_cache.seq_lens, m->h_seq_lens,
                          m->max_batch * sizeof(int), cudaMemcpyHostToDevice));
}

// =============================================================================
// qwen3_prefill
// =============================================================================

half* qwen3_prefill(Qwen3Model* m, const int* tokens_gpu, int B, int S,
                    cudaStream_t stream) {
    const Qwen3Config& c = m->config;
    const int T = B * S;

    CUBLAS_CHECK(cublasSetStream(m->cublas, stream));

    // Build position_ids and slot_mapping on host.
    // Assumes new sequences starting at position 0.
    for (int b = 0; b < B; b++) {
        for (int s = 0; s < S; s++) {
            int idx = b * S + s;
            m->h_pos_ids[idx]  = s;
            m->h_slot_map[idx] = paged_kv_cache_append_slot(m->kv_cache, m->h_block_tables, m->h_seq_lens, b);
        }
    }

    int mbps = m->kv_cache.max_blocks_per_seq;
    CUDA_CHECK(cudaMemcpyAsync(m->d_pos_ids,  m->h_pos_ids,
                               T * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m->d_slot_map, m->h_slot_map,
                               T * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m->kv_cache.block_tables, m->h_block_tables,
                               (long)m->max_batch * mbps * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m->kv_cache.seq_lens, m->h_seq_lens,
                               m->max_batch * sizeof(int), cudaMemcpyHostToDevice, stream));

    // Token embedding → d_hidden [T, hidden_size]
    launch_embedding(m->d_hidden, m->weights.embed_tokens,
                     tokens_gpu, T, c.vocab_size, c.hidden_size, stream);

    for (int l = 0; l < c.num_hidden_layers; l++)
        qwen3_layer_forward(m, l, T, B, S, /*is_prefill=*/true, stream);

    // Final norm + lm_head → d_logits [B, vocab_size]
    compute_logits(m, B, S, stream);

    return m->d_logits;
}

// =============================================================================
// qwen3_decode
// =============================================================================

half* qwen3_decode(Qwen3Model* m, const int* token_gpu, int B,
                   cudaStream_t stream) {
    const Qwen3Config& c = m->config;

    CUBLAS_CHECK(cublasSetStream(m->cublas, stream));

    // Build slot_mapping and position_ids.
    // paged_kv_cache_append_slot increments h_seq_lens[b], so position = new_len - 1.
    for (int b = 0; b < B; b++) {
        m->h_slot_map[b] = paged_kv_cache_append_slot(
            m->kv_cache, m->h_block_tables, m->h_seq_lens, b);
        m->h_pos_ids[b]  = m->h_seq_lens[b] - 1;
    }

    int mbps = m->kv_cache.max_blocks_per_seq;
    CUDA_CHECK(cudaMemcpyAsync(m->d_pos_ids,  m->h_pos_ids,
                               B * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m->d_slot_map, m->h_slot_map,
                               B * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m->kv_cache.block_tables, m->h_block_tables,
                               (long)m->max_batch * mbps * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m->kv_cache.seq_lens, m->h_seq_lens,
                               m->max_batch * sizeof(int), cudaMemcpyHostToDevice, stream));

    // Token embedding → d_hidden [B, hidden_size]
    launch_embedding(m->d_hidden, m->weights.embed_tokens,
                     token_gpu, B, c.vocab_size, c.hidden_size, stream);

    for (int l = 0; l < c.num_hidden_layers; l++)
        qwen3_layer_forward(m, l, B, B, /*S=*/1, /*is_prefill=*/false, stream);

    // S=1 → gather_last_tokens is a trivial copy of d_hidden → d_residual
    compute_logits(m, B, /*S=*/1, stream);

    return m->d_logits;
}
