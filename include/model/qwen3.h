#pragma once
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
#include <unordered_map>
#include <vector>
#include <string>
#include "model/sampling_parmas.h"

// cuda graph handler
struct CUDAGraphState {
    bool captured = false;
    std::unordered_map<int, cudaGraphExec_t> graphs;
    cudaGraphExec_t current_graph = nullptr;
    int* g_token_ids   = nullptr;
    int* g_pos_ids     = nullptr;
    int64_t* g_slot_map   = nullptr;
    int* g_block_tables = nullptr;
    int* g_seq_lens    = nullptr;
    void* g_cublas_ws  = nullptr;  // kept alive for captured graph lifetime
    std::vector<int> buckets;
};

// Qwen3 model struct
struct Qwen3Model {
    Qwen3Config    config;
    Qwen3Weights   weights;
    PagedKVCache   kv_cache;
    cublasHandle_t cublas;
    CUDAGraphState  graph_state;

    // RoPE tables: [max_position_embeddings, head_dim/2]  (FP32, GPU)
    float* cos_table = nullptr;
    float* sin_table = nullptr;

    // Scratch buffers — allocated once for max_T = max_batched_tokens
    half*    d_hidden    = nullptr;  // [max_T, hidden_size]
    half*    d_residual  = nullptr;  // [max_T, hidden_size]
    half*    d_Q         = nullptr;  // [max_T, q_dim]
    half*    d_K         = nullptr;  // [max_T, kv_dim]
    half*    d_V         = nullptr;  // [max_T, kv_dim]
    half*    d_attn_out  = nullptr;  // [max_T, q_dim]
    half*    d_gate      = nullptr;  // [max_T, intermediate_size]
    half*    d_up        = nullptr;  // [max_T, intermediate_size]
    half*    d_mlp_mid   = nullptr;  // [max_T, intermediate_size]
    half*    d_logits    = nullptr;  // [max_batch, vocab_size]
    int*     d_pos_ids   = nullptr;  // [max_T]
    int*     d_tokens    = nullptr;  // [max_T]
    int64_t* d_slot_map  = nullptr;  // [max_T]

    // Host-side mirrors — all pinned (cudaMallocHost) for fast H2D cudaMemcpyAsync
    int*     h_block_tables        = nullptr;  // [max_batch × max_blocks_per_seq]
    int*     h_seq_lens            = nullptr;  // [max_batch]
    int64_t* h_slot_map            = nullptr;  // [max_T]
    int*     h_pos_ids             = nullptr;  // [max_T]
    int*     h_tokens              = nullptr;  // [max_T]  prefill+decode input tokens
    int*     h_block_table_compact = nullptr;  // [max_batch × max_blocks_per_seq] decode compact
    int*     h_seq_lens_compact    = nullptr;  // [max_batch] decode compact

    int max_batch      = 0;
    int max_seq        = 0;
    int max_T          = 0;  // current scratch buffer capacity in tokens
    int infer_max_T    = 0;  // inference scratch size (restored on wakeup)
    int total_kv_blocks = 0;
};

Qwen3Model* qwen3_load(const std::string& model_dir, int max_batch, int max_seq);

void qwen3_init(Qwen3Model* m, int max_batch, int max_seq, int max_batched_tokens, int num_kv_blocks);

void qwen3_free(Qwen3Model* model);

// Ensure scratch buffers (d_hidden, d_Q, etc.) can hold at least T tokens.
// Re-allocates if T > current max_T. Called automatically by qwen3_forward.
void qwen3_ensure_scratch(Qwen3Model* m, int T);

void qwen3_reset(Qwen3Model* model);

// KV cache lifecycle for GRPO: sleep frees KV pools during training,
// wakeup re-allocates them before generation.
void qwen3_sleep(Qwen3Model* m);
void qwen3_wakeup(Qwen3Model* m, int total_blocks);

half* qwen3_prefill(Qwen3Model* m, const std::vector<Sequence*>& batch, cudaStream_t stream = 0);

half* qwen3_decode(Qwen3Model* model, const std::vector<Sequence*>& batch, cudaStream_t stream = 0);

// Graph-accelerated decode: copies inputs into graph's fixed device buffers,
// pads to bucket size, then fires a single cudaGraphLaunch instead of 400+ kernels.
// Falls back to qwen3_decode() if B > max captured bucket.
half* qwen3_decode_graph(Qwen3Model* model, const std::vector<Sequence*>& batch, cudaStream_t stream = 0);

void qwen3_layer_forward(Qwen3Model* m, int layer_idx, int T, int B, int S,
                         bool is_prefill, cudaStream_t stream);

void compute_logits(Qwen3Model* m, int B, int S, cudaStream_t stream);

// Free KV blocks for a finished sequence at batch slot b, reset seq state.
void qwen3_free_seq_slot(Qwen3Model* m, int b);

// =============================================================================
// Training forward pass
// =============================================================================

// Saved activations for backward pass with GRADIENT CHECKPOINTING.
//
// Only the layer input residual and flash-attention LSE are saved per layer.
// All other activations (Q, K, V, O, gate, up, ...) are recomputed from the
// residual at the start of each layer's backward pass.  This trades ~30% extra
// compute for ~10× less activation memory (70 GB → 6.6 GB at T=72000).
//
// Memory layout:
//   checkpoint_pool (half): [L × T×H]  layer_input  +  [T×H] final_hidden
//   lse_pool        (float): [L × T×H_q]  flash-attention LSE
//   recomp_pool     (half): single-layer recomputation scratch (Q_raw, K_raw,
//                           Q, K, V, O, post_attn, gate, up)
struct Qwen3TrainState {
    int B = 0, S = 0, T = 0;   // batch, seq_len, total tokens (B*S)
    int num_layers = 0;

    // --- Bulk GPU allocations ---
    half*  checkpoint_pool = nullptr;   // per-layer residuals + final_hidden
    float* lse_pool        = nullptr;   // per-layer LSE (FP32)
    half*  recomp_pool     = nullptr;   // single-layer recomputation scratch

    // Per-layer checkpointed activations (pointers into checkpoint_pool)
    half** layer_input     = nullptr;   // [L] → [T, hidden]   pre-layer residual
    float** layer_lse      = nullptr;   // [L] → [T, H_q]      FA2 logsumexp

    // Final stage (pointer into checkpoint_pool)
    half* final_hidden     = nullptr;   // [T, hidden] — after all layers

    // Single-layer recomputation buffers (pointers into recomp_pool)
    // Filled by recomputing forward for one layer during backward.
    half* recomp_Q_raw     = nullptr;   // [T, q_dim]
    half* recomp_K_raw     = nullptr;   // [T, kv_dim]
    half* recomp_Q         = nullptr;   // [T, q_dim]      after QK-norm + RoPE
    half* recomp_K         = nullptr;   // [T, kv_dim]     after QK-norm + RoPE
    half* recomp_V         = nullptr;   // [T, kv_dim]
    half* recomp_O         = nullptr;   // [T, q_dim]      attention output
    half* recomp_post_attn = nullptr;   // [T, hidden]     post-attention residual
    half* recomp_gate      = nullptr;   // [T, intermediate]
    half* recomp_up        = nullptr;   // [T, intermediate]

    // Logits scratch for lm_head (chunked to avoid [T, vocab] allocation)
    // Size: [TRAIN_LOGITS_CHUNK, vocab_size]
    half* d_logits_train   = nullptr;

    // Device copies of IDs
    int* d_token_ids       = nullptr;   // [T] input token ids
    int* d_target_ids      = nullptr;   // [T] target token ids (for log_prob gathering)
    int* d_pos_ids         = nullptr;   // [T] position ids
};

Qwen3TrainState* qwen3_train_state_alloc(const Qwen3Config& c, int B, int S);
void qwen3_train_state_free(Qwen3TrainState* state);

// Training forward pass.
// Runs full sequence through all layers (no KV cache), computes log_probs.
//   h_token_ids:  [B*S] input tokens (host)
//   h_target_ids: [B*S] target tokens (host), typically token_ids shifted left by 1
//   d_log_probs:  [B*S] output log-probabilities (device, FP32, pre-allocated)
// Saves all activations needed for backward into state.
void qwen3_forward(
    Qwen3Model* m,
    Qwen3TrainState* state,
    const int* h_token_ids,
    const int* h_target_ids,
    float* d_log_probs,
    int B, int S,
    cudaStream_t stream = 0
);

// =============================================================================
// Training backward pass
// =============================================================================

// Weight gradients for backward pass.
// All gradient buffers are bulk-allocated; per-layer pointers index into pools.
struct Qwen3Gradients {
    int num_layers = 0;

    // --- Bulk GPU allocations ---
    half*  half_pool  = nullptr;   // all FP16 weight gradients
    float* float_pool = nullptr;   // all FP32 weight gradients (norms) + D_buf
    size_t half_pool_bytes  = 0;
    size_t float_pool_bytes = 0;

    // Per-layer weight gradients (pointers into pools)
    half** dW_q_proj          = nullptr;  // [L] → [q_dim, hidden]
    half** dW_k_proj          = nullptr;  // [L] → [kv_dim, hidden]
    half** dW_v_proj          = nullptr;  // [L] → [kv_dim, hidden]
    half** dW_o_proj          = nullptr;  // [L] → [hidden, q_dim]
    half** dW_gate_proj       = nullptr;  // [L] → [inter, hidden]
    half** dW_up_proj         = nullptr;  // [L] → [inter, hidden]
    half** dW_down_proj       = nullptr;  // [L] → [hidden, inter]
    float** dW_input_norm     = nullptr;  // [L] → [hidden]
    float** dW_q_norm         = nullptr;  // [L] → [head_dim]
    float** dW_k_norm         = nullptr;  // [L] → [head_dim]
    float** dW_post_attn_norm = nullptr;  // [L] → [hidden]

    // Global weight gradients
    half*  dW_embed      = nullptr;  // [vocab, hidden] (tied = lm_head grad)
    float* dW_final_norm = nullptr;  // [hidden]

    // Backward workspace
    float* D_buf         = nullptr;  // [max_T * H_q] for flash attention backward
};

Qwen3Gradients* qwen3_gradients_alloc(const Qwen3Config& c, int max_T);
void qwen3_gradients_free(Qwen3Gradients* g);
void qwen3_gradients_zero(Qwen3Gradients* g, cudaStream_t stream = 0);

// Training backward pass.
// Computes weight gradients given upstream d_log_probs gradient.
// Gradients are accumulated into grads (caller must zero before first call).
// Uses model scratch buffers (d_hidden, d_Q, etc.) as workspace.
void qwen3_backward(
    Qwen3Model* m,
    Qwen3TrainState* state,
    Qwen3Gradients* grads,
    const float* d_log_probs,   // [B*S] upstream gradient (device, FP32)
    cudaStream_t stream = 0
);

