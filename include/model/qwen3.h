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

    // Scratch buffers — allocated once for max_T = max_batch * max_seq
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

    // Host-side mirrors (updated each forward pass on CPU, then H2D copied)
    int*     h_block_tables = nullptr;  // [max_batch, max_blocks_per_seq]
    int*     h_seq_lens     = nullptr;  // [max_batch]
    int64_t* h_slot_map     = nullptr;  // [max_T]
    int*     h_pos_ids      = nullptr;  // [max_T]

    int max_batch      = 0;
    int max_seq        = 0;
    int total_kv_blocks = 0;
};

Qwen3Model* qwen3_load(const std::string& model_dir, int max_batch, int max_seq);

void qwen3_init(Qwen3Model* m, int max_batch, int max_seq, int max_batched_tokens, int num_kv_blocks);

void qwen3_free(Qwen3Model* model);

void qwen3_reset(Qwen3Model* model);

half* qwen3_prefill(Qwen3Model* m, const std::vector<Sequence*>& batch, cudaStream_t stream = 0);

half* qwen3_decode(Qwen3Model* model, const std::vector<Sequence*>& batch, cudaStream_t stream = 0);

void qwen3_layer_forward(Qwen3Model* m, int layer_idx, int T, int B, int S,
                         bool is_prefill, cudaStream_t stream);

void compute_logits(Qwen3Model* m, int B, int S, cudaStream_t stream);

// Free KV blocks for a finished sequence at batch slot b, reset seq state.
void qwen3_free_seq_slot(Qwen3Model* m, int b);
