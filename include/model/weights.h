#pragma once
#include "config.h"
#include <cuda_fp16.h>
#include <string>
#include <vector>

// Qwen3Weights: All model weight pointers on GPU
//
// Safetensors tensor name patterns for Qwen3:
//   model.embed_tokens.weight                         [vocab_size, hidden_size]
//   lm_head.weight                                    tied to embed_tokens (tie_word_embeddings=true)
//   model.layers.{i}.input_layernorm.weight           [hidden_size]
//   model.layers.{i}.self_attn.q_proj.weight          [q_dim, hidden_size]
//   model.layers.{i}.self_attn.k_proj.weight          [kv_dim, hidden_size]
//   model.layers.{i}.self_attn.v_proj.weight          [kv_dim, hidden_size]
//   model.layers.{i}.self_attn.o_proj.weight          [hidden_size, q_dim]
//   model.layers.{i}.self_attn.q_norm.weight          [head_dim]
//   model.layers.{i}.self_attn.k_norm.weight          [head_dim]
//   model.layers.{i}.post_attention_layernorm.weight   [hidden_size]
//   model.layers.{i}.mlp.gate_proj.weight             [intermediate_size, hidden_size]
//   model.layers.{i}.mlp.up_proj.weight               [intermediate_size, hidden_size]
//   model.layers.{i}.mlp.down_proj.weight             [hidden_size, intermediate_size]
//   model.norm.weight                                 [hidden_size]

struct Qwen3LayerWeights {
    float* input_layernorm;      // [hidden_size]              FP32 — small, precision-critical
    half*  q_proj;               // [q_dim, hidden_size]
    half*  k_proj;               // [kv_dim, hidden_size]
    half*  v_proj;               // [kv_dim, hidden_size]
    half*  o_proj;               // [hidden_size, q_dim]
    float* q_norm;               // [head_dim]                 FP32 — QK-Norm weights
    float* k_norm;               // [head_dim]                 FP32 — QK-Norm weights
    float* post_attn_layernorm;  // [hidden_size]              FP32
    half*  gate_proj;            // [intermediate_size, hidden_size]
    half*  up_proj;              // [intermediate_size, hidden_size]
    half*  down_proj;            // [hidden_size, intermediate_size]
};

struct Qwen3Weights {
    // --- Flat GPU allocations (one cudaMalloc each) ---
    // Layout must match Qwen3Gradients pool layout for flat optimizer step.
    // FP16: per-layer [q, k, v, o, gate, up, down], then embed
    // FP32: per-layer [input_norm, q_norm, k_norm, post_attn_norm], then final_norm
    half*  fp16_pool = nullptr;
    float* fp32_pool = nullptr;
    size_t fp16_pool_elems = 0;   // total FP16 parameter count
    size_t fp32_pool_elems = 0;   // total FP32 parameter count

    // --- Convenience pointers (into pools, NOT independent allocations) ---
    half*  embed_tokens = nullptr; // [vocab_size, hidden_size]
    half*  lm_head      = nullptr; // alias of embed_tokens (tied)
    float* final_norm   = nullptr; // [hidden_size]
    std::vector<Qwen3LayerWeights> layers;

    // Total GPU memory allocated (bytes)
    size_t total_bytes = 0;
};

// Load weights from safetensors file(s) in model_dir onto GPU
Qwen3Weights load_weights(const std::string& model_dir, const Qwen3Config& cfg);

// Free all GPU weight memory
void free_weights(Qwen3Weights& w);