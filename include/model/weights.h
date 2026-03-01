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
    half*  embed_tokens;         // [vocab_size, hidden_size]  token embedding table
    half*  lm_head;              // [vocab_size, hidden_size]  alias of embed_tokens (tie_word_embeddings=true)
    float* final_norm;           // [hidden_size]              FP32
    std::vector<Qwen3LayerWeights> layers;

    // Total GPU memory allocated (bytes)
    size_t total_bytes = 0;
};

// Load weights from safetensors file(s) in model_dir onto GPU
Qwen3Weights load_weights(const std::string& model_dir, const Qwen3Config& cfg);

// Free all GPU weight memory
void free_weights(Qwen3Weights& w);