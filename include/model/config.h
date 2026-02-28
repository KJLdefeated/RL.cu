#pragma once
#include <string>
#include <cstdint>

// Qwen3Config: All fields from HuggingFace config.json
// Parsed at runtime so one binary supports 0.6B, 4B, 8B etc.
struct Qwen3Config {
    int vocab_size          = 151936;
    int hidden_size         = 1024;
    int intermediate_size   = 3072;
    int num_hidden_layers   = 28;
    int num_attention_heads = 16;   // Q heads
    int num_key_value_heads = 8;    // KV heads (GQA)
    int head_dim            = 128;
    int max_position_embeddings = 40960;
    float rope_theta        = 1000000.0f;
    float rms_norm_eps      = 1e-6f;
    bool tie_word_embeddings = true;
    bool attention_bias     = false;
    std::string hidden_act  = "silu";
    std::string torch_dtype = "bfloat16";

    // Derived (computed after parsing)
    int n_rep;              // num_attention_heads / num_key_value_heads
    int q_dim;              // num_attention_heads * head_dim
    int kv_dim;             // num_key_value_heads * head_dim

    void compute_derived() {
        n_rep  = num_attention_heads / num_key_value_heads;
        q_dim  = num_attention_heads * head_dim;
        kv_dim = num_key_value_heads * head_dim;
    }
};

// Parse config.json from model directory into Qwen3Config
Qwen3Config load_config(const std::string& model_dir);