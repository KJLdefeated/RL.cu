#pragma once
#include <string>
#include <cstdint>

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

    int n_rep;
    int q_dim;
    int kv_dim;

    void compute_derived() {
        n_rep  = num_attention_heads / num_key_value_heads;
        q_dim  = num_attention_heads * head_dim;
        kv_dim = num_key_value_heads * head_dim;
    }
};

Qwen3Config load_config(const std::string& model_dir);

struct Config {
    std::string model; // Model dir
    std::string tokenizer_path; // Tokenizer path
    int max_num_batched_tokens = 16384;
    int max_num_seqs = 512;
    int max_model_len = 4096;
    float gpu_memory_utilization = 0.9;
    bool enforce_eager = false;
    Qwen3Config model_config;
    int64_t eos = -1;
    int kv_block_size = 16;  // must match KV_BLOCK_SIZE in kv_cache.cuh
    int num_kv_blocks = -1;
    Config() = default;

    Config(std::string model_dir) {
        model = model_dir;
        tokenizer_path = model_dir + "/tokenizer.json";
        model_config = load_config(model_dir);
        eos = model_config.vocab_size - 1; // assuming EOS token is the last token in vocab
        // num_kv_blocks is set by compute_kv_budget() in ModelRunner based on actual
        // free GPU memory.  Leave at -1 so block_manager is not pre-constructed with
        // a wrong pool size (the old formula gave blocks-per-seq, not total pool size).
    }
};