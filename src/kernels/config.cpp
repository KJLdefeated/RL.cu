#include "model/config.h"
#include "third_party/json.hpp"
#include <fstream>
#include <stdexcept>
#include <cstdio>

using json = nlohmann::json;

Qwen3Config load_config(const std::string& model_dir) {
    Qwen3Config cfg;
    std::string config_path = model_dir + "/config.json";
    std::ifstream f(config_path);
    if (!f) {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }

    json j = json::parse(f);
    cfg.vocab_size          = j.value("vocab_size",          cfg.vocab_size);
    cfg.hidden_size         = j.value("hidden_size",         cfg.hidden_size);
    cfg.intermediate_size   = j.value("intermediate_size",   cfg.intermediate_size);
    cfg.num_hidden_layers   = j.value("num_hidden_layers",   cfg.num_hidden_layers);
    cfg.num_attention_heads = j.value("num_attention_heads", cfg.num_attention_heads);
    cfg.num_key_value_heads = j.value("num_key_value_heads", cfg.num_key_value_heads);
    cfg.head_dim            = j.value("head_dim",            cfg.head_dim);
    cfg.max_position_embeddings = j.value("max_position_embeddings", cfg.max_position_embeddings);
    cfg.rms_norm_eps        = j.value("rms_norm_eps",        cfg.rms_norm_eps);
    cfg.tie_word_embeddings = j.value("tie_word_embeddings", cfg.tie_word_embeddings);
    cfg.attention_bias      = j.value("attention_bias",      cfg.attention_bias);

    if (j.contains("rope_theta"))
        cfg.rope_theta = j["rope_theta"].get<float>();
    if (j.contains("hidden_act"))
        cfg.hidden_act = j["hidden_act"].get<std::string>();
    if (j.contains("torch_dtype"))
        cfg.torch_dtype = j["torch_dtype"].get<std::string>();

    cfg.compute_derived();

    printf("[CONFIG] %s\n", config_path.c_str());
    printf("  vocab_size=%d  hidden_size=%d  layers=%d\n",
           cfg.vocab_size, cfg.hidden_size, cfg.num_hidden_layers);
    printf("  Q_heads=%d  KV_heads=%d  head_dim=%d  n_rep=%d\n",
           cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.n_rep);
    printf("  intermediate=%d  rope_theta=%.0f  tie_embed=%d\n",
           cfg.intermediate_size, cfg.rope_theta, cfg.tie_word_embeddings);

    return cfg;
}
