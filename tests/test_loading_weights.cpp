#include "model/weights.h"
#include "model/config.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <filesystem>

// ── helpers ──────────────────────────────────────────────────────────────────

static int g_fails = 0;

#define PASS(name)         printf("[PASS] %s\n", (name))
#define FAIL(name, ...)    do { \
    printf("[FAIL] %s: ", (name)); printf(__VA_ARGS__); printf("\n"); g_fails++; \
} while(0)
#define EXPECT_EQ(name, got, expected) do { \
    if ((got) == (expected)) PASS(name); \
    else FAIL(name, "expected %d got %d", (int)(expected), (int)(got)); \
} while(0)
#define EXPECT_NONNULL(name, ptr) do { \
    if ((ptr) != nullptr) PASS(name); \
    else FAIL(name, "null pointer"); \
} while(0)

// ── tests ────────────────────────────────────────────────────────────────────

static void test_config(const Qwen3Config& cfg) {
    printf("\n--- config ---\n");
    EXPECT_EQ("vocab_size",          cfg.vocab_size,          151936);
    EXPECT_EQ("hidden_size",         cfg.hidden_size,         1024);
    EXPECT_EQ("num_hidden_layers",   cfg.num_hidden_layers,   28);
    EXPECT_EQ("num_attention_heads", cfg.num_attention_heads, 16);
    EXPECT_EQ("num_key_value_heads", cfg.num_key_value_heads, 8);
    EXPECT_EQ("head_dim",            cfg.head_dim,            128);
    EXPECT_EQ("intermediate_size",   cfg.intermediate_size,   3072);
    // Derived
    EXPECT_EQ("q_dim  (derived)",    cfg.q_dim,               2048);   // 16 * 128
    EXPECT_EQ("kv_dim (derived)",    cfg.kv_dim,              1024);   // 8  * 128
    EXPECT_EQ("n_rep  (derived)",    cfg.n_rep,               2);      // 16 / 8
}

static void test_weight_pointers(const Qwen3Weights& w, int num_layers) {
    printf("\n--- weight pointers ---\n");
    EXPECT_NONNULL("embed_tokens",  w.embed_tokens);
    EXPECT_NONNULL("final_norm",    w.final_norm);

    int null_count = 0;
    for (int i = 0; i < num_layers; i++) {
        const auto& L = w.layers[i];
        if (!L.input_layernorm)     null_count++;
        if (!L.q_proj)              null_count++;
        if (!L.k_proj)              null_count++;
        if (!L.v_proj)              null_count++;
        if (!L.o_proj)              null_count++;
        if (!L.q_norm)              null_count++;
        if (!L.k_norm)              null_count++;
        if (!L.post_attn_layernorm) null_count++;
        if (!L.gate_proj)           null_count++;
        if (!L.up_proj)             null_count++;
        if (!L.down_proj)           null_count++;
    }
    if (null_count == 0) PASS("all layer weight pointers non-null");
    else FAIL("layer weights", "%d null pointer(s) across %d layers", null_count, num_layers);
}

static void test_total_bytes(const Qwen3Weights& w) {
    printf("\n--- GPU memory ---\n");
    printf("  total_bytes = %.2f MB\n", w.total_bytes / (1024.0 * 1024.0));

    // Qwen3-0.6B in FP16: embed(151936×1024) + 28 layers × ~30 MB + final_norm ≈ 1.1 GB
    const size_t min_expected = 1000ULL * 1024 * 1024;  // 1000 MB
    const size_t max_expected = 2000ULL * 1024 * 1024;  // 2000 MB
    if (w.total_bytes >= min_expected && w.total_bytes <= max_expected)
        PASS("total_bytes in expected range [1000, 2000] MB");
    else
        FAIL("total_bytes", "%.0f MB not in expected range", w.total_bytes / (1024.0 * 1024.0));
}

static void test_data_nonzero(const Qwen3Weights& w, const Qwen3Config& cfg) {
    printf("\n--- data sanity (GPU->CPU peek) ---\n");

    // embed_tokens: first 8 elements
    {
        half buf[8];
        cudaMemcpy(buf, w.embed_tokens, sizeof(buf), cudaMemcpyDeviceToHost);
        bool any_nonzero = false;
        for (int i = 0; i < 8; i++) if (__half2float(buf[i]) != 0.f) any_nonzero = true;
        if (any_nonzero) PASS("embed_tokens[0:8] non-zero");
        else FAIL("embed_tokens[0:8]", "all zeros — likely bad load");
    }

    // Layer 0 q_proj: first 8 elements
    if (w.layers[0].q_proj) {
        half buf[8];
        cudaMemcpy(buf, w.layers[0].q_proj, sizeof(buf), cudaMemcpyDeviceToHost);
        bool any_nonzero = false;
        for (int i = 0; i < 8; i++) if (__half2float(buf[i]) != 0.f) any_nonzero = true;
        if (any_nonzero) PASS("layer0.q_proj[0:8] non-zero");
        else FAIL("layer0.q_proj[0:8]", "all zeros — likely bad load");
    }

    // Layer 0 q_norm: now float* — check finite and non-zero (trained values can be anything)
    if (w.layers[0].q_norm) {
        float buf[8];
        cudaMemcpy(buf, w.layers[0].q_norm, sizeof(buf), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < 8; i++) if (!std::isfinite(buf[i])) { ok = false; break; }
        if (ok) {
            printf("  layer0.q_norm[0:4] = %.4f %.4f %.4f %.4f\n",
                   buf[0], buf[1], buf[2], buf[3]);
            PASS("layer0.q_norm[0:8] finite FP32");
        } else FAIL("layer0.q_norm[0:8]", "contains NaN/Inf");
    }

    // Final norm: float* — same check
    {
        float buf[8];
        cudaMemcpy(buf, w.final_norm, sizeof(buf), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < 8; i++) if (!std::isfinite(buf[i])) { ok = false; break; }
        if (ok) {
            printf("  final_norm[0:4]    = %.4f %.4f %.4f %.4f\n",
                   buf[0], buf[1], buf[2], buf[3]);
            PASS("final_norm[0:8] finite FP32");
        } else FAIL("final_norm[0:8]", "contains NaN/Inf");
    }
}

static void test_free(Qwen3Weights& w) {
    printf("\n--- free_weights ---\n");
    free_weights(w);
    if (w.embed_tokens == nullptr) PASS("embed_tokens nulled after free");
    else FAIL("embed_tokens", "pointer not cleared");
    if (w.final_norm == nullptr)   PASS("final_norm nulled after free");
    else FAIL("final_norm", "pointer not cleared");
    if (w.layers.empty())          PASS("layers cleared after free");
    else FAIL("layers", "not empty after free");
    if (w.total_bytes == 0)        PASS("total_bytes zeroed after free");
    else FAIL("total_bytes", "not zeroed after free");
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    const char* model_dir = "model_weights/Qwen3-0.6B";

    if (!std::filesystem::exists(model_dir)) {
        printf("[SKIP] %s not found — download weights to run this test\n", model_dir);
        return 0;
    }

    // 1. Config
    Qwen3Config cfg = load_config(model_dir);
    test_config(cfg);

    // 2. Weights
    Qwen3Weights w = load_weights(model_dir, cfg);
    test_weight_pointers(w, cfg.num_hidden_layers);
    test_total_bytes(w);
    test_data_nonzero(w, cfg);
    test_free(w);

    printf("\n%s — %d failure(s)\n",
           g_fails == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED", g_fails);
    return g_fails > 0 ? 1 : 0;
}
