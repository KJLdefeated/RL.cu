// test_qwen3_forward.cu
//
// Tests the training forward pass (qwen3_forward) which computes per-token
// log-probabilities and saves activations for backward.
//
// Validates:
//   1. Log-probs are finite and ≤ 0 (log of a probability)
//   2. Log-probs match manual computation from inference logits
//   3. Batch consistency
//   4. Chunked lm_head produces correct results when T > max_batch
//
// Requires real weights at model_weights/Qwen3-0.6B.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "model/qwen3.h"
#include "model/tokenizer.h"

static const char* MODEL_DIR = "model_weights/Qwen3-0.6B";

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t e = (call);                                            \
        if (e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                    cudaGetErrorString(e), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

static int g_fails = 0;
#define PASS(name)      printf("[PASS] %s\n", (name))
#define FAIL(name, ...) do { \
    printf("[FAIL] %s: ", (name)); printf(__VA_ARGS__); printf("\n"); g_fails++; \
} while (0)

// CPU log-softmax for a single row, returning log_prob of target
static float cpu_log_softmax_gather(const float* logits, int V, int target) {
    float max_val = -INFINITY;
    for (int i = 0; i < V; i++)
        if (logits[i] > max_val) max_val = logits[i];
    double sum = 0.0;
    for (int i = 0; i < V; i++)
        sum += exp((double)(logits[i] - max_val));
    return (float)((double)(logits[target] - max_val) - log(sum));
}

// =============================================================================
// Test 1: Basic forward — log-probs finite and ≤ 0
// =============================================================================
static void test_basic_forward(Qwen3Model* m) {
    const char* name = "Basic forward: finite log-probs";
    const Qwen3Config& c = m->config;

    // Simple token sequence: "The capital of France is"
    std::vector<int> tokens = {785, 6722, 315, 9625, 374};
    int B = 1, S = (int)tokens.size();
    int T = B * S;

    // Target = shifted tokens (predict next token)
    // Last target is arbitrary (won't contribute to loss in practice)
    std::vector<int> targets(T);
    for (int i = 0; i < T - 1; i++) targets[i] = tokens[i + 1];
    targets[T - 1] = 0;  // padding target for last position

    // Allocate state and log_probs
    auto* state = qwen3_train_state_alloc(c, B, S);
    float* d_log_probs;
    CUDA_CHECK(cudaMalloc(&d_log_probs, T * sizeof(float)));

    qwen3_forward(m, state, tokens.data(), targets.data(), d_log_probs, B, S);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fetch results
    std::vector<float> h_log_probs(T);
    CUDA_CHECK(cudaMemcpy(h_log_probs.data(), d_log_probs, T * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Check: all finite and ≤ 0
    bool ok = true;
    for (int t = 0; t < T; t++) {
        if (!std::isfinite(h_log_probs[t])) {
            FAIL(name, "log_prob[%d] = %f (not finite)", t, h_log_probs[t]);
            ok = false; break;
        }
        if (h_log_probs[t] > 1e-5f) {  // small epsilon for numerical noise
            FAIL(name, "log_prob[%d] = %f (> 0)", t, h_log_probs[t]);
            ok = false; break;
        }
    }
    if (ok) {
        printf("  log_probs:");
        for (int t = 0; t < T; t++) printf(" %.4f", h_log_probs[t]);
        printf("\n");
        PASS(name);
    }

    cudaFree(d_log_probs);
    qwen3_train_state_free(state);
}

// =============================================================================
// Test 2: Cross-validate against inference path
//
// Run qwen3_prefill (inference) to get logits for the last token, then
// manually compute log_softmax_gather and compare with qwen3_forward.
//
// NOTE: qwen3_prefill modifies KV cache state, so we reset between runs.
// We compare the LAST position's log_prob since prefill only produces logits
// for the last token.
// =============================================================================
static void test_cross_validate(Qwen3Model* m) {
    const char* name = "Cross-validate: forward vs inference logits";
    const Qwen3Config& c = m->config;

    std::vector<int> tokens = {785, 6722, 315, 9625, 374};
    int B = 1, S = (int)tokens.size();
    int T = B * S;

    // Pick a target for the last position
    int last_target = 12095;  // "Paris"
    std::vector<int> targets(T);
    for (int i = 0; i < T - 1; i++) targets[i] = tokens[i + 1];
    targets[T - 1] = last_target;

    // --- Training forward ---
    auto* state = qwen3_train_state_alloc(c, B, S);
    float* d_log_probs;
    CUDA_CHECK(cudaMalloc(&d_log_probs, T * sizeof(float)));

    qwen3_forward(m, state, tokens.data(), targets.data(), d_log_probs, B, S);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_log_probs(T);
    CUDA_CHECK(cudaMemcpy(h_log_probs.data(), d_log_probs, T * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float train_log_prob = h_log_probs[T - 1];

    // --- Inference forward ---
    qwen3_reset(m);
    std::vector<int64_t> tok64(tokens.begin(), tokens.end());
    SamplingParams sp;
    Sequence seq(0, tok64, sp);
    seq.batch_slot = 0;
    seq.num_cached_tokens = 0;
    std::vector<Sequence*> batch = {&seq};

    half* d_logits = qwen3_prefill(m, batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fetch last-token logits
    std::vector<half> h_logits_half(c.vocab_size);
    CUDA_CHECK(cudaMemcpy(h_logits_half.data(), d_logits,
                          c.vocab_size * sizeof(half), cudaMemcpyDeviceToHost));
    std::vector<float> h_logits(c.vocab_size);
    for (int i = 0; i < c.vocab_size; i++)
        h_logits[i] = __half2float(h_logits_half[i]);

    float ref_log_prob = cpu_log_softmax_gather(h_logits.data(), c.vocab_size, last_target);

    float diff = fabsf(train_log_prob - ref_log_prob);
    float tol = 0.05f;  // FP16 accumulation differences across different compute paths
    if (diff < tol) {
        printf("  train=%.6f  inference=%.6f  diff=%.6f  (tol=%.3f)\n",
               train_log_prob, ref_log_prob, diff, tol);
        PASS(name);
    } else {
        FAIL(name, "train=%.6f  inference=%.6f  diff=%.6f > tol=%.3f",
             train_log_prob, ref_log_prob, diff, tol);
    }

    cudaFree(d_log_probs);
    qwen3_train_state_free(state);
    qwen3_reset(m);
}

// =============================================================================
// Test 3: Batch forward
// =============================================================================
static void test_batch_forward(Qwen3Model* m) {
    const char* name = "Batch forward: B=2";
    const Qwen3Config& c = m->config;

    // Two different sequences, padded to same length
    std::vector<int> seq1 = {785, 6722, 315, 9625, 374};       // "The capital of France is"
    std::vector<int> seq2 = {785, 6722, 315, 10765, 374};      // "The capital of Germany is" (approx)
    int B = 2, S = (int)seq1.size();
    int T = B * S;

    std::vector<int> tokens(T);
    for (int i = 0; i < S; i++) tokens[i] = seq1[i];
    for (int i = 0; i < S; i++) tokens[S + i] = seq2[i];

    std::vector<int> targets(T);
    for (int b = 0; b < B; b++)
        for (int s = 0; s < S - 1; s++)
            targets[b * S + s] = tokens[b * S + s + 1];
    targets[S - 1] = 0;
    targets[T - 1] = 0;

    auto* state = qwen3_train_state_alloc(c, B, S);
    float* d_log_probs;
    CUDA_CHECK(cudaMalloc(&d_log_probs, T * sizeof(float)));

    qwen3_forward(m, state, tokens.data(), targets.data(), d_log_probs, B, S);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_log_probs(T);
    CUDA_CHECK(cudaMemcpy(h_log_probs.data(), d_log_probs, T * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int t = 0; t < T; t++) {
        if (!std::isfinite(h_log_probs[t]) || h_log_probs[t] > 1e-5f) {
            FAIL(name, "log_prob[%d] = %f", t, h_log_probs[t]);
            ok = false; break;
        }
    }

    // Seq1 and seq2 should have DIFFERENT log_probs (different input)
    if (ok) {
        float diff_sum = 0.0f;
        for (int s = 0; s < S; s++)
            diff_sum += fabsf(h_log_probs[s] - h_log_probs[S + s]);
        if (diff_sum < 1e-6f) {
            FAIL(name, "batch sequences produced identical log_probs (sum_diff=%.8f)", diff_sum);
            ok = false;
        }
    }

    if (ok) {
        printf("  seq1:");
        for (int s = 0; s < S; s++) printf(" %.4f", h_log_probs[s]);
        printf("\n  seq2:");
        for (int s = 0; s < S; s++) printf(" %.4f", h_log_probs[S + s]);
        printf("\n");
        PASS(name);
    }

    cudaFree(d_log_probs);
    qwen3_train_state_free(state);
}

// =============================================================================
// Test 4: Determinism — running forward twice gives identical results
// =============================================================================
static void test_determinism(Qwen3Model* m) {
    const char* name = "Determinism: two forward passes match";
    const Qwen3Config& c = m->config;

    std::vector<int> tokens = {785, 6722, 315, 9625, 374};
    int B = 1, S = (int)tokens.size(), T = B * S;
    std::vector<int> targets(T);
    for (int i = 0; i < T - 1; i++) targets[i] = tokens[i + 1];
    targets[T - 1] = 0;

    auto* state = qwen3_train_state_alloc(c, B, S);
    float* d_log_probs;
    CUDA_CHECK(cudaMalloc(&d_log_probs, T * sizeof(float)));

    // Run 1
    qwen3_forward(m, state, tokens.data(), targets.data(), d_log_probs, B, S);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> run1(T);
    CUDA_CHECK(cudaMemcpy(run1.data(), d_log_probs, T * sizeof(float), cudaMemcpyDeviceToHost));

    // Run 2
    qwen3_forward(m, state, tokens.data(), targets.data(), d_log_probs, B, S);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> run2(T);
    CUDA_CHECK(cudaMemcpy(run2.data(), d_log_probs, T * sizeof(float), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (int t = 0; t < T; t++)
        max_diff = fmaxf(max_diff, fabsf(run1[t] - run2[t]));

    if (max_diff < 1e-6f) {
        PASS(name);
    } else {
        FAIL(name, "max_diff=%.8f between two identical forward passes", max_diff);
    }

    cudaFree(d_log_probs);
    qwen3_train_state_free(state);
}

// =============================================================================
// Test 5: Activation shapes — verify saved state has correct pointers
// =============================================================================
static void test_activation_save(Qwen3Model* m) {
    const char* name = "Activation save: non-null and non-zero";
    const Qwen3Config& c = m->config;

    std::vector<int> tokens = {785, 6722, 315};
    int B = 1, S = (int)tokens.size(), T = B * S;
    std::vector<int> targets = {6722, 315, 0};

    auto* state = qwen3_train_state_alloc(c, B, S);
    float* d_log_probs;
    CUDA_CHECK(cudaMalloc(&d_log_probs, T * sizeof(float)));

    qwen3_forward(m, state, tokens.data(), targets.data(), d_log_probs, B, S);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Spot-check: read first few elements of layer 0's saved Q
    bool ok = true;
    {
        std::vector<half> q_check(c.q_dim);
        CUDA_CHECK(cudaMemcpy(q_check.data(), state->layer_Q[0],
                              c.q_dim * sizeof(half), cudaMemcpyDeviceToHost));
        float sum = 0.0f;
        for (int i = 0; i < c.q_dim; i++) sum += fabsf(__half2float(q_check[i]));
        if (sum < 1e-10f) {
            FAIL(name, "layer_Q[0] is all zeros (sum=%.10f)", sum);
            ok = false;
        }
    }

    // Check LSE
    if (ok) {
        int lse_sz = T * c.num_attention_heads;
        std::vector<float> lse(lse_sz);
        CUDA_CHECK(cudaMemcpy(lse.data(), state->layer_lse[0],
                              lse_sz * sizeof(float), cudaMemcpyDeviceToHost));
        bool all_finite = true;
        for (int i = 0; i < lse_sz; i++) {
            if (!std::isfinite(lse[i])) { all_finite = false; break; }
        }
        if (!all_finite) {
            FAIL(name, "layer_lse[0] contains non-finite values");
            ok = false;
        }
    }

    // Check final_hidden
    if (ok) {
        std::vector<half> fh(c.hidden_size);
        CUDA_CHECK(cudaMemcpy(fh.data(), state->final_hidden,
                              c.hidden_size * sizeof(half), cudaMemcpyDeviceToHost));
        float sum = 0.0f;
        for (int i = 0; i < c.hidden_size; i++) sum += fabsf(__half2float(fh[i]));
        if (sum < 1e-10f) {
            FAIL(name, "final_hidden is all zeros");
            ok = false;
        }
    }

    if (ok) PASS(name);

    cudaFree(d_log_probs);
    qwen3_train_state_free(state);
}

// =============================================================================
// main
// =============================================================================
int main() {
    if (!std::filesystem::exists(MODEL_DIR)) {
        printf("Model weights not found at %s — skipping tests.\n", MODEL_DIR);
        return 0;
    }

    printf("=== qwen3_forward() training forward tests ===\n\n");

    // Load model with max_batch=4, max_seq=64 (enough for these tests)
    Qwen3Model* m = qwen3_load(MODEL_DIR, /*max_batch=*/4, /*max_seq=*/64);

    test_basic_forward(m);
    test_cross_validate(m);
    test_batch_forward(m);
    test_determinism(m);
    test_activation_save(m);

    qwen3_free(m);

    printf("\n%s\n", g_fails ? "SOME TESTS FAILED" : "All tests PASSED.");
    return g_fails ? 1 : 0;
}
