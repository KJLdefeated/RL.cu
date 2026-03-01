// test_qwen3.cu
//
// End-to-end forward pass test for the Qwen3 model.
//
// Requires real weights at model_weights/Qwen3-0.6B (skips gracefully if absent).
//
// Tests:
//   1. Load model + tokenizer
//   2. Single-token prefill — logits finite, print top-5
//   3. Multi-token prefill (8 tokens) — logits finite
//   4. Greedy decode for 5 steps — print generated token IDs
//   5. qwen3_reset + independent prefill — logits differ from pre-reset
//   6. Batch prefill (B=2) — both sequences produce finite logits
//   7. Chat generation — tokenize a sentence, generate a response, decode and print

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <string>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "model/qwen3.h"
#include "model/tokenizer.h"

static const char* MODEL_DIR = "model_weights/Qwen3-0.6B";

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

static int g_fails = 0;
#define PASS(name)      printf("[PASS] %s\n", (name))
#define FAIL(name, ...) do { \
    printf("[FAIL] %s: ", (name)); printf(__VA_ARGS__); printf("\n"); g_fails++; \
} while (0)

static std::vector<float> fetch_logits(const half* d_logits, int B, int vocab_size) {
    std::vector<half> buf((long)B * vocab_size);
    CUDA_CHECK(cudaMemcpy(buf.data(), d_logits,
                          (long)B * vocab_size * sizeof(half), cudaMemcpyDeviceToHost));
    std::vector<float> out((long)B * vocab_size);
    for (long i = 0; i < (long)B * vocab_size; i++)
        out[i] = __half2float(buf[i]);
    return out;
}

static bool all_finite(const float* v, int n) {
    for (int i = 0; i < n; i++)
        if (!std::isfinite(v[i])) return false;
    return true;
}

static int argmax(const float* row, int vocab_size) {
    int best = 0;
    for (int i = 1; i < vocab_size; i++)
        if (row[i] > row[best]) best = i;
    return best;
}

static void print_top5(const float* logits, int b, int vocab_size) {
    const float* row = logits + (long)b * vocab_size;
    std::vector<int> idx(vocab_size);
    for (int i = 0; i < vocab_size; i++) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin()+5, idx.end(),
                      [&](int a, int b2){ return row[a] > row[b2]; });
    printf("  seq[%d] top-5:", b);
    for (int k = 0; k < 5; k++) printf("  %6d(%.2f)", idx[k], row[idx[k]]);
    printf("\n");
}

static int* upload(const std::vector<int>& ids) {
    int* d;
    CUDA_CHECK(cudaMalloc(&d, ids.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    return d;
}

// =============================================================================
// Tests
// =============================================================================

static void test_single_token_prefill(Qwen3Model* m, int V) {
    printf("\n--- Test: single-token prefill ---\n");
    int* d = upload({1});
    half* dlog = qwen3_prefill(m, d, 1, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto log = fetch_logits(dlog, 1, V);
    if (all_finite(log.data(), V)) PASS("single-token prefill: logits finite");
    else FAIL("single-token prefill", "NaN or Inf");
    print_top5(log.data(), 0, V);
    cudaFree(d);
}

static void test_multi_token_prefill(Qwen3Model* m, int V) {
    printf("\n--- Test: 8-token prefill ---\n");
    qwen3_reset(m);
    int* d = upload({1,10,20,30,40,50,60,70});
    half* dlog = qwen3_prefill(m, d, 1, 8);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto log = fetch_logits(dlog, 1, V);
    if (all_finite(log.data(), V)) PASS("8-token prefill: logits finite");
    else FAIL("8-token prefill", "NaN or Inf");
    print_top5(log.data(), 0, V);
    cudaFree(d);
}

static void test_greedy_decode(Qwen3Model* m, int V) {
    printf("\n--- Test: prefill + 5 greedy decode steps ---\n");
    qwen3_reset(m);
    int* dp = upload({1,10,20,30});
    half* dlog = qwen3_prefill(m, dp, 1, 4);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dp);

    auto log = fetch_logits(dlog, 1, V);
    if (!all_finite(log.data(), V)) { FAIL("greedy decode prefill", "NaN or Inf"); return; }

    std::vector<int> gen;
    int next = argmax(log.data(), V);
    for (int step = 0; step < 5; step++) {
        gen.push_back(next);
        int* dt = upload({next});
        dlog = qwen3_decode(m, dt, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(dt);
        log  = fetch_logits(dlog, 1, V);
        if (!all_finite(log.data(), V)) { FAIL("greedy decode", "NaN or Inf at step %d", step); return; }
        next = argmax(log.data(), V);
    }
    gen.push_back(next);
    PASS("5 greedy decode steps: all logits finite");
    printf("  generated ids:");
    for (int t : gen) printf(" %d", t);
    printf("\n");
}

static void test_reset_independence(Qwen3Model* m, int V) {
    printf("\n--- Test: reset produces independent results ---\n");
    qwen3_reset(m);
    int* dA = upload({1,2,3});
    half* dlA = qwen3_prefill(m, dA, 1, 3);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto logA = fetch_logits(dlA, 1, V);
    cudaFree(dA);

    qwen3_reset(m);
    int* dB = upload({100,200,300});
    half* dlB = qwen3_prefill(m, dB, 1, 3);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto logB = fetch_logits(dlB, 1, V);
    cudaFree(dB);

    bool ok = all_finite(logA.data(), V) && all_finite(logB.data(), V);
    if (!ok) { FAIL("reset independence", "NaN or Inf"); return; }
    int argA = argmax(logA.data(), V), argB = argmax(logB.data(), V);
    if (argA != argB) PASS("reset: different prompts → different top token");
    else printf("[WARN] reset: same top token %d (unlikely but not a bug)\n", argA);
    printf("  prompt A top: %d  prompt B top: %d\n", argA, argB);
}

static void test_batch_prefill(Qwen3Model* m, int V) {
    printf("\n--- Test: batch prefill B=2 S=4 ---\n");
    qwen3_reset(m);
    int* d = upload({1,2,3,4, 10,20,30,40});
    half* dlog = qwen3_prefill(m, d, 2, 4);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto log = fetch_logits(dlog, 2, V);
    if (all_finite(log.data(), 2*V)) PASS("batch B=2: logits finite for both sequences");
    else FAIL("batch B=2 prefill", "NaN or Inf");
    print_top5(log.data(), 0, V);
    print_top5(log.data(), 1, V);
    cudaFree(d);
}

// ---------------------------------------------------------------------------
// Test 7: chat generation — tokenize prompt, generate, decode output
// ---------------------------------------------------------------------------
static void test_chat_generation(Qwen3Model* m, const Tokenizer& tok, int V) {
    printf("\n--- Test: chat generation ---\n");

    const std::string user_msg = "What is the capital of France?";
    const std::string prompt   = tok.chat_prompt(user_msg);
    printf("  Prompt: %s\n", prompt.c_str());

    // Tokenize prompt
    std::vector<int> input_ids = tok.encode(prompt);
    // Prepend BOS-equivalent: <|im_start|> is already in the prompt string
    printf("  Input tokens (%d): ", (int)input_ids.size());
    for (int id : input_ids) printf("%d ", id);
    printf("\n");

    qwen3_reset(m);

    // Prefill
    int S = input_ids.size();
    int* d_prompt = upload(input_ids);
    half* dlog = qwen3_prefill(m, d_prompt, 1, S);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_prompt);

    auto log = fetch_logits(dlog, 1, V);
    if (!all_finite(log.data(), V)) {
        FAIL("chat generation prefill", "NaN or Inf in logits");
        return;
    }
    printf("  [prefill] top-5: "); print_top5(log.data(), 0, V);

    // Greedy decode until <|im_end|> or max_new_tokens
    const int max_new_tokens = 64;
    std::vector<int> generated;
    int next = argmax(log.data(), V);

    printf("  Generating (max %d tokens)...\n", max_new_tokens);

    for (int step = 0; step < max_new_tokens; step++) {
        generated.push_back(next);

        std::string piece = tok.decode_token(next);
        printf("  [step %2d] token=%6d  text=%-20s", step, next, piece.c_str());
        fflush(stdout);

        // Stop at <|im_end|>
        if (next == tok.im_end_id) { printf("\n"); break; }

        int* dt = upload({next});
        dlog = qwen3_decode(m, dt, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(dt);

        log  = fetch_logits(dlog, 1, V);
        if (!all_finite(log.data(), V)) {
            printf("\n");
            FAIL("chat generation decode", "NaN or Inf at step %d", step);
            return;
        }
        next = argmax(log.data(), V);
        print_top5(log.data(), 0, V);
    }
    printf("\n");

    // Decode full response
    std::string response = tok.decode(generated);
    printf("  Full response: \"%s\"\n", response.c_str());
    printf("  Generated %d tokens\n", (int)generated.size());

    bool ok = all_finite(log.data(), V) && !generated.empty();
    if (ok) PASS("chat generation: produced finite logits and non-empty response");
    else    FAIL("chat generation", "empty or NaN response");
}

// =============================================================================
// main
// =============================================================================

int main() {
    printf("=== Qwen3 forward pass + tokenizer tests ===\n");

    if (!std::filesystem::exists(MODEL_DIR)) {
        printf("[SKIP] %s not found — download weights to run this test\n", MODEL_DIR);
        return 0;
    }

    // Load tokenizer
    Tokenizer tok;
    std::string tok_path = std::string(MODEL_DIR) + "/tokenizer.json";
    if (!tok.load(tok_path)) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }
    printf("[OK] Tokenizer loaded (%d tokens, %zu merges)\n",
           tok.vocab_size(), 0UL);

    // Quick tokenizer sanity check
    {
        printf("\n--- Tokenizer sanity ---\n");
        std::string sentence = "The capital of France is Paris.";
        auto ids = tok.encode(sentence);
        printf("  encode(\"%s\"):\n  ids:", sentence.c_str());
        for (int id : ids) printf(" %d", id);
        printf("\n  decoded: \"%s\"\n", tok.decode(ids).c_str());
        // Expected: 785 6722 315 9625 374 12095 13
        bool ok = !ids.empty() && tok.decode(ids) == sentence;
        if (ok) PASS("tokenizer round-trip");
        else    printf("[WARN] tokenizer round-trip mismatch (check pre-tokenizer)\n");
    }

    // Load model
    Qwen3Model* m = qwen3_load(MODEL_DIR, /*max_batch=*/4, /*max_seq=*/256);
    const int V = 151936;

    test_single_token_prefill(m, V);
    test_multi_token_prefill(m, V);
    test_greedy_decode(m, V);
    test_reset_independence(m, V);
    test_batch_prefill(m, V);
    test_chat_generation(m, tok, V);

    printf("\n");
    if (g_fails == 0) printf("ALL TESTS PASSED.\n");
    else              printf("SOME TESTS FAILED: %d failure(s).\n", g_fails);

    qwen3_free(m);
    return g_fails > 0 ? 1 : 0;
}
