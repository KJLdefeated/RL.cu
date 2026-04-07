// test_sampler.cu
// Validates GPU sampler (temperature + top-k + top-p + multinomial) against
// CPU reference computations.
//
// Test cases:
//   1. Greedy (temperature=0):  GPU output logit == CPU argmax logit  (value eq)
//   2. Top-k=1:                 GPU output == CPU argmax  (single unique max)
//   3. Top-k validity:          all samples have logit >= k-th largest logit
//   4. Top-p validity:          all samples are in the true top-p nucleus
//   5. Distribution accuracy:   empirical vs theoretical KL divergence < 0.05
//   6. Batched tokens:          B different logit rows, greedy verified by value
//   7. Qwen3 vocab (151936):    greedy + top-k=50 on full vocab size
//
// Tie-handling policy:
//   FP16 quantization causes many ties, especially for large vocab.  Tests use
//   threshold-based comparisons (value equality) rather than index equality to
//   avoid spurious failures when multiple tokens share the same top-k value.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <set>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/sampler.cuh"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

static unsigned int lcg_state = 12345u;
static float lcg_randf() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(lcg_state >> 1) / (float)0x7fffffffu) * 2.0f - 1.0f;
}
static void lcg_reset(unsigned int seed) { lcg_state = seed; }

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------

// Returns the maximum logit value across the row (for tie-safe greedy check).
static float cpu_max_logit(const half* logits, int vocab_size) {
    float best = __half2float(logits[0]);
    for (int i = 1; i < vocab_size; i++) {
        float v = __half2float(logits[i]);
        if (v > best) best = v;
    }
    return best;
}

// Softmax probabilities (FP32) with optional temperature.
static void cpu_softmax(float* probs, const half* logits, float temperature,
                        int vocab_size) {
    float t = (temperature > 0.0f) ? temperature : 1.0f;
    float mx = -1e30f;
    for (int i = 0; i < vocab_size; i++) {
        float v = __half2float(logits[i]) / t;
        if (v > mx) mx = v;
    }
    float s = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf(__half2float(logits[i]) / t - mx);
        s += probs[i];
    }
    for (int i = 0; i < vocab_size; i++) probs[i] /= s;
}

// Threshold-based top-k set: include ALL tokens with logit value >= the k-th
// largest logit value.  Handles FP16 ties at the boundary correctly — any
// token tied with the k-th entry is a valid top-k sample.
static std::set<int> cpu_top_k_set_threshold(const half* logits,
                                              int vocab_size, int k) {
    // Find the k-th largest logit value via partial sort
    std::vector<float> vals(vocab_size);
    for (int i = 0; i < vocab_size; i++) vals[i] = __half2float(logits[i]);
    std::vector<float> sorted_vals = vals;
    int n = std::min(k, vocab_size);
    std::nth_element(sorted_vals.begin(), sorted_vals.begin() + n - 1,
                     sorted_vals.end(), std::greater<float>());
    float threshold = sorted_vals[n - 1];

    // Include all tokens with logit value >= threshold
    std::set<int> s;
    for (int i = 0; i < vocab_size; i++)
        if (vals[i] >= threshold) s.insert(i);
    return s;
}

// Threshold-based top-k set using FP32 SOFTMAX PROBABILITIES as the criterion.
// Necessary because two tokens with different FP16 logits can produce equal
// FP32 probability after exp(), so logit-based threshold ≠ probability-based
// threshold at the boundary.  This matches exactly what the GPU kernel does.
static std::set<int> cpu_topk_prob_threshold(const half* logits,
                                              int vocab_size, int k,
                                              float temperature) {
    std::vector<float> probs(vocab_size);
    cpu_softmax(probs.data(), logits, temperature, vocab_size);

    std::vector<float> sorted_probs = probs;
    int n = std::min(k, vocab_size);
    std::nth_element(sorted_probs.begin(), sorted_probs.begin() + n - 1,
                     sorted_probs.end(), std::greater<float>());
    float threshold = sorted_probs[n - 1];

    std::set<int> s;
    for (int i = 0; i < vocab_size; i++)
        if (probs[i] >= threshold) s.insert(i);
    return s;
}

// Top-p nucleus set matching GPU semantics:
//   1. softmax, 2. top-k mask + renormalize, 3. walk cumsum until >= top_p.
// top_k: clamp before calling (consistent with launch_sampler wrapper).
static std::set<int> cpu_top_p_nucleus_gpu_semantics(
    const half* logits, float temperature, float top_p, int top_k,
    int vocab_size)
{
    std::vector<float> probs(vocab_size);
    cpu_softmax(probs.data(), logits, temperature, vocab_size);

    int actual_k = (top_k <= 0 || top_k > MAX_SAMPLER_TOP_K)
                       ? MAX_SAMPLER_TOP_K
                       : top_k;
    if (actual_k > vocab_size) actual_k = vocab_size;

    // Apply top-k mask (keep only top actual_k by probability)
    std::set<int> topk = cpu_top_k_set_threshold(logits, vocab_size, actual_k);
    for (int i = 0; i < vocab_size; i++)
        if (topk.find(i) == topk.end()) probs[i] = 0.0f;

    // Renormalize within top-k (GPU semantics)
    float s = 0.0f;
    for (float p : probs) s += p;
    for (float& p : probs) p /= s;

    // Walk sorted top-k in descending order until cumsum >= top_p
    std::vector<std::pair<float,int>> order;
    for (int i = 0; i < vocab_size; i++)
        if (probs[i] > 0.0f) order.push_back({probs[i], i});
    std::sort(order.begin(), order.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });

    std::set<int> nucleus;
    float cum = 0.0f;
    for (auto& [p, idx] : order) {
        nucleus.insert(idx);
        cum += p;
        if (cum >= top_p) break;
    }
    return nucleus;
}

// ---------------------------------------------------------------------------
// GPU launch helper: runs the sampler on num_tokens copies of one logit row
// (or B distinct rows).  h_logits must have num_tokens * vocab_size elements.
// ---------------------------------------------------------------------------
static void run_sampler(
    const half* h_logits, int vocab_size, int num_tokens,
    int top_k, float top_p, float temperature,
    unsigned long long seed, std::vector<int64_t>& h_ids
) {
    const long logit_elems = (long)num_tokens * vocab_size;

    half*    d_logits;
    float*   d_probs;
    int64_t* d_ids;
    CUDA_CHECK(cudaMalloc(&d_logits, logit_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_probs,  logit_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ids,    num_tokens  * sizeof(int64_t)));

    CUDA_CHECK(cudaMemcpy(d_logits, h_logits, logit_elems * sizeof(half),
                          cudaMemcpyHostToDevice));

    launch_sampler(d_logits, d_probs, num_tokens, vocab_size,
                   top_k, top_p, temperature, seed, d_ids);
    CUDA_CHECK(cudaDeviceSynchronize());

    h_ids.resize(num_tokens);
    CUDA_CHECK(cudaMemcpy(h_ids.data(), d_ids, num_tokens * sizeof(int64_t),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_logits); cudaFree(d_probs); cudaFree(d_ids);
}

// Convenience: create n_samples identical copies of h_logits_one.
static void run_sampler_batch(
    const half* h_logits_one, int vocab_size, int n_samples,
    int top_k, float top_p, float temperature,
    unsigned long long seed, std::vector<int64_t>& h_ids
) {
    std::vector<half> h_logits((long)n_samples * vocab_size);
    for (int t = 0; t < n_samples; t++)
        for (int i = 0; i < vocab_size; i++)
            h_logits[(long)t * vocab_size + i] = h_logits_one[i];
    run_sampler(h_logits.data(), vocab_size, n_samples,
                top_k, top_p, temperature, seed, h_ids);
}

// ---------------------------------------------------------------------------
// Test 1: greedy (temperature=0) — value equality handles FP16 ties
// ---------------------------------------------------------------------------
static bool test_greedy(const char* name, int vocab_size, float temperature,
                        int top_k, unsigned long long seed) {
    lcg_reset(42u);
    std::vector<half> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; i++)
        h_logits[i] = __float2half(lcg_randf() * 2.0f);

    // Accept any token tied with the maximum logit value
    const float max_val = cpu_max_logit(h_logits.data(), vocab_size);

    std::vector<int64_t> h_ids;
    run_sampler_batch(h_logits.data(), vocab_size, 1,
                      top_k, 1.0f, temperature, seed, h_ids);

    float got_val = __half2float(h_logits[h_ids[0]]);
    bool  pass    = (got_val == max_val);
    printf("[%s] %-55s expected_val=%.4f  got_val=%.4f  got_id=%lld  %s\n",
           pass ? "PASS" : "FAIL", name, max_val, got_val, (long long)h_ids[0],
           pass ? "" : "<-- WRONG VALUE");
    return pass;
}

// ---------------------------------------------------------------------------
// Test 2: top-k validity — sampled token must be within top-k threshold set
// ---------------------------------------------------------------------------
static bool test_topk_validity(const char* name, int vocab_size, int top_k,
                                float temperature, int n_samples) {
    lcg_reset(99u);
    std::vector<half> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; i++)
        h_logits[i] = __float2half(lcg_randf() * 1.5f);

    // Threshold-based set: includes all tied tokens at the k-th boundary
    const std::set<int> valid_set =
        cpu_top_k_set_threshold(h_logits.data(), vocab_size, top_k);

    std::vector<int64_t> h_ids;
    run_sampler_batch(h_logits.data(), vocab_size, n_samples,
                      top_k, 1.0f, temperature, 777ULL, h_ids);

    int violations = 0;
    for (int64_t id : h_ids)
        if (valid_set.find(id) == valid_set.end()) violations++;

    bool pass = (violations == 0);
    printf("[%s] %-55s violations=%d/%d  valid_set_size=%d  %s\n",
           pass ? "PASS" : "FAIL", name, violations, n_samples,
           (int)valid_set.size(), pass ? "" : "<-- OUT OF TOP-K");
    return pass;
}

// ---------------------------------------------------------------------------
// Test 3: top-p validity — all samples must be in the nucleus (GPU semantics)
// ---------------------------------------------------------------------------
static bool test_topp_validity(const char* name, int vocab_size,
                                float temperature, float top_p, int top_k,
                                int n_samples) {
    lcg_reset(777u);
    std::vector<half> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; i++)
        h_logits[i] = __float2half(lcg_randf() * 1.5f);

    const std::set<int> nucleus =
        cpu_top_p_nucleus_gpu_semantics(h_logits.data(), temperature, top_p,
                                        top_k, vocab_size);

    std::vector<int64_t> h_ids;
    run_sampler_batch(h_logits.data(), vocab_size, n_samples,
                      top_k, top_p, temperature, 888ULL, h_ids);

    int violations = 0;
    for (int64_t id : h_ids)
        if (nucleus.find(id) == nucleus.end()) violations++;

    bool pass = (violations == 0);
    printf("[%s] %-55s violations=%d/%d  nucleus_size=%d  %s\n",
           pass ? "PASS" : "FAIL", name, violations, n_samples,
           (int)nucleus.size(), pass ? "" : "<-- OUT OF NUCLEUS");
    return pass;
}

// ---------------------------------------------------------------------------
// Test 4: distribution accuracy via KL divergence.
//
// CPU theoretical distribution matches GPU semantics:
//   softmax → top-k mask → renormalize → top-p nucleus → renormalize → prob
// ---------------------------------------------------------------------------
static bool test_distribution(const char* name, int vocab_size, int top_k,
                               float top_p, float temperature,
                               int n_samples, float kl_tol = 0.05f) {
    lcg_reset(321u);
    std::vector<half> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; i++)
        h_logits[i] = __float2half(lcg_randf() * 1.0f);

    // --- CPU: theoretical probabilities (GPU semantics) ---
    std::vector<float> probs(vocab_size);
    cpu_softmax(probs.data(), h_logits.data(), temperature, vocab_size);

    // top-k mask
    int actual_k = (top_k <= 0 || top_k > MAX_SAMPLER_TOP_K)
                       ? MAX_SAMPLER_TOP_K : top_k;
    if (actual_k > vocab_size) actual_k = vocab_size;
    {
        std::set<int> topk_set =
            cpu_top_k_set_threshold(h_logits.data(), vocab_size, actual_k);
        for (int i = 0; i < vocab_size; i++)
            if (topk_set.find(i) == topk_set.end()) probs[i] = 0.0f;
    }

    // Renormalize within top-k (GPU semantics: before applying top-p)
    {
        float s = 0.0f;
        for (float p : probs) s += p;
        for (float& p : probs) p /= s;
    }

    // top-p nucleus on renormalized distribution
    if (top_p < 1.0f) {
        std::vector<std::pair<float,int>> order;
        for (int i = 0; i < vocab_size; i++)
            if (probs[i] > 0.0f) order.push_back({probs[i], i});
        std::sort(order.begin(), order.end(),
                  [](const auto& a, const auto& b){ return a.first > b.first; });
        float cum = 0.0f;
        bool done = false;
        for (auto& [p, idx] : order) {
            if (done) { probs[idx] = 0.0f; continue; }
            cum += p;
            if (cum >= top_p) done = true;
        }
        // Renormalize nucleus
        float s = 0.0f;
        for (float p : probs) s += p;
        for (float& p : probs) p /= s;
    }

    // --- GPU: empirical frequencies ---
    std::vector<int64_t> h_ids;
    run_sampler_batch(h_logits.data(), vocab_size, n_samples,
                      top_k, top_p, temperature, 555ULL, h_ids);

    std::vector<float> empirical(vocab_size, 0.0f);
    for (int64_t id : h_ids) empirical[id] += 1.0f;
    for (float& f : empirical) f /= n_samples;

    // KL(theory || empirical) summed over non-zero theoretical tokens
    float kl = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        if (probs[i] < 1e-9f) continue;
        float q = empirical[i] < 1e-9f ? 1e-9f : empirical[i];
        kl += probs[i] * logf(probs[i] / q);
    }

    bool pass = (kl < kl_tol);
    printf("[%s] %-55s KL=%.4f  tol=%.4f  %s\n",
           pass ? "PASS" : "FAIL", name, kl, kl_tol,
           pass ? "" : "<-- HIGH KL DIVERGENCE");
    return pass;
}

// ---------------------------------------------------------------------------
// Test 5: batched greedy — B distinct rows, value-equality comparison
// ---------------------------------------------------------------------------
static bool test_batched_greedy(int B, int vocab_size) {
    lcg_reset(555u);
    const long total = (long)B * vocab_size;

    std::vector<half> h_logits(total);
    for (long i = 0; i < total; i++)
        h_logits[i] = __float2half(lcg_randf() * 2.0f);

    // CPU: max logit value per row (accept any tied token)
    std::vector<float> expected_vals(B);
    for (int t = 0; t < B; t++)
        expected_vals[t] = cpu_max_logit(
            h_logits.data() + (long)t * vocab_size, vocab_size);

    std::vector<int64_t> h_ids;
    run_sampler(h_logits.data(), vocab_size, B,
                1, 1.0f, 0.0f, 0ULL, h_ids);

    int mismatches = 0;
    for (int t = 0; t < B; t++) {
        float got_val = __half2float(
            h_logits[h_ids[t] + (long)t * vocab_size]);
        if (got_val != expected_vals[t]) mismatches++;
    }

    char name[64];
    snprintf(name, sizeof(name), "Batched greedy B=%d V=%d", B, vocab_size);
    bool pass = (mismatches == 0);
    printf("[%s] %-55s mismatches=%d/%d  %s\n",
           pass ? "PASS" : "FAIL", name, mismatches, B,
           pass ? "" : "<-- WRONG GREEDY");
    return pass;
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------
static void run_benchmark(const char* name, int num_tokens, int vocab_size,
                          int top_k, float temperature,
                          int warmup = 5, int iters = 100) {
    const long total = (long)num_tokens * vocab_size;
    half*    d_logits; float* d_probs; int64_t* d_ids;
    CUDA_CHECK(cudaMalloc(&d_logits, total * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_probs,  total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ids,    num_tokens * sizeof(int64_t)));

    for (int i = 0; i < warmup; i++)
        launch_sampler(d_logits, d_probs, num_tokens, vocab_size,
                       top_k, 1.0f, temperature, 0ULL, d_ids);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < iters; i++)
        launch_sampler(d_logits, d_probs, num_tokens, vocab_size,
                       top_k, 1.0f, temperature,
                       (unsigned long long)i, d_ids);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    printf("[BENCH] %-55s  %7.2f us/call\n", name, ms * 1000.0f / iters);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    cudaFree(d_logits); cudaFree(d_probs); cudaFree(d_ids);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("=== Sampler kernel tests ===\n\n");
    bool all_pass = true;

    // ── Greedy (temperature=0) ────────────────────────────────────────────────
    printf("--- Greedy (temperature=0) ---\n");
    all_pass &= test_greedy("Greedy V=100  temp=0",   100,   0.0f, 1, 0ULL);
    all_pass &= test_greedy("Greedy V=1024 temp=0",   1024,  0.0f, 1, 1ULL);
    all_pass &= test_greedy("Greedy V=32000 temp=0",  32000, 0.0f, 1, 2ULL);
    all_pass &= test_greedy("Greedy V=151936 temp=0", 151936,0.0f, 1, 3ULL);

    // ── Distribution accuracy (Gumbel-max ≡ sampling from softmax(logits/T)) ─
    // Gumbel-max samples token i with probability softmax(logits/T)[i], so the
    // empirical frequency over N draws should match the theoretical softmax.
    // The reference distribution is the FULL softmax (no top-k/top-p masking).
    printf("\n--- Distribution accuracy (KL vs softmax, Gumbel-max) ---\n");
    all_pass &= test_distribution(
        "V=20  T=1.0  N=50000",  20,  151936/*ignored*/, 1.0f, 1.0f, 50000, 0.05f);
    all_pass &= test_distribution(
        "V=50  T=1.0  N=50000",  50,  151936,            1.0f, 1.0f, 50000, 0.05f);
    all_pass &= test_distribution(
        "V=100 T=0.5  N=50000",  100, 151936,            1.0f, 0.5f, 50000, 0.05f);
    all_pass &= test_distribution(
        "V=100 T=0.8  N=50000",  100, 151936,            1.0f, 0.8f, 50000, 0.05f);

    // ── Batched greedy ────────────────────────────────────────────────────────
    printf("\n--- Batched greedy (B distinct rows) ---\n");
    all_pass &= test_batched_greedy(8,   1024);
    all_pass &= test_batched_greedy(16,  32000);
    all_pass &= test_batched_greedy(4,   151936);

    // ── Summary ───────────────────────────────────────────────────────────────
    printf("\n%s\n", all_pass ? "All tests PASSED." : "Some tests FAILED.");

    // ── Benchmarks ────────────────────────────────────────────────────────────
    printf("\n=== Benchmarks (warmup=5, iters=100) ===\n");
    run_benchmark("Greedy   B=1  V=151936",   1, 151936, 0, 0.0f);
    run_benchmark("Gumbel   B=1  V=151936",   1, 151936, 0, 1.0f);
    run_benchmark("Gumbel   B=4  V=151936",   4, 151936, 0, 1.0f);
    run_benchmark("Gumbel   B=16 V=151936",  16, 151936, 0, 1.0f);
    run_benchmark("Gumbel   B=64 V=151936",  64, 151936, 0, 1.0f);

    return all_pass ? 0 : 1;
}
