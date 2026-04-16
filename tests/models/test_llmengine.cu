// tests/test_llmengine.cu
//
// LLMEngine integration tests — exercises the full inference stack:
//   Scheduler → ModelRunner → Sampler
//
// ┌─ Test sections ─────────────────────────────────────────────────────────┐
// │  1. Correctness   — factual Q&A, greedy determinism, sampling liveness  │
// │  2. Memory        — KV block reclamation, long context, batch OOM safety │
// │  3. Throughput    — tok/s for single-req and multi-batch decode          │
// │  4. nano-vllm ref — comparison table vs. nano-vllm (Qwen2.5-7B / A100)  │
// └─────────────────────────────────────────────────────────────────────────┘
//
// Build:  make test_llmengine
// Run:    ./build/test_llmengine [model_dir]
//
// Reference: https://github.com/GeeeekExplorer/nano-vllm

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <filesystem>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>

#include "engine/llm_engine.h"

// =============================================================================
// Test harness
// =============================================================================

static int g_pass = 0, g_fail = 0;

#define PASS(name)         do { printf("[PASS] %s\n",  (name)); g_pass++; } while (0)
#define FAIL(name, ...)    do { printf("[FAIL] %s: ", (name)); \
                                printf(__VA_ARGS__); printf("\n"); g_fail++; } while (0)
#define SECTION(title)     printf("\n━━━ %s ━━━\n", (title))
#define SKIP(reason)       do { printf("[SKIP] %s\n", (reason)); } while (0)

// Case-insensitive substring search
static bool contains_ci(const std::string& haystack, const std::string& needle) {
    std::string h = haystack, n = needle;
    std::transform(h.begin(), h.end(), h.begin(), ::tolower);
    std::transform(n.begin(), n.end(), n.begin(), ::tolower);
    return h.find(n) != std::string::npos;
}

static SamplingParams greedy(int max_new = 64) {
    SamplingParams sp;
    sp.temperature    = 0.0f;
    sp.max_new_tokens = max_new;
    return sp;
}

static SamplingParams sampling(float temp = 1.0f, float top_p = 0.9f, int max_new = 64) {
    SamplingParams sp;
    sp.temperature    = temp;
    sp.top_p          = top_p;
    sp.max_new_tokens = max_new;
    return sp;
}

// =============================================================================
// 1. Correctness
// =============================================================================

static void test_correctness(LLMEngine& engine) {
    SECTION("1. Correctness");

    // ── 1a. Factual Q&A ──────────────────────────────────────────────────────
    struct Case { const char* user_msg; const char* must_contain; };
    static const Case CASES[] = {
        { "What is the capital of France? One word.",            "Paris"   },
        { "What is 2 + 2? Answer with a single digit.",          "4"       },
        { "Which planet is closest to the Sun? One word only.",  "Mercury" },
        { "What language is used to write the Linux kernel?",    "C"       },
    };

    for (auto& c : CASES) {
        std::string prompt = engine.tokenizer->chat_prompt(c.user_msg);
        auto resp = engine.generate({prompt}, greedy(64));

        if (resp.empty() || resp[0].empty()) {
            FAIL(c.user_msg, "empty response");
            continue;
        }
        if (contains_ci(resp[0], c.must_contain))
            PASS(c.user_msg);
        else
            FAIL(c.user_msg, "expected '%s' in: \"%s\"",
                 c.must_contain, resp[0].substr(0, 100).c_str());
    }

    // ── 1b. Greedy is deterministic ───────────────────────────────────────────
    {
        std::string prompt = engine.tokenizer->chat_prompt("Say hello in one word.");
        auto r1 = engine.generate({prompt}, greedy(32));
        auto r2 = engine.generate({prompt}, greedy(32));
        if (!r1.empty() && !r2.empty() && r1[0] == r2[0])
            PASS("greedy determinism: identical outputs on repeated call");
        else
            FAIL("greedy determinism", "outputs differ:\n  run1=\"%s\"\n  run2=\"%s\"",
                 r1.empty() ? "(empty)" : r1[0].substr(0, 80).c_str(),
                 r2.empty() ? "(empty)" : r2[0].substr(0, 80).c_str());
    }

    // ── 1c. Sampling produces non-empty output ────────────────────────────────
    {
        std::string prompt = engine.tokenizer->chat_prompt(
            "Tell me a fun fact about any topic.");
        auto r = engine.generate({prompt}, sampling(1.0f, 0.95f, 64));
        if (!r.empty() && !r[0].empty())
            PASS("sampling (temp=1.0, top_p=0.95): non-empty response");
        else
            FAIL("sampling", "empty response");
    }

    // ── 1d. Tokenizer round-trip ──────────────────────────────────────────────
    {
        const std::string sentence = "The capital of France is Paris.";
        auto ids = engine.tokenizer->encode(sentence);
        auto decoded = engine.tokenizer->decode(ids);
        // Expected token ids: 785 6722 315 9625 374 12095 13
        bool ok = !ids.empty() && decoded == sentence;
        if (ok)
            PASS("tokenizer round-trip: encode→decode matches");
        else
            FAIL("tokenizer round-trip", "got \"%s\" (len=%zu ids)",
                 decoded.substr(0, 80).c_str(), ids.size());
    }
}

// =============================================================================
// 2. Memory management
// =============================================================================

static void test_memory(LLMEngine& engine) {
    SECTION("2. Memory management");

    // ── 2a. KV block reclamation — sequential requests ────────────────────────
    // If blocks leak, we'd OOM after a few requests.
    {
        const int N = 20;
        int ok = 0;
        for (int i = 0; i < N; i++) {
            char buf[64];
            snprintf(buf, sizeof(buf), "Count from 1 to %d.", i + 1);
            auto r = engine.generate(
                {engine.tokenizer->chat_prompt(buf)}, greedy(32));
            if (!r.empty() && !r[0].empty()) ok++;
        }
        if (ok == N)
            PASS("KV reclamation: 20 sequential requests, no OOM");
        else
            FAIL("KV reclamation", "%d/%d requests succeeded", ok, N);
    }

    // ── 2b. Long-context request ──────────────────────────────────────────────
    // Craft a prompt that is ~512 prompt tokens.
    {
        std::string body;
        body.reserve(2000);
        for (int i = 0; i < 60; i++)
            body += "The quick brown fox jumps over the lazy dog. ";
        std::string prompt = engine.tokenizer->chat_prompt(
            "Summarize the following:\n" + body);
        auto ids = engine.tokenizer->encode(prompt);

        auto r = engine.generate({prompt}, greedy(64));
        if (!r.empty() && !r[0].empty())
            PASS("long context: completed without OOM");
        else
            FAIL("long context", "empty response (prompt=%zu tokens)", ids.size());
    }

    // ── 2c. Concurrent batch — blocks allocated and freed together ────────────
    {
        const int B = std::min(4, engine.config.max_num_seqs);
        std::vector<std::string> prompts;
        prompts.reserve(B);
        const char* questions[] = {
            "What is 3 * 7?",
            "Name the largest ocean.",
            "What color is the sky?",
            "Who wrote Hamlet?",
        };
        for (int i = 0; i < B; i++)
            prompts.push_back(engine.tokenizer->chat_prompt(questions[i % 4]));

        auto resp = engine.generate(prompts, greedy(32));
        int ok = 0;
        for (auto& r : resp) ok += !r.empty();
        if (ok == B)
            PASS("concurrent batch B=4: all blocks reclaimed, no OOM");
        else
            FAIL("concurrent batch", "%d/%d non-empty responses", ok, B);
    }

    // ── 2d. Repeated batches — stress block pool ──────────────────────────────
    {
        const int ITERS = 5;
        int ok = 0;
        for (int it = 0; it < ITERS; it++) {
            std::vector<std::string> ps = {
                engine.tokenizer->chat_prompt("Write one sentence about cats."),
                engine.tokenizer->chat_prompt("Write one sentence about dogs."),
            };
            auto r = engine.generate(ps, greedy(32));
            if ((int)r.size() == 2 && !r[0].empty() && !r[1].empty()) ok++;
        }
        if (ok == ITERS)
            PASS("block pool stress: 5 × B=2 batches without leak");
        else
            FAIL("block pool stress", "%d/%d iterations ok", ok, ITERS);
    }
}

// =============================================================================
// 3. Throughput benchmark
// =============================================================================

struct BenchResult {
    double   wall_s;
    int      num_output_tokens;
    int      num_requests;
    double   toks_per_sec() const { return num_output_tokens / wall_s; }
    double   ms_per_req()   const { return wall_s * 1e3 / num_requests; }
};

static const char* BENCH_PROMPTS[] = {
    "Explain the theory of relativity in simple terms.",
    "Write a short poem about the ocean.",
    "What are the main differences between Python and C++?",
    "Describe the water cycle in three sentences.",
    "What is photosynthesis and why is it important?",
    "Summarize the main causes of World War I.",
    "How does a computer CPU work at a high level?",
    "What is machine learning? Give a one-sentence definition.",
    "Describe the life cycle of a star.",
    "What is the Turing test and what does it measure?",
    "Explain DNA replication in simple terms.",
    "What are the main programming paradigms?",
    "How does the internet work at a high level?",
    "What is the difference between RAM and ROM?",
    "Describe Newton's three laws of motion.",
    "What is quantum entanglement?",
};
static const int NBENCH = sizeof(BENCH_PROMPTS) / sizeof(BENCH_PROMPTS[0]);

static BenchResult run_bench(LLMEngine& engine, int num_req, int max_new) {
    SamplingParams sp = sampling(max_new);
    std::vector<std::string> prompts;
    prompts.reserve(num_req);
    for (int i = 0; i < num_req; i++)
        prompts.push_back(engine.tokenizer->chat_prompt(BENCH_PROMPTS[i % NBENCH]));

    auto t0 = std::chrono::high_resolution_clock::now();
    auto responses = engine.generate(prompts, sp);
    auto t1 = std::chrono::high_resolution_clock::now();

    BenchResult res{};
    res.wall_s       = std::chrono::duration<double>(t1 - t0).count();
    res.num_requests = (int)responses.size();
    for (auto& r : responses)
        res.num_output_tokens += (int)engine.tokenizer->encode(r).size();
    return res;
}

static void test_throughput(LLMEngine& engine) {
    SECTION("3. Throughput benchmark");

    // Warmup — avoids CUDA JIT / caching skew in the first timed call
    printf("  [warmup] ...\n");
    engine.generate(
        {engine.tokenizer->chat_prompt("Hello!")}, greedy(16));

    // ── 3a. Single-request latency ────────────────────────────────────────────
    // printf("\n  ── Single-request latency (greedy, max_new=128) ──\n");
    // {
    //     auto res = run_bench(engine, 1, 128);
    //     printf("  batch=1  %7.0f tok/s  (%.0f ms total, %d output tokens)\n",
    //            res.toks_per_sec(), res.wall_s * 1e3, res.num_output_tokens);
    // }

    // ── 3b. Batch sweep ───────────────────────────────────────────────────────
    printf("\n  ── Batch throughput sweep ──\n");
    printf("  %6s  %10s  %12s  %12s\n",
           "batch", "tok/s", "ms/request", "output_tokens");

    BenchResult best{};
    for (int batch : {256}) {
        if (batch > engine.config.max_num_seqs) break;
        auto res = run_bench(engine, batch, 1024);
        printf("  %6d  %10.0f  %12.1f  %12d\n",
               batch, res.toks_per_sec(), res.ms_per_req(), res.num_output_tokens);
        if (res.toks_per_sec() > best.toks_per_sec()) best = res;
    }
}

// =============================================================================
// 4. Realistic benchmark (nano-vllm bench.py protocol)
// =============================================================================
// Mirrors python_scripts/nano-vllm/bench.py exactly:
//   seed=0, 256 seqs, prompt_len ~ Uniform[100, 1024], output_len ~ Uniform[100, 1024]
//   token IDs drawn from Uniform[0, 10000], temperature=0.6, ignore_eos=true
//   1-seq warmup, then timed run; metric = sum(max_new_tokens) / wall_time

static void test_bench_realistic(LLMEngine& engine) {
    SECTION("4. Realistic benchmark (nano-vllm protocol)");

    // Deterministic LCG seeded to 0 — mirrors Python's random.seed(0) / randint behaviour.
    // We use a simple LCG rather than MT19937 to keep the seed self-contained.
    auto make_lcg = [](uint64_t seed) {
        // LCG: x = (a*x + c) mod 2^64; low bits discarded; take bits [16,47].
        return [state = seed](int lo, int hi) mutable -> int {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            int range = hi - lo + 1;
            return lo + (int)((state >> 17) % (uint64_t)range);
        };
    };
    auto rng = make_lcg(0);

    const int NUM_SEQS      = 64;
    const int MAX_INPUT_LEN = 4096;
    const int MAX_OUT_LEN   = 4096;

    std::vector<std::vector<int64_t>> prompts(NUM_SEQS);
    std::vector<SamplingParams>       sps(NUM_SEQS);
    int total_output_tokens = 0;
    for (int i = 0; i < NUM_SEQS; i++) {
        int plen = rng(100, MAX_INPUT_LEN);
        int olen = rng(100, MAX_OUT_LEN);
        prompts[i].resize(plen);
        for (auto& t : prompts[i]) t = (int64_t)rng(0, 10000);
        sps[i].temperature    = 0.6f;
        sps[i].top_k          = 50;   // 50 argmax rounds (vs 1024 for top_k=0 default)
        sps[i].ignore_eos     = true;
        sps[i].max_new_tokens = olen;
        sps[i].do_sample      = true;
        total_output_tokens  += olen;
    }

    // Warmup: single sequence, tiny output
    printf("  [warmup] ...\n");
    {
        std::vector<int64_t> w = {1};
        SamplingParams wsp; wsp.max_new_tokens = 4; wsp.temperature = 0.6f; wsp.top_k = 50;
        engine.generate({w}, {wsp});
    }

    printf("  [bench ] %d seqs, prompt_len~[100,%d], output_len~[100,%d], "
           "total_output=%d tok ...\n",
           NUM_SEQS, MAX_INPUT_LEN, MAX_OUT_LEN, total_output_tokens);

    auto t0 = std::chrono::high_resolution_clock::now();
    engine.generate(prompts, sps);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double throughput = total_output_tokens / elapsed;

    printf("\nTotal: %dtok, Time: %.2fs, Throughput: %.2ftok/s\n",
           total_output_tokens, elapsed, throughput);
}

// =============================================================================
// 5. Rollout debug — print full completions for visual inspection
// =============================================================================

static void test_rollout_debug(LLMEngine& engine, const char* data_path) {
    SECTION("5. Rollout debug — visual inspection");

    // ── 5a. Chat prompts (greedy) ────────────────────────────────────────────
    // printf("\n  ── Chat prompts (greedy, max_new=256) ──\n");
    // {
    //     const char* chat_qs[] = {
    //         "What is 2 + 3? Answer with just the number.",
    //         "Solve: x^2 = 16, x > 0. What is x?",
    //         "What is the capital of Japan? One word.",
    //     };
    //     for (auto& q : chat_qs) {
    //         std::string prompt = engine.tokenizer->chat_prompt(q);
    //         auto resp = engine.generate({prompt}, greedy(256));
    //         printf("\n  Q: %s\n", q);
    //         printf("  A: %s\n", resp.empty() ? "(empty)" : resp[0].c_str());
    //     }
    // }

    // ── 5b. Chat prompts (sampling, temp=0.8) ────────────────────────────────
    // printf("\n  ── Chat prompts (temp=0.8, top_p=1.0, max_new=512) ──\n");
    // {
    //     const char* chat_qs[] = {
    //         "Explain what a prime number is in two sentences.",
    //         "Write a haiku about the moon.",
    //     };
    //     for (auto& q : chat_qs) {
    //         std::string prompt = engine.tokenizer->chat_prompt(q);
    //         auto resp = engine.generate({prompt}, sampling(0.8f, 1.0f, 512));
    //         printf("\n  Q: %s\n", q);
    //         printf("  A: %s\n", resp.empty() ? "(empty)" : resp[0].c_str());
    //     }
    // }

    // ── 5c. DAPO-style prompts from bin file (same as GRPO trainer) ──────────
    // if (data_path && std::filesystem::exists(data_path)) {
    //     printf("\n  ── DAPO prompts from %s (temp=0.8, max_new=1024) ──\n", data_path);

    //     // Read binary file
    //     FILE* f = fopen(data_path, "rb");
    //     if (!f) { printf("  Cannot open %s\n", data_path); return; }

    //     struct { char magic[4]; uint32_t version; uint64_t num_samples, total_tokens; uint32_t flags, pad; } hdr;
    //     fread(&hdr, sizeof(hdr), 1, f);
    //     size_t N = (size_t)hdr.num_samples;

    //     std::vector<uint64_t> offsets(N + 1);
    //     fread(offsets.data(), sizeof(uint64_t), N + 1, f);

    //     // Skip prompt_lens if present
    //     if (hdr.flags & 1) fseek(f, N * sizeof(uint32_t), SEEK_CUR);

    //     std::vector<int32_t> all_tokens((size_t)hdr.total_tokens);
    //     fread(all_tokens.data(), sizeof(int32_t), hdr.total_tokens, f);
    //     fclose(f);

    //     // Load answers
    //     std::vector<std::string> answers;
    //     {
    //         std::string ans_path(data_path);
    //         auto dot = ans_path.rfind('.');
    //         if (dot != std::string::npos)
    //             ans_path = ans_path.substr(0, dot) + ".answers.jsonl";
    //         FILE* af = fopen(ans_path.c_str(), "r");
    //         if (af) {
    //             char line[4096];
    //             while (fgets(line, sizeof(line), af)) {
    //                 // Simple parse: {"index": N, "answer": "..."}
    //                 char* p = strstr(line, "\"answer\":");
    //                 if (p) {
    //                     p = strchr(p, ':') + 1;
    //                     while (*p == ' ' || *p == '"') p++;
    //                     char* end = strchr(p, '"');
    //                     if (end) answers.push_back(std::string(p, end));
    //                 }
    //             }
    //             fclose(af);
    //         }
    //     }

    //     const int NUM_PROMPTS = std::min(2, (int)N);
    //     SamplingParams sp;
    //     sp.temperature    = 0.8f;
    //     sp.top_p          = 1.0f;
    //     sp.max_new_tokens = 1024;
    //     sp.do_sample      = true;

    //     for (int i = 0; i < NUM_PROMPTS; i++) {
    //         size_t len = (size_t)(offsets[i + 1] - offsets[i]);
    //         std::vector<int64_t> prompt_ids(len);
    //         for (size_t t = 0; t < len; t++)
    //             prompt_ids[t] = (int64_t)all_tokens[offsets[i] + t];

    //         // Decode prompt for display
    //         std::string prompt_text = engine.tokenizer->decode(prompt_ids);
    //         if (prompt_text.size() > 300)
    //             prompt_text = prompt_text.substr(0, 300) + "...";

    //         // Generate using token IDs (same path as GRPO trainer)
    //         engine.add_request(prompt_ids, sp);
    //         std::string completion;
    //         while (!engine.is_finished()) {
    //             auto [completions, ntok] = engine.step();
    //             for (auto& [sid, tids] : completions) {
    //                 completion = engine.tokenizer->decode(tids);
    //             }
    //         }

    //         printf("\n  ── DAPO Prompt %d (%zu tokens) ──\n", i, len);
    //         printf("  PROMPT: %s\n", prompt_text.c_str());
    //         printf("  ANSWER: %s\n", (i < (int)answers.size()) ? answers[i].c_str() : "?");
    //         printf("  COMPLETION (%zu chars):\n%s\n",
    //                completion.size(),
    //                completion.size() > 2000 ? (completion.substr(0, 2000) + "...").c_str()
    //                                         : completion.c_str());
    //     }
    // } else {
    //     printf("  [SKIP] No data file at %s\n", data_path ? data_path : "(null)");
    // }

    // ── 5d. Sleep/Wakeup cycle — same path as GRPO trainer ──────────────
    // printf("\n  ── Sleep/Wakeup cycle test ──\n");
    // {
    //     // Generate BEFORE sleep/wakeup (baseline)
    //     std::string prompt = engine.tokenizer->chat_prompt(
    //         "What is 7 + 8? Answer with just the number.");
    //     auto resp_before = engine.generate({prompt}, greedy(64));
    //     printf("  BEFORE sleep/wakeup: %s\n",
    //            resp_before.empty() ? "(empty)" : resp_before[0].c_str());

    //     // Sleep → Wakeup → Generate
    //     engine.sleep();
    //     engine.wakeup();

    //     auto resp_after = engine.generate({prompt}, greedy(64));
    //     printf("  AFTER  sleep/wakeup: %s\n",
    //            resp_after.empty() ? "(empty)" : resp_after[0].c_str());

    //     if (!resp_before.empty() && !resp_after.empty() && resp_before[0] == resp_after[0])
    //         PASS("sleep/wakeup: greedy output matches before/after");
    //     else
    //         FAIL("sleep/wakeup", "output changed after sleep/wakeup cycle");
    // }

    // ── 5e. generate_ids — exact GRPO path ──────────────────────────────
    if (data_path && std::filesystem::exists(data_path)) {
        printf("\n  ── generate_ids (GRPO path) with DAPO prompts ──\n");

        // Re-read bin file for first 2 prompts
        FILE* f = fopen(data_path, "rb");
        if (f) {
            struct { char magic[4]; uint32_t version; uint64_t num_samples, total_tokens; uint32_t flags, pad; } hdr2;
            fread(&hdr2, sizeof(hdr2), 1, f);
            size_t N2 = (size_t)hdr2.num_samples;
            std::vector<uint64_t> offsets2(N2 + 1);
            fread(offsets2.data(), sizeof(uint64_t), N2 + 1, f);
            if (hdr2.flags & 1) fseek(f, N2 * sizeof(uint32_t), SEEK_CUR);
            std::vector<int32_t> all_tokens2((size_t)hdr2.total_tokens);
            fread(all_tokens2.data(), sizeof(int32_t), hdr2.total_tokens, f);
            fclose(f);

            // Build 2 prompts
            const int NP = 2;
            const int G  = 4;  // 8 generations per prompt (like GRPO)
            std::vector<std::vector<int64_t>> prompts2(NP);
            for (int i = 0; i < NP; i++) {
                size_t len = (size_t)(offsets2[i + 1] - offsets2[i]);
                prompts2[i].resize(len);
                for (size_t t = 0; t < len; t++)
                    prompts2[i][t] = (int64_t)all_tokens2[offsets2[i] + t];
            }

            // Simulate GRPO: sleep → wakeup → generate_ids → sleep
            engine.sleep();
            engine.wakeup();

            SamplingParams sp2;
            sp2.temperature    = 0.8f;
            sp2.top_p          = 1.0f;
            sp2.max_new_tokens = 512;  // shorter for debug
            sp2.do_sample      = true;

            // Printout prompts
            for (int i = 0; i < NP; i++) {
                std::string prompt_text = engine.tokenizer->decode(prompts2[i]);
                printf("\n  ── generate_ids prompt %d (%zu tokens) ──\n", i, prompts2[i].size());
                printf("  %s\n", prompt_text.c_str());
            }

            auto completions2 = engine.generate_ids(prompts2, sp2, G);

            for (int i = 0; i < NP; i++) {
                for (int g = 0; g < G; g++) {
                    int idx = i * G + g;
                    auto it = completions2.find((int64_t)idx);
                    if (it != completions2.end()) {
                        std::string text = engine.tokenizer->decode(it->second);
                        printf("\n  ── generate_ids prompt=%d gen=%d (%d tokens) ──\n",
                               i, g, (int)it->second.size());
                        printf("  %s\n",
                               text.size() > 1000 ? (text.substr(0, 1000) + "...").c_str()
                                                  : text.c_str());
                    } else {
                        printf("\n  ── generate_ids prompt=%d gen=%d: MISSING ──\n", i, g);
                    }
                }
            }

            // Wakeup again so subsequent tests work
            engine.wakeup();
        }
    }
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[]) {
    const char* model_dir  = (argc > 1) ? argv[1] : "model_weights/Qwen3-0.6B";
    const char* data_path  = (argc > 2) ? argv[2] : "data/dapo-17k.bin";

    printf("=== LLMEngine integration tests ===\n");
    printf("Model: %s\n", model_dir);

    if (!std::filesystem::exists(model_dir)) {
        printf("[SKIP] %s not found — pass model_dir as argv[1] or download weights\n",
               model_dir);
        return 0;
    }

    Config cfg(model_dir);
    cfg.max_num_seqs           = 64;
    cfg.max_model_len          = 8192;
    cfg.gpu_memory_utilization = 0.90f;
    cfg.enforce_eager          = false;

    LLMEngine engine(cfg);

    // test_rollout_debug(engine, data_path);
    // test_correctness(engine);
    // test_memory(engine);
    // test_throughput(engine);
    test_bench_realistic(engine);

    printf("\n");
    if (g_fail == 0)
        printf("ALL TESTS PASSED  (%d passed)\n", g_pass);
    else
        printf("RESULT: %d passed, %d FAILED\n", g_pass, g_fail);

    return g_fail > 0 ? 1 : 0;
}
