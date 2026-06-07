// test_rollout_engine.cu
// ---------------------------------------------------------------------------
// End-to-end test of the agentic multi-turn rollout API on LLMEngine
// (docs/AGENTIC_ROLLOUT_SPEC.md §2.1–2.3): rollout_add → rollout_run_turns →
// rollout_continue → rollout_run_turns → rollout_finish, with cross-turn KV
// reuse via the continued-prefill path wired into qwen3_prefill (§2.4).
//
// Correctness checks (greedy, deterministic):
//   T1)  Multi-turn turn-1 tokens  ==  single-shot generate(P, G1)
//          → proves the fresh-prefill + decode path is unchanged.
//   T2)  Multi-turn turn-2 first token  ==  single-shot generate(P+G1+O, G2)[0]
//          → proves the continued-prefill resume produced a hidden state
//            equivalent to a single full prefill of the concatenated context.
//
// T2 only requires the first-token argmax to match: continued prefill differs
// from a dense full prefill by ~paged-vs-WMMA numerical noise (~0.025 on
// logits) which never flips the argmax on confident next-tokens but can drift
// after many steps. The first-token match is the load-bearing property — the
// rest of T2 is reported as overlap for diagnostics.
//
// Requires weights at model_weights/Qwen3-0.6B (argv[1] override).
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <filesystem>
#include "engine/llm_engine.h"

static int g_pass = 0, g_fail = 0;
#define PASS(name)        do { printf("[PASS] %s\n",  (name)); g_pass++; } while (0)
#define FAIL(name, ...)   do { printf("[FAIL] %s: ",  (name)); printf(__VA_ARGS__); printf("\n"); g_fail++; } while (0)

static std::vector<int64_t> make_ids(int n, uint32_t seed, int vocab_cap = 10000) {
    std::vector<int64_t> v(n);
    uint32_t s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        v[i] = (int64_t)((s >> 9) % (uint32_t)vocab_cap);
    }
    return v;
}

static std::string ids_to_str(const std::vector<int64_t>& v, int max = 10) {
    std::string s = "[";
    int n = std::min((int)v.size(), max);
    for (int i = 0; i < n; i++) {
        s += std::to_string(v[i]);
        if (i + 1 < n) s += ",";
    }
    if ((int)v.size() > max) s += ",…";
    s += "]";
    return s;
}

int main(int argc, char** argv) {
    const char* model_dir = (argc > 1) ? argv[1] : "model_weights/Qwen3-0.6B";
    printf("=== Agentic rollout engine — end-to-end ===\nModel: %s\n", model_dir);
    if (!std::filesystem::exists(model_dir)) {
        printf("[SKIP] %s not found\n", model_dir);
        return 0;
    }

    Config cfg(model_dir);
    cfg.max_num_seqs           = 4;
    cfg.max_model_len          = 1024;
    cfg.gpu_memory_utilization = 0.30f;     // small footprint for the test
    cfg.enforce_eager          = true;      // skip CUDA graphs to keep paths simple
    LLMEngine engine(cfg);

    // Test prompt / observation sizes (small, all in-vocab).
    const int n_p = 24, G1 = 12, n_o = 6, G2 = 12;
    auto P = make_ids(n_p,  7u);
    auto O = make_ids(n_o, 99u);
    printf("prompt=%s  obs=%s  (n_p=%d G1=%d n_o=%d G2=%d)\n",
           ids_to_str(P).c_str(), ids_to_str(O).c_str(), n_p, G1, n_o, G2);

    SamplingParams sp_greedy;
    sp_greedy.temperature    = 0.0f;
    sp_greedy.do_sample      = false;
    sp_greedy.ignore_eos     = true;     // deterministic length

    // ── REF turn 1: single-shot generate(P, G1) ────────────────────────────
    sp_greedy.max_new_tokens = G1;
    std::vector<int64_t> ref_t1;
    {
        auto m = engine.generate_ids({P}, sp_greedy, /*num_generations=*/1);
        // generate_ids keys by 0..total_seqs-1
        ref_t1 = m.empty() ? std::vector<int64_t>{} : m.begin()->second;
    }

    // ── REF turn 2: single-shot generate(P+t1+O, G2) ───────────────────────
    std::vector<int64_t> ref_ctx = P;
    ref_ctx.insert(ref_ctx.end(), ref_t1.begin(), ref_t1.end());
    ref_ctx.insert(ref_ctx.end(), O.begin(),     O.end());
    sp_greedy.max_new_tokens = G2;
    std::vector<int64_t> ref_t2;
    {
        auto m = engine.generate_ids({ref_ctx}, sp_greedy, 1);
        ref_t2 = m.empty() ? std::vector<int64_t>{} : m.begin()->second;
    }
    printf("ref turn1=%s  ref turn2=%s\n",
           ids_to_str(ref_t1).c_str(), ids_to_str(ref_t2).c_str());

    // ── MULTI-TURN rollout via the agentic API ─────────────────────────────
    // Turn 1.
    SamplingParams sp_turn = sp_greedy;
    sp_turn.max_new_tokens  = G1;
    int64_t sid = engine.rollout_add(P, sp_turn);
    auto outs1 = engine.rollout_run_turns();
    if (outs1.find(sid) == outs1.end()) {
        FAIL("rollout turn1 produced output", "no entry for sid=%lld", (long long)sid);
        engine.rollout_finish(sid);
        printf("\n%d passed, %d failed\n", g_pass, g_fail);
        return g_fail > 0 ? 1 : 0;
    }
    auto t1 = outs1[sid].new_tokens;

    // Resume with observation.
    engine.rollout_continue(sid, O);
    // Turn 2: same sp_turn budget, but we need to re-set the engine's per-turn
    // limit — rollout_continue resets num_prompt_tokens; the sp on the seq
    // still carries G1's max_new_tokens. Force G2 by reassigning the seq's sp
    // is not exposed; instead, generate up to G2 by stopping when we have G2
    // tokens. Simpler: update the sequence's sp via a friend hook is overkill.
    // Workaround: re-add with sp_turn.max_new_tokens=G2? No — same seq.
    // Cleanest: the seq's sp stays G1 (which is what we want here, since
    // G1==G2 in our test). For the general case the spec calls for per-turn
    // sp; that's a future API knob.
    auto outs2 = engine.rollout_run_turns();
    auto t2 = (outs2.find(sid) == outs2.end()) ? std::vector<int64_t>{}
                                               : outs2[sid].new_tokens;
    engine.rollout_finish(sid);

    printf("rollout turn1=%s  rollout turn2=%s\n",
           ids_to_str(t1).c_str(), ids_to_str(t2).c_str());

    // ── T1: turn-1 tokens must exactly match the single-shot reference ────
    if (t1 == ref_t1) PASS("T1: rollout turn-1 tokens == generate(P, G1)");
    else              FAIL("T1: rollout turn-1 == ref turn-1",
                           "len got=%zu ref=%zu", t1.size(), ref_t1.size());

    // ── T2: turn-2 first token must match single-shot reference's first ───
    if (!t2.empty() && !ref_t2.empty() && t2[0] == ref_t2[0])
        PASS("T2: rollout turn-2 first-token == generate(P+t1+O, G2)[0]");
    else
        FAIL("T2: turn-2 first-token argmax",
             "got=%lld ref=%lld",
             t2.empty()    ? (int64_t)-1 : t2[0],
             ref_t2.empty()? (int64_t)-1 : ref_t2[0]);

    // Diagnostic: how many leading tokens overlap?
    int overlap = 0;
    for (size_t i = 0; i < std::min(t2.size(), ref_t2.size()); i++) {
        if (t2[i] == ref_t2[i]) overlap++; else break;
    }
    printf("[diag] turn-2 leading-token overlap: %d / %d (rollout=%zu, ref=%zu)\n",
           overlap, (int)std::min(t2.size(), ref_t2.size()),
           t2.size(), ref_t2.size());

    printf("\n%s  (%d passed, %d failed)\n",
           g_fail == 0 ? "ALL PASSED" : "SOME FAILED", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
