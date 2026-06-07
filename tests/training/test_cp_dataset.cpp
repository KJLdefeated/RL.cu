// test_cp_dataset.cpp
// ---------------------------------------------------------------------------
// Tests the JSONL loader (include/training/cp_dataset.h) and its integration
// with CodeContestsEnv + SubprocessPythonRunner.
//
// No network: writes a synthetic JSONL to /tmp, loads it, asserts contents,
// and runs an end-to-end env reward check on one of the synthetic problems.
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "training/cp_dataset.h"
#include "training/env.h"
#include "training/code_runner.h"

static int g_pass = 0, g_fail = 0;
#define PASS(name)        do { printf("[PASS] %s\n",  (name)); g_pass++; } while (0)
#define FAIL(name, ...)   do { printf("[FAIL] %s: ",  (name)); printf(__VA_ARGS__); printf("\n"); g_fail++; } while (0)

static const char* SYNTH_JSONL = R"DOC({"id":"add","description":"sum","public_tests":[{"input":"3 5\n","expected":"8\n"}],"hidden_tests":[{"input":"1 1\n","expected":"2\n"},{"input":"100 23\n","expected":"123\n"}]}
{"id":"mul","description":"prod","public_tests":[{"input":"2 4\n","expected":"8\n"}],"hidden_tests":[{"input":"3 7\n","expected":"21\n"}]}
{"id":"no-hidden-skip","description":"x","public_tests":[{"input":"a\n","expected":"a\n"}],"hidden_tests":[]}
malformed json line {{{
{"id":"no-public-skip","description":"x","public_tests":[],"hidden_tests":[{"input":"a\n","expected":"a\n"}]}
{"id":"caps","description":"clip me","public_tests":[{"input":"x\n","expected":"x\n"},{"input":"y\n","expected":"y\n"},{"input":"z\n","expected":"z\n"}],"hidden_tests":[{"input":"x\n","expected":"x\n"},{"input":"y\n","expected":"y\n"},{"input":"z\n","expected":"z\n"},{"input":"w\n","expected":"w\n"}]}
)DOC";

int main() {
    printf("=== CP dataset loader ===\n");

    // Write the synthetic JSONL to a tmp path.
    std::string tmp = "/tmp/cp_dataset_test.jsonl";
    {
        std::ofstream of(tmp);
        of << SYNTH_JSONL;
    }

    // --- 1. Basic load, no caps -------------------------------------------
    {
        std::vector<CPDatasetEntry> es;
        int n = load_cp_jsonl(tmp, es);
        // Expected to load 3 valid (add, mul, caps); skip the 2 empty-fields rows + 1 malformed.
        if (n == 3 && es.size() == 3) PASS("load: 3 entries (malformed + empty skipped)");
        else FAIL("load: count", "got n=%d size=%zu", n, es.size());

        // First entry sanity
        if (es[0].problem.id == "add" &&
            es[0].description == "sum" &&
            es[0].problem.public_tests.size() == 1 &&
            es[0].problem.public_tests[0].input == "3 5\n" &&
            es[0].problem.hidden_tests.size() == 2 &&
            es[0].problem.hidden_tests[1].expected == "123\n")
            PASS("load: fields of 'add' problem");
        else
            FAIL("load: fields of add", "id=%s desc=%s pub=%zu hid=%zu",
                 es[0].problem.id.c_str(), es[0].description.c_str(),
                 es[0].problem.public_tests.size(), es[0].problem.hidden_tests.size());
    }

    // --- 2. max_problems cap stops loading early --------------------------
    {
        std::vector<CPDatasetEntry> es;
        int n = load_cp_jsonl(tmp, es, /*max_problems=*/2);
        if (n == 2) PASS("load: max_problems cap = 2");
        else        FAIL("load: max_problems cap", "n=%d", n);
    }

    // --- 3. per-problem test caps applied ---------------------------------
    {
        std::vector<CPDatasetEntry> es;
        load_cp_jsonl(tmp, es, /*max_problems=*/100,
                              /*max_public=*/2, /*max_hidden=*/2);
        // Find the 'caps' problem and check caps were applied.
        bool ok = false;
        for (const auto& e : es) {
            if (e.problem.id == "caps") {
                ok = (e.problem.public_tests.size() == 2) &&
                     (e.problem.hidden_tests.size() == 2);
                break;
            }
        }
        if (ok) PASS("load: per-problem public/hidden caps");
        else    FAIL("load: caps", "didn't find or didn't clip");
    }

    // --- 4. convenience load_cp_problems mirrors the entry list -----------
    {
        std::vector<CPProblem> ps = load_cp_problems(tmp);
        if ((int)ps.size() == 3 && ps[0].id == "add" && ps[1].id == "mul")
            PASS("load_cp_problems: convenience wrapper");
        else
            FAIL("load_cp_problems", "size=%zu", ps.size());
    }

    // --- 5. missing file → 0 entries, stderr warning ----------------------
    {
        std::vector<CPDatasetEntry> es;
        int n = load_cp_jsonl("/tmp/__definitely_does_not_exist__.jsonl", es);
        if (n == 0 && es.empty()) PASS("load: missing file → 0 entries");
        else                      FAIL("load missing", "n=%d", n);
    }

    // --- 6. end-to-end: load → run a correct soln through env → reward 1.0 -
    {
        auto problems = load_cp_problems(tmp);
        SubprocessPythonRunner runner;
        ResourceLimits lim;
        lim.wall_time_sec   = 3;
        lim.isolate_network = false;  // not needed for this test
        CodeContestsEnv env(problems, &runner, RewardShape::AllOrNothing, lim);

        // problems[0] is the synthetic "add" problem.
        const std::string sol_add =
            "import sys\n"
            "a, b = map(int, sys.stdin.read().split())\n"
            "print(a + b)\n";
        float r = env.reward(sol_add, /*problem_idx=*/0);
        if (r == 1.0f) PASS("e2e: load+env+runner → reward 1.0 for add solution");
        else           FAIL("e2e add", "reward=%.3f", r);

        // problems[1] is "mul"; the add solution is wrong here → reward 0.0.
        float r2 = env.reward(sol_add, /*problem_idx=*/1);
        if (r2 == 0.0f) PASS("e2e: add soln on mul problem → reward 0.0");
        else            FAIL("e2e mul-wrong", "reward=%.3f", r2);
    }

    // Cleanup
    std::filesystem::remove(tmp);

    printf("\n%s  (%d passed, %d failed)\n",
           g_fail == 0 ? "ALL PASSED" : "SOME FAILED", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
