// test_cp_env.cpp
// ----------------------------------------------------------------------------
// Validates the agentic CP environment + sandboxed Python runner
// (docs/AGENTIC_ROLLOUT_SPEC.md §3.2–3.4).
//
// Tests:
//   1. extract_code parses fenced markdown blocks (incl. ```py, untagged).
//   2. SubprocessPythonRunner.run() executes correct code → exit_code=0,
//      stdout matches.
//   3. Timeout: infinite loop is killed by wall-time, status=Timeout, parent
//      doesn't hang.
//   4. Memory limit: oversized allocation gets killed (MemoryLimit signal).
//   5. Network isolation (best effort): code attempting to open a TCP socket
//      fails or hangs-then-timeouts inside the sandbox.  Soft-skipped if
//      unprivileged user namespaces are disabled on this host.
//   6. CodeContestsEnv:
//        - correct solution        → reward 1.0 in both AllOrNothing &
//          FractionPassed; observation says "PASS".
//        - intentionally wrong soln → reward 0.0 (AllOrNothing),
//          fraction < 1.0 (FractionPassed), observation says "FAIL".
//        - timing-out solution     → reward 0.0; observation status flagged.
// ----------------------------------------------------------------------------

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "training/env.h"

static int g_pass = 0, g_fail = 0;
#define PASS(name)        do { printf("[PASS] %s\n",  (name)); g_pass++; } while (0)
#define FAIL(name, ...)   do { printf("[FAIL] %s: ",  (name)); printf(__VA_ARGS__); printf("\n"); g_fail++; } while (0)

static bool contains(const std::string& h, const std::string& n) {
    return h.find(n) != std::string::npos;
}

int main() {
    printf("=== CP env + sandboxed runner ===\n");

    SubprocessPythonRunner runner("python3");

    // ─── 1. extract_code ─────────────────────────────────────────────────────
    {
        std::string t1 = "Sure!\n```python\nprint('hi')\n```\nDone.";
        std::string t2 = "before\n```py\nx = 1\nprint(x)\n```";
        std::string t3 = "no code here";
        std::string t4 = "before\n```cpp\nint main(){}\n```\n```python\nprint('y')\n```";
        std::string t5 = "untagged:\n```\nprint('z')\n```";

        if (extract_code(t1) == "print('hi')")           PASS("extract: ```python`");
        else FAIL("extract: ```python```", "got [%s]", extract_code(t1).c_str());
        if (extract_code(t2) == "x = 1\nprint(x)")       PASS("extract: ```py```");
        else FAIL("extract: ```py```", "got [%s]", extract_code(t2).c_str());
        if (extract_code(t3) == "")                      PASS("extract: no fence → empty");
        else FAIL("extract: no fence", "got [%s]", extract_code(t3).c_str());
        if (extract_code(t4) == "print('y')")            PASS("extract: skips ```cpp```");
        else FAIL("extract: skips ```cpp```", "got [%s]", extract_code(t4).c_str());
        if (extract_code(t5) == "print('z')")            PASS("extract: untagged fence");
        else FAIL("extract: untagged fence", "got [%s]", extract_code(t5).c_str());
    }

    // ─── 2. Runner: correct code ─────────────────────────────────────────────
    {
        ResourceLimits lim;
        lim.wall_time_sec = 3;
        auto r = runner.run("import sys; a,b=map(int,sys.stdin.read().split()); print(a+b)",
                            "3 5\n", lim);
        if (r.status == RunResult::Status::Ok && r.stdout_str.find("8") != std::string::npos)
            PASS("runner: 3+5 → 8 (exit 0)");
        else
            FAIL("runner: 3+5", "status=%d exit=%d stdout=[%s] stderr=[%s]",
                 (int)r.status, r.exit_code, r.stdout_str.c_str(), r.stderr_str.c_str());
    }

    // ─── 3. Runner: timeout ──────────────────────────────────────────────────
    {
        ResourceLimits lim;
        lim.wall_time_sec = 1;
        lim.cpu_time_sec  = 2;
        auto t0 = std::chrono::steady_clock::now();
        auto r = runner.run("while True: pass", "", lim);
        double waited = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        if (r.status == RunResult::Status::Timeout && waited < 3.0)
            PASS("runner: infinite loop → Timeout, parent didn't hang");
        else
            FAIL("runner: timeout", "status=%d waited=%.2fs", (int)r.status, waited);
    }

    // ─── 4. Runner: memory limit ────────────────────────────────────────────
    {
        ResourceLimits lim;
        lim.wall_time_sec = 3;
        lim.memory_mb     = 64;       // small cap
        // Build a list-of-zeros that should blow past 64 MB.
        auto r = runner.run("x = bytearray(256 * 1024 * 1024)", "", lim);
        // RLIMIT_AS makes the allocation fail → Python raises MemoryError →
        // non-zero exit *or* SIGKILL/SIGSEGV depending on overcommit settings.
        bool killed = (r.status == RunResult::Status::MemoryLimit ||
                       r.status == RunResult::Status::NonZeroExit);
        if (killed)
            PASS("runner: 256MB alloc under 64MB cap → blocked");
        else
            FAIL("runner: memory limit", "status=%d stderr=[%s]",
                 (int)r.status, r.stderr_str.c_str());
    }

    // ─── 5. Runner: network isolation (best effort, soft-skip if userns off) ─
    {
        ResourceLimits lim;
        lim.wall_time_sec   = 3;
        lim.isolate_network = true;
        // Try to open a TCP socket to 1.1.1.1:443.  With CLONE_NEWNET we
        // expect socket() or connect() to fail (no interfaces, no route);
        // without it we may actually connect.
        const char* code =
            "import socket\n"
            "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "s.settimeout(1.5)\n"
            "try:\n"
            "    s.connect(('1.1.1.1', 443))\n"
            "    print('CONNECTED')\n"
            "except Exception as e:\n"
            "    print('BLOCKED', type(e).__name__)\n";
        auto r = runner.run(code, "", lim);
        bool blocked = contains(r.stdout_str, "BLOCKED");
        bool userns_disabled = contains(r.stderr_str, "WARN unshare(NEWUSER)");
        if (blocked && !contains(r.stdout_str, "CONNECTED"))
            PASS("runner: network isolated (socket failed)");
        else if (userns_disabled)
            printf("[SKIP] network isolation: unprivileged user namespaces "
                   "disabled on this host (stderr: %s)\n", r.stderr_str.c_str());
        else
            FAIL("runner: network isolation",
                 "stdout=[%s] stderr=[%s]", r.stdout_str.c_str(), r.stderr_str.c_str());
    }

    // ─── 6. CodeContestsEnv: end-to-end ─────────────────────────────────────
    {
        CPProblem p;
        p.id = "add-two-numbers";
        p.public_tests = {
            {"3 5\n",    "8\n"},
            {"10 -2\n",  "8\n"},
        };
        p.hidden_tests = {
            {"1 1\n",    "2\n"},
            {"100 23\n", "123\n"},
            {"-5 5\n",   "0\n"},
        };
        std::vector<CPProblem> problems = {p};

        ResourceLimits lim;
        lim.wall_time_sec   = 3;
        lim.isolate_network = false;  // env tests don't need network checks here

        const std::string sol_correct =
            "import sys\n"
            "a, b = map(int, sys.stdin.read().split())\n"
            "print(a + b)\n";
        // Wrong: subtracts instead of adds.  Will pass test {1,1}->2 by
        // coincidence? No: 1-1=0 ≠ 2.  Passes {-5,5}->0 by coincidence:
        // -5-5=-10 ≠ 0.  So 0/3 hidden pass.
        const std::string sol_wrong =
            "import sys\n"
            "a, b = map(int, sys.stdin.read().split())\n"
            "print(a - b)\n";
        const std::string sol_timeout =
            "while True: pass\n";

        // 6a. AllOrNothing — correct soln → 1.0
        {
            CodeContestsEnv env(problems, &runner, RewardShape::AllOrNothing, lim);
            float r = env.reward(sol_correct, 0);
            if (r == 1.0f) PASS("env: correct → AllOrNothing reward 1.0");
            else           FAIL("env: correct AON", "reward=%.3f", r);
        }
        // 6b. AllOrNothing — wrong soln → 0.0
        {
            CodeContestsEnv env(problems, &runner, RewardShape::AllOrNothing, lim);
            float r = env.reward(sol_wrong, 0);
            if (r == 0.0f) PASS("env: wrong → AllOrNothing reward 0.0");
            else           FAIL("env: wrong AON", "reward=%.3f", r);
        }
        // 6c. FractionPassed — correct → 1.0
        {
            CodeContestsEnv env(problems, &runner, RewardShape::FractionPassed, lim);
            float r = env.reward(sol_correct, 0);
            if (r == 1.0f) PASS("env: correct → FractionPassed 1.0");
            else           FAIL("env: correct FP", "reward=%.3f", r);
        }
        // 6d. FractionPassed — wrong → 0.0 (no hidden test accidentally matches)
        {
            CodeContestsEnv env(problems, &runner, RewardShape::FractionPassed, lim);
            float r = env.reward(sol_wrong, 0);
            if (r < 0.5f) PASS("env: wrong → FractionPassed < 0.5");
            else          FAIL("env: wrong FP", "reward=%.3f", r);
        }
        // 6e. Timeout → 0 (no hidden test can pass an inf-looping prog)
        {
            ResourceLimits short_lim = lim;
            short_lim.wall_time_sec = 1;
            CodeContestsEnv env(problems, &runner, RewardShape::FractionPassed, short_lim);
            float r = env.reward(sol_timeout, 0);
            if (r == 0.0f) PASS("env: timeout solution → reward 0.0");
            else           FAIL("env: timeout reward", "reward=%.3f", r);
        }
        // 6f. feedback() observation contains pass/fail and stdout
        {
            CodeContestsEnv env(problems, &runner, RewardShape::AllOrNothing, lim);
            auto fb_ok  = env.feedback(sol_correct, 0);
            auto fb_bad = env.feedback(sol_wrong,   0);
            bool ok = contains(fb_ok.observation, "PASS") &&
                      contains(fb_ok.observation, "sample test");
            if (ok) PASS("env: feedback formats sample test result (correct)");
            else    FAIL("env: feedback correct fmt", "obs=[%s]",
                         fb_ok.observation.c_str());
            bool bad = contains(fb_bad.observation, "FAIL");
            if (bad) PASS("env: feedback formats sample test result (wrong)");
            else     FAIL("env: feedback wrong fmt", "obs=[%s]",
                          fb_bad.observation.c_str());
        }
    }

    printf("\n%s  (%d passed, %d failed)\n",
           g_fail == 0 ? "ALL PASSED" : "SOME FAILED", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
