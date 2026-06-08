#pragma once
// ============================================================================
// Multi-turn agentic environment for execution-grounded RL
// (docs/AGENTIC_ROLLOUT_SPEC.md §3.2).
//
//   Env::feedback(code, idx)  — runs code on PUBLIC tests; returns an
//                                observation string for the model to read next
//                                turn.  No grading.
//   Env::reward (code, idx)   — runs code on HIDDEN tests; returns a scalar
//                                reward for the GRPO trainer.
//
// Public-feedback / hidden-reward split mirrors real CP and prevents the model
// from memorizing the visible cases.
//
// The runtime is delegated to a CodeRunner (training/code_runner.h); the env
// only encodes problem data, judge logic, and the observation/reward shapes.
// ============================================================================

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "training/code_runner.h"

struct TestCase {
    std::string input;
    std::string expected;   // expected stdout (compared with whitespace-normalized equality)
};

struct CPProblem {
    std::string             id;
    std::vector<TestCase>   public_tests;
    std::vector<TestCase>   hidden_tests;
};

struct EnvFeedback {
    std::string observation;
    bool        stop = false;   // true → trainer should end the episode early
};

enum class RewardShape {
    AllOrNothing,    // 1.0 iff every hidden test passes, else 0.0
    FractionPassed   // (#passed / #total) — denser gradient, partial-credit gameable
};

class Env {
public:
    virtual ~Env() = default;
    virtual EnvFeedback feedback(const std::string& code, int problem_idx) = 0;
    virtual float       reward  (const std::string& code, int problem_idx) = 0;
};

// ----------------------------------------------------------------------------
// extract_code — pull the first fenced Python block out of model text.
//
//   ```python\n<body>\n```        →  <body>
//   ```py\n<body>\n```            →  <body>
//   ```\n<body>\n```              →  <body>   (un-tagged fence — accepted too,
//                                              since models often omit the tag)
//
// Returns "" if no fenced block is found.  The trailing newline before ``` is
// stripped; interior whitespace is preserved.
// ----------------------------------------------------------------------------
inline std::string extract_code(const std::string& text) {
    const std::string fence = "```";
    size_t i = 0;
    while (true) {
        size_t open = text.find(fence, i);
        if (open == std::string::npos) return "";
        // Skip an optional language tag up to the next newline.
        size_t after = open + fence.size();
        size_t nl    = text.find('\n', after);
        if (nl == std::string::npos) return "";
        // Accept "```", "```python", "```py" (case-insensitive).
        std::string tag = text.substr(after, nl - after);
        while (!tag.empty() && (tag.back() == ' ' || tag.back() == '\r' || tag.back() == '\t'))
            tag.pop_back();
        std::transform(tag.begin(), tag.end(), tag.begin(), ::tolower);

        // Find this block's closing fence.
        size_t body  = nl + 1;
        size_t close = text.find(fence, body);
        if (close == std::string::npos) return "";

        if (tag.empty() || tag == "python" || tag == "py" || tag == "python3") {
            std::string body_str = text.substr(body, close - body);
            while (!body_str.empty() &&
                   (body_str.back() == '\n' || body_str.back() == '\r'))
                body_str.pop_back();
            return body_str;
        }
        // Non-Python fence (e.g. ```cpp): skip past THIS block's closing fence
        // before searching for the next opener — don't mistake the close for an
        // untagged opener.
        i = close + fence.size();
    }
}

namespace cp_env_detail {

// Normalize a piece of stdout for output equality: trim trailing whitespace per
// line, strip trailing blank lines.  Matches what most CP judges do.
inline std::string normalize_output(const std::string& s) {
    std::string out; out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find('\n', i);
        if (j == std::string::npos) j = s.size();
        // Trim trailing whitespace on this line.
        size_t end = j;
        while (end > i && (s[end - 1] == ' ' || s[end - 1] == '\t' || s[end - 1] == '\r'))
            --end;
        out.append(s, i, end - i);
        if (j < s.size()) out.push_back('\n');
        i = j + 1;
    }
    // Strip trailing newlines.
    while (!out.empty() && out.back() == '\n') out.pop_back();
    return out;
}

inline std::string status_tag(RunResult::Status s) {
    switch (s) {
        case RunResult::Status::Ok:           return "OK";
        case RunResult::Status::NonZeroExit:  return "RUNTIME_ERROR";
        case RunResult::Status::Timeout:      return "TIME_LIMIT";
        case RunResult::Status::MemoryLimit:  return "MEMORY_LIMIT";
        case RunResult::Status::Signal:       return "SIGNAL";
        case RunResult::Status::SpawnFailed:  return "SPAWN_FAILED";
    }
    return "?";
}

// Bound a string to first N chars + "..." marker so observations stay compact.
inline std::string clip(const std::string& s, size_t n) {
    if (s.size() <= n) return s;
    return s.substr(0, n) + "\n[...truncated]";
}

}  // namespace cp_env_detail

// ----------------------------------------------------------------------------
// CodeContestsEnv — IO-based judge using public/hidden tests
// ----------------------------------------------------------------------------
class CodeContestsEnv : public Env {
public:
    CodeContestsEnv(std::vector<CPProblem> problems,
                    CodeRunner* runner,
                    RewardShape shape = RewardShape::AllOrNothing,
                    ResourceLimits limits = {},
                    int public_show = 1)
        : problems_(std::move(problems)),
          runner_(runner),
          shape_(shape),
          limits_(limits),
          public_show_(public_show) {}

    // Observation for the next turn: run code against (up to) the first
    // `public_show` public tests and format pass/fail + a small slice of
    // stdout/stderr/expected for the model to inspect.
    EnvFeedback feedback(const std::string& code, int problem_idx) override {
        EnvFeedback fb;
        if (problem_idx < 0 || problem_idx >= (int)problems_.size()) {
            fb.observation = "ENV_ERROR: invalid problem index\n";
            fb.stop = true;
            return fb;
        }
        const CPProblem& p = problems_[problem_idx];
        if (p.public_tests.empty()) {
            fb.observation = "(no public tests for this problem)\n";
            return fb;
        }

        std::ostringstream os;
        int show = std::min(public_show_, (int)p.public_tests.size());
        int passed = 0;
        bool any_pass = false;
        for (int t = 0; t < show; ++t) {
            const TestCase& tc = p.public_tests[t];
            RunResult r = runner_->run(code, tc.input, limits_);
            std::string got = cp_env_detail::normalize_output(r.stdout_str);
            std::string exp = cp_env_detail::normalize_output(tc.expected);
            bool ok = (r.status == RunResult::Status::Ok) && (got == exp);
            if (ok) { ++passed; any_pass = true; }
            os << "=== sample test " << (t + 1) << "/" << show
               << " [" << (ok ? "PASS" : "FAIL")
               << " | " << cp_env_detail::status_tag(r.status)
               << " | " << (int)(r.wall_time_s * 1000) << " ms]\n"
               << "input:\n"     << cp_env_detail::clip(tc.input,   512)  << "\n"
               << "expected:\n"  << cp_env_detail::clip(exp,        512)  << "\n"
               << "got stdout:\n"<< cp_env_detail::clip(got,        512)  << "\n";
            if (!r.stderr_str.empty())
                os << "stderr:\n" << cp_env_detail::clip(r.stderr_str, 512) << "\n";
        }
        os << "(public: " << passed << "/" << show << " passed)\n";
        fb.observation = os.str();
        // Early-stop: if the first public test crashed with SPAWN_FAILED, the
        // env itself is broken; bail rather than waste turns.
        fb.stop = !any_pass && (show > 0) &&
                  /* heuristic: any spawn failure → abort episode */
                  false;
        return fb;
    }

    // Final-episode reward: judge `code` against HIDDEN tests.
    float reward(const std::string& code, int problem_idx) override {
        if (problem_idx < 0 || problem_idx >= (int)problems_.size()) return 0.0f;
        const CPProblem& p = problems_[problem_idx];
        if (p.hidden_tests.empty()) return 0.0f;
        int passed = 0;
        for (const TestCase& tc : p.hidden_tests) {
            RunResult r = runner_->run(code, tc.input, limits_);
            if (r.status != RunResult::Status::Ok) continue;
            std::string got = cp_env_detail::normalize_output(r.stdout_str);
            std::string exp = cp_env_detail::normalize_output(tc.expected);
            if (got == exp) ++passed;
        }
        int n = (int)p.hidden_tests.size();
        switch (shape_) {
            case RewardShape::AllOrNothing:  return (passed == n) ? 1.0f : 0.0f;
            case RewardShape::FractionPassed:return (float)passed / (float)n;
        }
        return 0.0f;
    }

    // Direct access for tests / trainer instrumentation.
    int num_problems() const { return (int)problems_.size(); }
    const CPProblem& problem(int i) const { return problems_[i]; }

private:
    std::vector<CPProblem> problems_;
    CodeRunner*            runner_;
    RewardShape            shape_;
    ResourceLimits         limits_;
    int                    public_show_;
};
