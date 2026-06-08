#pragma once
// ============================================================================
// CP dataset loader — JSONL → std::vector<CPProblem>.
//
// Format produced by scripts/prepare_cp_data.py (one JSON object per line):
//
//   {
//     "id":           "1575_A. ...",
//     "description":  "..."  (optional, used by the trainer for the prompt),
//     "public_tests": [{"input":"...", "expected":"..."}, ...],
//     "hidden_tests": [{"input":"...", "expected":"..."}, ...]
//   }
//
// Skips malformed lines (logs to stderr) rather than throwing — long datasets
// occasionally have edge-case entries and we'd rather train on what's good.
// ============================================================================

#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "third_party/json.hpp"
#include "training/env.h"   // CPProblem, TestCase

// Description of the problem statement — used by the trainer when building the
// chat prompt for the model.  Kept separate from CPProblem itself so that
// CPProblem stays minimal and Env-focused.
struct CPDatasetEntry {
    CPProblem    problem;
    std::string  description;
};

// Load all problems from `path`. Caps:
//   max_problems     - stop after writing this many entries
//   max_public_tests - per-problem cap on public_tests kept
//   max_hidden_tests - per-problem cap on hidden_tests kept
//
// Returns the number of entries successfully loaded.  Out-vector is appended,
// not cleared.  On parse errors the line is skipped with a stderr warning.
inline int load_cp_jsonl(const std::string& path,
                         std::vector<CPDatasetEntry>& out,
                         int max_problems     = std::numeric_limits<int>::max(),
                         int max_public_tests = std::numeric_limits<int>::max(),
                         int max_hidden_tests = std::numeric_limits<int>::max())
{
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[cp_dataset] ERROR: cannot open %s\n", path.c_str());
        return 0;
    }
    int started = (int)out.size();
    int line_no = 0, parse_errs = 0, skipped_empty = 0;
    std::string line;
    while (std::getline(f, line) && (int)(out.size() - started) < max_problems) {
        line_no++;
        if (line.empty()) continue;
        try {
            auto j = nlohmann::json::parse(line);
            CPDatasetEntry e;
            e.problem.id   = j.value("id", std::string{});
            e.description  = j.value("description", std::string{});

            auto take_tests = [](const nlohmann::json& arr, int cap) {
                std::vector<TestCase> v;
                if (!arr.is_array()) return v;
                int n = std::min((int)arr.size(), cap);
                v.reserve(n);
                for (int i = 0; i < n; i++) {
                    const auto& t = arr[i];
                    TestCase tc;
                    tc.input    = t.value("input",    std::string{});
                    tc.expected = t.value("expected", std::string{});
                    v.push_back(std::move(tc));
                }
                return v;
            };

            e.problem.public_tests = take_tests(j.value("public_tests", nlohmann::json::array()),
                                                max_public_tests);
            e.problem.hidden_tests = take_tests(j.value("hidden_tests", nlohmann::json::array()),
                                                max_hidden_tests);

            if (e.problem.public_tests.empty() || e.problem.hidden_tests.empty()) {
                skipped_empty++;
                continue;
            }
            out.push_back(std::move(e));
        } catch (const std::exception& ex) {
            parse_errs++;
            if (parse_errs <= 3) {
                fprintf(stderr, "[cp_dataset] WARN: parse error at line %d: %s\n",
                        line_no, ex.what());
            } else if (parse_errs == 4) {
                fprintf(stderr, "[cp_dataset] WARN: further parse errors suppressed\n");
            }
        }
    }
    int loaded = (int)out.size() - started;
    fprintf(stderr, "[cp_dataset] %s: loaded %d "
                    "(parse_errs=%d skipped_empty=%d) from %d lines\n",
            path.c_str(), loaded, parse_errs, skipped_empty, line_no);
    return loaded;
}

// Convenience: extract just the CPProblem list (the usual env input).
inline std::vector<CPProblem> load_cp_problems(const std::string& path,
                                               int max_problems = std::numeric_limits<int>::max(),
                                               int max_public_tests = std::numeric_limits<int>::max(),
                                               int max_hidden_tests = std::numeric_limits<int>::max())
{
    std::vector<CPDatasetEntry> entries;
    load_cp_jsonl(path, entries, max_problems, max_public_tests, max_hidden_tests);
    std::vector<CPProblem> ps;
    ps.reserve(entries.size());
    for (auto& e : entries) ps.push_back(std::move(e.problem));
    return ps;
}
