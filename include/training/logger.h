#pragma once
#include <string>
#include <map>
#include <vector>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include "third_party/json.hpp"

// =============================================================================
// Logger — training metrics logger with JSON Lines output.
//
// Usage:
//   Logger logger("checkpoints/run_001");    // writes to run_001/train_log.jsonl
//   logger.log(step, "loss", 2.34f);
//   logger.log(step, "lr",   1e-5f);
//   logger.commit(step);                     // flush one JSON line to disk + stdout
//
// Output format (one JSON object per line):
//   {"step": 100, "loss": 2.3400, "lr": 1.00e-05}
//   {"step": 200, "loss": 2.1200, "lr": 9.80e-06}
//
// Accumulated values (e.g. loss over logging_steps) are averaged automatically
// when you call log() multiple times for the same key before commit().
// =============================================================================

class Logger {
public:
    Logger() = default;

    // log_dir: directory where train_log.jsonl will be written.
    // If empty, output goes to stdout only.
    explicit Logger(const std::string& log_dir) {
        open(log_dir);
    }

    ~Logger() {
        if (file_) fclose(file_);
    }

    // No copy
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void open(const std::string& log_dir) {
        if (log_dir.empty()) return;
        std::filesystem::create_directories(log_dir);
        std::string path = log_dir + "/train_log.jsonl";
        file_ = fopen(path.c_str(), "a");   // append — survives restarts
        if (!file_) {
            fprintf(stderr, "[Logger] WARNING: cannot open %s for writing\n", path.c_str());
        }
        log_path_ = path;
    }

    // Accumulate a scalar for this key. Multiple calls before commit() are averaged.
    void log(int step, const std::string& key, float value) {
        auto& acc = pending_[key];
        acc.sum   += value;
        acc.count += 1;
        pending_step_ = step;
    }

    // Log multiple metrics at once.
    void log(int step, const std::map<std::string, float>& metrics) {
        for (auto& [k, v] : metrics) log(step, k, v);
    }

    // Flush pending metrics: write one JSON line to disk and print to stdout.
    // Clears the accumulator for the next window.
    void commit(int step) {
        if (pending_.empty()) return;

        nlohmann::json obj;
        obj["step"] = step;

        printf("[step %d]", step);
        for (auto& [key, acc] : pending_) {
            float val = acc.count > 0 ? acc.sum / acc.count : 0.0f;
            obj[key] = val;

            // Pretty-print: small values in scientific notation
            if (fabsf(val) < 0.01f && val != 0.0f)
                printf("  %s=%.3e", key.c_str(), val);
            else
                printf("  %s=%.4f", key.c_str(), val);
        }
        printf("\n");

        // Write JSON line
        std::string line = obj.dump() + "\n";
        if (file_) {
            fputs(line.c_str(), file_);
            fflush(file_);
        }

        // Store in history
        history_.push_back(obj);

        pending_.clear();
    }

    // Read back all logged history (for tests / post-training analysis).
    const std::vector<nlohmann::json>& history() const { return history_; }

    // Return the most recent logged value for a key, or NaN if not found.
    float last(const std::string& key) const {
        for (int i = (int)history_.size() - 1; i >= 0; i--) {
            if (history_[i].contains(key))
                return history_[i][key].get<float>();
        }
        return std::numeric_limits<float>::quiet_NaN();
    }

    const std::string& log_path() const { return log_path_; }

private:
    struct Accumulator {
        float sum   = 0.0f;
        int   count = 0;
    };

    FILE*                          file_         = nullptr;
    std::string                    log_path_;
    std::map<std::string, Accumulator> pending_; // metrics buffered since last commit
    int                            pending_step_ = 0;
    std::vector<nlohmann::json>    history_;     // all committed entries
};
