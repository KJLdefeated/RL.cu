#include "training/GRPO_trainer.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <cctype>
#include <regex>

// Extract the last \boxed{...} content from text.
// Handles nested braces: \boxed{a{b}c} → "a{b}c"
static std::string extract_boxed(const std::string& text) {
    std::string key = "\\boxed{";
    size_t pos = text.rfind(key);
    if (pos == std::string::npos) return "";

    size_t start = pos + key.size();
    int depth = 1;
    size_t i = start;
    for (; i < text.size() && depth > 0; i++) {
        if (text[i] == '{') depth++;
        else if (text[i] == '}') depth--;
    }
    if (depth != 0) return "";
    return text.substr(start, i - start - 1);  // exclude closing '}'
}

// Extract content after the last "Answer:" line.
// Looks for "Answer:" (case-insensitive colon), takes rest of line.
static std::string extract_answer_line(const std::string& text) {
    size_t pos = text.rfind("Answer:");
    if (pos == std::string::npos) {
        pos = text.rfind("answer:");
        if (pos == std::string::npos) return "";
    }
    size_t start = pos + 7;  // skip "Answer:"
    // skip whitespace after colon
    while (start < text.size() && text[start] == ' ') start++;
    size_t end = start;
    while (end < text.size() && text[end] != '\n') end++;
    // trim trailing whitespace
    while (end > start && isspace((unsigned char)text[end - 1])) end--;
    return text.substr(start, end - start);
}

// Strip LaTeX wrapper commands: \text{Yes} → Yes, \textbf{3} → 3, \mathrm{cm} → cm
static std::string strip_latex_wrappers(const std::string& s) {
    std::string r = s;
    // Repeatedly strip \cmd{...} wrappers
    const char* cmds[] = {"\\text{", "\\textbf{", "\\textit{", "\\mathrm{",
                          "\\mathbf{", "\\mathit{", "\\operatorname{"};
    for (auto cmd : cmds) {
        size_t clen = strlen(cmd);
        while (r.size() > clen && r.compare(0, clen, cmd) == 0 && r.back() == '}') {
            r = r.substr(clen, r.size() - clen - 1);
        }
    }
    return r;
}

// Normalize a math answer string for comparison:
// strip LaTeX wrappers, whitespace, $, leading/trailing punctuation
static std::string normalize_answer(const std::string& s) {
    std::string t = strip_latex_wrappers(s);
    std::string r;
    for (char c : t) {
        if (!isspace((unsigned char)c) && c != '$')
            r += c;
    }
    // Strip wrapping parens/brackets if present: "(5)" → "5"
    if (r.size() >= 2 && r.front() == '(' && r.back() == ')')
        r = r.substr(1, r.size() - 2);
    // Strip trailing period
    if (!r.empty() && r.back() == '.')
        r.pop_back();
    return r;
}

// Extract answer from completion using multiple strategies:
// 1. Try \boxed{} (LaTeX format)
// 2. Try "Answer:" line
static std::string extract_predicted_answer(const std::string& completion) {
    // Try \boxed{} first (more precise)
    std::string ans = extract_boxed(completion);
    if (!ans.empty()) return ans;

    // Fall back to "Answer:" line
    ans = extract_answer_line(completion);
    // The answer line itself might contain \boxed{}
    if (!ans.empty()) {
        std::string boxed = extract_boxed(ans);
        if (!boxed.empty()) return boxed;
    }
    return ans;
}

// Reward: extract answer from completion, compare against ground truth.
// Returns 1.0 if match, 0.0 otherwise.
static float boxed_reward(const std::string& completion, const std::string& answer) {
    std::string pred = extract_predicted_answer(completion);
    if (pred.empty()) return 0.0f;
    return (normalize_answer(pred) == normalize_answer(answer)) ? 1.0f : 0.0f;
}

int main(int argc, char** argv) {
    std::string model_dir = (argc >= 2) ? argv[1] : "model_weights/Qwen3-0.6B";
    std::string data_path = (argc >= 3) ? argv[2] : "data/deepmath-103k.jsonl";
    int num_steps = (argc >= 4) ? atoi(argv[3]) : 1;

    // --- Create LLMEngine (owns the model, auto-computes KV budget) ---
    Config engine_cfg(model_dir);
    engine_cfg.max_model_len = 5120;
    engine_cfg.enforce_eager = true;
    engine_cfg.gpu_memory_utilization = 0.9f;
    printf("[test] Creating LLMEngine from %s ...\n", model_dir.c_str());
    LLMEngine engine(engine_cfg);

    // --- Configure GRPO ---
    GRPOConfig config(data_path);
    config.batch_size         = 64;  // total seqs = num_prompts * G = 8 * 8
    config.num_generations    = 8;
    config.grad_accum_steps   = 4;
    config.max_completion_len = 4096;
    config.max_seq_len        = 5120;
    config.gen_temperature    = 0.8f;
    config.gen_top_p          = 1.0f;
    config.base_lr            = 1e-5f;
    config.total_steps        = num_steps;
    config.logging_steps      = 1;
    config.warmup_steps       = 50;
    config.save_dir           = "checkpoints/grpo_ckpt";
    config.max_grad_norm      = 1.0f;
    config.recompute();

    printf("[test] GRPOConfig: batch_size=%d  num_prompts=%d  G=%d  acc=%d\n",
           config.batch_size, config.num_prompts,
           config.num_generations, config.grad_accum_steps);

    // --- Create trainer and run ---
    {
        GRPOTrainer trainer(config, &engine, (RewardFn)boxed_reward);

        printf("[test] Starting GRPO training for %d steps ...\n", config.total_steps);
        trainer.grpo_train();
    }

    printf("[test] GRPO training test completed successfully.\n");
    return 0;
}
