// train_grpo.cu
//
// GRPO training script for Qwen3-0.6B.
//
// Usage:
//   ./build/train_grpo [options]   (see --help)
//
// Default: 100 steps, 8 prompts x 8 generations, DeepMath-103K.

#include "training/GRPO_trainer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>

// ============================================================
// Reward helpers
// ============================================================

// Extract the last \boxed{...} content from text.
// Handles nested braces: \boxed{a{b}c} -> "a{b}c"
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
    return text.substr(start, i - start - 1);
}

// Extract content after the last "Answer:" line.
static std::string extract_answer_line(const std::string& text) {
    size_t pos = text.rfind("Answer:");
    if (pos == std::string::npos) {
        pos = text.rfind("answer:");
        if (pos == std::string::npos) return "";
    }
    size_t start = pos + 7;
    while (start < text.size() && text[start] == ' ') start++;
    size_t end = start;
    while (end < text.size() && text[end] != '\n') end++;
    while (end > start && isspace((unsigned char)text[end - 1])) end--;
    return text.substr(start, end - start);
}

// Strip LaTeX wrapper commands: \text{Yes} -> Yes, \textbf{3} -> 3
static std::string strip_latex_wrappers(const std::string& s) {
    std::string r = s;
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

// Normalize a math answer string for comparison.
static std::string normalize_answer(const std::string& s) {
    std::string t = strip_latex_wrappers(s);
    std::string r;
    for (char c : t) {
        if (!isspace((unsigned char)c) && c != '$')
            r += c;
    }
    if (r.size() >= 2 && r.front() == '(' && r.back() == ')')
        r = r.substr(1, r.size() - 2);
    if (!r.empty() && r.back() == '.')
        r.pop_back();
    return r;
}

// Extract answer from completion: try \boxed{} first, then "Answer:" line.
static std::string extract_predicted_answer(const std::string& completion) {
    std::string ans = extract_boxed(completion);
    if (!ans.empty()) return ans;

    ans = extract_answer_line(completion);
    if (!ans.empty()) {
        std::string boxed = extract_boxed(ans);
        if (!boxed.empty()) return boxed;
    }
    return ans;
}

// Reward: 1.0 if predicted answer matches ground truth, 0.0 otherwise.
static float boxed_reward(const std::string& completion, const std::string& answer) {
    std::string pred = extract_predicted_answer(completion);
    if (pred.empty()) return 0.0f;
    return (normalize_answer(pred) == normalize_answer(answer)) ? 1.0f : 0.0f;
}

// ============================================================
// Argument parsing
// ============================================================

static void print_usage(const char* prog) {
    printf(
        "Usage: %s [options]\n"
        "\nModel / data:\n"
        "  --model       DIR   model weights dir         (default: model_weights/Qwen3-0.6B)\n"
        "  --data        FILE  JSONL dataset              (default: data/deepmath-103k.jsonl)\n"
        "\nGRPO:\n"
        "  --batch-size  N     total seqs per step        (default: 64 = 8 prompts x 8 gens)\n"
        "  --num-gens    N     completions per prompt     (default: 8)\n"
        "  --accum       N     gradient accum steps       (default: 4)\n"
        "  --max-comp    N     max completion tokens      (default: 4096)\n"
        "  --max-seq     N     max total seq length       (default: 5120)\n"
        "  --temperature F     sampling temperature       (default: 0.8)\n"
        "  --top-p       F     nucleus sampling p         (default: 1.0)\n"
        "  --top-k       N     top-k sampling             (default: 0 = off)\n"
        "\nOptimizer:\n"
        "  --lr          F     learning rate              (default: 1e-6)\n"
        "  --min-lr      F     min LR after cosine decay  (default: 0)\n"
        "  --warmup      N     warmup steps               (default: 50)\n"
        "  --total-steps N     total optimizer steps      (default: 100)\n"
        "  --schedule    TYPE  cosine | constant          (default: constant)\n"
        "  --grad-clip   F     max gradient norm          (default: 1.0)\n"
        "\nEngine:\n"
        "  --max-model-len N   engine max model length    (default: 5120)\n"
        "  --gpu-util    F     GPU memory utilization     (default: 0.9)\n"
        "\nLogging / checkpointing:\n"
        "  --save-dir    DIR   checkpoint + log dir       (default: auto-generated with date)\n"
        "  --log-steps   N     log every N steps          (default: 1)\n"
        "  --save-steps  N     checkpoint every N steps   (default: 0 = off)\n"
        , prog);
}

static std::string make_default_save_dir() {
    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    struct tm tm;
    localtime_r(&tt, &tm);
    char buf[64];
    snprintf(buf, sizeof(buf), "checkpoints/grpo_%04d%02d%02d_%02d%02d%02d",
             tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec);
    return std::string(buf);
}

int main(int argc, char** argv) {
    // ── Defaults ──────────────────────────────────────────────────────────────
    std::string model_dir       = "model_weights/Qwen3-0.6B";
    std::string data_path       = "data/deepmath-103k.jsonl";
    int   batch_size            = 64;
    int   num_generations       = 8;
    int   grad_accum            = 4;
    int   max_completion_len    = 4096;
    int   max_seq_len           = 5120;
    float gen_temperature       = 0.8f;
    float gen_top_p             = 1.0f;
    int   gen_top_k             = 0;
    float base_lr               = 1e-6f;
    float min_lr                = 0.0f;
    int   warmup_steps          = 50;
    int   total_steps           = 100;
    float max_grad_norm         = 1.0f;
    LRScheduleType schedule     = LRScheduleType::Constant;
    int   max_model_len         = 5120;
    float gpu_util              = 0.9f;
    std::string save_dir        = "";  // empty = auto-generate
    int   log_steps             = 1;
    int   save_steps            = 0;

    // ── Parse args ────────────────────────────────────────────────────────────
    for (int i = 1; i < argc; i++) {
        auto eq   = [&](const char* f) { return strcmp(argv[i], f) == 0; };
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "ERROR: missing value for %s\n", argv[i]); exit(1);
            }
            return argv[++i];
        };

        if      (eq("--help") || eq("-h"))  { print_usage(argv[0]); return 0; }
        else if (eq("--model"))             model_dir          = next();
        else if (eq("--data"))              data_path          = next();
        else if (eq("--batch-size"))        batch_size         = atoi(next());
        else if (eq("--num-gens"))          num_generations    = atoi(next());
        else if (eq("--accum"))             grad_accum         = atoi(next());
        else if (eq("--max-comp"))          max_completion_len = atoi(next());
        else if (eq("--max-seq"))           max_seq_len        = atoi(next());
        else if (eq("--temperature"))       gen_temperature    = (float)atof(next());
        else if (eq("--top-p"))             gen_top_p          = (float)atof(next());
        else if (eq("--top-k"))             gen_top_k          = atoi(next());
        else if (eq("--lr"))                base_lr            = (float)atof(next());
        else if (eq("--min-lr"))            min_lr             = (float)atof(next());
        else if (eq("--warmup"))            warmup_steps       = atoi(next());
        else if (eq("--total-steps"))       total_steps        = atoi(next());
        else if (eq("--grad-clip"))         max_grad_norm      = (float)atof(next());
        else if (eq("--max-model-len"))     max_model_len      = atoi(next());
        else if (eq("--gpu-util"))          gpu_util           = (float)atof(next());
        else if (eq("--save-dir"))          save_dir           = next();
        else if (eq("--log-steps"))         log_steps          = atoi(next());
        else if (eq("--save-steps"))        save_steps         = atoi(next());
        else if (eq("--schedule")) {
            const char* s = next();
            if      (strcmp(s, "cosine")   == 0) schedule = LRScheduleType::Cosine;
            else if (strcmp(s, "constant") == 0) schedule = LRScheduleType::Constant;
            else { fprintf(stderr, "Unknown schedule: %s\n", s); return 1; }
        } else {
            fprintf(stderr, "ERROR: unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // ── Auto-generate save_dir if not specified ──────────────────────────────
    if (save_dir.empty()) {
        save_dir = make_default_save_dir();
    }

    // ── Validate paths ───────────────────────────────────────────────────────
    if (!std::filesystem::exists(model_dir)) {
        fprintf(stderr, "ERROR: model dir not found: %s\n", model_dir.c_str());
        return 1;
    }
    if (!std::filesystem::exists(data_path)) {
        fprintf(stderr, "ERROR: data file not found: %s\n", data_path.c_str());
        return 1;
    }

    // ── Print run config ────────────────────────────────────────────────────
    int num_prompts = std::max(1, batch_size / num_generations);
    printf("=== GRPO Training — Qwen3-0.6B ===\n");
    printf("  model          : %s\n",  model_dir.c_str());
    printf("  data           : %s\n",  data_path.c_str());
    printf("  batch_size     : %d  (%d prompts x %d gens)\n",
           batch_size, num_prompts, num_generations);
    printf("  grad accum     : %d\n",  grad_accum);
    printf("  max_comp / seq : %d / %d\n",  max_completion_len, max_seq_len);
    printf("  sampling       : temp=%.2f  top_p=%.2f  top_k=%d\n",
           gen_temperature, gen_top_p, gen_top_k);
    printf("  lr schedule    : %s  (%.2e -> %.2e, warmup=%d, total=%d)\n",
           schedule == LRScheduleType::Cosine ? "cosine" : "constant",
           base_lr, min_lr, warmup_steps, total_steps);
    printf("  grad clip      : %.1f\n", max_grad_norm);
    printf("  max_model_len  : %d\n",  max_model_len);
    printf("  gpu_util       : %.1f%%\n", gpu_util * 100);
    printf("  save_dir       : %s\n",  save_dir.c_str());
    printf("  log / save     : every %d / %d steps\n\n", log_steps, save_steps);

    // ── Create LLMEngine ─────────────────────────────────────────────────────
    Config engine_cfg(model_dir);
    engine_cfg.max_model_len         = max_model_len;
    engine_cfg.enforce_eager         = true;
    engine_cfg.gpu_memory_utilization = gpu_util;
    printf("[grpo] Creating LLMEngine from %s ...\n", model_dir.c_str());
    LLMEngine engine(engine_cfg);

    // ── Configure GRPO ───────────────────────────────────────────────────────
    GRPOConfig config(data_path);
    config.batch_size         = batch_size;
    config.num_generations    = num_generations;
    config.grad_accum_steps   = grad_accum;
    config.max_completion_len = max_completion_len;
    config.max_seq_len        = max_seq_len;
    config.gen_temperature    = gen_temperature;
    config.gen_top_p          = gen_top_p;
    config.gen_top_k          = gen_top_k;
    config.base_lr            = base_lr;
    config.min_lr             = min_lr;
    config.total_steps        = total_steps;
    config.warmup_steps       = warmup_steps;
    config.logging_steps      = log_steps;
    config.save_steps         = save_steps;
    config.save_dir           = save_dir;
    config.max_grad_norm      = max_grad_norm;
    config.lr_schedule_type   = schedule;
    config.recompute();

    // ── Run training ─────────────────────────────────────────────────────────
    {
        GRPOTrainer trainer(config, &engine, (RewardFn)boxed_reward);

        printf("[grpo] Starting GRPO training for %d steps ...\n", config.total_steps);
        trainer.grpo_train();
    }

    printf("\nDone. Log: %s/train_log.jsonl\n", save_dir.c_str());
    return 0;
}
