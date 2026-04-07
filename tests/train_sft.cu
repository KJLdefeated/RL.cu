// train_sft.cpp
//
// SFT training script for Qwen3-0.6B using data/sft_train.bin
// (prepared by: python scripts/prepare_data.py --mode sft --output data/sft_train.bin)
//
// Usage:
//   ./build/train_sft [options]   (see --help)
//
// Default smoke-test: 100 steps, batch=2, accum=4 (global_bs=8), seq=512.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <filesystem>

#include "model/qwen3.h"
#include "training/SFT_trainer.h"

static void print_usage(const char* prog) {
    printf(
        "Usage: %s [options]\n"
        "\nModel / data:\n"
        "  --model      DIR   model weights dir         (default: model_weights/Qwen3-0.6B)\n"
        "  --data       FILE  training binary            (default: data/sft_train.bin)\n"
        "\nBatch / sequence:\n"
        "  --batch-size N     micro-batch size           (default: 2)\n"
        "  --accum      N     gradient accum steps       (default: 4 -> global_bs=8)\n"
        "  --seq-len    N     max sequence length        (default: 512)\n"
        "\nOptimizer:\n"
        "  --lr         F     peak learning rate         (default: 1e-5)\n"
        "  --min-lr     F     min LR after cosine decay  (default: 1e-6)\n"
        "  --warmup     N     warmup steps               (default: 100)\n"
        "  --total-steps N    total optimizer steps      (default: 1000)\n"
        "  --schedule   TYPE  cosine | constant          (default: cosine)\n"
        "\nLogging / checkpointing:\n"
        "  --save-dir   DIR   checkpoint + log dir       (default: checkpoints/sft)\n"
        "  --log-steps  N     log every N steps          (default: 10)\n"
        "  --save-steps N     checkpoint every N steps   (default: 500, 0=off)\n"
        , prog);
}

int main(int argc, char** argv) {
    // ── Defaults ──────────────────────────────────────────────────────────────
    std::string model_dir   = "model_weights/Qwen3-0.6B";
    std::string data_path   = "data/sft_train.bin";
    int   batch_size        = 8;
    int   grad_accum        = 4;
    int   max_seq_len       = 2048;
    float base_lr           = 1e-5f;
    float min_lr            = 1e-6f;
    int   warmup_steps      = 100;
    int   total_steps       = 5000;
    std::string save_dir    = "checkpoints/sft";
    int   log_steps         = 1;
    int   save_steps        = 500;
    LRScheduleType schedule = LRScheduleType::Cosine;

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
        else if (eq("--model"))             model_dir    = next();
        else if (eq("--data"))              data_path    = next();
        else if (eq("--batch-size"))        batch_size   = atoi(next());
        else if (eq("--accum"))             grad_accum   = atoi(next());
        else if (eq("--seq-len"))           max_seq_len  = atoi(next());
        else if (eq("--lr"))                base_lr      = (float)atof(next());
        else if (eq("--min-lr"))            min_lr       = (float)atof(next());
        else if (eq("--warmup"))            warmup_steps = atoi(next());
        else if (eq("--total-steps"))       total_steps  = atoi(next());
        else if (eq("--save-dir"))          save_dir     = next();
        else if (eq("--log-steps"))         log_steps    = atoi(next());
        else if (eq("--save-steps"))        save_steps   = atoi(next());
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

    // ── Validate paths ────────────────────────────────────────────────────────
    if (!std::filesystem::exists(model_dir)) {
        fprintf(stderr, "ERROR: model dir not found: %s\n", model_dir.c_str());
        return 1;
    }
    if (!std::filesystem::exists(data_path)) {
        fprintf(stderr, "ERROR: data file not found: %s\n", data_path.c_str());
        return 1;
    }

    // ── Print run config ─────────────────────────────────────────────────────
    int global_bs = batch_size * grad_accum;
    printf("=== SFT Training — Qwen3-0.6B ===\n");
    printf("  model       : %s\n",  model_dir.c_str());
    printf("  data        : %s\n",  data_path.c_str());
    printf("  micro-batch : %d\n",  batch_size);
    printf("  grad accum  : %d  (global_bs=%d)\n", grad_accum, global_bs);
    printf("  seq_len     : %d\n",  max_seq_len);
    printf("  lr schedule : %s  (%.2e -> %.2e, warmup=%d, total=%d)\n",
           schedule == LRScheduleType::Cosine ? "cosine" : "constant",
           base_lr, min_lr, warmup_steps, total_steps);
    printf("  save_dir    : %s\n",  save_dir.c_str());
    printf("  log / save  : every %d / %d steps\n",   log_steps, save_steps);
    printf("  freeze_embed: yes (tied embed_tokens)\n\n");

    // ── Load model ────────────────────────────────────────────────────────────
    // qwen3_load allocates KV scratch for max_batch sequences; use micro-batch
    // size since training forward only processes one micro-batch at a time.
    Qwen3Model* model = qwen3_load(model_dir.c_str(), batch_size, max_seq_len);

    // ── TrainingConfig ────────────────────────────────────────────────────────
    TrainingConfig config(
        data_path,
        batch_size,
        grad_accum,
        max_seq_len,
        /*num_epochs=*/1,
        save_dir,
        base_lr,
        /*beta1=*/0.9f,
        /*beta2=*/0.999f,
        /*opt_eps=*/1e-8f,
        /*weight_decay=*/0.01f,
        min_lr,
        warmup_steps,
        total_steps,
        log_steps,
        save_steps,
        /*save_total_limit=*/5,
        schedule
    );
    // Qwen3-0.6B uses tied embeddings (embed_tokens == lm_head).
    // Freezing prevents the embedding table from drifting during SFT,
    // which would corrupt token representations shared with the output head.
    config.freeze_embed = true;

    // ── Run training ─────────────────────────────────────────────────────────
    SFTTrainer trainer(config, model);
    trainer.train();

    qwen3_free(model);
    printf("\nDone. Log: %s/train_log.jsonl\n", save_dir.c_str());
    return 0;
}
