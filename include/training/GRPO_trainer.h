#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>
#include "trainer.h"
#include "engine/llm_engine.h"
#include "third_party/json.hpp"

// ============================================================
// GRPO (Group Relative Policy Optimization) Trainer
//
// Algorithm (no KL penalty / no reference model):
//   For each batch of prompts:
//   1. wakeup KV cache, generate G completions per prompt
//   2. sleep KV cache (free memory for training)
//   3. Score completions with reward function
//   4. Compute group-normalized advantages
//   5. Forward/backward with GRPO loss, optimizer step
//
// Loss aggregation (DAPO):
//   L = (1/N_active) * sum_{active tokens} -advantage[seq] * log_prob[t]
//   where N_active = total active tokens across the global batch
//
// Architecture:
//   LLMEngine owns the model (via ModelRunner). GRPOTrainer uses
//   engine->model_runner->model for training. Generation is delegated
//   to the engine (sleep/wakeup/generate_ids). Optimizer updates weights
//   in-place, so the engine's model is always up-to-date.
// ============================================================

// ============================================================
// GRPO Loss Kernel
// ============================================================

__global__ static void grpo_loss_kernel(
    float*       __restrict__ grad,
    const float* __restrict__ log_probs,
    const int*   __restrict__ loss_mask,
    const float* __restrict__ advantages,  // [B] per-sequence
    float*       __restrict__ loss_sum,     // [1] atomic accumulator
    int T, int S, int N_active
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    int b = t / S;

    if (loss_mask[t] && N_active > 0) {
        float adv = advantages[b];
        float inv_n = 1.0f / (float)N_active;
        grad[t] = -adv * inv_n;
        atomicAdd(loss_sum, -adv * log_probs[t]);
    } else {
        grad[t] = 0.0f;
    }
}

// ============================================================
// GRPO Config
// ============================================================

struct GRPOConfig : public TrainingConfig {
    // batch_size (inherited): total training sequences per step = num_prompts * G
    // num_prompts is derived: batch_size / num_generations
    int num_generations    = 4;     // completions per prompt (G)
    int max_completion_len = 256;
    float gen_temperature  = 1.0f;
    float gen_top_p        = 1.0f;
    int gen_top_k          = 0;

    float advantage_eps    = 1e-8f;

    // Derived (call recompute() after changing batch_size / num_generations)
    int num_prompts        = 1;

    GRPOConfig(std::string data_path)
        : TrainingConfig(data_path)
    {
        recompute();
    }

    void recompute() {
        num_prompts = std::max(1, batch_size / num_generations);
        global_batch_size = num_prompts * num_generations;
    }
};

// ============================================================
// Reward function type
// ============================================================
using RewardFn = std::function<float(const std::string& completion,
                                     const std::string& answer)>;

// ============================================================
// GRPO Trainer
// ============================================================

class GRPOTrainer : public Trainer {
public:
    struct GRPOSample {
        std::string prompt;   // raw prompt text (with chat template)
        std::string answer;   // ground truth answer
    };

    struct Generation {
        std::vector<int64_t> prompt_tokens;
        std::vector<int64_t> completion_tokens;
        std::string completion_text;
        int prompt_idx;  // index within the current batch of prompts
    };

    GRPOTrainer(const GRPOConfig& config, LLMEngine* engine, RewardFn reward_fn)
        : Trainer(static_cast<const TrainingConfig&>(config),
                  engine->model_runner->model, /*load_data=*/false),
          grpo_config_(config),
          engine_(engine),
          reward_fn_(std::move(reward_fn))
    {
        load_dataset(config.train_data_path);

        CUDA_CHECK(cudaMalloc(&d_loss_sum_, sizeof(float)));

        // Sleep KV cache — free memory for training
        engine_->sleep();

        printf("[GRPOTrainer] batch_size=%d  num_prompts=%d  G=%d  "
               "grad_accum=%d  max_completion=%d  dataset=%zu samples\n",
               config_.batch_size, grpo_config_.num_prompts,
               grpo_config_.num_generations, config_.grad_accum_steps,
               grpo_config_.max_completion_len,
               dataset_.size());
    }

    ~GRPOTrainer() override {
        if (d_loss_sum_)    cudaFree(d_loss_sum_);
        if (d_advantages_)  cudaFree(d_advantages_);
    }

    // ================================================================
    // Main GRPO training loop
    // ================================================================
    void grpo_train() {
        const int B = grpo_config_.num_prompts;
        const int G = grpo_config_.num_generations;
        const size_t N = dataset_.size();
        int global_step = 0;

        // Build sample index for manual shuffling
        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);

        for (int epoch = 0; epoch < config_.num_epochs; epoch++) {
            std::mt19937_64 rng((uint64_t)(epoch + 42));
            // std::shuffle(indices.begin(), indices.end(), rng);

            for (size_t cursor = 0; cursor + B <= N; cursor += B) {
                if (global_step >= config_.total_steps) break;

                // --- Gather prompts and answers for this batch ---
                // Tokenize prompts on-the-fly from raw text
                std::vector<std::vector<int64_t>> prompts(B);
                std::vector<std::string> answers(B);
                for (int i = 0; i < B; i++) {
                    int idx = indices[cursor + i];
                    prompts[i] = engine_->tokenizer->encode(dataset_[idx].prompt);
                    answers[i] = dataset_[idx].answer;
                }

                float cur_lr = lr_scheduler_.get_lr(global_step);
                optimizer_.set_lr(cur_lr);

                auto t0 = std::chrono::steady_clock::now();

                // --- 1. Generate completions (wakeup -> generate -> sleep) ---
                auto gens = generate(prompts);

                // --- 2. Compute rewards ---
                auto rewards = compute_rewards(gens, answers);

                // --- 3. Compute advantages ---
                auto advantages = compute_advantages(rewards);

                // --- 4. Build training batch ---
                auto train_batch = build_train_batch(gens);

                // --- 5. GRPO training step ---
                float loss = grpo_training_step(train_batch, advantages);

                cudaDeviceSynchronize();
                auto t1 = std::chrono::steady_clock::now();
                float step_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

                // Metrics
                float mean_reward = 0;
                for (float r : rewards) mean_reward += r;
                mean_reward /= (float)rewards.size();

                int total_comp_tokens = 0;
                for (auto& gen : gens)
                    total_comp_tokens += (int)gen.completion_tokens.size();

                global_step++;

                logger_.log(global_step, "loss",          loss);
                logger_.log(global_step, "lr",            cur_lr);
                logger_.log(global_step, "grad_norm",     last_grad_norm_);
                logger_.log(global_step, "mean_reward",   mean_reward);
                logger_.log(global_step, "step_ms",       step_ms);
                logger_.log(global_step, "comp_tokens",   (float)total_comp_tokens);

                if (global_step % config_.logging_steps == 0)
                    logger_.commit(global_step);

                if (config_.save_steps > 0 && global_step % config_.save_steps == 0)
                    save_checkpoint(global_step);
            }

            if (global_step >= config_.total_steps) break;
        }

        printf("[GRPOTrainer] Done. %d steps completed.\n", global_step);
    }

protected:
    // Not used — GRPO has its own training step
    float compute_loss(const float* /*d_log_probs*/, const int* /*d_loss_mask*/,
                       int /*B*/, int /*S*/) override {
        return 0.0f;
    }

private:
    GRPOConfig  grpo_config_;
    LLMEngine*  engine_;        // not owned — caller manages lifetime
    RewardFn    reward_fn_;
    std::vector<GRPOSample> dataset_;  // raw text prompts + answers

    // Device buffers
    float*      d_loss_sum_    = nullptr;
    float*      d_advantages_  = nullptr;
    int         d_adv_cap_     = 0;

    // ================================================================
    // Load dataset from JSONL: {"prompt": "...", "answer": "..."}
    // ================================================================
    void load_dataset(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            fprintf(stderr, "[GRPOTrainer] ERROR: cannot open dataset %s\n",
                    path.c_str());
            return;
        }

        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            try {
                auto j = nlohmann::json::parse(line);
                GRPOSample s;
                s.prompt = j.at("prompt").get<std::string>();
                s.answer = j.at("answer").get<std::string>();
                dataset_.push_back(std::move(s));
            } catch (const std::exception& e) {
                fprintf(stderr, "[GRPOTrainer] Parse error: %s\n", e.what());
            }
        }

        printf("[GRPOTrainer] Loaded %zu samples from %s\n",
               dataset_.size(), path.c_str());
    }

    // Generate G completions per prompt — delegates to LLMEngine
    std::vector<Generation> generate(
        const std::vector<std::vector<int64_t>>& prompts)
    {
        const int B = (int)prompts.size();
        const int G = grpo_config_.num_generations;
        const int total_seqs = B * G;

        // Wakeup engine (re-allocate KV cache + scheduler)
        auto t_wake = std::chrono::steady_clock::now();
        engine_->wakeup();
        auto t_wake_done = std::chrono::steady_clock::now();

        // Configure sampling
        SamplingParams sp;
        sp.max_new_tokens = grpo_config_.max_completion_len;
        sp.temperature    = grpo_config_.gen_temperature;
        sp.top_p          = grpo_config_.gen_top_p;
        sp.top_k          = grpo_config_.gen_top_k;
        sp.do_sample      = true;
        sp.ignore_eos     = false;

        // Generate via engine
        auto t_gen = std::chrono::steady_clock::now();
        auto completions = engine_->generate_ids(prompts, sp, G);
        auto t_gen_done = std::chrono::steady_clock::now();

        // Sleep engine (free KV cache for training)
        auto t_sleep = std::chrono::steady_clock::now();
        engine_->sleep();
        auto t_sleep_done = std::chrono::steady_clock::now();

        // Build Generation structs
        std::vector<Generation> gens(total_seqs);
        for (int i = 0; i < B; i++) {
            for (int g = 0; g < G; g++) {
                int idx = i * G + g;
                gens[idx].prompt_tokens = prompts[i];
                gens[idx].prompt_idx = i;

                auto it = completions.find((int64_t)idx);
                if (it != completions.end()) {
                    gens[idx].completion_tokens = it->second;
                    gens[idx].completion_text =
                        engine_->tokenizer->decode(it->second);
                }
            }
        }

        int total_comp = 0;
        for (auto& gen : gens) total_comp += (int)gen.completion_tokens.size();

        float wake_ms  = std::chrono::duration<float, std::milli>(t_wake_done - t_wake).count();
        float gen_ms   = std::chrono::duration<float, std::milli>(t_gen_done - t_gen).count();
        float sleep_ms = std::chrono::duration<float, std::milli>(t_sleep_done - t_sleep).count();
        float tok_s    = gen_ms > 0 ? total_comp / (gen_ms * 1e-3f) : 0;
        printf("[GRPO] Generated %d completions, %d tokens  "
               "(wakeup=%.0fms  gen=%.0fms [%.0f tok/s]  sleep=%.0fms)\n",
               total_seqs, total_comp, wake_ms, gen_ms, tok_s, sleep_ms);

        return gens;
    }

    // Compute rewards
    std::vector<float> compute_rewards(
        const std::vector<Generation>& gens,
        const std::vector<std::string>& answers)
    {
        std::vector<float> rewards(gens.size());
        for (int i = 0; i < (int)gens.size(); i++) {
            rewards[i] = reward_fn_(gens[i].completion_text,
                                    answers[gens[i].prompt_idx]);
        }

        // --- Debug: print prompt, completion, answer, reward ---
        // const int G = grpo_config_.num_generations;
        // for (int i = 0; i < (int)gens.size(); i++) {
        //     auto& gen = gens[i];
        //     // Decode prompt tokens to text
        //     std::string prompt_text = engine_->tokenizer->decode(
        //         std::vector<int64_t>(gen.prompt_tokens.begin(), gen.prompt_tokens.end()));

        //     printf("\n── gen[%d] (prompt %d, g=%d) ──────────────────────\n",
        //            i, gen.prompt_idx, i % G);
        //     printf("  PROMPT:     %s\n", prompt_text.c_str());
        //     printf("  COMPLETION: %s\n", gen.completion_text.c_str());
        //     printf("  ANSWER:     %s\n", answers[gen.prompt_idx].c_str());
        //     printf("  REWARD:     %.1f  (comp_tokens=%d)\n",
        //            rewards[i], (int)gen.completion_tokens.size());
        // }
        // printf("────────────────────────────────────────────────────\n");

        return rewards;
    }

    // ================================================================
    // Group-normalized advantages (GRPO)
    // ================================================================
    std::vector<float> compute_advantages(const std::vector<float>& rewards) {
        const int G = grpo_config_.num_generations;
        const int num_groups = (int)rewards.size() / G;
        std::vector<float> advantages(rewards.size());

        for (int i = 0; i < num_groups; i++) {
            float sum = 0, sum_sq = 0;
            for (int g = 0; g < G; g++) {
                float r = rewards[i * G + g];
                sum += r;
                sum_sq += r * r;
            }
            float mean = sum / G;
            float var  = sum_sq / G - mean * mean;
            float stddev = sqrtf(std::max(0.0f, var) + grpo_config_.advantage_eps);

            for (int g = 0; g < G; g++)
                advantages[i * G + g] = (rewards[i * G + g] - mean) / stddev;
        }
        return advantages;
    }

    // ================================================================
    // Build TrainBatch from generations
    // ================================================================
    TrainBatch build_train_batch(const std::vector<Generation>& gens) {
        int total_seqs = (int)gens.size();

        int S = 0;
        for (auto& gen : gens) {
            int len = (int)gen.prompt_tokens.size() + (int)gen.completion_tokens.size();
            S = std::max(S, len);
        }
        S = std::min(S, config_.max_seq_len);

        TrainBatch batch;
        batch.B = total_seqs;
        batch.S = S;
        batch.token_ids.assign(total_seqs * S, 0);
        batch.target_ids.assign(total_seqs * S, 0);
        batch.loss_mask.assign(total_seqs * S, 0);
        batch.seq_lens.resize(total_seqs);
        batch.prompt_lens.resize(total_seqs);

        for (int b = 0; b < total_seqs; b++) {
            auto& gen = gens[b];
            int plen = (int)gen.prompt_tokens.size();
            int clen = (int)gen.completion_tokens.size();
            int total_len = std::min(plen + clen, S);

            batch.seq_lens[b]    = total_len;
            batch.prompt_lens[b] = plen;

            for (int t = 0; t < std::min(plen, total_len); t++)
                batch.token_ids[b * S + t] = (int)gen.prompt_tokens[t];
            for (int t = plen; t < total_len; t++)
                batch.token_ids[b * S + t] = (int)gen.completion_tokens[t - plen];

            // Target: shifted left by 1
            for (int t = 0; t < total_len - 1; t++)
                batch.target_ids[b * S + t] = batch.token_ids[b * S + t + 1];

            // Loss mask: completion tokens only (positions predicting tokens >= plen)
            int loss_start = std::max(0, plen - 1);
            for (int t = loss_start; t < total_len - 1; t++)
                batch.loss_mask[b * S + t] = 1;
        }

        return batch;
    }

    // ================================================================
    // GRPO training step with DAPO normalization
    // ================================================================
    float grpo_training_step(TrainBatch& global_batch,
                             const std::vector<float>& advantages) {
        const int B_total = global_batch.B;
        const int S       = global_batch.S;
        const int acc_steps = config_.grad_accum_steps > 0 ? config_.grad_accum_steps : 1;
        const int B = B_total / acc_steps;
        const int micro_T = B * S;

        ensure_buffers(B, S);

        // Total active tokens across global batch (DAPO denominator)
        int N_active = 0;
        for (int v : global_batch.loss_mask) N_active += v;

        ensure_advantage_buf(B);

        qwen3_gradients_zero(grads_);

        float total_loss = 0.0f;

        for (int acc = 0; acc < acc_steps; acc++) {
            const int* tokens  = global_batch.token_ids.data()  + acc * micro_T;
            const int* targets = global_batch.target_ids.data() + acc * micro_T;
            const int* mask_h  = global_batch.loss_mask.data()  + acc * micro_T;

            // Per-sequence advantages for this micro-batch
            std::vector<float> micro_adv(B);
            for (int b = 0; b < B; b++)
                micro_adv[b] = advantages[acc * B + b];

            CUDA_CHECK(cudaMemcpy(d_loss_mask_, mask_h,
                                  micro_T * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_advantages_, micro_adv.data(),
                                  B * sizeof(float), cudaMemcpyHostToDevice));

            // Forward
            qwen3_forward(model_, state_, tokens, targets, d_log_probs_, B, S);

            // GRPO gradient
            CUDA_CHECK(cudaMemset(d_loss_sum_, 0, sizeof(float)));
            int block = 256;
            int grid  = (micro_T + block - 1) / block;
            grpo_loss_kernel<<<grid, block>>>(
                d_loss_grad_, d_log_probs_, d_loss_mask_, d_advantages_,
                d_loss_sum_, micro_T, S, N_active);

            float micro_loss;
            CUDA_CHECK(cudaMemcpy(&micro_loss, d_loss_sum_,
                                  sizeof(float), cudaMemcpyDeviceToHost));
            total_loss += micro_loss;

            // Backward (accumulates into grads_)
            qwen3_backward(model_, state_, grads_, d_loss_grad_);
        }

        // Gradient clipping
        float grad_norm = 0.0f;
        if (config_.max_grad_norm > 0.0f)
            grad_norm = clip_grad_norm_(grads_, config_.max_grad_norm);
        last_grad_norm_ = grad_norm;

        // Optimizer step
        optimizer_.step(grads_);

        return (N_active > 0) ? total_loss / N_active : 0.0f;
    }

    // ================================================================
    // Helpers
    // ================================================================

    void ensure_advantage_buf(int B) {
        if (B <= d_adv_cap_) return;
        if (d_advantages_) cudaFree(d_advantages_);
        CUDA_CHECK(cudaMalloc(&d_advantages_, B * sizeof(float)));
        d_adv_cap_ = B;
    }
};
