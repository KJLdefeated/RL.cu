#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include "dataloader.h"
#include "lr_scheduler.h"
#include "optimizer.h"
#include "model/qwen3.h"

struct TrainingConfig {
    std::string train_data_path;
    int         batch_size       = 4;
    int         max_seq_len      = 512;
    int         num_epochs       = 1;
    int         log_interval     = 10;
    int         save_interval    = 100;
    std::string save_dir         = "checkpoints";
    float       base_lr          = 1e-5f;
    float       beta1            = 0.9f;
    float       beta2            = 0.999f;
    float       opt_eps          = 1e-8f;
    float       weight_decay     = 0.01f;
    float       min_lr           = 0.0f;
    int         warmup_steps     = 100;
    int         total_steps      = 1000;
    LRScheduleType lr_schedule_type = LRScheduleType::Cosine;
};

class Trainer {
public:
    Trainer(const TrainingConfig& config, Qwen3Model* model)
        : model_(model),
          optimizer_(model, config.base_lr, config.beta1, config.beta2,
                     config.opt_eps, config.weight_decay),
          data_loader_(config.train_data_path, config.batch_size, config.max_seq_len),
          config_(config),
          state_(nullptr), grads_(nullptr)
    {
        // LR scheduler
        if (config.lr_schedule_type == LRScheduleType::Cosine) {
            lr_scheduler_.init_cosine(config.base_lr, config.min_lr,
                                      config.warmup_steps, config.total_steps);
        } else {
            lr_scheduler_.init_constant(config.base_lr, config.warmup_steps);
        }
    }

    virtual ~Trainer() {
        if (state_) { qwen3_train_state_free(state_); state_ = nullptr; }
        if (grads_) { qwen3_gradients_free(grads_); grads_ = nullptr; }
    }

    // Main training loop
    void train() {
        int global_step = 0;
        float running_loss = 0.0f;

        for (int epoch = 0; epoch < config_.num_epochs; epoch++) {
            data_loader_.shuffle((uint64_t)(epoch + 42));

            TrainBatch batch;
            while (data_loader_.next_batch(batch)) {
                if (global_step >= config_.total_steps) break;

                // Update LR
                float cur_lr = lr_scheduler_.get_lr(global_step);
                optimizer_.set_lr(cur_lr);

                // Forward + backward + optimizer step
                float loss = training_step(batch);
                running_loss += loss;
                global_step++;

                // Logging
                if (global_step % config_.log_interval == 0) {
                    float avg = running_loss / config_.log_interval;
                    printf("[step %d] loss=%.4f  lr=%.2e\n", global_step, avg, cur_lr);
                    running_loss = 0.0f;
                }

                // Checkpoint
                if (config_.save_interval > 0 && global_step % config_.save_interval == 0) {
                    save_checkpoint(global_step);
                }
            }

            if (global_step >= config_.total_steps) break;
            data_loader_.reset();
        }

        printf("[Trainer] Done. %d steps completed.\n", global_step);
    }

    float training_step(const TrainBatch& batch) {
        int B = batch.B;
        int S = batch.S;
        int T = B * S;

        // Lazy-allocate train state and gradients for this (B, S)
        ensure_buffers(B, S);

        // Forward pass: compute log_probs
        qwen3_forward(model_, state_, batch.token_ids.data(),
                       batch.target_ids.data(), d_log_probs_, B, S);

        // Compute loss (subclass-specific)
        // Copy loss_mask to device
        CUDA_CHECK(cudaMemcpy(d_loss_mask_, batch.loss_mask.data(),
                              T * sizeof(int), cudaMemcpyHostToDevice));

        float loss = compute_loss(d_log_probs_, d_loss_mask_, B, S);

        // Backward pass
        qwen3_gradients_zero(grads_);
        qwen3_backward(model_, state_, grads_, d_loss_grad_);

        // Optimizer step
        optimizer_.step(grads_);

        return loss;
    }

protected:
    Qwen3Model*      model_;
    AdamWOptimizer   optimizer_;
    LRScheduler      lr_scheduler_;
    DataLoader       data_loader_;
    TrainingConfig   config_;

    // GPU buffers (lazy-allocated)
    Qwen3TrainState* state_ = nullptr;
    Qwen3Gradients*  grads_ = nullptr;
    float*           d_log_probs_  = nullptr;  // [T] output from forward
    float*           d_loss_grad_  = nullptr;  // [T] upstream gradient for backward
    int*             d_loss_mask_  = nullptr;  // [T] loss mask on device
    int              buf_B_ = 0, buf_S_ = 0;  // current buffer dimensions

    // Implement in subclass
    // Returns scalar loss for logging.
    //   d_log_probs: [B*S] log-probabilities from forward (device, FP32)
    //   d_loss_mask: [B*S] 1 where loss is computed, 0 for padding/prompt (device)
    //   Must write upstream gradient into d_loss_grad_ before returning.
    virtual float compute_loss(const float* d_log_probs, const int* d_loss_mask,
                               int B, int S) = 0;

    virtual void save_checkpoint(int step) {
        printf("[Trainer] Checkpoint at step %d (save not yet implemented)\n", step);
    }

private:
    void ensure_buffers(int B, int S) {
        int T = B * S;
        if (state_ && buf_B_ >= B && buf_S_ >= S) return;

        // Free old
        if (state_) { qwen3_train_state_free(state_); state_ = nullptr; }
        if (grads_) { qwen3_gradients_free(grads_); grads_ = nullptr; }
        if (d_log_probs_) { cudaFree(d_log_probs_); d_log_probs_ = nullptr; }
        if (d_loss_grad_) { cudaFree(d_loss_grad_); d_loss_grad_ = nullptr; }
        if (d_loss_mask_) { cudaFree(d_loss_mask_); d_loss_mask_ = nullptr; }

        state_ = qwen3_train_state_alloc(model_->config, B, S);
        grads_ = qwen3_gradients_alloc(model_->config, T);

        CUDA_CHECK(cudaMalloc(&d_log_probs_, T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_loss_grad_, T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_loss_mask_, T * sizeof(int)));

        buf_B_ = B;
        buf_S_ = S;
    }
};
