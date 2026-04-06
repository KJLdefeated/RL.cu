#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include "dataloader.h"
#include "lr_scheduler.h"
#include "logger.h"
#include "optimizer.h"
#include "model/qwen3.h"

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)
#endif

struct TrainingConfig {
    std::string train_data_path;
    int         global_batch_size = 16;   // effective batch = batch_size * grad_accum_steps
    int         batch_size        = 4;    // micro-batch size per forward/backward
    int         grad_accum_steps  = 1;
    int         max_seq_len       = 512;
    int         num_epochs        = 1;
    std::string save_dir          = "checkpoints";
    float       base_lr           = 1e-5f;
    float       beta1             = 0.9f;
    float       beta2             = 0.999f;
    float       opt_eps           = 1e-8f;
    float       weight_decay      = 0.01f;
    float       min_lr            = 0.0f;
    int         warmup_steps      = 100;
    int         total_steps       = 1000;
    int         logging_steps     = 10;
    int         save_steps        = 100;
    int         save_total_limit  = 5;
    LRScheduleType lr_schedule_type = LRScheduleType::Cosine;

    TrainingConfig(
        std::string train_data_path,
        int batch_size      = 4,
        int grad_accum_steps = 1,
        int max_seq_len     = 512,
        int num_epochs      = 1,
        std::string save_dir = "checkpoints",
        float base_lr       = 1e-5f,
        float beta1         = 0.9f,
        float beta2         = 0.999f,
        float opt_eps       = 1e-8f,
        float weight_decay  = 0.01f,
        float min_lr        = 0.0f,
        int warmup_steps    = 100,
        int total_steps     = 1000,
        int logging_steps   = 10,
        int save_steps      = 100,
        int save_total_limit = 5,
        LRScheduleType lr_schedule_type = LRScheduleType::Cosine
    ) {
        this->train_data_path  = train_data_path;
        this->batch_size       = batch_size;
        this->grad_accum_steps = grad_accum_steps;
        this->global_batch_size = batch_size * grad_accum_steps;
        this->max_seq_len      = max_seq_len;
        this->num_epochs       = num_epochs;
        this->save_dir         = save_dir;
        this->base_lr          = base_lr;
        this->beta1            = beta1;
        this->beta2            = beta2;
        this->opt_eps          = opt_eps;
        this->weight_decay     = weight_decay;
        this->min_lr           = min_lr;
        this->warmup_steps     = warmup_steps;
        this->total_steps      = total_steps;
        this->logging_steps    = logging_steps;
        this->save_steps       = save_steps;
        this->save_total_limit = save_total_limit;
        this->lr_schedule_type = lr_schedule_type;
    }
};

// Scale gradient in-place by a scalar factor.
// Used to divide each micro-batch's loss gradient by grad_accum_steps
// so the accumulated gradient equals the mean over the global batch.
__global__ static void scale_grad_kernel(float* __restrict__ grad, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] *= scale;
}

// =============================================================================
// Trainer — base class for SFTTrainer and GRPOTrainer.
//
// Training loop with gradient accumulation:
//   DataLoader yields global_batch_size samples at a time.
//   training_step() splits them into grad_accum_steps micro-batches,
//   accumulates gradients, then calls optimizer.step() once.
//
// Subclasses override compute_loss() to define their loss.
// =============================================================================
class Trainer {
public:
    Trainer(const TrainingConfig& config, Qwen3Model* model)
        : model_(model),
          optimizer_(model, config.base_lr, config.beta1, config.beta2,
                     config.opt_eps, config.weight_decay),
          // DataLoader uses global_batch_size so each next_batch() call
          // returns one full accumulation window worth of samples.
          data_loader_(config.train_data_path, config.global_batch_size, config.max_seq_len),
          logger_(config.save_dir),
          config_(config)
    {
        if (config.lr_schedule_type == LRScheduleType::Cosine) {
            lr_scheduler_.init_cosine(config.base_lr, config.min_lr,
                                      config.warmup_steps, config.total_steps);
        } else {
            lr_scheduler_.init_constant(config.base_lr, config.warmup_steps);
        }
    }

    virtual ~Trainer() { free_buffers(); }

    // Main training loop
    void train() {
        int global_step = 0;

        for (int epoch = 0; epoch < config_.num_epochs; epoch++) {
            data_loader_.shuffle((uint64_t)(epoch + 42));

            TrainBatch global_batch;
            while (data_loader_.next_batch(global_batch)) {
                if (global_step >= config_.total_steps) break;

                float cur_lr = lr_scheduler_.get_lr(global_step);
                optimizer_.set_lr(cur_lr);

                auto t0 = std::chrono::steady_clock::now();
                float loss = training_step(global_batch);
                // Sync GPU so the elapsed time reflects actual compute.
                cudaDeviceSynchronize();
                auto t1 = std::chrono::steady_clock::now();
                float step_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
                float tok_per_sec = (config_.global_batch_size * (float)global_batch.S)
                                    / (step_ms * 1e-3f);

                global_step++;

                // Accumulate metrics into logger
                logger_.log(global_step, "loss",        loss);
                logger_.log(global_step, "lr",          cur_lr);
                logger_.log(global_step, "step_ms",     step_ms);
                logger_.log(global_step, "tok_per_sec", tok_per_sec);

                // Flush log every logging_steps
                if (global_step % config_.logging_steps == 0)
                    logger_.commit(global_step);

                if (config_.save_steps > 0 && global_step % config_.save_steps == 0)
                    save_checkpoint(global_step);
            }

            if (global_step >= config_.total_steps) break;
            data_loader_.reset();
        }

        printf("[Trainer] Done. %d steps completed.\n", global_step);
    }

    // One optimizer step over a global batch, accumulating grads over micro-batches.
    // Returns mean loss across micro-batches.
    float training_step(const TrainBatch& global_batch) {
        const int acc_steps = config_.grad_accum_steps;
        const int B         = config_.batch_size;   // micro-batch size
        const int S         = global_batch.S;
        const int micro_T   = B * S;
        const float scale   = 1.0f / acc_steps;

        // Buffers sized for one micro-batch
        ensure_buffers(B, S);

        // Zero gradient accumulators once per optimizer step
        qwen3_gradients_zero(grads_);

        float accum_loss = 0.0f;

        for (int acc = 0; acc < acc_steps; acc++) {
            const int* tokens  = global_batch.token_ids.data()  + acc * micro_T;
            const int* targets = global_batch.target_ids.data() + acc * micro_T;
            const int* mask_h  = global_batch.loss_mask.data()  + acc * micro_T;

            CUDA_CHECK(cudaMemcpy(d_loss_mask_, mask_h,
                                  micro_T * sizeof(int), cudaMemcpyHostToDevice));

            // Forward
            qwen3_forward(model_, state_, tokens, targets, d_log_probs_, B, S);

            // compute_loss: subclass fills d_loss_grad_ with unscaled gradient
            float loss = compute_loss(d_log_probs_, d_loss_mask_, B, S);
            accum_loss += loss;

            // Scale gradient by 1/grad_accum_steps before accumulating
            int grid = (micro_T + 255) / 256;
            scale_grad_kernel<<<grid, 256>>>(d_loss_grad_, scale, micro_T);

            // Backward: *adds* into grads_ (not zeroed between micro-batches)
            qwen3_backward(model_, state_, grads_, d_loss_grad_);
        }

        // One optimizer step after all micro-batches
        optimizer_.step(grads_);

        return accum_loss / acc_steps;
    }

protected:
    Qwen3Model*      model_;
    AdamWOptimizer   optimizer_;
    LRScheduler      lr_scheduler_;
    DataLoader       data_loader_;
    Logger           logger_;
    TrainingConfig   config_;

    // GPU buffers — sized for one micro-batch (batch_size, S)
    Qwen3TrainState* state_        = nullptr;
    Qwen3Gradients*  grads_        = nullptr;
    float*           d_log_probs_  = nullptr;  // [micro_T]
    float*           d_loss_grad_  = nullptr;  // [micro_T] upstream gradient (pre-scale)
    int*             d_loss_mask_  = nullptr;  // [micro_T]
    int              buf_B_ = 0, buf_S_ = 0;

    // Subclass implements: compute loss on one micro-batch, write gradient into d_loss_grad_.
    // Do NOT scale by grad_accum_steps — the base class handles that.
    //   d_log_probs: [B*S] log-probs from forward (device, FP32)
    //   d_loss_mask: [B*S] 1 = compute loss, 0 = padding/prompt (device)
    virtual float compute_loss(const float* d_log_probs, const int* d_loss_mask,
                               int B, int S) = 0;

    virtual void save_checkpoint(int step) {
        printf("[Trainer] Checkpoint at step %d (not yet implemented)\n", step);
    }

private:
    // Lazy-allocate / reallocate GPU buffers when micro-batch shape changes
    void ensure_buffers(int B, int S) {
        if (state_ && buf_B_ == B && buf_S_ == S) return;

        free_buffers();

        int T = B * S;
        state_ = qwen3_train_state_alloc(model_->config, B, S);
        grads_ = qwen3_gradients_alloc(model_->config, T);

        CUDA_CHECK(cudaMalloc(&d_log_probs_, T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_loss_grad_, T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_loss_mask_, T * sizeof(int)));

        buf_B_ = B;
        buf_S_ = S;
    }

    void free_buffers() {
        if (state_)       { qwen3_train_state_free(state_); state_ = nullptr; }
        if (grads_)       { qwen3_gradients_free(grads_);   grads_ = nullptr; }
        if (d_log_probs_) { cudaFree(d_log_probs_); d_log_probs_ = nullptr; }
        if (d_loss_grad_) { cudaFree(d_loss_grad_); d_loss_grad_ = nullptr; }
        if (d_loss_mask_) { cudaFree(d_loss_mask_); d_loss_mask_ = nullptr; }
        buf_B_ = buf_S_ = 0;
    }
};
