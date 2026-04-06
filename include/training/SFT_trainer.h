#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include "trainer.h"

// ============================================================
// Masked NLL loss kernel (fused loss + gradient)
//
// SFT cross-entropy loss = -mean(log_probs * mask)
//
// Two-pass in one kernel launch:
//   Pass 1 (block reduction): each block sums log_probs*mask and mask count
//   Pass 2 (single block):    computes mean loss and writes gradient
//
// For simplicity, we use atomics for the partial sums, then a second
// kernel to finalize the gradient. But since T is small (batch*seq,
// typically <16K), a single-kernel approach with atomics is fine.
// ============================================================

// Kernel: compute partial sums of -log_probs*mask and mask count
__global__ static void nll_loss_reduce_kernel(
    const float* __restrict__ log_probs,   // [T]
    const int*   __restrict__ loss_mask,    // [T]
    float*       __restrict__ loss_sum,     // [1] atomic accumulator (init 0)
    int*         __restrict__ mask_count,   // [1] atomic accumulator (init 0)
    int T
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T) return;

    if (loss_mask[i]) {
        atomicAdd(loss_sum, -log_probs[i]);
        atomicAdd(mask_count, 1);
    }
}

// Kernel: write gradient = -mask[i] / num_loss_tokens
__global__ static void nll_loss_grad_kernel(
    float*       __restrict__ grad,         // [T] output gradient
    const int*   __restrict__ loss_mask,     // [T]
    const int*   __restrict__ mask_count,    // [1] total masked tokens
    int T
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T) return;

    int n = *mask_count;
    if (n > 0 && loss_mask[i]) {
        grad[i] = -1.0f / (float)n;
    } else {
        grad[i] = 0.0f;
    }
}

class SFTTrainer : public Trainer {
public:
    SFTTrainer(const TrainingConfig& config, Qwen3Model* model)
        : Trainer(config, model)
    {
        // Allocate scalar accumulators on device
        CUDA_CHECK(cudaMalloc(&d_loss_sum_,   sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mask_count_, sizeof(int)));
    }

    ~SFTTrainer() override {
        if (d_loss_sum_)   { cudaFree(d_loss_sum_);   d_loss_sum_   = nullptr; }
        if (d_mask_count_) { cudaFree(d_mask_count_); d_mask_count_ = nullptr; }
    }

protected:
    // Masked mean NLL loss: -sum(log_probs * mask) / sum(mask)
    // Writes gradient into d_loss_grad_.
    float compute_loss(const float* d_log_probs, const int* d_loss_mask,
                       int B, int S) override {
        int T = B * S;
        int block = 256;
        int grid = (T + block - 1) / block;

        // Zero accumulators
        CUDA_CHECK(cudaMemset(d_loss_sum_,   0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_mask_count_, 0, sizeof(int)));

        // Reduce: sum of -log_probs*mask and count of mask tokens
        nll_loss_reduce_kernel<<<grid, block>>>(
            d_log_probs, d_loss_mask, d_loss_sum_, d_mask_count_, T);

        // Write gradient: -mask[i] / count
        nll_loss_grad_kernel<<<grid, block>>>(
            d_loss_grad_, d_loss_mask, d_mask_count_, T);

        // Read scalar loss to host
        float h_loss_sum;
        int   h_mask_count;
        CUDA_CHECK(cudaMemcpy(&h_loss_sum,   d_loss_sum_,   sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_mask_count, d_mask_count_, sizeof(int),   cudaMemcpyDeviceToHost));

        return (h_mask_count > 0) ? h_loss_sum / h_mask_count : 0.0f;
    }

private:
    float* d_loss_sum_   = nullptr;  // [1] device scalar
    int*   d_mask_count_ = nullptr;  // [1] device scalar
};
