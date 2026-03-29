#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cassert>
#include "model/config.h"
#include "model/weights.h"
#include "kernels/adamw.cuh"
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

// =============================================================================
// AdamW Optimizer — flat buffer pattern
//
// All optimizer states are contiguous flat buffers co-indexed with the model
// weight pools (Qwen3Weights::fp16_pool / fp32_pool) and gradient pools
// (Qwen3Gradients::half_pool / float_pool).
//
// Layout (must match across weights, gradients, optimizer):
//   FP16: per-layer [q, k, v, o, gate, up, down], then embed
//   FP32: per-layer [input_norm, q_norm, k_norm, post_attn_norm], then final_norm
//
// step() does exactly 2 kernel launches:
//   1. FP16 params (projections + embed) — with weight decay
//   2. FP32 params (norms)               — no weight decay
//
// Usage:
//   AdamWOptimizer optimizer(model, lr=1e-4);
//   optimizer.step(grads);     // 2 kernel launches total
//   optimizer.set_lr(new_lr);
// =============================================================================

// Helper: copy FP16 weights to FP32 master copy (flat)
__global__ static void fp16_to_fp32_kernel(
    float*      __restrict__ dst,
    const half* __restrict__ src,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

static void fp16_to_fp32(float* dst, const half* src, int n, cudaStream_t stream) {
    fp16_to_fp32_kernel<<<(n + 255) / 256, 256, 0, stream>>>(dst, src, n);
}

class AdamWOptimizer {
public:
    // Hyperparameters (public, like torch)
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;

    AdamWOptimizer() : lr(1e-5f), beta1(0.9f), beta2(0.999f), eps(1e-8f), weight_decay(0.01f) {}

    AdamWOptimizer(Qwen3Model* model,
                   float lr           = 1e-4f,
                   float beta1        = 0.9f,
                   float beta2        = 0.999f,
                   float eps          = 1e-8f,
                   float weight_decay = 0.01f)
        : lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay)
    {
        init(model);
    }

    ~AdamWOptimizer() { free(); }

    // No copy (owns GPU memory)
    AdamWOptimizer(const AdamWOptimizer&) = delete;
    AdamWOptimizer& operator=(const AdamWOptimizer&) = delete;

    // Move
    AdamWOptimizer(AdamWOptimizer&& o) noexcept { move_from(o); }
    AdamWOptimizer& operator=(AdamWOptimizer&& o) noexcept {
        if (this != &o) { free(); move_from(o); }
        return *this;
    }

    void set_lr(float new_lr) { lr = new_lr; }

    int get_step() const { return step_; }

    // --- Flat buffer accessors ---
    float* fp16_master()       { return fp16_master_; }
    float* fp16_m()            { return fp16_m_; }
    float* fp16_v()            { return fp16_v_; }
    float* fp32_m()            { return fp32_m_; }
    float* fp32_v()            { return fp32_v_; }
    size_t num_fp16_params() const { return n16_; }
    size_t num_fp32_params() const { return n32_; }

    // --- Test accessors (compute offsets into flat buffers) ---
    // proj: 0=q, 1=k, 2=v, 3=o, 4=gate, 5=up, 6=down
    float* get_fp16_master(int layer, int proj) const { return fp16_master_ + fp16_offset(layer, proj); }
    float* get_fp16_m(int layer, int proj)      const { return fp16_m_      + fp16_offset(layer, proj); }
    float* get_fp16_v(int layer, int proj)      const { return fp16_v_      + fp16_offset(layer, proj); }
    // norm: 0=input, 1=q, 2=k, 3=post_attn
    float* get_fp32_m_norm(int layer, int norm) const { return fp32_m_ + fp32_offset(layer, norm); }
    float* get_fp32_v_norm(int layer, int norm) const { return fp32_v_ + fp32_offset(layer, norm); }
    float* get_final_norm_m() const { return fp32_m_ + fp32_final_norm_offset(); }
    float* get_final_norm_v() const { return fp32_v_ + fp32_final_norm_offset(); }

    // --- Optimizer step: 2 kernel launches ---
    void step(Qwen3Gradients* grads, cudaStream_t stream = 0) {
        step_++;
        float bc1 = 1.0f - powf(beta1, (float)step_);
        float bc2 = 1.0f - powf(beta2, (float)step_);

        // Launch 1: all FP16 params (projections + embed), with weight decay
        launch_adamw_fp16(
            fp16_master_, model_->weights.fp16_pool, grads->half_pool,
            fp16_m_, fp16_v_, (int)n16_,
            lr, beta1, beta2, eps, weight_decay, bc1, bc2, stream);

        // Launch 2: all FP32 params (norms), no weight decay
        launch_adamw_fp32(
            model_->weights.fp32_pool, (const float*)grads->float_pool,
            fp32_m_, fp32_v_, (int)n32_,
            lr, beta1, beta2, eps, /*weight_decay=*/0.0f, bc1, bc2, stream);
    }

private:
    Qwen3Model* model_ = nullptr;
    int step_ = 0;

    // Flat GPU allocations
    float* fp16_master_ = nullptr;  // [n16_] FP32 master copy of FP16 weights
    float* fp16_m_      = nullptr;  // [n16_] first moment
    float* fp16_v_      = nullptr;  // [n16_] second moment
    float* fp32_m_      = nullptr;  // [n32_] first moment for FP32 params
    float* fp32_v_      = nullptr;  // [n32_] second moment for FP32 params
    float* fp16_pool_   = nullptr;  // single alloc: [master | m | v]
    float* fp32_pool_   = nullptr;  // single alloc: [m | v]
    size_t n16_ = 0;                // total FP16 parameter count
    size_t n32_ = 0;                // total FP32 parameter count

    // Cached sizes for offset computation
    size_t per_layer_fp16_ = 0;
    size_t proj_sizes_[7] = {};     // q, k, v, o, gate, up, down
    size_t per_layer_fp32_ = 0;
    size_t norm_sizes_[4] = {};     // input_norm, q_norm, k_norm, post_attn_norm

    // Compute offset of (layer, proj) into the flat FP16 buffer
    size_t fp16_offset(int layer, int proj) const {
        size_t off = (size_t)layer * per_layer_fp16_;
        for (int p = 0; p < proj; p++) off += proj_sizes_[p];
        return off;
    }

    // Compute offset of (layer, norm) into the flat FP32 buffer
    size_t fp32_offset(int layer, int norm) const {
        size_t off = (size_t)layer * per_layer_fp32_;
        for (int n = 0; n < norm; n++) off += norm_sizes_[n];
        return off;
    }

    size_t fp32_final_norm_offset() const {
        return per_layer_fp32_ * model_->config.num_hidden_layers;
    }

    void init(Qwen3Model* model) {
        model_ = model;
        const Qwen3Config& c = model->config;
        const int L = c.num_hidden_layers;
        const int H = c.hidden_size;

        // Cache per-tensor sizes (same order as pool layout)
        proj_sizes_[0] = (size_t)c.q_dim * H;          // q
        proj_sizes_[1] = (size_t)c.kv_dim * H;         // k
        proj_sizes_[2] = (size_t)c.kv_dim * H;         // v
        proj_sizes_[3] = (size_t)H * c.q_dim;          // o
        proj_sizes_[4] = (size_t)c.intermediate_size * H; // gate
        proj_sizes_[5] = (size_t)c.intermediate_size * H; // up
        proj_sizes_[6] = (size_t)H * c.intermediate_size; // down

        per_layer_fp16_ = 0;
        for (int i = 0; i < 7; i++) per_layer_fp16_ += proj_sizes_[i];

        const size_t embed_sz = (size_t)c.vocab_size * H;
        n16_ = per_layer_fp16_ * L + embed_sz;

        norm_sizes_[0] = H;            // input_norm
        norm_sizes_[1] = c.head_dim;   // q_norm
        norm_sizes_[2] = c.head_dim;   // k_norm
        norm_sizes_[3] = H;            // post_attn_norm

        per_layer_fp32_ = 0;
        for (int i = 0; i < 4; i++) per_layer_fp32_ += norm_sizes_[i];

        const size_t final_norm_sz = H;
        n32_ = per_layer_fp32_ * L + final_norm_sz;

        // Verify sizes match weight pools
        assert(n16_ == model->weights.fp16_pool_elems);
        assert(n32_ == model->weights.fp32_pool_elems);

        // Allocate: FP16 states = [master_w | m | v], FP32 states = [m | v]
        size_t fp16_bytes = n16_ * 3 * sizeof(float);
        size_t fp32_bytes = n32_ * 2 * sizeof(float);

        CUDA_CHECK(cudaMalloc(&fp16_pool_, fp16_bytes));
        CUDA_CHECK(cudaMalloc(&fp32_pool_, fp32_bytes));
        CUDA_CHECK(cudaMemset(fp16_pool_, 0, fp16_bytes));
        CUDA_CHECK(cudaMemset(fp32_pool_, 0, fp32_bytes));

        // Partition flat pools
        fp16_master_ = fp16_pool_;
        fp16_m_      = fp16_master_ + n16_;
        fp16_v_      = fp16_m_      + n16_;

        fp32_m_ = fp32_pool_;
        fp32_v_ = fp32_m_ + n32_;

        // Initialize master weights: one flat copy from the weight pool
        fp16_to_fp32(fp16_master_, model->weights.fp16_pool, (int)n16_, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("[AdamW] Flat buffers: fp16=%zu params (%.1f MB state)  fp32=%zu params (%.2f MB state)  "
               "(lr=%.1e wd=%.2f)\n",
               n16_, fp16_bytes / (1024.0 * 1024.0),
               n32_, fp32_bytes / (1024.0 * 1024.0),
               lr, weight_decay);
    }

    void free() {
        if (fp16_pool_) { cudaFree(fp16_pool_); fp16_pool_ = nullptr; }
        if (fp32_pool_) { cudaFree(fp32_pool_); fp32_pool_ = nullptr; }
        fp16_master_ = fp16_m_ = fp16_v_ = nullptr;
        fp32_m_ = fp32_v_ = nullptr;
    }

    void move_from(AdamWOptimizer& o) {
        lr = o.lr; beta1 = o.beta1; beta2 = o.beta2; eps = o.eps; weight_decay = o.weight_decay;
        model_ = o.model_; step_ = o.step_;
        fp16_pool_ = o.fp16_pool_; fp32_pool_ = o.fp32_pool_;
        fp16_master_ = o.fp16_master_; fp16_m_ = o.fp16_m_; fp16_v_ = o.fp16_v_;
        fp32_m_ = o.fp32_m_; fp32_v_ = o.fp32_v_;
        n16_ = o.n16_; n32_ = o.n32_;
        per_layer_fp16_ = o.per_layer_fp16_; per_layer_fp32_ = o.per_layer_fp32_;
        memcpy(proj_sizes_, o.proj_sizes_, sizeof(proj_sizes_));
        memcpy(norm_sizes_, o.norm_sizes_, sizeof(norm_sizes_));
        // Null out source
        o.fp16_pool_ = nullptr; o.fp32_pool_ = nullptr;
        o.fp16_master_ = o.fp16_m_ = o.fp16_v_ = nullptr;
        o.fp32_m_ = o.fp32_v_ = nullptr;
    }
};
