// test_adamw.cu
//
// Tests the fused AdamW optimizer (AdamWOptimizer class).
//
// Validates:
//   1. CPU reference match for AdamW formula (FP32 norm weights)
//   2. CPU reference match for AdamW formula (FP16 linear weights)
//   3. Moments are non-zero after step
//   4. Step counter increments
//   5. Multi-step training reduces NLL (with full weight save/restore)
//
// Requires real weights at model_weights/Qwen3-0.6B.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "model/qwen3.h"
#include "training/optimizer.h"

static const char* MODEL_DIR = "model_weights/Qwen3-0.6B";

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t e = (call);                                            \
        if (e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                    cudaGetErrorString(e), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

static int g_fails = 0;
#define PASS(name)      printf("[PASS] %s\n", (name))
#define FAIL(name, ...) do { \
    printf("[FAIL] %s: ", (name)); printf(__VA_ARGS__); printf("\n"); g_fails++; \
} while (0)

// Helper: compute NLL = -sum(log_probs)
static double compute_nll(Qwen3Model* m, Qwen3TrainState* state,
                          const int* tokens, const int* targets,
                          float* d_log_probs, int B, int S) {
    int T = B * S;
    qwen3_forward(m, state, tokens, targets, d_log_probs, B, S);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> lp(T);
    CUDA_CHECK(cudaMemcpy(lp.data(), d_log_probs, T * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double nll = 0.0;
    for (int t = 0; t < T; t++) nll -= lp[t];
    return nll;
}

// Helper: one forward + backward + optimizer step
static void train_step(Qwen3Model* m, Qwen3TrainState* state,
                       Qwen3Gradients* grads, AdamWOptimizer* opt,
                       const int* tokens, const int* targets,
                       float* d_log_probs, float* d_upstream,
                       int B, int S) {
    qwen3_forward(m, state, tokens, targets, d_log_probs, B, S);
    CUDA_CHECK(cudaDeviceSynchronize());
    qwen3_gradients_zero(grads);
    qwen3_backward(m, state, grads, d_upstream);
    CUDA_CHECK(cudaDeviceSynchronize());
    opt->step(grads);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Helper: save all model weights to host vectors
struct WeightSnapshot {
    std::vector<float> final_norm;
    std::vector<std::vector<float>> input_norm, q_norm, k_norm, post_norm;
    std::vector<half> embed;
    struct LayerFP16 { std::vector<half> q, k, v, o, gate, up, down; };
    std::vector<LayerFP16> layers;
};

static WeightSnapshot save_weights(Qwen3Model* m) {
    const auto& c = m->config;
    WeightSnapshot s;
    int L = c.num_hidden_layers;

    s.final_norm.resize(c.hidden_size);
    CUDA_CHECK(cudaMemcpy(s.final_norm.data(), m->weights.final_norm,
                          c.hidden_size * sizeof(float), cudaMemcpyDeviceToHost));

    s.input_norm.resize(L); s.q_norm.resize(L);
    s.k_norm.resize(L); s.post_norm.resize(L);
    for (int l = 0; l < L; l++) {
        auto& lw = m->weights.layers[l];
        s.input_norm[l].resize(c.hidden_size);
        s.q_norm[l].resize(c.head_dim);
        s.k_norm[l].resize(c.head_dim);
        s.post_norm[l].resize(c.hidden_size);
        CUDA_CHECK(cudaMemcpy(s.input_norm[l].data(), lw.input_layernorm, c.hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(s.q_norm[l].data(), lw.q_norm, c.head_dim * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(s.k_norm[l].data(), lw.k_norm, c.head_dim * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(s.post_norm[l].data(), lw.post_attn_layernorm, c.hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    }

    long embed_sz = (long)c.vocab_size * c.hidden_size;
    s.embed.resize(embed_sz);
    CUDA_CHECK(cudaMemcpy(s.embed.data(), m->weights.embed_tokens, embed_sz * sizeof(half), cudaMemcpyDeviceToHost));

    s.layers.resize(L);
    for (int l = 0; l < L; l++) {
        auto& lw = m->weights.layers[l];
        auto& sv = s.layers[l];
        sv.q.resize(c.q_dim * c.hidden_size);
        sv.k.resize(c.kv_dim * c.hidden_size);
        sv.v.resize(c.kv_dim * c.hidden_size);
        sv.o.resize(c.hidden_size * c.q_dim);
        sv.gate.resize(c.intermediate_size * c.hidden_size);
        sv.up.resize(c.intermediate_size * c.hidden_size);
        sv.down.resize(c.hidden_size * c.intermediate_size);
        CUDA_CHECK(cudaMemcpy(sv.q.data(), lw.q_proj, sv.q.size() * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sv.k.data(), lw.k_proj, sv.k.size() * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sv.v.data(), lw.v_proj, sv.v.size() * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sv.o.data(), lw.o_proj, sv.o.size() * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sv.gate.data(), lw.gate_proj, sv.gate.size() * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sv.up.data(), lw.up_proj, sv.up.size() * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sv.down.data(), lw.down_proj, sv.down.size() * sizeof(half), cudaMemcpyDeviceToHost));
    }
    return s;
}

static void restore_weights(Qwen3Model* m, const WeightSnapshot& s) {
    const auto& c = m->config;
    int L = c.num_hidden_layers;

    CUDA_CHECK(cudaMemcpy(m->weights.final_norm, s.final_norm.data(),
                          c.hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    long embed_sz = (long)c.vocab_size * c.hidden_size;
    CUDA_CHECK(cudaMemcpy(m->weights.embed_tokens, s.embed.data(), embed_sz * sizeof(half), cudaMemcpyHostToDevice));

    for (int l = 0; l < L; l++) {
        auto& lw = m->weights.layers[l];
        CUDA_CHECK(cudaMemcpy(lw.input_layernorm, s.input_norm[l].data(), c.hidden_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lw.q_norm, s.q_norm[l].data(), c.head_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lw.k_norm, s.k_norm[l].data(), c.head_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lw.post_attn_layernorm, s.post_norm[l].data(), c.hidden_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lw.q_proj, s.layers[l].q.data(), s.layers[l].q.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lw.k_proj, s.layers[l].k.data(), s.layers[l].k.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lw.v_proj, s.layers[l].v.data(), s.layers[l].v.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lw.o_proj, s.layers[l].o.data(), s.layers[l].o.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lw.gate_proj, s.layers[l].gate.data(), s.layers[l].gate.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lw.up_proj, s.layers[l].up.data(), s.layers[l].up.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lw.down_proj, s.layers[l].down.data(), s.layers[l].down.size() * sizeof(half), cudaMemcpyHostToDevice));
    }
}

// =============================================================================
// Test 1: CPU reference for FP32 AdamW (final_norm)
// =============================================================================
static void test_cpu_reference_fp32(Qwen3Model* m) {
    const char* name = "CPU reference: FP32 AdamW (final_norm)";
    const Qwen3Config& c = m->config;

    std::vector<int> tokens = {785, 6722, 315, 9625, 374};
    int B = 1, S = (int)tokens.size(), T = B * S;
    std::vector<int> targets(T);
    for (int i = 0; i < T - 1; i++) targets[i] = tokens[i + 1];
    targets[T - 1] = 0;

    auto snap = save_weights(m);
    auto* state = qwen3_train_state_alloc(c, B, S);
    auto* grads = qwen3_gradients_alloc(c, T);
    auto* opt = new AdamWOptimizer(m, /*lr=*/1e-3f);

    float* d_log_probs;
    CUDA_CHECK(cudaMalloc(&d_log_probs, T * sizeof(float)));
    float* d_upstream;
    CUDA_CHECK(cudaMalloc(&d_upstream, T * sizeof(float)));
    std::vector<float> ones(T, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_upstream, ones.data(), T * sizeof(float), cudaMemcpyHostToDevice));

    // Forward + backward (don't step yet — read gradient first)
    qwen3_forward(m, state, tokens.data(), targets.data(), d_log_probs, B, S);
    CUDA_CHECK(cudaDeviceSynchronize());
    qwen3_gradients_zero(grads);
    qwen3_backward(m, state, grads, d_upstream);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read gradient before step
    std::vector<float> grad(c.hidden_size);
    CUDA_CHECK(cudaMemcpy(grad.data(), grads->dW_final_norm,
                          c.hidden_size * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference: step 1 of AdamW (no weight decay on norms)
    float _beta1 = opt->beta1, _beta2 = opt->beta2, _lr = opt->lr, _eps = opt->eps;
    float bc1 = 1.0f - _beta1;
    float bc2 = 1.0f - _beta2;
    std::vector<float> cpu_w(c.hidden_size);
    for (int i = 0; i < c.hidden_size; i++) {
        float g = grad[i];
        float mi = (1.0f - _beta1) * g;
        float vi = (1.0f - _beta2) * g * g;
        float m_hat = mi / bc1;
        float v_hat = vi / bc2;
        cpu_w[i] = snap.final_norm[i] - _lr * m_hat / (sqrtf(v_hat) + _eps);
    }

    // GPU step
    opt->step(grads);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_w(c.hidden_size);
    CUDA_CHECK(cudaMemcpy(gpu_w.data(), m->weights.final_norm,
                          c.hidden_size * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < c.hidden_size; i++)
        max_err = fmaxf(max_err, fabsf(gpu_w[i] - cpu_w[i]));

    printf("  max_err(GPU vs CPU) = %.8e\n", max_err);

    restore_weights(m, snap);

    if (max_err < 1e-6f) { PASS(name); }
    else { FAIL(name, "max_err=%.8e", max_err); }

    cudaFree(d_log_probs);
    cudaFree(d_upstream);
    qwen3_train_state_free(state);
    qwen3_gradients_free(grads);
    delete opt;
}

// =============================================================================
// Test 2: CPU reference for FP16 AdamW (q_proj layer 0)
// =============================================================================
static void test_cpu_reference_fp16(Qwen3Model* m) {
    const char* name = "CPU reference: FP16 AdamW (q_proj[0])";
    const Qwen3Config& c = m->config;

    std::vector<int> tokens = {785, 6722, 315, 9625, 374};
    int B = 1, S = (int)tokens.size(), T = B * S;
    std::vector<int> targets(T);
    for (int i = 0; i < T - 1; i++) targets[i] = tokens[i + 1];
    targets[T - 1] = 0;

    int qsz = c.q_dim * c.hidden_size;
    auto snap = save_weights(m);
    auto* state = qwen3_train_state_alloc(c, B, S);
    auto* grads = qwen3_gradients_alloc(c, T);
    auto* opt = new AdamWOptimizer(m, /*lr=*/1e-3f, /*beta1=*/0.9f, /*beta2=*/0.999f,
                                   /*eps=*/1e-8f, /*weight_decay=*/0.01f);

    float* d_log_probs;
    CUDA_CHECK(cudaMalloc(&d_log_probs, T * sizeof(float)));
    float* d_upstream;
    CUDA_CHECK(cudaMalloc(&d_upstream, T * sizeof(float)));
    std::vector<float> ones(T, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_upstream, ones.data(), T * sizeof(float), cudaMemcpyHostToDevice));

    qwen3_forward(m, state, tokens.data(), targets.data(), d_log_probs, B, S);
    CUDA_CHECK(cudaDeviceSynchronize());
    qwen3_gradients_zero(grads);
    qwen3_backward(m, state, grads, d_upstream);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read FP16 gradient
    std::vector<half> grad_h(qsz);
    CUDA_CHECK(cudaMemcpy(grad_h.data(), grads->dW_q_proj[0],
                          qsz * sizeof(half), cudaMemcpyDeviceToHost));

    // Read original FP16 weights (= master weights at this point)
    std::vector<float> w0(qsz);
    for (int i = 0; i < qsz; i++) w0[i] = __half2float(snap.layers[0].q[i]);

    // CPU reference
    float _beta1 = opt->beta1, _beta2 = opt->beta2, _lr = opt->lr;
    float _eps = opt->eps, _wd = opt->weight_decay;
    float bc1 = 1.0f - _beta1;
    float bc2 = 1.0f - _beta2;
    std::vector<float> cpu_master(qsz);
    std::vector<half> cpu_fp16(qsz);
    for (int i = 0; i < qsz; i++) {
        float g = __half2float(grad_h[i]);
        float mi = (1.0f - _beta1) * g;
        float vi = (1.0f - _beta2) * g * g;
        float m_hat = mi / bc1;
        float v_hat = vi / bc2;
        float w = w0[i] * (1.0f - _lr * _wd)
                  - _lr * m_hat / (sqrtf(v_hat) + _eps);
        cpu_master[i] = w;
        cpu_fp16[i] = __float2half(w);
    }

    // GPU step
    opt->step(grads);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read GPU FP16 result
    std::vector<half> gpu_fp16(qsz);
    CUDA_CHECK(cudaMemcpy(gpu_fp16.data(), m->weights.layers[0].q_proj,
                          qsz * sizeof(half), cudaMemcpyDeviceToHost));

    // Read GPU master copy
    std::vector<float> gpu_master(qsz);
    CUDA_CHECK(cudaMemcpy(gpu_master.data(), opt->get_fp16_master(0, 0),
                          qsz * sizeof(float), cudaMemcpyDeviceToHost));

    // Check master (FP32 exact match)
    float max_err_master = 0.0f;
    for (int i = 0; i < qsz; i++)
        max_err_master = fmaxf(max_err_master, fabsf(gpu_master[i] - cpu_master[i]));

    // Check FP16 (should be identical since both do __float2half of same value)
    int fp16_mismatches = 0;
    for (int i = 0; i < qsz; i++)
        if (__half2float(gpu_fp16[i]) != __half2float(cpu_fp16[i])) fp16_mismatches++;

    printf("  master max_err=%.8e  fp16 mismatches=%d/%d\n",
           max_err_master, fp16_mismatches, qsz);

    restore_weights(m, snap);

    if (max_err_master < 1e-6f && fp16_mismatches == 0) { PASS(name); }
    else { FAIL(name, "master_err=%.8e fp16_mismatch=%d", max_err_master, fp16_mismatches); }

    cudaFree(d_log_probs);
    cudaFree(d_upstream);
    qwen3_train_state_free(state);
    qwen3_gradients_free(grads);
    delete opt;
}

// =============================================================================
// Test 3: Moments are non-zero after step
// =============================================================================
static void test_moments_nonzero(Qwen3Model* m) {
    const char* name = "Moments non-zero after step";
    const Qwen3Config& c = m->config;

    std::vector<int> tokens = {785, 6722, 315, 9625, 374};
    int B = 1, S = (int)tokens.size(), T = B * S;
    std::vector<int> targets(T);
    for (int i = 0; i < T - 1; i++) targets[i] = tokens[i + 1];
    targets[T - 1] = 0;

    auto snap = save_weights(m);
    auto* state = qwen3_train_state_alloc(c, B, S);
    auto* grads = qwen3_gradients_alloc(c, T);
    auto* opt = new AdamWOptimizer(m);

    float* d_log_probs;
    CUDA_CHECK(cudaMalloc(&d_log_probs, T * sizeof(float)));
    float* d_upstream;
    CUDA_CHECK(cudaMalloc(&d_upstream, T * sizeof(float)));
    std::vector<float> ones(T, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_upstream, ones.data(), T * sizeof(float), cudaMemcpyHostToDevice));

    train_step(m, state, grads, opt, tokens.data(), targets.data(),
               d_log_probs, d_upstream, B, S);

    // Check final_norm moments (FP32 path)
    std::vector<float> m_buf(c.hidden_size), v_buf(c.hidden_size);
    CUDA_CHECK(cudaMemcpy(m_buf.data(), opt->get_final_norm_m(), c.hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_buf.data(), opt->get_final_norm_v(), c.hidden_size * sizeof(float), cudaMemcpyDeviceToHost));

    float m_sum = 0, v_sum = 0;
    bool finite = true;
    for (int i = 0; i < c.hidden_size; i++) {
        if (!std::isfinite(m_buf[i]) || !std::isfinite(v_buf[i])) finite = false;
        m_sum += fabsf(m_buf[i]);
        v_sum += fabsf(v_buf[i]);
    }

    // Check q_proj[0] moments (FP16 path)
    int check_n = 1024;
    std::vector<float> qm(check_n), qv(check_n);
    CUDA_CHECK(cudaMemcpy(qm.data(), opt->get_fp16_m(0, 0), check_n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qv.data(), opt->get_fp16_v(0, 0), check_n * sizeof(float), cudaMemcpyDeviceToHost));

    float qm_sum = 0, qv_sum = 0;
    for (int i = 0; i < check_n; i++) {
        if (!std::isfinite(qm[i]) || !std::isfinite(qv[i])) finite = false;
        qm_sum += fabsf(qm[i]);
        qv_sum += fabsf(qv[i]);
    }

    printf("  final_norm: m=%.6f v=%.6f  q_proj[0]: m=%.6f v=%.6f\n",
           m_sum, v_sum, qm_sum, qv_sum);

    restore_weights(m, snap);

    if (finite && m_sum > 1e-10f && v_sum > 1e-10f && qm_sum > 1e-10f && qv_sum > 1e-10f) {
        PASS(name);
    } else {
        FAIL(name, "finite=%d m=%.8f v=%.8f qm=%.8f qv=%.8f",
             finite, m_sum, v_sum, qm_sum, qv_sum);
    }

    cudaFree(d_log_probs);
    cudaFree(d_upstream);
    qwen3_train_state_free(state);
    qwen3_gradients_free(grads);
    delete opt;
}

// =============================================================================
// Test 4: Step counter increments
// =============================================================================
static void test_step_counter(Qwen3Model* m) {
    const char* name = "Step counter increments";
    const Qwen3Config& c = m->config;

    std::vector<int> tokens = {785, 6722, 315};
    int B = 1, S = (int)tokens.size(), T = B * S;
    std::vector<int> targets = {6722, 315, 0};

    auto snap = save_weights(m);
    auto* state = qwen3_train_state_alloc(c, B, S);
    auto* grads = qwen3_gradients_alloc(c, T);
    auto* opt = new AdamWOptimizer(m);

    float* d_log_probs;
    CUDA_CHECK(cudaMalloc(&d_log_probs, T * sizeof(float)));
    float* d_upstream;
    CUDA_CHECK(cudaMalloc(&d_upstream, T * sizeof(float)));
    std::vector<float> ones(T, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_upstream, ones.data(), T * sizeof(float), cudaMemcpyHostToDevice));

    bool ok = (opt->get_step() == 0);

    for (int i = 0; i < 3; i++)
        train_step(m, state, grads, opt, tokens.data(), targets.data(),
                   d_log_probs, d_upstream, B, S);

    ok = ok && (opt->get_step() == 3);

    restore_weights(m, snap);

    if (ok) { PASS(name); }
    else { FAIL(name, "step=%d, expected 3", opt->get_step()); }

    cudaFree(d_log_probs);
    cudaFree(d_upstream);
    qwen3_train_state_free(state);
    qwen3_gradients_free(grads);
    delete opt;
}

// =============================================================================
// Test 5: Multi-step NLL reduction
// =============================================================================
static void test_multi_step(Qwen3Model* m) {
    const char* name = "Multi-step: 5 steps reduce NLL";
    const Qwen3Config& c = m->config;

    std::vector<int> tokens = {785, 6722, 315, 9625, 374};
    int B = 1, S = (int)tokens.size(), T = B * S;
    std::vector<int> targets(T);
    for (int i = 0; i < T - 1; i++) targets[i] = tokens[i + 1];
    targets[T - 1] = 12095;

    auto snap = save_weights(m);
    auto* state = qwen3_train_state_alloc(c, B, S);
    auto* grads = qwen3_gradients_alloc(c, T);
    auto* opt = new AdamWOptimizer(m, /*lr=*/1e-4f);

    float* d_log_probs;
    CUDA_CHECK(cudaMalloc(&d_log_probs, T * sizeof(float)));
    float* d_upstream;
    CUDA_CHECK(cudaMalloc(&d_upstream, T * sizeof(float)));
    // d_upstream = -1 → backward gives d(NLL)/dw → AdamW minimizes NLL
    std::vector<float> neg_ones(T, -1.0f);
    CUDA_CHECK(cudaMemcpy(d_upstream, neg_ones.data(), T * sizeof(float), cudaMemcpyHostToDevice));

    const int N = 5;
    double nlls[N + 1];
    nlls[0] = compute_nll(m, state, tokens.data(), targets.data(), d_log_probs, B, S);

    for (int s = 0; s < N; s++) {
        train_step(m, state, grads, opt, tokens.data(), targets.data(),
                   d_log_probs, d_upstream, B, S);
        nlls[s + 1] = compute_nll(m, state, tokens.data(), targets.data(),
                                  d_log_probs, B, S);
    }

    printf("  NLLs:");
    for (int s = 0; s <= N; s++) printf(" %.4f", nlls[s]);
    printf("\n");

    restore_weights(m, snap);

    if (nlls[N] < nlls[0]) { PASS(name); }
    else { FAIL(name, "NLL did not decrease: %.6f -> %.6f", nlls[0], nlls[N]); }

    cudaFree(d_log_probs);
    cudaFree(d_upstream);
    qwen3_train_state_free(state);
    qwen3_gradients_free(grads);
    delete opt;
}

// =============================================================================
// main
// =============================================================================
int main() {
    if (!std::filesystem::exists(MODEL_DIR)) {
        printf("Model weights not found at %s — skipping tests.\n", MODEL_DIR);
        return 0;
    }

    printf("=== AdamW Optimizer tests ===\n\n");

    Qwen3Model* m = qwen3_load(MODEL_DIR, /*max_batch=*/4, /*max_seq=*/64);

    test_cpu_reference_fp32(m);
    test_cpu_reference_fp16(m);
    test_moments_nonzero(m);
    test_step_counter(m);
    test_multi_step(m);

    qwen3_free(m);

    printf("\n%s\n", g_fails ? "SOME TESTS FAILED" : "All tests PASSED.");
    return g_fails ? 1 : 0;
}
