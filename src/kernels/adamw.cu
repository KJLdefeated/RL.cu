#include "kernels/adamw.cuh"
#include <cstdio>

// ============================================================
// Kernel: Fused AdamW for FP16 weights with FP32 master copy
// ============================================================
__global__ static void adamw_fp16_kernel(
    float*       __restrict__ master_w,
    half*        __restrict__ model_w,
    const half*  __restrict__ grad,
    float*       __restrict__ m,
    float*       __restrict__ v,
    int          n,
    float        lr,
    float        beta1,
    float        beta2,
    float        eps,
    float        weight_decay,
    float        bc1_inv,     // 1 / (1 - beta1^t)
    float        bc2_inv      // 1 / (1 - beta2^t)
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = __half2float(grad[i]);
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;

    m[i] = mi;
    v[i] = vi;

    float m_hat = mi * bc1_inv;
    float v_hat = vi * bc2_inv;

    float w = master_w[i];
    w = w * (1.0f - lr * weight_decay) - lr * m_hat / (sqrtf(v_hat) + eps);

    master_w[i] = w;
    model_w[i]  = __float2half(w);
}

void launch_adamw_fp16(
    float* master_w, half* model_w, const half* grad,
    float* m, float* v, int n,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2,
    cudaStream_t stream
) {
    if (n <= 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    float bc1_inv = 1.0f / bias_correction1;
    float bc2_inv = 1.0f / bias_correction2;
    adamw_fp16_kernel<<<grid, block, 0, stream>>>(
        master_w, model_w, grad, m, v, n,
        lr, beta1, beta2, eps, weight_decay, bc1_inv, bc2_inv
    );
}

// ============================================================
// Kernel: Fused AdamW for FP32 weights (norms)
// ============================================================
__global__ static void adamw_fp32_kernel(
    float*       __restrict__ w,
    const float* __restrict__ grad,
    float*       __restrict__ m,
    float*       __restrict__ v,
    int          n,
    float        lr,
    float        beta1,
    float        beta2,
    float        eps,
    float        weight_decay,
    float        bc1_inv,
    float        bc2_inv
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;

    m[i] = mi;
    v[i] = vi;

    float m_hat = mi * bc1_inv;
    float v_hat = vi * bc2_inv;

    w[i] = w[i] * (1.0f - lr * weight_decay) - lr * m_hat / (sqrtf(v_hat) + eps);
}

void launch_adamw_fp32(
    float* w, const float* grad, float* m, float* v, int n,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2,
    cudaStream_t stream
) {
    if (n <= 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    float bc1_inv = 1.0f / bias_correction1;
    float bc2_inv = 1.0f / bias_correction2;
    adamw_fp32_kernel<<<grid, block, 0, stream>>>(
        w, grad, m, v, n,
        lr, beta1, beta2, eps, weight_decay, bc1_inv, bc2_inv
    );
}
