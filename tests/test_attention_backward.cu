// test_attention_backward.cu
// Validates FA2 backward (dQ, dK, dV) against a CPU FP32 naive reference.
//
// The backward recomputes P from saved LSE, then:
//   D[i]    = dot(O[i], dO[i])
//   dP[i,j] = dot(dO[i], V[j])
//   dS[i,j] = P[i,j] * (dP[i,j] - D[i])
//   dQ[i]   = scale * Σ_j dS[i,j] * K[j]
//   dK[j]   = scale * Σ_i dS[i,j] * Q[i]     (summed over GQA Q heads)
//   dV[j]   = Σ_i P[i,j] * dO[i]              (summed over GQA Q heads)
//
// Pass criterion: max absolute error < 5e-2
//   (backward accumulates more FP16 rounding than forward)
//
// Test cases:
//   1. B=1, S=16,  H_q=2, H_kv=1  — small, GQA 2:1
//   2. B=1, S=32,  H_q=2, H_kv=2  — MHA (no GQA)
//   3. B=1, S=64,  H_q=4, H_kv=2  — single KV tile (Bc=64)
//   4. B=2, S=48,  H_q=4, H_kv=2  — batched
//   5. B=1, S=128, H_q=16, H_kv=8 — Qwen3 head config

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/attention.cuh"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

static unsigned int lcg_state = 77777u;
static float lcg_randf() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(lcg_state >> 1) / (float)0x7fffffffu) * 2.0f - 1.0f;
}

// ---------------------------------------------------------------------------
// CPU reference: naive causal attention backward
// Q, K, V, O, dO layout: see forward
// dQ: [B, S, H_q, D]
// dK: [B, S, H_kv, D]
// dV: [B, S, H_kv, D]
// ---------------------------------------------------------------------------
static void ref_attention_backward(
    float* dQ, float* dK, float* dV,
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO_arr,
    int B, int S, int H_q, int H_kv, int D
) {
    const float scale = 1.0f / sqrtf((float)D);
    float* scores = new float[S];

    memset(dQ, 0, (long)B * S * H_q  * D * sizeof(float));
    memset(dK, 0, (long)B * S * H_kv * D * sizeof(float));
    memset(dV, 0, (long)B * S * H_kv * D * sizeof(float));

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H_q; h++) {
            const int hkv = h * H_kv / H_q;

            for (int i = 0; i < S; i++) {
                const float* qi  = Q  + ((long)(b*S+i)*H_q +h)*D;
                const float* oi  = O  + ((long)(b*S+i)*H_q +h)*D;
                const float* doi = dO_arr + ((long)(b*S+i)*H_q +h)*D;
                float*       dqi = dQ + ((long)(b*S+i)*H_q +h)*D;

                // Recompute softmax P[i,:]
                float max_s = -1e30f;
                for (int j = 0; j <= i; j++) {
                    const float* kj = K + ((long)(b*S+j)*H_kv+hkv)*D;
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++) dot += qi[d] * kj[d];
                    scores[j] = dot * scale;
                    if (scores[j] > max_s) max_s = scores[j];
                }
                float sum_e = 0.0f;
                for (int j = 0; j <= i; j++) {
                    scores[j] = expf(scores[j] - max_s);
                    sum_e += scores[j];
                }
                for (int j = 0; j <= i; j++) scores[j] /= sum_e;

                // D[i] = dot(O[i], dO[i])
                float D_val = 0.0f;
                for (int d = 0; d < D; d++) D_val += oi[d] * doi[d];

                // Backward through each (i, j) pair
                for (int j = 0; j <= i; j++) {
                    const float* kj  = K  + ((long)(b*S+j)*H_kv+hkv)*D;
                    const float* vj  = V  + ((long)(b*S+j)*H_kv+hkv)*D;
                    float*       dkj = dK + ((long)(b*S+j)*H_kv+hkv)*D;
                    float*       dvj = dV + ((long)(b*S+j)*H_kv+hkv)*D;

                    // dP[i,j] = dot(dO[i], V[j])
                    float dp = 0.0f;
                    for (int d = 0; d < D; d++) dp += doi[d] * vj[d];

                    // dS[i,j] = P[i,j] * (dP[i,j] - D[i])
                    float ds = scores[j] * (dp - D_val);

                    // dQ[i] += dS * K[j] * scale
                    for (int d = 0; d < D; d++)
                        dqi[d] += ds * kj[d] * scale;

                    // dK[j] += dS * Q[i] * scale  (accumulated over Q heads in GQA)
                    for (int d = 0; d < D; d++)
                        dkj[d] += ds * qi[d] * scale;

                    // dV[j] += P[i,j] * dO[i]  (accumulated over Q heads in GQA)
                    for (int d = 0; d < D; d++)
                        dvj[d] += scores[j] * doi[d];
                }
            }
        }
    }

    delete[] scores;
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
static bool run_test(
    const char* name, int B, int S, int H_q, int H_kv,
    int D = 128, float tol = 5e-2f
) {
    const long N_q  = (long)B * S * H_q  * D;
    const long N_kv = (long)B * S * H_kv * D;
    const long lse_sz = (long)B * S * H_q;

    // Host FP32 + FP16 round-tripped
    float* h_Q_f32  = new float[N_q];
    float* h_K_f32  = new float[N_kv];
    float* h_V_f32  = new float[N_kv];
    float* h_dO_f32 = new float[N_q];

    half* h_Q  = new half[N_q];
    half* h_K  = new half[N_kv];
    half* h_V  = new half[N_kv];
    half* h_dO = new half[N_q];

    // Scale inputs down to prevent softmax saturation
    for (long i = 0; i < N_q;  i++) h_Q[i]  = __float2half(lcg_randf() * 0.5f);
    for (long i = 0; i < N_kv; i++) h_K[i]  = __float2half(lcg_randf() * 0.5f);
    for (long i = 0; i < N_kv; i++) h_V[i]  = __float2half(lcg_randf() * 0.5f);
    for (long i = 0; i < N_q;  i++) h_dO[i] = __float2half(lcg_randf() * 0.5f);

    // Round-trip for CPU reference
    for (long i = 0; i < N_q;  i++) h_Q_f32[i]  = __half2float(h_Q[i]);
    for (long i = 0; i < N_kv; i++) h_K_f32[i]  = __half2float(h_K[i]);
    for (long i = 0; i < N_kv; i++) h_V_f32[i]  = __half2float(h_V[i]);
    for (long i = 0; i < N_q;  i++) h_dO_f32[i] = __half2float(h_dO[i]);

    // CPU forward to get O (for D computation in backward)
    float* h_O_f32 = new float[N_q];
    {
        const float scale = 1.0f / sqrtf((float)D);
        float* scores = new float[S];
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H_q; h++) {
                const int hkv = h * H_kv / H_q;
                for (int i = 0; i < S; i++) {
                    const float* qi = h_Q_f32 + ((long)(b*S+i)*H_q+h)*D;
                    float max_s = -1e30f;
                    for (int j = 0; j <= i; j++) {
                        const float* kj = h_K_f32 + ((long)(b*S+j)*H_kv+hkv)*D;
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++) dot += qi[d] * kj[d];
                        scores[j] = dot * scale;
                        if (scores[j] > max_s) max_s = scores[j];
                    }
                    float sum_e = 0.0f;
                    for (int j = 0; j <= i; j++) {
                        scores[j] = expf(scores[j] - max_s);
                        sum_e += scores[j];
                    }
                    for (int j = 0; j <= i; j++) scores[j] /= sum_e;

                    float* oi = h_O_f32 + ((long)(b*S+i)*H_q+h)*D;
                    for (int d = 0; d < D; d++) {
                        float acc = 0.0f;
                        for (int j = 0; j <= i; j++) {
                            const float* vj = h_V_f32 + ((long)(b*S+j)*H_kv+hkv)*D;
                            acc += scores[j] * vj[d];
                        }
                        oi[d] = acc;
                    }
                }
            }
        }
        delete[] scores;
    }

    // CPU backward reference
    float* h_ref_dQ = new float[N_q];
    float* h_ref_dK = new float[N_kv];
    float* h_ref_dV = new float[N_kv];
    ref_attention_backward(h_ref_dQ, h_ref_dK, h_ref_dV,
                           h_Q_f32, h_K_f32, h_V_f32, h_O_f32, h_dO_f32,
                           B, S, H_q, H_kv, D);

    // GPU: allocate
    half *d_Q, *d_K, *d_V, *d_O, *d_dO, *d_dQ, *d_dK, *d_dV;
    float *d_lse, *d_D;

    CUDA_CHECK(cudaMalloc(&d_Q,   N_q  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K,   N_kv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V,   N_kv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_O,   N_q  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dO,  N_q  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dQ,  N_q  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dK,  N_kv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dV,  N_kv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_lse, lse_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_D,   lse_sz * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q,  h_Q,  N_q  * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K,  h_K,  N_kv * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V,  h_V,  N_kv * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dO, h_dO, N_q  * sizeof(half), cudaMemcpyHostToDevice));

    // GPU forward (to get O and LSE on device)
    launch_flash_attention_prefill(d_Q, d_K, d_V, d_O,
                                   B, S, H_q, H_kv, D, 0, d_lse);
    CUDA_CHECK(cudaDeviceSynchronize());

    // GPU backward
    launch_flash_attention_backward(d_Q, d_K, d_V, d_O, d_dO,
                                    d_lse, d_D, d_dQ, d_dK, d_dV,
                                    B, S, H_q, H_kv, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    half* h_dQ = new half[N_q];
    half* h_dK = new half[N_kv];
    half* h_dV = new half[N_kv];
    CUDA_CHECK(cudaMemcpy(h_dQ, d_dQ, N_q  * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dK, d_dK, N_kv * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dV, d_dV, N_kv * sizeof(half), cudaMemcpyDeviceToHost));

    // Compare
    float max_err_dQ = 0.0f, max_err_dK = 0.0f, max_err_dV = 0.0f;
    for (long i = 0; i < N_q; i++) {
        float diff = fabsf(__half2float(h_dQ[i]) - h_ref_dQ[i]);
        if (diff > max_err_dQ) max_err_dQ = diff;
    }
    for (long i = 0; i < N_kv; i++) {
        float diff = fabsf(__half2float(h_dK[i]) - h_ref_dK[i]);
        if (diff > max_err_dK) max_err_dK = diff;
    }
    for (long i = 0; i < N_kv; i++) {
        float diff = fabsf(__half2float(h_dV[i]) - h_ref_dV[i]);
        if (diff > max_err_dV) max_err_dV = diff;
    }

    // Scale tolerance with S (more accumulation = more FP16 error)
    float scaled_tol = tol * sqrtf((float)S / 16.0f);
    bool passed = (max_err_dQ < scaled_tol) &&
                  (max_err_dK < scaled_tol) &&
                  (max_err_dV < scaled_tol);

    printf("[%s] %-52s dQ=%.5f dK=%.5f dV=%.5f (tol=%.3f)\n",
           passed ? "PASS" : "FAIL", name,
           max_err_dQ, max_err_dK, max_err_dV, scaled_tol);

    // Cleanup
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    cudaFree(d_dO); cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV);
    cudaFree(d_lse); cudaFree(d_D);

    delete[] h_Q_f32; delete[] h_K_f32; delete[] h_V_f32; delete[] h_dO_f32;
    delete[] h_O_f32;
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_dO;
    delete[] h_ref_dQ; delete[] h_ref_dK; delete[] h_ref_dV;
    delete[] h_dQ; delete[] h_dK; delete[] h_dV;

    return passed;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("=== Flash Attention 2 Backward tests ===\n\n");

    bool all_pass = true;

    all_pass &= run_test(
        "B=1 S=16  H_q=2  H_kv=1 (small GQA 2:1)",
        1, 16, 2, 1);

    all_pass &= run_test(
        "B=1 S=32  H_q=2  H_kv=2 (MHA, no GQA)",
        1, 32, 2, 2);

    all_pass &= run_test(
        "B=1 S=64  H_q=4  H_kv=2 (single KV tile)",
        1, 64, 4, 2);

    all_pass &= run_test(
        "B=2 S=48  H_q=4  H_kv=2 (batched)",
        2, 48, 4, 2);

    all_pass &= run_test(
        "B=1 S=128 H_q=16 H_kv=8 (Qwen3 heads)",
        1, 128, 16, 8);

    printf("\n%s\n", all_pass ? "All tests PASSED." : "Some tests FAILED.");
    return all_pass ? 0 : 1;
}
