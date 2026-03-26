// test_rope_backward.cu
// Validates rope_backward (inverse rotation) against CPU reference.
//
// Forward: out[i] = x[i]*cos - x[i+D/2]*sin;  out[i+D/2] = x[i+D/2]*cos + x[i]*sin
// Backward: dX[i] = dOut[i]*cos + dOut[i+D/2]*sin;  dX[i+D/2] = dOut[i+D/2]*cos - dOut[i]*sin
//
// Test cases:
//   1. Single token, single head — basic correctness
//   2. Qwen3 dims: 4 tokens, H_q=16, H_kv=8, D=128
//   3. Round-trip: forward then backward recovers original input
//   4. Different position_ids (non-sequential)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/rope.cuh"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                     \
    } while (0)

static unsigned int lcg = 13579u;
static float lcg_randf() {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(lcg >> 1) / (float)0x7fffffffu) * 2.0f - 1.0f;
}

// CPU reference: inverse rotation
static void ref_rope_backward(
    float* dQ, float* dK,
    const float* cos_tbl, const float* sin_tbl,
    const int* pos_ids,
    int num_tokens, int H_q, int H_kv, int D
) {
    const int hd2 = D / 2;
    for (int tok = 0; tok < num_tokens; tok++) {
        int pos = pos_ids[tok];
        for (int h = 0; h < H_q; h++) {
            float* dq = dQ + ((long)tok * H_q + h) * D;
            for (int i = 0; i < hd2; i++) {
                float c = cos_tbl[pos * hd2 + i];
                float s = sin_tbl[pos * hd2 + i];
                float d0 = dq[i], d1 = dq[i + hd2];
                dq[i]       = d0 * c + d1 * s;
                dq[i + hd2] = d1 * c - d0 * s;
            }
        }
        for (int h = 0; h < H_kv; h++) {
            float* dk = dK + ((long)tok * H_kv + h) * D;
            for (int i = 0; i < hd2; i++) {
                float c = cos_tbl[pos * hd2 + i];
                float s = sin_tbl[pos * hd2 + i];
                float d0 = dk[i], d1 = dk[i + hd2];
                dk[i]       = d0 * c + d1 * s;
                dk[i + hd2] = d1 * c - d0 * s;
            }
        }
    }
}

static bool run_test(
    const char* name,
    int num_tokens, int H_q, int H_kv, int D,
    const int* custom_pos = nullptr,
    float tol = 1e-3f
) {
    const int hd2 = D / 2;
    const long q_sz = (long)num_tokens * H_q  * D;
    const long k_sz = (long)num_tokens * H_kv * D;
    const int max_pos = 512;

    // Precompute cos/sin tables on CPU (match GPU single-precision)
    float* h_cos = new float[max_pos * hd2];
    float* h_sin = new float[max_pos * hd2];

    // Use GPU precompute for exact match
    float *d_cos, *d_sin;
    CUDA_CHECK(cudaMalloc(&d_cos, max_pos * hd2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, max_pos * hd2 * sizeof(float)));
    launch_rope_precompute(d_cos, d_sin, max_pos, D, 1000000.0f, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_cos, d_cos, max_pos * hd2 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sin, d_sin, max_pos * hd2 * sizeof(float), cudaMemcpyDeviceToHost));

    // Position ids
    int* h_pos = new int[num_tokens];
    if (custom_pos) {
        for (int i = 0; i < num_tokens; i++) h_pos[i] = custom_pos[i];
    } else {
        for (int i = 0; i < num_tokens; i++) h_pos[i] = i;
    }

    // Generate random dQ, dK
    float* h_dQ_f32 = new float[q_sz];
    float* h_dK_f32 = new float[k_sz];
    half*  h_dQ_h16 = new half[q_sz];
    half*  h_dK_h16 = new half[k_sz];
    for (long i = 0; i < q_sz; i++) {
        h_dQ_h16[i] = __float2half(lcg_randf());
        h_dQ_f32[i] = __half2float(h_dQ_h16[i]);
    }
    for (long i = 0; i < k_sz; i++) {
        h_dK_h16[i] = __float2half(lcg_randf());
        h_dK_f32[i] = __half2float(h_dK_h16[i]);
    }

    // CPU reference (modifies in place)
    ref_rope_backward(h_dQ_f32, h_dK_f32, h_cos, h_sin, h_pos,
                      num_tokens, H_q, H_kv, D);

    // GPU
    half *d_dQ, *d_dK;
    int  *d_pos;
    CUDA_CHECK(cudaMalloc(&d_dQ, q_sz * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dK, k_sz * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_pos, num_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_dQ, h_dQ_h16, q_sz * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dK, h_dK_h16, k_sz * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos, h_pos, num_tokens * sizeof(int), cudaMemcpyHostToDevice));

    launch_rope_backward(d_dQ, d_dK, d_cos, d_sin, d_pos,
                         num_tokens, H_q, H_kv, D, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify
    half* h_out_dQ = new half[q_sz];
    half* h_out_dK = new half[k_sz];
    CUDA_CHECK(cudaMemcpy(h_out_dQ, d_dQ, q_sz * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_dK, d_dK, k_sz * sizeof(half), cudaMemcpyDeviceToHost));

    float max_err_q = 0.0f, max_err_k = 0.0f;
    for (long i = 0; i < q_sz; i++) {
        float diff = fabsf(__half2float(h_out_dQ[i]) - h_dQ_f32[i]);
        if (diff > max_err_q) max_err_q = diff;
    }
    for (long i = 0; i < k_sz; i++) {
        float diff = fabsf(__half2float(h_out_dK[i]) - h_dK_f32[i]);
        if (diff > max_err_k) max_err_k = diff;
    }

    bool passed = (max_err_q < tol) && (max_err_k < tol);
    printf("[%s] %-50s dQ_err=%.6f  dK_err=%.6f  (tol=%.0e)\n",
           passed ? "PASS" : "FAIL", name, max_err_q, max_err_k, tol);

    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_dQ));
    CUDA_CHECK(cudaFree(d_dK));
    CUDA_CHECK(cudaFree(d_pos));
    delete[] h_cos; delete[] h_sin; delete[] h_pos;
    delete[] h_dQ_f32; delete[] h_dQ_h16; delete[] h_out_dQ;
    delete[] h_dK_f32; delete[] h_dK_h16; delete[] h_out_dK;

    return passed;
}

// Round-trip test: forward(backward(x)) should ≈ x
static bool run_roundtrip_test(
    int num_tokens, int H_q, int H_kv, int D,
    float tol = 1e-3f
) {
    const long q_sz = (long)num_tokens * H_q * D;
    const long k_sz = (long)num_tokens * H_kv * D;
    const int hd2 = D / 2;
    const int max_pos = 512;

    // Precompute tables
    float *d_cos, *d_sin;
    CUDA_CHECK(cudaMalloc(&d_cos, max_pos * hd2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, max_pos * hd2 * sizeof(float)));
    launch_rope_precompute(d_cos, d_sin, max_pos, D, 1000000.0f, 0);

    // Position ids
    int* h_pos = new int[num_tokens];
    for (int i = 0; i < num_tokens; i++) h_pos[i] = i;
    int* d_pos;
    CUDA_CHECK(cudaMalloc(&d_pos, num_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pos, h_pos, num_tokens * sizeof(int), cudaMemcpyHostToDevice));

    // Original data
    half* h_Q_orig = new half[q_sz];
    half* h_K_orig = new half[k_sz];
    for (long i = 0; i < q_sz; i++) h_Q_orig[i] = __float2half(lcg_randf());
    for (long i = 0; i < k_sz; i++) h_K_orig[i] = __float2half(lcg_randf());

    half *d_Q, *d_K;
    CUDA_CHECK(cudaMalloc(&d_Q, q_sz * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K, k_sz * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q_orig, q_sz * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K_orig, k_sz * sizeof(half), cudaMemcpyHostToDevice));

    // Forward then backward — should recover original
    launch_rope(d_Q, d_K, d_cos, d_sin, d_pos, num_tokens, H_q, H_kv, D, 0);
    launch_rope_backward(d_Q, d_K, d_cos, d_sin, d_pos, num_tokens, H_q, H_kv, D, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    half* h_Q_rt = new half[q_sz];
    half* h_K_rt = new half[k_sz];
    CUDA_CHECK(cudaMemcpy(h_Q_rt, d_Q, q_sz * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_K_rt, d_K, k_sz * sizeof(half), cudaMemcpyDeviceToHost));

    float max_err_q = 0.0f, max_err_k = 0.0f;
    for (long i = 0; i < q_sz; i++) {
        float diff = fabsf(__half2float(h_Q_rt[i]) - __half2float(h_Q_orig[i]));
        if (diff > max_err_q) max_err_q = diff;
    }
    for (long i = 0; i < k_sz; i++) {
        float diff = fabsf(__half2float(h_K_rt[i]) - __half2float(h_K_orig[i]));
        if (diff > max_err_k) max_err_k = diff;
    }

    bool passed = (max_err_q < tol) && (max_err_k < tol);
    printf("[%s] %-50s Q_rt_err=%.6f  K_rt_err=%.6f  (tol=%.0e)\n",
           passed ? "PASS" : "FAIL",
           "round-trip (fwd→bwd recovers input)", max_err_q, max_err_k, tol);

    CUDA_CHECK(cudaFree(d_cos)); CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_Q));   CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_pos));
    delete[] h_pos; delete[] h_Q_orig; delete[] h_K_orig;
    delete[] h_Q_rt; delete[] h_K_rt;

    return passed;
}

int main() {
    printf("=== rope_backward tests ===\n\n");

    bool all_pass = true;

    // 1. Single token, single head
    all_pass &= run_test("1 tok, H_q=1, H_kv=1, D=128", 1, 1, 1, 128);

    // 2. Qwen3 dims
    all_pass &= run_test("4 tok, H_q=16, H_kv=8, D=128 (Qwen3)", 4, 16, 8, 128);

    // 3. Larger batch
    all_pass &= run_test("32 tok, H_q=16, H_kv=8, D=128", 32, 16, 8, 128);

    // 4. Non-sequential position ids
    {
        int pos[] = {0, 5, 100, 200};
        all_pass &= run_test("non-sequential pos_ids", 4, 16, 8, 128, pos);
    }

    // 5. Round-trip: forward then backward = identity
    all_pass &= run_roundtrip_test(8, 16, 8, 128);

    printf("\n%s\n", all_pass ? "All tests PASSED." : "Some tests FAILED.");
    return all_pass ? 0 : 1;
}
