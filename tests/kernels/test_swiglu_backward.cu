// test_swiglu_backward.cu
// Validates swiglu_backward against CPU FP32 reference.
//
// Forward: out_i = silu(gate_i) * up_i
// Backward:
//   dGate_i = dOut_i * up_i * σ(g_i) * (1 + g_i * (1 - σ(g_i)))
//   dUp_i   = dOut_i * silu(g_i)
//
// Test cases:
//   1. Small n=64
//   2. n=3072 (one token, intermediate_size)
//   3. n=32*3072 (batch=32)
//   4. n=512*3072 (prefill)
//   5. Edge: gate near 0 (silu'(0) = 0.5)
//   6. Edge: large |gate| (silu saturates)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/swiglu.cuh"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                     \
    } while (0)

static unsigned int lcg = 42069u;
static float lcg_randf() {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(lcg >> 1) / (float)0x7fffffffu) * 2.0f - 1.0f;
}

static void ref_swiglu_backward(
    float* dGate, float* dUp,
    const float* dOut, const float* gate, const float* up,
    int n
) {
    for (int i = 0; i < n; i++) {
        float g   = gate[i];
        float u   = up[i];
        float dy  = dOut[i];
        float sig = 1.0f / (1.0f + expf(-g));
        float silu = g * sig;
        float silu_grad = sig * (1.0f + g * (1.0f - sig));

        dGate[i] = dy * u * silu_grad;
        dUp[i]   = dy * silu;
    }
}

static bool run_test(
    const char* name, int n,
    const float* custom_gate = nullptr,
    float tol = 1e-3f
) {
    float* h_gate_f32 = new float[n];
    float* h_up_f32   = new float[n];
    float* h_dOut_f32 = new float[n];
    half*  h_gate_h16 = new half[n];
    half*  h_up_h16   = new half[n];
    half*  h_dOut_h16 = new half[n];

    for (int i = 0; i < n; i++) {
        float g = custom_gate ? custom_gate[i] : lcg_randf();
        h_gate_h16[i] = __float2half(g);
        h_gate_f32[i] = __half2float(h_gate_h16[i]);
        h_up_h16[i]   = __float2half(lcg_randf());
        h_up_f32[i]   = __half2float(h_up_h16[i]);
        h_dOut_h16[i] = __float2half(lcg_randf());
        h_dOut_f32[i] = __half2float(h_dOut_h16[i]);
    }

    // CPU reference
    float* h_ref_dGate = new float[n];
    float* h_ref_dUp   = new float[n];
    ref_swiglu_backward(h_ref_dGate, h_ref_dUp, h_dOut_f32, h_gate_f32, h_up_f32, n);

    // GPU
    half *d_gate, *d_up, *d_dOut, *d_dGate, *d_dUp;
    CUDA_CHECK(cudaMalloc(&d_gate,  n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_up,    n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dOut,  n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dGate, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dUp,   n * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_gate,  h_gate_h16, n * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up,    h_up_h16,   n * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dOut,  h_dOut_h16, n * sizeof(half), cudaMemcpyHostToDevice));

    launch_swiglu_backward(d_dGate, d_dUp, d_dOut, d_gate, d_up, n, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    half* h_dGate = new half[n];
    half* h_dUp   = new half[n];
    CUDA_CHECK(cudaMemcpy(h_dGate, d_dGate, n * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dUp,   d_dUp,   n * sizeof(half), cudaMemcpyDeviceToHost));

    float max_err_gate = 0.0f, max_err_up = 0.0f;
    for (int i = 0; i < n; i++) {
        float dg = fabsf(__half2float(h_dGate[i]) - h_ref_dGate[i]);
        float du = fabsf(__half2float(h_dUp[i])   - h_ref_dUp[i]);
        if (dg > max_err_gate) max_err_gate = dg;
        if (du > max_err_up)   max_err_up   = du;
    }

    bool passed = (max_err_gate < tol) && (max_err_up < tol);
    printf("[%s] %-50s dGate_err=%.6f  dUp_err=%.6f  (tol=%.0e)\n",
           passed ? "PASS" : "FAIL", name, max_err_gate, max_err_up, tol);

    CUDA_CHECK(cudaFree(d_gate));
    CUDA_CHECK(cudaFree(d_up));
    CUDA_CHECK(cudaFree(d_dOut));
    CUDA_CHECK(cudaFree(d_dGate));
    CUDA_CHECK(cudaFree(d_dUp));
    delete[] h_gate_f32; delete[] h_gate_h16;
    delete[] h_up_f32;   delete[] h_up_h16;
    delete[] h_dOut_f32; delete[] h_dOut_h16;
    delete[] h_ref_dGate; delete[] h_ref_dUp;
    delete[] h_dGate;     delete[] h_dUp;

    return passed;
}

int main() {
    printf("=== swiglu_backward tests ===\n\n");

    bool all_pass = true;

    all_pass &= run_test("small n=64",                   64);
    all_pass &= run_test("n=3072 (1 token)",           3072);
    all_pass &= run_test("n=32*3072 (batch=32)",    32*3072);
    all_pass &= run_test("n=512*3072 (prefill)",   512*3072);

    // Edge: gate near zero
    {
        const int n = 256;
        float gates[256];
        for (int i = 0; i < n; i++) gates[i] = (float)i / (float)n * 0.01f - 0.005f;
        all_pass &= run_test("gate near zero", n, gates);
    }

    // Edge: large |gate| (sigmoid saturates)
    {
        const int n = 256;
        float gates[256];
        for (int i = 0; i < n; i++) gates[i] = (i % 2 == 0) ? 6.0f : -6.0f;
        // Higher tol: silu(6)≈5.985, FP16 ULP near 6.0 is ~0.003
        all_pass &= run_test("large |gate| (saturated sigmoid)", n, gates, 3e-3f);
    }

    printf("\n%s\n", all_pass ? "All tests PASSED." : "Some tests FAILED.");
    return all_pass ? 0 : 1;
}
