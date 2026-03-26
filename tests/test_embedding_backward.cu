// test_embedding_backward.cu
// Validates embedding_backward (scatter-add) against CPU reference.
//
// Backward: dW[token_ids[tok], d] += dOut[tok, d]
//
// Pass criterion: exact FP16 match for unique token_ids (pure scatter, no arithmetic).
// For duplicate token_ids, atomicAdd(half) introduces no rounding beyond FP16 addition,
// so we allow max_err < 1e-3 (one FP16 ULP near typical values).
//
// Test cases:
//   1. Single token                       — one row gets gradient
//   2. S=32 unique tokens                 — each row updated once (exact)
//   3. Duplicate token_ids (all same)     — all gradients scatter to one row (atomicAdd)
//   4. Mixed duplicates                   — some rows get multiple adds
//   5. S=512, vocab=151936, hidden=1024   — full Qwen3 scale
//   6. Non-touched rows stay zero         — verify dW rows not in token_ids remain 0

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/embedding.cuh"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                     \
    } while (0)

static unsigned int lcg = 77777u;
static float lcg_randf() {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(lcg >> 1) / (float)0x7fffffffu) * 2.0f - 1.0f;
}

// CPU reference: dW[token_ids[tok], d] += dOut[tok, d]
static void ref_embedding_backward(
    float*       dW,         // [vocab_size, hidden_size]  (accumulated)
    const float* dOut,       // [num_tokens, hidden_size]
    const int*   token_ids,
    int num_tokens, int hidden_size
) {
    for (int tok = 0; tok < num_tokens; tok++) {
        int vid = token_ids[tok];
        for (int d = 0; d < hidden_size; d++)
            dW[(long)vid * hidden_size + d] += dOut[(long)tok * hidden_size + d];
    }
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
static bool run_test(
    const char* name,
    int num_tokens, int vocab_size, int hidden_size,
    const int* custom_ids = nullptr,  // if non-null, use these token_ids
    float tol = 0.0f                  // 0 = exact match
) {
    const long dout_sz = (long)num_tokens * hidden_size;
    const long dw_sz   = (long)vocab_size * hidden_size;

    // Generate dOut
    float* h_dOut_f32 = new float[dout_sz];
    half*  h_dOut_h16 = new half[dout_sz];
    for (long i = 0; i < dout_sz; i++) {
        h_dOut_h16[i] = __float2half(lcg_randf());
        h_dOut_f32[i] = __half2float(h_dOut_h16[i]);
    }

    // Generate token_ids
    int* h_ids = new int[num_tokens];
    if (custom_ids) {
        for (int i = 0; i < num_tokens; i++) h_ids[i] = custom_ids[i];
    } else {
        for (int i = 0; i < num_tokens; i++) {
            lcg = lcg * 1664525u + 1013904223u;
            h_ids[i] = (int)(lcg % (unsigned)vocab_size);
        }
    }

    // CPU reference
    float* h_ref_dW = new float[dw_sz]();
    ref_embedding_backward(h_ref_dW, h_dOut_f32, h_ids, num_tokens, hidden_size);

    // GPU
    half *d_dOut, *d_dW;
    int  *d_ids;
    CUDA_CHECK(cudaMalloc(&d_dOut, dout_sz * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dW,  dw_sz   * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ids, num_tokens * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_dOut, h_dOut_h16, dout_sz * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ids,  h_ids,      num_tokens * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dW, 0, dw_sz * sizeof(half)));

    launch_embedding_backward(d_dW, d_dOut, d_ids, num_tokens, hidden_size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify
    half* h_dW = new half[dw_sz];
    CUDA_CHECK(cudaMemcpy(h_dW, d_dW, dw_sz * sizeof(half), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    int   err_count = 0;
    for (long i = 0; i < dw_sz; i++) {
        float gpu_val = __half2float(h_dW[i]);
        float ref_val = h_ref_dW[i];
        float diff = fabsf(gpu_val - ref_val);
        if (diff > max_err) max_err = diff;
        if (diff > tol + 1e-7f) err_count++;
    }

    bool passed = (max_err <= tol + 1e-7f);
    if (tol == 0.0f) {
        printf("[%s] %-55s max_err=%.6f  (exact)  %s\n",
               passed ? "PASS" : "FAIL", name, max_err,
               passed ? "" : "<-- FAIL");
    } else {
        printf("[%s] %-55s max_err=%.6f  (tol=%.0e)  %s\n",
               passed ? "PASS" : "FAIL", name, max_err, tol,
               passed ? "" : "<-- FAIL");
    }

    CUDA_CHECK(cudaFree(d_dOut));
    CUDA_CHECK(cudaFree(d_dW));
    CUDA_CHECK(cudaFree(d_ids));
    delete[] h_dOut_f32; delete[] h_dOut_h16;
    delete[] h_ids; delete[] h_ref_dW; delete[] h_dW;

    return passed;
}

int main() {
    printf("=== embedding_backward tests ===\n\n");

    bool all_pass = true;

    // 1. Single token — one row gets gradient
    {
        int ids[] = {42};
        all_pass &= run_test("single token", 1, 1000, 1024, ids);
    }

    // 2. S=32 unique tokens — each row updated once (exact)
    {
        int ids[32];
        for (int i = 0; i < 32; i++) ids[i] = i * 31;  // spread out, unique
        all_pass &= run_test("32 unique tokens", 32, 1000, 1024, ids);
    }

    // 3. All same token_id — all 32 gradients scatter-add to one row
    {
        int ids[32];
        for (int i = 0; i < 32; i++) ids[i] = 7;
        // 32 atomicAdds of FP16 values: expect small rounding
        all_pass &= run_test("32 tokens all same id (atomicAdd stress)",
                             32, 1000, 1024, ids, 1e-1f);
    }

    // 4. Mixed duplicates: some ids appear 2-4 times
    {
        int ids[] = {10, 20, 10, 30, 20, 10, 40, 20, 30, 10, 50, 60, 70, 80, 90, 100};
        // id=10 appears 4x, id=20 appears 3x, id=30 appears 2x
        all_pass &= run_test("mixed duplicates (4x/3x/2x/1x)",
                             16, 1000, 1024, ids, 1e-2f);
    }

    // 5. Full Qwen3 scale — random ids, some duplicates possible
    all_pass &= run_test("Qwen3 scale S=512 V=151936 H=1024",
                         512, 151936, 1024, nullptr, 1e-1f);

    // 6. Non-touched rows stay zero
    {
        int ids[] = {5, 10};
        // Only rows 5 and 10 should be nonzero in a vocab of 100
        const int V = 100, H = 64, T = 2;
        const long dw_sz = (long)V * H;

        float* h_dOut_f32 = new float[(long)T * H];
        half*  h_dOut_h16 = new half[(long)T * H];
        for (long i = 0; i < (long)T * H; i++) {
            h_dOut_h16[i] = __float2half(lcg_randf());
            h_dOut_f32[i] = __half2float(h_dOut_h16[i]);
        }

        half *d_dOut, *d_dW;
        int  *d_ids;
        CUDA_CHECK(cudaMalloc(&d_dOut, T * H * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_dW,  dw_sz * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_ids, T * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dOut, h_dOut_h16, T * H * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ids,  ids,        T * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_dW, 0, dw_sz * sizeof(half)));

        launch_embedding_backward(d_dW, d_dOut, d_ids, T, H, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        half* h_dW = new half[dw_sz];
        CUDA_CHECK(cudaMemcpy(h_dW, d_dW, dw_sz * sizeof(half), cudaMemcpyDeviceToHost));

        bool zero_ok = true;
        for (int v = 0; v < V; v++) {
            if (v == 5 || v == 10) continue;
            for (int d = 0; d < H; d++) {
                if (__half2float(h_dW[v * H + d]) != 0.0f) {
                    zero_ok = false;
                    break;
                }
            }
        }

        printf("[%s] %-55s %s\n",
               zero_ok ? "PASS" : "FAIL",
               "non-touched rows stay zero",
               zero_ok ? "" : "<-- FAIL");
        all_pass &= zero_ok;

        CUDA_CHECK(cudaFree(d_dOut));
        CUDA_CHECK(cudaFree(d_dW));
        CUDA_CHECK(cudaFree(d_ids));
        delete[] h_dOut_f32; delete[] h_dOut_h16; delete[] h_dW;
    }

    printf("\n%s\n", all_pass ? "All tests PASSED." : "Some tests FAILED.");
    return all_pass ? 0 : 1;
}
