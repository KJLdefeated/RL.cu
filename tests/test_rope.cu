// test_rope.cu
// Validates the RoPE precompute and apply kernels against a CPU FP32 reference.
//
// Pass criterion (from DESIGN.md): max absolute error < 1e-3 (FP16 budget).
//
// Test cases:
//   1. Single token at position 0  — trivial (cos=1, sin=0 → identity)
//   2. Single token at position 1  — basic rotation
//   3. B=1 S=32, H_q=16 H_kv=8 D=128 — Qwen3 head config
//   4. B=1 S=16, H_q=4  H_kv=2 D=128 — small, GQA 2:1
//   5. Non-sequential positions  — position_ids = [0, 3, 7, 15, ...]
//   6. Long sequence S=256, full Qwen3 config — multi-tile correctness

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/rope.cuh"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                     \
    } while (0)

static unsigned int lcg = 1234567u;
static float lcg_randf() {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(lcg >> 1) / (float)0x7fffffffu) * 2.0f - 1.0f;
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
// Implements HuggingFace Qwen3 RoPE in FP32.
// x_out[i]        = x[i] * cos - x[i+D/2] * sin   (i < D/2)
// x_out[i + D/2]  = x[i+D/2] * cos + x[i] * sin   (i < D/2)
// where cos/sin are looked up from the precomputed FP32 table.
// ---------------------------------------------------------------------------
static void ref_rope_apply(
    float*       out,            // [num_tokens, H, D]  FP32
    const float* x,              // [num_tokens, H, D]  FP32
    const float* cos_table,      // [max_seq_len, D/2]  FP32
    const float* sin_table,
    const int*   position_ids,   // [num_tokens]
    int          num_tokens,
    int          H,
    int          head_dim
) {
    const int half = head_dim / 2;
    for (int tok = 0; tok < num_tokens; tok++) {
        const int pos = position_ids[tok];
        for (int h = 0; h < H; h++) {
            const float* xh  = x   + (tok * H + h) * head_dim;
            float*       oh  = out + (tok * H + h) * head_dim;
            for (int i = 0; i < half; i++) {
                const float c = cos_table[pos * half + i];
                const float s = sin_table[pos * half + i];
                oh[i]      = xh[i]      * c - xh[i + half] * s;
                oh[i+half] = xh[i+half] * c + xh[i]        * s;
            }
        }
    }
}

// CPU precompute (FP32) — used to build the reference table
static void ref_precompute(
    float* cos_table, float* sin_table,
    int max_seq_len, int head_dim, float rope_theta
) {
    const int half = head_dim / 2;
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half; i++) {
            float inv_freq = expf(-2.0f * i * logf(rope_theta) / (float)head_dim);
            float angle    = (float)pos * inv_freq;
            cos_table[pos * half + i] = cosf(angle);
            sin_table[pos * half + i] = sinf(angle);
        }
    }
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
// Exercises both precompute and apply.  The GPU sin/cos table is compared
// against the CPU reference first, then the rotated output is compared.
// ---------------------------------------------------------------------------
static bool run_test(
    const char* name,
    int num_tokens, int H_q, int H_kv,
    const int* position_ids_host,  // [num_tokens]
    int head_dim  = 128,
    float rope_theta = 1e6f,
    float tol = 1e-3f
) {
    const int max_seq_len = 2048;
    const int hd2 = head_dim / 2;  // avoid shadowing CUDA 'half' type
    const int N_q  = num_tokens * H_q  * head_dim;
    const int N_kv = num_tokens * H_kv * head_dim;

    // ── CPU reference table ───────────────────────────────────────────────
    float* ref_cos = new float[(long)max_seq_len * hd2];
    float* ref_sin = new float[(long)max_seq_len * hd2];
    ref_precompute(ref_cos, ref_sin, max_seq_len, head_dim, rope_theta);

    // ── Host Q, K (FP32 master + FP16 kernel input) ───────────────────────
    float* h_Q_f32 = new float[N_q];
    float* h_K_f32 = new float[N_kv];
    half*  h_Q     = new half[N_q];
    half*  h_K     = new half[N_kv];
    half*  h_Q_out = new half[N_q];
    half*  h_K_out = new half[N_kv];

    for (int i = 0; i < N_q;  i++) h_Q[i] = __float2half(lcg_randf() * 0.5f);
    for (int i = 0; i < N_kv; i++) h_K[i] = __float2half(lcg_randf() * 0.5f);

    // Round-trip to FP16 so reference uses the exact same values
    for (int i = 0; i < N_q;  i++) h_Q_f32[i] = __half2float(h_Q[i]);
    for (int i = 0; i < N_kv; i++) h_K_f32[i] = __half2float(h_K[i]);

    // ── CPU reference output ──────────────────────────────────────────────
    float* ref_Q_out = new float[N_q];
    float* ref_K_out = new float[N_kv];
    ref_rope_apply(ref_Q_out, h_Q_f32, ref_cos, ref_sin, position_ids_host,
                   num_tokens, H_q, head_dim);
    ref_rope_apply(ref_K_out, h_K_f32, ref_cos, ref_sin, position_ids_host,
                   num_tokens, H_kv, head_dim);

    // ── GPU allocations ───────────────────────────────────────────────────
    float *d_cos, *d_sin;
    half  *d_Q, *d_K;
    int   *d_pos_ids;

    CUDA_CHECK(cudaMalloc(&d_cos,     (long)max_seq_len * hd2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin,     (long)max_seq_len * hd2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q,       N_q  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K,       N_kv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_pos_ids, num_tokens * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_Q,       h_Q,            N_q  * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K,       h_K,            N_kv * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_ids, position_ids_host, num_tokens * sizeof(int), cudaMemcpyHostToDevice));

    // ── Precompute ────────────────────────────────────────────────────────
    launch_rope_precompute(d_cos, d_sin, max_seq_len, head_dim, rope_theta);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check sin/cos table accuracy
    float* gpu_cos = new float[(long)max_seq_len * hd2];
    float* gpu_sin = new float[(long)max_seq_len * hd2];
    CUDA_CHECK(cudaMemcpy(gpu_cos, d_cos, (long)max_seq_len * hd2 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_sin, d_sin, (long)max_seq_len * hd2 * sizeof(float), cudaMemcpyDeviceToHost));

    float max_tbl_err = 0.0f;
    for (long i = 0; i < (long)max_seq_len * hd2; i++) {
        max_tbl_err = fmaxf(max_tbl_err, fabsf(gpu_cos[i] - ref_cos[i]));
        max_tbl_err = fmaxf(max_tbl_err, fabsf(gpu_sin[i] - ref_sin[i]));
    }
    // GPU single-precision cosf/sinf vs CPU reference can differ by ~1e-4 (ULP differences)
    if (max_tbl_err > 1e-4f) {
        printf("[FAIL] %s — sin/cos table error %.2e > 1e-4\n", name, max_tbl_err);
        return false;
    }

    // ── Apply RoPE ────────────────────────────────────────────────────────
    launch_rope(d_Q, d_K, d_cos, d_sin, d_pos_ids,
                num_tokens, H_q, H_kv, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_Q_out, d_Q, N_q  * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_K_out, d_K, N_kv * sizeof(half), cudaMemcpyDeviceToHost));

    // Compare Q
    float max_err_q = 0.0f;
    for (int i = 0; i < N_q; i++) {
        float diff = fabsf(__half2float(h_Q_out[i]) - ref_Q_out[i]);
        if (diff > max_err_q) max_err_q = diff;
    }

    // Compare K
    float max_err_k = 0.0f;
    for (int i = 0; i < N_kv; i++) {
        float diff = fabsf(__half2float(h_K_out[i]) - ref_K_out[i]);
        if (diff > max_err_k) max_err_k = diff;
    }

    const float max_err = fmaxf(max_err_q, max_err_k);
    const bool passed   = (max_err < tol);

    printf("[%s] %-54s max_err=%.6f  %s\n",
           passed ? "PASS" : "FAIL", name, max_err,
           passed ? "" : "<-- EXCEEDS 1e-3");

    // cleanup
    CUDA_CHECK(cudaFree(d_cos)); CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_Q));   CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_pos_ids));
    delete[] ref_cos; delete[] ref_sin;
    delete[] h_Q_f32; delete[] h_K_f32;
    delete[] h_Q;     delete[] h_K;
    delete[] h_Q_out; delete[] h_K_out;
    delete[] ref_Q_out; delete[] ref_K_out;
    delete[] gpu_cos; delete[] gpu_sin;

    return passed;
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------
static void run_benchmark(
    const char* name,
    int num_tokens, int H_q, int H_kv, int head_dim,
    float rope_theta = 1e6f,
    int warmup = 20, int iters = 500
) {
    const int max_seq_len = 2048;
    const int hd2 = head_dim / 2;  // avoid shadowing CUDA 'half' type

    float *d_cos, *d_sin;
    half  *d_Q, *d_K;
    int   *d_pos;

    CUDA_CHECK(cudaMalloc(&d_cos, (long)max_seq_len * hd2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, (long)max_seq_len * hd2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q,   (long)num_tokens * H_q  * head_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K,   (long)num_tokens * H_kv * head_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_pos, num_tokens * sizeof(int)));

    launch_rope_precompute(d_cos, d_sin, max_seq_len, head_dim, rope_theta);

    // Sequential position IDs
    int* h_pos = new int[num_tokens];
    for (int i = 0; i < num_tokens; i++) h_pos[i] = i;
    CUDA_CHECK(cudaMemcpy(d_pos, h_pos, num_tokens * sizeof(int), cudaMemcpyHostToDevice));
    delete[] h_pos;

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    for (int i = 0; i < warmup; i++)
        launch_rope(d_Q, d_K, d_cos, d_sin, d_pos, num_tokens, H_q, H_kv, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < iters; i++)
        launch_rope(d_Q, d_K, d_cos, d_sin, d_pos, num_tokens, H_q, H_kv, head_dim);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    const float us = ms * 1000.0f / iters;

    // Bytes: read+write Q (H_q) + read+write K (H_kv), each num_tokens * D halfs
    const long elems = (long)num_tokens * (H_q + H_kv) * head_dim;
    const float bw   = (float)(2 * elems * sizeof(half)) / (us * 1e-6f) / 1e9f;

    printf("[BENCH] %-50s  %6.2f us  %6.1f GB/s\n", name, us, bw);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaFree(d_cos)); CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_Q));   CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_pos));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("=== RoPE kernel tests ===\n\n");

    bool all_pass = true;

    // Test 1: single token, position 0  → identity (cos=1, sin=0 everywhere)
    {
        int pos[] = {0};
        all_pass &= run_test("pos=0 identity (single token)",
                             1, 4, 2, pos);
    }

    // Test 2: single token, position 1
    {
        int pos[] = {1};
        all_pass &= run_test("pos=1 basic rotation (single token)",
                             1, 4, 2, pos);
    }

    // Test 3: Qwen3 config — sequential positions
    {
        const int S = 32;
        int pos[S];
        for (int i = 0; i < S; i++) pos[i] = i;
        all_pass &= run_test("S=32 H_q=16 H_kv=8 D=128 (Qwen3 config)",
                             S, 16, 8, pos);
    }

    // Test 4: small GQA
    {
        const int S = 16;
        int pos[S];
        for (int i = 0; i < S; i++) pos[i] = i;
        all_pass &= run_test("S=16 H_q=4 H_kv=2 D=128 (small GQA)",
                             S, 4, 2, pos);
    }

    // Test 5: non-sequential positions (decode scenario)
    {
        const int S = 8;
        int pos[] = {0, 5, 12, 31, 63, 127, 255, 511};
        all_pass &= run_test("S=8 non-sequential positions",
                             S, 4, 2, pos);
    }

    // Test 6: long sequence — multiple blocks
    {
        const int S = 256;
        int pos[S];
        for (int i = 0; i < S; i++) pos[i] = i;
        all_pass &= run_test("S=256 H_q=16 H_kv=8 D=128 (long seq)",
                             S, 16, 8, pos);
    }

    // Test 7: decode step — single token at a large position
    {
        int pos[] = {2047};  // near max_seq_len
        all_pass &= run_test("decode pos=2047 single token",
                             1, 16, 8, pos);
    }

    printf("\n%s\n", all_pass ? "All tests PASSED." : "Some tests FAILED.");

    printf("\n=== RoPE benchmarks (warmup=20, iters=500) ===\n");
    run_benchmark("Decode: S=1   H_q=16 H_kv=8",   1,   16, 8, 128);
    run_benchmark("Prefill: S=128  H_q=16 H_kv=8", 128, 16, 8, 128);
    run_benchmark("Prefill: S=512  H_q=16 H_kv=8", 512, 16, 8, 128);
    run_benchmark("Prefill: S=2048 H_q=16 H_kv=8", 2048,16, 8, 128);

    return all_pass ? 0 : 1;
}
