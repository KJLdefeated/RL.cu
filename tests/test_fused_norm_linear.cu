#include "kernels/fused_norm_linear.cuh"
#include "kernels/rmsnorm.cuh"
#include "kernels/linear.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t _s = (x); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static unsigned long long lcg = 12345678ULL;
static float lcg_rand() {
    lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((lcg >> 33) & 0xFFFF) / 65535.f * 2.f - 1.f;
}

// Returns elapsed microseconds between two events (event must be recorded and sync'd)
static float event_us(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms * 1e3f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Correctness check
// ─────────────────────────────────────────────────────────────────────────────
static void run_correctness(const char* name, int T, int N, int H,
                             float eps, float tol, cublasHandle_t cublas)
{
    printf("  %-40s T=%-4d N=%-5d  ", name, T, N);
    fflush(stdout);

    const size_t xsz   = (size_t)T * H;
    const size_t wsz   = (size_t)N * H;
    const size_t outsz = (size_t)T * N;

    std::vector<float> h_x(xsz), h_w(wsz), h_g(H);
    for (auto& v : h_x) v = lcg_rand() * 0.3f;
    for (auto& v : h_w) v = lcg_rand() * 0.1f;
    for (auto& v : h_g) v = 0.8f + lcg_rand() * 0.2f;

    half *d_x, *d_w, *d_ref_norm, *d_ref_out, *d_fused_out;
    float *d_gamma;
    CUDA_CHECK(cudaMalloc(&d_x,         xsz  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w,         wsz  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_gamma,     H    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref_norm,  xsz  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ref_out,   outsz * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_fused_out, outsz * sizeof(half)));

    std::vector<half> h_xh(xsz), h_wh(wsz);
    for (size_t i = 0; i < xsz; i++) h_xh[i] = __float2half(h_x[i]);
    for (size_t i = 0; i < wsz; i++) h_wh[i] = __float2half(h_w[i]);
    CUDA_CHECK(cudaMemcpy(d_x,    h_xh.data(), xsz * sizeof(half),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w,    h_wh.data(), wsz * sizeof(half),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_g.data(), H   * sizeof(float), cudaMemcpyHostToDevice));

    // Reference: norm → cuBLAS GEMM
    launch_rmsnorm(d_ref_norm, d_x, d_gamma, T, H, eps, 0);
    linear_half(cublas, d_ref_norm, d_w, d_ref_out, T, N, H, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fused
    launch_fused_rmsnorm_linear(d_x, d_gamma, d_w, d_fused_out, T, N, H, eps, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_ref(outsz), h_fused(outsz);
    CUDA_CHECK(cudaMemcpy(h_ref.data(),   d_ref_out,   outsz * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fused.data(), d_fused_out, outsz * sizeof(half), cudaMemcpyDeviceToHost));

    float max_err = 0.f;
    for (size_t i = 0; i < outsz; i++)
        max_err = fmaxf(max_err, fabsf(__half2float(h_ref[i]) - __half2float(h_fused[i])));

    printf("%s  max_err=%.2e\n", (max_err < tol) ? "[PASS]" : "[FAIL]", max_err);
    if (max_err >= tol) exit(1);

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_gamma);
    cudaFree(d_ref_norm); cudaFree(d_ref_out); cudaFree(d_fused_out);
}

// ─────────────────────────────────────────────────────────────────────────────
// Timing benchmark: fused vs norm+cuBLAS
//   Warms up both paths, then times ITERS back-to-back launches.
// ─────────────────────────────────────────────────────────────────────────────
static void run_bench(const char* name, int T, int N, int H,
                      float eps, cublasHandle_t cublas,
                      int warmup = 5, int iters = 50)
{
    const size_t xsz   = (size_t)T * H;
    const size_t wsz   = (size_t)N * H;
    const size_t outsz = (size_t)T * N;

    half *d_x, *d_w, *d_norm, *d_out_ref, *d_out_fused;
    float *d_gamma;
    CUDA_CHECK(cudaMalloc(&d_x,          xsz  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w,          wsz  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_gamma,      H    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm,       xsz  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out_ref,    outsz * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out_fused,  outsz * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_x,     0, xsz  * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_w,     0, wsz  * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_gamma, 0, H    * sizeof(float)));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    // ── norm + cuBLAS ──
    for (int i = 0; i < warmup; i++) {
        launch_rmsnorm(d_norm, d_x, d_gamma, T, H, eps, 0);
        linear_half(cublas, d_norm, d_w, d_out_ref, T, N, H, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < iters; i++) {
        launch_rmsnorm(d_norm, d_x, d_gamma, T, H, eps, 0);
        linear_half(cublas, d_norm, d_w, d_out_ref, T, N, H, 0);
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    const float ref_us = event_us(ev0, ev1) / iters;

    // ── fused ──
    for (int i = 0; i < warmup; i++)
        launch_fused_rmsnorm_linear(d_x, d_gamma, d_w, d_out_fused, T, N, H, eps, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < iters; i++)
        launch_fused_rmsnorm_linear(d_x, d_gamma, d_w, d_out_fused, T, N, H, eps, 0);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    const float fused_us = event_us(ev0, ev1) / iters;

    // Arithmetic intensity (for context)
    // Reads: x (T*H*2B) + W (N*H*2B) + gamma (H*4B); writes: out (T*N*2B)
    const float bytes_ref  = 2.f * (float)(xsz + wsz) * 2 + H * 4 + outsz * 2; // norm write + re-read
    const float bytes_fused = (float)(xsz + wsz) * 2 + H * 4 + outsz * 2;      // no norm HBM write
    (void)bytes_ref; (void)bytes_fused;

    printf("  %-30s T=%-4d N=%-5d  ref=%6.1f µs  fused=%6.1f µs  ratio=%4.2fx%s\n",
           name, T, N,
           ref_us, fused_us,
           ref_us / fused_us,
           (fused_us < ref_us) ? " ← faster" : "");

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_gamma);
    cudaFree(d_norm); cudaFree(d_out_ref); cudaFree(d_out_fused);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    printf("=== test_fused_norm_linear ===\n\n");

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    constexpr float eps = 1e-6f;
    constexpr float tol = 5e-3f;

    // ─── Correctness ───────────────────────────────────────────────────────
    printf("── Correctness ──\n");
    run_correctness("q_proj  (B=1)",   1,   2048, 1024, eps, tol, cublas);
    run_correctness("k_proj  (B=1)",   1,   1024, 1024, eps, tol, cublas);
    run_correctness("v_proj  (B=1)",   1,   1024, 1024, eps, tol, cublas);
    run_correctness("gate    (B=1)",   1,   3072, 1024, eps, tol, cublas);
    run_correctness("up      (B=1)",   1,   3072, 1024, eps, tol, cublas);
    run_correctness("q_proj  (B=16)",  16,  2048, 1024, eps, tol, cublas);
    run_correctness("gate    (B=16)",  16,  3072, 1024, eps, tol, cublas);
    run_correctness("q_proj  (B=64)",  64,  2048, 1024, eps, tol, cublas);
    run_correctness("gate    (B=64)",  64,  3072, 1024, eps, tol, cublas);
    run_correctness("q_proj  (B=128)", 128, 2048, 1024, eps, tol, cublas);
    run_correctness("q_proj  (B=256)", 256, 2048, 1024, eps, tol, cublas);
    run_correctness("gate    (B=256)", 256, 3072, 1024, eps, tol, cublas);
    run_correctness("q_proj  (T=17)",  17,  2048, 1024, eps, tol, cublas);
    run_correctness("gate    (T=33)",  33,  3072, 1024, eps, tol, cublas);
    printf("\nAll correctness tests passed.\n\n");

    // ─── Performance ───────────────────────────────────────────────────────
    // Columns: kernel name | T | N | norm+cuBLAS µs | fused µs | ratio
    printf("── Performance: fused vs norm+cuBLAS (µs per call, avg over 50 iters) ──\n");
    printf("  %-30s %-6s %-7s  %-14s  %-14s  %s\n",
           "kernel", "T", "N", "norm+cuBLAS", "fused", "ratio");
    printf("  %s\n", "--------------------------------------------------------------------------------");

    // Decode shapes (small T)
    run_bench("q_proj",   1,   2048, 1024, eps, cublas);
    run_bench("q_proj",   4,   2048, 1024, eps, cublas);
    run_bench("q_proj",   8,   2048, 1024, eps, cublas);
    run_bench("q_proj",   16,  2048, 1024, eps, cublas);
    run_bench("q_proj",   32,  2048, 1024, eps, cublas);
    run_bench("q_proj",   64,  2048, 1024, eps, cublas);
    run_bench("q_proj",   128, 2048, 1024, eps, cublas);
    run_bench("q_proj",   256, 2048, 1024, eps, cublas);
    printf("\n");
    run_bench("k_proj",   1,   1024, 1024, eps, cublas);
    run_bench("k_proj",   64,  1024, 1024, eps, cublas);
    run_bench("k_proj",   256, 1024, 1024, eps, cublas);
    printf("\n");
    run_bench("gate_proj",1,   3072, 1024, eps, cublas);
    run_bench("gate_proj",64,  3072, 1024, eps, cublas);
    run_bench("gate_proj",256, 3072, 1024, eps, cublas);

    printf("\n");
    cublasDestroy(cublas);
    return 0;
}
