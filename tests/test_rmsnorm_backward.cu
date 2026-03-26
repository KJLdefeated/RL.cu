// test_rmsnorm_backward.cu
// Validates rmsnorm_backward against CPU FP32 reference.
//
// Forward: y_i = w_i * x_i / sqrt(mean(x^2) + eps)
// Backward:
//   dX_j = rms_inv * (dY_j * w_j  -  x_j * rms_inv^2 * c / N)
//     where c = sum_i(dY_i * w_i * x_i)
//   dW_i = sum_rows(dY_i * x_i * rms_inv)
//
// Test cases:
//   1. Single row, cols=128 (QK-norm head_dim)
//   2. Single row, cols=1024 (hidden_size)
//   3. 32 rows, cols=1024 (batch of tokens)
//   4. 512 rows, cols=1024 (prefill)
//   5. dW accumulation: verify sum across rows

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/rmsnorm.cuh"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                     \
    } while (0)

static unsigned int lcg = 31337u;
static float lcg_randf() {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(lcg >> 1) / (float)0x7fffffffu) * 2.0f - 1.0f;
}

// CPU reference backward
static void ref_rmsnorm_backward(
    float*       dX,      // [rows, cols]
    float*       dW,      // [cols]  (accumulated)
    const float* dY,      // [rows, cols]
    const float* x,       // [rows, cols]
    const float* weight,  // [cols]
    int rows, int cols, float eps
) {
    for (int r = 0; r < rows; r++) {
        const float* x_row  = x  + (long)r * cols;
        const float* dy_row = dY + (long)r * cols;
        float*       dx_row = dX + (long)r * cols;

        // sum_sq
        float sum_sq = 0.0f;
        for (int i = 0; i < cols; i++)
            sum_sq += x_row[i] * x_row[i];
        float rms_inv = 1.0f / sqrtf(sum_sq / (float)cols + eps);

        // c = sum(dY_i * w_i * x_i)
        float c = 0.0f;
        for (int i = 0; i < cols; i++)
            c += dy_row[i] * weight[i] * x_row[i];

        float coeff = c * rms_inv * rms_inv / (float)cols;

        // dX and dW
        for (int i = 0; i < cols; i++) {
            dx_row[i] = rms_inv * (dy_row[i] * weight[i] - x_row[i] * coeff);
            dW[i] += dy_row[i] * x_row[i] * rms_inv;
        }
    }
}

static bool run_test(
    const char* name,
    int rows, int cols,
    float eps = 1e-6f,
    float tol_dx = 1e-3f,
    float tol_dw = 1e-2f
) {
    const long x_sz = (long)rows * cols;

    // Generate data, round-trip through FP16
    float* h_dY_f32 = new float[x_sz];
    float* h_x_f32  = new float[x_sz];
    float* h_w_f32  = new float[cols];
    half*  h_dY_h16 = new half[x_sz];
    half*  h_x_h16  = new half[x_sz];

    for (long i = 0; i < x_sz; i++) {
        h_x_h16[i]  = __float2half(lcg_randf());
        h_x_f32[i]  = __half2float(h_x_h16[i]);
        h_dY_h16[i] = __float2half(lcg_randf());
        h_dY_f32[i] = __half2float(h_dY_h16[i]);
    }
    for (int i = 0; i < cols; i++)
        h_w_f32[i] = lcg_randf();  // weight is FP32, no round-trip needed

    // CPU reference
    float* h_ref_dX = new float[x_sz];
    float* h_ref_dW = new float[cols]();
    ref_rmsnorm_backward(h_ref_dX, h_ref_dW, h_dY_f32, h_x_f32, h_w_f32, rows, cols, eps);

    // GPU
    half *d_dY, *d_x, *d_dX;
    float *d_w, *d_dW;
    CUDA_CHECK(cudaMalloc(&d_dY, x_sz * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x,  x_sz * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dX, x_sz * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w,  cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW, cols * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_dY, h_dY_h16, x_sz * sizeof(half),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,  h_x_h16,  x_sz * sizeof(half),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w,  h_w_f32,  cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dW, 0, cols * sizeof(float)));

    launch_rmsnorm_backward(d_dX, d_dW, d_dY, d_x, d_w, rows, cols, eps, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify dX
    half* h_dX = new half[x_sz];
    CUDA_CHECK(cudaMemcpy(h_dX, d_dX, x_sz * sizeof(half), cudaMemcpyDeviceToHost));
    float max_err_dx = 0.0f;
    for (long i = 0; i < x_sz; i++) {
        float diff = fabsf(__half2float(h_dX[i]) - h_ref_dX[i]);
        if (diff > max_err_dx) max_err_dx = diff;
    }

    // Verify dW (FP32 output, but accumulated via atomicAdd across rows)
    float* h_dW = new float[cols];
    CUDA_CHECK(cudaMemcpy(h_dW, d_dW, cols * sizeof(float), cudaMemcpyDeviceToHost));
    float max_err_dw = 0.0f;
    for (int i = 0; i < cols; i++) {
        float diff = fabsf(h_dW[i] - h_ref_dW[i]);
        if (diff > max_err_dw) max_err_dw = diff;
    }

    // dW tolerance scales with rows (atomicAdd FP32 accumulation from FP16 inputs)
    float adj_tol_dw = tol_dw * sqrtf((float)rows / 32.0f);
    if (adj_tol_dw < tol_dw) adj_tol_dw = tol_dw;

    bool passed = (max_err_dx < tol_dx) && (max_err_dw < adj_tol_dw);
    printf("[%s] %-50s dX_err=%.6f(tol=%.0e)  dW_err=%.6f(tol=%.3f)\n",
           passed ? "PASS" : "FAIL", name,
           max_err_dx, tol_dx, max_err_dw, adj_tol_dw);

    CUDA_CHECK(cudaFree(d_dY));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_dX));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_dW));
    delete[] h_dY_f32; delete[] h_dY_h16;
    delete[] h_x_f32;  delete[] h_x_h16;
    delete[] h_w_f32;
    delete[] h_ref_dX; delete[] h_ref_dW;
    delete[] h_dX;     delete[] h_dW;

    return passed;
}

int main() {
    printf("=== rmsnorm_backward tests ===\n\n");

    bool all_pass = true;

    // 1. Single row, head_dim=128 (QK-norm)
    all_pass &= run_test("1 row, cols=128 (QK-norm)",        1, 128);

    // 2. Single row, hidden=1024
    all_pass &= run_test("1 row, cols=1024 (hidden)",         1, 1024);

    // 3. 32 rows (decode batch)
    all_pass &= run_test("32 rows, cols=1024",               32, 1024);

    // 4. 512 rows (prefill)
    all_pass &= run_test("512 rows, cols=1024 (prefill)",   512, 1024);

    // 5. Large cols
    all_pass &= run_test("16 rows, cols=3072",               16, 3072);

    printf("\n%s\n", all_pass ? "All tests PASSED." : "Some tests FAILED.");
    return all_pass ? 0 : 1;
}
