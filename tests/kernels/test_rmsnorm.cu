// GoogleTest correctness tests; pass --benchmark to also print kernel timings.
#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <tuple>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "kernels/rmsnorm.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

class RMSNormTest : public testing::TestWithParam<std::tuple<int, int>> {
protected:
    half *d_x = nullptr, *d_out = nullptr;
    float *d_w = nullptr;

    void SetUp() override {
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && count == 0)) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
        const int rows = std::get<0>(GetParam()), cols = std::get<1>(GetParam());
        ASSERT_EQ(cudaMalloc(&d_x, rows * cols * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_out, rows * cols * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_w, cols * sizeof(float)), cudaSuccess);
    }

    void TearDown() override {
        if (d_x) cudaFree(d_x);
        if (d_out) cudaFree(d_out);
        if (d_w) cudaFree(d_w);
    }

    void CheckAgainstCPU(bool zeros) {
        const int rows = std::get<0>(GetParam()), cols = std::get<1>(GetParam());
        constexpr float eps = 1e-5f;
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<half> x(rows * cols), out(rows * cols);
        std::vector<float> weights(cols), expected(rows * cols);
        for (auto& value : x) value = __float2half(zeros ? 0.0f : dist(rng));
        for (auto& value : weights) value = 1.0f + 0.5f * dist(rng);

        // Reference uses the rounded FP16 input and the FP32 weights consumed by CUDA.
        for (int row = 0; row < rows; ++row) {
            float sum_sq = 0.0f;
            for (int col = 0; col < cols; ++col) {
                const float value = __half2float(x[row * cols + col]);
                sum_sq += value * value;
            }
            const float inv = 1.0f / std::sqrt(sum_sq / cols + eps);
            for (int col = 0; col < cols; ++col) {
                expected[row * cols + col] = __half2float(x[row * cols + col]) * inv * weights[col];
            }
        }

        ASSERT_EQ(cudaMemcpy(d_x, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_w, weights.data(), cols * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        launch_rmsnorm(d_out, d_x, d_w, rows, cols, eps);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(out.data(), d_out, out.size() * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);
        for (size_t i = 0; i < out.size(); ++i) {
            const float actual = __half2float(out[i]);
            // NaN comparisons cannot be allowed to silently leave max_err at zero.
            ASSERT_TRUE(std::isfinite(actual)) << "non-finite output at index " << i;
            ASSERT_TRUE(std::isfinite(expected[i])) << "non-finite reference at index " << i;
            EXPECT_LT(std::fabs(actual - expected[i]), 1e-3f)
                << "index=" << i << " actual=" << actual << " expected=" << expected[i];
        }
    }
};

TEST_P(RMSNormTest, MatchesCPUReference) { CheckAgainstCPU(false); }
TEST_P(RMSNormTest, ZeroInputRemainsFinite) { CheckAgainstCPU(true); }

INSTANTIATE_TEST_SUITE_P(Shapes, RMSNormTest, testing::Values(
    std::make_tuple(4, 128), std::make_tuple(4, 1024),
    std::make_tuple(16, 1024), std::make_tuple(1, 128),
    std::make_tuple(1, 1), std::make_tuple(3, 31),
    std::make_tuple(3, 33), std::make_tuple(3, 127),
    std::make_tuple(3, 129), std::make_tuple(3, 1025)));

static void run_benchmark(const char* name, int rows, int cols,
                          float eps = 1e-5f,
                          int warmup = 10, int iters = 200) {
    const int N = rows * cols;
    half  *d_x, *d_out;
    float *d_w;
    CUDA_CHECK(cudaMalloc(&d_x,   N    * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w,   cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N    * sizeof(half)));

    // Use defined input data for timing as well as correctness tests.
    CUDA_CHECK(cudaMemset(d_x, 0, N * sizeof(half)));
    std::vector<float> weights(cols, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_w, weights.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    for (int i = 0; i < warmup; ++i)
        launch_rmsnorm(d_out, d_x, d_w, rows, cols, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < iters; ++i)
        launch_rmsnorm(d_out, d_x, d_w, rows, cols, eps);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    const float us    = ms * 1000.0f / iters;
    // x (half) + out (half) + weight (float)
    const float bytes = (float)(2 * N) * sizeof(half) + (float)cols * sizeof(float);
    const float bw_gb = bytes / (us * 1e-6f) / 1e9f;

    printf("[BENCH] %-44s  %7.2f us  %6.1f GB/s\n", name, us, bw_gb);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_out);
}

int main(int argc, char** argv) {
    bool benchmark = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--benchmark") == 0) {
            benchmark = true;
            for (int j = i; j + 1 < argc; ++j) argv[j] = argv[j + 1];
            argv[--argc] = nullptr;
            --i;
        }
    }
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    int count = 0;
    if (benchmark && result == 0 && cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
        run_benchmark("RMSNorm rows=4    cols=128", 4, 128);
        run_benchmark("RMSNorm rows=4    cols=1024", 4, 1024);
        run_benchmark("RMSNorm rows=16   cols=1024", 16, 1024);
        run_benchmark("RMSNorm rows=2048 cols=128", 2048, 128);
        run_benchmark("RMSNorm rows=2048 cols=1024", 2048, 1024);
    }
    return result;
}
