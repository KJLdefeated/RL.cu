#include "kernels/rmsnorm.cuh"

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
// One block per row.  Shared memory holds one float per warp for the
// inter-warp reduction step (max 32 warps = 1024-thread block).
//
// Reduction strategy:
//   1. Each thread accumulates sum(x^2) over its assigned columns (FP32).
//   2. Warp-level reduction via __shfl_xor_sync.
//   3. Warp leaders write to smem; thread 0 does a sequential reduction over
//      n_warps values and broadcasts the total back through smem.
//   4. All threads normalise their elements: out[i] = x[i] * rsqrt(mean_sq + eps) * w[i]
// ---------------------------------------------------------------------------

__global__ void rmsnorm_kernel(
    half*       out,
    const half* x,
    const half* weight,
    int         cols,
    float       eps
) {
    extern __shared__ float smem[];   // n_warps floats

    const int row    = blockIdx.x;
    const int tid    = threadIdx.x;
    const int lane   = tid & 31;
    const int warp_id = tid >> 5;
    const int n_warps = (blockDim.x + 31) >> 5;

    const half* x_row  = x   + (long)row * cols;
    half*       o_row  = out + (long)row * cols;

    // ------------------------------------------------------------------
    // Step 1: accumulate sum of squares (FP32)
    // ------------------------------------------------------------------
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float xi = __half2float(x_row[i]);
        sum_sq  += xi * xi;
    }

    // Warp-level reduction
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, off);

    // Block-level reduction (only needed when n_warps > 1)
    if (n_warps > 1) {
        if (lane == 0) smem[warp_id] = sum_sq;
        __syncthreads();
        if (tid == 0) {
            float total = smem[0];
            for (int w = 1; w < n_warps; ++w)
                total += smem[w];
            smem[0] = total;
        }
        __syncthreads();
        sum_sq = smem[0];
    }

    const float rms_inv = rsqrtf(sum_sq / (float)cols + eps);

    // ------------------------------------------------------------------
    // Step 2: normalise
    // ------------------------------------------------------------------
    for (int i = tid; i < cols; i += blockDim.x) {
        float xi = __half2float(x_row[i]);
        float wi = __half2float(weight[i]);
        o_row[i] = __float2half(xi * rms_inv * wi);
    }
}

// ---------------------------------------------------------------------------
// Launch wrapper
// ---------------------------------------------------------------------------
void launch_rmsnorm(
    half*        out,
    const half*  x,
    const half*  weight,
    int          rows,
    int          cols,
    float        eps,
    cudaStream_t stream
) {
    // 32 threads  for cols=128  (head_dim  — 1 warp,  4 elements/thread)
    // 128 threads for cols=1024 (hidden    — 4 warps, 8 elements/thread)
    // 256 threads for anything larger
    int threads;
    if      (cols <= 128)  threads = 32;
    else if (cols <= 1024) threads = 128;
    else                   threads = 256;

    const int    n_warps    = threads / 32;
    const size_t smem_bytes = n_warps * sizeof(float);

    rmsnorm_kernel<<<rows, threads, smem_bytes, stream>>>(out, x, weight, cols, eps);
}
