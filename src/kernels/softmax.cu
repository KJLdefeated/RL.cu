#include "kernels/softmax.cuh"

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
// Row-wise softmax. One block per row; shared memory holds one float per
// warp for block-level reductions.
//
// Two reduction passes per row (two-pass numerically stable softmax):
//   Pass 1: global max via warp + block max reduction
//   Pass 2: sum of exp(x - max) via warp + block sum reduction
//   Pass 3: write exp(x - max) / sum
//
// Handles arbitrarily large cols (threads stride over the row).
// ---------------------------------------------------------------------------

__device__ inline void block_reduce_max(float& val, float* smem, int tid, int n_warps) {
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;

    // Warp-level max
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, off));

    if (n_warps > 1) {
        if (lane == 0) smem[warp_id] = val;
        __syncthreads();
        if (tid == 0) {
            float total = smem[0];
            for (int w = 1; w < n_warps; ++w)
                total = fmaxf(total, smem[w]);
            smem[0] = total;
        }
        __syncthreads();
        val = smem[0];
    }
}

__device__ inline void block_reduce_sum(float& val, float* smem, int tid, int n_warps) {
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;

    // Warp-level sum
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xffffffffu, val, off);

    if (n_warps > 1) {
        if (lane == 0) smem[warp_id] = val;
        __syncthreads();
        if (tid == 0) {
            float total = smem[0];
            for (int w = 1; w < n_warps; ++w)
                total += smem[w];
            smem[0] = total;
        }
        __syncthreads();
        val = smem[0];
    }
}

__global__ void softmax_kernel(
    half*       out,
    const half* x,
    int         cols
) {
    extern __shared__ float smem[];   // n_warps floats (reused across passes)

    const int row     = blockIdx.x;
    const int tid     = threadIdx.x;
    const int n_warps = (blockDim.x + 31) >> 5;

    const half* x_row = x   + (long)row * cols;
    half*       o_row = out + (long)row * cols;

    // Pass 1: find row max
    float local_max = -1e20f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float xi = __half2float(x_row[i]);
        if (xi > local_max) local_max = xi;
    }
    block_reduce_max(local_max, smem, tid, n_warps);
    const float global_max = local_max;

    // Pass 2: sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float xi   = __half2float(x_row[i]);
        local_sum += expf(xi - global_max);
    }
    block_reduce_sum(local_sum, smem, tid, n_warps);
    const float inv_sum = 1.0f / local_sum;

    // Pass 3: write normalised values
    for (int i = tid; i < cols; i += blockDim.x) {
        float xi  = __half2float(x_row[i]);
        o_row[i] = __float2half(expf(xi - global_max) * inv_sum);
    }
}

// Launch wrapper
void launch_softmax(
    half*        out,
    const half*  x,
    int          rows,
    int          cols,
    cudaStream_t stream
) {
    // 256 threads: each handles cols/256 elements per row.
    // Good balance for cols ranging from 1024 (hidden) to 151936 (vocab).
    const int    threads    = 256;
    const int    n_warps    = threads / 32;
    const size_t smem_bytes = n_warps * sizeof(float);

    softmax_kernel<<<rows, threads, smem_bytes, stream>>>(out, x, cols);
}
