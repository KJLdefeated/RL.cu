#include "kernels/rmsnorm.cuh"

__global__ void rmsnorm_kernel(
    half*        out,
    const half*  x,
    const float* weight,
    int          cols,
    float        eps
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
        float wi = weight[i];
        o_row[i] = __float2half(xi * rms_inv * wi);
    }
}

// ============================================================
// Backward kernel for RMSNorm
// Forward: y_i = w_i * x_i * rms_inv,  rms_inv = rsqrt(mean(x^2) + eps)
//
// dX_j = rms_inv * (dY_j * w_j  -  x_j * rms_inv^2 * c / N)
//   where c = sum_i(dY_i * w_i * x_i)
// dW_i += dY_i * x_i * rms_inv   (accumulated across rows via atomicAdd)
//
// Grid: (rows,);  Block: same sizing as forward
// 3 passes per row:
//   Pass 1: sum_sq = sum(x_i^2)  → rms_inv
//   Pass 2: c = sum(dY_i * w_i * x_i)
//   Pass 3: dX_j = ...,  atomicAdd dW_i
// ============================================================

__global__ void rmsnorm_backward_kernel(
    half*        __restrict__ dX,      // [rows, cols]
    float*       __restrict__ dW,      // [cols]  (accumulated via atomicAdd)
    const half*  __restrict__ dY,      // [rows, cols]
    const half*  __restrict__ x,       // [rows, cols]
    const float* __restrict__ weight,  // [cols]
    int          cols,
    float        eps
) {
    extern __shared__ float smem[];   // n_warps floats

    const int row     = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    const int n_warps = (blockDim.x + 31) >> 5;

    const half* x_row  = x  + (long)row * cols;
    const half* dy_row = dY + (long)row * cols;
    half*       dx_row = dX + (long)row * cols;

    // --- Pass 1: sum_sq = sum(x_i^2) → rms_inv ---
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float xi = __half2float(x_row[i]);
        sum_sq += xi * xi;
    }
    // Warp reduce
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, off);
    // Block reduce
    if (n_warps > 1) {
        if (lane == 0) smem[warp_id] = sum_sq;
        __syncthreads();
        if (tid == 0) {
            float total = smem[0];
            for (int w = 1; w < n_warps; ++w) total += smem[w];
            smem[0] = total;
        }
        __syncthreads();
        sum_sq = smem[0];
    }
    const float rms_inv = rsqrtf(sum_sq / (float)cols + eps);

    // --- Pass 2: c = sum(dY_i * w_i * x_i) ---
    float c = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float dyi = __half2float(dy_row[i]);
        float xi  = __half2float(x_row[i]);
        float wi  = weight[i];
        c += dyi * wi * xi;
    }
    // Warp reduce
    for (int off = 16; off > 0; off >>= 1)
        c += __shfl_xor_sync(0xffffffffu, c, off);
    // Block reduce
    if (n_warps > 1) {
        if (lane == 0) smem[warp_id] = c;
        __syncthreads();
        if (tid == 0) {
            float total = smem[0];
            for (int w = 1; w < n_warps; ++w) total += smem[w];
            smem[0] = total;
        }
        __syncthreads();
        c = smem[0];
    }

    const float coeff = c * rms_inv * rms_inv / (float)cols;

    // --- Pass 3: dX and dW ---
    for (int i = tid; i < cols; i += blockDim.x) {
        float dyi = __half2float(dy_row[i]);
        float xi  = __half2float(x_row[i]);
        float wi  = weight[i];

        // dX_j = rms_inv * (dY_j * w_j  -  x_j * coeff)
        float dx_val = rms_inv * (dyi * wi - xi * coeff);
        dx_row[i] = __float2half(dx_val);

        // dW_i += dY_i * x_i * rms_inv  (accumulated across rows)
        atomicAdd(&dW[i], dyi * xi * rms_inv);
    }
}

// ---------------------------------------------------------------------------
// Launch wrapper (backward)
// ---------------------------------------------------------------------------
void launch_rmsnorm_backward(
    half*         dX,
    float*        dW,
    const half*   dY,
    const half*   x,
    const float*  weight,
    int           rows,
    int           cols,
    float         eps,
    cudaStream_t  stream
) {
    int threads;
    if      (cols <= 128)  threads = 32;
    else if (cols <= 1024) threads = 128;
    else                   threads = 256;

    const int    n_warps    = threads / 32;
    const size_t smem_bytes = n_warps * sizeof(float);

    rmsnorm_backward_kernel<<<rows, threads, smem_bytes, stream>>>(
        dX, dW, dY, x, weight, cols, eps
    );
}

// ---------------------------------------------------------------------------
// Launch wrapper (forward)
// ---------------------------------------------------------------------------
void launch_rmsnorm(
    half*         out,
    const half*   x,
    const float*  weight,
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
