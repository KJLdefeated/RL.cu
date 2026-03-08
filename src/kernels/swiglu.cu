#include "kernels/swiglu.cuh"

__global__ void swiglu_kernel(
    half*       out,
    const half* gate,
    const half* up,
    int         n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float g    = __half2float(gate[i]);
    const float u    = __half2float(up[i]);
    const float silu = g / (1.0f + expf(-g));   // silu(g)
    out[i] = __float2half(silu * u);
}

// Launch wrapper
void launch_swiglu(
    half*        out,
    const half*  gate,
    const half*  up,
    int          n_elements,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks  = (n_elements + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads, 0, stream>>>(out, gate, up, n_elements);
}
