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

// ============================================================
// Backward: out = silu(gate) * up
//   dGate_i = dOut_i * up_i * σ(g_i) * (1 + g_i * (1 - σ(g_i)))
//   dUp_i   = dOut_i * silu(g_i)
// Elementwise — no reductions, no shared memory.
// ============================================================

__global__ void swiglu_backward_kernel(
    half*       __restrict__ dGate,
    half*       __restrict__ dUp,
    const half* __restrict__ dOut,
    const half* __restrict__ gate,
    const half* __restrict__ up,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float g   = __half2float(gate[i]);
    const float u   = __half2float(up[i]);
    const float dy  = __half2float(dOut[i]);

    const float sig  = 1.0f / (1.0f + expf(-g));      // σ(g)
    const float silu = g * sig;                         // silu(g)

    // silu'(g) = σ(g) * (1 + g * (1 - σ(g)))
    const float silu_grad = sig * (1.0f + g * (1.0f - sig));

    dGate[i] = __float2half(dy * u * silu_grad);
    dUp[i]   = __float2half(dy * silu);
}

void launch_swiglu_backward(
    half*        dGate,
    half*        dUp,
    const half*  dOut,
    const half*  gate,
    const half*  up,
    int          n_elements,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks  = (n_elements + threads - 1) / threads;
    swiglu_backward_kernel<<<blocks, threads, 0, stream>>>(
        dGate, dUp, dOut, gate, up, n_elements
    );
}
