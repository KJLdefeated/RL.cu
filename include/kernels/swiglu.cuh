#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// SwiGLU activation (fused): out[i] = silu(gate[i]) * up[i]
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// All tensors: [n_elements] in half (interpret as flat).
// For a [batch, seq, intermediate_size] tensor, pass n = batch*seq*intermediate_size.
// FP16 I/O, FP32 compute.
void launch_swiglu(
    half*        out,
    const half*  gate,
    const half*  up,
    int          n_elements,
    cudaStream_t stream = 0
);

// Backward pass for SwiGLU: out = silu(gate) * up
//   dGate_i = dOut_i * up_i * σ(g_i) * (1 + g_i * (1 - σ(g_i)))
//   dUp_i   = dOut_i * silu(g_i)
// where σ = sigmoid, silu(g) = g * σ(g)
void launch_swiglu_backward(
    half*        dGate,      // [n_elements] gradient for gate input
    half*        dUp,        // [n_elements] gradient for up input
    const half*  dOut,       // [n_elements] upstream gradient
    const half*  gate,       // [n_elements] saved gate from forward
    const half*  up,         // [n_elements] saved up from forward
    int          n_elements,
    cudaStream_t stream = 0
);
