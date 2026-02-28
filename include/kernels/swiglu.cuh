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
