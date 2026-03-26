#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void launch_embedding(
    half*        out,
    const half*  weight,
    const int*   token_ids,
    int          num_tokens,
    int          vocab_size,
    int          hidden_size,
    cudaStream_t stream = 0
);

// Backward pass for embedding: scatter-add gradients into embedding weight table.
//   dW[token_ids[tok], d] += dOut[tok, d]
// dW must be zeroed by the caller before the first backward call.
// Uses atomicAdd(half*) for correctness when multiple tokens share the same vocab_id.
void launch_embedding_backward(
    half*        dW,         // [vocab_size, hidden_size] gradient for embedding weight (accumulated)
    const half*  dOut,       // [num_tokens, hidden_size] upstream gradient
    const int*   token_ids,  // [num_tokens]
    int          num_tokens,
    int          hidden_size,
    cudaStream_t stream = 0
);
