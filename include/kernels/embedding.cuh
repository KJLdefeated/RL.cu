#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Embedding — token lookup (gather)
// ---------------------------------------------------------------------------
// out[tok] = weight[token_ids[tok]]
//
//   weight    : [vocab_size, hidden_size]  (FP16)
//   token_ids : [num_tokens]              (int32)
//   out       : [num_tokens, hidden_size] (FP16)
//
// num_tokens = B * S (flattened).
// FP16 throughout — no accumulation needed (pure memory copy per row).
//
// Note: the output projection (logits = hidden @ weight^T) reuses the same
// weight matrix but is handled via cuBLAS, not this kernel.
// ---------------------------------------------------------------------------
void launch_embedding(
    half*        out,
    const half*  weight,
    const int*   token_ids,
    int          num_tokens,
    int          vocab_size,
    int          hidden_size,
    cudaStream_t stream = 0
);
