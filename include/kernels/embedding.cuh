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
