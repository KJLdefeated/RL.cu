#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

static constexpr int MAX_SAMPLER_TOP_K = 1024;

void launch_sampler(
    const half*        logits,
    float*             temp_probs,     // pre-allocated [num_tokens, vocab_size]
    int                num_tokens,
    int                vocab_size,
    int                top_k,
    float              top_p,
    float              temperature,    // 0 = greedy argmax
    unsigned long long seed,
    int64_t*           output_ids,
    cudaStream_t       stream = 0
);
