#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void launch_rmsnorm(
    half*         out,
    const half*   x,
    const float*  weight,
    int          rows,
    int          cols,
    float        eps,
    cudaStream_t stream = 0
);
