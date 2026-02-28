#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Softmax along the last dimension (row-wise).
// out[row][i] = exp(x[row][i] - max_row) / sum_row
//
// Shape: x, out -> [rows, cols]  (half)
// FP16 I/O, FP32 accumulation internally (online max + sum).
// One block per row; threads iterate over cols in chunks.
void launch_softmax(
    half*        out,
    const half*  x,
    int          rows,
    int          cols,
    cudaStream_t stream = 0
);
