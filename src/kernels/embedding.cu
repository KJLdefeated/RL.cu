#include "kernels/embedding.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void embedding_kernel(
    half*       __restrict__ out,       // [num_tokens, hidden_size]
    const half* __restrict__ weight,    // [vocab_size, hidden_size]
    const int*  __restrict__ token_ids, // [num_tokens]
    int num_tokens,
    int hidden_size
) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;

    const int vocab_id = token_ids[tok];

    const half* src = weight + (long)vocab_id * hidden_size;
    half*       dst = out    + (long)tok       * hidden_size;

    for (int d = threadIdx.x; d < hidden_size; d += blockDim.x)
        dst[d] = src[d];
}

void launch_embedding(
    half*        out,
    const half*  weight,
    const int*   token_ids,
    int          num_tokens,
    int          vocab_size,
    int          hidden_size,
    cudaStream_t stream
) {
    if (num_tokens == 0) return;
    (void)vocab_size;  // bounds-checking omitted for performance

    // 256 threads per token: 4 passes for hidden_size=1024
    dim3 grid(num_tokens, 1);
    dim3 block(256, 1);

    embedding_kernel<<<grid, block, 0, stream>>>(
        out, weight, token_ids, num_tokens, hidden_size
    );
}
