#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ============================================================
// Fused RMSNorm + Linear (forward only, inference path)
//
// Computes:  out[T, N] = (x[T, H] / rms(x) * gamma[H]) @ W[N, H]^T
//
// Advantage over separate RMSNorm + cuBLAS GEMM:
//   - x[T, H] is read once from HBM (loaded into SHMEM per block)
//   - Normalised values never written to HBM — avoids the T×H write
//     and T×H read that the cuBLAS path requires for its input pointer
//   - At T=256, H=1024: saves 2 × 256 × 1024 × 2 B = 1 MB per call
//
// Constraints:
//   H must equal 1024 (Qwen3-0.6B hidden size)
//   H must be a multiple of 64 (WMMA tile size constraint)
//   N and T may be any non-negative integer
//
// Tiling: BM=16, BN=64, BK=64  (WMMA 16×16×16 fragments)
//   Grid:  (ceil(N/64), ceil(T/16))
//   Block: 128 threads = 4 warps
//   SHMEM: ~44 KB per block (x=32KB, W=8KB, normed_x=2KB, out=2KB)
// ============================================================

void launch_fused_rmsnorm_linear(
    const half*  x,       // [T, H] — input hidden states (pre-norm)
    const float* gamma,   // [H]    — RMSNorm scale (FP32, same layout as launch_rmsnorm)
    const half*  weight,  // [N, H] — linear weight (row-major, same layout as linear_half)
    half*        out,     // [T, N] — output
    int T, int N, int H,
    float        eps,
    cudaStream_t stream
);
