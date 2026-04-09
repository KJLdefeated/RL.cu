#include "kernels/fused_norm_linear.cuh"
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
using namespace nvcuda;

// ============================================================
// Kernel constants
// ============================================================
static constexpr int FUSED_H     = 1024;  // input hidden dim (must match model)
static constexpr int FUSED_BM    = 16;    // output rows per block  (= one WMMA m-tile)
static constexpr int FUSED_BN    = 64;    // output cols per block  (= 4 WMMA n-tiles)
static constexpr int FUSED_BK    = 64;    // hidden-dim tile size   (= 4 WMMA k-tiles)
static constexpr int FUSED_WARPS = FUSED_BN / 16;        // 4 — one warp per n-tile
static constexpr int FUSED_BLOCK = FUSED_WARPS * 32;     // 128 threads

// ============================================================
// fused_rmsnorm_linear_kernel
//
// Grid:  (ceil(N/BN), ceil(T/BM))
// Block: 128 threads = 4 warps
//
// Warp layout: each warp accumulates a [BM=16, 16] output sub-tile.
//   warp 0 → output cols [0,  16)
//   warp 1 → output cols [16, 32)
//   warp 2 → output cols [32, 48)
//   warp 3 → output cols [48, 64)
//
// SHMEM (static, ~44 KB):
//   x_smem   [BM][H]   half   32 KB  — full input tile (kept for all k-tiles)
//   w_smem   [BN][BK]  half    8 KB  — weight macro-tile for current k-iteration
//   nx_smem  [BM][BK]  half    2 KB  — normed x for current k-macro-tile
//   out_smem [BM][BN]  half    2 KB  — staging for FP32 → FP16 result store
//   rms_inv  [BM]      float  64  B  — per-row scale
// ============================================================

__global__ void fused_rmsnorm_linear_kernel(
    const half*  __restrict__ x,      // [T, H]
    const float* __restrict__ gamma,  // [H]
    const half*  __restrict__ weight, // [N, H]
    half*        __restrict__ out,    // [T, N]
    int T, int N, float eps)
{
    const int row_base = blockIdx.y * FUSED_BM;
    const int col_base = blockIdx.x * FUSED_BN;
    const int wid      = threadIdx.x >> 5;
    const int lid      = threadIdx.x & 31;

    __shared__ half  x_smem  [FUSED_BM * FUSED_H];   // 32 KB
    __shared__ half  w_smem  [FUSED_BN * FUSED_BK];  //  8 KB
    __shared__ half  nx_smem [FUSED_BM * FUSED_BK];  //  2 KB
    __shared__ float out_smem[FUSED_BM * FUSED_BN];  //  4 KB (FP32 — required by store_matrix_sync)
    __shared__ float rms_inv [FUSED_BM];              // 64  B

    // ----------------------------------------------------------------
    // Phase 1: load x tile [BM, H] into x_smem
    //   128 threads × 128 iterations = 16384 half elements
    // ----------------------------------------------------------------
    #pragma unroll 1
    for (int i = threadIdx.x; i < FUSED_BM * FUSED_H; i += FUSED_BLOCK) {
        const int r = i / FUSED_H, c = i % FUSED_H;
        const int gr = row_base + r;
        x_smem[i] = (gr < T) ? x[(long)gr * FUSED_H + c] : __float2half(0.f);
    }
    __syncthreads();

    // ----------------------------------------------------------------
    // Phase 2: compute per-row RMS inverse
    //   4 warps × 4 rows each = 16 rows; each warp reduces H=1024 per row
    // ----------------------------------------------------------------
    {
        constexpr int rows_per_warp = FUSED_BM / FUSED_WARPS;  // 4
        #pragma unroll
        for (int ri = 0; ri < rows_per_warp; ri++) {
            const int row = wid * rows_per_warp + ri;
            const half* rp = x_smem + row * FUSED_H;
            float sum_sq = 0.f;
            for (int c = lid; c < FUSED_H; c += 32) {
                const float v = __half2float(rp[c]);
                sum_sq += v * v;
            }
            // Warp reduction
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, off);
            if (lid == 0)
                rms_inv[row] = rsqrtf(sum_sq / FUSED_H + eps);
        }
    }
    __syncthreads();

    // ----------------------------------------------------------------
    // Phase 3: WMMA accumulation over k-macro-tiles
    // ----------------------------------------------------------------
    wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major> B_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>                 C_frag;
    wmma::fill_fragment(C_frag, 0.f);

    constexpr int num_km = FUSED_H / FUSED_BK;  // 16

    #pragma unroll 1
    for (int km = 0; km < num_km; km++) {
        const int k_off = km * FUSED_BK;

        // Load W tile [BN, BK] from weight[col_base:col_base+BN, k_off:k_off+BK]
        for (int i = threadIdx.x; i < FUSED_BN * FUSED_BK; i += FUSED_BLOCK) {
            const int bn = i / FUSED_BK, bk = i % FUSED_BK;
            const int gn = col_base + bn;
            w_smem[i] = (gn < N) ? weight[(long)gn * FUSED_H + k_off + bk]
                                  : __float2half(0.f);
        }

        // Compute normed x tile [BM, BK]:
        //   nx_smem[m][bk] = x_smem[m][k_off+bk] * rms_inv[m] * gamma[k_off+bk]
        for (int i = threadIdx.x; i < FUSED_BM * FUSED_BK; i += FUSED_BLOCK) {
            const int m  = i / FUSED_BK, bk = i % FUSED_BK;
            const int gk = k_off + bk;
            const float xv = __half2float(x_smem[m * FUSED_H + gk]);
            nx_smem[i] = __float2half(xv * rms_inv[m] * gamma[gk]);
        }
        __syncthreads();

        // 4 WMMA k-sub-tiles of 16 each (BK/16 = 4)
        //
        // A [BM=16, 16] row-major from nx_smem  at offset ks*16, ld=BK
        //   A[m,k] = nx_smem[m*BK + ks*16 + k]  ✓
        //
        // B [K=16, N_sub=16] col-major from w_smem at warp's sub-column
        //   We want B[k,n] = weight[col_base + wid*16+n, k_off + ks*16+k]
        //                  = w_smem[(wid*16+n)*BK + ks*16 + k]
        //   With base = w_smem + wid*16*BK + ks*16:
        //     base[n*BK + k] → wmma col_major B with ld=BK  ✓
        #pragma unroll
        for (int ks = 0; ks < FUSED_BK / 16; ks++) {
            wmma::load_matrix_sync(A_frag,
                nx_smem + ks * 16,
                FUSED_BK);
            wmma::load_matrix_sync(B_frag,
                w_smem + wid * 16 * FUSED_BK + ks * 16,
                FUSED_BK);
            wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        }
        __syncthreads();
    }

    // ----------------------------------------------------------------
    // Phase 4: store result
    //   Each warp stores its [BM=16, 16] tile into out_smem at col wid*16.
    //   store_matrix_sync row-major: C[m,n] → out_smem[m*BN + wid*16 + n]
    // ----------------------------------------------------------------
    wmma::store_matrix_sync(out_smem + wid * 16, C_frag, FUSED_BN, wmma::mem_row_major);
    __syncthreads();

    // Convert FP32 accumulator → FP16 and write to global with bounds check
    for (int i = threadIdx.x; i < FUSED_BM * FUSED_BN; i += FUSED_BLOCK) {
        const int m = i / FUSED_BN, n = i % FUSED_BN;
        const int gm = row_base + m, gn = col_base + n;
        if (gm < T && gn < N)
            out[(long)gm * N + gn] = __float2half(out_smem[i]);
    }
}

// ============================================================
// Launch wrapper
// ============================================================
void launch_fused_rmsnorm_linear(
    const half*  x,
    const float* gamma,
    const half*  weight,
    half*        out,
    int T, int N, int H,
    float eps,
    cudaStream_t stream)
{
    if (H != FUSED_H) {
        fprintf(stderr,
            "[fused_rmsnorm_linear] H=%d unsupported (expected %d) — skipping\n",
            H, FUSED_H);
        return;
    }
    if (T <= 0 || N <= 0) return;

    const dim3 grid((N + FUSED_BN - 1) / FUSED_BN,
                    (T + FUSED_BM - 1) / FUSED_BM);
    fused_rmsnorm_linear_kernel<<<grid, FUSED_BLOCK, 0, stream>>>(
        x, gamma, weight, out, T, N, eps);
}
