#include "kernels/attention.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <mma.h>

using namespace nvcuda;

// =============================================================================
// Flash Attention kernels
// =============================================================================
//
// Prefill: WMMA Tensor Core kernel
//   block = (32) — 1 warp handles one Br=16 Q tile against all KV tiles.
//   Q×K^T and P×V computed via 16×16×16 WMMA ops (8 steps over HEAD_DIM=128).
//   Online softmax maintained in shared memory per row.
//   ~22 KB shared memory per block; one warp → high occupancy.
//
// Backward (dQ, dK, dV): warp-parallel reduction kernel (unchanged)
//   block = (32, Br) — 1 warp per Q/KV row, 4 dims per thread.
//   DIMS_PER_THREAD = HEAD_DIM / WARP_SIZE = 128 / 32 = 4
//   Dot products via __shfl_xor_sync warp reduction.
//
// =============================================================================

static constexpr int WARP_SIZE        = 32;
static constexpr int DIMS_PER_THREAD  = 4;   // HEAD_DIM(128) / WARP_SIZE(32)

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

// =============================================================================
// FA2 Prefill Kernel  (WMMA Tensor Core, 4 warps, Bc=64, async double-buffer K/V)
// =============================================================================
//
// Grid:  (ceil(S/Br), H_q, B)
// Block: (NUM_WARPS * 32 = 128)  — 4 warps
//
// Warp roles (each warp handles a non-overlapping tile):
//   QK^T : warp w → S_smem[:, w*16:(w+1)*16]   (1 WMMA tile of [Br=16, 16])
//   PV   : warp w → O_smem[:, w*32:(w+1)*32]   (2 WMMA tiles of [Br=16, 16])
//
// Shared memory per block (~95 KB): compile-time offsets (no runtime allocation chain).
//   K/V are double-buffered (ping-pong) with cp.async to hide memory latency.
//   Buffer 0 is loaded while buffer 1 is computed, and vice versa.
//
//   Q_smem  [16][ 136] half:         4352 B  @ offset 0
//   K_smem[0][64][136] half:        17408 B  @ offset 4352
//   K_smem[1][64][136] half:        17408 B  @ offset 21760
//   V_smem[0][64][136] half:        17408 B  @ offset 39168
//   V_smem[1][64][136] half:        17408 B  @ offset 56576
//   S_smem  [16][  64] float:        4096 B  @ offset 73984
//   P_smem  [16][  64] half:         2048 B  @ offset 78080
//   O_smem  [16][ 128] float:        8192 B  @ offset 80128
//   warp_tmp[ 4][ 16][32] float:     8192 B  @ offset 88320
//   s_row_max/sum/alpha[16] float:    192 B  @ offset 96512
//   Total: 96704 B (~94.4 KB)
//
// Pipeline per KV tile:
//   1. Prologue (before loop): cp.async tile-0 → buf-0, commit_group
//   2. Each iteration i: cp.async tile-(i+1) → buf-(1-cur), commit_group,
//      wait_group(1) [cur buf ready], __syncthreads, compute, __syncthreads
//   3. Last iteration: wait_group(0) [drain final group]
//
// OOB K/V rows (kv_row >= S): cp.async skipped (smem retains prev. tile's data,
//   correctness preserved: OOB K scores masked to -INF; OOB V rows have P=0).
//
// Barriers per KV tile: 7 (same as single-buffer; async hides load latency)
// =============================================================================

// =============================================================================

template<int HEAD_DIM, int Br, int Bc, int NUM_WARPS>
__global__ void flash_attention_prefill_kernel(
    const half* __restrict__ Q,   // [B, S, H_q,  D]
    const half* __restrict__ K,   // [B, S, H_kv, D]
    const half* __restrict__ V,   // [B, S, H_kv, D]
    half*       __restrict__ O,   // [B, S, H_q,  D]
    float*      __restrict__ LSE, // [B, S, H_q] or nullptr
    int B, int S, int H_q, int H_kv, float scale
) {
    static_assert(Br == 16 && HEAD_DIM % 16 == 0,
                  "WMMA FA2: Br must be 16; HEAD_DIM must be divisible by 16");
    static_assert(Bc == NUM_WARPS * 16,
                  "WMMA FA2: each warp handles exactly 16 KV cols → Bc = NUM_WARPS*16");
    static_assert(HEAD_DIM % (NUM_WARPS * 16) == 0,
                  "WMMA FA2: HEAD_DIM must be divisible by NUM_WARPS*16");

    // Each warp owns KV_COLS_PER_WARP=16 score cols and OUT_COLS_PER_WARP=32 output cols
    constexpr int TOTAL_THREADS     = NUM_WARPS * WARP_SIZE;
    constexpr int KV_COLS_PER_WARP  = Bc / NUM_WARPS;           // 64/4 = 16
    constexpr int OUT_COLS_PER_WARP = HEAD_DIM / NUM_WARPS;     // 128/4 = 32
    constexpr int OUT_FRAGS         = OUT_COLS_PER_WARP / 16;   // 32/16 = 2

    const int warp_id     = threadIdx.x / WARP_SIZE;
    const int tx          = threadIdx.x;
    const int q_tile      = blockIdx.x;
    const int h_q         = blockIdx.y;
    const int b           = blockIdx.z;
    const int h_kv        = h_q * H_kv / H_q;
    const int q_row_start = q_tile * Br;

    constexpr int PAD = 8;   // half-row padding for bank-conflict avoidance

    // Dynamic smem (~94 KB); K/V double-buffered with cp.async.  Launcher sets carveout.
    extern __shared__ char dyn_smem[];

    // Double-buffered K/V smem — all offsets computed at compile time from template params.
    // Layout (bytes): Q=0  K0=4352  K1=21760  V0=39168  V1=56576
    //                 S=73984  P=78080  O=80128  WT=88320  stats=96512
    constexpr int KV_STRIDE   = HEAD_DIM + PAD;   // half-elements per padded row (136)
    constexpr int SZ_Q        = Br * KV_STRIDE * sizeof(half);   //  4352 B
    constexpr int SZ_KV       = Bc * KV_STRIDE * sizeof(half);   // 17408 B
    constexpr int OFF_K0      = SZ_Q;
    constexpr int OFF_K1      = OFF_K0 + SZ_KV;
    constexpr int OFF_V0      = OFF_K1 + SZ_KV;
    constexpr int OFF_V1      = OFF_V0 + SZ_KV;
    constexpr int OFF_S       = OFF_V1 + SZ_KV;            // 73984: first float region
    constexpr int OFF_P       = OFF_S  + Br * Bc * sizeof(float);             // 78080
    constexpr int OFF_O       = OFF_P  + Br * Bc * sizeof(half);              // 80128
    constexpr int OFF_WT      = OFF_O  + Br * HEAD_DIM * sizeof(float);       // 88320
    constexpr int OFF_RMAX    = OFF_WT + NUM_WARPS * Br * OUT_COLS_PER_WARP * sizeof(float); // 96512
    constexpr int OFF_RSUM    = OFF_RMAX + Br * sizeof(float);   // 96576
    constexpr int OFF_ALPHA   = OFF_RSUM + Br * sizeof(float);   // 96640

    half*  Q_smem    = (half*) (dyn_smem + 0);
    half*  K_buf[2]  = { (half*)(dyn_smem + OFF_K0), (half*)(dyn_smem + OFF_K1) };
    half*  V_buf[2]  = { (half*)(dyn_smem + OFF_V0), (half*)(dyn_smem + OFF_V1) };
    float* S_smem    = (float*)(dyn_smem + OFF_S);
    half*  P_smem    = (half*) (dyn_smem + OFF_P);
    float* O_smem    = (float*)(dyn_smem + OFF_O);
    float* warp_tmp  = (float*)(dyn_smem + OFF_WT);
    float* s_row_max = (float*)(dyn_smem + OFF_RMAX);
    float* s_row_sum = (float*)(dyn_smem + OFF_RSUM);
    float* s_alpha   = (float*)(dyn_smem + OFF_ALPHA);

    // 2D indexing helpers
    auto Qs  = [&](int r, int c) -> half&  { return Q_smem[r*(HEAD_DIM+PAD)+c]; };
    auto Ss  = [&](int r, int c) -> float& { return S_smem[r*Bc+c]; };
    auto Ps  = [&](int r, int c) -> half&  { return P_smem[r*Bc+c]; };
    auto Os  = [&](int r, int c) -> float& { return O_smem[r*HEAD_DIM+c]; };
    auto Wt  = [&](int w, int r, int c) -> float& { return warp_tmp[(w*Br+r)*OUT_COLS_PER_WARP+c]; };

    // Initialize O_smem and softmax stats
    for (int i = tx; i < Br * HEAD_DIM; i += TOTAL_THREADS)
        Os(i / HEAD_DIM, i % HEAD_DIM) = 0.0f;
    if (tx < Br) { s_row_max[tx] = -INFINITY; s_row_sum[tx] = 0.0f; }

    // Load Q tile [Br, HEAD_DIM] — cooperatively across all 4 warps
    for (int i = tx; i < Br * HEAD_DIM; i += TOTAL_THREADS) {
        const int r = i / HEAD_DIM, c = i % HEAD_DIM;
        const int q_row = q_row_start + r;
        Qs(r, c) = (q_row < S)
            ? Q[((long)(b * S + q_row) * H_q + h_q) * HEAD_DIM + c]
            : __float2half(0.0f);
    }
    __syncthreads();

    const int num_kv_active = min((S + Bc - 1) / Bc, (q_row_start + Br - 1) / Bc + 1);

    // -------------------------------------------------------------------------
    // Async K/V load: cp.async.ca (16-byte / 8-half chunks, L1-cached).
    // OOB rows (kv_row >= S) are skipped; correctness is preserved because:
    //   - OOB K scores are masked to -INF after QK^T
    //   - OOB V rows have P=0 so contribute nothing to the output
    // The smem for skipped rows retains stale data from a previous tile, but
    // that data is never read (P=0 or score=-INF).
    // Pipeline structure (stage depth = 2):
    //   Prologue : issue tile-0 → buf-0, commit_group
    //   Iter i   : issue tile-(i+1) → buf-(1-cur), commit_group,
    //              wait_group(1) [cur ready], sync, compute, sync
    //   Last iter: wait_group(0)  [drain final group]
    // -------------------------------------------------------------------------
    constexpr int ASYNC_STEPS = Bc * HEAD_DIM / 8;  // 16-byte (8-half) chunks per tile

    // Prologue: issue tile-0 into buf-0 before the loop
    {
        const int kv_base = 0;
        for (int i = tx; i < ASYNC_STEPS; i += TOTAL_THREADS) {
            const int r  = i / (HEAD_DIM / 8);
            const int c8 = (i % (HEAD_DIM / 8)) * 8;
            const int kv_row = kv_base + r;
            if (kv_row < S) {
                const long g = ((long)(b*S+kv_row)*H_kv+h_kv)*HEAD_DIM + c8;
                unsigned int sk = __cvta_generic_to_shared(&K_buf[0][r*KV_STRIDE+c8]);
                unsigned int sv = __cvta_generic_to_shared(&V_buf[0][r*KV_STRIDE+c8]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                             :: "r"(sk), "l"((unsigned long long)(K + g)) : "memory");
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                             :: "r"(sv), "l"((unsigned long long)(V + g)) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }

    for (int kv_tile = 0; kv_tile < num_kv_active; kv_tile++) {
        const int kv_start = kv_tile * Bc;
        const int cur      = kv_tile & 1;
        half* K_cur        = K_buf[cur];
        half* V_cur        = V_buf[cur];

        // Issue prefetch for next tile into the opposite buffer, then wait for cur.
        if (kv_tile + 1 < num_kv_active) {
            const int nxt_base = (kv_tile + 1) * Bc;
            const int nxt      = 1 - cur;
            for (int i = tx; i < ASYNC_STEPS; i += TOTAL_THREADS) {
                const int r  = i / (HEAD_DIM / 8);
                const int c8 = (i % (HEAD_DIM / 8)) * 8;
                const int kv_row = nxt_base + r;
                if (kv_row < S) {
                    const long g = ((long)(b*S+kv_row)*H_kv+h_kv)*HEAD_DIM + c8;
                    unsigned int sk = __cvta_generic_to_shared(&K_buf[nxt][r*KV_STRIDE+c8]);
                    unsigned int sv = __cvta_generic_to_shared(&V_buf[nxt][r*KV_STRIDE+c8]);
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                                 :: "r"(sk), "l"((unsigned long long)(K + g)) : "memory");
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                                 :: "r"(sv), "l"((unsigned long long)(V + g)) : "memory");
                }
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
            // wait_group(1): all but the last committed group (next tile's) must be done
            asm volatile("cp.async.wait_group 1;" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        __syncthreads();

        // --- QK^T: warp w → S_smem[:, w*KV_COLS_PER_WARP] ---
        {
            const int kvc = warp_id * KV_COLS_PER_WARP;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_acc;
            wmma::fill_fragment(s_acc, 0.0f);
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 16) {
                wmma::load_matrix_sync(q_frag, Q_smem + d, HEAD_DIM + PAD);
                wmma::load_matrix_sync(k_frag, K_cur + kvc*KV_STRIDE + d, KV_STRIDE);
                wmma::mma_sync(s_acc, q_frag, k_frag, s_acc);
            }
            // Each warp stores to its non-overlapping column slice; stride = Bc
            wmma::store_matrix_sync(S_smem + kvc, s_acc, Bc, wmma::mem_row_major);
        }
        __syncthreads();

        // --- Scale + causal mask (128 threads, 8 elements each for Br*Bc=1024) ---
        for (int i = tx; i < Br * Bc; i += TOTAL_THREADS) {
            const int r = i / Bc, c = i % Bc;
            const int q_pos = q_row_start + r, kv_pos = kv_start + c;
            float val = Ss(r, c) * scale;
            if (kv_pos > q_pos || q_pos >= S || kv_pos >= S) val = -INFINITY;
            Ss(r, c) = val;
        }
        __syncthreads();

        // --- Softmax: all 128 threads participate ---
        // TPR=8 threads per row, CPT=8 cols per thread.
        // shfl_xor(4,2,1) reduces within each 8-thread row-group without crossing rows:
        //   row r occupies warp lanes (r%4)*8 .. (r%4)*8+7; xor with 4,2,1 stays in [0..7] sub-group.
        {
            constexpr int TPR = TOTAL_THREADS / Br;   // 128/16 = 8 threads per row
            constexpr int CPT = Bc / TPR;             // 64/8  = 8 cols per thread
            static_assert(Bc % TPR == 0, "Bc must divide evenly across TOTAL_THREADS/Br");

            const int row      = tx / TPR;
            const int col_start = (tx % TPR) * CPT;
            const bool valid_row = (q_row_start + row < S);

            // Phase 1: find new row-max (initialize with running max from previous tiles)
            const float old_max = s_row_max[row];
            float thread_max    = old_max;
            if (valid_row) {
                for (int c = col_start; c < col_start + CPT; c++)
                    thread_max = fmaxf(thread_max, Ss(row, c));
            }
            // 3-step butterfly reduce within 8-thread row-group
            thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, 4));
            thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, 2));
            thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, 1));
            // all 8 threads now hold the same new_max for their row

            const float alpha = expf(old_max - thread_max);

            // Phase 2: compute exp(x - new_max) and sum
            float thread_sum = 0.0f;
            if (valid_row) {
                for (int c = col_start; c < col_start + CPT; c++) {
                    const float p = expf(Ss(row, c) - thread_max);
                    Ps(row, c) = __float2half(p);
                    thread_sum += p;
                }
            } else {
                for (int c = col_start; c < col_start + CPT; c++)
                    Ps(row, c) = __float2half(0.0f);
            }
            // Reduce sum within 8-thread row-group
            thread_sum += __shfl_xor_sync(0xFFFFFFFF, thread_sum, 4);
            thread_sum += __shfl_xor_sync(0xFFFFFFFF, thread_sum, 2);
            thread_sum += __shfl_xor_sync(0xFFFFFFFF, thread_sum, 1);

            // First thread in each row-group writes back softmax stats
            if (tx % TPR == 0) {
                s_row_max[row] = thread_max;
                s_row_sum[row] = s_row_sum[row] * alpha + thread_sum;
                s_alpha  [row] = valid_row ? alpha : 1.0f;
            }
        }
        __syncthreads();

        // --- Rescale O_smem by alpha (128 threads, all warps, 16 elements each) ---
        for (int i = tx; i < Br * HEAD_DIM; i += TOTAL_THREADS)
            Os(i / HEAD_DIM, i % HEAD_DIM) *= s_alpha[i / HEAD_DIM];
        __syncthreads();

        // --- PV: warp w → warp_tmp[w][:][0..OUT_COLS_PER_WARP] ---
        // P_smem[16, 64] × V_smem[cur_buf][64, w*32:(w+1)*32] → O contribution [16, 32]
        {
            const int outc = warp_id * OUT_COLS_PER_WARP;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag[OUT_FRAGS];
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_acc[OUT_FRAGS];
            #pragma unroll
            for (int f = 0; f < OUT_FRAGS; f++) wmma::fill_fragment(o_acc[f], 0.0f);

            // Inner loop over Bc=64 K-dimension in steps of 16 (4 iterations)
            #pragma unroll
            for (int k = 0; k < Bc; k += 16) {
                wmma::load_matrix_sync(p_frag, P_smem + k, Bc);
                #pragma unroll
                for (int f = 0; f < OUT_FRAGS; f++) {
                    wmma::load_matrix_sync(v_frag[f], V_cur + k*KV_STRIDE + outc + f*16,
                                           HEAD_DIM + PAD);
                    wmma::mma_sync(o_acc[f], p_frag, v_frag[f], o_acc[f]);
                }
            }
            // Store [16, 32] result into per-warp staging buffer
            #pragma unroll
            for (int f = 0; f < OUT_FRAGS; f++)
                wmma::store_matrix_sync(&Wt(warp_id, 0, f * 16),
                                        o_acc[f], OUT_COLS_PER_WARP, wmma::mem_row_major);
        }
        __syncthreads();

        // --- Accumulate warp_tmp into O_smem (128 threads, 16 elements each) ---
        for (int i = tx; i < Br * HEAD_DIM; i += TOTAL_THREADS) {
            const int r = i / HEAD_DIM, c = i % HEAD_DIM;
            Os(r, c) += Wt(c / OUT_COLS_PER_WARP, r, c % OUT_COLS_PER_WARP);
        }
        __syncthreads();
    } // end KV tiles

    // --- Normalize and write output ---
    for (int i = tx; i < Br * HEAD_DIM; i += TOTAL_THREADS) {
        const int r = i / HEAD_DIM, d = i % HEAD_DIM;
        const int q_row = q_row_start + r;
        if (q_row >= S) continue;
        const float inv = (s_row_sum[r] > 0.0f) ? 1.0f / s_row_sum[r] : 0.0f;
        O[((long)(b * S + q_row) * H_q + h_q) * HEAD_DIM + d] =
            __float2half(Os(r, d) * inv);
    }

    // Write LSE for backward pass
    if (LSE && tx < Br) {
        const int q_row = q_row_start + tx;
        if (q_row < S)
            LSE[((long)(b * S + q_row) * H_q + h_q)] =
                s_row_max[tx] + logf(fmaxf(s_row_sum[tx], 1e-10f));
    }
}

// =============================================================================
// Paged Attention Decode Kernel  (unchanged — inference only, already fast)
// =============================================================================

template<int HEAD_DIM, int BLOCK_SIZE>
__global__ void paged_attention_decode_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    half*       __restrict__ out,
    const int*  __restrict__ block_tables,
    const int*  __restrict__ seq_lens,
    float scale,
    int max_blocks_per_seq,
    int H_q, int H_kv
) {
    const int seq_idx = blockIdx.x;
    const int h_q     = blockIdx.y;
    const int h_kv    = h_q * H_kv / H_q;

    const int context_len = seq_lens[seq_idx];
    const int num_blocks  = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float q_reg[HEAD_DIM];
    {
        const half* q_ptr = q + (seq_idx * H_q + h_q) * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++)
            q_reg[d] = __half2float(q_ptr[d]);
    }

    float max_score = -INFINITY;
    float sum_exp   = 0.0f;
    float acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) acc[d] = 0.0f;

    const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    for (int blk = 0; blk < num_blocks; blk++) {
        const int physical_block  = seq_block_table[blk];
        const int tokens_in_block = min(BLOCK_SIZE, context_len - blk * BLOCK_SIZE);

        const long block_base =
            ((long)physical_block * H_kv + h_kv) * BLOCK_SIZE * HEAD_DIM;

        for (int tok = 0; tok < tokens_in_block; tok++) {
            const half* k_ptr = k_cache + block_base + (long)tok * HEAD_DIM;
            const half* v_ptr = v_cache + block_base + (long)tok * HEAD_DIM;

            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++)
                score += q_reg[d] * __half2float(k_ptr[d]);
            score *= scale;

            const float new_max = fmaxf(max_score, score);
            const float alpha   = expf(max_score - new_max);
            const float p       = expf(score - new_max);

            sum_exp = sum_exp * alpha + p;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++)
                acc[d] = acc[d] * alpha + p * __half2float(v_ptr[d]);

            max_score = new_max;
        }
    }

    half* out_ptr = out + (seq_idx * H_q + h_q) * HEAD_DIM;
    const float inv = 1.0f / sum_exp;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++)
        out_ptr[d] = __float2half(acc[d] * inv);
}

// =============================================================================
// FA2 Backward — Kernel 1: Precompute D  (unchanged)
// =============================================================================

template<int HEAD_DIM>
__global__ void flash_attention_bwd_compute_D_kernel(
    const half* __restrict__ O,
    const half* __restrict__ dO,
    float*      __restrict__ D,
    int total_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;

    const half* o_ptr  = O  + (long)row * HEAD_DIM;
    const half* do_ptr = dO + (long)row * HEAD_DIM;

    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++)
        sum += __half2float(o_ptr[d]) * __half2float(do_ptr[d]);

    D[row] = sum;
}

// =============================================================================
// FA2 Backward — Kernel 2 (WMMA): dQ
// =============================================================================
//
// Grid:  (ceil(S/Br), H_q, B)  — same as forward
// Block: (NUM_WARPS * 32 = 128)  — 4 warps (same as forward)
//
// Replaces per-column warp_reduce_sum with 3 WMMA passes per KV tile:
//   a) QK^T:   Q_smem[Br,D] × K_smem[Bc,D]^T → S_smem[Br,Bc]    (same as forward)
//   b) dOV^T:  dO_smem[Br,D] × V_smem[Bc,D]^T → S_smem[Br,Bc]   (reuse S_smem as dp)
//   c) ds×K:   P_smem[Br,Bc] × K_smem[Bc,D] → dQ_acc[Br,D]      (same as forward PV)
//
// P[r,c] = exp(S[r,c] - LSE[r]) using precomputed LSE (no online softmax).
//
// Smem layout (~65 KB, allows 3 blocks/SM on Blackwell 228 KB):
//   Q_smem   [Br][KV_STRIDE] half: 4352 B @ OFF_Q
//   dO_smem  [Br][KV_STRIDE] half: 4352 B @ OFF_DO
//   K_smem   [Bc][KV_STRIDE] half: 17408 B @ OFF_K
//   V_smem   [Bc][KV_STRIDE] half: 17408 B @ OFF_V
//   S_smem   [Br][Bc] float:  4096 B @ OFF_S   (reused as dp)
//   P_smem   [Br][Bc] half:   2048 B @ OFF_P   (reused as ds)
//   dQ_smem  [Br][D]  float:  8192 B @ OFF_DQ
//   warp_tmp [4][Br][OCP] float: 8192 B @ OFF_WT
//   lse/D scalars [Br] float x2:  128 B @ OFF_LSE/D
// =============================================================================

template<int HEAD_DIM, int Br, int Bc, int NUM_WARPS>
__global__ void flash_attention_bwd_dq_wmma_kernel(
    const half*  __restrict__ Q,
    const half*  __restrict__ K,
    const half*  __restrict__ V,
    const half*  __restrict__ dO,
    const float* __restrict__ LSE,
    const float* __restrict__ D,
    half*        __restrict__ dQ,
    int B, int S, int H_q, int H_kv, float scale
) {
    static_assert(Br == 16 && HEAD_DIM % 16 == 0, "WMMA bwd dQ: Br=16 required");
    static_assert(Bc == NUM_WARPS * 16,            "WMMA bwd dQ: Bc = NUM_WARPS*16");

    constexpr int TOTAL_THREADS     = NUM_WARPS * WARP_SIZE;
    constexpr int KV_COLS_PER_WARP  = Bc / NUM_WARPS;
    constexpr int OUT_COLS_PER_WARP = HEAD_DIM / NUM_WARPS;
    constexpr int OUT_FRAGS         = OUT_COLS_PER_WARP / 16;
    constexpr int PAD               = 8;
    constexpr int KV_STRIDE         = HEAD_DIM + PAD;

    constexpr int OFF_Q   = 0;
    constexpr int OFF_DO  = OFF_Q  + Br * KV_STRIDE * sizeof(half);
    constexpr int OFF_K   = OFF_DO + Br * KV_STRIDE * sizeof(half);
    constexpr int OFF_V   = OFF_K  + Bc * KV_STRIDE * sizeof(half);
    constexpr int OFF_S   = OFF_V  + Bc * KV_STRIDE * sizeof(half);  // also dp_smem
    constexpr int OFF_P   = OFF_S  + Br * Bc * sizeof(float);         // also ds_smem
    constexpr int OFF_DQ  = OFF_P  + Br * Bc * sizeof(half);
    constexpr int OFF_WT  = OFF_DQ + Br * HEAD_DIM * sizeof(float);
    constexpr int OFF_LSE = OFF_WT + NUM_WARPS * Br * OUT_COLS_PER_WARP * sizeof(float);
    constexpr int OFF_D   = OFF_LSE + Br * sizeof(float);

    extern __shared__ char dyn_smem[];
    half*  Q_smem   = (half*) (dyn_smem + OFF_Q);
    half*  dO_smem  = (half*) (dyn_smem + OFF_DO);
    half*  K_smem   = (half*) (dyn_smem + OFF_K);
    half*  V_smem   = (half*) (dyn_smem + OFF_V);
    float* S_smem   = (float*)(dyn_smem + OFF_S);
    half*  P_smem   = (half*) (dyn_smem + OFF_P);
    float* dQ_smem  = (float*)(dyn_smem + OFF_DQ);
    float* warp_tmp = (float*)(dyn_smem + OFF_WT);
    float* lse_smem = (float*)(dyn_smem + OFF_LSE);
    float* d_smem   = (float*)(dyn_smem + OFF_D);

    auto Ss  = [&](int r, int c) -> float& { return S_smem [r*Bc+c]; };
    auto Ps  = [&](int r, int c) -> half&  { return P_smem [r*Bc+c]; };
    auto DQs = [&](int r, int c) -> float& { return dQ_smem[r*HEAD_DIM+c]; };
    auto Wt  = [&](int w, int r, int c) -> float& {
                   return warp_tmp[(w*Br+r)*OUT_COLS_PER_WARP+c]; };

    const int warp_id     = threadIdx.x / WARP_SIZE;
    const int tx          = threadIdx.x;
    const int q_tile      = blockIdx.x;
    const int h_q         = blockIdx.y;
    const int b           = blockIdx.z;
    const int h_kv        = h_q * H_kv / H_q;
    const int q_row_start = q_tile * Br;

    // Initialize dQ accumulator
    for (int i = tx; i < Br * HEAD_DIM; i += TOTAL_THREADS)
        DQs(i / HEAD_DIM, i % HEAD_DIM) = 0.0f;

    // Load Q and dO tiles [Br, D] cooperatively
    for (int i = tx; i < Br * HEAD_DIM; i += TOTAL_THREADS) {
        const int r = i / HEAD_DIM, c = i % HEAD_DIM;
        const int q_row = q_row_start + r;
        if (q_row < S) {
            const long idx = ((long)(b*S+q_row)*H_q+h_q)*HEAD_DIM + c;
            Q_smem [r*KV_STRIDE+c] = Q [idx];
            dO_smem[r*KV_STRIDE+c] = dO[idx];
        } else {
            Q_smem [r*KV_STRIDE+c] = __float2half(0.0f);
            dO_smem[r*KV_STRIDE+c] = __float2half(0.0f);
        }
    }
    // Load LSE and D scalars for this Q tile
    if (tx < Br) {
        const int q_row = q_row_start + tx;
        if (q_row < S) {
            const long idx = (long)(b*S+q_row)*H_q + h_q;
            lse_smem[tx] = LSE[idx];
            d_smem  [tx] = D  [idx];
        } else {
            lse_smem[tx] = 0.0f;
            d_smem  [tx] = 0.0f;
        }
    }
    __syncthreads();

    const int num_kv_active = min((S + Bc - 1) / Bc, (q_row_start + Br - 1) / Bc + 1);

    for (int kv_tile = 0; kv_tile < num_kv_active; kv_tile++) {
        const int kv_start = kv_tile * Bc;

        // Load K, V tile [Bc, D]
        for (int i = tx; i < Bc * HEAD_DIM; i += TOTAL_THREADS) {
            const int r = i / HEAD_DIM, c = i % HEAD_DIM;
            const int kv_row = kv_start + r;
            if (kv_row < S) {
                const long idx = ((long)(b*S+kv_row)*H_kv+h_kv)*HEAD_DIM + c;
                K_smem[r*KV_STRIDE+c] = K[idx];
                V_smem[r*KV_STRIDE+c] = V[idx];
            } else {
                K_smem[r*KV_STRIDE+c] = __float2half(0.0f);
                V_smem[r*KV_STRIDE+c] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // (a) QK^T WMMA: warp w → S_smem[:, w*KVC_PER_WARP:(w+1)*KVC_PER_WARP]
        {
            const int kvc = warp_id * KV_COLS_PER_WARP;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_acc;
            wmma::fill_fragment(s_acc, 0.0f);
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 16) {
                wmma::load_matrix_sync(q_frag, Q_smem + d,              KV_STRIDE);
                wmma::load_matrix_sync(k_frag, K_smem + kvc*KV_STRIDE + d, KV_STRIDE);
                wmma::mma_sync(s_acc, q_frag, k_frag, s_acc);
            }
            wmma::store_matrix_sync(S_smem + kvc, s_acc, Bc, wmma::mem_row_major);
        }
        __syncthreads();

        // Scale + causal mask (identical to forward)
        for (int i = tx; i < Br * Bc; i += TOTAL_THREADS) {
            const int r = i / Bc, c = i % Bc;
            const int q_pos = q_row_start + r, kv_pos = kv_start + c;
            float val = Ss(r, c) * scale;
            if (kv_pos > q_pos || q_pos >= S || kv_pos >= S) val = -INFINITY;
            Ss(r, c) = val;
        }
        __syncthreads();

        // P[r,c] = exp(S[r,c] - LSE[r]); all 128 threads, 8 per row, 8 cols each
        {
            constexpr int TPR = TOTAL_THREADS / Br;
            constexpr int CPT = Bc / TPR;
            const int row      = tx / TPR;
            const int col_start = (tx % TPR) * CPT;
            const float lse_r  = lse_smem[row];
            const bool valid_row = (q_row_start + row < S);
            if (valid_row) {
                for (int c = col_start; c < col_start + CPT; c++)
                    Ps(row, c) = __float2half(expf(Ss(row, c) - lse_r));
            } else {
                for (int c = col_start; c < col_start + CPT; c++)
                    Ps(row, c) = __float2half(0.0f);
            }
        }
        __syncthreads();

        // (b) dOV^T WMMA: reuse S_smem as dp_smem
        {
            const int kvc = warp_id * KV_COLS_PER_WARP;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> do_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> v_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> dp_acc;
            wmma::fill_fragment(dp_acc, 0.0f);
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 16) {
                wmma::load_matrix_sync(do_frag, dO_smem + d,             KV_STRIDE);
                wmma::load_matrix_sync(v_frag,  V_smem + kvc*KV_STRIDE + d, KV_STRIDE);
                wmma::mma_sync(dp_acc, do_frag, v_frag, dp_acc);
            }
            wmma::store_matrix_sync(S_smem + kvc, dp_acc, Bc, wmma::mem_row_major);
        }
        __syncthreads();

        // ds = P * (dp - D) → overwrite P_smem with ds; all 128 threads
        {
            constexpr int TPR = TOTAL_THREADS / Br;
            constexpr int CPT = Bc / TPR;
            const int row      = tx / TPR;
            const int col_start = (tx % TPR) * CPT;
            const float D_r    = d_smem[row];
            for (int c = col_start; c < col_start + CPT; c++) {
                const float p_val  = __half2float(Ps(row, c));
                const float dp_val = Ss(row, c);
                Ps(row, c) = __float2half(p_val * (dp_val - D_r));
            }
        }
        __syncthreads();

        // (c) scale × ds × K WMMA: warp w → warp_tmp[w][:][0..OCP]
        {
            const int outc = warp_id * OUT_COLS_PER_WARP;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> ds_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> k_frag[OUT_FRAGS];
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> dq_acc[OUT_FRAGS];
            #pragma unroll
            for (int f = 0; f < OUT_FRAGS; f++) wmma::fill_fragment(dq_acc[f], 0.0f);
            #pragma unroll
            for (int k = 0; k < Bc; k += 16) {
                wmma::load_matrix_sync(ds_frag, P_smem + k, Bc);
                #pragma unroll
                for (int f = 0; f < OUT_FRAGS; f++) {
                    wmma::load_matrix_sync(k_frag[f],
                                           K_smem + k*KV_STRIDE + outc + f*16, KV_STRIDE);
                    wmma::mma_sync(dq_acc[f], ds_frag, k_frag[f], dq_acc[f]);
                }
            }
            #pragma unroll
            for (int f = 0; f < OUT_FRAGS; f++)
                wmma::store_matrix_sync(&Wt(warp_id, 0, f*16),
                                        dq_acc[f], OUT_COLS_PER_WARP, wmma::mem_row_major);
        }
        __syncthreads();

        // Accumulate warp_tmp into dQ_smem (with scale factor)
        for (int i = tx; i < Br * HEAD_DIM; i += TOTAL_THREADS) {
            const int r = i / HEAD_DIM, c = i % HEAD_DIM;
            DQs(r, c) += scale * Wt(c / OUT_COLS_PER_WARP, r, c % OUT_COLS_PER_WARP);
        }
        __syncthreads();
    } // end KV tiles

    // Write dQ to global memory
    for (int i = tx; i < Br * HEAD_DIM; i += TOTAL_THREADS) {
        const int r = i / HEAD_DIM, c = i % HEAD_DIM;
        const int q_row = q_row_start + r;
        if (q_row < S)
            dQ[((long)(b*S+q_row)*H_q+h_q)*HEAD_DIM + c] = __float2half(DQs(r, c));
    }
}

// =============================================================================
// FA2 Backward — Kernel 3 (WMMA): dK, dV  (KV-outer loop)
// =============================================================================
//
// Grid:  (ceil(S/Bkv), H_kv, B)
// Block: (NUM_WARPS * 32 = 128)  — 4 warps
//
// Per Q tile, 4 WMMA passes:
//   a) KQ^T:   K_smem[Bkv,D] × Q_smem[Bq,D]^T → S_smem[Bkv,Bq]
//   b) VdO^T:  V_smem[Bkv,D] × dO_smem[Bq,D]^T → S_smem (dp, reuse)
//   c) P×dO:   P_smem[Bkv,Bq] × dO_smem[Bq,D] → dV_smem
//   d) ds×Q:   P_smem(ds)[Bkv,Bq] × Q_smem[Bq,D] → dK_smem
//
// Smem layout (~73 KB):
//   K_smem   [Bkv][KV_STRIDE] half: 4352 B @ OFF_K
//   V_smem   [Bkv][KV_STRIDE] half: 4352 B @ OFF_V
//   Q_smem   [Bq][KV_STRIDE]  half: 17408 B @ OFF_Q
//   dO_smem  [Bq][KV_STRIDE]  half: 17408 B @ OFF_DO
//   S_smem   [Bkv][Bq] float:  4096 B @ OFF_S  (score, reused as dp)
//   P_smem   [Bkv][Bq] half:   2048 B @ OFF_P  (reused as ds)
//   dV_smem  [Bkv][D] float:   8192 B @ OFF_DV
//   dK_smem  [Bkv][D] float:   8192 B @ OFF_DK
//   warp_tmp [4][Bkv][OCP] float: 8192 B @ OFF_WT
//   lse/d scalars [Bq] float x2:   512 B @ OFF_LSE/D
// =============================================================================

template<int HEAD_DIM, int Bkv, int Bq, int NUM_WARPS>
__global__ void flash_attention_bwd_dkdv_wmma_kernel(
    const half*  __restrict__ Q,
    const half*  __restrict__ K,
    const half*  __restrict__ V,
    const half*  __restrict__ dO,
    const float* __restrict__ LSE,
    const float* __restrict__ D,
    half*        __restrict__ dK,
    half*        __restrict__ dV,
    int B, int S, int H_q, int H_kv, float scale
) {
    static_assert(Bkv == 16 && HEAD_DIM % 16 == 0, "WMMA bwd dkdv: Bkv=16 required");
    static_assert(Bq == NUM_WARPS * 16,             "WMMA bwd dkdv: Bq = NUM_WARPS*16");

    constexpr int TOTAL_THREADS     = NUM_WARPS * WARP_SIZE;
    constexpr int KV_COLS_PER_WARP  = Bq / NUM_WARPS;
    constexpr int OUT_COLS_PER_WARP = HEAD_DIM / NUM_WARPS;
    constexpr int OUT_FRAGS         = OUT_COLS_PER_WARP / 16;
    constexpr int PAD               = 8;
    constexpr int KV_STRIDE         = HEAD_DIM + PAD;

    constexpr int OFF_K   = 0;
    constexpr int OFF_V   = OFF_K  + Bkv * KV_STRIDE * sizeof(half);
    constexpr int OFF_Q   = OFF_V  + Bkv * KV_STRIDE * sizeof(half);
    constexpr int OFF_DO  = OFF_Q  + Bq  * KV_STRIDE * sizeof(half);
    constexpr int OFF_S   = OFF_DO + Bq  * KV_STRIDE * sizeof(half);  // also dp
    constexpr int OFF_P   = OFF_S  + Bkv * Bq * sizeof(float);         // also ds
    constexpr int OFF_DV  = OFF_P  + Bkv * Bq * sizeof(half);
    constexpr int OFF_DK  = OFF_DV + Bkv * HEAD_DIM * sizeof(float);
    constexpr int OFF_WT  = OFF_DK + Bkv * HEAD_DIM * sizeof(float);
    constexpr int OFF_LSE = OFF_WT + NUM_WARPS * Bkv * OUT_COLS_PER_WARP * sizeof(float);
    constexpr int OFF_D   = OFF_LSE + Bq * sizeof(float);

    extern __shared__ char dyn_smem[];
    half*  K_smem   = (half*) (dyn_smem + OFF_K);
    half*  V_smem   = (half*) (dyn_smem + OFF_V);
    half*  Q_smem   = (half*) (dyn_smem + OFF_Q);
    half*  dO_smem  = (half*) (dyn_smem + OFF_DO);
    float* S_smem   = (float*)(dyn_smem + OFF_S);
    half*  P_smem   = (half*) (dyn_smem + OFF_P);
    float* dV_smem  = (float*)(dyn_smem + OFF_DV);
    float* dK_smem  = (float*)(dyn_smem + OFF_DK);
    float* warp_tmp = (float*)(dyn_smem + OFF_WT);
    float* lse_smem = (float*)(dyn_smem + OFF_LSE);
    float* d_smem   = (float*)(dyn_smem + OFF_D);

    // S_smem[r][c] and P_smem[r][c]: r=kv_row (0..Bkv-1), c=q_row (0..Bq-1)
    auto Ss  = [&](int r, int c) -> float& { return S_smem[r*Bq+c]; };
    auto Ps  = [&](int r, int c) -> half&  { return P_smem[r*Bq+c]; };
    auto DVs = [&](int r, int c) -> float& { return dV_smem[r*HEAD_DIM+c]; };
    auto DKs = [&](int r, int c) -> float& { return dK_smem[r*HEAD_DIM+c]; };
    auto Wt  = [&](int w, int r, int c) -> float& {
                   return warp_tmp[(w*Bkv+r)*OUT_COLS_PER_WARP+c]; };

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int tx      = threadIdx.x;
    const int kv_tile = blockIdx.x;
    const int h_kv    = blockIdx.y;
    const int b       = blockIdx.z;
    const int kv_start = kv_tile * Bkv;

    const int n_rep    = H_q / H_kv;
    const int h_q_base = h_kv * n_rep;

    // Initialize dK, dV accumulators
    for (int i = tx; i < Bkv * HEAD_DIM; i += TOTAL_THREADS) {
        DKs(i / HEAD_DIM, i % HEAD_DIM) = 0.0f;
        DVs(i / HEAD_DIM, i % HEAD_DIM) = 0.0f;
    }

    // Load K, V tiles [Bkv, D] once (they don't change across Q tiles or h_rep)
    for (int i = tx; i < Bkv * HEAD_DIM; i += TOTAL_THREADS) {
        const int r = i / HEAD_DIM, c = i % HEAD_DIM;
        const int kv_row = kv_start + r;
        if (kv_row < S) {
            const long idx = ((long)(b*S+kv_row)*H_kv+h_kv)*HEAD_DIM + c;
            K_smem[r*KV_STRIDE+c] = K[idx];
            V_smem[r*KV_STRIDE+c] = V[idx];
        } else {
            K_smem[r*KV_STRIDE+c] = __float2half(0.0f);
            V_smem[r*KV_STRIDE+c] = __float2half(0.0f);
        }
    }
    __syncthreads();

    const int num_q_tiles  = (S + Bq - 1) / Bq;
    const int q_tile_start = kv_start / Bq;  // causal: skip Q tiles before this KV tile

    for (int h_rep = 0; h_rep < n_rep; h_rep++) {
        const int h_q = h_q_base + h_rep;

        for (int q_tile = q_tile_start; q_tile < num_q_tiles; q_tile++) {
            const int q_start = q_tile * Bq;

            // Load Q, dO tiles [Bq, D] and LSE/D scalars for this Q tile
            for (int i = tx; i < Bq * HEAD_DIM; i += TOTAL_THREADS) {
                const int r = i / HEAD_DIM, c = i % HEAD_DIM;
                const int q_row = q_start + r;
                if (q_row < S) {
                    const long idx = ((long)(b*S+q_row)*H_q+h_q)*HEAD_DIM + c;
                    Q_smem [r*KV_STRIDE+c] = Q [idx];
                    dO_smem[r*KV_STRIDE+c] = dO[idx];
                } else {
                    Q_smem [r*KV_STRIDE+c] = __float2half(0.0f);
                    dO_smem[r*KV_STRIDE+c] = __float2half(0.0f);
                }
            }
            if (tx < Bq) {
                const int q_row = q_start + tx;
                if (q_row < S) {
                    const long idx = (long)(b*S+q_row)*H_q + h_q;
                    lse_smem[tx] = LSE[idx];
                    d_smem  [tx] = D  [idx];
                } else {
                    lse_smem[tx] = 0.0f;
                    d_smem  [tx] = 0.0f;
                }
            }
            __syncthreads();

            // (a) KQ^T WMMA: score[Bkv,Bq] = K_smem × Q_smem^T
            // warp w → S_smem[:, w*KVC_PER_WARP:(w+1)*KVC_PER_WARP]
            {
                const int kvc = warp_id * KV_COLS_PER_WARP;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> k_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> q_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_acc;
                wmma::fill_fragment(s_acc, 0.0f);
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d += 16) {
                    wmma::load_matrix_sync(k_frag, K_smem + d,              KV_STRIDE);
                    wmma::load_matrix_sync(q_frag, Q_smem + kvc*KV_STRIDE + d, KV_STRIDE);
                    wmma::mma_sync(s_acc, k_frag, q_frag, s_acc);
                }
                wmma::store_matrix_sync(S_smem + kvc, s_acc, Bq, wmma::mem_row_major);
            }
            __syncthreads();

            // Scale + causal mask: S[kv_row, q_col]
            for (int i = tx; i < Bkv * Bq; i += TOTAL_THREADS) {
                const int r = i / Bq, c = i % Bq;
                const int kv_row = kv_start + r, q_row = q_start + c;
                float val = Ss(r, c) * scale;
                if (kv_row > q_row || kv_row >= S || q_row >= S) val = -INFINITY;
                Ss(r, c) = val;
            }
            __syncthreads();

            // P[kv_row, q_col] = exp(S[kv_row,q_col] - LSE[q_col])
            // All 128 threads: TPR=8 threads per kv_row, CPT=8 q_cols per thread
            {
                constexpr int TPR = TOTAL_THREADS / Bkv;
                constexpr int CPT = Bq / TPR;
                const int row       = tx / TPR;   // kv_row index
                const int col_start = (tx % TPR) * CPT;
                for (int c = col_start; c < col_start + CPT; c++) {
                    const int q_row = q_start + c;
                    const float lse_c = (q_row < S) ? lse_smem[c] : 0.0f;
                    Ps(row, c) = __float2half(expf(Ss(row, c) - lse_c));
                }
            }
            __syncthreads();

            // (b) VdO^T WMMA: dp[Bkv,Bq] = V_smem × dO_smem^T (reuse S_smem as dp)
            {
                const int kvc = warp_id * KV_COLS_PER_WARP;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> v_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> do_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> dp_acc;
                wmma::fill_fragment(dp_acc, 0.0f);
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d += 16) {
                    wmma::load_matrix_sync(v_frag,  V_smem + d,               KV_STRIDE);
                    wmma::load_matrix_sync(do_frag, dO_smem + kvc*KV_STRIDE + d, KV_STRIDE);
                    wmma::mma_sync(dp_acc, v_frag, do_frag, dp_acc);
                }
                wmma::store_matrix_sync(S_smem + kvc, dp_acc, Bq, wmma::mem_row_major);
            }
            __syncthreads();

            // (c) dV += P × dO WMMA (before overwriting P_smem with ds)
            {
                const int outc = warp_id * OUT_COLS_PER_WARP;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> do_frag[OUT_FRAGS];
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> dv_acc[OUT_FRAGS];
                #pragma unroll
                for (int f = 0; f < OUT_FRAGS; f++) wmma::fill_fragment(dv_acc[f], 0.0f);
                #pragma unroll
                for (int k = 0; k < Bq; k += 16) {
                    wmma::load_matrix_sync(p_frag, P_smem + k, Bq);
                    #pragma unroll
                    for (int f = 0; f < OUT_FRAGS; f++) {
                        wmma::load_matrix_sync(do_frag[f],
                                               dO_smem + k*KV_STRIDE + outc + f*16, KV_STRIDE);
                        wmma::mma_sync(dv_acc[f], p_frag, do_frag[f], dv_acc[f]);
                    }
                }
                #pragma unroll
                for (int f = 0; f < OUT_FRAGS; f++)
                    wmma::store_matrix_sync(&Wt(warp_id, 0, f*16),
                                            dv_acc[f], OUT_COLS_PER_WARP, wmma::mem_row_major);
            }
            __syncthreads();
            // Accumulate warp_tmp into dV_smem
            for (int i = tx; i < Bkv * HEAD_DIM; i += TOTAL_THREADS) {
                const int r = i / HEAD_DIM, c = i % HEAD_DIM;
                DVs(r, c) += Wt(c / OUT_COLS_PER_WARP, r, c % OUT_COLS_PER_WARP);
            }
            __syncthreads();

            // ds = P * (dp - D_q) → overwrite P_smem; all 128 threads
            {
                constexpr int TPR = TOTAL_THREADS / Bkv;
                constexpr int CPT = Bq / TPR;
                const int row       = tx / TPR;
                const int col_start = (tx % TPR) * CPT;
                for (int c = col_start; c < col_start + CPT; c++) {
                    const float p_val  = __half2float(Ps(row, c));
                    const float dp_val = Ss(row, c);
                    const float D_q    = d_smem[c];
                    Ps(row, c) = __float2half(p_val * (dp_val - D_q));
                }
            }
            __syncthreads();

            // (d) dK += scale × ds × Q WMMA
            {
                const int outc = warp_id * OUT_COLS_PER_WARP;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> ds_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> q_frag[OUT_FRAGS];
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> dk_acc[OUT_FRAGS];
                #pragma unroll
                for (int f = 0; f < OUT_FRAGS; f++) wmma::fill_fragment(dk_acc[f], 0.0f);
                #pragma unroll
                for (int k = 0; k < Bq; k += 16) {
                    wmma::load_matrix_sync(ds_frag, P_smem + k, Bq);
                    #pragma unroll
                    for (int f = 0; f < OUT_FRAGS; f++) {
                        wmma::load_matrix_sync(q_frag[f],
                                               Q_smem + k*KV_STRIDE + outc + f*16, KV_STRIDE);
                        wmma::mma_sync(dk_acc[f], ds_frag, q_frag[f], dk_acc[f]);
                    }
                }
                #pragma unroll
                for (int f = 0; f < OUT_FRAGS; f++)
                    wmma::store_matrix_sync(&Wt(warp_id, 0, f*16),
                                            dk_acc[f], OUT_COLS_PER_WARP, wmma::mem_row_major);
            }
            __syncthreads();
            // Accumulate warp_tmp into dK_smem (with scale)
            for (int i = tx; i < Bkv * HEAD_DIM; i += TOTAL_THREADS) {
                const int r = i / HEAD_DIM, c = i % HEAD_DIM;
                DKs(r, c) += scale * Wt(c / OUT_COLS_PER_WARP, r, c % OUT_COLS_PER_WARP);
            }
            __syncthreads();
        } // end Q tiles
    } // end h_rep

    // Write dK, dV to global memory
    for (int i = tx; i < Bkv * HEAD_DIM; i += TOTAL_THREADS) {
        const int r = i / HEAD_DIM, c = i % HEAD_DIM;
        const int kv_row = kv_start + r;
        if (kv_row < S) {
            const long idx = ((long)(b*S+kv_row)*H_kv+h_kv)*HEAD_DIM + c;
            dK[idx] = __float2half(DKs(r, c));
            dV[idx] = __float2half(DVs(r, c));
        }
    }
}

// =============================================================================
// Launch wrappers
// =============================================================================

void launch_flash_attention_prefill(
    const half*  Q,
    const half*  K,
    const half*  V,
    half*        O,
    int B, int S, int H_q, int H_kv, int head_dim,
    cudaStream_t stream,
    float*       lse
) {
    constexpr int Br        = 16;   // Q rows per tile (WMMA M dim)
    constexpr int NUM_WARPS = 4;    // warps per block
    constexpr int Bc        = NUM_WARPS * 16;  // = 64: KV cols per tile
    const float scale = 1.0f / sqrtf((float)head_dim);

    dim3 grid((S + Br - 1) / Br, H_q, B);
    dim3 block(NUM_WARPS * WARP_SIZE);  // 128 threads; 4 warps cover Bc=64 in parallel

    // Request dynamic smem > 48 KB default limit (sm_120 supports up to 228 KB).
    // K/V are double-buffered → 2× the single-buffer size.
    constexpr int PAD_LAUNCHER = 8;
    constexpr int KV_STRIDE_L  = 128 + PAD_LAUNCHER;
    const int smem_bytes =
        Br * KV_STRIDE_L * sizeof(half)               // Q_smem
      + 2 * Bc * KV_STRIDE_L * sizeof(half)           // K_buf[2]
      + 2 * Bc * KV_STRIDE_L * sizeof(half)           // V_buf[2]
      + Br * Bc              * sizeof(float)           // S_smem
      + Br * Bc              * sizeof(half)            // P_smem
      + Br * 128             * sizeof(float)           // O_smem
      + NUM_WARPS * Br * (128 / NUM_WARPS) * sizeof(float) // warp_tmp
      + Br * 3               * sizeof(float);          // s_row_max/sum/alpha

    auto kernel_fn = flash_attention_prefill_kernel<128, Br, Bc, NUM_WARPS>;
    cudaFuncSetAttribute(kernel_fn,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);   // 96704 B < device opt-in max (101376 B)

    kernel_fn<<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, lse, B, S, H_q, H_kv, scale
    );
}

void launch_paged_attention_decode(
    const half*  q,
    const half*  k_cache,
    const half*  v_cache,
    half*        out,
    const int*   block_tables,
    const int*   seq_lens,
    int num_seqs, int H_q, int H_kv, int head_dim,
    int max_blocks_per_seq, int block_size,
    cudaStream_t stream
) {
    const float scale = 1.0f / sqrtf((float)head_dim);

    dim3 grid(num_seqs, H_q);
    dim3 block(1);

    paged_attention_decode_kernel<128, 16><<<grid, block, 0, stream>>>(
        q, k_cache, v_cache, out,
        block_tables, seq_lens,
        scale, max_blocks_per_seq, H_q, H_kv
    );
}

void launch_flash_attention_backward(
    const half*  Q,
    const half*  K,
    const half*  V,
    const half*  O,
    const half*  dO,
    const float* lse,
    float*       D_buf,
    half*        dQ,
    half*        dK,
    half*        dV,
    int B, int S, int H_q, int H_kv, int head_dim,
    cudaStream_t stream
) {
    const float scale     = 1.0f / sqrtf((float)head_dim);
    const int   total_rows = B * S * H_q;

    // Step 1: D[i] = dot(O[i], dO[i])  — block=256 is already optimal
    {
        int block_sz = 256;
        int grid_sz  = (total_rows + block_sz - 1) / block_sz;
        flash_attention_bwd_compute_D_kernel<128><<<grid_sz, block_sz, 0, stream>>>(
            O, dO, D_buf, total_rows
        );
    }

    // Step 2: dQ WMMA (Q-outer loop, 4 warps)
    {
        constexpr int Br        = 16;
        constexpr int Bc        = 64;
        constexpr int NUM_WARPS = 4;
        constexpr int PAD_L     = 8;
        constexpr int KV_STRIDE_L = 128 + PAD_L;
        constexpr int OCP = 128 / NUM_WARPS;  // OUT_COLS_PER_WARP

        // OFF_K = SZ_Q*2 + SZ_KV*2 + Br*Bc*(4+2) + Br*128*4 + NUM_WARPS*Br*OCP*4 + Br*4*2
        const int smem_dq =
              2 * Br * KV_STRIDE_L * sizeof(half)          // Q, dO
            + 2 * Bc * KV_STRIDE_L * sizeof(half)          // K, V
            + Br * Bc  * sizeof(float)                      // S_smem
            + Br * Bc  * sizeof(half)                       // P_smem
            + Br * 128 * sizeof(float)                      // dQ_smem
            + NUM_WARPS * Br * OCP * sizeof(float)          // warp_tmp
            + Br * 2 * sizeof(float);                       // lse + D scalars

        dim3 grid((S + Br - 1) / Br, H_q, B);
        dim3 block(NUM_WARPS * WARP_SIZE);
        auto dq_fn = flash_attention_bwd_dq_wmma_kernel<128, Br, Bc, NUM_WARPS>;
        cudaFuncSetAttribute(dq_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_dq);
        dq_fn<<<grid, block, smem_dq, stream>>>(
            Q, K, V, dO, lse, D_buf, dQ, B, S, H_q, H_kv, scale
        );
    }

    // Step 3: dK, dV WMMA (KV-outer loop, 4 warps)
    {
        constexpr int Bkv       = 16;
        constexpr int Bq        = 64;
        constexpr int NUM_WARPS = 4;
        constexpr int PAD_L     = 8;
        constexpr int KV_STRIDE_L = 128 + PAD_L;
        constexpr int OCP = 128 / NUM_WARPS;

        const int smem_dkdv =
              2 * Bkv * KV_STRIDE_L * sizeof(half)         // K, V
            + 2 * Bq  * KV_STRIDE_L * sizeof(half)         // Q, dO
            + Bkv * Bq * sizeof(float)                      // S_smem
            + Bkv * Bq * sizeof(half)                       // P_smem
            + 2 * Bkv * 128 * sizeof(float)                 // dV_smem, dK_smem
            + NUM_WARPS * Bkv * OCP * sizeof(float)         // warp_tmp
            + Bq * 2 * sizeof(float);                       // lse + D scalars

        dim3 grid((S + Bkv - 1) / Bkv, H_kv, B);
        dim3 block(NUM_WARPS * WARP_SIZE);
        auto dkdv_fn = flash_attention_bwd_dkdv_wmma_kernel<128, Bkv, Bq, NUM_WARPS>;
        cudaFuncSetAttribute(dkdv_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_dkdv);
        dkdv_fn<<<grid, block, smem_dkdv, stream>>>(
            Q, K, V, dO, lse, D_buf, dK, dV, B, S, H_q, H_kv, scale
        );
    }
}
