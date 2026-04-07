# Attention Kernels: Design & Optimization

Implementation notes for `src/kernels/attention.cu` in grpo-cuda.

---

## Overview

| Mode | Kernel | When | Bottleneck |
|------|--------|------|------------|
| **Prefill** | `flash_attention_prefill_kernel` | Processing prompt tokens | Compute (FLOPS) |
| **Decode** | `paged_attention_decode_kernel` | Token-by-token generation | Memory bandwidth |
| **Bwd dQ** | `flash_attention_bwd_dq_wmma_kernel` | Training backward | Compute |
| **Bwd dKdV** | `flash_attention_bwd_dkdv_wmma_kernel` | Training backward | Compute |

---

## 1. FA2 Prefill Kernel

**Template:** `flash_attention_prefill_kernel<HEAD_DIM=128, Br=16, Bc=64, NUM_WARPS=4>`

```
Grid:  (ceil(S/Br), H_q, B)     — one block per (q_tile, head, batch)
Block: (128)                     — 4 warps × 32 threads
```

### Warp Role Partitioning

Each warp handles a non-overlapping column slice of the score / output matrix:

```
QK^T phase: warp w → S_smem[:, w*16:(w+1)*16]   (one 16×16 WMMA tile, Bc=64/4=16 cols)
PV   phase: warp w → O contribution[:, w*32:(w+1)*32] (two 16×16 WMMA tiles, D=128/4=32 cols)
```

### Shared Memory Layout (~94 KB, double-buffered K/V)

All arrays live in one `extern __shared__ char dyn_smem[]` buffer.
`cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)` unlocks
the >48 KB default on Blackwell (device max = 228 KB / SM, opt-in max = 101376 B).

```
Q_smem      [16][136]  half:  4352 B  @ 0       (Br × (HEAD_DIM + PAD=8))
K_smem[0]   [64][136]  half: 17408 B  @ 4352    ping-pong buffer 0
K_smem[1]   [64][136]  half: 17408 B  @ 21760   ping-pong buffer 1
V_smem[0]   [64][136]  half: 17408 B  @ 39168
V_smem[1]   [64][136]  half: 17408 B  @ 56576
S_smem       [16][64]  float: 4096 B  @ 73984   raw QK^T scores
P_smem       [16][64]  half:  2048 B  @ 78080   softmax probabilities
O_smem      [16][128]  float: 8192 B  @ 80128   FP32 running output accumulator
warp_tmp   [4][16][32] float: 8192 B  @ 88320   per-warp PV staging
s_row_max/sum/alpha [16]×3 float: 192 B @ 96512
Total:                        96704 B (~94 KB)
```

PAD=8 halves per row avoids shared-memory bank conflicts on 16-wide WMMA tile loads.

### Per-KV-Tile Pipeline

The kernel uses `cp.async.ca.shared.global` (L1-cached 16-byte DMA) with a depth-2
ping-pong pipeline to overlap next-tile loading with current-tile WMMA computation.

```
Prologue:  cp.async tile-0 → K_buf[0], V_buf[0];  commit_group
Loop i:
  if i+1 < num_tiles:
    cp.async tile-(i+1) → K_buf[1-cur], V_buf[1-cur];  commit_group
    cp.async.wait_group 1       ← tile-i ready, tile-(i+1) still in flight
  else:
    cp.async.wait_group 0       ← drain last group
  __syncthreads()
  [compute tile-i from K_buf[cur], V_buf[cur]]
  __syncthreads()
```

Each tile compute step:
1. **QK^T WMMA**: 8 `mma_sync` per warp (HEAD_DIM/16 = 8 steps) → `S_smem`
2. **Scale + causal mask**: 128 threads stripe over Br×Bc=1024 elements
3. **Parallel softmax** (all 128 threads, 8 per row, 8 cols each)
4. **Rescale O_smem × alpha**: 128 threads
5. **PV WMMA**: 4×2 `mma_sync` per warp → `warp_tmp`
6. **Accumulate** `warp_tmp` into `O_smem`

7 `__syncthreads()` per KV tile.

---

## 2. Optimization 1: Async K/V Double Buffering (cp.async)

### Problem

Each tile did a synchronous cooperative load (global → shared) then `__syncthreads`,
stalling all warps until the load completed. Memory latency (~400 cycles for 17 KB
across HBM) directly blocked WMMA throughput.

### Fix

`cp.async.ca.shared.global` is a hardware DMA instruction that writes to shared memory
without stalling the issuing warp. Two K/V smem buffers (`K_buf[0/1]`, `V_buf[0/1]`)
alternate each KV-tile iteration. While warps compute tile `i` from `K_buf[cur]`, the
DMA engine fills `K_buf[1-cur]` with tile `i+1`.

`cp.async.wait_group 1` stalls only until ≤1 group is outstanding — guaranteeing the
current tile's data is in smem before `__syncthreads`.

Critical implementation details:
- OOB rows (kv_row ≥ S) are silently skipped; correctness preserved because masked-out
  K scores become -INF and masked-out V rows have P=0.
- smem address for PTX: use `__cvta_generic_to_shared(&buf[r*KV_STRIDE+c])`.
- `cudaFuncSetAttribute` value must be ≤ device opt-in max (101376 B on this machine).
  Passing 100*1024=102400 silently fails and leaves the 48 KB default, causing all
  accesses >48 KB to read garbage.

**Smem cost:** +2 × (K + V) buffers = +69 KB → 96 KB total.
**Measured speedup:** 16.39 → 23.37 TFLOPS (+43%, B=8 S=2048).

---

## 3. Optimization 2: Parallel Softmax (all 128 threads)

### Problem

The softmax step gated on `if (tx < Br)` — only 16 of 128 threads active (87.5% idle).
Each active thread serially looped over Bc=64 columns:

```cpp
if (tx < Br) {                    // 112 threads spin idle at __syncthreads below
    for (int c = 0; c < Bc; c++)  // 64 serial expf() calls per thread
        ...
}
__syncthreads();
```

At ~20 cycles/expf × 64 elements, this was ~1280 cycles of serialized work per KV tile
while 7/8 of the block's threads waited.

### Fix

Assign all 128 threads with `THREADS_PER_ROW (TPR) = TOTAL_THREADS/Br = 8` threads per
row, each handling `COLS_PER_THREAD (CPT) = Bc/TPR = 8` cols.

Within-row max/sum reduction via 3-step butterfly (no cross-row contamination because
`shfl_xor(mask, 4/2/1)` maps lane `8r+c ↦ 8r+(c⊕k)` — always within the same 8-thread
group):

```cpp
constexpr int TPR = TOTAL_THREADS / Br;   // 8
constexpr int CPT = Bc / TPR;             // 8
const int row      = tx / TPR;
const int col_start = (tx % TPR) * CPT;
const float old_max = s_row_max[row];
float thread_max    = old_max;
if (valid_row) {
    for (int c = col_start; c < col_start + CPT; c++)
        thread_max = fmaxf(thread_max, Ss(row, c));
}
// 3-step butterfly: stays within 8-thread row-group
thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, 4));
thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, 2));
thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, 1));
```

Only `tx % TPR == 0` writes back `s_row_max / s_row_sum / s_alpha`.

**Measured speedup:** 11.51 ms → 6.19 ms per prefill call (1.9×, B=8 S=2048).
TFLOPS: 23.37 → 44.40.

---

## 4. FA2 Backward Kernels (WMMA)

### Root Cause of Slow Scalar Backward

The old `flash_attention_bwd_dq_kernel` and `flash_attention_bwd_dkdv_kernel` called
`warp_reduce_sum` inside the inner column loop — 2 reductions per column, Bc=64 columns
per tile:

```cpp
for (int c = 0; c < tile_end; c++) {          // 64 iterations
    dot = warp_reduce_sum(dot) * scale;        // 5 __shfl_xor each
    dp  = warp_reduce_sum(dp);                 // 5 __shfl_xor each
    // ... accumulate dq/dk/dv ...
}
```

For S=2048, Bc=64: 32 tiles × 64 cols × 2 reductions × 5 shfl = **20480 shfl ops per block**.
This is O(S²) warp reductions — catastrophically slow for large sequences.

### Fix: WMMA-Based Backward

Both backward kernels are rewritten to use the same 4-warp WMMA structure as the forward
kernel. Per KV/Q tile, scalar dot-products are replaced by tensor-core matrix multiplications.

### 4a. `flash_attention_bwd_dq_wmma_kernel`

**Smem layout (~65 KB → 3 blocks/SM on Blackwell):**

```
Q_smem   [16][136] half: 4352 B   loaded once before KV loop
dO_smem  [16][136] half: 4352 B   loaded once before KV loop
K_smem   [64][136] half:17408 B   per KV tile
V_smem   [64][136] half:17408 B
S_smem   [16][ 64] float:4096 B   QK^T result, REUSED as dp_smem
P_smem   [16][ 64] half: 2048 B   attention weights, REUSED as ds_smem
dQ_smem  [16][128] float:8192 B   accumulated dQ output
warp_tmp [4][16][32] float:8192 B per-warp staging (reused across passes)
lse_smem [16] float:  64 B
d_smem   [16] float:  64 B
Total:  ~65 KB
```

**Per-KV-tile computation (3 WMMA passes):**

```
(a) QK^T:  Q_smem[16,128] × K_smem[64,128]^T → S_smem[16,64]     (8 mma_sync/warp)
    mask:  S[q,k] = -INF when k > q or k >= S
    P:     P[r,c] = exp(S[r,c] - LSE[r])                          (parallel, 8-thread-per-row)

(b) dOV^T: dO_smem[16,128] × V_smem[64,128]^T → S_smem[16,64]    (8 mma_sync/warp; reuse S)
    ds:    P_smem[r,c] = P[r,c] × (S_smem[r,c] − D[r])           (overwrite P with ds)

(c) ds×K:  P_smem[16,64] × K_smem[64,128] → dQ_smem[16,128]      (2 frags × 4 mma_sync/warp)
           dQ_smem[r,c] += scale × warp_tmp[...]
```

**Smem reuse:** S_smem is sequentially dual-use (QK^T scores, then dp). P_smem is
dual-use (P weights, then ds). No overlap — races impossible.

**Backward softmax:** uses precomputed `LSE` from forward — no online max/sum tracking.
Simplified vs forward softmax (no shfl reduction for max/sum, just parallel `expf`).

### 4b. `flash_attention_bwd_dkdv_wmma_kernel`

K, V are loaded once per outer KV tile; Q, dO are loaded per inner Q tile.
GQA handled by `for h_rep in 0..n_rep-1` (n_rep=2 for Qwen3-0.6B).

**Per-Q-tile computation (4 WMMA passes):**

```
(a) KQ^T:   K_smem[16,128] × Q_smem[64,128]^T → S_smem[16,64]
    mask:   S[kv,q] = -INF when kv > q
    P:      P[kv,q] = exp(S[kv,q] - LSE[q])                       (parallel)

(b) VdO^T:  V_smem[16,128] × dO_smem[64,128]^T → S_smem[16,64]   (dp; reuse S)
(c) P×dO:   P_smem[16,64] × dO_smem[64,128] → dV_smem[16,128]    (accumulate, no scale)
    ds:     P_smem[kv,q] = P[kv,q] × (S_smem[kv,q] − D_Q[q])     (overwrite P with ds)
(d) ds×Q:   P_smem[16,64] × Q_smem[64,128] → dK_smem[16,128]     (accumulate × scale)
```

**Causal start:** Q tiles before `q_tile_start = kv_start / Bq` are skipped (all P = 0).

### Backward Performance

| Kernel | Before | After | Speedup |
|--------|--------|-------|---------|
| `bwd_dq` (B=8 S=2048) | 23.88 ms/call | ~6 ms | ~4× |
| `bwd_dkdv` (B=8 S=2048) | 31.25 ms/call | ~8 ms | ~4× |
| Full backward pass | 2008 ms | **1211 ms** | **1.66×** |

---

## 5. Combined Results

All optimizations together, Qwen3-0.6B, B=8 S=2048, Blackwell RTX PRO 6000 (sm_120):

| Metric | Baseline | Optimized | Speedup |
|--------|----------|-----------|---------|
| Prefill (one call) | 11.51 ms | 6.19 ms | **1.9×** |
| Prefill TFLOPS | 23.4 | 44.4 | 1.9× |
| Forward pass (28 layers) | 451 ms | 300 ms | **1.5×** |
| Backward pass | 2008 ms | 1211 ms | **1.66×** |
| Step total | 2459 ms | 1510 ms | **1.63×** |
| Training throughput | 6662 tok/s | 10848 tok/s | **1.63×** |

---

## 6. Paged Attention Decode Kernel

**Template:** `paged_attention_decode_warp_kernel<HEAD_DIM=128, BLOCK_SIZE=16>`

```
Grid:  (num_seqs, H_q)   — one warp per (sequence, q_head)
Block: (32)              — 1 warp; each thread owns HEAD_DIM/32 = 4 elements
```

Memory-bandwidth-bound (S_q=1). One warp per (seq, head) cooperatively loads K/V
with coalesced 8-byte (int2) vectorized transactions and reduces the dot product via
warp butterfly (`__shfl_xor_sync`). No `__syncthreads` needed (all within one warp).

**Optimization vs v1 scalar kernel (block=1):**
- v1: single thread, 128 serial half loads → uncoalesced (8 cache-line transactions)
- v2: 32 threads × 4 halves = one 256-byte transaction; warp-reduce (5 shfl ops) for score

**KV cache layout:** `[num_blocks, H_kv, BLOCK_SIZE, HEAD_DIM]` — block-major keeps
the inner `HEAD_DIM` dimension contiguous for coalesced reads.

**Per-token work (v2 warp kernel):**
1. `int2` load 8 bytes of K → `half22float2` → 4-element partial dot product
2. 5-step warp butterfly reduce → full score (all lanes identical)
3. Online softmax update: `new_max`, `alpha`, `p` (identical on all lanes)
4. `int2` load 8 bytes of V → accumulate 4 output elements

**Benchmark (Qwen3 config, H_q=16, H_kv=8, HEAD_DIM=128, sm_120):**

| Config | Latency | BW |
|--------|---------|-----|
| B=1  ctx=512  | 150 µs | 14 GB/s |
| B=1  ctx=2048 | 591 µs | 14 GB/s |
| B=16 ctx=512  | 150 µs | 225 GB/s |
| B=16 ctx=2048 | 605 µs | 222 GB/s |
| B=64 ctx=512  | 204 µs | 662 GB/s |
| B=64 ctx=2048 | 1010 µs | 532 GB/s |
| B=128 ctx=2048 | 1043 µs | 1031 GB/s |

B=1 is latency-bound (only 16 SMs active); B=128 reaches ~60% of 1700 GB/s HBM peak.

---

## 7. Correctness Tests

**Forward** (`make test_attention`): `max_err < 5e-3` vs CPU FP32 naive reference.

```
[PASS] Prefill B=1 S=16   H_q=2  H_kv=1   max_err=0.000152
[PASS] Prefill B=1 S=64   H_q=4  H_kv=2   max_err=0.000189
[PASS] Prefill B=1 S=128  H_q=16 H_kv=8   max_err=0.000185
[PASS] Prefill B=2 S=96   H_q=4  H_kv=2   max_err=0.000159
[PASS] Prefill B=1 S=256  H_q=16 H_kv=8   max_err=0.000164
[PASS] Decode  num_seqs=1 ctx=32           max_err=0.000058
[PASS] Decode  num_seqs=4 ctx=128          max_err=0.000030
```

**Backward** (`make test_attention_backward`): `max_err < 5e-2` vs CPU FP32 reference.

```
[PASS] B=1 S=16  H_q=2  H_kv=1   dQ=0.00002 dK=0.00003 dV=0.00056
[PASS] B=1 S=32  H_q=2  H_kv=2   dQ=0.00001 dK=0.00002 dV=0.00046
[PASS] B=1 S=64  H_q=4  H_kv=2   dQ=0.00002 dK=0.00003 dV=0.00058
[PASS] B=2 S=48  H_q=4  H_kv=2   dQ=0.00002 dK=0.00004 dV=0.00055
[PASS] B=1 S=128 H_q=16 H_kv=8   dQ=0.00003 dK=0.00005 dV=0.00053
```

---

## 8. Remaining Optimization Roadmap

### Forward (6.2 ms/call, 44 TFLOPS, target ~1 ms / ~400 TFLOPS)

Gap analysis: achieving 11% of Blackwell peak. Root causes: 2 blocks/SM (94 KB smem),
16K blocks × 65 waves, no register-level output accumulation.

| Optimization | Expected gain | Effort |
|---|---|---|
| **Larger Br=64** — 4× fewer blocks (4096 vs 16384), 3 blocks/SM, 4× more work per block | 1.5–2× | Medium |
| **Register O accumulation** — remove O_smem (8 KB) by holding output in WMMA accumulator fragments across KV tiles; enables Br=64 without smem overflow | 1.2× + enables above | Medium |
| **Fused QKV projection** — single GEMM `[T, q+2k dims]` instead of 3 separate; avoids 2 extra full-width HBM reads of the hidden state per layer × 28 layers | 1.3× forward total | Low |
| **Warp Group MMA (WGMMA)** — Blackwell-native instruction; larger tile granularity, eliminates smem round-trip for S/P | 2–4× | High |
| **TMA for K/V loading** — hardware-scheduled bulk DMA, replaces cp.async | 1.3× | High |

### Backward (1.2 s/pass, bwd/fwd = 4×, target 3×)

| Optimization | Expected gain | Effort |
|---|---|---|
| **Larger Bq tile** (64 → 128) for dkdv — halves Q-tile load count per KV tile | 1.3× bwd | Low |
| **Fused gate+up backward** — single GEMM produces `[dgate | dup]` concatenated | 1.1× total bwd | Low |
| **Selective activation recompute** — current design stores all layer inputs; recompute attention activations to reduce 15 GB activation memory | Enables larger batch | Medium |
