# Attention Kernels: Design & Optimization

Implementation notes for `src/kernels/attention.cu` in grpo-cuda.

---

## Overview

Two attention kernels cover the full inference path:

| Mode | Kernel | When | Bottleneck |
|------|--------|------|------------|
| **Prefill** | `flash_attention_prefill_kernel` | Processing prompt tokens | Compute (FLOPS) |
| **Decode** | `paged_attention_decode_kernel` | Token-by-token generation | Memory bandwidth |

---

## 1. FA2 Prefill Kernel

### Current Design (4-warp WMMA, Bc=64)

**File:** `src/kernels/attention.cu`  
**Template:** `flash_attention_prefill_kernel<HEAD_DIM=128, Br=16, Bc=64, NUM_WARPS=4>`

```
Grid:  (ceil(S/Br), H_q, B)     — one block per (q_tile, head, batch)
Block: (128)                     — 4 warps × 32 threads
```

#### Warp Role Partitioning

Each warp handles a non-overlapping tile to keep all 4 warps busy in parallel:

```
QK^T phase:  warp w → S_smem[:, w*16:(w+1)*16]   (one 16×16 WMMA tile)
PV   phase:  warp w → O_smem[:, w*32:(w+1)*32]   (two 16×16 WMMA tiles)
```

This gives 4× the useful WMMA parallelism vs the original 1-warp Bc=16 design.

#### Shared Memory Layout (~60 KB)

All arrays are in a single dynamic smem buffer (`extern __shared__ char dyn_smem[]`).
`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, 70*1024)` unlocks the >48 KB default limit.

```
Q_smem   [16][136]  half:         4352 B   (Br × (HEAD_DIM + PAD=8))
K_smem   [64][136]  half:        17408 B   (Bc × (HEAD_DIM + PAD=8))
V_smem   [64][136]  half:        17408 B
S_smem   [16][ 64]  float:        4096 B   (raw QK^T scores, all warps write in parallel)
P_smem   [16][ 64]  half:         2048 B   (softmax probabilities)
O_smem   [16][128]  float:        8192 B   (running FP32 output accumulator)
warp_tmp [4][16][32] float:       8192 B   (per-warp PV staging before O_smem add)
s_row_max/sum/alpha [16]:          192 B
Total:                           ~59.8 KB
```

PAD=8 halves per row avoids bank conflicts when loading 16-wide WMMA tiles.

#### Per-KV-Tile Loop

```
1. Load K_smem, V_smem  [Bc × HEAD_DIM]  cooperatively (128 threads, __syncthreads)
2. QK^T WMMA:   warp w → S_smem[:, w*16:(w+1)*16]        (__syncthreads)
3. Scale + causal mask (128 threads stripe over Br*Bc=1024 elements) (__syncthreads)
4. Online softmax (tx < Br threads, one row each):
     new_max = max(old_max, row); alpha = exp(old_max - new_max)
     Ps(r,c) = exp(Ss(r,c) - new_max); new_sum = old_sum*alpha + sum(P) (__syncthreads)
5. Rescale O_smem *= alpha  (128 threads)                 (__syncthreads)
6. PV WMMA:  warp w → warp_tmp[w][:][0..32]              (__syncthreads)
7. Accumulate: O_smem += warp_tmp[warp_id][r][c]          (__syncthreads)
```

7 barriers per KV tile (vs 22 in the old 1-warp Bc=16 design).

#### Performance

| Config | Time | TFLOPS |
|--------|------|--------|
| 1-warp Bc=16 (baseline) | 42104 µs | 6.53 |
| 4-warp Bc=64 (current) | 16775 µs | **16.39** |
| HF FA2 (reference) | ~3740 µs | ~73.5 |

Remaining gap: ~4.5× vs HF FA2.

---

## 2. Paged Attention Decode Kernel

**Template:** `paged_attention_decode_kernel<HEAD_DIM=128, BLOCK_SIZE=16>`

```
Grid:  (num_seqs, H_q)   — one thread per (sequence, q_head)
Block: (1)               — single thread walks the block table sequentially
```

Memory-bound (S_q=1). Single-thread-per-head design keeps the code simple and avoids
warp-divergence on irregular context lengths. Walks `block_tables[seq][logical_block]`
to find physical KV blocks and computes online softmax in registers.

**KV cache layout:** `[num_blocks, H_kv, BLOCK_SIZE, HEAD_DIM]`  
(block-major → minimizes stride on the inner `HEAD_DIM` dimension)

---

## 3. Paged KV Cache

**File:** `src/model/kv_cache.cu`, `include/model/kv_cache.cuh`

```
Pool: [num_layers, total_blocks, num_kv_heads, KV_BLOCK_SIZE=16, head_dim]
slot_mapping[token] = physical_block * KV_BLOCK_SIZE + in_block_offset  (int64)
```

`reshape_and_cache_half_kernel` appends tokens from a flat `[num_tokens, H_kv, D]` projection
into their assigned physical slots using pre-computed `slot_mapping`.

`paged_kv_cache_fork()` copies only block-table pointers (no KV data copy) — this is the
key GRPO optimization: G rollouts sharing one prompt share physical KV blocks with zero copy.

---

## 4. Optimization Roadmap

### Done

- [x] **WMMA QK^T and PV** — tensor core GEMM for both attention matmuls
- [x] **4-warp Bc=64** — 4× KV parallelism, 2.5× speedup vs 1-warp baseline
- [x] **Dynamic smem** — 60 KB smem unlocked via `cudaFuncSetAttribute`
- [x] **Bank-conflict padding** — PAD=8 halves per row

### Tier 1 — High impact

**1. Async K/V prefetch with `cuda::pipeline`**  
Currently serializes: load K/V tile → barrier → compute. With `cp.async` and a 2-stage
double buffer, the next K/V tile loads while the current tile is being computed by WMMA.
This fully hides memory latency for all but the first tile.

```
// Double-buffer pattern (2 × K_smem, 2 × V_smem):
Stage  0: prefetch KV tile 0
Loop   i: compute tile i  ||  prefetch tile i+1
Final:    compute last tile (no prefetch)
```

Smem cost: 2 × (K_smem + V_smem) = 4 × 17408 B = ~70 KB total; within the 70 KB carveout.
Expected: 1.5–2× prefill speedup.

**2. Larger Q tile (Br=32 via register O accumulation)**  
Br=16 is the WMMA M dimension minimum. Growing to Br=32 would halve the number of KV tile
loop iterations, reducing barrier and smem-load overhead. Requires accumulating O in registers
(`float o_reg[HEAD_DIM/NUM_WARPS]`) rather than shared memory to avoid smem overflow.

**3. cuBLAS algorithm tuning (cublasLt)**  
GEMMs (QKV proj, O proj, FFN) dominate non-attention time. `cublasLtMatmul` exposes tile/
pipeline algorithm selection; the default `cublasGemmEx` may not pick the best algo for our
exact shapes (hidden=1024, intermediate=3072).

### Tier 2 — Moderate impact

**4. Fused RMSNorm + linear**  
Each layer round-trips through HBM: norm writes to scratch, QKV proj reads from scratch.
A fused kernel avoids one full-size read + write per layer (×28 layers, ×2 norm+proj pairs).

**5. Fused QK-Norm + RoPE**  
Two separate kernel launches currently. Per-head RMSNorm and RoPE apply to the same Q/K
tensors sequentially — trivially fusible into one pass.

### Tier 3 — High impact, high effort

**6. Warp-specialized FA2 (producer/consumer)**  
Split warps: producer warps issue `cp.async` loads, consumer warps run WMMA. This is
the FlashAttention-3 / CUTLASS approach and would require restructuring the block entirely.

**7. FP8 GEMMs**  
sm_120 (Blackwell) has native FP8 tensor cores — 2× FP16 throughput for GEMMs.
Would need per-tensor quantization + dequantization at layer boundaries.

---

## 5. Correctness Tests

All tests pass with `max_err < 5e-3` vs CPU FP32 naive reference:

```
[PASS] Prefill B=1 S=16  H_q=2  H_kv=1   max_err=0.000152
[PASS] Prefill B=1 S=64  H_q=4  H_kv=2   max_err=0.000189
[PASS] Prefill B=1 S=128 H_q=16 H_kv=8   max_err=0.000185
[PASS] Prefill B=2 S=96  H_q=4  H_kv=2   max_err=0.000159
[PASS] Prefill B=1 S=256 H_q=16 H_kv=8   max_err=0.000164
[PASS] Decode  num_seqs=1 ctx=32          max_err=0.000058
[PASS] Decode  num_seqs=1 ctx=128         max_err=0.000029
[PASS] Decode  num_seqs=2 ctx=64          max_err=0.000031
[PASS] Decode  num_seqs=4 ctx=128         max_err=0.000030
```

Build and run: `make test_attention`
