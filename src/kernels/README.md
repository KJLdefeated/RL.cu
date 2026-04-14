# src/kernels/ — CUDA Kernel Implementations

Each `.cu` file implements one kernel (forward + backward). Headers are in `include/kernels/*.cuh`.

---

## File Map

| File | Kernel | Grid / Block | Key Technique |
|------|--------|-------------|---------------|
| `rmsnorm.cu` | RMSNorm | Grid(rows), Block(32 or 128) | Warp `__shfl_xor_sync` butterfly reduction; block-level via shared memory when >1 warp |
| `rope.cu` | Rotary Position Embedding | Grid(num_tokens, H_q), Block(D/2=64) | FP32 sin/cos precomputed table; NeoX split-half rotation; Q and K in one dispatch |
| `attention.cu` | FA2 prefill + paged decode + backward | See below | WMMA tensor cores, async double-buffered K/V, parallel softmax |
| `swiglu.cu` | SwiGLU activation | Grid(ceil(N/256)), Block(256) | Fused `silu(gate) * up` in one pass; 1 thread per element |
| `embedding.cu` | Token embedding | Grid(num_tokens), Block(256) | Gather: `out[t] = weight[ids[t]]`; stride loop over hidden_size |
| `linear.cu` | Dense projection | cuBLAS managed | `cublasGemmEx` with FP16 I/O, FP32 accumulation, tensor core ops |
| `softmax.cu` | Softmax | Grid(rows), Block(warp) | Numerically stable: subtract row max before exp |
| `sampler.cu` | Gumbel-max sampler | Grid(num_tokens), Block(256) | Single-pass argmax over perturbed logits; no softmax or top-k needed |
| `adamw.cu` | Fused AdamW | Grid(ceil(N/256)), Block(256) | FP16 grad -> FP32 master weight update -> FP16 write-back in one kernel |
| `fused_norm_linear.cu` | Fused RMSNorm + Linear | Grid(ceil(N/64), ceil(T/16)), Block(128) | WMMA matmul; normalized hidden states never written to HBM |
| `config.cpp` | Config parser | CPU | `load_config()` from Qwen3 `config.json` via nlohmann/json |
| `weights.cpp` | Weight loader | CPU | Safetensors `mmap` loader; BF16 -> FP16 (projections) / FP32 (norms) |

---

## Optimization Details

### Attention (attention.cu)

The largest and most optimized kernel. Three sub-kernels:

**FA2 Prefill** (`flash_attention_prefill_kernel<128, 16, 64, 4>`):
- **4-warp WMMA**: each warp owns a non-overlapping column slice of QK^T scores and output accumulation. 8 `mma.sync` per warp per KV tile for QK^T; 8 more for PV multiply.
- **cp.async double buffering**: two K/V shared memory buffers alternate with `cp.async.ca.shared.global` DMA. While warps compute tile `i`, hardware fills tile `i+1`. Hides ~400-cycle HBM latency. (+43% TFLOPS)
- **Parallel softmax**: all 128 threads participate (8 threads per row, 8 cols each). 3-step warp butterfly reduction within each 8-thread row group. Previous: only 16/128 threads active. (+1.9x speedup)
- **Shared memory**: ~94 KB per block (Q + K[2] + V[2] + S + P + O + warp_tmp). PAD=8 halves per row avoids bank conflicts on WMMA tile loads.
- **Causal mask**: `break` on kv_tile when all columns are masked.

**FA2 Backward** (`flash_attention_bwd_dq_wmma_kernel`, `flash_attention_bwd_dkdv_wmma_kernel`):
- Same 4-warp WMMA structure as forward. Replaces O(S^2) warp reductions with tensor-core matmuls.
- Shared memory reuse: S_smem is dual-purpose (QK^T scores, then dp). P_smem is dual-purpose (attention weights, then ds).
- Uses precomputed LSE from forward — no online max/sum tracking in backward.
- 4x faster than scalar backward at S=2048.

**Paged Decode** (`paged_attention_decode_warp_kernel<128, 16>`):
- 1 warp per (seq, head). 32 threads cooperatively load K/V via coalesced `int2` (8-byte) vectorized transactions.
- Online softmax with warp butterfly reduction (5 `__shfl_xor_sync` ops per KV token).
- Walks block_table for paged KV cache lookup.

### Sampler (sampler.cu) — Gumbel-Max Trick

Instead of softmax + top-k + CDF sampling (53 vocab passes), computes:

```
argmax_i ( logit_i / T + Gumbel(0,1)_i )
```

- **1 pass** over vocabulary (vs 53 in v1, 2 in v2)
- **No softmax, no top-k sort** — mathematically equivalent to `sample(softmax(logits/T))`
- Counter-based noise via splitmix64 hash of `(seed, tok_idx, vocab_idx)` — fully stateless, parallelizable
- Block-level argmax reduction with warp shuffles -> shared memory -> warp 0
- Greedy fast-path (`T=0`): pure argmax, no noise
- 13x faster than v1 at B=1 (51 us vs 679 us for V=151936)

### Fused RMSNorm + Linear (fused_norm_linear.cu)

Eliminates the HBM round-trip between RMSNorm and the following GEMM:

- Input `x[T, H=1024]` loaded once into shared memory (32 KB)
- RMS inverse computed per-row in shared memory
- Normalized values fed directly into WMMA tiles — never written to global memory
- Tiling: BM=16, BN=64, BK=64 with 4-warp WMMA (one warp per 16-col output tile)
- Saves `2 * T * 1024 * 2 bytes` per fusion point (2 fusion points per layer)

### AdamW (adamw.cu)

Two fused kernels — FP16 params (projections + embed) and FP32 params (norms):

- Reads FP16 gradient, updates FP32 master weight and momentum/variance in-place, writes back FP16 model weight — all in one kernel launch
- Decoupled weight decay: `w *= (1 - lr * wd)` applied to master weight before Adam step
- Bias correction: precomputed `1/(1-beta^t)` on host, passed as kernel args

### Linear (linear.cu) — CUTLASS/cuBLAS

- Forward: `cublasGemmEx` with FP16 I/O, FP32 accumulation, tensor core operations
- Column-major trick: computes `C^T = B * A^T` to avoid explicit transpose
- Backward: `dX = dY @ W` and `dW += dY^T @ X` via cuBLAS (dW accumulated with `beta=1`)
- Pre-allocated 32 MB cuBLAS workspace avoids async pool exhaustion under memory pressure
