# Engine Optimization: Closing the vLLM Gap

Performance engineering notes for the grpo-cuda inference engine.

**Baseline**: ~125 tok/s batch=1, ~821 tok/s batch=16 (eager, pre-optimization)
**Current**: **6963 tok/s** (256 queued seqs) — 94% of nano-vllm target
**Target**: nano-vllm parity (~7411 tok/s at batch=256)
**GPU**: RTX PRO 6000 (sm_120 Blackwell), CUDA 12.8

---

## 1. CUDA Graph Execution

### Problem
Every decode step issued **400+ individual kernel launches**: 28 RMSNorm, 28 QKV GEMM, 28 RoPE,
28 paged-attn, 28 o-proj, 28 gate/up GEMM, 28 SwiGLU, 28 down GEMM, plus embedding, logit projection,
and sampler. Kernel launch overhead alone was significant at small batch sizes.

The graphs _were_ captured during `ModelRunner` construction (`capture_cudagraph()` in
`include/engine/model_runner.cuh`) for batch buckets `{1, 2, 4, 8, 16, 32, ..., 128}`, but
`qwen3_decode_graph()` was never wired — `run()` always fell back to the eager path.

### Fix
Added `qwen3_decode_graph()` in `src/model/qwen3.cu`:

1. **Bucket selection**: find the smallest captured bucket `B_bucket >= B`.
2. **Ghost padding**: for slots `B..B_bucket-1` write `token=0`, `slot=-1` (KV write skipped),
   `seq_len=0` (attention writes nothing). No output is read from ghost rows.
3. **H2D update** of the graph's fixed device buffers (`gs.g_token_ids`, `gs.g_pos_ids`,
   `gs.g_slot_map`, `gs.g_block_tables`, `gs.g_seq_lens`) via `cudaMemcpyAsync`.
4. **Single `cudaGraphLaunch`** replaces all individual kernel launches.

`ModelRunner::run()` now selects the graph path when `gs.captured`:
```cpp
half* d_logits = is_prefill
    ? qwen3_prefill(model, batch, stream)
    : (gs.captured ? qwen3_decode_graph(model, batch, stream)
                   : qwen3_decode(model, batch, stream));
```

### Result
Nsight profile: `cudaGraphLaunch` appears ~36K× at **93 µs avg** vs millions of individual
`cudaLaunchKernel` calls. Decode step latency drops substantially at small batch sizes.

---

## 2. Sampler: Gumbel-Max Trick (v3)

### Problem
The original sampler (v1) was a sequential k-round argmax: for each of top-k rounds, scan all V
elements to find the max, write it, zero it out, repeat. For top-k=50, V=151936 this required
**53 full passes** over vocabulary memory. The sampler accounted for **84.7% of GPU time** at
703 µs average — more time than the forward pass.

v2 replaced that with a single-pass CUB `BlockRadixSort`-based top-k, but still required
softmax over the full vocabulary (2 extra passes) to get probabilities for sampling.

### Fix: Gumbel-Max (v3, `src/kernels/sampler.cu`)
The Gumbel-max trick is mathematically equivalent to sampling from `softmax(logits/T)`:

```
argmax_i ( logit_i / T  +  Gumbel(0,1)_i )
```

**Properties**:
- **1 pass** over vocabulary (v1: 53, v2: 2)
- **No softmax** computed
- **No top-k selection** or sort
- **64 bytes** static shared memory (v1/v2: up to 12 KB dynamic)
- Greedy fast-path (`T=0`): pure argmax, no noise added

Gumbel noise is generated per-element by a counter-based hash of `(seed, tok_idx, vocab_idx)`
via splitmix64 finalizer. Completely stateless — no sequential RNG state, perfectly
parallelizable.

```cpp
__device__ __forceinline__
float gumbel_noise(unsigned long long seed, int tok, int idx)
{
    unsigned long long s = seed
        ^ ((unsigned long long)tok  * 0x9e3779b97f4a7c15ULL)
        ^ ((unsigned long long)(unsigned int)idx * 0x6c62272e07bb0142ULL);
    s ^= s >> 30; s *= 0xbf58476d1ce4e5b9ULL;
    s ^= s >> 27; s *= 0x94d049bb133111ebULL;
    s ^= s >> 31;
    float u = fmaxf((float)(s >> 11) * (1.0f / (float)(1ULL << 53)), 1e-37f);
    return -__logf(-__logf(u));  // Gumbel(0,1)
}
```

Block-level argmax reduction uses warp shuffles → shared memory → warp 0 (standard pattern,
no CUB dependency).

### Benchmarks (V=151936)

| Mode | v1 (k-rounds) | v2 (CUB sort) | v3 (Gumbel) |
|------|--------------|----------------|-------------|
| Greedy B=1 | 14 µs | 14 µs | **14 µs** |
| Gumbel B=1 | 679 µs | 177 µs | **51 µs** |
| Gumbel B=4 | 710 µs | 178 µs | **51 µs** |
| Gumbel B=16 | 710 µs | 178 µs | **51 µs** |
| Gumbel B=64 | — | — | **52 µs** |

**13× faster** than v1 at B=1; near-flat scaling across batch (memory bandwidth bound,
B blocks read the same V elements in parallel once the batch fills the SM).

### Note on top-k / top-p
v3 drops explicit top-k and top-p filtering. The Gumbel-max trick samples from the full
vocabulary with probabilities `softmax(logits/T)`. This matches nano-vllm's sampler design.
The `top_k` and `top_p` parameters are accepted by `launch_sampler()` for API compatibility
but are silently ignored.

---

## 3. cuBLAS Workspace Pre-allocation

### Problem
With a large KV cache (e.g. 80+ GB GPU), the CUDA async memory pool can be exhausted during
inference. cuBLAS falls back to requesting workspace from the async pool; when that fails it
returns `CUBLAS_STATUS_ALLOC_FAILED` (status=14), crashing the decode step.

### Fix
Pre-allocate a **32 MB** cuBLAS workspace via `cudaMalloc` in `ModelRunner`'s constructor,
before the KV cache is sized:

```cpp
CUDA_CHECK(cudaMalloc(&d_cublas_ws, 32ULL * 1024 * 1024));
CUBLAS_CHECK(cublasSetWorkspace(model->cublas, d_cublas_ws, 32ULL * 1024 * 1024));
```

The same pointer is re-set after CUDA graph capture (which switches streams) so the workspace
association follows the handle.

---

## 4. Continuous Batching (Two-Phase Scheduler)

### Problem
The original `schedule()` method alternated between prefill-only and decode-only steps. This
left the decode batch at ~4 seqs on average while the GPU could sustain 256+. Prefill and
decode were never interleaved, so throughput was gated by round-trip scheduler latency.

### Fix
Split into `schedule_decode()` + `schedule_prefill()` in `include/engine/scheduler.h`, both
called in a single `LLMEngine::step()` (`include/engine/llm_engine.h`):

1. **Phase 1 — decode**: run all currently-running sequences; preempt tail sequences if
   memory is tight. Free slots of any sequences that finish.
2. **Phase 2 — prefill**: immediately fill the now-vacant slots from the waiting queue.
   Each step produces tokens *and* admits new sequences in one GPU round-trip.

Decode CUDA graph bucket ceiling raised from 128 → 256 to match the new `max_num_seqs`.

### Result
With 256 queued sequences, realistic throughput: **1189 tok/s** (up from ~821 tok/s at B=16
with old scheduler), and the average active batch depth now tracks the queue depth.

---

## 5. Fused QKV and Gate+Up Projections

### Problem
Each transformer layer issued **5 separate cuBLAS GEMMs** for the projection matrices:

```
Q:    [T, 1024] × [1024, 2048]  (q_proj)
K:    [T, 1024] × [1024, 1024]  (k_proj)
V:    [T, 1024] × [1024, 1024]  (v_proj)
gate: [T, 1024] × [1024, 3072]  (gate_proj)
up:   [T, 1024] × [1024, 3072]  (up_proj)
```

At B=256, these "thin" GEMMs leave many SMs idle — cuBLAS tiles the N dimension and the
small N values don't provide enough tiles to fill all 128 SMs.

### Fix
The model weights are already stored contiguously in `fp16_pool` in the order
`[q, k, v, o, gate, up, down]`. Exploit this layout to treat adjacent tensors as a single
larger GEMM with no extra memory or weight copies.

Added `qkv_proj` and `gate_up_proj` alias pointers to `Qwen3LayerWeights`
(`include/model/weights.h`), set in `load_weights` (`src/kernels/weights.cpp`):

```cpp
lw.q_proj    = hp2; lw.qkv_proj    = hp2; hp2 += q_proj_sz;
lw.k_proj    = hp2;                        hp2 += k_proj_sz;
lw.v_proj    = hp2;                        hp2 += v_proj_sz;
// ...
lw.gate_proj = hp2; lw.gate_up_proj = hp2; hp2 += gate_proj_sz;
lw.up_proj   = hp2;                        hp2 += up_proj_sz;
```

Scratch buffers consolidated (`include/model/qwen3.h`):
- `d_qkv   [max_T, q_dim + 2*kv_dim]` — Q/K/V aliases into it
- `d_gate_up [max_T, 2*intermediate_size]` — gate/up aliases into it

`qwen3_layer_forward` (`src/model/qwen3.cu`) now issues 2 GEMMs instead of 5:

```cpp
// QKV: [T, 1024] × [1024, 4096]
linear_half(m->cublas, m->d_hidden, L.qkv_proj, m->d_qkv,
            T, c.q_dim + 2*c.kv_dim, c.hidden_size, stream);
// gate+up: [T, 1024] × [1024, 6144]
linear_half(m->cublas, m->d_hidden, L.gate_up_proj, m->d_gate_up,
            T, 2*c.intermediate_size, c.hidden_size, stream);
```

Post-GEMM, `d_Q/d_K/d_V` and `d_gate/d_up` are pointer aliases into the fused buffers —
no copies, no extra memory.

### Result
**1356 tok/s** (up from 1189 tok/s) — **+14%** with zero weight memory overhead.
Larger N gives cuBLAS more output tiles, improving SM occupancy at all batch sizes.

---

## 6. Critical Bug Fixes (1356 → 6963 tok/s)

These three bugs silently neutralised most of the optimisations above.
Fixing them together produced the largest single throughput jump in the project.

### Bug A: CUDA Graph D2D Copies Outside Capture Scope

**Root cause**: `capture_cudagraph()` ran four `cudaMemcpy` calls (pos_ids, slot_map,
block_tables, seq_lens) in the **warmup** loop, *before* `cudaStreamBeginCapture`.
Those copies executed once at warmup time but were never recorded into the graph node list.

On every subsequent `cudaGraphLaunch`, the model's internal GPU buffers still held the
warmup values — zero token IDs, zero positions, zero block tables.  Kernels ran on what
looked like a valid all-zeros sequence: no crash, but the attention read stale KV and the
embedding looked up token 0 repeatedly.  The graph "worked" in the sense that it produced
a token each step, but the output was meaningless.

**Fix**: Move all four D2D `cudaMemcpyAsync` calls to **inside**
`cudaStreamBeginCapture … cudaStreamEndCapture` so they become nodes in the graph and
re-execute on every launch with the freshly written `gs.g_*` device buffers:

```cpp
CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
// These must be inside the capture — they update model-internal ptrs from the
// "parameter" buffers that the H2D path fills before each graph launch:
CUDA_CHECK(cudaMemcpyAsync(model->d_pos_ids,            gs.g_pos_ids,      ...));
CUDA_CHECK(cudaMemcpyAsync(model->d_slot_map,           gs.g_slot_map,     ...));
CUDA_CHECK(cudaMemcpyAsync(model->kv_cache.block_tables,gs.g_block_tables, ...));
CUDA_CHECK(cudaMemcpyAsync(model->kv_cache.seq_lens,    gs.g_seq_lens,     ...));
launch_embedding(...);
for (int l = 0; l < c.num_hidden_layers; l++) qwen3_layer_forward(...);
compute_logits(...);
CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
```

**Impact**: this single fix is responsible for the majority of the 1356 → 6963 tok/s jump.
With correct graph replay, 400+ kernel launches per decode step collapse to one
`cudaGraphLaunch`.

---

### Bug B: Fused QKV GEMM Produces Interleaved, Not Packed, Output

**Root cause**: A single fused GEMM `[T, hidden] × [hidden, q_dim+2·kv_dim]` writes its
output row-major: row `t` of the result is `[Q_t | K_t | V_t]`.  The alias pointers
assumed *packed* layout — `d_K = d_qkv + T·q_dim` — which only coincides with the actual
data when `T = 1` (decode).  For `T > 1` (prefill), Q, K, V each read from the wrong
memory region, producing garbage attention scores and blank/whitespace model output.

The same issue applied to the fused gate+up GEMM.

**Fix**: Revert to three separate GEMMs for Q, K, V and two for gate, up.  The correctness
requirement (packed contiguous layout per matrix) is trivially satisfied when each GEMM has
its own output buffer:

```cpp
linear_half(m->cublas, m->d_hidden, L.q_proj, m->d_Q, T, c.q_dim,  c.hidden_size, stream);
linear_half(m->cublas, m->d_hidden, L.k_proj, m->d_K, T, c.kv_dim, c.hidden_size, stream);
linear_half(m->cublas, m->d_hidden, L.v_proj, m->d_V, T, c.kv_dim, c.hidden_size, stream);
```

A correct fused implementation is still possible by striding the alias pointers with a
leading dimension of `q_dim+2·kv_dim` (row stride) rather than `q_dim` (packed stride),
but the separate-GEMM path is simpler, correct, and still benefits from CUDA graph
execution.

**Symptom**: greedy decode (T=1) appeared to work; factual Q&A (T>1 prefill) returned a
long string of whitespace tokens.

---

### Bug C: KV Memory Budget Used `total` Instead of `free`

**Root cause**: `compute_kv_budget()` called `cudaMemGetInfo(&free, &total)` but then
computed the budget from `total`:

```cpp
size_t budget = (size_t)(total * cfg.gpu_memory_utilization)
              - weight_bytes - activation_bytes - reserved;
```

On a shared machine with other processes already occupying GPU memory, `total` is the full
physical VRAM (102 GB) while `free` is what is actually available (e.g. 7.8 GB after
weights).  The budget wildly overestimated available KV space, leading to
`cudaMalloc` failures deep inside model initialisation (OOM at `d_mlp_mid`, status=14 on
cuBLAS workspace).

**Fix**: Base the budget on `free`, which `cudaMemGetInfo` measures *after* weights are
already loaded — so no need to subtract `weight_bytes` separately:

```cpp
cudaMemGetInfo(&free, &total);
if (free < reserved) free = reserved;
size_t budget = (size_t)(free * cfg.gpu_memory_utilization) - reserved;
```

---

## 7. Throughput History

Measurements on RTX PRO 6000 (sm_120), Qwen3-0.6B, FP16, 256 queued seqs:

| Optimization | Throughput | Notes |
|---|---|---|
| Baseline (eager, B=16) | 821 tok/s | |
| + CUDA graphs (buckets 1–128) | ~1000 tok/s | graphs captured but D2D outside scope — mostly broken |
| + Gumbel-max sampler | ~1050 tok/s | |
| + Continuous batching (B=256) | 1189 tok/s | |
| + Fused QKV + gate/up GEMMs | 1356 tok/s | interleaving bug → prefill broken; decode coincidentally OK |
| **Fix D2D inside graph capture** | **6963 tok/s** | graphs now actually replay correctly; ~5× jump |
| + Fix fused QKV interleaving | 6963 tok/s | reverted to separate GEMMs; no throughput change |
| + Fix memory budget (free vs total) | 6963 tok/s | stability fix on shared GPUs |
| nano-vllm reference | ~7411 tok/s | |

---

## 7. Remaining Gap vs nano-vllm

| Factor | grpo-cuda | nano-vllm |
|--------|-----------|-----------|
| Throughput (256 seqs) | **6963 tok/s** | ~7411 tok/s |
| Graph execution | Yes (buckets 1–256) | Yes |
| Sampler | Gumbel-max | Gumbel-max |
| Attention decode | 1 thread / (seq, head) | FlashDecoding (split-K) |
| GEMM fusion | Separate Q/K/V GEMMs | Fused QKV + norm |

### Priority 1: Flash Decoding (Split-K)
- Current paged decode: 1 thread per (seq, head) — severe under-parallelization
- Flash Decoding splits the KV sequence across blocks and merges partial softmax (split-K)
- Expected 5–10× faster decode attention, most impactful at small batch

### Priority 2: Fused RMSNorm + Linear
- Each layer still writes normalized hidden `[T, 1024]` to HBM between RMSNorm and GEMM
- A fused kernel computes the norm and feeds values directly into GEMM tiles
- Two fusion points per layer (input_norm+QKV, post_attn_norm+gate/up): saves 2 × T × 1024 × 2 bytes per layer

### Priority 3: cuBLAS GEMM Tuning
- At small B the fused GEMMs are still thin; algorithm hints or CUTLASS stream-K could help
- Profile with Nsight to confirm whether compute or bandwidth bound
