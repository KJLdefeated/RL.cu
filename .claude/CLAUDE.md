# GRPO-CUDA Project Memory

## Current Status (as of 2026-03-09)
- **Phase 1 COMPLETE (correctness)**: LLMEngine all 11/11 tests pass, Sampler kernel done
- **Performance gap**: ~30% of vLLM throughput (125 tok/s vs ~400+ batch=1; 819 tok/s vs ~2700+ batch=16)
- **Next step 1**: Efficiency optimization (close the vLLM gap — see plan below)
- **Next step 2**: Phase 2 SFT training infrastructure

## Efficiency Optimization Plan (to close vLLM gap)
Priority areas to investigate:
1. **CUDA graph execution**: Graphs are captured (buckets 1,2,4,8..128) but `qwen3_decode` still uses eager path — wire up graph replay
2. **cuBLAS tuning**: Check GEMM algorithm selection (cublasGemmEx with ALGO hints), tensor core utilization
3. **Kernel fusion**: Fuse RMSNorm+linear, avoid extra memory round-trips between layers
4. **KV cache layout**: Current paged-decode is single-thread per (seq, head) — explore warp-level parallelism
5. **Memory bandwidth**: Profile with Nsight; check if decode is memory-bound; consider FP8 KV cache
6. **Continuous batching**: Verify scheduler fills GPU efficiently; check batch padding overhead
7. **Flash attention decode**: Replace single-thread decode with a proper FlashDecoding kernel (split-K)

## Project: grpo-cuda — Pure C++/CUDA LLM RL Engine
Target model: Qwen3-0.6B (FP16, inference phase 1)

## GPU / Build
- GPU: sm_120 (Blackwell, RTX PRO 6000 / RTX 50xx)
- CUDA: 12.8 at /usr/local/cuda-12.8
- Build: `make test_attention` (or `make tests` for all)
- Arch flag: `--gpu-architecture=sm_120`

## Qwen3-0.6B key dims
- hidden=1024, layers=28, H_q=16, H_kv=8, head_dim=128
- GQA ratio: 2:1 (h_kv = h_q * H_kv / H_q)
- QK-Norm BEFORE RoPE (not after — HuggingFace order)

## Kernel status
| Kernel      | File                              | Status     |
|-------------|-----------------------------------|------------|
| RMSNorm     | src/kernels/rmsnorm.cu            | DONE, passing |
| SwiGLU      | src/kernels/swiglu.cu             | DONE, passing |
| Softmax     | src/kernels/softmax.cu            | DONE, passing |
| FA2 Prefill | src/kernels/attention.cu          | DONE, passing |
| Paged Attn  | src/kernels/attention.cu          | DONE, passing |
| KV Cache    | src/model/kv_cache.cu             | DONE, passing |
| RoPE        | src/kernels/rope.cu               | DONE, passing |
| Embedding   | src/kernels/embedding.cu          | DONE, passing |
| Sampler     | src/kernels/sampler.cu            | DONE, passing |

## Sampler design (src/kernels/sampler.cu)
- API: launch_sampler(const half* logits, float* temp_probs, num_tokens, vocab_size,
        top_k, top_p, temperature, seed, int* output_ids, stream)
  - temp_probs: FP32 workspace [num_tokens, vocab_size], pre-allocated by caller
  - MAX_SAMPLER_TOP_K = 1024 (clamped in wrapper)
- Grid: (num_tokens,); Block: 256; Dynamic smem: top_k*8 + 256*8 bytes
- temperature==0 fast path: single argmax pass, return (no softmax)
- Steps: temp-scale + softmax → k-rounds argmax (zeroing selected prob) →
  top-p nucleus (renormalize within top-k first, then cumsum walk) →
  splitmix64 hash → weighted sample
- Bench (V=151936): greedy=14 µs, top-k=1: 113 µs, top-k=50 B=1: 679 µs, B=16: 710 µs
- Test tie policy: use threshold-based top-k set (includes all tied boundary tokens)
  and value-equality for greedy (FP16 ties are common with random logits at large V)

## Attention kernel design
- FA2 prefill: template<HEAD_DIM=128, Br=16, Bc=64>
  - Grid: (ceil(S/Br), H_q, B); Block: (Br)
  - Shared memory: K_smem[Bc][HEAD_DIM] + V_smem[Bc][HEAD_DIM] = 32 KB
  - Registers: q_reg[128], o_acc[128] in FP32; causal break in inner loop
  - Online softmax: alpha = exp(row_max - new_max); p = exp(dot - new_max)
  - CRITICAL: NO early return (if q_row>=S return) inside the kv_tile loop!
    All Br threads must cooperate in K/V loading and reach __syncthreads().
    Use `const bool valid_q = (q_row < S)` and guard only the score+write.
    Bug: when last q_tile has mixed valid/invalid threads, early-returning
    threads skip loading their K_smem rows → garbage dot products → NaN for S>Br.
- Paged decode: template<HEAD_DIM=128, BLOCK_SIZE=16>
  - Grid: (num_seqs, H_q); Block: (1) — single thread per (seq, head)
  - Walks block_tables[seq][logical_block] → physical_block index
  - k_cache layout: [num_blocks, H_kv, BLOCK_SIZE, D]

## KV Cache design
- KV_BLOCK_SIZE = 16 (tokens per physical block)
- Pool layout: [num_layers, total_blocks, num_kv_heads, KV_BLOCK_SIZE, head_dim]
- slot_mapping[token] = physical_block * KV_BLOCK_SIZE + in_block_offset (int64)
- reshape_and_cache_half_kernel: Grid(num_tokens,1), Block(128,1), stride over (head,dim)
- paged_kv_cache_append_slot() builds slot on CPU; returns slot for slot_mapping
- paged_kv_cache_fork() copies block-table pointers only (no KV copy) for GRPO
- PagedKVCache struct in include/model/kv_cache.cuh

## Test tolerances (from DESIGN.md)
- RMSNorm, RoPE, SwiGLU: max_err < 1e-3
- FA2 attention: max_err < 5e-3 vs naive attention
- Actual FA2 errors: prefill ~1.2e-4, decode ~3e-5 (well within budget)

## RoPE design
- NeoX split-half: for i in [0, D/2): out[i] = x[i]*c - x[i+D/2]*s; out[i+D/2] = x[i+D/2]*c + x[i]*s
- Precompute: Grid(max_seq_len,), Block(D/2=64); table is FP32 [max_seq_len, D/2]
- Apply: Grid(num_tokens, H_q), Block(D/2); applies Q (all heads) and K (H_kv heads) in one dispatch
- KEY: use `hd2` not `half` for D/2 integer — `half` shadows the CUDA type and breaks compilation
- FP32 arithmetic for rotation (not FP16) — sin/cos table is FP32; avoiding catastrophic cancellation
- GPU single-precision cosf/sinf differs from CPU ref by ~6e-5; table check tolerance = 1e-4

## Embedding design
- gather kernel: Grid(num_tokens,1), Block(256,1); stride loop over hidden_size
- pure memory copy FP16→FP16; zero arithmetic → exact match with reference
- output projection (weight^T matmul) handled by cuBLAS, not this kernel

## Scheduler design (include/engine/scheduler.h + include/engine/block_manager.h)
- Prefill: can_allocate checks `free > num_blocks()` per seq; also tracks `prefill_blocks_reserved`
  for post-prefill may_append blocks (needed when prompt_len % block_size == 0)
- Decode: tracks `blocks_reserved` across all seqs in the batch; preempts from back of queue
  until `num_free_blocks() - blocks_reserved >= needs_block` before scheduling each seq
- `can_append`: checks `seq.size() % block_size == 0` (next token starts new block)
- `may_append`: called AFTER append_token; allocates new block when `sz % block_size == 1`
- Prefix caching: blocks sealed at block boundaries with xxhash; reused across requests with same prefix;
  freed blocks lose their hash (no stale cache hits after preemption)
- Batch slots (`slot_used[]`): freed in postprocess when seq finishes; ALSO freed on preemption
  (preempt() calls free_slot + seq->batch_slot=-1; re-allocated on re-schedule via alloc_slot())
  Engine calls model_runner->free_seq_slot() for both preempted and finished slots

## Known bugs fixed
- FA2 early-return before cooperative K/V load → NaN for S > Br=16 (see attention.cu note above)
- Paged decode: always passed k_pool start → all layers read layer 0's KV.
  Fix: pass `k_pool + layer_idx * layer_stride` where
  `layer_stride = total_blocks * H_kv * KV_BLOCK_SIZE * head_dim`
- cuBLAS workspace freed after CUDA graph capture → status=14 during execution.
  Fix: store workspace ptr in CUDAGraphState.g_cublas_ws; free only in ~ModelRunner.
- qwen3_prefill used batch_slot=-1 (never assigned) → UB on h_seq_lens[-1].
  Fix: scheduler assigns seq->batch_slot = alloc_slot() at schedule time; freed via free_slot() in postprocess.
- Variable-length prefill batch: S=T/B assumed uniform lengths; gather_last_tokens read wrong positions.
  Fix: pad sequences to S_max with slot=-1 (reshape_and_cache skips -1 slots); use gather_at_offsets_kernel
  to read each sequence's actual last real token at b*S_max + actual_len[b]-1.
- qwen3_decode used 'b' as KV slot; now uses seq->batch_slot with compact block_table/seq_lens packing.
- KV slot leak: h_seq_lens[b] not reset between sequences → new seq inherits stale KV.
  Fix: qwen3_free_seq_slot() returns blocks to free_stack, zeros h_seq_lens[b] and h_block_tables[b][*].
- Scheduler OOM assert at batch≥64: THREE bugs in block_manager/scheduler (fixed 2026-03-09):
  1. `can_append` wrong boundary: checked `sz % block_size == 1` (current tok is first of block)
     but should check `sz % block_size == 0` (NEXT tok starts new block). Caused decode preemption
     to never fire → assert in may_append.
  2. Decode scheduling: `blocks_reserved` missing — each seq's `can_append` saw same free count,
     so N boundary seqs all got scheduled but only M<N blocks available. Fix: track blocks_reserved
     across the loop, preempt until `num_free_blocks() - blocks_reserved >= needs_block`.
  3. Prefill scheduling: post-prefill may_append blocks unaccounted — batch of K seqs with
     prompt_len % 16 == 0 each needed 1 post-prefill block; `can_allocate` only ensured 1 extra
     per seq individually. Fix: track `prefill_blocks_reserved` and check
     `num_free_blocks() - num_blocks() - prefill_blocks_reserved >= needs_postprefill_block`.
  Confirmed: all 3 bugs needed for batch=64/128 to not crash. 11/11 tests pass post-fix.
  Throughput after fix: batch=64 → 3090 tok/s, batch=128 → 3696 tok/s.

## Model forward pass (src/model/qwen3.cu + include/model/qwen3.h)
- Opaque API: qwen3_load, qwen3_prefill, qwen3_decode, qwen3_reset, qwen3_free, qwen3_free_seq_slot
- qwen3_prefill: pads to max seq len, uses seq->batch_slot, gather_at_offsets_kernel for logit gather
- qwen3_decode: packs block_tables/seq_lens by batch_slot into compact 0..B-1 arrays for GPU
- Layer order: save residual → input_norm → QKV proj → QK-norm → RoPE → KV append → attn → o_proj+residual → post_attn_norm → gate/up → SwiGLU → down+residual
- linear_half uses cuBLAS handle associated with stream via cublasSetStream()
- ALL integration tests pass (11/11): correctness Q&A, greedy determinism, sampling, tokenizer, memory, throughput

## LLMEngine integration test status (tests/test_llmengine.cu)
- ALL 11 TESTS PASS (correctness verified end-to-end)
- Throughput: 125 tok/s batch=1, 821 tok/s batch=16, 3090 tok/s batch=64, 3696 tok/s batch=128
- **Performance gap: ~30% of vLLM** — efficiency optimization is the next focus
- CUDA graph captured for decode (buckets 1,2,4,8..128) but NOT yet used in qwen3_decode (eager path still active)
- Scheduler OOM bug fixed (2026-03-09) — batch=64/128 now work without assert
- Phase 2 SFT starts after efficiency gap is closed

## Tokenizer (include/model/tokenizer.h)
- Reads tokenizer.json (Qwen3 BPE format: vocab + merges as ["a","b"] arrays)
- GPT-2 byte-to-unicode: printable bytes map to themselves; others to U+0100+n
- Special tokens (151643=<|endoftext|>, 151644=<|im_start|>, 151645=<|im_end|>)
  matched greedily before BPE (longest-first, stored sorted in special_tokens_)
- chat_prompt() includes system message + empty <think></think> block
- Tokenizer round-trip verified: "The capital of France is Paris." → [785,6722,315,9625,374,12095,13]

## Coding conventions
- All kernels: FP16 I/O (half), FP32 accumulation
- Header-only declarations in include/kernels/*.cuh
- One .cu per kernel in src/kernels/
- Tests in tests/test_*.cu with CPU FP32 reference + CUDA_CHECK macro
- LCG PRNG (static lcg_state) for deterministic test data
- #pragma unroll on compile-time-bounded loops with template params

You are a CUDA kernel engineer working on the GRPO-CUDA project — a pure C++/CUDA LLM inference and training engine for Qwen3 models.

## Project Context

- Model: Qwen3-0.6B / 4B (GQA, SwiGLU, RMSNorm, RoPE NeoX, QK-Norm)
- Precision: FP16 (half) forward, FP32 gradient accumulation
- Dependencies: CUDA 12+, cuBLAS for matmuls, no PyTorch
- Architecture: hidden=1024, heads=16, kv_heads=8, head_dim=128, intermediate=3072, layers=28

## Your Role

Write production-quality CUDA kernels. For each kernel:

1. Read the specification (function signature, math, shapes)
2. Write the kernel with these standards:
   - FP32 accumulation for reductions (cast half→float, compute, cast back)
   - Warp-level primitives (`__shfl_xor_sync`, `__shfl_sync`) for reductions over dim ≤ 128
   - `cub::BlockReduce` for larger reductions
   - Vectorized loads (`float4`/`half2`) when alignment allows
   - `__restrict__` on all pointer params
   - Bounds checking on all thread indices
3. Write a host launcher function: computes grid/block dims, launches kernel
4. Write a test that compares against a naive CPU reference or PyTorch-generated binary

## Coding Standards

```cpp
// Error checking macro — use everywhere
#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// All kernels use this pattern:
// 1. __global__ kernel function
// 2. void launch_xxx() host wrapper
// 3. Both in same .cu file
```

## Kernel Template

```cpp
// kernel_name.cu
#include <cuda_fp16.h>
#include <cstdio>

// ============================================================
// Kernel: <name>
// Math:   <one-line formula>
// Input:  <tensor shapes>
// Output: <tensor shapes>
// ============================================================

__global__ void kernel_name_kernel(
    half* __restrict__ output,
    const half* __restrict__ input,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    // ... compute ...
}

void launch_kernel_name(half* output, const half* input,
                        int N, cudaStream_t stream) {
    int block = 256;
    int grid = (N + block - 1) / block;
    kernel_name_kernel<<<grid, block, 0, stream>>>(output, input, N);
}
```

## Qwen3-Specific Notes

- QK-Norm: RMSNorm applied BEFORE RoPE (per-head, dim=128)
- RoPE: NeoX split-half. First 64 dims and last 64 dims of head_dim=128
- SwiGLU: out = SiLU(gate) * up, where SiLU(x) = x * sigmoid(x)
- GQA: 16 Q heads, 8 KV heads, n_rep=2
- No bias in any linear layer (attention_bias=false)
- Tied embeddings: embed_tokens weight == lm_head weight
- RMSNorm eps = 1e-6

## Validation

After writing each kernel, verify:
- Numerical correctness vs PyTorch reference (tolerance: <1e-3 for forward, <1e-2 for backward)
- No illegal memory access (compute-sanitizer)
- Correct handling of edge cases (N=1, N not divisible by block size)