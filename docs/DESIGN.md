# GRPO-CUDA: Project Design Document

**Pure C++/CUDA LLM RL Engine — Phase 1: Inference for Qwen3-0.6B**
*v0.2 · February 2026 · GPU: sm_120 (Blackwell) · CUDA: 12.8*

---

## 1. Qwen3-0.6B Architecture

Exact values from `config.json`:

| Parameter | Value |
|---|---|
| hidden_size | 1024 |
| num_hidden_layers | 28 |
| num_attention_heads (Q) | 16 |
| num_key_value_heads (KV) | 8 |
| head_dim | 128 |
| intermediate_size | 3072 |
| vocab_size | 151,936 |
| rope_theta | 1,000,000 |
| tie_word_embeddings | true |
| Precision | **FP16** (half) weights/activations; **FP32** norm weights + accumulation |

**Memory**: ~1.1 GB weights on GPU (FP16 projections + FP32 norms).

### Per-Layer Forward Pass

```
residual = hidden
hidden = RMSNorm(hidden, input_layernorm)            // weight: float[1024]
Q = hidden @ q_proj.W                               // [B*S, 1024] -> [B*S, 2048]
K = hidden @ k_proj.W                               // [B*S, 1024] -> [B*S, 1024]
V = hidden @ v_proj.W                               // [B*S, 1024] -> [B*S, 1024]
Q, K = reshape to [B*S, heads, 128]
Q = RMSNorm(Q, q_norm)                              // weight: float[128]  ⚠️ BEFORE RoPE
K = RMSNorm(K, k_norm)                              // weight: float[128]  ⚠️ BEFORE RoPE
Q, K = apply_RoPE(Q, K, position_ids)               // NeoX split-half, theta=1M
KV_cache.append(K, V)                               // paged, slot_mapping
attn_out = FlashAttention2(Q, K_cached, V_cached)   // GQA: 2 Q heads per KV head
hidden = attn_out @ o_proj.W + residual

residual = hidden
hidden = RMSNorm(hidden, post_attn_layernorm)        // weight: float[1024]
hidden = SiLU(hidden @ gate_proj.W) * (hidden @ up_proj.W)   // SwiGLU
hidden = hidden @ down_proj.W + residual
```

> **Critical**: QK-Norm is applied BEFORE RoPE. TorchTune had a bug reversing this order.
> Match HuggingFace exactly: `q_norm → rope`, not `rope → q_norm`.

---

## 2. Project Structure

```
RL_cuda/
├── Makefile                     # make test_<name> or make tests
├── docs/
│   ├── DESIGN.md                # this file
│   └── ATTENTION.md             # FA2 + paged attention design notes
│
├── include/
│   ├── kernels/
│   │   ├── rmsnorm.cuh          # launch_rmsnorm (float* weight)
│   │   ├── rope.cuh             # launch_rope_precompute, launch_rope
│   │   ├── attention.cuh        # launch_flash_attention_prefill, launch_paged_attention_decode
│   │   ├── swiglu.cuh           # launch_swiglu
│   │   ├── embedding.cuh        # launch_embedding
│   │   └── softmax.cuh          # launch_softmax
│   ├── model/
│   │   ├── config.h             # Qwen3Config struct + load_config()
│   │   ├── weights.h            # Qwen3Weights + Qwen3LayerWeights structs
│   │   └── kv_cache.cuh         # PagedKVCache struct + all declarations
│   └── third_party/
│       └── json.hpp             # nlohmann/json v3.11.3 (single-header)
│
├── src/
│   ├── kernels/                 # One .cu/.cpp per kernel
│   │   ├── rmsnorm.cu
│   │   ├── rope.cu
│   │   ├── attention.cu         # FA2 prefill + paged decode
│   │   ├── swiglu.cu
│   │   ├── embedding.cu
│   │   ├── softmax.cu
│   │   ├── config.cpp           # Parse config.json via nlohmann/json
│   │   └── weights.cpp          # Safetensors mmap loader (BF16→FP16/FP32)
│   └── model/
│       ├── kv_cache.cu          # Paged KV cache: alloc, append, fork
│       └── qwen3.cu             # Full forward pass orchestration (WIP)
│
└── tests/
    ├── test_rmsnorm.cu          # DONE ✓
    ├── test_softmax.cu          # DONE ✓
    ├── test_swiglu.cu           # DONE ✓
    ├── test_attention.cu        # DONE ✓ (FA2 prefill + paged decode)
    ├── test_kv_cache.cu         # DONE ✓
    ├── test_rope.cu             # DONE ✓
    ├── test_embedding.cu        # DONE ✓
    └── test_loading_weights.cpp # DONE ✓
```

---

## 3. Kernel Design

All kernels: FP16 (`half`) I/O; FP32 accumulation internally.
**Exception**: norm weights (`input_layernorm`, `q_norm`, `k_norm`, `post_attn_layernorm`, `final_norm`) stored and passed as `float*` — BF16→FP32 on load. Avoids FP16 exponent range loss for trained norm weights that drift from 1.0.

| Kernel | Grid / Block | Notes |
|---|---|---|
| **RMSNorm** | Grid(rows), Block(32 or 128) | Warp `__shfl_xor_sync` reduction; 1 warp for cols=128, 4 for cols=1024 |
| **RoPE precompute** | Grid(max_seq_len), Block(D/2=64) | FP32 sin/cos table `[max_seq_len, 64]` |
| **RoPE apply** | Grid(num_tokens, H_q), Block(D/2=64) | NeoX split-half; FP32 rotation arithmetic; applies Q (all heads) and K (H_kv heads) |
| **FA2 prefill** | Grid(⌈S/Br⌉, H_q, B), Block(Br=16) | template<HEAD_DIM=128, Br=16, Bc=64>; K_smem+V_smem=32 KB; online softmax; causal `break` |
| **Paged decode** | Grid(num_seqs, H_q), Block(1) | template<HEAD_DIM=128, BLOCK_SIZE=16>; single thread per (seq, head); walks block_table |
| **KV cache** | Grid(num_tokens), Block(128) | `reshape_and_cache_half_kernel`; slot = phys_block×16 + offset |
| **SwiGLU** | Grid(⌈N/256⌉), Block(256) | Fused: `out[i] = silu(gate[i]) * up[i]`; one thread per element |
| **Embedding** | Grid(num_tokens), Block(256) | Gather: `out[t] = weight[ids[t]]`; stride loop over hidden_size; exact FP16 copy |
| **Softmax** | — | Numerically stable: subtract row max before exp |
| **Matmuls** | — | All linear projections via **cuBLAS** `cublasHgemm` (FP16) |

### FA2 Design Details

```
template<HEAD_DIM=128, Br=16, Bc=64>
Grid:  (⌈S/Br⌉, H_q, B)    — one block per (Q-tile, Q-head, batch)
Block: (Br=16 threads)
SMEM:  K_smem[Bc][HEAD_DIM] + V_smem[Bc][HEAD_DIM] = 64 × 128 × 2 × 2B = 32 KB
Regs:  q_reg[128] (FP32), o_acc[128] (FP32), row_max, row_sum
GQA:   kv_head = q_head * H_kv / H_q  (integer, 2:1 ratio for 0.6B)
```

### Paged KV Cache Design

```
Pool layout: [num_layers, total_blocks, num_kv_heads, KV_BLOCK_SIZE=16, head_dim]
block_tables: [max_batch_size, max_blocks_per_seq]  — GPU int32
slot_mapping: int64  =  physical_block * 16 + in_block_offset
fork (GRPO): copies block_table pointers only — no KV data copy
```

### RoPE Design

NeoX split-half rotation for i ∈ [0, D/2):
```
out[i]       = x[i]      * cos - x[i+D/2] * sin
out[i + D/2] = x[i+D/2]  * cos + x[i]     * sin
```
- FP32 arithmetic for rotation (catastrophic cancellation in FP16: `a*c - b*s` loses 1–2 digits)
- Use `hd2 = head_dim / 2` (not `half`) — `half` is a reserved CUDA type name

---

## 4. Weight Loading

**Safetensors format**: `[uint64 header_size][JSON header][raw tensor data]`.
Offsets in JSON are relative to the start of tensor data (after header).

Loader uses `mmap` for zero-copy access. `SafetensorsFile` has a move constructor that nulls `fd` and `mapped` on the moved-from object — prevents double-`munmap` when stored in `std::vector`.

### Weight Types

```cpp
struct Qwen3LayerWeights {
    float* input_layernorm;      // [hidden_size=1024]          FP32
    half*  q_proj;               // [q_dim=2048,   hidden=1024] FP16
    half*  k_proj;               // [kv_dim=1024,  hidden=1024] FP16
    half*  v_proj;               // [kv_dim=1024,  hidden=1024] FP16
    half*  o_proj;               // [hidden=1024,  q_dim=2048]  FP16
    float* q_norm;               // [head_dim=128]              FP32
    float* k_norm;               // [head_dim=128]              FP32
    float* post_attn_layernorm;  // [hidden_size=1024]          FP32
    half*  gate_proj;            // [intermediate=3072, h=1024] FP16
    half*  up_proj;              // [intermediate=3072, h=1024] FP16
    half*  down_proj;            // [hidden=1024, inter=3072]   FP16
};

struct Qwen3Weights {
    half*  embed_tokens;   // [vocab=151936, hidden=1024]  FP16 (tied with lm_head)
    float* final_norm;     // [hidden=1024]                FP32
    std::vector<Qwen3LayerWeights> layers;  // 28 layers
};
```

Qwen3-0.6B ships as BF16 in safetensors. The loader converts:
- Projection/FFN weights: **BF16 → FP16** (same size, same GPU memory)
- Norm weights: **BF16 → FP32** (2× memory, but only ~64 KB total for all layers)

---

## 5. Execution Flow

**Prefill** (prompt, compute-bound):
```
token_ids [B, S] → Embedding → 28× Layer(full seq) → RMSNorm(final) → Logits → Sample
                                                              ↓
                                                     KV cache filled (paged)
```

**Decode** (token-by-token, memory-bound):
```
token_id [B, 1] → Embedding → 28× Layer(S=1, paged KV read) → RMSNorm(final) → Logits → Sample
                                          ↓
                                   KV cache append
```

All computation on GPU. Only transfers: token IDs in, sampled token out (~4 bytes/step).

---

## 6. Testing Strategy

Tests use a CPU FP32 reference (not PyTorch) computed inline, with deterministic LCG PRNG.

| Test | Kernel(s) | Pass Criteria | Actual |
|---|---|---|---|
| `test_rmsnorm` | RMSNorm | max_err < 1e-3 | ~1e-4 |
| `test_softmax` | Softmax | max_err < 1e-3 | — |
| `test_swiglu` | SwiGLU | max_err < 1e-3 | — |
| `test_attention` (5 prefill + 4 decode) | FA2, Paged attn | max_err < 5e-3 | prefill ~1.2e-4, decode ~3e-5 |
| `test_kv_cache` (8 cases) | KV append, fork | exact slot mapping | ✓ |
| `test_rope` (7 cases) | RoPE precompute + apply | max_err < 1e-3 | ~2.4e-4 |
| `test_embedding` (5 cases) | Embedding gather | exact match | 0 error |
| `test_loading_weights` | Weight loader | pointers non-null, values finite | ✓ |

Build and run:
```bash
make test_<name>     # build + run a single test
make tests           # build + run all tests
```

---

## 7. Build

```makefile
CUDA_HOME := /usr/local/cuda-12.8
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := sm_120          # Blackwell (RTX PRO 6000 / RTX 50xx)
NVCCFLAGS := -O2 -std=c++17 -I include --gpu-architecture=$(ARCH)
```

Override architecture: `make tests ARCH=sm_89` (Ada) / `sm_86` (Ampere) / `sm_80` (A100).

| Dependency | Version | Purpose |
|---|---|---|
| CUDA Toolkit | 12.8 | nvcc, cuBLAS, cudart |
| nlohmann/json | 3.11.3 | config.json + safetensors header parsing (vendored in `include/third_party/`) |

---

## 8. Roadmap (Phase 1: Inference)

| Status | Goal | Details |
|---|---|---|
| ✅ **Done** | CUDA kernels | RMSNorm, RoPE, FA2 (prefill + paged decode), SwiGLU, Embedding, Softmax, KV Cache — all tested |
| ✅ **Done** | Weight loading | Safetensors mmap loader; BF16→FP16/FP32 conversion; config.json parser |
| 🔧 **Next** | Full forward pass | Wire kernels into `src/model/qwen3.cu`; cuBLAS for all projections; single-token decode + prefill |
| ⬜ **Planned** | Tokenizer | Load HF BPE tokenizer (`tokenizer.json`), encode/decode |
| ⬜ **Planned** | Generation engine | Prefill + decode loop, top-k/top-p sampling on GPU |
| ⬜ **Planned** | Benchmark | vs llama.cpp on same hardware; tokens/s for batch=1 |

---

## 9. Extension Points for Training & GRPO

The inference engine is designed to extend into training:

- **Backward kernels**: Each `.cu` gains `_backward()` functions (same file, same data structures).
- **Activation storage**: Forward pass gains a `train_mode` flag to cache intermediates for backprop.
- **Log-probs**: Log-softmax + gather for per-token log-probabilities (GRPO loss input).
- **Batched rollouts**: KV cache has batch dim. Set `B=G` (group size) for G parallel GRPO completions. `paged_kv_cache_fork()` shares prompt KV blocks across all G sequences — zero copy.
- **Reference model**: Instantiate `Qwen3Weights` twice — active policy + frozen reference for KL divergence.
- **Optimizer**: FP16 model weights, **FP32 master weights** + FP32 optimizer states (AdamW m, v). Standard mixed-precision training.
