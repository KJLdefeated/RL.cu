# GRPO-CUDA: Project Design Document

**Pure C++/CUDA LLM RL Engine — Phase 1: Inference for Qwen3-0.6B**
*v0.1 · February 2026 · Precision: FP16 (half)*

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
| Precision | **FP16** (half), FP32 accumulation |

**Memory**: ~1.2 GB weights (FP16), ~56 MB KV cache (B=1, seq=2048).

### Per-Layer Forward Pass

```
residual = hidden
hidden = RMSNorm(hidden)                         // input_layernorm [1024]
Q = hidden @ q_proj.W                            // [1024] -> [16*128=2048]
K = hidden @ k_proj.W                            // [1024] -> [8*128=1024]
V = hidden @ v_proj.W                            // [1024] -> [8*128=1024]
Q, K = reshape to [B, S, heads, 128]
Q = RMSNorm(Q, per_head)                         // q_norm on head_dim ⚠️ BEFORE RoPE
K = RMSNorm(K, per_head)                         // k_norm on head_dim ⚠️ BEFORE RoPE
Q, K = apply_RoPE(Q, K)                          // NeoX split-half, theta=1M
KV_cache.append(K, V)
attn_out = FlashAttention2(Q, K_cached, V_cached) // GQA: 2 Q heads per KV head
hidden = attn_out @ o_proj.W + residual

residual = hidden
hidden = RMSNorm(hidden)                         // post_attention_layernorm
hidden = SiLU(hidden @ gate_proj.W) * (hidden @ up_proj.W)  // SwiGLU
hidden = hidden @ down_proj.W + residual
```

> **Critical**: QK-Norm is applied BEFORE RoPE. TorchTune had a bug reversing this order. Match HuggingFace exactly.

---

## 2. Project Structure

```
grpo-cuda/
├── CMakeLists.txt
├── include/
│   ├── grpo/
│   │   ├── common.cuh          # Types (half), error macros, CUDA utils
│   │   ├── config.h            # Qwen3Config struct (parsed from JSON)
│   │   ├── tensor.cuh          # Lightweight GPU tensor (ptr + shape)
│   │   └── model.h             # Qwen3Model: weights + forward()
│   └── kernels/
│       ├── rmsnorm.cuh
│       ├── rope.cuh
│       ├── attention.cuh       # FA2 forward
│       ├── swiglu.cuh
│       ├── embedding.cuh
│       └── softmax.cuh
│
├── src/
│   ├── kernels/                # One .cu per kernel
│   │   ├── rmsnorm.cu          # Warp-reduction, FP32 accum
│   │   ├── rope.cu             # NeoX split-half, precomputed freqs
│   │   ├── attention.cu        # FA2 forward, tiled + online softmax
│   │   ├── swiglu.cu           # Fused SiLU(gate)*up
│   │   ├── embedding.cu        # Gather + output projection
│   │   └── softmax.cu          # For sampling
│   ├── model/
│   │   ├── config.cpp          # Parse config.json
│   │   ├── weights.cpp         # Safetensors mmap loader
│   │   ├── qwen3.cu            # Full forward pass orchestration
│   │   └── kv_cache.cu         # Pre-allocated, append-on-decode
│   ├── inference/
│   │   ├── generate.cu         # Prefill + decode loop
│   │   └── sampler.cu          # Top-k, top-p, temperature (on GPU)
│   ├── tokenizer/
│   │   └── bpe.cpp             # Load HF tokenizer
│   └── main.cpp
│
├── tests/
│   ├── test_rmsnorm.cu         # Each kernel vs PyTorch reference
│   ├── test_rope.cu
│   ├── test_attention.cu
│   ├── test_forward.cu         # Full forward: logits match HF
│   ├── test_generate.cu        # Greedy decode = identical tokens
│   └── generate_references.py  # Dump PyTorch tensors to binary
│
├── scripts/
│   ├── download_model.py
│   └── benchmark.sh
│
└── third_party/
    ├── json.hpp                # nlohmann/json (header-only)
    └── safetensors.h           # Minimal parser
```

---

## 3. Kernel Design

All kernels: FP16 (`half`) I/O, FP32 accumulation internally where needed (reductions, softmax).

| Kernel | Implementation Notes |
|---|---|
| **RMSNorm** | Warp-level `__shfl_xor_sync` reduction. 1 warp for head_dim=128, 4 warps for hidden=1024. |
| **RoPE** | NeoX split-half: split dim into [0:64] and [64:128], rotate. Precompute sin/cos table at load. |
| **FA2 Forward** | Tiled Q/K/V in SMEM, online softmax. Br=64, Bc=64 for head_dim=128. GQA: `kv_head = q_head / 2`. |
| **SwiGLU** | Fused element-wise: `out[i] = silu(gate[i]) * up[i]`. One thread per element. |
| **Embedding** | Gather: `out[b][s] = weight[ids[b][s]]`. Output projection reuses same weight via cuBLAS. |
| **Softmax + Sampling** | Online softmax on GPU, then top-k filter + multinomial sample. Avoids 600KB logit transfer to host. |
| **Matmuls** | All linear layers via **cuBLAS** `cublasHgemm` (FP16). Not custom — this is the pragmatic choice. |

---

## 4. Weight Loading

**Safetensors format**: 8-byte header length → JSON header → raw tensor bytes.

```cpp
struct Qwen3Weights {
    half* embed_tokens;             // [151936, 1024] — shared with output proj
    struct Layer {
        half *input_layernorm;      // [1024]
        half *q_proj, *k_proj, *v_proj, *o_proj;  // projection weights
        half *q_norm, *k_norm;      // [128] each — per-head RMSNorm
        half *post_attn_layernorm;  // [1024]
        half *gate_proj, *up_proj, *down_proj;     // FFN weights
    } layers[28];
    half* final_norm;               // [1024]
};
```

Loader: mmap safetensors file → parse JSON header → `cudaMemcpy` each tensor to pre-allocated GPU buffer. One-time cost at startup.

---

## 5. Execution Flow

**Prefill** (prompt, compute-bound):
```
tokens [1, S] → Embedding → 28x Layer(full seq) → RMSNorm → Logits → Sample
                                                    ↓
                                              KV cache filled
```

**Decode** (token-by-token, memory-bound):
```
token [1, 1] → Embedding → 28x Layer(S=1, read KV cache) → RMSNorm → Logits → Sample
                                     ↓
                              KV cache append
```

All computation on GPU. Only transfers: token IDs in, sampled token out.

---

## 6. Testing Strategy

| Level | Test | Pass Criteria |
|---|---|---|
| Kernel | `test_rmsnorm`, `test_rope`, `test_swiglu` | < 1e-3 vs PyTorch FP16 |
| Kernel | `test_attention` (FA2) | < 5e-3 vs naive attention |
| Integration | `test_forward` (full model) | Logits < 1e-2 vs HuggingFace |
| E2E | `test_generate` (greedy) | Identical token sequence as HF `generate()` |

Reference data generated by `tests/generate_references.py` which runs HuggingFace Qwen3-0.6B and dumps tensors.

---

## 7. Build & Dependencies

```cmake
cmake_minimum_required(VERSION 3.24)
project(grpo-cuda LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)  # Ampere → Hopper
find_package(CUDAToolkit REQUIRED)
target_link_libraries(grpo-cuda CUDA::cublas CUDA::cudart)
```

| Dependency | Required? | Purpose |
|---|---|---|
| CUDA Toolkit ≥ 12.0 | Yes | Runtime, cuBLAS, nvcc |
| nlohmann/json | Vendored | config.json + safetensors header |
| Python + PyTorch | Dev only | Generate test reference data |

---

## 8. Roadmap (Phase 1: Inference)

| Month | Goal | Milestone |
|---|---|---|
| **March** | CUDA kernels + FA2 | RMSNorm, RoPE, SwiGLU, Embedding done. FA2 forward WIP. All kernels validated vs PyTorch. |
| **April** | Full forward pass | Wire kernels → Qwen3 forward. KV cache. Logits match HuggingFace. |
| **May** | Generation engine | Prefill/decode loop, tokenizer, sampling. Generate coherent text. Benchmark vs llama.cpp. Release. |

---

## 9. Extension Points for Training & GRPO

The inference engine is designed to extend into training:

- **Backward kernels**: Each `.cu` file will add `_backward()` functions (same file, same data structures).
- **Activation storage**: Forward pass gains a `train_mode` flag to cache intermediates for backprop.
- **Log-probs**: Add log-softmax + gather for per-token log-probabilities (needed by GRPO loss).
- **Batched rollouts**: KV cache has batch dim. Set `B=G` (group size) for parallel GRPO completions.
- **Reference model**: Instantiate `Qwen3Weights` twice — active policy + frozen reference for KL divergence.
- **Optimizer**: FP16 model weights, **FP32 master weights** + FP32 optimizer states (AdamW m, v). Mixed-precision training pattern.