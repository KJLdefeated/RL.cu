# include/ — Header Modules

All public API declarations for RL.cu. Each subdirectory is a self-contained module.

## Directory Layout

```
include/
├── cuda_utils.h          # CUDA_CHECK / CUBLAS_CHECK error macros
├── kernels/              # Kernel launch declarations (forward + backward)
├── model/                # Model structs, config, weights, tokenizer, KV cache
├── engine/               # Inference engine: scheduler, model runner, LLM engine
└── training/             # Training infrastructure: trainers, optimizer, dataloader
```

---

## kernels/

CUDA kernel **declarations only** — one `.cuh` per kernel. Implementations live in `src/kernels/`.

| Header | Kernel | API |
|--------|--------|-----|
| `rmsnorm.cuh` | RMSNorm (fwd + bwd) | `launch_rmsnorm`, `launch_rmsnorm_backward` |
| `rope.cuh` | Rotary Position Embedding | `launch_rope_precompute`, `launch_rope`, `launch_rope_backward` |
| `attention.cuh` | Flash Attention 2 prefill + paged decode + backward | `launch_flash_attention_prefill`, `launch_paged_attention_decode`, `launch_flash_attention_backward` |
| `swiglu.cuh` | SwiGLU activation (fwd + bwd) | `launch_swiglu`, `launch_swiglu_backward` |
| `embedding.cuh` | Token embedding gather (fwd + bwd) | `launch_embedding`, `launch_embedding_backward` |
| `linear.cuh` | Dense projection via CUTLASS/cuBLAS | `linear_half`, `linear_backward_half` |
| `softmax.cuh` | Numerically stable softmax | `launch_softmax` |
| `sampler.cuh` | Gumbel-max GPU sampler | `launch_sampler` |
| `adamw.cuh` | Fused AdamW parameter update | `launch_adamw_fp16`, `launch_adamw_fp32` |
| `fused_norm_linear.cuh` | Fused RMSNorm + Linear (WMMA) | `launch_fused_rmsnorm_linear` |

All kernels follow the convention: **FP16 (`half`) I/O, FP32 internal accumulation**.

---

## model/

Model definition, weight management, and inference primitives.

| Header | Purpose |
|--------|---------|
| `config.h` | `Qwen3Config` struct + `load_config()` from `config.json` |
| `weights.h` | `Qwen3Weights` / `Qwen3LayerWeights` — safetensors mmap loader (BF16 -> FP16/FP32) |
| `qwen3.h` | `Qwen3Model` struct + forward pass API: `qwen3_init`, `qwen3_prefill`, `qwen3_decode`, `qwen3_decode_graph` |
| `kv_cache.cuh` | `PagedKVCache` — block-based paged KV cache with append, fork (copy-on-write for GRPO), and free |
| `tokenizer.h` | BPE tokenizer (Qwen3 `tokenizer.json` format) with `encode()`, `decode()`, `chat_prompt()` |
| `loss.cuh` | Loss function declarations (cross-entropy for SFT, advantage computation for GRPO) |
| `sampling_parmas.h` | `SamplingParams` struct (temperature, top_k, top_p, max_new_tokens) |

Implementations: `src/model/qwen3.cu` (forward pass orchestration), `src/model/kv_cache.cu` (cache kernels).

---

## engine/

Inference serving engine — vLLM-style continuous batching with CUDA graph execution. See [engine/README.md](engine/README.md) for architecture details.

| Header | Purpose |
|--------|---------|
| `llm_engine.h` | `LLMEngine` — top-level API: `add_request()`, `step()` |
| `scheduler.h` | `Scheduler` — two-phase continuous batching (decode then prefill) with preemption |
| `block_manager.h` | `BlockManager` — physical KV block allocation with prefix caching (xxhash) |
| `model_runner.cuh` | `ModelRunner` — GPU execution: warmup, CUDA graph capture/replay, sampling |

---

## training/

Training loop infrastructure for SFT and GRPO reinforcement learning.

| Header | Purpose |
|--------|---------|
| `trainer.h` | `Trainer` base class — gradient accumulation, clipping, checkpoint save/load |
| `SFT_trainer.h` | `SFTTrainer` — supervised fine-tuning with masked NLL loss |
| `GRPO_trainer.h` | `GRPOTrainer` — Group Relative Policy Optimization with rollout generation, KL penalty, and advantage normalization |
| `optimizer.h` | `AdamWOptimizer` — flat-buffer AdamW with FP32 master weights and per-parameter weight decay |
| `lr_scheduler.h` | Learning rate schedules: cosine with warmup, constant with warmup |
| `dataloader.h` | `RLDTDataLoader` — binary RLDT format reader (prompt/completion pairs) |
| `logger.h` | JSONL training logger for metrics (loss, reward, grad norm, etc.) |
