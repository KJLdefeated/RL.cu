<p align="center">
  <h1 align="center">RL.cu</h1>
  <p align="center">
    <b>LLM Reinforcement Learning in Pure CUDA — From Kernels to GRPO</b>
  </p>
  <p align="center">
    Zero PyTorch. One GPU. Full RL loop.
  </p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#kernels">Kernels</a> &bull;
  <a href="#inference-engine">Inference Engine</a> &bull;
  <a href="#training">Training</a> &bull;
  <a href="#benchmarks">Benchmarks</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#develop-with-claude-code">Claude Code</a>
</p>

---

A from-scratch implementation of the complete LLM RL pipeline — hand-written CUDA kernels, a vLLM-style inference engine with continuous batching, and GRPO training — all in a single binary with no Python runtime dependency.

**Built for learning.** Every layer is explicit: when you read `qwen3_forward()`, you see exactly what happens — `embedding → rmsnorm → qkv_proj → qk_norm → rope → flash_attention → o_proj → swiglu → ...`. No hidden graph. No autograd tape. No framework magic. If you want to understand how LLM inference and RL training actually work at the GPU level, this is the codebase to read.

## What We Built

| Layer | What | From Scratch? |
|-------|------|:---:|
| **CUDA Kernels** | FlashAttention-2 (fwd+bwd), RMSNorm, RoPE, SwiGLU, Embedding, Sampler, AdamW, GRPO loss — all with forward AND backward passes | Yes |
| **Model** | Qwen3-0.6B full forward + backward pass, safetensors weight loading | Yes |
| **KV Cache** | Paged KV cache with block manager (same design as vLLM) | Yes |
| **Inference Engine** | Continuous batching, CUDA graph capture, two-phase scheduling | Yes |
| **Tokenizer** | BPE tokenizer reading HuggingFace `tokenizer.json` directly | Yes |
| **Training** | SFT + GRPO with gradient checkpointing, mixed-precision AdamW | Yes |

## Quick Start

### Requirements
- CUDA Toolkit >= 12.0
- GPU: Ampere or newer (sm_80+)
- Python 3.8+ (only for downloading model weights / preparing data)

### Build & Run

```bash
git clone https://github.com/KJLdefeated/RL.cu.git && cd RL.cu

# Download model weights
pip install huggingface_hub
python scripts/download_model.py Qwen/Qwen3-0.6B model_weights/Qwen3-0.6B

# Build
make build/test_llmengine

# Run inference (includes correctness tests + throughput benchmark)
./build/test_llmengine model_weights/Qwen3-0.6B
```

### GRPO Training

```bash
# Prepare dataset (downloads DeepMath-103K as JSONL)
pip install datasets
python scripts/prepare_data.py --mode grpo-text \
    --dataset trl-lib/DeepMath-103K --output data/deepmath-103k.jsonl

# Build & run GRPO training
make build/train_grpo
./build/train_grpo model_weights/Qwen3-0.6B data/deepmath-103k.jsonl 100
```

## Kernels

Every kernel is hand-written with FP16 I/O and FP32 accumulation. Each has a standalone test comparing against a CPU/PyTorch reference.

| Kernel | File | Lines | Key Design |
|--------|------|------:|------------|
| **FlashAttention-2** | `src/kernels/attention.cu` | 1,367 | Forward + backward, GQA, online softmax, tiled WMMA, causal masking |
| **RMSNorm** | `src/kernels/rmsnorm.cu` | 211 | Forward + backward, warp-level reduction |
| **RoPE** | `src/kernels/rope.cu` | 160 | Forward + backward, NeoX split-half, FP32 sin/cos tables |
| **SwiGLU** | `src/kernels/swiglu.cu` | 77 | Forward + backward, fused SiLU * up |
| **Embedding** | `src/kernels/embedding.cu` | 86 | Forward gather + backward scatter-add |
| **Fused Norm+Linear** | `src/kernels/fused_norm_linear.cu` | 188 | RMSNorm fused with cuBLAS GEMM (saves one HBM round-trip) |
| **Sampler** | `src/kernels/sampler.cu` | 147 | Top-k, top-p, temperature, Gumbel-max single-pass |
| **AdamW** | `src/kernels/adamw.cu` | 108 | Mixed-precision (FP16 params, FP32 moments), fused update |
| **Softmax** | `src/kernels/softmax.cu` | 115 | Numerically stable, warp-level |
| **Linear** | `src/kernels/linear.cu` | 103 | cuBLAS FP16 GEMM + backward (dX, dW) |

Build and run any individual kernel test:
```bash
make build/test_attention && ./build/test_attention
make build/test_rmsnorm  && ./build/test_rmsnorm
# ... etc
```

## Inference Engine

A vLLM-style engine with continuous batching, paged KV cache, and CUDA graph acceleration.

**Features:**
- **Continuous batching** — two-phase decode + prefill per step, new requests start immediately
- **Paged KV cache** — block-level memory management, no wasted pre-allocation
- **CUDA graphs** — decode captured at bucket sizes (1, 2, 4, 8, ..., 256), single `cudaGraphLaunch` replaces 400+ kernel calls
- **Fused projections** — QKV and gate+up projections fused into single GEMMs (zero extra memory)

```
Engine step:
  schedule_decode() → prefill new seqs → sample → postprocess
  └── continuous: finished slots instantly reused by waiting requests
```

## Training

### SFT (Supervised Fine-Tuning)
Standard cross-entropy training with chunked lm_head to control memory.

### GRPO (Group Relative Policy Optimization)
Full RL training loop in a single CUDA binary:

```
For each step:
  1. Generate G completions per prompt using the inference engine
  2. Score with reward function (e.g., boxed-answer matching)
  3. Compute GRPO advantages (group-relative normalization)
  4. Forward pass with gradient checkpointing
  5. Backward pass (recompute activations per layer)
  6. AdamW update with gradient accumulation + clipping
```

**Gradient checkpointing** saves only per-layer input residuals + FlashAttention LSE, recomputing all other activations during backward. This reduces activation memory from **~61 GB to ~6.4 GB** (54.8 GB saved) for batch of 64 sequences.

**Sleep/wakeup lifecycle:** KV cache pools are freed during training and re-allocated before generation, so the same GPU memory is shared between inference and training phases.

## Benchmarks

### Inference Throughput (Qwen3-0.6B, RTX PRO 6000)

| Batch Size | Throughput (tok/s) |
|:----------:|:------------------:|
| **256** | **6,963** |

94% of nano-vllm throughput (7,411 tok/s). See [docs/ENGINE.md](docs/ENGINE.md) for the full optimization journey.

### GRPO Training (Qwen3-0.6B, 8 prompts x 8 generations)

| Metric | Value |
|--------|------:|
| Generation throughput | 3,834 tok/s |
| Activation memory (with checkpointing) | 6.4 GB |
| Activation memory (without checkpointing) | 61 GB |
| Training step time | ~107s |

## Architecture

```
RL.cu
├── src/kernels/          # Hand-written CUDA kernels (fwd + bwd)
│   ├── attention.cu      #   FlashAttention-2 with GQA
│   ├── rmsnorm.cu        #   RMSNorm
│   ├── rope.cu           #   Rotary Position Embedding (NeoX)
│   ├── swiglu.cu         #   SwiGLU activation
│   ├── embedding.cu      #   Token embedding
│   ├── sampler.cu        #   Top-k/p sampling
│   ├── adamw.cu          #   Mixed-precision optimizer
│   └── ...
├── src/model/
│   ├── qwen3.cu          #   Full forward + backward pass (1,432 lines)
│   └── kv_cache.cu       #   Paged KV cache operations
├── include/engine/       # Inference engine
│   ├── llm_engine.h      #   LLMEngine: top-level API
│   ├── scheduler.h       #   Continuous batching scheduler
│   ├── block_manager.h   #   Paged KV block allocation
│   └── model_runner.cuh  #   Model execution + CUDA graphs
├── include/training/     # Training infrastructure
│   ├── GRPO_trainer.h    #   GRPO training loop
│   ├── SFT_trainer.h     #   SFT training loop
│   ├── optimizer.h       #   AdamW with flat buffer
│   └── lr_scheduler.h    #   Cosine + warmup
├── include/model/
│   ├── tokenizer.h       #   BPE tokenizer (reads HF tokenizer.json)
│   ├── config.h          #   Model + engine config
│   └── weights.h         #   Safetensors loader (mmap)
└── tests/                # 26 test files, each kernel validated independently
```

## Design Philosophy

**One binary, one process, one GPU.** Rollout generation and policy training happen in the same CUDA context. No serialization overhead. No weight transfer. No Python GIL.

**Train-inference mismatch = 0 by construction.** The GRPO policy ratio needs `log_probs_old` (rollout) and `log_probs_new` (training). Both come from the same `qwen3_forward()` — same kernels, same FP reduction order. The ratio is exactly 1.0 at the first iteration. This isn't a feature; it's a property of the architecture.

**llm.c for RL.** [llm.c](https://github.com/karpathy/llm.c) showed that pretraining can be done in pure CUDA. We extend this to the full RL loop: generate → reward → advantage → loss → backward → optimize.

## Comparison

| | llm.c | vLLM | TRL | **RL.cu** |
|---|---|---|---|---|
| Language | C/CUDA | Python + CUDA | Python + PyTorch | **C++/CUDA** |
| Inference engine | - | Yes | via vLLM | **Yes** |
| Continuous batching | - | Yes | via vLLM | **Yes** |
| Paged KV cache | - | Yes | via vLLM | **Yes** |
| CUDA graphs | - | Yes | - | **Yes** |
| SFT training | Yes (GPT-2) | - | Yes | **Yes (Qwen3)** |
| RL training (GRPO) | - | - | Yes | **Yes** |
| Unified inference + training | N/A | N/A | Bridge needed | **Yes** |
| Zero train-infer mismatch | N/A | N/A | Requires mitigation | **By design** |
| Runtime dependencies | None | Python + PyTorch | Python + PyTorch | **None** |

## Tests

Every component has a standalone test. Run all tests:
```bash
make tests
```

Or run individually:
```bash
make build/test_attention && ./build/test_attention      # FlashAttention-2 fwd+bwd
make build/test_rmsnorm  && ./build/test_rmsnorm         # RMSNorm
make build/test_rope     && ./build/test_rope            # RoPE
make build/test_sampler  && ./build/test_sampler          # Top-k/p sampling
make build/test_llmengine && ./build/test_llmengine model_weights/Qwen3-0.6B  # Full engine
# ... 26 tests total
```

## Develop with Claude Code

This project is built to work with [Claude Code](https://docs.anthropic.com/en/docs/claude-code). The repo includes project-level docs and skills so Claude can contribute effectively from the first message.

### What's configured

**`.claude/CLAUDE.md`** — 270-line project memory that gives Claude full context:
- GPU architecture (sm_120, CUDA 12.8), model dimensions (Qwen3-0.6B), build system
- Every kernel's API, grid/block config, and known pitfalls
- Scheduler design, KV cache layout, attention kernel gotchas
- All bugs we've fixed and why (so Claude doesn't reintroduce them)
- Coding conventions: FP16 I/O, FP32 accumulation, `#pragma unroll`, test patterns

**`.claude/skills/add_new_kernel/`** — Step-by-step skill for implementing new CUDA kernels:
- Header in `include/kernels/*.cuh`, source in `src/kernels/*.cu`, test in `tests/test_*.cu`
- Includes full examples (FlashAttention-2, RMSNorm) showing the kernel → launcher → test pattern
- Warp-level reduction, shared memory tiling, vectorized loads, bounds checking
- CPU reference + tolerance comparison template

### How to use it

```bash
# Install Claude Code
npm install -g @anthropic-ai/claude-code

# Start working — Claude already knows the project
cd RL.cu
claude

# Examples of what Claude can do with the project context:
# "Add a flash decoding kernel for paged attention"
# "Why is my attention kernel producing NaN for S > 16?"
# "Optimize the RMSNorm backward kernel with warp shuffle"
# "Add INT8 quantization support for linear layers"
```

Claude understands the full architecture — kernel APIs, memory layouts, known bugs, and conventions — so it can write production-quality CUDA code that fits the project from the start.

## Contributing

Contributions welcome! Some areas where help would be great:

- **Flash Decoding (split-K)** — the decode attention kernel is currently single-threaded per (seq, head); a proper split-K implementation would give 5-10x speedup
- **Multi-GPU support** — tensor parallelism for larger models
- **More model architectures** — Llama, Gemma, etc.
- **Speculative decoding** — draft model + verification
- **Quantization** — INT8/INT4 weight quantization

## Acknowledgments

- [llm.c](https://github.com/karpathy/llm.c) by Andrej Karpathy — inspiration for "pure CUDA" approach
- [vLLM](https://github.com/vllm-project/vllm) — paged attention and continuous batching design
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — attention algorithm
- [CUTLASS](https://github.com/NVIDIA/cutlass) — WMMA primitives

## License

MIT
