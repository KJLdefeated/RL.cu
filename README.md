# GRPO-CUDA: Pure C++/CUDA LLM Reinforcement Learning Training

## Project Vision

Build an open-source, pure C++/CUDA framework for LLM reinforcement learning training — specifically GRPO (Group Relative Policy Optimization) — targeting Qwen3-4B, with **zero Python/PyTorch dependency** at runtime.

---

## Landscape & Related Work

### Existing Projects (What Exists)

| Project | What It Does | Gap for You |
|---------|-------------|-------------|
| **llm.c** (Karpathy) | Pure C/CUDA LLM *pretraining* (GPT-2/3). ~7% faster than PyTorch. | No RL training, no GRPO, no modern architectures (Qwen) |
| **veRL** (ByteDance) | Full RLHF/GRPO framework. Python + PyTorch + vLLM + Ray | Heavy Python stack, not pure C++/CUDA |
| **OpenRLHF** | Scalable RLHF/GRPO via Ray + vLLM + DeepSpeed | Same — Python ecosystem |
| **NeMo-RL** (NVIDIA) | GRPO training, reproduced DeepScaleR | Python + Megatron-LM + vLLM |
| **TRL** (HuggingFace) | GRPOTrainer in Python | Pure Python, not optimized |
| **llama.cpp** | C/C++ inference for many architectures including Qwen3 | Inference only, no training, no RL |
| **Unsloth** | Memory-efficient GRPO with custom CUDA kernels | Still Python/PyTorch wrapper |
| **ROLL** (Alibaba) | Scalable RL for LLMs with Megatron + SGLang | Python ecosystem |

### Key Insight: The Gap

**Nobody has built pure C++/CUDA GRPO training.** llm.c proved C/CUDA pretraining is viable and even faster. The RL training loop (GRPO specifically) has never been done outside the Python ecosystem. This is a genuine open-source contribution.

### Key References to Study

- **GRPO Algorithm**: DeepSeek-R1 paper, Cameron Wolfe's "GRPO" and "GRPO++ Tricks" overviews
- **llm.c Architecture**: The `train_gpt2.cu` codebase as the template for pure CUDA training
- **Flash Attention from Scratch**: lubits.ch 10-part series, gau-nernst's FA for 5090, Stephen Diehl's line-by-line walkthrough
- **Qwen3 Architecture**: Standard transformer with GQA, QK-LayerNorm, NeoX RoPE, SwiGLU — architecture details in llama.cpp's Qwen3 support

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  GRPO Training Loop                  │
│                                                      │
│  ┌─────────────┐    ┌──────────┐    ┌─────────────┐ │
│  │ Inferencer   │───▶│ Reward   │───▶│ Policy      │ │
│  │ (Generation) │    │ Compute  │    │ Update      │ │
│  └─────────────┘    └──────────┘    └─────────────┘ │
│        │                                     │       │
│        ▼                                     ▼       │
│  ┌─────────────┐                    ┌─────────────┐  │
│  │ FA2 Kernel  │                    │ AdamW Opt   │  │
│  │ KV Cache    │                    │ Grad Accum  │  │
│  │ Sampling    │                    │ KL Penalty  │  │
│  └─────────────┘                    └─────────────┘  │
└─────────────────────────────────────────────────────┘
```

### GRPO-Specific Components

Unlike standard pretraining (llm.c), GRPO requires:

1. **Generation/Rollout Engine**: Sample G completions per prompt (autoregressive inference with KV cache)
2. **Reward Computation**: Score each completion, compute group-normalized advantages
3. **Policy Gradient**: Clipped surrogate loss with group-relative advantages
4. **KL Divergence**: Between current policy and reference policy (frozen copy)
5. **Reference Model**: Frozen copy of initial weights for KL computation

### Qwen3-4B Architecture Details

```
- Layers: 36
- Hidden dim: 2560
- Attention heads: 32 (Q) / 8 (KV) — Grouped Query Attention
- Head dim: 80
- FFN intermediate: 9728 (SwiGLU)
- Vocab: 151,936
- Context: 32,768 native
- QK-LayerNorm: Yes (per-head RMSNorm on Q and K)
- RoPE: NeoX (split-half, not interleaved)
- Weights: ~8GB in BF16
```

---

## Detailed Monthly Roadmap

### Phase 0: Foundation (March 2026)

**Goal**: Master CUDA fundamentals and FA2; establish project skeleton.

#### Week 1-2: CUDA Fundamentals & Project Setup
- [ ] Set up project repo with build system (CMake + CUDA)
- [ ] Study llm.c codebase in depth — understand `train_gpt2.cu` architecture
- [ ] Implement basic CUDA kernels: RMSNorm, SwiGLU, RoPE embedding
- [ ] Write safetensors loader in C++ (parse header JSON, mmap tensors)
- [ ] **Milestone**: Load Qwen3-4B weights into GPU memory

#### Week 3-4: Flash Attention 2
- [ ] Study FA2 algorithm (Tri Dao paper + lubits.ch blog series)
- [ ] Implement naive attention kernel → tiled attention → FA2 forward pass
- [ ] Implement FA2 backward pass (needed for training!)
- [ ] Support GQA (Qwen3 uses 32Q/8KV heads)
- [ ] Benchmark against cuDNN/FlashAttention reference
- [ ] **Milestone**: FA2 forward+backward matching reference within 20%

**Resources**:
- lubits.ch Flash Attention from Scratch (10-part series)
- gau-nernst.github.io FA in CUDA C++ walkthrough
- stephendiehl.com Flash Attention CUDA line-by-line
- GPU MODE Lecture 12 on Flash Attention

---

### Phase 1: Inference Engine (April 2026)

**Goal**: Complete forward pass + autoregressive generation for Qwen3-4B.

#### Week 1-2: Complete Forward Pass
- [ ] Implement all Qwen3 layers in CUDA:
  - RMSNorm (with QK-LayerNorm variant)
  - GQA Attention (with RoPE, using FA2)
  - SwiGLU FFN (gate_proj, up_proj, down_proj)
  - Embedding lookup + output projection (tied weights)
- [ ] Implement BF16 matmul kernels (or use cuBLAS for matmuls initially)
- [ ] Validate: match HuggingFace forward pass output to <1e-3 error
- [ ] **Milestone**: Single forward pass producing correct logits

#### Week 3-4: Autoregressive Generation + KV Cache
- [ ] Implement KV cache management (pre-allocated, per-layer)
- [ ] Implement autoregressive token generation loop
- [ ] Add sampling strategies: temperature, top-p, top-k
- [ ] Implement batched generation (G completions per prompt simultaneously)
- [ ] Profile and optimize: target >100 tok/s on single GPU for Qwen3-4B
- [ ] **Milestone**: Generate coherent text from Qwen3-4B in pure C++/CUDA

---

### Phase 2: Training Infrastructure (May 2026)

**Goal**: Backward pass + optimizer + basic SFT training loop.

#### Week 1-2: Backward Pass & Gradient Computation
- [ ] Implement backward kernels for every forward op:
  - Attention backward (FA2 backward already done in Phase 0)
  - RMSNorm backward
  - SwiGLU backward
  - Linear layer backward (matmul transposed)
  - Embedding backward
  - RoPE backward
- [ ] Implement gradient accumulation across micro-batches
- [ ] Implement gradient checkpointing (recompute activations to save memory)
- [ ] Validate: match PyTorch autograd gradients to <1e-3
- [ ] **Milestone**: Correct gradients for full Qwen3-4B

#### Week 3-4: Optimizer + SFT Training
- [ ] Implement AdamW optimizer in CUDA:
  - FP32 optimizer states (m, v) with BF16 model weights
  - Weight decay, bias correction
  - Fused kernel: update + decay in single pass
- [ ] Implement learning rate scheduler (cosine with warmup)
- [ ] Build basic SFT training loop: load data → forward → loss → backward → step
- [ ] Validate on tiny dataset: loss should decrease matching PyTorch
- [ ] **Milestone**: Successful SFT fine-tuning matching PyTorch training curves

---

### Phase 3: GRPO Implementation (June 2026)

**Goal**: Full GRPO training loop working end-to-end.

#### Week 1-2: GRPO Core Algorithm
- [ ] Implement reference model (frozen weight copy, shared memory where possible)
- [ ] Implement GRPO rollout pipeline:
  ```
  For each prompt:
    Generate G completions using current policy
    Score each with reward function
    Compute group-normalized advantages: A_i = (r_i - mean(r)) / std(r)
  ```
- [ ] Implement per-token log-probability computation (forward pass storing logprobs)
- [ ] Implement KL divergence computation between policy and reference
- [ ] Implement GRPO loss:
  ```
  L = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] + β * KL
  ```
- [ ] **Milestone**: Single GRPO training step producing correct loss

#### Week 3-4: Integration & Reward Functions
- [ ] Build end-to-end GRPO training loop:
  ```
  Loop:
    1. Sample batch of prompts
    2. Generate G completions per prompt (inference engine)
    3. Compute rewards (verifiable: math correctness, format, etc.)
    4. Compute advantages (group-normalized)
    5. Forward pass on completions (get logprobs)
    6. Compute GRPO loss + KL penalty
    7. Backward pass
    8. Optimizer step
    9. (Optional) Update reference model periodically
  ```
- [ ] Implement pluggable reward functions (C++ function pointers / callbacks)
- [ ] Implement basic math reward: parse answer, check correctness
- [ ] Handle variable-length sequences: padding, attention masks
- [ ] **Milestone**: GRPO training loop running end-to-end on math dataset

---

### Phase 4: Optimization & Release (July 2026)

**Goal**: Performance optimization, multi-GPU support, documentation, open-source release.

#### Week 1-2: Performance & Memory Optimization
- [ ] Memory optimization:
  - Gradient checkpointing tuning
  - Mixed precision: BF16 forward/backward, FP32 optimizer states
  - Activation offloading to CPU if needed
- [ ] Compute optimization:
  - Fuse element-wise kernels (RMSNorm + residual, SwiGLU activation)
  - CUDA graphs for inference loop
  - Async memory operations (generation ↔ training overlap)
- [ ] Implement "GRPO++ tricks" from latest research:
  - Clip-higher (asymmetric clipping)
  - Advantage filtering/clipping
  - Dynamic sampling (skip low-entropy prompts)
- [ ] **Milestone**: Within 2x throughput of veRL/OpenRLHF on equivalent setup

#### Week 3-4: Multi-GPU + Release
- [ ] Multi-GPU support via NCCL:
  - Data parallelism for training
  - Tensor parallelism for large batch generation
- [ ] Write comprehensive documentation + examples
- [ ] Create benchmark suite: compare with veRL, OpenRLHF, TRL
- [ ] Open-source release: proper README, build instructions, examples
- [ ] **Milestone**: Public release with reproducible GRPO training results

---

## Technical Decision Guide

### Build vs. Use Libraries

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| MatMul | Use cuBLAS initially, custom later | cuBLAS is ~optimal; custom matmul is a separate project |
| Flash Attention | Build from scratch | Core learning goal; needed for both fwd + bwd |
| RMSNorm / SwiGLU | Build from scratch | Simple kernels, good CUDA practice |
| RoPE | Build from scratch | Qwen3 uses specific NeoX variant |
| AdamW | Build from scratch | Single fused kernel, educational |
| NCCL (multi-GPU) | Use library | Don't reinvent collective comms |
| Safetensors parser | Build from scratch | Simple format: JSON header + raw bytes |

### Memory Budget (Single A100 80GB)

```
Model weights (BF16):           ~8 GB
Reference model (BF16):         ~8 GB
Optimizer states (FP32 m,v):   ~16 GB
Gradients (BF16):               ~8 GB
KV Cache (G=8, seq=2048):      ~4 GB
Activations + buffers:         ~20 GB
──────────────────────────────────────
Total:                         ~64 GB  ✓ Fits A100-80GB
```

For smaller GPUs (24GB), you'll need:
- Gradient checkpointing (aggressive)
- LoRA instead of full fine-tuning (implement LoRA adapters)
- Smaller group size G (4 instead of 8)
- Shorter sequences

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| FA2 backward pass complexity | High | Start with naive backward, optimize incrementally. Can fallback to cuDNN |
| Numerical precision mismatches | Medium | Validate every kernel against PyTorch; use FP32 reference |
| GRPO training instability | Medium | Start with small model (Qwen3-0.6B) for debugging, scale up |
| Memory pressure on consumer GPU | Medium | Implement gradient checkpointing + LoRA early |
| Scope creep | High | Hard cutoff: single-GPU first, multi-GPU is stretch goal |

---

## Suggested Starting Point

**Day 1**: Clone llm.c, read `train_gpt2.cu` end-to-end. Then start your project:

```bash
mkdir grpo-cuda && cd grpo-cuda
# Structure:
# src/
#   kernels/     # CUDA kernels: attention, rmsnorm, swiglu, rope, adamw
#   model/       # Qwen3 model definition, weight loading
#   inference/   # Autoregressive generation, KV cache, sampling
#   training/    # Backward pass, gradient computation
#   grpo/        # GRPO algorithm: rollout, advantage, policy loss
# include/       # Headers
# tests/         # Validation against PyTorch reference
# scripts/       # Python scripts for data prep, validation helpers
```

Start with the smallest possible end-to-end: load Qwen3-0.6B (tiny), forward one token, verify output matches HuggingFace. Build up from there.