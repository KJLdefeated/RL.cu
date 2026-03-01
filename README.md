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

---

## Detailed Monthly Roadmap

### Phase 0: Foundation (Feb 2026)

**Goal**: Master CUDA fundamentals and FA2; establish project skeleton.

#### Week 1-2: CUDA Fundamentals & Project Setup
- [x] Set up project repo with build system (CMake + CUDA)
- [x] Implement basic CUDA kernels: RMSNorm, SwiGLU, RoPE embedding, Softmax, linear layer (cublas wrapper)
- [x] Implement FA2 kernel + PagedKV
- [x] Write safetensors loader in C++ (parse header JSON, mmap tensors)
- [x] Run Qwen3-0.6B forward: Prefill + Decode
- [ ] Sampler
- [ ] Benchmark forward speed with huggingface implementation (FA2 + KV Cache)
- [ ] Qwen3-4B forward & benchmark

### Phase 1: Inference Engine

#### 2 Weeks: vLLM Style reproduce
- [ ] Reference to nano vllm
- [ ] Model runner (allocate correct KV cache size, batch forward)
- [ ] Scheduler (schedule input request)
- [ ] LLM Engine
- [ ] Benchmark with vLLM (Aim for 90% performance for Qwen3 0.6B, 4B)

### Phase 2: Training Infrastructure

#### 3 Weeks: SFT Full Finetuning
- [ ] Dataset loading (maybe need apache arrow)
- [ ] Implement backward kernels for every forward op:
  - Attention backward (FA2 backward already done in Phase 0)
  - RMSNorm backward
  - SwiGLU backward
  - Linear layer backward (matmul transposed)
  - Embedding backward
  - RoPE backward
- [ ] Implement gradient accumulation across micro-batches
- [ ] Validate: match PyTorch autograd gradients to <1e-3
- [ ] Implement AdamW optimizer in CUDA:
  - FP32 optimizer states (m, v) with BF16 model weights
  - Weight decay, bias correction
  - Fused kernel: update + decay in single pass
- [ ] Implement learning rate scheduler (cosine with warmup)
- [ ] Build basic SFT training loop: load data → forward → loss → backward → step
- [ ] Validate on tiny dataset: loss should decrease matching PyTorch
- [ ] **Milestone**: Successful SFT fine-tuning matching PyTorch training curves

### Phase 3: GRPO Implementation (April 2026)

**Goal**: Full GRPO training loop working end-to-end.

#### Week 1: GRPO Core Algorithm
- [ ] Implement reference model (frozen weight copy, shared memory where possible)
- [ ] Implement basic math reward: parse answer, check correctness
- [ ] Implement GRPO rollout pipeline (connect with inference engine)
- [ ] Implement per-token log-probability computation (forward pass storing logprobs)
- [ ] Implement KL divergence computation between policy and reference
- [ ] Implement GRPO loss
- [ ] Implement basic math reward: parse answer, check correctness
- [ ] **Milestone**: Single GRPO training step producing correct loss

#### Week 2: Integration
- [ ] Build end-to-end GRPO training pipeline
- [ ] Train qwen3-4B on dapo-math-17k & test on AIME/24,25
- [ ] Writing documents and interface
- [ ] Release on Github (in May hopely)
- [ ] Benchmark with Slime & VeRL


### Future: Performance optimization, multi-GPU support.