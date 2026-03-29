# RL.cu: Pure C++/CUDA LLM Reinforcement Learning Training

## Project Vision

Build an open-source, pure C++/CUDA framework for LLM reinforcement learning training — specifically GRPO (Group Relative Policy Optimization) — targeting Qwen3-4B, with **zero Python/PyTorch dependency** at runtime.

## Features
1. nano-vLLM like inference engine written by CUDA/C++
2. pure C++/CUDA training infra for SFT / RL
3. optimized inference CUDA kernel
4. Ene-to-end RL training in CUDA/C++

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
│  │ Paged KV    │                    │ Grad Accum  │  │
│  │ Sampling    │                    │ KL Penalty  │  │
│  └─────────────┘                    └─────────────┘  │
└─────────────────────────────────────────────────────┘
```
---

## Roadmap

#### Fundamental Kernels & Project Setup
- [x] Set up project repo with build system (CMake + CUDA)
- [x] Implement basic CUDA kernels: RMSNorm, SwiGLU, RoPE embedding, Softmax, linear layer (cublas wrapper)
- [x] Implement FA2 kernel + PagedKV
- [x] Write safetensors loader in C++ (parse header JSON, mmap tensors)
- [x] Run Qwen3-0.6B forward: Prefill + Decode
- [x] Sampler

### vLLM Style reproduce
- [x] Reference to nano vllm
- [x] Model runner (allocate correct KV cache size, batch forward)
- [x] Scheduler (schedule input request)
- [x] LLM Engine
- [ ] Align with vLLM (Aim for 90% performance for Qwen3 0.6B, 4B)

### Training
- [x] Dataset loading
- [x] Implement backward kernels for every forward op:
  - Attention backward
  - RMSNorm backward
  - SwiGLU backward
  - Linear layer backward
  - Embedding backward
  - RoPE backward
- [x] Implement AdamW optimizer in CUDA:
  - FP32 optimizer states (m, v) with FP16 model weights
  - Weight decay, bias correction
  - Fused kernel: update + decay in single pass
  - Flat buffer pattern for contiguous memory access
- [x] Implement learning rate scheduler (cosine with warmup)
- [ ] Trainer
  - [ ] Implement gradient accumulation across micro-batches
  - [ ] Logging and training curve
  - [ ] Build basic SFT training loop: load data → forward → loss → backward → step
  - [ ] Validate on tiny dataset: loss should decrease matching PyTorch
- [ ] Successful SFT fine-tuning matching PyTorch training curves

### GRPO Core Algorithm
- [ ] Implement reference model (frozen weight copy, shared memory where possible)
- [ ] Implement basic math reward: parse answer, check correctness
- [ ] Implement GRPO rollout pipeline (connect with inference engine)
- [ ] Implement per-token log-probability computation (forward pass storing logprobs)
- [ ] Implement KL divergence computation between policy and reference
- [ ] Implement GRPO loss
- [ ] Implement basic math reward: parse answer, check correctness
- [ ] **Milestone**: Single GRPO training step producing correct loss

#### Integration
- [ ] Build end-to-end GRPO training pipeline
- [ ] Train qwen3-4B on dapo-math-17k & test on AIME/24,25
- [ ] Writing documents and interface
- [ ] Release on Github (in May hopely)
- [ ] Benchmark with Slime & VeRL

### Future: Performance optimization, multi-GPU support.
