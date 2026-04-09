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
- [x] Align with vLLM (Aim for 90% performance for Qwen3 0.6B, 4B)
  - [x] Prefill Kernel 6ms forward
  - [x] Decode Kernel Optimize
- [ ] 100% match with vllm or outperform it
  - [ ] Flash-Decoding Split-KV

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
- [x] Trainer
  - [x] Implement gradient accumulation across micro-batches
  - [x] Logging and training curve
  - [x] Build basic SFT training loop: load data → forward → loss → backward → step
  - [x] Validate on tiny dataset: loss should decrease matching PyTorch

### GRPO Core Algorithm
- [x] Implement basic math reward: parse answer, check correctness
- [x] Implement GRPO rollout pipeline (connect with inference engine)
- [x] Implement per-token log-probability computation (forward pass storing logprobs)
- [x] Implement KL divergence computation between policy and reference
- [x] Implement GRPO loss
- [x] Implement basic math reward: parse answer, check correctness
- [x] Single GRPO training step producing correct loss
- [ ] Pipeline validation & Optimization
  - [ ] Improve memory efficiency (vllm wakeup & sleep, gradient checkpoints)

#### Integration
- [x] Build end-to-end GRPO training pipeline
- [ ] Train qwen3-0.6B on deepmath-103k
- [ ] Writing documents and interface
- [ ] Release on Github (in May hopely)
- [ ] Benchmark with TRL (w. vLLM)

### Future: Performance optimization, multi-GPU support.
