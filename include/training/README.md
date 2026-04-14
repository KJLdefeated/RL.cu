# include/training/ — Training Infrastructure

SFT and GRPO training loops, optimizer, data loading, and logging. All training runs on the same `Qwen3Model` used for inference — no PyTorch dependency.

---

## Architecture

```
                          ┌──────────────┐
                          │   Trainer    │  (base class)
                          │              │
                          │ • gradient accumulation
                          │ • gradient clipping
                          │ • AdamW optimizer step
                          │ • LR scheduling
                          │ • checkpoint save
                          └──────┬───────┘
                                 │
                   ┌─────────────┴─────────────┐
                   │                           │
            ┌──────▼──────┐             ┌──────▼──────┐
            │ SFTTrainer  │             │ GRPOTrainer │
            │             │             │             │
            │ masked NLL  │             │ rollout gen │
            │ loss        │             │ reward fn   │
            └─────────────┘             │ advantage   │
                                        │ KL penalty  │
                                        └─────────────┘
```

---

## Trainer (trainer.h) — Base Class

Provides the shared training loop infrastructure. Subclasses override `compute_loss()`.

### Training Step Flow

```
training_step(global_batch):
  zero_gradients()
  for acc in 0..grad_accum_steps:
      micro_batch = global_batch[acc * B : (acc+1) * B]
      log_probs = qwen3_forward(tokens, targets)      # forward pass
      loss = compute_loss(log_probs, loss_mask)        # subclass-defined
      scale_grad(d_loss_grad, 1/grad_accum_steps)      # normalize
      qwen3_backward(d_loss_grad)                       # accumulate grads
  clip_grad_norm(max_grad_norm)
  optimizer.step()
```

### Key Components

| Component | Description |
|-----------|-------------|
| **TrainingConfig** | Hyperparameters: batch_size, grad_accum_steps, max_seq_len, base_lr, weight_decay, max_grad_norm, warmup_steps, etc. |
| **Gradient clipping** | Global L2 norm clipping across both FP16 (projection) and FP32 (norm) gradient pools. Uses warp-shuffle + atomicAdd reduction. |
| **Gradient accumulation** | Micro-batches accumulate into a single gradient buffer. Loss gradient is scaled by `1/grad_accum_steps` before backward. |
| **Embedding freeze** | Optional: zeros the embedding gradient before the optimizer step. |

---

## SFTTrainer (SFT_trainer.h)

Supervised fine-tuning with masked negative log-likelihood loss.

### Loss

```
L = -sum(log_probs[t] * mask[t]) / sum(mask[t])
```

Where `mask[t] = 1` for completion tokens (positions the model should learn to predict) and `mask[t] = 0` for prompt tokens and padding.

### Implementation

Two GPU kernels per micro-batch:
1. **nll_loss_reduce_kernel**: Atomic sum of `-log_probs * mask` and mask count.
2. **nll_loss_grad_kernel**: Writes gradient `= -1/N` for masked tokens, 0 elsewhere.

---

## GRPOTrainer (GRPO_trainer.h)

Group Relative Policy Optimization — on-policy RL for language models.

### Algorithm (per step)

```
1. Wakeup engine    — re-allocate KV cache for inference
2. Generate         — G completions per prompt via LLMEngine
3. Sleep engine     — free KV cache for training memory
4. Score            — reward_fn(completion, answer) per completion
5. Advantages       — group-normalize within each prompt's G completions
6. Forward/backward — GRPO loss with optional KL penalty
7. Optimizer step
```

### GRPO Loss

Per active token `t` in sequence `b`:

```
L[t] = -advantage[b] * log_p_new[t]  +  beta * KL[t]
```

**KL estimator** (Schulman, unbiased, non-negative):
```
delta  = log_p_ref[t] - log_p_new[t]
KL[t]  = exp(delta) - delta - 1
```

The gradient on `log_p_new[t]`:
```
dL/d(log_p_new[t]) = (-advantage[b] + beta * (1 - exp(delta))) / N_active
```

Where `N_active` is the total active token count across the global batch (DAPO-style normalization).

### Group-Normalized Advantages

For each prompt group of G completions:
```
advantage[i] = (reward[i] - mean(rewards)) / (std(rewards) + eps)
```

### Features

| Feature | Description |
|---------|-------------|
| **KL penalty** | Optional `kl_beta`-weighted KL divergence to frozen reference weights (snapshot at init). Prevents policy collapse. |
| **Dynamic KL targeting** | Proportional log-space controller: adjusts `kl_beta` each step to keep per-token KL near `kl_target`. Rate-limited by `kl_horizon`. Clamped to `[target*0.1, target*2]`. |
| **Overlong filtering** (DAPO) | Completions that hit `max_completion_len` get loss_mask zeroed — they likely failed to produce an answer and would dilute the gradient signal. |
| **Sleep/wakeup** | Shares GPU memory between inference (KV cache) and training (optimizer states, gradients, activations) by releasing and re-allocating the KV cache each step. |
| **Reference forward** | When KL is enabled, runs a second forward pass with frozen initial weights. Weight snapshot is a D2D copy of both pools (~1.2 GB for 0.6B). Weight pointers are relocated via offset arithmetic. |

### Dataset Format

JSONL with `prompt` and `answer` fields:
```json
{"prompt": "Solve: What is 2+2?", "answer": "4"}
```

The reward function is user-provided (passed as `std::function<float(completion, answer)>`).

### GRPOConfig (extends TrainingConfig)

| Field | Default | Description |
|-------|---------|-------------|
| `num_generations` | 4 | Completions per prompt (G) |
| `max_completion_len` | 256 | Max tokens per completion |
| `gen_temperature` | 1.0 | Sampling temperature for rollouts |
| `kl_beta` | 0.0 | KL penalty weight (0 = disabled) |
| `kl_target` | 0.0 | Dynamic KL target (0 = disabled) |
| `kl_horizon` | 0.05 | Per-step max adaptation rate |
| `filter_overlong` | true | Zero loss_mask for max-length completions |

---

## Supporting Modules

| File | Purpose |
|------|---------|
| `optimizer.h` | `AdamWOptimizer` — flat-buffer AdamW. FP32 master weights + FP32 m/v states. Two kernel launches per step (FP16 projections with weight decay, FP32 norms without). |
| `lr_scheduler.h` | `LRScheduler` — cosine with warmup or constant with warmup. `get_lr(step)` returns the current learning rate. |
| `dataloader.h` | `DataLoader` — reads binary RLDT format (header + offsets + prompt_lens + token IDs). Supports shuffling and epoch reset. |
| `logger.h` | `Logger` — accumulates per-step metrics, flushes to `train_log.jsonl` every N steps. |
