# Agentic RL.cu — Pivot Plan

**Ultra-Fast Agentic RL in Pure CUDA/C++**

From single-turn LLM-reasoning GRPO to a multi-turn, execution-grounded agentic RL engine — same unified inference+training process, zero weight transfer, now with cross-turn KV reuse and 2-GPU distributed support.

---

## 1. Positioning

RL.cu today is a GRPO engine on Qwen3-0.6B with a unified inference+training design (no weight transfer, zero train-infer mismatch). The pivot keeps that core intact and reframes it around **agentic RL**, which is the current trend, while leaning on the one property that gets *more* valuable in a multi-turn setting:

> **The wedge:** multi-turn rollouts amplify any train-inference drift. RL.cu's same-weights, same-kernels, same-process design eliminates that drift by construction — exactly where TRL+vLLM has to fight it.

Headline claim to build toward: *a multi-turn agentic RL engine where the same process does (TP-sharded) rollouts and training, with KV reuse across turns and zero weight transfer.*

---

## 2. Environment: Competitive Programming (execution-verified, multi-turn)

### Why CP
- **Executably verifiable reward** — run generated code against test cases; pass/fail is the reward. No reward model, no judge LLM, deterministic. Same philosophy as the existing boxed-answer matcher, upgraded to a sandbox.
- **CPU-cheap, in-process** — the env is a sandboxed subprocess; no GPU contention with rollout/training.

### Multi-turn design (the version that earns the name "agentic")
A turn = one code block emitted by the model. Loop:

```
model writes code → env runs it on PUBLIC/sample tests → returns stdout/stderr/diff
  → model sees feedback as observation tokens → revises → runs again
  → ... until pass or turn-budget exhausted
episode reward computed on HIDDEN tests
```

- **Public-feedback / hidden-reward split** — the model only sees sample-test output during the episode; final reward uses hidden tests. Mirrors real CP and prevents overfitting to visible cases.
- **Cross-turn KV reuse** — across turns, prefill only the *new* observation, not the full history. The existing paged KV cache already handles prefix reuse; this becomes a centerpiece of the pitch.

> Decision (locked): **multi-turn with execution feedback**, not single-turn-with-verifier. Single-turn is structurally identical to the current math loop and does not justify the "agentic" framing.

### Dataset
Candidates to evaluate: **CodeContests** (DeepMind), **TACO**, recent code-RL sets. Selection criteria:
- Reliable hidden test suites.
- Difficulty range the 1.7B/4B can *sometimes-but-not-always* solve — reward variance is required for GRPO (all-pass or all-fail groups give zero advantage signal).

### Reward signal — open decision
- **All-or-nothing on hidden tests** — clean, ungameable, but sparse.
- **Fraction-of-tests-passed** — denser gradient, but partial-credit shaping can be gamed.
- Sparsity mitigations regardless of choice: **curriculum** (start easy), **oversample solvable problems**.

> Open item: pick reward shaping before writing the trainer changes — it materially affects whether the reward curve moves.

### Sandbox (operational risk — do not under-budget)
Executing model-generated code. Requirements:
- Hard isolation: container / seccomp / firejail.
- **No network.** CPU + memory + wall-time limits. Killed on timeout.
- Do **not** hand-roll loosely; a runaway or malicious generation can wedge the training run. This is boring but it is where these projects break.

---

## 3. Model Scaling: 1.7B → 4B

| Model | Training state (params + AdamW FP32 moments) | Single-GPU? | Role |
|-------|----------------------------------------------|-------------|------|
| 0.6B  | trivial | yes | current baseline |
| 1.7B  | ~14 GB | yes (comfortable) | **v0.1 flagship** |
| 4B    | ~32 GB before activations/KV | tight→infeasible w/ real KV + multi-turn | **v0.2 flagship, gated on TP** |

- 1.7B trains comfortably single-GPU → validate the agentic loop and env here first.
- 4B's reasoning is meaningfully better for CP and makes the demo convincing, but its training state does not coexist with a real KV cache + longer multi-turn sequences on one card.
- **Do not** debug the multi-turn rollout loop and a barely-fitting 4B simultaneously — you won't know which thing is OOMing.

---

## 4. Distributed: 2 GPUs

With exactly 2 GPUs, TP=2 and DP=2 cannot run simultaneously (that needs 4). Two operating modes, one shared NCCL/process plumbing:

| Mode | Config | Purpose | When |
|------|--------|---------|------|
| **Throughput** | **DP=2** | double rollouts; gradient all-reduce only | 1.7B (fits on one card) |
| **Capacity** | **TP=2 + SP** | fit the bigger model | 4B |

### DP=2 (easy)
- Replicate model; each GPU runs independent rollouts; one gradient all-reduce per step (`ncclAllReduce`).
- Agentic RL is **rollout-bound**, so DP's rollout parallelism is a direct, high-value throughput win.
- Also serves to validate the multi-process / NCCL plumbing cheaply.

### TP=2 + SP (the hard systems chunk)
- **TP=2 (Megatron-style):** shard QKV/out and gate-up/down GEMMs column/row-wise. All-reduce after attention out-proj and after MLP down-proj, **forward and backward**, via NCCL. The existing **fused QKV and fused gate-up projections make the sharding cleaner**.
- **SP (Megatron sequence parallelism):** shard the replicated RMSNorm/residual regions along the sequence dimension. Converts the TP all-reduces into **reduce-scatter (entering) + all-gather (exiting)** pairs — same comms volume, but each rank holds only 1/TP of the activations in those regions. Roughly free given comms is already paid; composes with the existing fused-norm kernels.
  - Caveat: the **backward pass for the reduce-scatter/all-gather swap is the fiddly part** — budget debugging time.
- This is the single biggest code chunk of the pivot — likely more than the agentic loop itself — and also the most impressive systems artifact.

> Scope guard: SP here means **Megatron sequence parallelism** (activation-memory partner to TP). NOT Ring/context parallelism for ultra-long sequences — that's a different, larger project and is not needed (CP + multi-turn histories aren't 100K+ tokens).

---

## 5. Sequencing

### v0.1 — "Agentic RL.cu" (shippable on its own)
1. **Multi-turn rollout loop + CP env + sandbox** on **1.7B single-GPU.**
   - Where the current trainer breaks: the "generate G completions, score once" assumption. Need: generate → parse code → exec in env → feed observation back → repeat → reward at end.
   - Deliver cross-turn KV reuse on top of the existing paged cache.
2. **DP=2** — gradient all-reduce. Throughput win + validates multi-process plumbing. Pairs perfectly with rollout-bound agentic training.

### v0.2 — Capacity
3. **TP=2 + SP** — unlocks **4B**. The hard NCCL forward/backward work.

> Critical guard: **do not let TP+SP block the announcement.** Steps 1–2 are a complete, nameable "Ultra-Fast Agentic RL.cu." TP+SP+4B is the impressive follow-up.

The agentic rollout loop (step 1) is orthogonal to the distributed work and can proceed in parallel with step 2's plumbing.

---

## 6. Open Decisions (resolve before coding)

1. **Reward shaping:** all-or-nothing (hidden tests) vs fraction-of-tests-passed. Affects GRPO gradient signal substantially.
2. **Dataset:** CodeContests vs TACO vs code-RL sets — gated on hidden-test reliability + difficulty distribution matching model capability.
3. **Turn budget** per episode and **public/hidden test split** policy.
4. **Sandbox stack:** container vs seccomp vs firejail.

---

## 7. Immediate Next Step

Spec the multi-turn rollout loop against the current `GRPO_trainer.h`: identify exactly where the single-shot generation assumption breaks, and define the turn / observation / KV-reuse interface.