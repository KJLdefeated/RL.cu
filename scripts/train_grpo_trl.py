#!/usr/bin/env python3
"""
train_grpo_trl.py — GRPO training with TRL (reference implementation)

Mirrors the CUDA GRPO trainer (tests/train_grpo.cu) for comparison.

Usage (matching CUDA command):
  python scripts/train_grpo_trl.py \
      --max-comp 2048 --max-seq 3072 \
      --kl-beta 0.04 --kl-target 0.04 \
      --temperature 0.7 \
      --reward shaped \
      --total-steps 1000

Dependencies:
  pip install trl transformers datasets accelerate
  # Optional for faster generation:
  pip install vllm
"""

import argparse
import json
import os
import time
from datetime import datetime

from datasets import load_dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer


# ─── Reward helpers (mirrors tests/train_grpo.cu exactly) ─────────────────────

def extract_boxed(text: str) -> str:
    """Extract last \\boxed{...} content, handling nested braces."""
    key = "\\boxed{"
    pos = text.rfind(key)
    if pos == -1:
        return ""
    start = pos + len(key)
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth != 0:
        return ""
    return text[start:i - 1]


def extract_answer_line(text: str) -> str:
    """Extract content after the last 'Answer:' line."""
    for pattern in ["Answer:", "answer:"]:
        pos = text.rfind(pattern)
        if pos != -1:
            start = pos + len(pattern)
            while start < len(text) and text[start] == ' ':
                start += 1
            end = text.find('\n', start)
            if end == -1:
                end = len(text)
            return text[start:end].rstrip()
    return ""


def strip_latex_wrappers(s: str) -> str:
    cmds = ["\\text{", "\\textbf{", "\\textit{", "\\mathrm{",
            "\\mathbf{", "\\mathit{", "\\operatorname{"]
    for cmd in cmds:
        while s.startswith(cmd) and s.endswith('}'):
            s = s[len(cmd):-1]
    return s


def normalize_answer(s: str) -> str:
    s = strip_latex_wrappers(s)
    r = ''.join(c for c in s if not c.isspace() and c != '$')
    if len(r) >= 2 and r[0] == '(' and r[-1] == ')':
        r = r[1:-1]
    if r.endswith('.'):
        r = r[:-1]
    return r


def extract_predicted_answer(completion: str) -> str:
    ans = extract_boxed(completion)
    if ans:
        return ans
    ans = extract_answer_line(completion)
    if ans:
        boxed = extract_boxed(ans)
        if boxed:
            return boxed
    return ans


# ─── Reward functions ─────────────────────────────────────────────────────────

# Shared state for logging extra metrics from the reward function
_reward_stats = {
    "n_correct": 0,
    "n_total": 0,
    "n_overlong": 0,
    "total_comp_len": 0,
}


def _reset_reward_stats():
    _reward_stats["n_correct"] = 0
    _reward_stats["n_total"] = 0
    _reward_stats["n_overlong"] = 0
    _reward_stats["total_comp_len"] = 0


def boxed_reward_fn(completions, answer, **kwargs) -> list[float]:
    """1.0 if correct, 0.0 otherwise."""
    rewards = []
    for comp, ans in zip(completions, answer):
        if isinstance(comp, list):
            comp = " ".join(m.get("content", "") for m in comp if m.get("content"))
        _reward_stats["n_total"] += 1
        _reward_stats["total_comp_len"] += len(comp.split())
        pred = extract_predicted_answer(comp)
        if not pred:
            rewards.append(0.0)
        elif normalize_answer(pred) == normalize_answer(ans):
            rewards.append(1.0)
            _reward_stats["n_correct"] += 1
        else:
            rewards.append(0.0)
    return rewards


def shaped_reward_fn(completions, answer, max_comp_len=0, **kwargs) -> list[float]:
    """1.0 if correct, 0.1 if \\boxed{} present but wrong, 0.0 otherwise."""
    rewards = []
    for comp, ans in zip(completions, answer):
        if isinstance(comp, list):
            comp = " ".join(m.get("content", "") for m in comp if m.get("content"))
        _reward_stats["n_total"] += 1
        _reward_stats["total_comp_len"] += len(comp.split())
        pred = extract_predicted_answer(comp)
        if not pred:
            rewards.append(0.0)
        elif normalize_answer(pred) == normalize_answer(ans):
            rewards.append(1.0)
            _reward_stats["n_correct"] += 1
        else:
            rewards.append(0.1)
    return rewards


# ─── JSONL Logger (matches CUDA trainer's train_log.jsonl format) ─────────────

class JSONLLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w")

    def log(self, row: dict):
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# ─── Custom callback to log metrics matching CUDA trainer ────────────────────

class GRPOMetricsCallback(TrainerCallback):
    """Log per-step metrics in the same format as the CUDA GRPO trainer."""

    def __init__(self, logger: JSONLLogger, kl_beta: float, kl_target: float,
                 kl_horizon: float):
        self.logger = logger
        self.kl_beta = kl_beta
        self.kl_target = kl_target
        self.kl_horizon = kl_horizon
        self.step_start_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
        _reset_reward_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step == 0:
            return

        step_ms = 0.0
        if self.step_start_time is not None:
            step_ms = (time.time() - self.step_start_time) * 1000.0

        # Extract TRL's logged metrics
        loss = logs.get("loss", 0.0)
        kl = logs.get("kl", 0.0)
        grad_norm = logs.get("grad_norm", 0.0)
        lr = logs.get("learning_rate", 0.0)
        mean_reward = logs.get("reward", 0.0)

        # Completion length from TRL
        mean_comp_len = logs.get("completions/mean_length", 0.0)

        # Our custom stats from reward function
        n_total = max(_reward_stats["n_total"], 1)
        frac_correct = _reward_stats["n_correct"] / n_total
        frac_overlong = _reward_stats["n_overlong"] / n_total

        # Dynamic KL targeting (mirrors CUDA trainer's controller)
        if self.kl_target > 0 and self.kl_beta > 0 and kl > 0:
            import math
            ratio = kl / (self.kl_target + 1e-8)
            log_ratio = math.log(max(0.1, min(10.0, ratio)))
            delta = self.kl_horizon * log_ratio
            delta = max(-self.kl_horizon, min(self.kl_horizon, delta))
            self.kl_beta *= math.exp(delta)
            init_beta = self.kl_target
            self.kl_beta = max(init_beta * 0.1, min(init_beta * 2.0, self.kl_beta))

        # Completion tokens this step (approximate from mean_comp_len * num_gens)
        comp_tokens = int(logs.get("completions/mean_length", 0)
                          * _reward_stats["n_total"])

        row = {
            "step": state.global_step,
            "loss": round(loss, 6),
            "mean_reward": round(mean_reward, 4),
            "grad_norm": round(grad_norm, 4),
            "lr": lr,
            "kl": round(kl, 6),
            "kl_beta": round(self.kl_beta, 6),
            "step_ms": round(step_ms, 1),
            "comp_tokens": comp_tokens,
            "frac_correct": round(frac_correct, 4),
            "frac_overlong": round(frac_overlong, 4),
            "mean_comp_len": round(mean_comp_len, 1),
        }

        self.logger.log(row)

        # Print summary line
        print(f"[step {state.global_step:4d}] "
              f"reward={mean_reward:.3f}  loss={loss:.4f}  "
              f"grad={grad_norm:.3f}  kl={kl:.5f}  "
              f"correct={frac_correct:.1%}  "
              f"comp_len={mean_comp_len:.0f}  "
              f"time={step_ms / 1000:.1f}s")


# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_local_jsonl(path: str):
    """Load local JSONL with 'prompt' (chat-templated) and 'answer' fields."""
    dataset = load_dataset("json", data_files=path, split="train")
    return dataset


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO training with TRL")

    # Model / data
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data", default="data/deepmath-103k.jsonl")

    # GRPO
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Total seqs per step (num_prompts * num_gens)")
    parser.add_argument("--num-gens", type=int, default=8)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--max-comp", type=int, default=4096)
    parser.add_argument("--max-seq", type=int, default=5120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--kl-beta", type=float, default=0.04)
    parser.add_argument("--kl-target", type=float, default=0.04)
    parser.add_argument("--kl-horizon", type=float, default=0.05)
    parser.add_argument("--reward", default="shaped", choices=["boxed", "shaped"])

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--total-steps", type=int, default=100)
    parser.add_argument("--schedule", default="constant",
                        choices=["cosine", "constant"])
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Engine
    parser.add_argument("--max-model-len", type=int, default=5120)
    parser.add_argument("--gpu-util", type=float, default=0.9)
    parser.add_argument("--no-vllm", action="store_true")

    # Logging
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--log-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=0)

    args = parser.parse_args()

    # Auto-generate save_dir
    if not args.save_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"checkpoints/grpo_trl_{ts}"

    os.makedirs(args.save_dir, exist_ok=True)
    jsonl_logger = JSONLLogger(os.path.join(args.save_dir, "train_log.jsonl"))

    # Derived
    num_prompts = args.batch_size // args.num_gens
    max_prompt_len = args.max_seq - args.max_comp

    print(f"[config] model={args.model}")
    print(f"[config] data={args.data}")
    print(f"[config] batch_size={args.batch_size}  "
          f"num_prompts={num_prompts}  G={args.num_gens}")
    print(f"[config] max_comp={args.max_comp}  max_seq={args.max_seq}")
    print(f"[config] temperature={args.temperature}  kl_beta={args.kl_beta}  "
          f"kl_target={args.kl_target}")
    print(f"[config] lr={args.lr}  warmup={args.warmup}  "
          f"total_steps={args.total_steps}")
    print(f"[config] reward={args.reward}  save_dir={args.save_dir}")

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"[data] Loading {args.data}...")
    dataset = load_local_jsonl(args.data)
    print(f"[data] {len(dataset)} samples")

    # ── Select reward function ────────────────────────────────────────────
    reward_fn = shaped_reward_fn if args.reward == "shaped" else boxed_reward_fn

    # ── TRL GRPOConfig ────────────────────────────────────────────────────
    # TRL uses per_device_train_batch_size = num_prompts per device.
    # Total seqs per step = per_device_batch_size * grad_accum * num_gens
    # We want: num_prompts * num_gens = batch_size
    # With grad_accum: per_device_batch_size * grad_accum = num_prompts
    per_device_batch = num_prompts
    actual_accum = args.accum

    lr_scheduler_type = ("cosine" if args.schedule == "cosine"
                         else "constant_with_warmup")

    training_args = GRPOConfig(
        output_dir=args.save_dir,

        # Batch / generation
        per_device_train_batch_size=per_device_batch,
        num_generations=args.num_gens,
        gradient_accumulation_steps=actual_accum,

        # Generation params
        max_completion_length=args.max_comp,
        max_prompt_length=max_prompt_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else None,

        # Optimizer
        learning_rate=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.01,
        max_grad_norm=args.grad_clip,
        warmup_steps=args.warmup,
        max_steps=args.total_steps,
        lr_scheduler_type=lr_scheduler_type,

        # GRPO-specific
        beta=args.kl_beta,

        # Logging
        logging_steps=args.log_steps,
        save_steps=args.save_steps if args.save_steps > 0 else args.total_steps,
        report_to="none",

        # vLLM
        use_vllm=not args.no_vllm,
        vllm_mode="colocate" if not args.no_vllm else None,
        vllm_gpu_memory_utilization=args.gpu_util * 0.5,

        # Precision
        bf16=True,
    )

    # ── Metrics callback ──────────────────────────────────────────────────
    metrics_cb = GRPOMetricsCallback(
        logger=jsonl_logger,
        kl_beta=args.kl_beta,
        kl_target=args.kl_target,
        kl_horizon=args.kl_horizon,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    print(f"[train] Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
        callbacks=[metrics_cb],
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"[train] Starting GRPO training for {args.total_steps} steps...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    jsonl_logger.close()

    print(f"\n{'='*60}")
    print(f"[train] Done in {elapsed:.1f}s "
          f"({elapsed / max(args.total_steps, 1):.1f}s/step)")
    print(f"[train] Log saved to {args.save_dir}/train_log.jsonl")
    print(f"[train] Plot with: python scripts/plot_training.py "
          f"--log {args.save_dir}/train_log.jsonl")


if __name__ == "__main__":
    main()
