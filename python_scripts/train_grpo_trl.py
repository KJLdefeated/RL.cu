#!/usr/bin/env python3
"""
train_grpo_trl.py — GRPO training with TRL + vLLM (reference implementation)

Mirrors the exact training settings from our CUDA GRPO trainer (tests/train_grpo.cu)
for validation and benchmarking.

Usage:
  # Single GPU, default settings (matches CUDA trainer)
  python python_scripts/train_grpo_trl.py

  # Custom settings
  python python_scripts/train_grpo_trl.py \
      --model Qwen/Qwen3-0.6B \
      --dataset trl-lib/DeepMath-103K \
      --num-steps 100

Dependencies:
  pip install trl transformers datasets vllm accelerate
"""

import argparse
import re
import time

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ─── System prompt (must match prepare_data.py / GRPO_trainer.h) ─────────────

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step using thinking mode. "
    "Use <think>...<think/> tags for your reasoning steps. "
)


# ─── Reward function (mirrors tests/train_grpo.cu) ──────────────────────────

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
            # skip whitespace
            while start < len(text) and text[start] == ' ':
                start += 1
            end = text.find('\n', start)
            if end == -1:
                end = len(text)
            return text[start:end].rstrip()
    return ""


def strip_latex_wrappers(s: str) -> str:
    """Strip LaTeX wrapper commands: \\text{Yes} -> Yes."""
    cmds = ["\\text{", "\\textbf{", "\\textit{", "\\mathrm{",
            "\\mathbf{", "\\mathit{", "\\operatorname{"]
    for cmd in cmds:
        while s.startswith(cmd) and s.endswith('}'):
            s = s[len(cmd):-1]
    return s


def normalize_answer(s: str) -> str:
    """Normalize a math answer for comparison."""
    s = strip_latex_wrappers(s)
    # Strip whitespace and $
    r = ''.join(c for c in s if not c.isspace() and c != '$')
    # Strip wrapping parens
    if len(r) >= 2 and r[0] == '(' and r[-1] == ')':
        r = r[1:-1]
    # Strip trailing period
    if r.endswith('.'):
        r = r[:-1]
    return r


def extract_predicted_answer(completion: str) -> str:
    """Extract answer using \\boxed{} or Answer: line."""
    ans = extract_boxed(completion)
    if ans:
        return ans
    ans = extract_answer_line(completion)
    if ans:
        boxed = extract_boxed(ans)
        if boxed:
            return boxed
    return ans


def boxed_reward_fn(completions, solution, **kwargs) -> list[float]:
    """Batch reward function for TRL GRPOTrainer.
    Returns 1.0 if extracted answer matches ground truth, 0.0 otherwise."""
    rewards = []
    for completion, answer in zip(completions, solution):
        # TRL passes chat-format completions as list of message dicts
        if isinstance(completion, list):
            completion = " ".join(m.get("content", "") for m in completion if m.get("content"))
        pred = extract_predicted_answer(completion)
        if not pred:
            rewards.append(0.0)
        elif normalize_answer(pred) == normalize_answer(answer):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# ─── Dataset formatting ─────────────────────────────────────────────────────

def build_prompt(messages: list[dict]) -> list[dict]:
    """Build chat messages with system prompt for the model."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages,
    ]


def format_dataset(example):
    """Format dataset example for TRL GRPOTrainer."""
    example["prompt"] = build_prompt(example["prompt"])
    # Rename 'solution' to keep it accessible in reward function
    return example


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO training with TRL + vLLM")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="Model name or path")
    parser.add_argument("--dataset", default="trl-lib/DeepMath-103K",
                        help="HuggingFace dataset name")
    parser.add_argument("--num-steps", type=int, default=10,
                        help="Total training steps")
    parser.add_argument("--output-dir", default="checkpoints/grpo_trl",
                        help="Output directory")
    parser.add_argument("--no-vllm", action="store_true",
                        help="Disable vLLM (use HF generate instead)")
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"[data] Loading {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    dataset = dataset.map(format_dataset)
    print(f"[data] {len(dataset)} samples")

    # ── Training config (mirrors tests/train_grpo.cu settings) ────────────
    #
    # CUDA trainer settings:
    #   batch_size=8, num_generations=8  →  num_prompts=1 per step
    #   grad_accum_steps=4
    #   max_completion_len=4096, max_seq_len=8192
    #   temperature=0.8, top_p=1.0
    #   lr=1e-6, warmup_steps=1, max_grad_norm=1.0
    #   No KL penalty, no reference model (DAPO-style)

    training_args = GRPOConfig(
        output_dir=args.output_dir,

        # ── Batch / generation ────────────────────────────────────────
        per_device_train_batch_size=1,          # num_prompts per device
        num_generations=8,                      # G completions per prompt
        gradient_accumulation_steps=64,

        # ── Generation params ─────────────────────────────────────────
        max_completion_length=4096,
        max_prompt_length=4096,                 # max_seq_len=8192, split
        temperature=0.8,
        top_p=1.0,
        top_k=0,

        # ── Optimizer ─────────────────────────────────────────────────
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=1,
        max_steps=args.num_steps,

        # ── GRPO-specific ─────────────────────────────────────────────
        beta=0.0,                               # No KL penalty (DAPO)

        # ── Logging ───────────────────────────────────────────────────
        logging_steps=1,
        save_steps=100,
        report_to="none",

        # ── vLLM ──────────────────────────────────────────────────────
        use_vllm=not args.no_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.6,

        # ── Precision ─────────────────────────────────────────────────
        bf16=True,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    print(f"[train] Initializing GRPOTrainer...")
    print(f"  model={args.model}")
    print(f"  num_generations={training_args.num_generations}")
    print(f"  grad_accum={training_args.gradient_accumulation_steps}")
    print(f"  max_completion={training_args.max_completion_length}")
    print(f"  lr={training_args.learning_rate}")
    print(f"  max_steps={training_args.max_steps}")
    print(f"  vllm={training_args.use_vllm}")

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=boxed_reward_fn,
        args=training_args,
        train_dataset=dataset,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"[train] Starting GRPO training for {args.num_steps} steps...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"[train] Done in {elapsed:.1f}s ({elapsed / args.num_steps:.1f}s/step)")


if __name__ == "__main__":
    main()
