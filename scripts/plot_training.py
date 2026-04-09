#!/usr/bin/env python3
"""Plot GRPO training curves from train_log.jsonl."""

import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

def load_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def smooth(values, window=10):
    """Simple moving average."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="checkpoints/grpo_ckpt/train_log.jsonl")
    parser.add_argument("--smooth", type=int, default=20, help="Smoothing window")
    parser.add_argument("--output", default="training_curves.png")
    args = parser.parse_args()

    rows = load_log(args.log)
    if not rows:
        print("No data found", file=sys.stderr)
        sys.exit(1)

    steps       = [r["step"] for r in rows]
    rewards     = [r["mean_reward"] for r in rows]
    losses      = [r["loss"] for r in rows]
    grad_norms  = [r["grad_norm"] for r in rows]
    comp_tokens = [r["comp_tokens"] for r in rows]
    step_ms     = [r["step_ms"] for r in rows]
    lrs         = [r["lr"] for r in rows]

    w = args.smooth

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"GRPO Training ({len(rows)} steps)", fontsize=14, fontweight='bold')

    # 1. Mean Reward
    ax = axes[0, 0]
    ax.plot(steps, rewards, alpha=0.25, color='C0', linewidth=0.5)
    if len(rewards) >= w:
        s = smooth(rewards, w)
        ax.plot(steps[w-1:], s, color='C0', linewidth=1.5, label=f'smooth({w})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Mean Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Loss
    ax = axes[0, 1]
    ax.plot(steps, losses, alpha=0.25, color='C1', linewidth=0.5)
    if len(losses) >= w:
        s = smooth(losses, w)
        ax.plot(steps[w-1:], s, color='C1', linewidth=1.5, label=f'smooth({w})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('GRPO Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Gradient Norm
    ax = axes[0, 2]
    ax.plot(steps, grad_norms, alpha=0.25, color='C2', linewidth=0.5)
    if len(grad_norms) >= w:
        s = smooth(grad_norms, w)
        ax.plot(steps[w-1:], s, color='C2', linewidth=1.5, label=f'smooth({w})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Grad Norm')
    ax.set_title('Gradient Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Completion Tokens
    ax = axes[1, 0]
    ax.plot(steps, comp_tokens, alpha=0.25, color='C3', linewidth=0.5)
    if len(comp_tokens) >= w:
        s = smooth(comp_tokens, w)
        ax.plot(steps[w-1:], s, color='C3', linewidth=1.5, label=f'smooth({w})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Tokens')
    ax.set_title('Completion Tokens / Step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Step Time
    ax = axes[1, 1]
    step_s = [t / 1000.0 for t in step_ms]
    ax.plot(steps, step_s, alpha=0.25, color='C4', linewidth=0.5)
    if len(step_s) >= w:
        s = smooth(step_s, w)
        ax.plot(steps[w-1:], s, color='C4', linewidth=1.5, label=f'smooth({w})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Time (s)')
    ax.set_title('Step Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Learning Rate
    ax = axes[1, 2]
    ax.plot(steps, lrs, color='C5', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('LR')
    ax.set_title('Learning Rate')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.output}")

    # Print summary stats
    print(f"\n{'='*50}")
    print(f"Steps: {len(rows)}")
    print(f"Mean Reward:  {np.mean(rewards):.3f}  (last 50: {np.mean(rewards[-50:]):.3f})")
    print(f"Loss:         {np.mean(losses):.4f}  (last 50: {np.mean(losses[-50:]):.4f})")
    print(f"Grad Norm:    {np.mean(grad_norms):.3f}  (last 50: {np.mean(grad_norms[-50:]):.3f})")
    print(f"Step Time:    {np.mean(step_s):.1f}s  (total: {sum(step_s)/3600:.1f}h)")
    print(f"Comp Tokens:  {np.mean(comp_tokens):.0f}/step  (total: {sum(comp_tokens)/1e6:.1f}M)")


if __name__ == "__main__":
    main()
