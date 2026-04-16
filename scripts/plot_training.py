#!/usr/bin/env python3
"""Plot GRPO training curves from train_log.jsonl. Supports comparing two runs."""

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

def extract(rows):
    return {
        "steps":       [r["step"] for r in rows],
        "rewards":     [r["mean_reward"] for r in rows],
        "losses":      [r["loss"] for r in rows],
        "grad_norms":  [r["grad_norm"] for r in rows],
        "comp_tokens": [r["comp_tokens"] for r in rows],
        "step_ms":     [r["step_ms"] for r in rows],
        "lrs":         [r["lr"] for r in rows],
        "kls":         [r.get("kl", 0.0) for r in rows],
        "frac_correct":[r.get("frac_correct", 0.0) for r in rows],
        "frac_overlong":[r.get("frac_overlong", 0.0) for r in rows],
    }

def plot_metric(ax, datasets, key, ylabel, title, w, colors, labels, log_scale=False):
    for i, (d, color, label) in enumerate(zip(datasets, colors, labels)):
        if i >= 500:
            break
        steps = d["steps"]
        vals = d[key]
        ax.plot(steps, vals, alpha=0.15, color=color, linewidth=0.5)
        if len(vals) >= w:
            s = smooth(vals, w)
            ax.plot(steps[w-1:], s, color=color, linewidth=1.5, label=f'{label} smooth({w})')
        else:
            ax.plot(steps, vals, color=color, linewidth=1.0, label=label)
    if log_scale:
        ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", nargs='+', required=True, help="One or two log files")
    parser.add_argument("--label", nargs='+', default=None, help="Labels for each log")
    parser.add_argument("--smooth", type=int, default=20, help="Smoothing window")
    parser.add_argument("--max-steps", type=int, default=0, help="Only plot first N steps (0=all)")
    parser.add_argument("--metrics", nargs='+', default=None,
                        help="Which metrics to plot (e.g. --metrics reward step_time). "
                             "Options: reward, loss, grad_norm, kl, frac_correct, "
                             "step_time, lr, frac_overlong, comp_tokens")
    parser.add_argument("--output", default="training_curves.png")
    args = parser.parse_args()

    all_rows = [load_log(p) for p in args.log]
    if args.max_steps > 0:
        all_rows = [[r for r in rows if r["step"] <= args.max_steps] for rows in all_rows]
    datasets = [extract(rows) for rows in all_rows]

    if args.label:
        labels = args.label
    else:
        labels = [p.split('/')[-2] if '/' in p else f'run{i}' for i, p in enumerate(args.log)]

    colors_list = [('C0', 'C1'), ('C3', 'C4')]  # pairs per run
    run_colors = ['C0', 'C3'] if len(datasets) == 2 else ['C0']

    has_kl = any(any(k > 0 for k in d["kls"]) for d in datasets)
    has_frac = any(any(f > 0 for f in d["frac_correct"]) for d in datasets)

    # Precompute step_s for all datasets
    for d in datasets:
        d["step_s"] = [t / 1000.0 for t in d["step_ms"]]

    w = args.smooth
    comparing = len(datasets) > 1

    # Metric definitions: (key, ylabel, title)
    metric_defs = {
        "reward":       ("rewards",      "Mean Reward",   "Mean Reward"),
        "loss":         ("losses",       "Loss",          "GRPO Loss"),
        "grad_norm":    ("grad_norms",   "Grad Norm",     "Gradient Norm"),
        "kl":           ("kls",          "KL (per token)", "KL to Reference"),
        "frac_correct": ("frac_correct", "Frac Correct",  "Fraction Correct"),
        "step_time":    ("step_s",       "Time (s)",      "Step Time"),
        "lr":           ("lrs",          "LR",            "Learning Rate"),
        "frac_overlong":("frac_overlong","Frac Overlong",  "Fraction Overlong"),
        "comp_tokens":  ("comp_tokens",  "Tokens",        "Completion Tokens / Step"),
    }

    if args.metrics:
        selected = args.metrics
    else:
        selected = ["reward", "loss", "grad_norm"]
        if has_kl:
            selected.append("kl")
        if has_frac:
            selected += ["frac_correct", "frac_overlong"]
        else:
            selected.append("comp_tokens")
        selected += ["step_time", "lr"]

    n = len(selected)
    n_cols = min(n, 4)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    title = "GRPO Training Comparison" if comparing else f"GRPO Training ({len(all_rows[0])} steps)"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for idx, metric_name in enumerate(selected):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        key, ylabel, mtitle = metric_defs[metric_name]
        if metric_name == "lr":
            for d, color, label in zip(datasets, run_colors, labels):
                ax.plot(d["steps"], d[key], color=color, linewidth=1.5, label=label)
            ax.set_xlabel('Step'); ax.set_ylabel(ylabel); ax.set_title(mtitle)
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        else:
            plot_metric(ax, datasets, key, ylabel, mtitle, w, run_colors, labels)

    # Hide unused subplot slots
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.output}")

    # Print summary stats for each run
    for label, rows, d in zip(labels, all_rows, datasets):
        print(f"\n{'='*50}")
        print(f"Run: {label}  ({len(rows)} steps)")
        print(f"Mean Reward:  {np.mean(d['rewards']):.3f}  (last 50: {np.mean(d['rewards'][-50:]):.3f})")
        print(f"Loss:         {np.mean(d['losses']):.4f}  (last 50: {np.mean(d['losses'][-50:]):.4f})")
        print(f"Grad Norm:    {np.mean(d['grad_norms']):.3f}  (last 50: {np.mean(d['grad_norms'][-50:]):.3f})")
        print(f"Step Time:    {np.mean(d['step_s']):.1f}s  (total: {sum(d['step_s'])/3600:.1f}h)")
        print(f"Comp Tokens:  {np.mean(d['comp_tokens']):.0f}/step  (total: {sum(d['comp_tokens'])/1e6:.1f}M)")
        if any(k > 0 for k in d['kls']):
            print(f"KL:           {np.mean(d['kls']):.5f}  (last 50: {np.mean(d['kls'][-50:]):.5f})")
        if any(f > 0 for f in d['frac_correct']):
            print(f"Frac Correct: {np.mean(d['frac_correct']):.3f}  (last 50: {np.mean(d['frac_correct'][-50:]):.3f})")


if __name__ == "__main__":
    main()
