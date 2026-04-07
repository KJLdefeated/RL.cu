#!/usr/bin/env python3
"""
bench_sampler.py — Torch sampler vs our CUDA sampler latency comparison.

nano-vllm's sampler (nanovllm/layers/sampler.py) uses the Gumbel-max trick:
    logits/T → softmax → divide by Exponential(1) → argmax
This samples proportionally to softmax(logits/T) without explicit top-k/top-p.
It's @torch.compile'd and runs the full vocab in one fused kernel.

Our CUDA sampler uses:
    single-pass logit scan (max + top-k insertion sort) → CUB BlockRadixSort
    → thread-0 serial: exp(top-k) + top-p nucleus + weighted sample

This script benchmarks both across batch sizes with V=151936 (Qwen3 vocab).

Usage:
    cd /home/kjl0508/RL_cuda
    python python_scripts/bench_sampler.py
"""

import torch
import subprocess, os

VOCAB_SIZE  = 151936
TEMPERATURE = 0.6
TOP_K       = 50


# ──────────────────────────────────────────────────────────────────────────────
# Torch sampler implementations
# ──────────────────────────────────────────────────────────────────────────────

def _gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """nano-vllm style: Gumbel-max trick, no top-k/top-p."""
    scaled = logits.float().div_(temperature)
    probs  = torch.softmax(scaled, dim=-1)
    return probs.div_(
        torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    ).argmax(dim=-1)


def _topk_sample(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    """Explicit top-k + multinomial: comparable semantics to our CUDA sampler."""
    scaled = logits.float() / temperature
    topk_vals, topk_idx = torch.topk(scaled, top_k, dim=-1)
    probs = torch.softmax(topk_vals, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return topk_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)


# Compiled versions (like nano-vllm's @torch.compile)
_gumbel_compiled = torch.compile(_gumbel_sample)
_topk_compiled   = torch.compile(_topk_sample)


# ──────────────────────────────────────────────────────────────────────────────
# Timing helper
# ──────────────────────────────────────────────────────────────────────────────

def bench_us(fn, warmup: int = 30, iters: int = 300) -> float:
    """Return average latency in µs using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1_000 / iters   # ms → µs


# ──────────────────────────────────────────────────────────────────────────────
# Get our CUDA sampler numbers by running test_sampler
# ──────────────────────────────────────────────────────────────────────────────

def get_our_numbers() -> dict:
    """Run make test_sampler, parse benchmark lines, return {label: µs}."""
    root = os.path.join(os.path.dirname(__file__), "..")
    try:
        out = subprocess.check_output(
            ["make", "-s", "test_sampler"],
            cwd=root, stderr=subprocess.STDOUT, text=True, timeout=120
        )
    except Exception as e:
        print(f"  [warn] could not run test_sampler: {e}")
        return {}

    import re
    nums = {}
    for line in out.splitlines():
        if "[BENCH]" not in line:
            continue
        # e.g. "[BENCH] top-k=50 B=16 V=151936   123.98 us/call"
        # Extract the float before "us/call"
        m = re.search(r'([\d.]+)\s+us/call', line)
        if not m:
            continue
        us = float(m.group(1))
        # Extract B= and top-k= values to build a lookup key
        bm = re.search(r'B=(\d+)', line)
        km = re.search(r'top-k=(\d+)', line)
        gm = re.search(r'Greedy', line)
        if not bm:
            continue
        B = int(bm.group(1))
        if gm:
            nums[("greedy", B)] = us
        elif km:
            nums[(f"top-k={km.group(1)}", B)] = us
    return nums


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    device = "cuda"
    print(f"Device     : {torch.cuda.get_device_name()}")
    print(f"Vocab size : {VOCAB_SIZE:,}")
    print(f"Temperature: {TEMPERATURE}   top_k={TOP_K}")
    print()

    # Warm up torch.compile (triggers JIT on first call)
    print("Compiling torch samplers (first call triggers JIT) ...")
    dummy = torch.randn(1, VOCAB_SIZE, device=device, dtype=torch.float16)
    _gumbel_compiled(dummy, TEMPERATURE)
    _topk_compiled(dummy, TEMPERATURE, TOP_K)
    torch.cuda.synchronize()
    print("Done.\n")

    # Fetch our CUDA kernel numbers
    print("Building & running our CUDA sampler benchmarks ...")
    our = get_our_numbers()
    print("Done.\n")

    # ── Header ──
    w = 16
    print(f"{'Batch':>6}  "
          f"{'Torch Gumbel':>{w}}  "
          f"{'Torch top-k=50':>{w}}  "
          f"{'Ours greedy':>{w}}  "
          f"{'Ours top-k=50':>{w}}")
    print("─" * (6 + 4 * (w + 2) + 8))

    batch_sizes = [1, 4, 16, 64, 128, 256]

    for B in batch_sizes:
        logits = torch.randn(B, VOCAB_SIZE, device=device, dtype=torch.float16)

        t_gumbel = bench_us(lambda: _gumbel_compiled(logits, TEMPERATURE))
        t_topk   = bench_us(lambda: _topk_compiled(logits, TEMPERATURE, TOP_K))

        our_greedy = f"{our[('greedy', B)]:.1f} µs" if ("greedy", B) in our else "—"
        our_topk   = f"{our[('top-k=50', B)]:.1f} µs" if ("top-k=50", B) in our else "—"

        print(f"B={B:<4}   "
              f"{t_gumbel:>{w-3}.1f} µs     "
              f"{t_topk:>{w-3}.1f} µs     "
              f"{our_greedy:>{w}}  "
              f"{our_topk:>{w}}")

    print()
    print("Notes")
    print("─" * 60)
    print("Torch Gumbel  : logits/T → softmax → div(Exp(1)) → argmax")
    print("                No top-k/top-p. nano-vllm uses this with")
    print("                @torch.compile. Samples from the full vocab.")
    print()
    print("Torch top-k=50: topk() → softmax → multinomial()")
    print("                Comparable semantics to our CUDA sampler.")
    print()
    print("Ours greedy   : temperature=0, single argmax pass")
    print("Ours top-k=50 : single logit scan (max+topk) + CUB")
    print("                BlockRadixSort + serial exp/sample for top-k")
    print()
    print("Key insight: nano-vllm batches ~128 seqs per sampler call")
    print("(large batches amortize kernel launch). Our scheduler produces")
    print("~4 seqs/call on average — see the B=4 row for a fair comparison.")


if __name__ == "__main__":
    main()
