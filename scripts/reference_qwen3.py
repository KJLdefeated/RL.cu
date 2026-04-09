#!/usr/bin/env python3
"""
reference_qwen3.py — HuggingFace + PyTorch reference for Qwen3-0.6B

Runs the same chat generation as tests/test_qwen3.cu (step-by-step greedy
decode with top-5 logit logging), then profiles decode throughput for both
the HF implementation and the pure-CUDA implementation.

Usage:
  # HF correctness + speed only
  python python_scripts/reference_qwen3.py

  # Also run the CUDA benchmark binary (build first with: make bench_decode)
  python python_scripts/reference_qwen3.py --cuda-bench
"""

import argparse
import os
import re
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.join(SCRIPT_DIR, "..")
MODEL_DIR   = os.path.join(REPO_ROOT, "model_weights", "Qwen3-0.6B")
CUDA_BENCH  = os.path.join(REPO_ROOT, "build", "bench_decode")

USER_MSG       = "What is the capital of France?"
MAX_NEW_TOKENS = 64
WARMUP_STEPS   = 10
BENCH_STEPS    = 100

IM_END_ID = 151645   # <|im_end|>

# ---------------------------------------------------------------------------
# Chat template  (must match tok.chat_prompt() in include/model/tokenizer.h)
# ---------------------------------------------------------------------------

def build_chat_prompt(user_msg: str, enable_thinking: bool = False) -> str:
    system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    prefix = "" if enable_thinking else "<think>\n\n</think>\n"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{prefix}"
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def top5_str(logits: torch.Tensor) -> str:
    """Return top-5 'id(score)' pairs from a [vocab_size] float tensor."""
    vals, idx = torch.topk(logits.float(), 5)
    return "  ".join(f"{i.item():>6d}({v.item():.2f})" for i, v in zip(idx, vals))

# ---------------------------------------------------------------------------
# Correctness: step-by-step greedy generation with top-5 logging
# ---------------------------------------------------------------------------

def run_chat_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[int]:
    """
    Mirrors the output format of test_chat_generation() in test_qwen3.cu:
      [prefill] top-5: ...
      [step  0] token=   785  text=The          <top-5 of *next* logits>
      ...
    Returns the list of generated token IDs (excluding stop token).
    """
    prompt   = build_chat_prompt(USER_MSG)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    S         = len(input_ids)

    print(f"  Prompt: {prompt!r}")
    print(f"  Input tokens ({S}): {' '.join(map(str, input_ids))}")

    d_input = torch.tensor([input_ids], dtype=torch.long, device="cuda")

    # Prefill
    with torch.no_grad():
        out = model(d_input, use_cache=True)
    logits = out.logits[0, -1]          # [vocab_size]
    past   = out.past_key_values

    print(f"  [prefill] top-5:   seq[0] top-5: {top5_str(logits)}")

    generated: list[int] = []
    next_tok = int(logits.argmax())

    print(f"  Generating (max {max_new_tokens} tokens)...")

    for step in range(max_new_tokens):
        generated.append(next_tok)
        piece = tokenizer.decode([next_tok])
        print(f"  [step {step:2d}] token={next_tok:6d}  text={piece!r:<20}", end="")

        if next_tok == IM_END_ID:
            print()
            break

        d_tok = torch.tensor([[next_tok]], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out  = model(d_tok, past_key_values=past, use_cache=True)
        logits = out.logits[0, -1]
        past   = out.past_key_values
        next_tok = int(logits.argmax())
        print(f"  seq[0] top-5: {top5_str(logits)}")

    response = tokenizer.decode(generated, skip_special_tokens=False)
    print(f"\n  Full response: {response!r}")
    print(f"  Generated {len(generated)} tokens")
    return generated

# ---------------------------------------------------------------------------
# Speed benchmark — HuggingFace
# ---------------------------------------------------------------------------

def benchmark_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    warmup: int = WARMUP_STEPS,
    n_steps: int = BENCH_STEPS,
) -> tuple[float, float]:
    """
    Returns (tok/s, ms/step).

    Protocol (matches bench_decode.cu):
      1. Prefill the same chat prompt.
      2. Warm-up `warmup` decode steps (growing KV context, argmax each step).
      3. Timed `n_steps` decode steps from a fixed token (no argmax inside
         the timed region) so Python overhead doesn't distort the GPU time.
         KV cache keeps growing — same as the CUDA benchmark.
    """
    prompt    = build_chat_prompt(USER_MSG)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    d_input   = torch.tensor([input_ids], dtype=torch.long, device="cuda")

    # ── Prefill ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        out = model(d_input, use_cache=True)
    past     = out.past_key_values
    next_tok = int(out.logits[0, -1].argmax())

    # ── Warm-up (growing context) ─────────────────────────────────────────────
    for _ in range(warmup):
        d_tok = torch.tensor([[next_tok]], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out  = model(d_tok, past_key_values=past, use_cache=True)
        past     = out.past_key_values
        next_tok = int(out.logits[0, -1].argmax())

    # ── Timed (fixed token, growing context) ─────────────────────────────────
    # Reuse the same tensor to minimise Python allocation overhead.
    d_fixed = torch.tensor([[next_tok]], dtype=torch.long, device="cuda")

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(n_steps):
        with torch.no_grad():
            out  = model(d_fixed, past_key_values=past, use_cache=True)
        past = out.past_key_values   # grow context — realistic decode

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tps        = n_steps / elapsed
    ms_per_step = elapsed * 1000.0 / n_steps
    return tps, ms_per_step

# ---------------------------------------------------------------------------
# Speed benchmark — CUDA binary
# ---------------------------------------------------------------------------

def run_cuda_benchmark() -> tuple[float, float] | None:
    """
    Invokes build/bench_decode and parses its output line:
      BENCH_RESULT: tok/s=<float>
    Returns (tok/s, ms/step) or None if the binary is missing / fails.
    """
    if not os.path.exists(CUDA_BENCH):
        return None

    print(f"  Running {CUDA_BENCH} ...")
    result = subprocess.run(
        [CUDA_BENCH],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print(f"  [WARN] bench_decode exited with code {result.returncode}")
        print(result.stderr[-400:])
        return None

    # Print the binary's output indented
    for line in result.stdout.splitlines():
        print(f"    {line}")

    m = re.search(r"BENCH_RESULT: tok/s=([0-9.]+)", result.stdout)
    if not m:
        print("  [WARN] could not parse BENCH_RESULT from bench_decode output")
        return None

    tps = float(m.group(1))
    # Parse ms/step from the "Decode:" line
    ms_match = re.search(r"([\d.]+) ms/step", result.stdout)
    ms_per_step = float(ms_match.group(1)) if ms_match else 1000.0 / tps
    return tps, ms_per_step

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HuggingFace reference for Qwen3-0.6B vs pure-CUDA impl"
    )
    parser.add_argument(
        "--cuda-bench",
        action="store_true",
        help="Also run the compiled CUDA benchmark (make bench_decode first)",
    )
    args = parser.parse_args()

    if not os.path.isdir(MODEL_DIR):
        print(f"[SKIP] {MODEL_DIR} not found — download weights first")
        sys.exit(0)

    print("=" * 65)
    print("  Qwen3-0.6B  —  HuggingFace reference vs pure-CUDA engine")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\nLoading model (float16)...")
    t0        = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",  # use PyTorch's built-in flash attention if available
    ).cuda().eval()
    load_secs = time.perf_counter() - t0
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"  Loaded in {load_secs:.1f}s  ({n_params/1e6:.0f}M params, float16)\n")

    # ── Speed ─────────────────────────────────────────────────────────────────
    print()
    print("─" * 65)
    print(f"  Speed benchmark  ({WARMUP_STEPS} warmup + {BENCH_STEPS} timed decode steps)")
    print("─" * 65)

    print(f"\n  HuggingFace (float16) — benchmarking...")
    hf_tps, hf_ms = benchmark_hf(model, tokenizer)
    print(f"  HuggingFace:          {hf_tps:8.1f} tok/s  ({hf_ms:.2f} ms/step)")

    if args.cuda_bench:
        print()
        cuda_result = run_cuda_benchmark()
        if cuda_result is not None:
            cuda_tps, cuda_ms = cuda_result
            speedup = cuda_tps / hf_tps
            w = 52
            print()
            print(f"  ┌{'─'*w}┐")
            hf_col   = f"{hf_tps:7.1f} tok/s  ({hf_ms:.2f} ms/step)"
            cuda_col = f"{cuda_tps:7.1f} tok/s  ({cuda_ms:.2f} ms/step)"
            sp_col   = f"{speedup:.2f}x"
            print(f"  │  {'HuggingFace (float16):':<24} {hf_col:<{w-28}}│")
            print(f"  │  {'CUDA engine (float16):':<24} {cuda_col:<{w-28}}│")
            print(f"  │  {'Speedup:':<24} {sp_col:<{w-28}}│")
            print(f"  └{'─'*w}┘")
        else:
            print(f"  [INFO] Build with: make bench_decode")
    else:
        print(f"\n  [TIP]  Run with --cuda-bench to compare against the CUDA engine")

    print()


if __name__ == "__main__":
    main()
