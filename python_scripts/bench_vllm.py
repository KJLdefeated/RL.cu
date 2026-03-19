#!/usr/bin/env python3
"""
bench_vllm.py — Compare RL.cu engine against vLLM for Qwen3-0.6B

Runs the same prompts and batch-size sweep through both engines and prints
a side-by-side throughput + correctness table.

Usage:
    # vLLM only (no CUDA engine binary required)
    python python_scripts/bench_vllm.py

    # Also compare against our CUDA engine
    python python_scripts/bench_vllm.py --cuda-engine

    # Skip correctness check (throughput only)
    python python_scripts/bench_vllm.py --no-correctness

Requirements:
    pip install vllm
"""

import argparse
import os
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.join(SCRIPT_DIR, "..")
MODEL_DIR   = os.path.join(REPO_ROOT, "model_weights", "Qwen3-0.6B")
ENGINE_BIN  = os.path.join(REPO_ROOT, "build", "test_llmengine")

# Prompts used by test_llmengine section 3 throughput sweep
BENCH_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a short poem about the ocean.",
    "What are the main differences between Python and C++?",
    "Describe the water cycle in three sentences.",
    "What is photosynthesis and why is it important?",
    "Summarize the main causes of World War I.",
    "How does a computer CPU work at a high level?",
    "What is machine learning? Give a one-sentence definition.",
    "Describe the life cycle of a star.",
    "What is the Turing test and what does it measure?",
    "Explain DNA replication in simple terms.",
    "What are the main programming paradigms?",
    "How does the internet work at a high level?",
    "What is the difference between RAM and ROM?",
    "Describe Newton's three laws of motion.",
    "What is quantum entanglement?",
]

# Correctness cases — same as test_llmengine section 1
CORRECTNESS_CASES = [
    ("What is the capital of France? One word.",           "paris"),
    ("What is 2 + 2? Answer with a single digit.",         "4"),
    ("Which planet is closest to the Sun? One word only.", "mercury"),
    ("What language is used to write the Linux kernel?",   "c"),
]

BATCH_SIZES  = [64, 128, 256]
MAX_NEW_TOKENS = 2048


# ---------------------------------------------------------------------------
# Chat template (matches tokenizer.h chat_prompt())
# ---------------------------------------------------------------------------

def chat_prompt(user_msg: str) -> str:
    system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n"
    )


# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------

def load_vllm(model_dir: str):
    from vllm import LLM, SamplingParams  # type: ignore
    print(f"  Loading vLLM engine ({model_dir}) ...")
    t0 = time.perf_counter()
    llm = LLM(
        model=model_dir,
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
    )
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    return llm


def vllm_sampling_params(max_new: int):
    from vllm import SamplingParams  # type: ignore
    return SamplingParams(temperature=0.8, max_tokens=max_new)


def vllm_generate(llm, prompts: list[str], max_new: int) -> tuple[list[str], float]:
    """Returns (decoded_outputs, wall_seconds)."""
    sp = vllm_sampling_params(max_new)
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    wall = time.perf_counter() - t0
    texts = [o.outputs[0].text for o in outputs]
    return texts, wall


def count_output_tokens(llm, texts: list[str]) -> int:
    """Approximate token count via vLLM tokenizer."""
    tokenizer = llm.get_tokenizer()
    return sum(len(tokenizer.encode(t)) for t in texts)


# ---------------------------------------------------------------------------
# CUDA engine helpers
# ---------------------------------------------------------------------------

def run_cuda_engine() -> dict:
    """
    Invoke build/test_llmengine and parse its BENCH_RESULT line.
    Returns dict with keys: tok_s (float), batch (int), max_new (int).
    Returns None if binary not found or parse fails.
    """
    if not os.path.exists(ENGINE_BIN):
        return None

    print(f"\n  Running {ENGINE_BIN} ...")
    result = subprocess.run(
        [ENGINE_BIN, MODEL_DIR],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    stdout = result.stdout + result.stderr  # progress bar goes to stderr

    # Print test results (section 1 pass/fail lines)
    for line in result.stdout.splitlines():
        if line.startswith("[PASS]") or line.startswith("[FAIL]") or line.startswith("━"):
            print(f"    {line}")

    # Parse BENCH_RESULT: tok/s=<float>  batch=<int>  max_new=<int>
    m = re.search(r"BENCH_RESULT: tok/s=([0-9.]+)\s+batch=(\d+)\s+max_new=(\d+)",
                  result.stdout)
    if not m:
        print("  [WARN] Could not parse BENCH_RESULT from test_llmengine output")
        return None

    return {
        "tok_s":   float(m.group(1)),
        "batch":   int(m.group(2)),
        "max_new": int(m.group(3)),
    }


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def check_correctness(llm) -> list[tuple[str, str, bool]]:
    """Returns list of (question, answer, passed)."""
    prompts = [chat_prompt(q) for q, _ in CORRECTNESS_CASES]
    texts, _ = vllm_generate(llm, prompts, max_new=64)
    results = []
    for (question, must_contain), answer in zip(CORRECTNESS_CASES, texts):
        passed = must_contain.lower() in answer.lower()
        results.append((question, answer.strip()[:100], passed))
    return results


# ---------------------------------------------------------------------------
# vLLM throughput sweep
# ---------------------------------------------------------------------------

def bench_sweep(llm, batch_sizes: list[int], max_new: int) -> list[dict]:
    """
    For each batch size, generate max_new tokens for each prompt and measure
    tok/s (output tokens / wall-clock time).
    """
    records = []
    for B in batch_sizes:
        prompts = [chat_prompt(BENCH_PROMPTS[i % len(BENCH_PROMPTS)]) for i in range(B)]
        # Warmup
        vllm_generate(llm, prompts[:1], max_new=16)
        # Timed
        texts, wall = vllm_generate(llm, prompts, max_new)
        out_tokens  = count_output_tokens(llm, texts)
        tok_s       = out_tokens / wall if wall > 0 else 0.0
        ms_per_req  = wall * 1e3 / B
        records.append({
            "batch":       B,
            "out_tokens":  out_tokens,
            "wall_s":      wall,
            "tok_s":       tok_s,
            "ms_per_req":  ms_per_req,
        })
        print(f"    batch={B:2d}  {tok_s:8.0f} tok/s  {ms_per_req:8.1f} ms/req"
              f"  ({out_tokens} output tokens in {wall:.2f}s)")
    return records


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM vs RL.cu CUDA engine on Qwen3-0.6B"
    )
    parser.add_argument("--cuda-engine", action="store_true",
                        help="Also run build/test_llmengine for comparison")
    parser.add_argument("--no-correctness", action="store_true",
                        help="Skip correctness Q&A section")
    parser.add_argument("--model-dir", default=MODEL_DIR,
                        help=f"Path to model weights (default: {MODEL_DIR})")
    parser.add_argument("--max-new", type=int, default=MAX_NEW_TOKENS,
                        help=f"Max new tokens per request (default: {MAX_NEW_TOKENS})")
    parser.add_argument("--batches", type=int, nargs="+", default=BATCH_SIZES,
                        metavar="B",
                        help=f"Batch sizes to sweep (default: {BATCH_SIZES})")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"[SKIP] {args.model_dir} not found — download weights first")
        sys.exit(0)

    try:
        import vllm  # noqa: F401
    except ImportError:
        print("[ERROR] vllm not installed. Run: pip install vllm")
        sys.exit(1)

    print("=" * 68)
    print("  Qwen3-0.6B  —  vLLM vs RL.cu CUDA engine")
    print("=" * 68)

    # ── Load vLLM ────────────────────────────────────────────────────────────
    print("\n[1/3] Loading vLLM...")
    llm = load_vllm(args.model_dir)

    # ── Correctness ───────────────────────────────────────────────────────────
    if not args.no_correctness:
        print("\n[2/3] Correctness (vLLM greedy, max_new=64)")
        print("-" * 68)
        results = check_correctness(llm)
        n_pass = 0
        for q, ans, ok in results:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}]  Q: {q}")
            print(f"          A: {ans}")
            if ok:
                n_pass += 1
        print(f"\n  vLLM correctness: {n_pass}/{len(results)} passed")
    else:
        print("\n[2/3] Correctness: skipped")

    # ── vLLM throughput sweep ─────────────────────────────────────────────────
    print(f"\n[3/3] vLLM throughput sweep (greedy, max_new={args.max_new})")
    print("-" * 68)
    print(f"  {'batch':>5}  {'tok/s':>10}  {'ms/req':>10}  {'out_tokens':>12}")
    sweep = bench_sweep(llm, args.batches, args.max_new)

    best_vllm = max(sweep, key=lambda r: r["tok_s"])

    # ── CUDA engine (optional) ─────────────────────────────────────────────────
    cuda_result = None
    if args.cuda_engine:
        print("\n[4/4] RL.cu CUDA engine (build/test_llmengine)")
        print("-" * 68)
        cuda_result = run_cuda_engine()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  SUMMARY")
    print("=" * 68)

    print(f"\n  vLLM (Qwen3-0.6B / greedy / max_new={args.max_new})")
    print(f"  {'batch':>5}  {'tok/s':>10}  {'ms/req':>10}")
    for r in sweep:
        marker = " ◀ best" if r is best_vllm else ""
        print(f"  {r['batch']:5d}  {r['tok_s']:10.0f}  {r['ms_per_req']:10.1f}{marker}")

    if cuda_result:
        cuda_tps  = cuda_result["tok_s"]
        cmp_batch = cuda_result["batch"]
        cmp_vllm  = next((r for r in sweep if r["batch"] == cmp_batch), None)

        print(f"\n  RL.cu engine  (batch={cmp_batch}, max_new={cuda_result['max_new']})")
        print(f"  {'engine':<20}  {'tok/s':>10}")
        print(f"  {'RL.cu (CUDA)':<20}  {cuda_tps:10.0f}")
        if cmp_vllm:
            ratio = cuda_tps / cmp_vllm["tok_s"]
            print(f"  {'vLLM (same batch)':<20}  {cmp_vllm['tok_s']:10.0f}")
            print(f"  {'ratio (RL.cu/vLLM)':<20}  {ratio:10.2f}x"
                  f"  {'(faster)' if ratio >= 1.0 else '(slower)'}")

        # nano-vllm normalisation (7B equivalent, A100 baseline)
        # nano-vllm reports ~2600 tok/s for Qwen2.5-7B at batch=256 on A100.
        cuda_norm = cuda_tps / 12.0   # 0.6B → 7B equivalent (model-size ratio ≈12)
        vllm_norm = best_vllm["tok_s"] / 12.0
        print(f"\n  nano-vllm comparison (÷12 normalisation to 7B-equiv, A100 baseline ≈ 2600 tok/s)")
        print(f"  {'engine':<25}  {'raw tok/s':>10}  {'7B-equiv tok/s':>15}")
        print(f"  {'nano-vllm (A100, batch=256)':<25}  {'~2600':>10}  {'~2600':>15}")
        print(f"  {'vLLM (this GPU, best batch)':<25}  {best_vllm['tok_s']:10.0f}  {vllm_norm:15.0f}")
        print(f"  {'RL.cu (this GPU)':<25}  {cuda_tps:10.0f}  {cuda_norm:15.0f}")
    else:
        vllm_norm = best_vllm["tok_s"] / 12.0
        print(f"\n  nano-vllm comparison (÷12 normalisation to 7B-equiv, A100 baseline ≈ 2600 tok/s)")
        print(f"  vLLM best: {best_vllm['tok_s']:.0f} tok/s  →  7B-equiv: {vllm_norm:.0f} tok/s")
        print(f"\n  Tip: run with --cuda-engine to add RL.cu numbers to this table")

    print()


if __name__ == "__main__":
    main()
