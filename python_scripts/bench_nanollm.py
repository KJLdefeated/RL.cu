#!/usr/bin/env python3
"""
bench_nanollm.py — Run the nano-vllm bench.py protocol on nano-vllm.

Mirrors exactly what our C++ test_bench_realistic() does:
  seed=0, 256 seqs, prompt_len~Uniform[100,1024], output_len~Uniform[100,1024]
  token IDs ~ Uniform[0,10000], temperature=0.6, top_k=50, ignore_eos=True
  warmup with 1 seq, timed run, metric = sum(max_tokens) / wall_time

Usage:
    cd /home/kjl0508/RL_cuda
    python python_scripts/bench_nanollm.py [--model MODEL_DIR]
"""

import os
import sys
import time
from random import randint, seed as set_seed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.join(SCRIPT_DIR, "..")
MODEL_DIR  = os.path.join(REPO_ROOT, "model_weights", "Qwen3-0.6B")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_DIR)
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        print(f"[SKIP] {args.model} not found")
        sys.exit(0)

    try:
        from nanovllm import LLM, SamplingParams  # type: ignore
    except ImportError:
        print("[ERROR] nanovllm not installed — run: pip install nanovllm")
        sys.exit(1)

    NUM_SEQS      = 256
    MAX_INPUT_LEN = 1024
    MAX_OUT_LEN   = 1024

    set_seed(0)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, MAX_INPUT_LEN))]
        for _ in range(NUM_SEQS)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, top_k=50, ignore_eos=True,
                       max_tokens=randint(100, MAX_OUT_LEN))
        for _ in range(NUM_SEQS)
    ]
    total_tokens = sum(sp.max_tokens for sp in sampling_params)

    print(f"Model  : {args.model}")
    print(f"Seqs   : {NUM_SEQS}  prompt_len~[100,{MAX_INPUT_LEN}]  "
          f"output_len~[100,{MAX_OUT_LEN}]")
    print(f"Total output tokens: {total_tokens}")
    print()

    llm = LLM(args.model, enforce_eager=False, max_model_len=4096)

    # Warmup
    print("[warmup] ...")
    llm.generate([[1]], [SamplingParams(temperature=0.6, top_k=50, max_tokens=4)])

    # Timed run
    print("[bench ] running ...")
    t = time.perf_counter()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    t = time.perf_counter() - t

    throughput = total_tokens / t
    print(f"\nTotal: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
