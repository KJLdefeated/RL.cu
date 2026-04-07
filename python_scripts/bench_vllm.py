#!/usr/bin/env python3
"""
bench_vllm.py — Run the nano-vllm bench.py protocol on official vLLM.

Mirrors exactly what bench_nanollm.py (and our C++ test_bench_realistic()) does:
  seed=0, 256 seqs, prompt_len~Uniform[100,1024], output_len~Uniform[100,1024]
  token IDs ~ Uniform[0,10000], temperature=0.6, ignore_eos=True
  warmup with 1 seq, timed run, metric = sum(max_tokens) / wall_time

Usage:
    cd /home/kjl0508/RL_cuda
    python python_scripts/bench_vllm.py [--model MODEL_DIR]

Requirements:
    pip install vllm
"""

import argparse
import os
import sys
import time
from random import randint, seed as set_seed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.join(SCRIPT_DIR, "..")
MODEL_DIR  = os.path.join(REPO_ROOT, "model_weights", "Qwen3-0.6B")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_DIR)
    parser.add_argument("--num-seqs",    type=int, default=256)
    parser.add_argument("--max-input",   type=int, default=1024)
    parser.add_argument("--max-output",  type=int, default=1024)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graphs in vLLM (eager mode)")
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        print(f"[SKIP] {args.model} not found")
        sys.exit(0)

    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except ImportError:
        print("[ERROR] vllm not installed — run: pip install vllm")
        sys.exit(1)

    NUM_SEQS      = args.num_seqs
    MAX_INPUT_LEN = args.max_input
    MAX_OUT_LEN   = args.max_output

    set_seed(0)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, MAX_INPUT_LEN))]
        for _ in range(NUM_SEQS)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True,
                       max_tokens=randint(100, MAX_OUT_LEN))
        for _ in range(NUM_SEQS)
    ]
    total_tokens = sum(sp.max_tokens for sp in sampling_params)

    print(f"Model  : {args.model}")
    print(f"Seqs   : {NUM_SEQS}  prompt_len~[100,{MAX_INPUT_LEN}]  "
          f"output_len~[100,{MAX_OUT_LEN}]")
    print(f"Total output tokens: {total_tokens}")
    print()

    llm = LLM(
        model=args.model,
        dtype="float16",
        max_model_len=MAX_INPUT_LEN + MAX_OUT_LEN,
        gpu_memory_utilization=0.85,
        enforce_eager=args.enforce_eager,
    )

    # Warmup
    print("[warmup] ...")
    llm.generate(
        [{"prompt_token_ids": [1]}],
        SamplingParams(temperature=0.6, max_tokens=4),
        use_tqdm=False,
    )

    # Timed run
    print("[bench ] running ...")
    inputs = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
    t = time.perf_counter()
    llm.generate(inputs, sampling_params, use_tqdm=True)
    t = time.perf_counter() - t

    throughput = total_tokens / t
    print(f"\nTotal: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
