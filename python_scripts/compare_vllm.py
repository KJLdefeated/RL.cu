#!/usr/bin/env python3
"""
Compare rollout quality: Python vLLM vs. GRPO-CUDA engine.

Usage:
    python3 scripts/compare_vllm.py [--model Qwen/Qwen3-0.6B] [--n-prompts 4] [--G 2]

Reads prompts from data/dapo-17k.bin (same binary the C++ trainer reads),
generates completions with vLLM, and prints them side-by-side.
"""

import argparse
import struct
import json

def load_bin_prompts(path, n=4):
    """Read pre-tokenized prompts from the RLDT binary."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"RLDT"
        version, = struct.unpack("<I", f.read(4))
        num_samples, = struct.unpack("<Q", f.read(8))
        total_tokens, = struct.unpack("<Q", f.read(8))
        flags, = struct.unpack("<I", f.read(4))
        _pad, = struct.unpack("<I", f.read(4))

        offsets = struct.unpack(f"<{num_samples+1}Q", f.read(8 * (num_samples + 1)))

        has_prompt_lens = flags & 1
        if has_prompt_lens:
            f.read(4 * num_samples)  # skip prompt_lens

        all_tokens = struct.unpack(f"<{total_tokens}i", f.read(4 * total_tokens))

    prompts = []
    for i in range(min(n, num_samples)):
        toks = list(all_tokens[offsets[i]:offsets[i+1]])
        prompts.append(toks)
    return prompts


def load_answers(path, n=4):
    """Load answers from JSONL."""
    answers = {}
    try:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                j = json.loads(line)
                answers[j["index"]] = j["answer"]
    except FileNotFoundError:
        pass
    return [answers.get(i, "") for i in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data", default="data/dapo-17k.bin")
    parser.add_argument("--answers", default="data/dapo-17k.answers.jsonl")
    parser.add_argument("--n-prompts", type=int, default=4)
    parser.add_argument("--G", type=int, default=2, help="completions per prompt")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    print(f"Loading model: {args.model}")
    llm = LLM(model=args.model, dtype="float16", max_model_len=2048)
    tokenizer = llm.get_tokenizer()

    prompts = load_bin_prompts(args.data, args.n_prompts)
    answers = load_answers(args.answers, args.n_prompts)

    print(f"\nLoaded {len(prompts)} prompts from {args.data}")
    inputs = []
    for i, p in enumerate(prompts):
        text = tokenizer.decode(p)
        inputs.append(text)
        print(f"  Prompt {i}: {len(p)} tokens — {text[:100]}...")

    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.G,
    )

    # Build prompt_token_ids: replicate each prompt G times
    # vLLM's generate() with n=G handles this natively
    outputs = llm.generate(inputs, sampling_params=sp)

    print(f"\n{'='*80}")
    print(f"vLLM Results (model={args.model}, temp={args.temperature}, top_p={args.top_p})")
    print(f"{'='*80}")

    for i, output in enumerate(outputs):
        print(prompts[i])
        print(f"  ANSWER: {answers[i]}")
        for g, completion in enumerate(output.outputs):
            text = completion.text
            # Check for \boxed{}
            import re
            boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
            boxed_str = boxed[-1] if boxed else "(none)"
            trunc = text[:500] + "..." if len(text) > 500 else text
            print(f"  GEN[{g}] ({len(completion.token_ids)} tok): {trunc}")
            print(f"    \\boxed: {boxed_str}  | correct: {boxed_str.strip() == answers[i].strip()}")

    print(f"\n{'='*80}")
    print("Done.")


if __name__ == "__main__":
    main()
