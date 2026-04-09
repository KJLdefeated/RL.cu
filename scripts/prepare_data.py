#!/usr/bin/env python3
"""
prepare_data.py — Download and preprocess training data for RL.cu

Downloads datasets from HuggingFace, tokenizes with Qwen3 tokenizer,
and saves as a binary format optimized for C++ DataLoader consumption.

Usage:
  # SFT data (PrimeIntellect/SYNTHETIC-1-SFT-Data)
  python scripts/prepare_data.py --mode sft --output data/sft_train.bin

  # GRPO data (BytedTsinghua-SIA/DAPO-Math-17k)
  python scripts/prepare_data.py --mode grpo --output data/grpo_train.bin

  # Custom max samples / sequence length
  python scripts/prepare_data.py --mode sft --output data/sft_small.bin \\
      --max-samples 10000 --max-seq-len 1024

  # Use local parquet file(s)
  python scripts/prepare_data.py --mode sft --output data/sft_train.bin \\
      --local-parquet data/my_data.parquet

Dependencies:
  pip install pyarrow huggingface_hub transformers

Binary format (.bin):
  Header (32 bytes):
    magic     : char[4]  = "RLDT"
    version   : uint32   = 1
    num_samples: uint64
    total_tokens: uint64
    flags     : uint32   (bit 0: has_prompt_lens, bit 1: has_answers)
    pad       : uint32
  Offsets    : uint64[num_samples + 1]  (token index where each sample starts)
  PromptLens : uint32[num_samples]      (if has_prompt_lens flag)
  Tokens     : int32[total_tokens]      (all token IDs concatenated)

  Companion file (.answers.jsonl) for GRPO mode:
    One JSON object per line: {"index": i, "answer": "..."}
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    print("ERROR: pyarrow not installed. Run: pip install pyarrow", file=sys.stderr)
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: transformers not installed. Run: pip install transformers", file=sys.stderr)
    sys.exit(1)


# ─── Constants ────────────────────────────────────────────────────────────────

MAGIC = b"RLDT"
VERSION = 1
FLAG_HAS_PROMPT_LENS = 1 << 0
FLAG_HAS_ANSWERS     = 1 << 1

SFT_DATASET  = "PrimeIntellect/SYNTHETIC-1-SFT-Data"
GRPO_DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
TOKENIZER_NAME = "Qwen/Qwen3-0.6B"


# ─── Download helpers ─────────────────────────────────────────────────────────

def download_parquet_files(repo_id: str, cache_dir: str = None) -> list[str]:
    """Download all parquet files from a HuggingFace dataset repo."""
    print(f"[download] Listing files in {repo_id}...")
    all_files = list_repo_files(repo_id, repo_type="dataset")
    parquet_files = [f for f in all_files if f.endswith(".parquet")]

    if not parquet_files:
        raise ValueError(f"No parquet files found in {repo_id}")

    print(f"[download] Found {len(parquet_files)} parquet file(s)")
    local_paths = []
    for pf in parquet_files:
        print(f"  Downloading {pf}...")
        path = hf_hub_download(
            repo_id, pf, repo_type="dataset", cache_dir=cache_dir
        )
        local_paths.append(path)
    return local_paths


def load_parquet_table(paths: list[str]) -> pa.Table:
    """Load and concatenate parquet files into a single Arrow table."""
    tables = []
    for p in paths:
        t = pq.read_table(p)
        tables.append(t)
        print(f"  Loaded {p}: {len(t)} rows")
    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables)


# ─── Chat formatting ─────────────────────────────────────────────────────────

def format_sft_chat(messages: list[dict]) -> tuple[str, str]:
    """
    Format SFT messages into (prompt_text, full_text) using Qwen3 chat template.
    Returns the prompt portion and the full conversation text.
    """
    # Build prompt (everything up to assistant response)
    prompt = ""
    full = ""

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        turn = f"<|im_start|>{role}\n{content}<|im_end|>\n"
        full += turn
        if role != "assistant":
            prompt += turn

    # Add assistant prefix to prompt (model continues from here)
    prompt += "<|im_start|>assistant\n"

    return prompt, full


GRPO_SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step using thinking mode. Use <think>...<think/> tags for your reasoning steps. "
)


def format_grpo_prompt(prompt_messages: list[dict]) -> str:
    """
    Format GRPO prompt messages into text using Qwen3 chat template.
    Uses a chain-of-thought system prompt and enables thinking mode.
    """
    text = f"<|im_start|>system\n{GRPO_SYSTEM_PROMPT}<|im_end|>\n"
    for msg in prompt_messages:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text


# ─── Tokenization + binary writing ───────────────────────────────────────────

def write_binary(
    output_path: str,
    all_tokens: list[np.ndarray],
    prompt_lens: list[int] | None,
    answers: list[str] | None,
):
    """Write tokenized data to binary format."""
    num_samples = len(all_tokens)
    total_tokens = sum(len(t) for t in all_tokens)

    flags = 0
    if prompt_lens is not None:
        flags |= FLAG_HAS_PROMPT_LENS
    if answers is not None:
        flags |= FLAG_HAS_ANSWERS

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "wb") as f:
        # Header (32 bytes)
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<Q", num_samples))
        f.write(struct.pack("<Q", total_tokens))
        f.write(struct.pack("<I", flags))
        f.write(struct.pack("<I", 0))  # padding

        # Offsets: uint64[num_samples + 1]
        offset = 0
        for tokens in all_tokens:
            f.write(struct.pack("<Q", offset))
            offset += len(tokens)
        f.write(struct.pack("<Q", offset))  # sentinel

        # Prompt lengths: uint32[num_samples]
        if prompt_lens is not None:
            for pl in prompt_lens:
                f.write(struct.pack("<I", pl))

        # Token data: int32[total_tokens]
        for tokens in all_tokens:
            f.write(tokens.astype(np.int32).tobytes())

    file_size = os.path.getsize(output_path)
    print(f"[write] {output_path}: {num_samples} samples, {total_tokens} tokens, "
          f"{file_size / 1024 / 1024:.1f} MB")

    # Write companion answers file for GRPO
    if answers is not None:
        answers_path = output_path.replace(".bin", ".answers.jsonl")
        with open(answers_path, "w") as f:
            for i, ans in enumerate(answers):
                f.write(json.dumps({"index": i, "answer": ans}) + "\n")
        print(f"[write] {answers_path}: {len(answers)} answers")


# ─── SFT processing ──────────────────────────────────────────────────────────

def process_sft(
    table: pa.Table,
    tokenizer,
    max_samples: int | None,
    max_seq_len: int,
    min_score: float,
) -> tuple[list[np.ndarray], list[int]]:
    """Process SFT dataset: tokenize (prompt + response), compute prompt lengths."""
    messages_col = table.column("messages")
    score_col = table.column("score") if "score" in table.column_names else None

    all_tokens = []
    prompt_lens = []
    skipped = 0
    n = len(table) if max_samples is None else min(len(table), max_samples)

    for i in range(n):
        if i % 10000 == 0 and i > 0:
            print(f"  [{i}/{n}] processed, {skipped} skipped")

        # Filter by score
        if score_col is not None:
            score = score_col[i].as_py()
            if score is not None and score < min_score:
                skipped += 1
                continue

        messages = messages_col[i].as_py()
        if not messages or len(messages) < 2:
            skipped += 1
            continue

        prompt_text, full_text = format_sft_chat(messages)

        # Tokenize prompt (for loss mask boundary)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        # Tokenize full sequence
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # Truncate to max_seq_len
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]

        if len(full_ids) < 4:  # skip very short samples
            skipped += 1
            continue

        all_tokens.append(np.array(full_ids, dtype=np.int32))
        prompt_lens.append(min(len(prompt_ids), len(full_ids)))

    print(f"  Final: {len(all_tokens)} samples ({skipped} skipped)")
    return all_tokens, prompt_lens


# ─── GRPO processing ─────────────────────────────────────────────────────────

def process_grpo(
    table: pa.Table,
    tokenizer,
    max_samples: int | None,
    max_seq_len: int,
) -> tuple[list[np.ndarray], list[str]]:
    """Process GRPO dataset: tokenize prompts, extract ground truth answers."""
    prompt_col = table.column("prompt")
    reward_col = table.column("reward_model")

    all_tokens = []
    answers = []
    skipped = 0
    seen_prompts = set()

    n = len(table) if max_samples is None else min(len(table), max_samples)

    for i in range(n):
        if i % 50000 == 0 and i > 0:
            print(f"  [{i}/{n}] processed, {len(all_tokens)} unique, {skipped} skipped")

        prompt_messages = prompt_col[i].as_py()
        if not prompt_messages:
            skipped += 1
            continue

        # Extract ground truth answer
        reward_info = reward_col[i].as_py()
        answer = reward_info.get("ground_truth", "") if reward_info else ""

        # Deduplicate by prompt content
        prompt_key = prompt_messages[0]["content"][:200] if prompt_messages else ""
        if prompt_key in seen_prompts:
            skipped += 1
            continue
        seen_prompts.add(prompt_key)

        prompt_text = format_grpo_prompt(prompt_messages)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        if len(prompt_ids) > max_seq_len:
            prompt_ids = prompt_ids[:max_seq_len]

        if len(prompt_ids) < 4:
            skipped += 1
            continue

        all_tokens.append(np.array(prompt_ids, dtype=np.int32))
        answers.append(answer)

    print(f"  Final: {len(all_tokens)} unique prompts ({skipped} skipped)")
    return all_tokens, answers


def process_grpo_text(
    table: pa.Table,
    max_samples: int | None,
) -> list[dict]:
    """Process GRPO dataset: extract raw prompt text + ground truth answers.
    Returns list of {"prompt": str, "answer": str} dicts (no tokenization)."""
    prompt_col = table.column("prompt")
    reward_col = table.column("reward_model")

    samples = []
    skipped = 0
    seen_prompts = set()

    n = len(table) if max_samples is None else min(len(table), max_samples)

    for i in range(n):
        if i % 50000 == 0 and i > 0:
            print(f"  [{i}/{n}] processed, {len(samples)} unique, {skipped} skipped")

        prompt_messages = prompt_col[i].as_py()
        if not prompt_messages:
            skipped += 1
            continue

        # Extract ground truth answer
        reward_info = reward_col[i].as_py()
        answer = reward_info.get("ground_truth", "") if reward_info else ""

        # Deduplicate by prompt content
        prompt_key = prompt_messages[0]["content"][:200] if prompt_messages else ""
        if prompt_key in seen_prompts:
            skipped += 1
            continue
        seen_prompts.add(prompt_key)

        prompt_text = format_grpo_prompt(prompt_messages)

        if len(prompt_text) < 20:
            skipped += 1
            continue

        samples.append({"prompt": prompt_text, "answer": answer})

    print(f"  Final: {len(samples)} unique prompts ({skipped} skipped)")
    return samples


def write_grpo_jsonl(output_path: str, samples: list[dict]):
    """Write GRPO data as JSONL: one {"prompt": ..., "answer": ...} per line."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    file_size = os.path.getsize(output_path)
    print(f"[write] {output_path}: {len(samples)} samples, {file_size / 1024 / 1024:.1f} MB")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for RL.cu")
    parser.add_argument("--mode", required=True, choices=["sft", "grpo", "grpo-text"],
                        help="Dataset mode: sft, grpo (tokenized binary), or grpo-text (JSONL, tokenize at runtime)")
    parser.add_argument("--output", required=True,
                        help="Output binary file path (e.g., data/sft_train.bin)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to process (default: all)")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Max sequence length in tokens (default: 2048)")
    parser.add_argument("--min-score", type=float, default=0.9,
                        help="Min quality score for SFT data (default: 0.9)")
    parser.add_argument("--tokenizer", default=TOKENIZER_NAME,
                        help=f"Tokenizer name or path (default: {TOKENIZER_NAME})")
    parser.add_argument("--local-parquet", nargs="+", default=None,
                        help="Use local parquet file(s) instead of downloading")
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace cache directory")
    args = parser.parse_args()

    # Load data
    if args.local_parquet:
        print(f"[data] Loading {len(args.local_parquet)} local parquet file(s)...")
        table = load_parquet_table(args.local_parquet)
    else:
        repo_id = SFT_DATASET if args.mode == "sft" else GRPO_DATASET
        paths = download_parquet_files(repo_id, cache_dir=args.cache_dir)
        table = load_parquet_table(paths)

    print(f"[data] Total rows: {len(table)}")
    print(f"[data] Columns: {table.column_names}")

    # Process
    if args.mode == "grpo-text":
        # Text-only JSONL mode: no tokenization, stores raw prompt + answer
        print(f"[grpo-text] Processing...")
        samples = process_grpo_text(table, args.max_samples)
        write_grpo_jsonl(args.output, samples)
        prompt_lens = [len(s["prompt"]) for s in samples]
        print(f"\n[stats] Samples: {len(samples)}")
        print(f"  Prompt chars: min={min(prompt_lens)}, max={max(prompt_lens)}, "
              f"mean={sum(prompt_lens)/len(prompt_lens):.0f}")
    else:
        # Tokenized binary modes (sft, grpo)
        print(f"[tokenizer] Loading {args.tokenizer}...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        print(f"  Vocab size: {tokenizer.vocab_size}")

        if args.mode == "sft":
            print(f"[sft] Processing (max_seq_len={args.max_seq_len}, "
                  f"min_score={args.min_score})...")
            all_tokens, prompt_lens = process_sft(
                table, tokenizer, args.max_samples, args.max_seq_len, args.min_score
            )
            write_binary(args.output, all_tokens, prompt_lens, answers=None)
        else:
            print(f"[grpo] Processing (max_seq_len={args.max_seq_len})...")
            all_tokens, answers = process_grpo(
                table, tokenizer, args.max_samples, args.max_seq_len
            )
            write_binary(args.output, all_tokens, prompt_lens=None, answers=answers)

        lengths = [len(t) for t in all_tokens]
        print(f"\n[stats] Samples: {len(all_tokens)}")
        print(f"  Tokens: {sum(lengths):,}")
        print(f"  Seq len: min={min(lengths)}, max={max(lengths)}, "
              f"mean={sum(lengths)/len(lengths):.0f}, "
              f"median={sorted(lengths)[len(lengths)//2]}")


if __name__ == "__main__":
    main()
