#!/usr/bin/env python3
"""
Prepare a competitive-programming problem set for RL.cu agentic rollouts.

Downloads a HuggingFace dataset (default: deepmind/code_contests), extracts the
public test cases (shown to the model as feedback) and hidden test cases (used
for the final reward), and writes a JSONL file consumed by
include/training/cp_dataset.h.

Output JSONL — one line per problem:
  {
    "id": "1575_A. Another Sorting Problem",
    "description": "...",                       # optional, for the prompt
    "public_tests": [{"input":"...", "expected":"..."}, ...],
    "hidden_tests": [{"input":"...", "expected":"..."}, ...]
  }

Examples:
  # Default: download CodeContests test split (165 problems, 1 shard)
  python scripts/prepare_cp_data.py --output data/code_contests.jsonl

  # Smaller smoke set + per-problem cap on hidden tests
  python scripts/prepare_cp_data.py --output data/cc_small.jsonl \\
      --max-samples 20 --max-hidden 5

  # Train split (39 shards) — pass --max-samples to bound the download time
  python scripts/prepare_cp_data.py --output data/cc_train.jsonl \\
      --split train --shards 2 --max-samples 1000
"""
import argparse
import json
import os
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow not installed. Run: pip install pyarrow", file=sys.stderr)
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub",
          file=sys.stderr)
    sys.exit(1)


def list_shards(repo_id: str, split: str) -> list[str]:
    files = list_repo_files(repo_id, repo_type="dataset")
    prefix = f"data/{split}-"
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))


def download_shards(repo_id: str, shard_paths: list[str], cache_dir: str | None) -> list[str]:
    out = []
    for p in shard_paths:
        print(f"[download] {p} ...")
        local = hf_hub_download(repo_id, p, repo_type="dataset", cache_dir=cache_dir)
        out.append(local)
    return out


def _collect_pairs(field: dict | None, cap: int) -> list[dict]:
    """Convert HF's {'input': [...], 'output': [...]} into [{'input', 'expected'}]."""
    if not field:
        return []
    inputs  = field.get("input")  or []
    outputs = field.get("output") or []
    n = min(len(inputs), len(outputs))
    if cap is not None:
        n = min(n, cap)
    return [{"input": inputs[i], "expected": outputs[i]} for i in range(n)]


def extract_problem(row: dict, max_hidden: int) -> dict | None:
    name = row.get("name") or ""
    if not name:
        return None
    public = _collect_pairs(row.get("public_tests"), cap=None)
    if not public:
        return None  # no way to give per-turn feedback → skip
    # Hidden = private + generated (private is gold-standard, generated is large).
    # Take private first to prioritize quality, then top off with generated.
    hidden  = _collect_pairs(row.get("private_tests"),   cap=None)
    hidden += _collect_pairs(row.get("generated_tests"), cap=None)
    if not hidden:
        return None  # no way to score → skip
    if max_hidden is not None:
        hidden = hidden[:max_hidden]
    out = {
        "id": name,
        "description": row.get("description", "") or "",
        "public_tests": public,
        "hidden_tests": hidden,
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, help="JSONL output path")
    ap.add_argument("--dataset", default="deepmind/code_contests",
                    help="HF dataset repo id (default: deepmind/code_contests)")
    ap.add_argument("--split", default="test", choices=["train", "test", "valid", "validation"],
                    help="Dataset split (default: test — small, fast)")
    ap.add_argument("--shards", type=int, default=None,
                    help="Number of parquet shards to download (default: all)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap on total problems written (default: no cap)")
    ap.add_argument("--max-hidden", type=int, default=10,
                    help="Cap on hidden tests per problem (default: 10)")
    ap.add_argument("--min-public", type=int, default=1,
                    help="Drop problems with fewer public tests (default: 1)")
    ap.add_argument("--difficulty-min", type=int, default=None,
                    help="Filter on CodeContests `difficulty` field (>= this).")
    ap.add_argument("--difficulty-max", type=int, default=None,
                    help="Filter on CodeContests `difficulty` field (<= this).")
    ap.add_argument("--cache-dir", default=None,
                    help="HuggingFace cache directory (default: HF default)")
    args = ap.parse_args()

    # Normalize HF split naming.
    split = "valid" if args.split == "validation" else args.split

    shards = list_shards(args.dataset, split)
    if not shards:
        print(f"ERROR: no {split} shards found in {args.dataset}", file=sys.stderr)
        sys.exit(2)
    if args.shards is not None:
        shards = shards[: args.shards]
    print(f"[plan] dataset={args.dataset} split={split} "
          f"shards={len(shards)} max_samples={args.max_samples} "
          f"max_hidden={args.max_hidden}")

    local_paths = download_shards(args.dataset, shards, args.cache_dir)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    n_dropped_difficulty = 0
    n_dropped_public = 0
    n_dropped_hidden = 0

    with out_path.open("w", encoding="utf-8") as f:
        for lp in local_paths:
            tbl = pq.read_table(lp)
            print(f"  [shard] {os.path.basename(lp)}: {len(tbl)} rows")
            for row in tbl.to_pylist():
                if args.max_samples is not None and n_written >= args.max_samples:
                    break

                # Difficulty filter (if requested + field exists)
                d = row.get("difficulty")
                if args.difficulty_min is not None and (d is None or d < args.difficulty_min):
                    n_dropped_difficulty += 1
                    continue
                if args.difficulty_max is not None and (d is None or d > args.difficulty_max):
                    n_dropped_difficulty += 1
                    continue

                p = extract_problem(row, args.max_hidden)
                if p is None:
                    # Distinguish reason for diagnostics
                    pub = _collect_pairs((row.get("public_tests") or {}), cap=None)
                    if len(pub) < args.min_public:
                        n_dropped_public += 1
                    else:
                        n_dropped_hidden += 1
                    n_skipped += 1
                    continue
                if len(p["public_tests"]) < args.min_public:
                    n_dropped_public += 1
                    n_skipped += 1
                    continue

                f.write(json.dumps(p, ensure_ascii=False) + "\n")
                n_written += 1

            if args.max_samples is not None and n_written >= args.max_samples:
                break

    print(f"[write] {out_path}: {n_written} problems "
          f"(skipped: {n_skipped} — public={n_dropped_public}, "
          f"hidden={n_dropped_hidden}, difficulty={n_dropped_difficulty})")
    # Quick stats
    if n_written > 0:
        sz = out_path.stat().st_size
        print(f"[stats] file size: {sz/1024:.1f} KB"
              f"  avg per-problem: {sz/n_written:.0f} bytes")


if __name__ == "__main__":
    main()
