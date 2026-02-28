#!/usr/bin/env python3
"""
generate_references.py
======================
Generate PyTorch reference tensors for the three kernel unit tests:
  - RMSNorm   (tests/test_rmsnorm.cu)
  - Softmax   (tests/test_softmax.cu)
  - SwiGLU    (tests/test_swiglu.cu)

For every test case the script saves:
  tests/reference_data/<kernel>/<case>/
      x.fp16.bin          raw float16 bytes  (C++ fread-friendly)
      weight.fp16.bin     (RMSNorm only)
      gate.fp16.bin       (SwiGLU only)
      up.fp16.bin         (SwiGLU only)
      output.fp16.bin     PyTorch output, float16
      meta.json           shape, dtype, eps

Usage:
    python tests/generate_references.py [--outdir tests/reference_data]
    python tests/generate_references.py --device cpu   # no GPU required
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# PyTorch reference implementations
# All operate in FP32 internally; I/O is float16 (matches kernel contract).
# ──────────────────────────────────────────────────────────────────────────────

def ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm: out = x / rms(x) * weight  (FP32 accumulation, FP16 I/O)."""
    x32  = x.float()
    w32  = weight.float()
    var  = x32.pow(2).mean(dim=-1, keepdim=True)          # [rows, 1]
    norm = x32 * torch.rsqrt(var + eps)                   # [rows, cols]
    return (norm * w32).half()


def ref_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax  (FP32 accumulation, FP16 I/O)."""
    return F.softmax(x.float(), dim=-1).half()


def ref_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU: silu(gate) * up  (FP32 accumulation, FP16 I/O)."""
    g32 = gate.float()
    u32 = up.float()
    return (F.silu(g32) * u32).half()


# ──────────────────────────────────────────────────────────────────────────────
# Test-case definitions  (shapes must match the C++ test files)
# ──────────────────────────────────────────────────────────────────────────────

RMSNORM_CASES = [
    # (name,               rows,  cols,  description)
    ("rows4_cols128",       4,    128,   "head_dim  — q_norm / k_norm"),
    ("rows4_cols1024",      4,   1024,   "hidden_size — input_layernorm"),
    ("rows16_cols1024",    16,   1024,   "batched hidden"),
    ("rows1_cols128",       1,    128,   "single head"),
]

SOFTMAX_CASES = [
    # (name,                rows,   cols,    description)
    ("rows4_cols1024",       4,     1024,    "small  (hidden dim)"),
    ("rows2_cols32768",      2,    32768,    "medium (attention, seq=32k)"),
    ("rows1_cols151936",     1,   151936,    "large  (vocab logits, Qwen3)"),
    ("rows8_cols2048",       8,     2048,    "batched attention"),
]

SWIGLU_CASES = [
    # (name,         n,            description)
    ("n3072",        3072,         "Qwen3-0.6B intermediate, 1 token"),
    ("n3072x16",     3072 * 16,    "Qwen3-0.6B intermediate, seq=16"),
    ("n9728",        9728,         "Qwen3-4B  intermediate, 1 token"),
    ("n777",         777,          "non-multiple of block_size 256"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Saving helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_fp16_bin(t: torch.Tensor, path: str) -> None:
    """Save tensor as raw float16 bytes (no header). Readable with fread in C++."""
    arr = t.cpu().contiguous().numpy().astype(np.float16)
    arr.tofile(path)


def save_meta(meta: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Per-kernel generators
# ──────────────────────────────────────────────────────────────────────────────

def gen_rmsnorm(outdir: str, device: str, seed: int = 10) -> list[dict]:
    torch.manual_seed(seed)
    results = []
    for name, rows, cols, desc in RMSNORM_CASES:
        case_dir = os.path.join(outdir, "rmsnorm", name)
        ensure_dir(case_dir)

        x      = torch.randn(rows, cols, dtype=torch.float16, device=device)
        weight = (torch.randn(cols, dtype=torch.float16, device=device) * 0.5 + 1.0)
        out    = ref_rmsnorm(x, weight)

        eps = 1e-5
        save_fp16_bin(x,      os.path.join(case_dir, "x.fp16.bin"))
        save_fp16_bin(weight, os.path.join(case_dir, "weight.fp16.bin"))
        save_fp16_bin(out,    os.path.join(case_dir, "output.fp16.bin"))
        save_meta(
            {"kernel": "rmsnorm", "rows": rows, "cols": cols, "eps": eps,
             "dtype": "float16", "desc": desc},
            os.path.join(case_dir, "meta.json"),
        )

        # Spot-check: compare FP16 output against FP32 recomputation
        err = _max_err_vs_fp32_rmsnorm(x, weight, out, eps)
        results.append({"case": name, "desc": desc, "max_err": err})

    return results


def gen_softmax(outdir: str, device: str, seed: int = 20) -> list[dict]:
    torch.manual_seed(seed)
    results = []
    for name, rows, cols, desc in SOFTMAX_CASES:
        case_dir = os.path.join(outdir, "softmax", name)
        ensure_dir(case_dir)

        # Scale ∈ [-3, 3] — stays in FP16 range and avoids degenerate distributions
        x   = torch.randn(rows, cols, dtype=torch.float16, device=device) * 3.0
        out = ref_softmax(x)

        save_fp16_bin(x,   os.path.join(case_dir, "x.fp16.bin"))
        save_fp16_bin(out, os.path.join(case_dir, "output.fp16.bin"))
        save_meta(
            {"kernel": "softmax", "rows": rows, "cols": cols,
             "dtype": "float16", "desc": desc},
            os.path.join(case_dir, "meta.json"),
        )

        err = _max_err_vs_fp32_softmax(x, out)
        results.append({"case": name, "desc": desc, "max_err": err})

    return results


def gen_swiglu(outdir: str, device: str, seed: int = 30) -> list[dict]:
    torch.manual_seed(seed)
    results = []
    for name, n, desc in SWIGLU_CASES:
        case_dir = os.path.join(outdir, "swiglu", name)
        ensure_dir(case_dir)

        gate = torch.randn(n, dtype=torch.float16, device=device) * 2.0
        up   = torch.randn(n, dtype=torch.float16, device=device) * 2.0
        out  = ref_swiglu(gate, up)

        save_fp16_bin(gate, os.path.join(case_dir, "gate.fp16.bin"))
        save_fp16_bin(up,   os.path.join(case_dir, "up.fp16.bin"))
        save_fp16_bin(out,  os.path.join(case_dir, "output.fp16.bin"))
        save_meta(
            {"kernel": "swiglu", "n": n, "dtype": "float16", "desc": desc},
            os.path.join(case_dir, "meta.json"),
        )

        err = _max_err_vs_fp32_swiglu(gate, up, out)
        results.append({"case": name, "desc": desc, "max_err": err})

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Spot-check helpers: compare FP16 PyTorch output vs FP32 PyTorch recomputation.
# This validates the reference itself is accurate (no FP16 precision loss in ref).
# ──────────────────────────────────────────────────────────────────────────────

def _max_err_vs_fp32_rmsnorm(x, weight, out_fp16, eps):
    x32 = x.float(); w32 = weight.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    ref_fp32 = (x32 * torch.rsqrt(var + eps) * w32)
    return (out_fp16.float() - ref_fp32).abs().max().item()

def _max_err_vs_fp32_softmax(x, out_fp16):
    ref_fp32 = F.softmax(x.float(), dim=-1)
    return (out_fp16.float() - ref_fp32).abs().max().item()

def _max_err_vs_fp32_swiglu(gate, up, out_fp16):
    ref_fp32 = F.silu(gate.float()) * up.float()
    return (out_fp16.float() - ref_fp32).abs().max().item()


# ──────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_results(kernel: str, results: list[dict]) -> None:
    print(f"\n── {kernel} " + "─" * (50 - len(kernel)))
    tol = 1e-3
    for r in results:
        status = "ok" if r["max_err"] < tol else "WARN"
        print(f"  [{status}] {r['case']:<22}  ref_err={r['max_err']:.2e}  ({r['desc']})")


# ──────────────────────────────────────────────────────────────────────────────
# C++ loading instructions (printed at end)
# ──────────────────────────────────────────────────────────────────────────────

CPPLOAD_SNIPPET = """
── How to load in C++ ─────────────────────────────────────────────────────
  // Load reference output for rmsnorm rows=4 cols=128:
  std::vector<half> ref(rows * cols);
  FILE* f = fopen("tests/reference_data/rmsnorm/rows4_cols128/output.fp16.bin", "rb");
  fread(ref.data(), sizeof(half), ref.size(), f);
  fclose(f);
  // Then compare against your CUDA kernel output.
────────────────────────────────────────────────────────────────────────────
"""

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default="tests/reference_data",
                        help="Root output directory (default: tests/reference_data)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device (default: cuda if available, else cpu)")
    args = parser.parse_args()

    print(f"PyTorch : {torch.__version__}")
    print(f"Device  : {args.device}")
    print(f"Out dir : {args.outdir}")

    rmsnorm_res = gen_rmsnorm(args.outdir, args.device)
    softmax_res = gen_softmax(args.outdir, args.device)
    swiglu_res  = gen_swiglu (args.outdir, args.device)

    _print_results("RMSNorm", rmsnorm_res)
    _print_results("Softmax", softmax_res)
    _print_results("SwiGLU",  swiglu_res)

    # Write global manifest
    manifest = {
        "rmsnorm": [r["case"] for r in rmsnorm_res],
        "softmax": [r["case"] for r in softmax_res],
        "swiglu":  [r["case"] for r in swiglu_res],
    }
    manifest_path = os.path.join(args.outdir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total = sum(len(v) for v in manifest.values())
    print(f"\n{total} reference cases written → {args.outdir}")
    print(f"Manifest → {manifest_path}")
    print(CPPLOAD_SNIPPET)

    # Non-zero exit if any ref_err >= 1e-3 (reference itself is suspect)
    all_ok = all(
        r["max_err"] < 1e-3
        for results in [rmsnorm_res, softmax_res, swiglu_res]
        for r in results
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
