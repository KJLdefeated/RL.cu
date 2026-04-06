#!/usr/bin/env python3
"""
bench_hf_forward.py — Benchmark HuggingFace Transformers prefill forward pass
                      for Qwen3-0.6B and compare against our CUDA kernel times.

Measures:
  - Full model forward (all 28 layers)
  - Attention-only forward time (estimated via hooks)
  - Throughput in tokens/s and TFLOPS (attention)

Usage:
    python python_scripts/bench_hf_forward.py
    python python_scripts/bench_hf_forward.py --model model_weights/Qwen3-0.6B
    python python_scripts/bench_hf_forward.py --dtype float16 --compile

Requirements:
    pip install transformers torch
"""

import argparse
import time
import sys

import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR   = "model_weights/Qwen3-0.6B"
WARMUP      = 5
ITERS       = 20
H_Q         = 16
H_KV        = 8
HEAD_DIM    = 128

# (B, S) pairs to sweep — matches our CUDA benchmark configs
CONFIGS = [
    (1,  128),
    (1,  512),
    (1, 2048),
    (8, 2048),
]


# ---------------------------------------------------------------------------
# Attention timing hook
# ---------------------------------------------------------------------------
class AttentionTimer:
    """Hooks into every attention module to measure cumulative attention time."""

    def __init__(self):
        self.hooks   = []
        self.elapsed = 0.0   # seconds
        self._start  = {}

    def _pre_hook(self, module, args):
        torch.cuda.synchronize()
        self._start[id(module)] = time.perf_counter()

    def _post_hook(self, module, args, output):
        torch.cuda.synchronize()
        self.elapsed += time.perf_counter() - self._start.pop(id(module), 0)

    def register(self, model):
        for name, module in model.named_modules():
            if "attention" in type(module).__name__.lower() and hasattr(module, "forward"):
                self.hooks.append(module.register_forward_pre_hook(self._pre_hook))
                self.hooks.append(module.register_forward_hook(self._post_hook))

    def reset(self):
        self.elapsed = 0.0

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(model, tokenizer, B: int, S: int, dtype, device,
                  attn_timer: AttentionTimer) -> dict:
    """Run warmup + timed iterations for a single (B, S) config."""
    # Build synthetic input_ids
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (B, S), device=device)

    def forward_once():
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False)
        return out

    # Warmup
    for _ in range(WARMUP):
        forward_once()
    torch.cuda.synchronize()

    # --- Full forward ---
    t0 = time.perf_counter()
    for _ in range(ITERS):
        forward_once()
    torch.cuda.synchronize()
    full_us = (time.perf_counter() - t0) / ITERS * 1e6

    # --- Attention time (accumulated across all 28 layers) ---
    attn_timer.reset()
    for _ in range(ITERS):
        forward_once()
    torch.cuda.synchronize()
    attn_us = attn_timer.elapsed / ITERS * 1e6

    # Compute attention TFLOPS using same formula as our CUDA benchmark:
    #   flops = 2 * B * H_q * S^2 * D * 2
    flops    = 2.0 * B * H_Q * S * S * HEAD_DIM * 2
    # multiply by num_layers (HF runs all layers, our kernel is per-layer)
    num_layers = model.config.num_hidden_layers
    attn_tflops = (flops * num_layers) / (attn_us * 1e-6) / 1e12

    tok_per_sec = B * S / (full_us * 1e-6)

    return {
        "B": B, "S": S,
        "full_us": full_us,
        "attn_us": attn_us,
        "attn_tflops": attn_tflops,
        "tok_per_sec": tok_per_sec,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark HF Qwen3 forward pass")
    parser.add_argument("--model",   default=MODEL_DIR,
                        help="Path to model dir (default: %(default)s)")
    parser.add_argument("--dtype",   default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype (default: %(default)s)")
    parser.add_argument("--compile", action="store_true",
                        help="Apply torch.compile (torch 2.x)")
    parser.add_argument("--sdpa",    action="store_true",
                        help="Force SDPA attention (instead of eager)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    dtype  = {"float16": torch.float16,
               "bfloat16": torch.bfloat16,
               "float32": torch.float32}[args.dtype]

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"Loading {args.model} …", flush=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers",
              file=sys.stderr)
        sys.exit(1)

    attn_impl = "sdpa" if args.sdpa else "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.compile:
        print("Compiling model with torch.compile …", flush=True)
        model = torch.compile(model)

    num_layers = model.config.num_hidden_layers
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {num_layers} layers  {num_params:.0f} M params  "
          f"dtype={args.dtype}  attn={attn_impl}")
    print(f"GPU:   {torch.cuda.get_device_name(0)}")
    print()

    # ── Register attention hooks ────────────────────────────────────────────
    attn_timer = AttentionTimer()
    attn_timer.register(model)

    # ── Sweep configs ───────────────────────────────────────────────────────
    header = (f"{'Config':<18}  {'full(ms)':>9}  {'attn(ms)':>9}  "
              f"{'tok/s':>9}  {'attnTFLOPS':>11}  {'note'}")
    print(header)
    print("-" * len(header))

    for B, S in CONFIGS:
        try:
            r = run_benchmark(model, tokenizer, B, S, dtype, device, attn_timer)
        except torch.cuda.OutOfMemoryError:
            print(f"  B={B} S={S:<6}  OOM — skipping")
            torch.cuda.empty_cache()
            continue

        full_ms = r["full_us"] / 1000
        attn_ms = r["attn_us"] / 1000
        note = ""
        if B == 8 and S == 2048:
            note = "← our CUDA bench config"
        print(f"  B={B} S={S:<6}  "
              f"{full_ms:>9.2f}  {attn_ms:>9.2f}  "
              f"{r['tok_per_sec']:>9.0f}  {r['attn_tflops']:>11.2f}  {note}")

    attn_timer.remove()

    # ── Summary note ────────────────────────────────────────────────────────
    print()
    print("Notes:")
    print("  full(ms)      = total forward including all 28 layers (no KV cache)")
    print("  attn(ms)      = cumulative attention module time across all layers")
    print("  attnTFLOPS    = effective TFLOPS for attention (all layers, same formula")
    print("                  as our CUDA benchmark: 2*B*H_q*S^2*D*2 per layer)")
    print("  Our CUDA kernel (B=8 S=2048): 42104 us total / 28 layers ≈ 1504 us/layer")
    print()
    print("  To compare per-layer attention time against our kernel:")
    print("    HF per-layer attn ≈ attn(ms) / 28 layers")


if __name__ == "__main__":
    main()
