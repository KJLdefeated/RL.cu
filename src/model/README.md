# src/model/ — Model Implementation

Full forward/backward pass orchestration for Qwen3 and the paged KV cache.
Headers are in `include/model/`.

---

## Files

| File | Purpose |
|------|---------|
| `qwen3.cu` | Layer-level and full-model forward pass (prefill + decode + CUDA graph decode), backward pass, training forward, weight init, buffer management |
| `kv_cache.cu` | Paged KV cache: init, append, fork (copy-on-write), free, reshape_and_cache kernel |

---

## qwen3.cu — Forward Pass Architecture

### Per-Layer Pipeline

Each of the 28 transformer layers executes the same sequence:

```
                    ┌─────────────────────────────────────────────┐
  d_hidden ────────>│ save residual ─> RMSNorm(input_layernorm)   │
                    │                                             │
                    │ Q = hidden @ q_proj    [T, 2048]            │
                    │ K = hidden @ k_proj    [T, 1024]            │
                    │ V = hidden @ v_proj    [T, 1024]            │
                    │                                             │
                    │ Q = RMSNorm(Q, q_norm) ─> RoPE(Q)           │  QK-Norm BEFORE RoPE
                    │ K = RMSNorm(K, k_norm) ─> RoPE(K)           │
                    │                                             │
                    │ KV cache append (paged, per-layer slice)    │
                    │                                             │
                    │ attn_out = FA2_prefill(Q, K, V)             │  or paged_decode
                    │ d_hidden = attn_out @ o_proj + residual     │
                    │                                             │
                    │ save residual ─> RMSNorm(post_attn_norm)    │
                    │                                             │
                    │ gate = hidden @ gate_proj   [T, 3072]       │
                    │ up   = hidden @ up_proj     [T, 3072]       │
                    │ d_hidden = SwiGLU(gate, up) @ down_proj     │
                    │           + residual                        │
                    └─────────────────────────────────────────────┘
```

After all 28 layers: `final_norm -> lm_head projection (tied embed weights) -> logits`.

### Prefill vs Decode

- **Prefill** (`qwen3_prefill`): Processes full prompt sequences. Pads variable-length inputs to `S_max` with `slot=-1` (KV cache skips these). Uses `gather_at_offsets_kernel` to extract each sequence's last real token for logit projection.
- **Decode** (`qwen3_decode`): Single-token generation. Packs `block_tables`/`seq_lens` into compact `0..B-1` arrays by `batch_slot` for GPU. Uses paged attention to read cached KV.
- **Graph decode** (`qwen3_decode_graph`): Selects the smallest captured CUDA graph bucket `>= B`, ghost-pads extra slots (`token=0, slot=-1, seq_len=0`), updates graph input buffers via H2D, then `cudaGraphLaunch`.

### Training Forward

- `qwen3_forward_train`: Full forward pass storing intermediate activations (hidden states, Q, K, V, gate, up, attention output, LSE) per layer for backward.
- `log_softmax_gather_kernel`: Fused log-softmax + target gather for cross-entropy loss. Warp shuffle + shared memory reduction over V=151936.
- Logits computed in chunks of 256 tokens to bound scratch memory.

### Backward Pass

- `qwen3_backward`: Reverse layer traversal. Each layer backprops through: down_proj -> SwiGLU -> gate/up_proj -> post_attn_norm -> o_proj -> FA2 backward -> RoPE backward -> QK-norm backward -> Q/K/V proj -> input_norm -> residual add.
- All gradient accumulation in FP32 via `linear_backward_half` (cuBLAS `beta=1`).

---

## kv_cache.cu — Paged KV Cache

Block-based virtual memory for KV states, inspired by vLLM's PagedAttention.

### Layout

```
Pool:  [num_layers, total_blocks, num_kv_heads, KV_BLOCK_SIZE=16, head_dim=128]
       One flat cudaMalloc per pool (k_pool, v_pool).

Slot mapping:  slot = physical_block * 16 + in_block_offset
               -1 = skip (no KV write)
```

### Operations

| Function | Description |
|----------|-------------|
| `paged_kv_cache_init` | Allocate pools + block tables + free stack |
| `paged_kv_cache_append_slot` | CPU-side: pop free block if needed, compute slot mapping |
| `reshape_and_cache_half_kernel` | GPU: scatter K/V tokens into cache at given slots |
| `paged_kv_cache_fork` | Copy block-table pointers only (zero KV copy) for GRPO rollouts |
| `qwen3_free_seq_slot` | Return blocks to free stack, zero seq_lens and block_tables |

### Fork for GRPO

When generating G completions from the same prompt, `fork()` copies only the block table pointers (not KV data). All G sequences share the prompt's physical KV blocks. New blocks are allocated only for divergent completion tokens. This makes GRPO rollout memory O(G * completion_len) instead of O(G * full_seq_len).
