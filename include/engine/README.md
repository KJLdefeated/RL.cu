# include/engine/ — Inference Engine

vLLM-style serving engine with continuous batching, paged KV cache, and CUDA graph execution.

---

## Architecture Overview

```
                           ┌─────────────────────────────┐
                           │         LLMEngine           │
                           │  add_request() / step()     │
                           └──────┬──────────┬───────────┘
                                  │          │
                     ┌────────────▼──┐  ┌────▼──────────┐
                     │   Scheduler   │  │  ModelRunner   │
                     │  batch policy │  │  GPU execution │
                     └──────┬────────┘  └────┬──────────┘
                            │                │
                     ┌──────▼────────┐  ┌────▼──────────┐
                     │ BlockManager  │  │  Qwen3Model   │
                     │ KV block pool │  │  fwd + sample │
                     └───────────────┘  └───────────────┘
```

---

## Components

### LLMEngine (`llm_engine.h`)

Top-level API. Owns the scheduler, model runner, and tokenizer.

- `add_request(prompt, sampling_params)` — tokenize and enqueue.
- `step()` — run one iteration of continuous batching (see below). Returns finished completions.
- `sleep()` / `wakeup()` — release / re-acquire KV cache for training (GRPO uses this to share GPU memory between inference rollouts and gradient updates).
- `generate_ids(prompts, params)` — convenience wrapper: enqueue all prompts, loop `step()` until done.

### Scheduler (`scheduler.h`)

Two-phase continuous batching scheduler. Each `step()` call runs both phases:

**Phase 1 — Decode**: Schedule all currently running sequences for one decode step. If the KV pool is near capacity, preempt sequences from the back of the queue (lowest priority) to free blocks. Preempted sequences return to the front of the waiting queue.

**Phase 2 — Prefill**: Fill vacated batch slots from the waiting queue. Checks three constraints per candidate:
1. Padded token count `<= max_num_batched_tokens`
2. KV blocks available (`can_allocate`)
3. Post-prefill may_append blocks reserved (when `prompt_len % block_size == 0`)

This two-phase design keeps the decode batch at max capacity — as soon as a slot opens, a new sequence fills it in the same call.

**Batch slot management**: Fixed-size boolean array `slot_used[]`. Each running sequence gets an assigned `batch_slot` (0-indexed) that maps to its position in KV cache block tables and sampler buffers. Freed on finish or preemption.

### BlockManager (`block_manager.h`)

Physical KV block allocator with prefix caching:

- **Free list**: deque-based LIFO stack of physical block IDs.
- **Prefix caching**: Sealed blocks (full `KV_BLOCK_SIZE=16` tokens) are hashed with xxhash. Identical prefix blocks across different sequences share the same physical block (ref-counted).
- **Preemption**: Deallocates all blocks; hash entries removed to prevent stale cache hits.
- **Fork** (GRPO): Increments ref counts on shared blocks — no KV data copy.

### ModelRunner (`model_runner.cuh`)

Bridges the scheduler to GPU execution:

- **Initialization**: Loads model weights, computes KV budget from free GPU memory, allocates sampler workspace.
- **Warmup**: Runs a dummy forward pass to trigger JIT compilation and cuBLAS autotuning.
- **CUDA graph capture**: Captures decode forward pass for batch buckets `{1, 2, 4, 8, ..., 256}`. Each graph includes D2D copies for input buffers, all 28 layer forwards, logit projection, and sampling.
- **run(batch, is_prefill)**: Dispatches to `qwen3_prefill`, `qwen3_decode_graph`, or `qwen3_decode` (eager fallback). Returns sampled token IDs.

---

## Request Lifecycle

```
add_request("What is 2+2?")
  │
  ▼
Waiting Queue ──[schedule_prefill]──> Running (prefill step)
  │                                      │
  │                                      ▼
  │                                   Running (decode steps)
  │                                      │
  │                          ┌───────────┼──────────┐
  │                          ▼           ▼          ▼
  │                    EOS token    max_len     preemption
  │                          │           │          │
  │                          ▼           ▼          ▼
  │                      FINISHED    FINISHED   back to
  │                      (output)    (output)   Waiting
  └──────────────────────────────────────────────────┘
```

---

## CUDA Graph Execution

Decode steps use captured CUDA graphs to eliminate per-step kernel launch overhead (~400 individual launches -> 1 `cudaGraphLaunch`).

**Bucket selection**: For batch size B, use the smallest captured bucket `B_bucket >= B`.

**Ghost padding**: Slots `B..B_bucket-1` get `token=0, slot=-1, seq_len=0`. The KV cache skips `slot=-1` writes; attention produces no output for `seq_len=0`. No results are read from ghost rows.

**Per-launch update**: H2D copies to graph input buffers (`g_token_ids`, `g_pos_ids`, `g_slot_map`, `g_block_tables`, `g_seq_lens`), then `cudaGraphLaunch`. The D2D copies from these buffers to model-internal pointers are captured inside the graph.

---

## Key Design Decisions

1. **Sleep/wakeup for GRPO**: The engine releases its KV cache during training so the optimizer and gradients can use that GPU memory, then re-allocates with a fresh budget on wakeup.

2. **Slot-based batch indexing**: Each sequence gets a fixed batch slot (not position-in-batch). This allows CUDA graph input buffers to be indexed by slot, avoiding re-packing when sequences finish mid-batch.

3. **Prefix caching with xxhash**: Reduces prefill cost when multiple GRPO rollouts share the same prompt prefix. Sealed blocks are identified by content hash, not sequence ID.

4. **Preemption over OOM**: When the KV pool is tight, the scheduler preempts the lowest-priority running sequence rather than failing. The preempted sequence re-enters the waiting queue at the front.
