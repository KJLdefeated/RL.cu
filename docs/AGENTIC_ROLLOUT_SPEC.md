# Multi-Turn Agentic Rollout — Implementation Spec

Spec for §7 of [agentic_rl_plan.md](../agentic_rl_plan.md): turn the single-shot GRPO
loop into a multi-turn, execution-grounded rollout with cross-turn KV reuse, on
1.7B single-GPU. Written against the current code (file:line refs are live).

---

## Progress (live)

| Item | Status |
|------|--------|
| Stage 0 — continued-prefill spike ([test_chunked_prefill.cu](../tests/kernels/test_chunked_prefill.cu)) | ✅ done — proved chunk-only prefill diverges (~0.5 err), cache-reading path needed |
| Risk #1 kernel — `launch_flash_attention_prefill_paged` ([attention.cu](../src/kernels/attention.cu)) | ✅ done — matches golden ~5e-5 (GQA, block boundaries, empty prefix, multi-seq, padding) |
| Stage 2 wiring — `qwen3_prefill` continued-prefill path | ✅ done — routes to the paged kernel when any seq has `num_cached_tokens>0` ([test_continued_prefill_model.cu](../tests/models/test_continued_prefill_model.cu)) |
| Engine resume API — `PAUSED`/`rollout_*`, scheduler resume + continued-prefill scheduling, `BlockManager::append_blocks_for` | ✅ done — multi-turn rollout via engine == single-shot reference token-for-token through 12 decode steps ([test_rollout_engine.cu](../tests/models/test_rollout_engine.cu)) |
| **Prerequisite bug fix** — Flash-Decode split-K NaN (see below) | ✅ done — was blocking 100% of generation on this box |
| CP env + sandboxed Python runner — `Env` / `CodeContestsEnv` / `extract_code` ([env.h](../include/training/env.h)) + `SubprocessPythonRunner` w/ setrlimit + unshare net ([code_runner.h](../include/training/code_runner.h)) | ✅ done — 16/16 tests pass: AllOrNothing + FractionPassed reward, sample-test feedback formatting, hard timeout, RLIMIT_AS, CLONE_NEWNET isolation verified on this box ([test_cp_env.cpp](../tests/training/test_cp_env.cpp)) |
| Trainer rewrite (`Episode`, rollout driver, obs masking) | ⬜ not started (§3.1, 3.3, 3.5–3.8) |
| CP dataset loader (CodeContests / TACO → `CPProblem` list) | ⬜ not started (gated on §6.2) |
| Stage 3 — DP=2 | ⬜ not started |

### Prerequisite bug fixed: Flash-Decode split-K NaN (not part of the agentic spec, but blocked it)

Found while bringing up `test_llmengine` on the H100 box: **every** greedy generation
collapsed to token id 0 ("!") after the first token. Root cause — the **paged decode
split-K kernel** ([attention.cu](../src/kernels/attention.cu)
`flash_decode_splitk_kernel` / `flash_decode_reduce_kernel`) produced **all-NaN
logits** whenever `num_splits > num_blocks`:

- The launcher picks `num_splits` from `total_heads` (16 for batch=1), independent of
  context length. At the first decode step (~3 KV blocks), **13 of 16 splits are empty**.
- An empty split ran the cross-warp merge with `global_max = -inf`, computing
  `expf(-inf - (-inf)) = expf(NaN) = NaN` → wrote NaN `partial_sum`/`partial_out`.
- The reduce then did `NaN × 0 = NaN`, poisoning the output → NaN logits → sampler
  returns id 0.

It escaped CI because the split-K unit tests only ever used `num_splits ≤ num_blocks`,
and `test_llmengine`'s correctness section was commented out in `main`.

**Fix:** empty splits now write neutral partials (`max=-inf, sum=0, out=0`) instead of
the NaN merge; the reduce skips `-inf` splits and zero-outputs the all-empty case.
After the fix: 11/11 `test_llmengine` pass, 16.8k tok/s @ batch 64.

**Relevance to this spec:** continued prefill resumes a sequence and immediately
decodes it — multi-turn rollouts do this every turn with *short* new contexts, i.e.
exactly the `num_splits > num_blocks` regime that triggered this bug. Without the fix,
no rollout (agentic or single-turn) could emit a valid token. A regression guard for
the short-context/empty-split regime belongs in the split-K unit tests.

---

## 0. TL;DR of the change

Today's rollout is **one** call: `engine_->generate_ids(prompts, sp, G)` runs every
sequence to EOS/maxlen in a single pass, then we score once. Multi-turn requires
the policy to *pause* mid-rollout, hand control to an environment (run code on
public tests), feed the result back as observation tokens, and *resume* — N times
per episode, with the KV cache for everything-so-far kept in place.

The single-shot assumption is concentrated in exactly **three** places:

| # | Symbol | File:line | Assumption that breaks |
|---|--------|-----------|------------------------|
| 1 | `GRPOTrainer::generate()` | [GRPO_trainer.h:426](../include/training/GRPO_trainer.h#L426) | one `generate_ids` → complete completion, then `sleep()` |
| 2 | `GRPOTrainer::Generation` + `build_train_batch()` | [GRPO_trainer.h:156](../include/training/GRPO_trainer.h#L156), [:548](../include/training/GRPO_trainer.h#L548) | a single contiguous `[prompt][completion]` span to mask |
| 3 | `LLMEngine::generate_ids` / `step()` | [llm_engine.h:137](../include/engine/llm_engine.h#L137), [:62](../include/engine/llm_engine.h#L62) | a finished sequence is **deleted** and its KV/slot freed (no resume) |

Everything downstream of `build_train_batch` — advantages, `grpo_loss_kernel`,
`grpo_training_step`, optimizer — is **per-sequence** and needs **no change**:
`grpo_loss_kernel` already keys advantage by `b = t / S`
([GRPO_trainer.h:72](../include/training/GRPO_trainer.h#L72)), and an episode is
just one sequence `b`. The work is entirely in (a) the engine's lifecycle and (b)
the trainer's rollout driver + loss masking.

---

## 1. The turn / observation / KV-reuse interface

### 1.1 Turn boundary = existing EOS stop

Qwen3 chat format ends every assistant message with `<|im_end|>` (151645), which is
already `config.eos` and already stops a sequence in `Scheduler::postprocess`
([scheduler.h:74](../include/engine/scheduler.h#L74)). **We reuse that as the turn
boundary** — no stop-string / partial-BPE detection needed. A "turn" = generate
until `<|im_end|>` or `max_new_tokens`.

An episode's realized token stream is the interleaving:

```
[ prompt ][ asst turn 1 ][ obs 1 ][ asst turn 2 ][ obs 2 ] … [ asst turn K ]
  mask 0     mask 1         mask 0    mask 1         mask 0      mask 1
```

- **asst spans** = policy-generated tokens → loss-active.
- **obs spans** = environment feedback, chat-wrapped (`<|im_start|>user … <|im_end|><|im_start|>assistant`) → loss-masked, but **fully attended** (they're real context the policy conditioned on).

### 1.2 KV-reuse contract (the headline)

Across turns we want to prefill **only the new observation span**, not the whole
history. Two viable mechanisms, both needing the same kernel capability:

- **Option A — resident KV (recommended).** The `Sequence*` stays alive across
  turns, keeping its `batch_slot` and `block_table`. Resuming appends obs tokens and
  prefills only them; existing KV is untouched. No hashing, no prefix re-walk.
- **Option B — prefix-cache resubmit.** Let the sequence finish, re-`add_request`
  the full `prompt+turn1+obs1+…`; `BlockManager` prefix caching sets
  `num_cached_tokens` so `schedule_prefill` prefills only `uncached =
  size − num_cached_tokens` ([scheduler.h:134](../include/engine/scheduler.h#L134)).
  Reuses existing machinery but re-walks/hashes the prefix each turn.

> **Decision: Option A.** It's the genuine "prefill only the new tokens" story and
> avoids per-turn hashing. Option B is the fallback if resident-slot bookkeeping
> proves fiddly.

**Both options depend on one kernel capability — continued (chunked) prefill:** a
prefill of `m` new tokens whose attention reads K/V for the `n` already-cached
tokens from the paged cache (so new Q attends cached+new K/V), writing only the new
tokens' KV. This is **Risk #1** (§5) and gets a spike before anything else.

### 1.3 Proposed engine API (`llm_engine.h`)

A KV-preserving rollout surface that hands episode lifecycle to the trainer:

```cpp
struct TurnOutput {
    std::vector<int64_t> new_tokens;  // tokens emitted THIS turn (incl. trailing <|im_end|>)
    bool hit_eos   = false;           // ended on <|im_end|>  → episode may continue
    bool truncated = false;           // ended on max_new_tokens → likely failed turn
};

class LLMEngine {
    // Register a new episode. Sequence parked in `paused_`, KV not yet allocated.
    int64_t rollout_add(const std::vector<int64_t>& prompt_tokens, SamplingParams sp);

    // Append observation tokens to a paused episode and queue its next turn.
    // KV/block_table/batch_slot preserved; num_cached_tokens := pre-append length
    // so only obs_tokens are prefilled (continued prefill).
    void rollout_continue(int64_t seq_id, const std::vector<int64_t>& obs_tokens);

    // Run continuous batching until every *active* episode has emitted a full turn
    // (hit_eos) or truncated. Paused on turn-end (KV retained), NOT deleted.
    std::map<int64_t, TurnOutput> rollout_run_turns();

    // Episode done: free KV + batch slot.
    void rollout_finish(int64_t seq_id);
};
```

`rollout_run_turns()` loops `step()` until `running.empty()`, but in **agentic
mode** `postprocess` routes EOS/maxlen sequences to a `paused_` deque (KV+slot kept)
instead of `FINISHED`+delete.

---

## 2. Engine / scheduler changes (concrete edits)

### 2.1 `sampling_parmas.h`
- Add `SeqStatus::PAUSED` ([sampling_parmas.h:16](../include/model/sampling_parmas.h#L16)).
- `SamplingParams`: add `bool agentic = false;` — when set, turn-end pauses instead of finishing.
- `Sequence`: add `int turn_count = 0;` and a per-turn record for the trainer:
  `std::vector<std::pair<int,int>> asst_spans;` (start,end of each assistant span in `token_ids`). Pushed in `rollout_continue`/`postprocess`.

### 2.2 `scheduler.h`
- `std::deque<Sequence*> paused_;` + `std::vector<Sequence*> take_paused();`
- `postprocess` ([:64](../include/engine/scheduler.h#L64)): if `seq.sampling_params.agentic` and the stop is EOS/maxlen, set `status=PAUSED`, remove from `running`, push to `paused_`, **do not** `deallocate`/`free_slot` (KV+slot retained).
- New `resume(Sequence* seq, const std::vector<int64_t>& obs)`:
  - `seq->num_cached_tokens = seq->size();` (everything so far is already in KV)
  - append `obs` to `token_ids`, record the obs span, open a new asst span at the new tail
  - `status=WAITING`; push to `waiting` (it **keeps** `batch_slot`).
- `schedule_prefill` ([:123](../include/engine/scheduler.h#L123)): detect a **resumed** seq (`batch_slot>=0 && num_cached_tokens>0`) → *continued-prefill* branch: skip `alloc_slot()`, call a new `block_manager.append_blocks_for(seq)` that only allocates blocks for the `uncached` tail, leave `num_cached_tokens` intact. Existing fresh-prefill path unchanged.

### 2.3 `block_manager.h`
- Add `append_blocks_for(Sequence&)`: extend `block_table` to cover
  `ceil(size / block_size)` blocks, allocating only the delta. (The decode-time
  `may_append` already does single-block growth; generalize to a multi-block span.)

### 2.4 `model_runner.cuh` / `qwen3.cu` — **continued prefill** (Risk #1) ✅ DONE
The spike confirmed the chunk-only path diverges, so a cache-reading prefill kernel
was built and `qwen3_prefill` now routes to it:
- `launch_flash_attention_prefill_paged` ([attention.cu](../src/kernels/attention.cu)):
  the `S_q>1` sibling of paged decode — new-chunk queries attend over the paged KV
  cache causally by absolute position. Verified to ~5e-5 vs golden.
- `qwen3_prefill` ([qwen3.cu](../src/model/qwen3.cu)): when **any** seq in the batch
  has `num_cached_tokens > 0`, the whole batch takes the paged path (`m->cont_prefill`
  flag → `qwen3_layer_forward` calls the paged kernel with per-token `q_seq_idx` /
  `q_abs_pos` and the per-slot block table). Q is built over the `uncached` tail only;
  KV is written for the tail via `reshape_and_cache`; attention reads prefix+tail from
  the cache. Pure-fresh batches (`num_cached==0` everywhere) keep the faster dense
  WMMA FA2 path. Validated by [test_continued_prefill_model.cu](../tests/models/test_continued_prefill_model.cu):
  2-chunk prefill last-token logits == single full prefill.
- **Remaining:** this is the *model*-level capability. The *engine* still never sets
  `num_cached_tokens>0` (no resume API yet) — §2.1–2.3 wire the scheduler/engine to
  actually drive it across turns.

---

## 3. Trainer changes (`GRPO_trainer.h`)

### 3.1 New types
Replace `Generation` ([:156](../include/training/GRPO_trainer.h#L156)) with:

```cpp
struct Episode {
    int prompt_idx;
    std::vector<int64_t> tokens;                  // full realized stream (exact ids)
    std::vector<std::pair<int,int>> asst_spans;   // [start,end) of each policy turn
    int  num_turns   = 0;
    bool solved      = false;   // env says hidden tests passed
    bool truncated   = false;   // hit turn/length budget without solving
    std::string last_code;      // most recent extracted code block (for reward)
};
```

Store **exact token ids** (never re-tokenize text for training) so the training
forward sees a byte-identical sequence to what was generated → preserves the
zero train-infer-mismatch property that is the whole pitch.

### 3.2 New environment interface (`include/training/env.h`, new file)

```cpp
struct EnvFeedback { std::string observation; bool stop; };  // stop = end episode early
class Env {
public:
    virtual ~Env() = default;
    // Run model-emitted code on PUBLIC/sample tests → observation text the model sees.
    virtual EnvFeedback feedback(const std::string& code, int sample_idx) = 0;
    // Final reward on HIDDEN tests at episode end.
    virtual float reward(const Episode& ep, int sample_idx) = 0;
};
```

CP impl `CodeContestsEnv : Env` wraps a sandboxed `CodeRunner` (§3.4). Reward
shaping (all-or-nothing vs fraction-passed) is a config switch (§4, open decision).

### 3.3 Rollout driver — replaces `generate()` ([:426](../include/training/GRPO_trainer.h#L426))

```cpp
std::vector<Episode> rollout(const std::vector<std::vector<int64_t>>& prompts) {
    engine_->wakeup();
    // init B*G episodes, one engine sequence each (sp.agentic = true)
    std::map<int64_t, Episode*> live;
    for (i in prompts) for (g in G)
        live[engine_->rollout_add(prompts[i], sp)] = new Episode{prompt_idx=i,...};

    for (int turn = 0; turn < cfg.max_turns && !live.empty(); ++turn) {
        auto outs = engine_->rollout_run_turns();          // GPU: all active turns
        std::vector<int64_t> to_continue;
        for (auto& [sid, out] : outs) {                    // CPU: env barrier
            Episode* ep = live[sid];
            ep->append_turn(out.new_tokens);  ep->num_turns++;
            std::string code = extract_code(out.new_tokens);
            bool last_turn = (turn == cfg.max_turns - 1) || out.truncated || code.empty();
            if (last_turn) { finalize(ep); engine_->rollout_finish(sid); live.erase(sid); }
            else {
                auto fb = env_->feedback(code, ep->prompt_idx);
                if (fb.stop) { finalize(ep); engine_->rollout_finish(sid); live.erase(sid); }
                else { engine_->rollout_continue(sid, wrap_observation(fb.observation)); }
            }
        }
    }
    engine_->sleep();
    return collected_episodes;   // size B*G, group-aligned by prompt_idx
}
```

- **v0.1 uses a synchronous env barrier** (all turns generate, then all envs run).
  Env is CPU-cheap (plan §2), so GPU idle during env is acceptable. Overlapping env
  with generation is a later optimization (note it; don't build it now).
- `wrap_observation` emits `<|im_end|><|im_start|>user\n{obs}\n<|im_end|><|im_start|>assistant\n`
  tokens (whatever the dataset's template uses) and returns the **token ids** to append.
- `extract_code` parses the ```` ```python ... ``` ```` block from the turn text.

### 3.4 Sandbox (`include/training/code_runner.h`, new file)
- `class CodeRunner { virtual RunResult run(code, stdin, limits); };`
- Default `SubprocessRunner`: `fork` → `setrlimit` (CPU, AS, FSIZE) + `alarm`
  timeout + **no network** (`unshare(CLONE_NEWNET)` or run under `firejail
  --net=none`/`bubblewrap`), kill on timeout, capture stdout/stderr.
- **Do not hand-roll loosely** (plan §2). Recommend `bubblewrap`/`firejail` over a
  bare `fork` for v0.1; this is its own work item with its own test.

### 3.5 Reward — replace `compute_rewards` ([:489](../include/training/GRPO_trainer.h#L489))
Call `env_->reward(ep, ep.prompt_idx)` per episode (hidden tests). Advantages
([:523](../include/training/GRPO_trainer.h#L523)) and the per-group normalization
are **unchanged** — still G episodes per prompt, group mean/std.

### 3.6 `build_train_batch` — the masking rewrite ([:548](../include/training/GRPO_trainer.h#L548))

Generalize from one completion span to the union of assistant spans. Lay out each
episode's full `tokens`, `target_ids[t]=tokens[t+1]`, and:

```
loss_mask[t] = 1  iff  is_assistant[t+1] == true        (for t in 0 .. len-2)
```

where `is_assistant[]` is built from `ep.asst_spans`. This rule (mask keyed on the
**predicted** token) correctly:
- trains the first token of each turn (predicted from prompt/obs context),
- trains the turn-ending `<|im_end|>`,
- excludes prompt tokens and **all** observation tokens from the gradient,

and stays consistent with `qwen3_forward`'s gather (`d_log_probs[t] = log
p(tokens[t+1] | ≤t)`), exactly as the current single-span code does. Keep the DAPO
**overlong filter**: zero an episode's whole mask if `truncated` and unsolved.

### 3.7 `grpo_training_step` — essentially unchanged ([:610](../include/training/GRPO_trainer.h#L610))
Per-episode advantage + `grpo_loss_kernel` already correct. Only practical change:
`S` is now much larger (prompt + K turns + K observations), so
- `max_seq_len` / KV budget / `max_turns × max_obs_len` must be sized together (§4),
- expect higher `grad_accum_steps`; gradient checkpointing already supports long T.

### 3.8 Config additions (`GRPOConfig`)
`int max_turns = 4;  int max_obs_tokens = 512;  enum RewardShaping {AllOrNothing, FractionPassed} reward_shaping;  std::string sandbox = "bwrap";` plus public/hidden split policy fields.

---

## 4. Open decisions to lock before coding (plan §6)
1. **Reward shaping** — all-or-nothing (clean, sparse) vs fraction-passed (dense,
   gameable). Affects whether the GRPO curve moves; config switch either way.
2. **Dataset** — CodeContests vs TACO; gated on hidden-test reliability + a
   difficulty band the 1.7B solves *sometimes* (reward variance is required — an
   all-pass or all-fail group gives zero advantage, [:537](../include/training/GRPO_trainer.h#L537)).
3. **Turn budget** `max_turns` and **public/hidden split** policy.
4. **Sandbox stack** — `bubblewrap` vs `firejail` vs `seccomp`.

---

## 5. Risks / spikes (do these first)
1. **Continued-prefill kernel (Risk #1).** ✅ **RESOLVED.** Spike showed chunk-only
   prefill diverges; built `launch_flash_attention_prefill_paged` and wired it into
   `qwen3_prefill` (§2.4). 2-chunk prefill logits now == single full prefill.
   *Also surfaced & fixed the split-K decode NaN (see Progress section) — a hard
   prerequisite, since resume→decode hits the short-context split-K regime.*
2. **Resident-slot accounting.** Paused episodes hold `batch_slot`+KV during the env
   barrier. With `B*G` episodes all paused at once, that's `B*G` slots + full KV
   reserved → must fit `max_num_seqs` and the KV budget simultaneously. Size
   `num_prompts*G` against `model_runner` budget up front.
3. **Token exactness.** Training must replay the **generated ids verbatim** (obs ids
   included). Any re-tokenization reintroduces train-infer drift — the one thing
   this project exists to avoid.

---

## 6. Sequencing (de-risked)
- **Stage 0 — spike:** continued-prefill correctness test (Risk #1). No loop yet.
- **Stage 1 — correctness, full re-prefill:** build the multi-turn loop + env + mask
  rewrite, but on `rollout_continue` **re-prefill the whole sequence each turn** (no
  KV reuse). Slow but correct; unblocks the trainer + masking + env + sandbox in
  parallel with kernel work. Validate reward curve moves on 1.7B single-GPU.
- **Stage 2 — KV reuse:** flip `rollout_continue` to resident-KV continued prefill
  (Option A). Measure prefill-tokens-per-turn drop → the headline number.
- **Stage 3 — DP=2:** unchanged rollout, add `ncclAllReduce` on grads (plan §4).

Stages 1 and 0/2 are parallelizable: the trainer/env/sandbox/masking (Stage 1) does
not block on the kernel (Stage 0/2).

---

## 7. New / touched files
| Action | Path | Status |
|--------|------|--------|
| new | `tests/kernels/test_chunked_prefill.cu` — Risk #1 spike (+ regression for the paged kernel) | ✅ done |
| edit | `src/kernels/attention.cu` / `.cuh` — `launch_flash_attention_prefill_paged` (+ split-K NaN fix) | ✅ done |
| edit | `src/model/qwen3.cu` / `include/model/qwen3.h` — continued-prefill routing in `qwen3_prefill` | ✅ done |
| new | `tests/models/test_continued_prefill_model.cu` — 2/3-chunk prefill == full prefill | ✅ done |
| edit | `include/model/sampling_parmas.h` — `PAUSED`, `agentic`, span record | ✅ done |
| edit | `include/engine/scheduler.h` — `paused_`, `resume`, continued-prefill schedule | ✅ done |
| edit | `include/engine/block_manager.h` — `append_blocks_for` | ✅ done |
| edit | `include/engine/llm_engine.h` — `rollout_*` API | ✅ done |
| new  | `tests/models/test_rollout_engine.cu` — multi-turn rollout end-to-end | ✅ done |
| new | `include/training/env.h` — `Env`, `CodeContestsEnv`, `extract_code` | ✅ done |
| new | `include/training/code_runner.h` — `SubprocessPythonRunner` (sandbox) | ✅ done |
| new | `tests/training/test_cp_env.cpp` — env + sandbox end-to-end | ✅ done |
| edit | `include/training/GRPO_trainer.h` — `Episode`, `rollout`, reward, mask rewrite | ⬜ |
| new | `tests/training/train_agentic_grpo.cu` — driver | ⬜ |
| edit | `scripts/prepare_data.py` — `--mode code-rl` | ⬜ |
</content>
