#pragma once
#include <vector>
#include <deque>
#include <algorithm>
#include <cstdint>
#include "model/config.h"
#include "model/sampling_parmas.h"
#include "engine/block_manager.h"

class Scheduler {
private:
    int max_num_seqs;
    int max_num_batched_tokens;
    int max_model_len;
    int64_t eos;
    BlockManager block_manager;
    std::deque<Sequence*> waiting;
    std::deque<Sequence*> running;
    // Agentic rollout: sequences that hit EOS / max_new with agentic=true are
    // parked here with KV + batch_slot retained, awaiting rollout_continue.
    std::deque<Sequence*> paused_;
    std::vector<bool> slot_used;  // tracks which batch slots are in use
    std::vector<int> preempted_slots_;  // batch slots freed by preemption this schedule() call

    int alloc_slot() {
        for (int i = 0; i < (int)slot_used.size(); i++) {
            if (!slot_used[i]) { slot_used[i] = true; return i; }
        }
        throw std::runtime_error("No free batch slots");
    }
    void free_slot(int s) { slot_used[s] = false; }

public:
    Scheduler(const Config& cfg)
        : max_num_seqs(cfg.max_num_seqs),
          max_num_batched_tokens(cfg.max_num_batched_tokens),
          max_model_len(cfg.max_model_len),
          eos(cfg.eos),
          block_manager(cfg.num_kv_blocks, cfg.kv_block_size),
          slot_used(cfg.max_num_seqs, false) {}

    bool is_finished() const {
        return waiting.empty() && running.empty();
    }

    // True when no sequence is *actively* being processed; agentic paused seqs
    // are excluded (they're waiting for caller-driven rollout_continue).
    bool no_active() const {
        return waiting.empty() && running.empty();
    }

    // Agentic rollout API surface used by LLMEngine ────────────────────────
    const std::deque<Sequence*>& paused() const  { return paused_; }
    const std::deque<Sequence*>& waiting_view() const { return waiting; }
    const std::deque<Sequence*>& running_view() const { return running; }
    // Free KV + batch slot for a paused seq that the caller is about to delete.
    // (rollout_finish takes the seq out of paused_ first via take_paused.)
    void release_paused(Sequence* seq) {
        block_manager.deallocate(*seq);
        if (seq->batch_slot >= 0) free_slot(seq->batch_slot);
    }
    Sequence* take_paused(int64_t seq_id) {
        for (auto it = paused_.begin(); it != paused_.end(); ++it) {
            if ((*it)->seq_id == seq_id) {
                Sequence* s = *it; paused_.erase(it); return s;
            }
        }
        return nullptr;
    }
    void erase_paused(int64_t seq_id) {
        for (auto it = paused_.begin(); it != paused_.end(); ++it) {
            if ((*it)->seq_id == seq_id) { paused_.erase(it); return; }
        }
    }
    // Resume a paused agentic seq by appending observation tokens. KV prefix +
    // batch_slot are retained; only the obs span is prefilled on the next step.
    void resume(Sequence* seq, const std::vector<int64_t>& obs_tokens) {
        // Pre-resume size = end of just-closed assistant span; everything in
        // [0, size()) is already in KV → mark as cached.
        seq->num_cached_tokens = seq->size();
        for (int64_t t : obs_tokens) seq->append_token(t);
        // Next assistant turn starts after the observation.
        seq->next_turn_start = seq->size();
        // Reset the max_new_tokens budget to count from the start of the next
        // turn: postprocess uses (num_tokens - num_prompt_tokens) >= max_new.
        seq->num_prompt_tokens = seq->size();
        seq->status = SeqStatus::WAITING;
        waiting.push_front(seq);   // priority over fresh waiting seqs
    }

    void add(Sequence* seq) {
        waiting.push_back(seq);
    }

    void preempt(Sequence* seq) {
        block_manager.deallocate(*seq);
        if (seq->batch_slot >= 0) {
            preempted_slots_.push_back(seq->batch_slot);
            free_slot(seq->batch_slot);
            seq->batch_slot = -1;
        }
        seq->status = SeqStatus::WAITING;
        waiting.push_front(seq);  // re-queue at front for priority
    }

    // Slots freed by preemption in the most recent schedule() call.
    // Caller should invoke model_runner->free_seq_slot() for each.
    const std::vector<int>& preempted_slots() const { return preempted_slots_; }

    // Called after model_runner->run() returns new token IDs.
    // Appends each token, marks finished sequences, frees their blocks.
    void postprocess(std::vector<Sequence*>& batch, const std::vector<int64_t>& new_token_ids, bool is_prefill = false) {
        for (int i = 0; i < (int)batch.size(); ++i) {
            Sequence* seq = batch[i];
            int max_new_tokens = seq->sampling_params.max_new_tokens;
            int64_t new_token_id = new_token_ids[i];
            seq->append_token(new_token_id);
            block_manager.may_append(*seq);

            bool hit_eos = !seq->sampling_params.ignore_eos && new_token_id == eos;
            bool max_len_reached = (seq->num_tokens - seq->num_prompt_tokens) >= max_new_tokens
                                || seq->num_tokens >= max_model_len;
            if (hit_eos || max_len_reached) {
                // Close this assistant turn span [next_turn_start, size()).
                seq->asst_spans.emplace_back(seq->next_turn_start, seq->size());
                seq->turn_count++;
                running.erase(std::remove(running.begin(), running.end(), seq), running.end());
                if (seq->sampling_params.agentic) {
                    // Agentic: pause without deallocating KV / freeing slot.
                    // Caller drives rollout_continue (resume) or rollout_finish (cleanup).
                    seq->status = SeqStatus::PAUSED;
                    paused_.push_back(seq);
                } else {
                    seq->status = SeqStatus::FINISHED;
                    block_manager.deallocate(*seq);
                    free_slot(seq->batch_slot);
                }
            }
        }
    }

    // ── Phase 1 of step(): schedule decode for all currently running sequences.
    // Preempts from the back of `running` when the KV pool is too full to store
    // one new token per sequence.  Populates preempted_slots_ as a side-effect;
    // caller must free model KV state for those slots before calling schedule_prefill().
    std::vector<Sequence*> schedule_decode() {
        preempted_slots_.clear();
        std::vector<Sequence*> scheduled;
        if (running.empty()) return scheduled;

        std::deque<Sequence*> to_process = running;
        running.clear();
        int blocks_reserved = 0;
        while (!to_process.empty()) {
            Sequence* seq = to_process.front();
            to_process.pop_front();
            int needs_block = (seq->size() % seq->block_size == 0) ? 1 : 0;
            while (block_manager.num_free_blocks() - blocks_reserved < needs_block) {
                if (!to_process.empty()) {
                    preempt(to_process.back());
                    to_process.pop_back();
                } else {
                    preempt(seq);
                    seq = nullptr;
                    break;
                }
            }
            if (seq && (int)scheduled.size() < max_num_seqs) {
                blocks_reserved += needs_block;
                running.push_back(seq);
                scheduled.push_back(seq);
            }
        }
        return scheduled;
    }

    // ── Phase 2 of step(): fill vacant batch slots from the waiting queue.
    // Called after schedule_decode() + postprocess() so that slots freed by
    // finished/preempted sequences are available.
    // Fills up to (max_num_seqs - running.size()) new sequences, subject to
    // token-budget and KV-block availability.
    std::vector<Sequence*> schedule_prefill() {
        if (waiting.empty()) return {};
        int available_slots = max_num_seqs - (int)running.size();
        if (available_slots <= 0) return {};

        std::vector<Sequence*> scheduled;
        int num_batched_tokens = 0;
        int max_seq_in_batch   = 0;  // track S_max for padded size check
        int prefill_blks_rsvd  = 0;  // post-prefill may_append blocks reserved
        while (!waiting.empty() && (int)scheduled.size() < available_slots) {
            Sequence* seq = waiting.front();
            int uncached = seq->size() - seq->num_cached_tokens;
            int needs_postprefill = (seq->size() % seq->block_size == 0) ? 1 : 0;
            // Check padded size: qwen3_prefill pads to B * S_max
            int new_smax = std::max(max_seq_in_batch, uncached);
            int padded_tokens = ((int)scheduled.size() + 1) * new_smax;

            // Resumed agentic seq: batch_slot + KV prefix already resident, so
            // we only need to allocate blocks for the new observation span — and
            // we MUST NOT alloc_slot() (it would clobber a still-live slot).
            const bool is_resumed = (seq->batch_slot >= 0 && seq->num_cached_tokens > 0);

            if (is_resumed) {
                int delta_blocks = block_manager.blocks_needed_for(*seq);
                if (padded_tokens > max_num_batched_tokens ||
                    block_manager.num_free_blocks()
                        - delta_blocks - prefill_blks_rsvd < needs_postprefill) {
                    break;
                }
                waiting.pop_front();
                block_manager.append_blocks_for(*seq);
                prefill_blks_rsvd += needs_postprefill;
                seq->status        = SeqStatus::RUNNING;
                // batch_slot retained
                running.push_back(seq);
                scheduled.push_back(seq);
                num_batched_tokens += uncached;
                max_seq_in_batch    = new_smax;
                continue;
            }

            if (padded_tokens > max_num_batched_tokens ||
                !block_manager.can_allocate(*seq) ||
                block_manager.num_free_blocks()
                    - (int)seq->num_blocks()
                    - prefill_blks_rsvd < needs_postprefill) {
                break;
            }
            waiting.pop_front();
            block_manager.allocate(*seq);
            prefill_blks_rsvd  += needs_postprefill;
            seq->status         = SeqStatus::RUNNING;
            seq->batch_slot     = alloc_slot();
            running.push_back(seq);
            scheduled.push_back(seq);
            num_batched_tokens += uncached;
            max_seq_in_batch    = new_smax;
        }
        return scheduled;
    }
};
