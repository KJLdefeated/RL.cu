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
    int64_t eos;
    BlockManager block_manager;
    std::deque<Sequence*> waiting;
    std::deque<Sequence*> running;
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
          eos(cfg.eos),
          block_manager(cfg.num_kv_blocks, cfg.kv_block_size),
          slot_used(cfg.max_num_seqs, false) {}

    bool is_finished() const {
        return waiting.empty() && running.empty();
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

            bool max_len_reached = (seq->num_tokens - seq->num_prompt_tokens) >= max_new_tokens;
            if ((!seq->sampling_params.ignore_eos && new_token_id == eos) || max_len_reached) {
                seq->status = SeqStatus::FINISHED;
                block_manager.deallocate(*seq);
                free_slot(seq->batch_slot);
                running.erase(std::remove(running.begin(), running.end(), seq), running.end());
            }
        }
    }

    // Returns {batch, is_prefill}.
    std::pair<std::vector<Sequence*>, bool> schedule() {
        std::vector<Sequence*> scheduled;
        preempted_slots_.clear();

        // Prefill
        // Track post-prefill may_append reservations: a sequence needs 1 extra block
        // if its size after the first generated token will cross a block boundary,
        // i.e., if current size % block_size == 0 (so size+1 % block_size == 1).
        int num_seqs = 0;
        int num_batched_tokens = 0;
        int prefill_blocks_reserved = 0;  // post-prefill may_append blocks needed
        while (!waiting.empty() && num_seqs < max_num_seqs) {
            Sequence* seq = waiting.front();
            int uncached = seq->size() - seq->num_cached_tokens;
            int needs_postprefill_block = (seq->size() % seq->block_size == 0) ? 1 : 0;
            if (num_batched_tokens + uncached > max_num_batched_tokens ||
                !block_manager.can_allocate(*seq) ||
                block_manager.num_free_blocks() - (int)seq->num_blocks() - prefill_blocks_reserved < needs_postprefill_block) {
                break;
            }
            waiting.pop_front();
            block_manager.allocate(*seq);
            prefill_blocks_reserved += needs_postprefill_block;
            seq->status = SeqStatus::RUNNING;
            seq->batch_slot = alloc_slot();
            running.push_back(seq);
            scheduled.push_back(seq);
            num_batched_tokens += uncached;
            num_seqs++;
        }
        if (!scheduled.empty()) {
            return {scheduled, true};
        }

        // Decode pass
        // Collect all running sequences; preempt from the back if we're OOM.
        // Track "blocks_reserved": blocks already committed to scheduled seqs
        // but not yet physically allocated (allocation happens in postprocess).
        std::deque<Sequence*> to_process = running;
        running.clear();
        int blocks_reserved = 0;
        while (!to_process.empty()) {
            Sequence* seq = to_process.front();
            to_process.pop_front();
            // Does the next token start a new block? (current size is at boundary)
            int needs_block = (seq->size() % seq->block_size == 0) ? 1 : 0;
            // Ensure there's space after accounting for already-reserved blocks.
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
        return {scheduled, false};
    }
};
