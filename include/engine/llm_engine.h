#pragma once
#include <vector>
#include <map>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include "model/config.h"
#include "model/sampling_parmas.h"
#include "model/tokenizer.h"
#include "engine/scheduler.h"
#include "engine/model_runner.cuh"

class LLMEngine {
public:
    Config       config;
    ModelRunner* model_runner = nullptr;
    Scheduler*   scheduler    = nullptr;
    Tokenizer*   tokenizer    = nullptr;
    int64_t      seq_id_counter = 0;

    LLMEngine(const Config& cfg) : config(cfg) {
        // Load tokenizer first so config.eos is correct before Scheduler reads it.
        tokenizer = new Tokenizer();
        tokenizer->load(config.tokenizer_path);
        config.eos = tokenizer->eos_id;

        model_runner = new ModelRunner(config);
        // Propagate compute_kv_budget's adjusted max_num_seqs so the Scheduler
        // and ModelRunner agree on the batch slot limit. Without this, the
        // Scheduler assigns batch_slots up to the original max_num_seqs while
        // ModelRunner only allocated KV/sampler buffers for the budget-limited value.
        config.max_num_seqs           = model_runner->config.max_num_seqs;
        config.num_kv_blocks          = model_runner->config.num_kv_blocks;
        config.max_num_batched_tokens = model_runner->config.max_num_batched_tokens;
        scheduler    = new Scheduler(config);
    }
    ~LLMEngine() {
        delete model_runner;
        delete scheduler;
        delete tokenizer;
    }

    void add_request(const std::string& prompt, SamplingParams sampling_params) {
        std::vector<int64_t> token_ids = tokenizer->encode(prompt);
        scheduler->add(new Sequence(seq_id_counter++, token_ids, sampling_params));
    }

    void add_request(const std::vector<int64_t>& token_ids, SamplingParams sampling_params) {
        scheduler->add(new Sequence(seq_id_counter++, token_ids, sampling_params));
    }

    // Returns ({(seq_id, completion_token_ids), ...}, total_new_tokens)
    //
    // Two-phase continuous batching:
    //   Phase 1 — decode all currently running sequences.
    //             Sequences that finish free their batch slots.
    //   Phase 2 — prefill waiting sequences into the newly vacated slots.
    //             Those sequences will be decoded starting from the next step().
    //
    // This keeps the decode batch at max capacity: as soon as a slot opens up
    // a new sequence fills it in the same call, so the next decode step is full.
    std::pair<std::vector<std::pair<int64_t, std::vector<int64_t>>>, int> step() {
        std::vector<std::pair<int64_t, std::vector<int64_t>>> output;
        int num_tokens = 0;

        // Helper: collect finished sequences, free their KV/slot, accumulate output.
        auto collect_finished = [&](std::vector<Sequence*>& batch) {
            for (Sequence* seq : batch) {
                if (seq->status == SeqStatus::FINISHED) {
                    model_runner->free_seq_slot(seq->batch_slot);
                    std::vector<int64_t> completion = seq->completion_token_ids();
                    num_tokens += (int)completion.size();
                    output.push_back({seq->seq_id, std::move(completion)});
                    delete seq;
                }
            }
        };

        // ── Phase 1: decode ──────────────────────────────────────────────────────
        auto decode_batch = scheduler->schedule_decode();

        // Free KV state for any sequences preempted while building the decode batch.
        for (int slot : scheduler->preempted_slots())
            model_runner->free_seq_slot(slot);

        if (!decode_batch.empty()) {
            auto toks = model_runner->run(decode_batch, /*is_prefill=*/false);
            scheduler->postprocess(decode_batch, toks, /*is_prefill=*/false);
            collect_finished(decode_batch);
        }

        // ── Phase 2: prefill into vacated slots ──────────────────────────────────
        auto prefill_batch = scheduler->schedule_prefill();
        if (!prefill_batch.empty()) {
            auto toks = model_runner->run(prefill_batch, /*is_prefill=*/true);
            scheduler->postprocess(prefill_batch, toks, /*is_prefill=*/true);
            collect_finished(prefill_batch);
        }

        return {output, num_tokens};
    }

    bool is_finished() const {
        return scheduler->is_finished();
    }

    // Variant: accepts pre-tokenized inputs with per-sequence SamplingParams.
    // Returns an empty vector of strings (callers should use seq_id→token_ids map).
    // total_output_tokens is set to the sum of generated token counts.
    int generate(const std::vector<std::vector<int64_t>>& token_id_batches,
                 const std::vector<SamplingParams>& sp_vec) {
        for (int i = 0; i < (int)token_id_batches.size(); i++)
            add_request(token_id_batches[i], sp_vec[i]);

        const int   total_seqs  = (int)token_id_batches.size();
        int         done_seqs   = 0;
        int         total_toks  = 0;
        int         step_count  = 0;
        auto        t0          = std::chrono::steady_clock::now();
        const int   bar_width   = 30;

        auto print_bar = [&]() {
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            double tok_s = elapsed > 0.0 ? total_toks / elapsed : 0.0;
            int filled = (total_seqs > 0)
                ? (done_seqs * bar_width / total_seqs) : 0;
            fprintf(stderr, "\r  [");
            for (int i = 0; i < bar_width; ++i)
                fputc(i < filled ? '#' : '.', stderr);
            fprintf(stderr, "] %d/%d seqs  %d tok  %.0f tok/s  %.1fs ",
                    done_seqs, total_seqs, total_toks, tok_s, elapsed);
            fflush(stderr);
        };

        print_bar();
        while (!is_finished()) {
            auto [completions, ntok] = step();
            total_toks += ntok;
            ++step_count;
            if (step_count % 4 == 0 || done_seqs == total_seqs)
                print_bar();
            for (auto& [sid, tids] : completions) { (void)sid; (void)tids; }
        }
        fprintf(stderr, "\n");  // newline after bar
        return total_toks;
    }

    std::vector<std::string> generate(const std::vector<std::string>& prompts,
                                      SamplingParams sampling_params) {
        for (const auto& prompt : prompts) {
            add_request(prompt, sampling_params);
        }

        const int   total_seqs  = (int)prompts.size();
        int         done_seqs   = 0;
        int         total_toks  = 0;
        int         step_count  = 0;
        auto        t0          = std::chrono::steady_clock::now();
        const int   bar_width   = 30;

        auto print_bar = [&]() {
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            double tok_s = elapsed > 0.0 ? total_toks / elapsed : 0.0;
            int filled = (total_seqs > 0)
                ? (done_seqs * bar_width / total_seqs) : 0;
            fprintf(stderr, "\r  [");
            for (int i = 0; i < bar_width; ++i)
                fputc(i < filled ? '#' : '.', stderr);
            fprintf(stderr, "] %d/%d seqs  %d tok  %.0f tok/s  %.1fs ",
                    done_seqs, total_seqs, total_toks, tok_s, elapsed);
            fflush(stderr);
        };

        std::map<int64_t, std::vector<int64_t>> generated;
        print_bar();
        while (!is_finished()) {
            auto [completions, ntok] = step();
            total_toks += ntok;
            ++step_count;
            for (auto& [seq_id, token_ids] : completions) {
                ++done_seqs;
                auto& vec = generated[seq_id];
                vec.insert(vec.end(), token_ids.begin(), token_ids.end());
            }
            if (step_count % 4 == 0 || done_seqs == total_seqs)
                print_bar();
        }
        fprintf(stderr, "\n");  // newline after bar

        std::vector<std::string> results;
        results.reserve(generated.size());
        for (auto& [seq_id, token_ids] : generated) {
            results.push_back(tokenizer->decode(token_ids));
        }
        return results;
    }
};
