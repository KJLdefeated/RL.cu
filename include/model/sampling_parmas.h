#pragma once
#include <string>
#include <cstdint>
#include <vector>

struct SamplingParams {
    int64_t max_new_tokens = 256;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int64_t top_k = 0;
    bool do_sample = true;
    bool ignore_eos = false;  // if true, never stop on EOS token (only on max_new_tokens)
};

// sequence status
enum class SeqStatus : uint8_t {
    WAITING = 0,
    RUNNING = 1,
    FINISHED = 2
};

struct Sequence {
    int block_size = 16;  // must match KV_BLOCK_SIZE in kv_cache.cuh
    int64_t seq_id = 0;
    int batch_slot = -1;  // stable KV-cache slot; assigned at schedule time, freed at finish
    int num_tokens = 0;
    int64_t last_token_id = 0;
    int num_prompt_tokens = 0;
    int num_cached_tokens = 0;
    SeqStatus status = SeqStatus::WAITING;
    SamplingParams sampling_params;
    std::vector<int64_t> token_ids;
    std::vector<int> block_table;
    Sequence(int64_t seq_id, std::vector<int64_t> tok_ids, SamplingParams sampling_params) : 
    seq_id(seq_id), token_ids(tok_ids), sampling_params(sampling_params)  {
        num_tokens = tok_ids.size();
        if (num_tokens > 0) {
            last_token_id = tok_ids.back();
        }
        num_prompt_tokens = num_tokens;
    }
    int num_blocks() {
        return (token_ids.size() + block_size - 1) / block_size;
    }
    std::vector<int64_t> completion_token_ids() {
        return std::vector<int64_t>(token_ids.begin() + num_prompt_tokens, token_ids.end());
    }
    int size() {
        return token_ids.size();
    }
    std::vector<int64_t> block(int block_idx) const {
        int start = block_idx * block_size;
        int end = std::min(start + block_size, (int)token_ids.size());
        return std::
        vector<int64_t>(token_ids.begin() + start, token_ids.begin() + end);
    }
    void append_token(int64_t token_id) {
        token_ids.push_back(token_id);
        last_token_id = token_id;
        num_tokens++;
    }
};