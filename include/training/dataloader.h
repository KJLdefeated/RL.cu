#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <cassert>

// ============================================================
// DataLoader — reads pre-tokenized binary files produced by
// scripts/prepare_data.py and provides batches for training.
//
// Binary format:
//   Header (32 bytes):
//     magic[4]       : "RLDT"
//     version        : uint32
//     num_samples    : uint64
//     total_tokens   : uint64
//     flags          : uint32 (bit 0: has_prompt_lens)
//     pad            : uint32
//   Offsets          : uint64[num_samples + 1]
//   PromptLens       : uint32[num_samples]  (if flag set)
//   Tokens           : int32[total_tokens]
//
// Usage:
//   DataLoader loader("data/sft_train.bin");
//   loader.set_batch_size(4);
//   loader.set_max_seq_len(512);
//   loader.shuffle(42);
//
//   TrainBatch batch;
//   while (loader.next_batch(batch)) {
//       // batch.token_ids[B*S], batch.target_ids[B*S], ...
//       // feed to qwen3_forward / qwen3_backward
//   }
//   loader.reset();  // new epoch
// ============================================================

static constexpr uint32_t RLDT_MAGIC   = 0x54444C52;  // "RLDT" little-endian
static constexpr uint32_t RLDT_VERSION = 1;
static constexpr uint32_t FLAG_HAS_PROMPT_LENS = 1 << 0;
static constexpr uint32_t FLAG_HAS_ANSWERS     = 1 << 1;

struct TrainBatch {
    int B = 0;                          // actual batch size (may be < requested at end)
    int S = 0;                          // padded sequence length (max in batch)
    std::vector<int> token_ids;         // [B * S] input tokens (padded with 0)
    std::vector<int> target_ids;        // [B * S] target tokens (shifted left by 1)
    std::vector<int> loss_mask;         // [B * S] 1 where loss is computed, 0 for padding/prompt
    std::vector<int> seq_lens;          // [B] actual sequence lengths
    std::vector<int> prompt_lens;       // [B] prompt lengths (0 if no prompt info)
};

struct DataLoaderHeader {
    char     magic[4];
    uint32_t version;
    uint64_t num_samples;
    uint64_t total_tokens;
    uint32_t flags;
    uint32_t pad;
};

class DataLoader {
public:
    DataLoader() = default;

    DataLoader(const std::string& path, int batch_size = 1, int max_seq_len = 2048)
        : batch_size_(batch_size), max_seq_len_(max_seq_len)
    {
        if (!load(path)) {
            fprintf(stderr, "[DataLoader] Failed to load %s\n", path.c_str());
        }
    }

    bool load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "[DataLoader] Cannot open %s\n", path.c_str());
            return false;
        }

        // Read header
        DataLoaderHeader hdr;
        if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
            fprintf(stderr, "[DataLoader] Failed to read header\n");
            fclose(f);
            return false;
        }
        if (memcmp(hdr.magic, "RLDT", 4) != 0) {
            fprintf(stderr, "[DataLoader] Bad magic: expected RLDT\n");
            fclose(f);
            return false;
        }
        if (hdr.version != RLDT_VERSION) {
            fprintf(stderr, "[DataLoader] Bad version: %u (expected %u)\n",
                    hdr.version, RLDT_VERSION);
            fclose(f);
            return false;
        }

        num_samples_  = (size_t)hdr.num_samples;
        total_tokens_ = (size_t)hdr.total_tokens;
        has_prompt_lens_ = (hdr.flags & FLAG_HAS_PROMPT_LENS) != 0;

        // Read offsets: uint64[num_samples + 1]
        offsets_.resize(num_samples_ + 1);
        if (fread(offsets_.data(), sizeof(uint64_t), num_samples_ + 1, f)
            != num_samples_ + 1) {
            fprintf(stderr, "[DataLoader] Failed to read offsets\n");
            fclose(f);
            return false;
        }

        // Read prompt lengths if present
        if (has_prompt_lens_) {
            prompt_lens_.resize(num_samples_);
            if (fread(prompt_lens_.data(), sizeof(uint32_t), num_samples_, f)
                != num_samples_) {
                fprintf(stderr, "[DataLoader] Failed to read prompt_lens\n");
                fclose(f);
                return false;
            }
        }

        // Read all tokens: int32[total_tokens]
        tokens_.resize(total_tokens_);
        if (fread(tokens_.data(), sizeof(int32_t), total_tokens_, f)
            != total_tokens_) {
            fprintf(stderr, "[DataLoader] Failed to read tokens\n");
            fclose(f);
            return false;
        }

        fclose(f);

        // Initialize index for sequential access
        indices_.resize(num_samples_);
        for (size_t i = 0; i < num_samples_; i++) indices_[i] = (int)i;
        cursor_ = 0;

        printf("[DataLoader] Loaded %s: %zu samples, %zu tokens",
               path.c_str(), num_samples_, total_tokens_);
        if (has_prompt_lens_) printf(", has prompt_lens");
        printf("\n");

        return true;
    }

    // Configuration
    void set_batch_size(int bs)    { batch_size_ = bs; }
    void set_max_seq_len(int msl)  { max_seq_len_ = msl; }

    // Shuffle sample order (call at start of each epoch)
    void shuffle(uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::shuffle(indices_.begin(), indices_.end(), rng);
        cursor_ = 0;
    }

    // Reset cursor to start (for next epoch without reshuffling)
    void reset() { cursor_ = 0; }

    // Number of samples
    size_t num_samples() const { return num_samples_; }

    // Number of batches per epoch
    size_t num_batches() const {
        return (num_samples_ + batch_size_ - 1) / batch_size_;
    }

    // Get sample i's tokens (raw, unpadded)
    const int32_t* sample_tokens(size_t i, size_t& len) const {
        len = (size_t)(offsets_[i + 1] - offsets_[i]);
        return tokens_.data() + offsets_[i];
    }

    // Get sample i's prompt length (0 if not available)
    int sample_prompt_len(size_t i) const {
        if (!has_prompt_lens_ || i >= prompt_lens_.size()) return 0;
        return (int)prompt_lens_[i];
    }

    // Fill next batch. Returns false when epoch is done.
    bool next_batch(TrainBatch& batch) {
        if (cursor_ >= num_samples_) return false;

        int B = std::min(batch_size_, (int)(num_samples_ - cursor_));

        // Find max seq len in this batch
        int S = 0;
        for (int b = 0; b < B; b++) {
            int idx = indices_[cursor_ + b];
            int len = (int)(offsets_[idx + 1] - offsets_[idx]);
            len = std::min(len, max_seq_len_);
            S = std::max(S, len);
        }

        batch.B = B;
        batch.S = S;
        batch.token_ids.assign(B * S, 0);
        batch.target_ids.assign(B * S, 0);
        batch.loss_mask.assign(B * S, 0);
        batch.seq_lens.resize(B);
        batch.prompt_lens.resize(B);

        for (int b = 0; b < B; b++) {
            int idx = indices_[cursor_ + b];
            int raw_len = (int)(offsets_[idx + 1] - offsets_[idx]);
            int len = std::min(raw_len, max_seq_len_);
            int pl = has_prompt_lens_ ? std::min((int)prompt_lens_[idx], len) : 0;

            batch.seq_lens[b] = len;
            batch.prompt_lens[b] = pl;

            const int32_t* src = tokens_.data() + offsets_[idx];

            // token_ids: [b*S .. b*S + len-1] = src[0..len-1], rest padded 0
            for (int t = 0; t < len; t++)
                batch.token_ids[b * S + t] = src[t];

            // target_ids: shifted left by 1
            // target[t] = token[t+1] for t < len-1, target[len-1] = 0
            for (int t = 0; t < len - 1; t++)
                batch.target_ids[b * S + t] = src[t + 1];

            // loss_mask: 1 for response tokens (after prompt), 0 for prompt/padding
            // For SFT: loss on positions [pl-1, len-2] (predicting tokens [pl, len-1])
            // For GRPO (no prompt_lens): loss on all tokens [0, len-2]
            int loss_start = (pl > 0) ? pl - 1 : 0;
            for (int t = loss_start; t < len - 1; t++)
                batch.loss_mask[b * S + t] = 1;
        }

        cursor_ += B;
        return true;
    }

private:
    // Data from file
    size_t                 num_samples_  = 0;
    size_t                 total_tokens_ = 0;
    bool                   has_prompt_lens_ = false;
    std::vector<uint64_t>  offsets_;       // [num_samples + 1]
    std::vector<uint32_t>  prompt_lens_;   // [num_samples] (if present)
    std::vector<int32_t>   tokens_;        // [total_tokens]

    // Iteration state
    std::vector<int>       indices_;       // shuffled sample indices
    size_t                 cursor_ = 0;
    int                    batch_size_   = 1;
    int                    max_seq_len_  = 2048;
};
