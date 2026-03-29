// test_dataloader.cpp
//
// Tests the DataLoader class by:
//   1. Writing a synthetic binary file with known data
//   2. Loading it with DataLoader
//   3. Verifying batches, shuffling, loss masks, etc.
//
// No GPU or model weights required.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cassert>
#include <filesystem>

#include "training/dataloader.h"

static int g_fails = 0;
#define PASS(name)      printf("[PASS] %s\n", (name))
#define FAIL(name, ...) do { \
    printf("[FAIL] %s: ", (name)); printf(__VA_ARGS__); printf("\n"); g_fails++; \
} while (0)

static const char* TEST_BIN = "/tmp/test_dataloader.bin";
static const char* TEST_BIN_NO_PL = "/tmp/test_dataloader_nopl.bin";

// Helper: write a test binary file
static void write_test_file(
    const char* path,
    const std::vector<std::vector<int32_t>>& samples,
    const std::vector<uint32_t>* prompt_lens = nullptr
) {
    FILE* f = fopen(path, "wb");
    assert(f);

    size_t num_samples = samples.size();
    size_t total_tokens = 0;
    for (auto& s : samples) total_tokens += s.size();

    uint32_t flags = 0;
    if (prompt_lens) flags |= FLAG_HAS_PROMPT_LENS;

    // Header
    fwrite("RLDT", 1, 4, f);
    uint32_t ver = 1;
    fwrite(&ver, 4, 1, f);
    uint64_t ns = num_samples, tt = total_tokens;
    fwrite(&ns, 8, 1, f);
    fwrite(&tt, 8, 1, f);
    fwrite(&flags, 4, 1, f);
    uint32_t pad = 0;
    fwrite(&pad, 4, 1, f);

    // Offsets
    uint64_t offset = 0;
    for (auto& s : samples) {
        fwrite(&offset, 8, 1, f);
        offset += s.size();
    }
    fwrite(&offset, 8, 1, f);

    // Prompt lens
    if (prompt_lens) {
        fwrite(prompt_lens->data(), 4, num_samples, f);
    }

    // Tokens
    for (auto& s : samples) {
        fwrite(s.data(), 4, s.size(), f);
    }

    fclose(f);
}

// =============================================================================
// Test 1: Basic load and iterate
// =============================================================================
static void test_basic_load() {
    const char* name = "Basic load and iterate";

    // 5 samples with known content
    std::vector<std::vector<int32_t>> samples = {
        {10, 20, 30, 40, 50},         // len=5
        {100, 200, 300},              // len=3
        {1, 2, 3, 4, 5, 6, 7},       // len=7
        {42, 43},                     // len=2
        {500, 501, 502, 503},         // len=4
    };
    std::vector<uint32_t> prompt_lens = {2, 1, 3, 1, 2};

    write_test_file(TEST_BIN, samples, &prompt_lens);

    DataLoader loader;
    if (!loader.load(TEST_BIN)) {
        FAIL(name, "failed to load");
        return;
    }

    if (loader.num_samples() != 5) {
        FAIL(name, "num_samples=%zu, expected 5", loader.num_samples());
        return;
    }

    // Iterate with batch_size=2
    loader.set_batch_size(2);
    loader.set_max_seq_len(1024);

    TrainBatch batch;
    int total_batches = 0;
    int total_samples = 0;

    while (loader.next_batch(batch)) {
        total_batches++;
        total_samples += batch.B;

        // Verify padding: positions >= seq_len should be 0
        for (int b = 0; b < batch.B; b++) {
            for (int t = batch.seq_lens[b]; t < batch.S; t++) {
                if (batch.token_ids[b * batch.S + t] != 0) {
                    FAIL(name, "non-zero padding at batch %d, seq %d, pos %d",
                         total_batches, b, t);
                    return;
                }
            }
        }
    }

    if (total_batches != 3) {
        FAIL(name, "total_batches=%d, expected 3", total_batches);
        return;
    }
    if (total_samples != 5) {
        FAIL(name, "total_samples=%d, expected 5", total_samples);
        return;
    }

    PASS(name);
}

// =============================================================================
// Test 2: Target IDs are shifted left by 1
// =============================================================================
static void test_target_shift() {
    const char* name = "Target IDs shifted left by 1";

    std::vector<std::vector<int32_t>> samples = {
        {10, 20, 30, 40, 50},
    };

    write_test_file(TEST_BIN, samples);

    DataLoader loader;
    loader.load(TEST_BIN);
    loader.set_batch_size(1);

    TrainBatch batch;
    loader.next_batch(batch);

    // target[t] = token[t+1] for t < len-1
    bool ok = true;
    for (int t = 0; t < 4; t++) {
        if (batch.target_ids[t] != samples[0][t + 1]) {
            FAIL(name, "target[%d]=%d, expected %d", t, batch.target_ids[t], samples[0][t + 1]);
            ok = false;
            break;
        }
    }
    // target[len-1] = 0 (end of sequence)
    if (ok && batch.target_ids[4] != 0) {
        FAIL(name, "target[4]=%d, expected 0", batch.target_ids[4]);
        ok = false;
    }

    if (ok) PASS(name);
}

// =============================================================================
// Test 3: Loss mask with prompt_lens (SFT mode)
// =============================================================================
static void test_loss_mask_sft() {
    const char* name = "Loss mask with prompt_lens (SFT)";

    // Sample: 5 tokens, prompt_len=2
    // Positions: 0  1  2  3  4
    // Prompt:    P  P  R  R  R
    // Loss:      0  1  1  1  0   (loss on [pl-1=1, len-2=3])
    std::vector<std::vector<int32_t>> samples = {
        {10, 20, 30, 40, 50},
    };
    std::vector<uint32_t> prompt_lens = {2};

    write_test_file(TEST_BIN, samples, &prompt_lens);

    DataLoader loader;
    loader.load(TEST_BIN);
    loader.set_batch_size(1);

    TrainBatch batch;
    loader.next_batch(batch);

    std::vector<int> expected = {0, 1, 1, 1, 0};
    bool ok = true;
    for (int t = 0; t < 5; t++) {
        if (batch.loss_mask[t] != expected[t]) {
            FAIL(name, "loss_mask[%d]=%d, expected %d", t, batch.loss_mask[t], expected[t]);
            ok = false;
            break;
        }
    }

    if (ok) PASS(name);
}

// =============================================================================
// Test 4: Loss mask without prompt_lens (GRPO mode)
// =============================================================================
static void test_loss_mask_grpo() {
    const char* name = "Loss mask without prompt_lens (GRPO)";

    // No prompt_lens → loss on all positions [0, len-2]
    std::vector<std::vector<int32_t>> samples = {
        {10, 20, 30, 40, 50},
    };

    write_test_file(TEST_BIN_NO_PL, samples);

    DataLoader loader;
    loader.load(TEST_BIN_NO_PL);
    loader.set_batch_size(1);

    TrainBatch batch;
    loader.next_batch(batch);

    std::vector<int> expected = {1, 1, 1, 1, 0};
    bool ok = true;
    for (int t = 0; t < 5; t++) {
        if (batch.loss_mask[t] != expected[t]) {
            FAIL(name, "loss_mask[%d]=%d, expected %d", t, batch.loss_mask[t], expected[t]);
            ok = false;
            break;
        }
    }

    if (ok) PASS(name);
}

// =============================================================================
// Test 5: Shuffle changes order, covers all samples
// =============================================================================
static void test_shuffle() {
    const char* name = "Shuffle covers all samples";

    std::vector<std::vector<int32_t>> samples;
    for (int i = 0; i < 20; i++) {
        samples.push_back({i * 10, i * 10 + 1, i * 10 + 2});
    }

    write_test_file(TEST_BIN, samples);

    DataLoader loader;
    loader.load(TEST_BIN);
    loader.set_batch_size(4);

    // Collect first tokens from unshuffled pass
    std::vector<int> order1;
    TrainBatch batch;
    while (loader.next_batch(batch)) {
        for (int b = 0; b < batch.B; b++)
            order1.push_back(batch.token_ids[b * batch.S]);
    }

    // Shuffle and collect again
    loader.shuffle(12345);
    std::vector<int> order2;
    while (loader.next_batch(batch)) {
        for (int b = 0; b < batch.B; b++)
            order2.push_back(batch.token_ids[b * batch.S]);
    }

    if (order1.size() != 20 || order2.size() != 20) {
        FAIL(name, "sizes: %zu, %zu (expected 20)", order1.size(), order2.size());
        return;
    }

    // Check all 20 samples are present in both orders
    auto sorted1 = order1; std::sort(sorted1.begin(), sorted1.end());
    auto sorted2 = order2; std::sort(sorted2.begin(), sorted2.end());
    if (sorted1 != sorted2) {
        FAIL(name, "shuffled order doesn't contain all samples");
        return;
    }

    // Check order actually changed
    if (order1 == order2) {
        FAIL(name, "shuffle didn't change order (unlikely with seed 12345)");
        return;
    }

    PASS(name);
}

// =============================================================================
// Test 6: max_seq_len truncation
// =============================================================================
static void test_max_seq_len() {
    const char* name = "Max seq len truncation";

    std::vector<std::vector<int32_t>> samples = {
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},  // len=10
    };
    std::vector<uint32_t> prompt_lens = {3};

    write_test_file(TEST_BIN, samples, &prompt_lens);

    DataLoader loader;
    loader.load(TEST_BIN);
    loader.set_batch_size(1);
    loader.set_max_seq_len(6);  // truncate to 6

    TrainBatch batch;
    loader.next_batch(batch);

    if (batch.S != 6) {
        FAIL(name, "S=%d, expected 6", batch.S);
        return;
    }
    if (batch.seq_lens[0] != 6) {
        FAIL(name, "seq_lens[0]=%d, expected 6", batch.seq_lens[0]);
        return;
    }

    // Verify tokens are first 6
    bool ok = true;
    for (int t = 0; t < 6; t++) {
        if (batch.token_ids[t] != t + 1) {
            FAIL(name, "token[%d]=%d, expected %d", t, batch.token_ids[t], t + 1);
            ok = false;
            break;
        }
    }

    // Prompt len should be clamped to min(3, 6) = 3
    if (ok && batch.prompt_lens[0] != 3) {
        FAIL(name, "prompt_lens[0]=%d, expected 3", batch.prompt_lens[0]);
        ok = false;
    }

    if (ok) PASS(name);
}

// =============================================================================
// Test 7: Reset resets cursor
// =============================================================================
static void test_reset() {
    const char* name = "Reset resets cursor";

    std::vector<std::vector<int32_t>> samples = {
        {10, 20, 30},
        {40, 50, 60},
    };

    write_test_file(TEST_BIN, samples);

    DataLoader loader;
    loader.load(TEST_BIN);
    loader.set_batch_size(1);

    TrainBatch batch;

    // First pass
    loader.next_batch(batch);
    int first_token_a = batch.token_ids[0];
    loader.next_batch(batch);
    bool done = !loader.next_batch(batch);

    if (!done) {
        FAIL(name, "expected end of epoch after 2 samples");
        return;
    }

    // Reset and verify same first token
    loader.reset();
    loader.next_batch(batch);
    int first_token_b = batch.token_ids[0];

    if (first_token_a != first_token_b) {
        FAIL(name, "first token after reset: %d vs %d", first_token_a, first_token_b);
        return;
    }

    PASS(name);
}

// =============================================================================
// Test 8: Batch padding with different-length sequences
// =============================================================================
static void test_batch_padding() {
    const char* name = "Batch padding with mixed lengths";

    std::vector<std::vector<int32_t>> samples = {
        {10, 20, 30},                  // len=3
        {40, 50, 60, 70, 80, 90},     // len=6
    };

    write_test_file(TEST_BIN, samples);

    DataLoader loader;
    loader.load(TEST_BIN);
    loader.set_batch_size(2);

    TrainBatch batch;
    loader.next_batch(batch);

    if (batch.S != 6) {
        FAIL(name, "S=%d, expected 6 (max in batch)", batch.S);
        return;
    }

    // First sequence: tokens at [0..2], padding at [3..5]
    bool ok = true;
    for (int t = 3; t < 6; t++) {
        if (batch.token_ids[0 * 6 + t] != 0) {
            FAIL(name, "seq0 not padded at pos %d: %d", t, batch.token_ids[t]);
            ok = false;
            break;
        }
    }

    // Loss mask should be 0 for padding
    if (ok) {
        for (int t = 3; t < 6; t++) {
            if (batch.loss_mask[0 * 6 + t] != 0) {
                FAIL(name, "loss_mask not 0 for padding at pos %d", t);
                ok = false;
                break;
            }
        }
    }

    if (ok) PASS(name);
}

// =============================================================================
// Test 9: sample_tokens raw access
// =============================================================================
static void test_sample_access() {
    const char* name = "Raw sample access";

    std::vector<std::vector<int32_t>> samples = {
        {10, 20, 30},
        {40, 50},
        {60, 70, 80, 90},
    };

    write_test_file(TEST_BIN, samples);

    DataLoader loader;
    loader.load(TEST_BIN);

    bool ok = true;
    for (size_t i = 0; i < samples.size(); i++) {
        size_t len;
        const int32_t* toks = loader.sample_tokens(i, len);
        if (len != samples[i].size()) {
            FAIL(name, "sample %zu: len=%zu, expected %zu", i, len, samples[i].size());
            ok = false;
            break;
        }
        for (size_t t = 0; t < len; t++) {
            if (toks[t] != samples[i][t]) {
                FAIL(name, "sample %zu token %zu: %d != %d", i, t, toks[t], samples[i][t]);
                ok = false;
                break;
            }
        }
        if (!ok) break;
    }

    if (ok) PASS(name);
}

// =============================================================================
// main
// =============================================================================
int main() {
    printf("=== DataLoader tests ===\n\n");

    test_basic_load();
    test_target_shift();
    test_loss_mask_sft();
    test_loss_mask_grpo();
    test_shuffle();
    test_max_seq_len();
    test_reset();
    test_batch_padding();
    test_sample_access();

    // Cleanup
    std::remove(TEST_BIN);
    std::remove(TEST_BIN_NO_PL);

    printf("\n%s\n", g_fails ? "SOME TESTS FAILED" : "All tests PASSED.");
    return g_fails ? 1 : 0;
}
