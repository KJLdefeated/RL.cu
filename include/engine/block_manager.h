#pragma once
#include <vector>
#include <deque>
#include <map>
#include <set>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <stdexcept>
#define XXH_INLINE_ALL
#include "third_party/xxhash.h"
#include "model/sampling_parmas.h"

struct Block{
    int block_id = 0;
    int ref_cnt = 0;
    int64_t hash = -1;  // -1 = not yet hashed (block is partial)
    std::vector<int64_t> token_ids;
    Block() = default;
    void reset(){
        block_id = 0;
        ref_cnt = 0;
        hash = -1;
        token_ids.clear();
    }
    void update(int64_t new_hash, const std::vector<int64_t>& new_token_ids){
        hash = new_hash;
        token_ids = new_token_ids;
    }
};

class BlockManager {
private:
    std::vector<Block> blocks;
    int block_size;
    std::map<int64_t, int> hash_to_block_id;  // hash → block ID for prefix caching
    std::deque<int> free_block_ids;
    std::set<int> used_block_ids;

    // Allocate a block from the pool (caller must have already pop'd it from free list).
    Block& _allocate_block(int block_id) {
        if (block_id < 0 || block_id >= (int)blocks.size()) {
            throw std::out_of_range("Block ID out of range");
        }
        Block& block = blocks[block_id];
        assert(block.ref_cnt == 0);
        block.reset();
        block.block_id = block_id;
        block.ref_cnt  = 1;
        used_block_ids.insert(block_id);
        return block;
    }

    void _deallocate_block(int block_id) {
        if (block_id < 0 || block_id >= (int)blocks.size()) {
            throw std::out_of_range("Block ID out of range");
        }
        Block& block = blocks[block_id];
        assert(block.ref_cnt == 0);
        if (block.hash != -1) hash_to_block_id.erase(block.hash);
        block.reset();
        used_block_ids.erase(block_id);
        free_block_ids.push_back(block_id);
    }

    static int64_t compute_hash(const std::vector<int64_t>& token_ids, int64_t prefix = -1) {
        XXH64_state_t* state = XXH64_createState();
        XXH64_reset(state, 0);
        if (prefix != -1) {
            XXH64_update(state, &prefix, sizeof(int64_t));
        }
        XXH64_update(state, token_ids.data(), token_ids.size() * sizeof(int64_t));
        int64_t result = static_cast<int64_t>(XXH64_digest(state));
        XXH64_freeState(state);
        return result;
    }

public:
    BlockManager() : block_size(16) {}
    BlockManager(int num_blocks, int block_size) : blocks(num_blocks), block_size(block_size) {
        for (int i = 0; i < num_blocks; ++i) {
            blocks[i].block_id = i;
            free_block_ids.push_back(i);
        }
    }

    bool can_allocate(Sequence& seq) {
        return (int)free_block_ids.size() > seq.num_blocks();
    }

    void allocate(Sequence& seq) {
        for (int i = 0; i < seq.num_blocks(); ++i) {
            int block_id = free_block_ids.front();
            free_block_ids.pop_front();
            _allocate_block(block_id);
            seq.block_table.push_back(block_id);
        }
    }

    void deallocate(Sequence& seq) {
        for (int block_id : seq.block_table) {
            if (block_id < 0 || block_id >= (int)blocks.size()) {
                throw std::out_of_range("Block ID out of range");
            }
            Block& block = blocks[block_id];
            block.ref_cnt--;
            if (block.ref_cnt == 0) {
                _deallocate_block(block_id);
            }
        }
        seq.block_table.clear();
    }

    // Returns true if we have room to append one more token.
    // A new block is needed when the current size is exactly at a block boundary,
    // i.e., the next token will be the first token of a new block.
    bool can_append(Sequence& seq) {
        return free_block_ids.size() >= (size_t)(seq.size() % block_size == 0);
    }

    // Called after a new token has been appended to seq.token_ids.
    void may_append(Sequence& seq) {
        int sz = seq.size();
        if (sz % block_size == 1) {
            // First token of a new block: allocate a fresh physical block.
            assert(!free_block_ids.empty());
            int block_id = free_block_ids.front();
            free_block_ids.pop_front();
            _allocate_block(block_id);
            seq.block_table.push_back(block_id);
        } else if (sz % block_size == 0) {
            // Last token of current block: seal it with a hash for prefix caching.
            Block& last_block = blocks[seq.block_table.back()];
            assert(last_block.hash == -1);
            int last_blk_idx = (int)seq.block_table.size() - 1;
            std::vector<int64_t> blk_tokens = seq.block(last_blk_idx);
            int64_t prefix = (seq.block_table.size() > 1)
                ? blocks[seq.block_table[seq.block_table.size() - 2]].hash
                : -1;
            int64_t h = compute_hash(blk_tokens, prefix);
            last_block.update(h, blk_tokens);
            hash_to_block_id[h] = last_block.block_id;
        }
        // else: mid-block, nothing to do
    }

    int num_free_blocks() const { return (int)free_block_ids.size(); }
};
