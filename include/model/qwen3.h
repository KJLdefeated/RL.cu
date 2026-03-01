#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <string>

// Opaque handle.  Full definition is in src/model/qwen3.cu.
struct Qwen3Model;

// ---------------------------------------------------------------------------
// Load / free
// ---------------------------------------------------------------------------
// Allocates GPU memory for weights, KV cache, RoPE tables, and scratch buffers.
//   max_batch : maximum number of concurrent sequences
//   max_seq   : maximum sequence length (tokens per sequence)
Qwen3Model* qwen3_load(const std::string& model_dir, int max_batch, int max_seq);

// Free all GPU/CPU memory and delete the model.
void qwen3_free(Qwen3Model* model);

// Reset KV cache state.  Call between independent generation requests.
void qwen3_reset(Qwen3Model* model);

// ---------------------------------------------------------------------------
// Forward passes
// ---------------------------------------------------------------------------

// Prefill: encode B sequences of S tokens each (new sequences, pos 0..S-1).
//   tokens_gpu : [B * S]  on device (int32, sequences packed row-major)
// Returns d_logits [B, vocab_size] — last-token logits for each sequence.
// Side effect: K/V for all T=B*S tokens are written to the KV cache.
half* qwen3_prefill(Qwen3Model* model,
                    const int* tokens_gpu, int B, int S,
                    cudaStream_t stream = 0);

// Decode: one new token per sequence (must follow a prefill or prior decode).
//   token_gpu : [B]  on device (int32)
// Returns d_logits [B, vocab_size].
// Side effect: K/V for the B new tokens are written to the KV cache.
half* qwen3_decode(Qwen3Model* model,
                   const int* token_gpu, int B,
                   cudaStream_t stream = 0);
