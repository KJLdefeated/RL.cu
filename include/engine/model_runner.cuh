#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include "cuda_utils.h"
#include "model/config.h"
#include "model/sampling_parmas.h"
#include "model/qwen3.h"
#include "kernels/sampler.cuh"
#include "model/weights.h"

class ModelRunner {
    public:
    Qwen3Model*        model;
    Config             config;
    Qwen3Config        model_config;
    int                block_size;
    bool               enforce_eager;
    float*             d_temp_probs = nullptr;  // [max_num_seqs, vocab_size] FP32 sampler workspace
    int64_t*           d_output_ids = nullptr;  // [max_num_seqs] sampled token IDs
    unsigned long long sample_seed  = 42ULL;    // PRNG seed, incremented each step

    ModelRunner(const Config& cfg) : config(cfg) {
        model_config = load_config(cfg.model);
        model_config.compute_derived();
        model = new Qwen3Model{};
        model->config = model_config;
        model->weights = load_weights(cfg.model, model_config);
        // Compute KV cache budget and adjust max_num_seqs
        compute_kv_budget(config, model_config, model->weights.total_bytes);
        // Initialize model with max batch/seq for buffer allocation.
        qwen3_init(model, config.max_num_seqs, cfg.max_model_len, cfg.max_num_batched_tokens, config.num_kv_blocks);
        block_size    = cfg.kv_block_size;
        enforce_eager = cfg.enforce_eager;

        // Sampler buffers — allocated once for the maximum batch size.
        int V = model_config.vocab_size;
        CUDA_CHECK(cudaMalloc(&d_temp_probs, (long)config.max_num_seqs * V * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_ids, config.max_num_seqs * sizeof(int64_t)));
        printf("[Qwen3] Loaded: %d layers  H_q=%d H_kv=%d  hidden=%d  vocab=%d\n",
           model_config.num_hidden_layers, model_config.num_attention_heads,
           model_config.num_key_value_heads, model_config.hidden_size, model_config.vocab_size);
        // Model warmup
        warmup();
        // Capture CUDA graphs for decoding
        if (!enforce_eager)
            capture_cudagraph();
    }

    ~ModelRunner() {
        // Free CUDA graph executables and their fixed input buffers.
        // These are owned by ModelRunner, not by qwen3_free.
        CUDAGraphState& gs = model->graph_state;
        if (gs.captured) {
            for (auto& [b, exec] : gs.graphs)
                cudaGraphExecDestroy(exec);
            cudaFree(gs.g_token_ids);
            cudaFree(gs.g_pos_ids);
            cudaFree(gs.g_slot_map);
            cudaFree(gs.g_block_tables);
            cudaFree(gs.g_seq_lens);
            cudaFree(gs.g_cublas_ws);
        }
        cudaFree(d_temp_probs);
        cudaFree(d_output_ids);
        // Free weights, KV cache, scratch buffers, cuBLAS, host mirrors.
        // qwen3_free also calls delete model.
        qwen3_free(model);
    }

    void warmup() {
        cudaDeviceSynchronize();
        // Use a short sequence to avoid register-spilling slowness in FA2 prefill
        // (HEAD_DIM=128 → ~256 float registers/thread → kernel spills for long S).
        const int warmup_seq_len = 64;
        const int num_seqs       = std::min(4, config.max_num_seqs);
        SamplingParams sp;
        std::vector<Sequence*> dummy_batch;
        for (int i = 0; i < num_seqs; i++) {
            std::vector<int64_t> tokens(warmup_seq_len, 0);
            auto* seq = new Sequence(i, tokens, sp);
            seq->batch_slot = i;
            dummy_batch.push_back(seq);
        }
        printf("[WARMUP] Running warmup with %d seqs × %d tokens...\n", num_seqs, warmup_seq_len);
        run(dummy_batch, true);
        cudaDeviceSynchronize();
        qwen3_reset(model);
        for (auto* seq : dummy_batch) delete seq;
        printf("[WARMUP] Done.\n");
    }
    
    // Run one forward pass then sample one token per sequence.
    // All sequences in the batch share the same SamplingParams.
    // Returns sampled token IDs (int64_t), one per sequence.
    std::vector<int64_t> run(const std::vector<Sequence*>& batch, bool is_prefill,
                             cudaStream_t stream = 0) {
        int B = (int)batch.size();
        int V = model_config.vocab_size;

        half* d_logits = is_prefill
            ? qwen3_prefill(model, batch, stream)
            : qwen3_decode(model, batch, stream);

        const SamplingParams& sp = batch[0]->sampling_params;
        launch_sampler(
            d_logits, d_temp_probs,
            B, V,
            (int)sp.top_k, sp.top_p, sp.temperature,
            sample_seed,
            d_output_ids, stream
        );
        sample_seed += B;  // each token in the batch gets a distinct seed offset

        std::vector<int64_t> result(B);
        CUDA_CHECK(cudaMemcpyAsync(result.data(), d_output_ids, B * sizeof(int64_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return result;
    }

    void capture_cudagraph() {
        // CUDA graph capture requires a non-null stream.
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        CUDAGraphState& gs = model->graph_state;
        const Qwen3Config& c = model->config;
        int max_bucket = std::min(model->max_batch, 128);
        int mbps = model->kv_cache.max_blocks_per_seq;

        // Allocate graph capture buffers
        CUDA_CHECK(cudaMalloc(&gs.g_token_ids,    max_bucket * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&gs.g_pos_ids,      max_bucket * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&gs.g_slot_map,     max_bucket * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&gs.g_block_tables, max_bucket * mbps * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&gs.g_seq_lens,     max_bucket * sizeof(int)));

        CUDA_CHECK(cudaMemset(gs.g_token_ids,    0, max_bucket * sizeof(int)));
        CUDA_CHECK(cudaMemset(gs.g_pos_ids,      0, max_bucket * sizeof(int)));
        CUDA_CHECK(cudaMemset(gs.g_slot_map,     0, max_bucket * sizeof(int64_t)));
        CUDA_CHECK(cudaMemset(gs.g_block_tables, 0, max_bucket * mbps * sizeof(int)));
        CUDA_CHECK(cudaMemset(gs.g_seq_lens,     0, max_bucket * sizeof(int)));

        gs.buckets = {1, 2, 4, 8};
        for (int b = 16; b <= max_bucket; b += 16)
            gs.buckets.push_back(b);

        CUBLAS_CHECK(cublasSetStream(model->cublas, stream));

        // Pre-allocate cuBLAS workspace so it doesn't try to allocate during capture.
        void* d_cublas_ws = nullptr;
        const size_t cublas_ws_bytes = 32ULL * 1024 * 1024;  // 32 MB
        CUDA_CHECK(cudaMalloc(&d_cublas_ws, cublas_ws_bytes));
        CUBLAS_CHECK(cublasSetWorkspace(model->cublas, d_cublas_ws, cublas_ws_bytes));

        for (int i = gs.buckets.size() - 1; i >= 0; i--) {
            int B = gs.buckets[i];
            CUDA_CHECK(cudaMemcpy(model->kv_cache.block_tables, gs.g_block_tables, B * mbps * sizeof(int), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(model->kv_cache.seq_lens, gs.g_seq_lens, B * sizeof(int), cudaMemcpyDeviceToDevice));
            // warmup
            launch_embedding(model->d_hidden, model->weights.embed_tokens, gs.g_token_ids, B, c.vocab_size, c.hidden_size, stream);
            CUDA_CHECK(cudaMemcpy(model->d_pos_ids, gs.g_pos_ids, B * sizeof(int), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(model->d_slot_map, gs.g_slot_map, B * sizeof(int64_t), cudaMemcpyDeviceToDevice));
            for (int l = 0;l < c.num_hidden_layers; l++) {
                qwen3_layer_forward(model, l, B, B, 1, false, stream);
            }
            compute_logits(model, B, 1, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            // capture
            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            launch_embedding(model->d_hidden, model->weights.embed_tokens, gs.g_token_ids, B, c.vocab_size, c.hidden_size, stream);
            for (int l = 0; l < c.num_hidden_layers; l++)
                qwen3_layer_forward(model, l, B, B, 1, false, stream);
            compute_logits(model, B, 1, stream);
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
            cudaGraphExec_t exec;
            CUDA_CHECK(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));  // only need the executable
            gs.graphs[B] = exec;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        gs.captured = true;
        // Keep d_cublas_ws alive — the captured graph nodes reference it.
        // Free it only when the graphs are destroyed (in ~ModelRunner).
        gs.g_cublas_ws = d_cublas_ws;
        // Restore default stream for eager-mode (prefill) cuBLAS calls.
        CUBLAS_CHECK(cublasSetStream(model->cublas, 0));
        CUDA_CHECK(cudaStreamDestroy(stream));
        printf("[CUDA_GRAPH] Captured %zu decode graphs (buckets 1..%d)\n", gs.buckets.size(), max_bucket);
    }

    // Free model KV state for a finished sequence at batch slot b.
    void free_seq_slot(int b) {
        qwen3_free_seq_slot(model, b);
    }

    void compute_kv_budget(Config& cfg, const Qwen3Config& model_cfg, size_t weight_bytes) {
        size_t free, total;
        cudaMemGetInfo(&free, &total);

        // Bytes per KV block: 2 (K+V) × layers × block_size × kv_heads × head_dim × sizeof(half)
        size_t block_bytes = 2ULL
            * model_cfg.num_hidden_layers
            * cfg.kv_block_size
            * model_cfg.num_key_value_heads
            * model_cfg.head_dim
            * sizeof(half);

        // Budget for activation scratch (estimate from max batch forward)
        // Dominant term: ffn intermediates = max_tokens × intermediate_size × 2
        size_t max_tokens = cfg.max_num_batched_tokens;
        size_t activation_bytes = max_tokens * (
            3ULL * model_cfg.hidden_size     // hidden + residual + normed
            + model_cfg.q_dim                // Q
            + 2ULL * model_cfg.kv_dim        // K + V
            + model_cfg.q_dim               // attn_out
            + 3ULL * model_cfg.intermediate_size  // gate + up + ffn_inter
        ) * sizeof(half);

        // Reserve some headroom for cuBLAS workspace, CUDA context, etc.
        size_t reserved = 256ULL * 1024 * 1024;  // 256 MB safety margin

        size_t budget = (size_t)(total * cfg.gpu_memory_utilization)
                    - weight_bytes
                    - activation_bytes
                    - reserved;

        cfg.num_kv_blocks = budget / block_bytes;

        // Sanity checks
        int min_blocks_per_seq = (cfg.max_model_len + cfg.kv_block_size - 1)
                            / cfg.kv_block_size;
        int max_seqs_by_memory = cfg.num_kv_blocks / min_blocks_per_seq;
        cfg.max_num_seqs = std::min(cfg.max_num_seqs, max_seqs_by_memory);

        printf("[ENGINE] GPU: %.1f GB total, %.1f GB for KV cache\n",
            total / 1e9, (cfg.num_kv_blocks * block_bytes) / 1e9);
        printf("[ENGINE] %d KV blocks (block_size=%d) → supports %d seqs × %d len\n",
            cfg.num_kv_blocks, cfg.kv_block_size,
            cfg.max_num_seqs, cfg.max_model_len);
    }
};