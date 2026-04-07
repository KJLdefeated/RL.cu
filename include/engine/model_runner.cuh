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
    float*             d_temp_probs  = nullptr;  // [max_num_seqs, vocab_size] FP32 sampler workspace
    int64_t*           d_output_ids  = nullptr;  // [max_num_seqs] sampled token IDs
    void*              d_cublas_ws   = nullptr;  // pre-allocated cuBLAS workspace (32 MB)
    unsigned long long sample_seed   = 42ULL;    // PRNG seed, incremented each step

    ModelRunner(const Config& cfg) : config(cfg) {
        model_config = load_config(cfg.model);
        model_config.compute_derived();
        model = new Qwen3Model{};
        model->config = model_config;
        model->weights = load_weights(cfg.model, model_config);
        // Compute KV cache budget and adjust max_num_seqs
        compute_kv_budget(config, model_config, model->weights.total_bytes);
        // Initialize model with max batch/seq for buffer allocation.
        qwen3_init(model, config.max_num_seqs, config.max_model_len, config.max_num_batched_tokens, config.num_kv_blocks);
        block_size    = cfg.kv_block_size;
        enforce_eager = cfg.enforce_eager;

        // Sampler buffers — allocated once for the maximum batch size.
        int V = model_config.vocab_size;
        CUDA_CHECK(cudaMalloc(&d_temp_probs, (long)config.max_num_seqs * V * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_ids, config.max_num_seqs * sizeof(int64_t)));

        // Pre-allocate cuBLAS workspace so cuBLAS never tries to allocate from the
        // async memory pool during warmup or inference.  With a large KV cache (e.g.
        // 80+ GB), the async pool can be exhausted, causing cublasGemmEx → status=14.
        // 32 MB covers all GEMM shapes in Qwen3 (max: T×3072×1024 with T≤max_tokens).
        CUDA_CHECK(cudaMalloc(&d_cublas_ws, 32ULL * 1024 * 1024));
        CUBLAS_CHECK(cublasSetWorkspace(model->cublas, d_cublas_ws, 32ULL * 1024 * 1024));
        // Flush any pending CUDA errors before warmup so they don't pollute cuBLAS.
        CUDA_CHECK(cudaDeviceSynchronize());

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
            // d_cublas_ws is freed below (it's the same d_cublas_ws pre-allocated
            // in the constructor; capture_cudagraph no longer allocates its own).
        }
        cudaFree(d_cublas_ws);   // free once, unconditionally
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

        CUDAGraphState& gs = model->graph_state;
        half* d_logits = is_prefill
            ? qwen3_prefill(model, batch, stream)
            : qwen3_decode(model, batch, stream);  // TODO: re-enable graph path after debugging

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
        int max_bucket = std::min(model->max_batch, 256);
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

        gs.buckets.clear();
        for (int b : {1, 2, 4, 8}) {
            if (b <= max_bucket) gs.buckets.push_back(b);
        }
        for (int b = 16; b < max_bucket; b += 16)
            gs.buckets.push_back(b);
        if (gs.buckets.empty() || gs.buckets.back() != max_bucket)
            gs.buckets.push_back(max_bucket);  // always cover the actual max batch

        CUBLAS_CHECK(cublasSetStream(model->cublas, stream));

        // Workspace is pre-allocated by the constructor (d_cublas_ws) and already
        // set on the handle.  Re-set it here after switching to the capture stream
        // so the workspace association follows the handle's current stream.
        CUBLAS_CHECK(cublasSetWorkspace(model->cublas, d_cublas_ws, 32ULL * 1024 * 1024));

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
            // D2D: update model buffers from graph's fixed "parameter" buffers.
            // These D2D copies MUST be inside the capture so they re-execute on
            // every graph launch (the H2D into gs.g_* happens outside the graph,
            // but the model-internal buffers that layer kernels read from are
            // updated here inside the graph).
            CUDA_CHECK(cudaMemcpyAsync(model->d_pos_ids, gs.g_pos_ids,
                                       B * sizeof(int), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(model->d_slot_map, gs.g_slot_map,
                                       B * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(model->kv_cache.block_tables, gs.g_block_tables,
                                       (long)B * mbps * sizeof(int), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(model->kv_cache.seq_lens, gs.g_seq_lens,
                                       B * sizeof(int), cudaMemcpyDeviceToDevice, stream));
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

        size_t block_bytes = 2ULL
            * model_cfg.num_hidden_layers * cfg.kv_block_size
            * model_cfg.num_key_value_heads * model_cfg.head_dim * sizeof(half);

        size_t per_token_act = (
            3ULL * model_cfg.hidden_size      // hidden + residual + normed
            + model_cfg.q_dim                 // Q
            + 2ULL * model_cfg.kv_dim         // K + V
            + model_cfg.q_dim                 // attn_out
            + 3ULL * model_cfg.intermediate_size  // gate + up + ffn_inter
        ) * sizeof(half);

        size_t reserved     = 256ULL * 1024 * 1024;
        // Use `free` (not `total`) so we respect other processes already occupying
        // GPU memory.  `free` is measured after weights are loaded, so weight_bytes
        // are already excluded — no separate subtraction needed.
        if (free < reserved) free = reserved;  // safety: avoid underflow
        size_t total_budget = (size_t)(free * cfg.gpu_memory_utilization) - reserved;
        (void)weight_bytes;  // no longer needed; kept in signature for documentation

        int min_blocks_per_seq = (cfg.max_model_len + cfg.kv_block_size - 1)
                               / cfg.kv_block_size;

        // Iterative solver: scratch = est_max_seqs × max_model_len tokens (the actual
        // ceiling — we never process more tokens than that in one forward pass), capped
        // at the user's max_num_batched_tokens.  Iterate until max_num_seqs stabilises.
        int est = cfg.max_num_seqs;
        for (int iter = 0; iter < 10; iter++) {
            size_t scratch_tokens = std::min(
                (size_t)est * (size_t)cfg.max_model_len,
                (size_t)cfg.max_num_batched_tokens);
            size_t act_bytes = scratch_tokens * per_token_act;
            if (act_bytes >= total_budget) { est = 1; break; }

            int kv_blocks  = (int)((total_budget - act_bytes) / block_bytes);
            int new_est    = std::min(kv_blocks / min_blocks_per_seq, cfg.max_num_seqs);
            if (std::abs(new_est - est) <= 1) { est = std::min(new_est, est); break; }
            est = new_est;
        }
        cfg.max_num_seqs = est;

        // Recompute final values with converged max_num_seqs.
        size_t scratch_tokens = std::min(
            (size_t)cfg.max_num_seqs * (size_t)cfg.max_model_len,
            (size_t)cfg.max_num_batched_tokens);
        cfg.num_kv_blocks = (int)((total_budget - scratch_tokens * per_token_act) / block_bytes);

        // Update max_num_batched_tokens so qwen3_init allocates the right-sized scratch
        // buffers.  Keeping it at 262 K would re-waste the 4-8 GB we freed for KV blocks.
        cfg.max_num_batched_tokens = (int)scratch_tokens;

        printf("[ENGINE] GPU: %.1f GB total  %.1f GB free after weights, %.1f GB for KV cache\n",
            total / 1e9, free / 1e9, (cfg.num_kv_blocks * block_bytes) / 1e9);
        printf("[ENGINE] %d KV blocks (block_size=%d) → supports %d seqs × %d len\n",
            cfg.num_kv_blocks, cfg.kv_block_size, cfg.max_num_seqs, cfg.max_model_len);
    }
};