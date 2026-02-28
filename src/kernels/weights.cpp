#include "model/weights.h"
#include "third_party/json.hpp"
#include "model/config.h"
#include <cuda_runtime.h>
#include <fstream>
#include <unordered_map>
#include <filesystem>
#include <cstring>
#include <cstdio>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

using json = nlohmann::json;

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Safetensors parser
// ============================================================================
// File format:
//   [0..7]       uint64_le  header_size
//   [8..8+N-1]   JSON       {"tensor_name": {"dtype":"BF16","shape":[...],"data_offsets":[start,end]}, ...}
//   [8+N..]      bytes      raw tensor data (offsets are relative to data start = 8+N)

struct SafetensorsFile {
    int fd = -1;
    void* mapped = MAP_FAILED;
    size_t file_size = 0;
    uint64_t header_size = 0;
    const uint8_t* data_start = nullptr;  // pointer to first byte of tensor data
    json header;                          // parsed JSON header
    
    SafetensorsFile() = default;

    // Move constructor: transfer ownership, null out source so destructor is a no-op
    SafetensorsFile(SafetensorsFile&& o) noexcept
        : fd(o.fd), mapped(o.mapped), file_size(o.file_size),
          header_size(o.header_size), data_start(o.data_start),
          header(std::move(o.header))
    {
        o.fd     = -1;
        o.mapped = MAP_FAILED;
    }

    SafetensorsFile(const SafetensorsFile&) = delete;  // prevent accidental copy

    ~SafetensorsFile() {
        if (mapped != MAP_FAILED) munmap(mapped, file_size);
        if (fd >= 0) close(fd);
    }
};

static SafetensorsFile open_safetensors(const std::string& path) {
    SafetensorsFile sf;

    sf.fd = open(path.c_str(), O_RDONLY);
    if (sf.fd < 0) {
        fprintf(stderr, "Cannot open %s\n", path.c_str());
        exit(1);
    }

    struct stat st;
    fstat(sf.fd, &st);
    sf.file_size = st.st_size;

    sf.mapped = mmap(nullptr, sf.file_size, PROT_READ, MAP_PRIVATE, sf.fd, 0);
    if (sf.mapped == MAP_FAILED) {
        fprintf(stderr, "mmap failed for %s\n", path.c_str());
        exit(1);
    }

    // Read 8-byte header size (little-endian uint64)
    memcpy(&sf.header_size, sf.mapped, sizeof(uint64_t));

    // Parse JSON header
    const char* json_start = (const char*)sf.mapped + 8;
    sf.header = json::parse(json_start, json_start + sf.header_size);

    // Data starts right after the header
    sf.data_start = (const uint8_t*)sf.mapped + 8 + sf.header_size;

    return sf;
}

// Get raw pointer to tensor data in mmaped file
static const void* get_tensor_data(const SafetensorsFile& sf, const std::string& name,
                                    size_t* out_bytes = nullptr) {
    if (!sf.header.contains(name)) return nullptr;

    auto& info = sf.header[name];
    auto offsets = info["data_offsets"];
    size_t start = offsets[0].get<size_t>();
    size_t end   = offsets[1].get<size_t>();

    if (out_bytes) *out_bytes = end - start;
    return sf.data_start + start;
}

// dtype string -> element size in bytes
static size_t dtype_size(const std::string& dtype) {
    if (dtype == "BF16" || dtype == "F16") return 2;
    if (dtype == "F32") return 4;
    if (dtype == "F64") return 8;
    if (dtype == "I32") return 4;
    if (dtype == "I64") return 8;
    fprintf(stderr, "Unknown dtype: %s\n", dtype.c_str());
    return 2;
}

// ============================================================================
// GPU allocation + upload helpers
// ============================================================================

static half* alloc_and_upload(const void* src, size_t bytes, size_t* total) {
    half* dst = nullptr;
    CUDA_CHECK(cudaMalloc(&dst, bytes));
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
    *total += bytes;
    return dst;
}

// Upload with BF16->FP16 conversion on CPU (for large weight tensors)
static half* alloc_upload_bf16_to_fp16(const void* src_bf16, size_t num_elements, size_t* total) {
    std::vector<half> fp16_buf(num_elements);
    const uint16_t* src = (const uint16_t*)src_bf16;
    for (size_t i = 0; i < num_elements; i++) {
        uint32_t fp32_bits = ((uint32_t)src[i]) << 16;
        float f;
        memcpy(&f, &fp32_bits, sizeof(float));
        fp16_buf[i] = __float2half(f);
    }
    size_t bytes = num_elements * sizeof(half);
    half* dst = nullptr;
    CUDA_CHECK(cudaMalloc(&dst, bytes));
    CUDA_CHECK(cudaMemcpy(dst, fp16_buf.data(), bytes, cudaMemcpyHostToDevice));
    *total += bytes;
    return dst;
}

// Upload with BF16->FP32 conversion (for norm weights — avoids FP16 range loss)
static float* alloc_upload_bf16_to_fp32(const void* src_bf16, size_t num_elements, size_t* total) {
    std::vector<float> fp32_buf(num_elements);
    const uint16_t* src = (const uint16_t*)src_bf16;
    for (size_t i = 0; i < num_elements; i++) {
        uint32_t fp32_bits = ((uint32_t)src[i]) << 16;
        memcpy(&fp32_buf[i], &fp32_bits, sizeof(float));
    }
    size_t bytes = num_elements * sizeof(float);
    float* dst = nullptr;
    CUDA_CHECK(cudaMalloc(&dst, bytes));
    CUDA_CHECK(cudaMemcpy(dst, fp32_buf.data(), bytes, cudaMemcpyHostToDevice));
    *total += bytes;
    return dst;
}

// ============================================================================
// Main loader
// ============================================================================

Qwen3Weights load_weights(const std::string& model_dir, const Qwen3Config& cfg) {
    Qwen3Weights w;
    w.layers.resize(cfg.num_hidden_layers);

    // Find safetensors files
    // Qwen3-0.6B has a single model.safetensors
    // Larger models have model-00001-of-NNNNN.safetensors + model.safetensors.index.json
    std::vector<SafetensorsFile> files;
    std::unordered_map<std::string, size_t> tensor_to_file;  // tensor_name -> file index

    std::string single = model_dir + "/model.safetensors";
    std::string index_path = model_dir + "/model.safetensors.index.json";

    if (std::filesystem::exists(single) && !std::filesystem::exists(index_path)) {
        // Single file (e.g., Qwen3-0.6B)
        files.push_back(open_safetensors(single));
        for (auto& [key, val] : files[0].header.items()) {
            if (key == "__metadata__") continue;
            tensor_to_file[key] = 0;
        }
    } else if (std::filesystem::exists(index_path)) {
        // Sharded model: read index to find which file has which tensor
        std::ifstream idx_f(index_path);
        json idx = json::parse(idx_f);
        auto& weight_map = idx["weight_map"];

        // Collect unique filenames
        std::unordered_map<std::string, size_t> file_indices;
        for (auto& [tensor_name, filename] : weight_map.items()) {
            std::string fname = filename.get<std::string>();
            if (file_indices.find(fname) == file_indices.end()) {
                size_t idx = files.size();
                file_indices[fname] = idx;
                files.push_back(open_safetensors(model_dir + "/" + fname));
            }
            tensor_to_file[tensor_name] = file_indices[fname];
        }
    } else {
        fprintf(stderr, "No safetensors found in %s\n", model_dir.c_str());
        exit(1);
    }

    printf("[WEIGHTS] Loading from %zu safetensors file(s)...\n", files.size());

    // Helper: find and upload a tensor by name
    bool is_bf16 = (cfg.torch_dtype == "bfloat16");

    // Helper: look up a tensor's data pointer + element count
    auto find_tensor = [&](const std::string& name, size_t expected_elements,
                           const void** out_data, size_t* out_num_elements,
                           std::string* out_dtype) -> bool {
        auto it = tensor_to_file.find(name);
        if (it == tensor_to_file.end()) {
            fprintf(stderr, "WARNING: tensor '%s' not found in safetensors\n", name.c_str());
            return false;
        }
        size_t file_idx = it->second;
        size_t bytes = 0;
        *out_data = get_tensor_data(files[file_idx], name, &bytes);
        if (!*out_data) {
            fprintf(stderr, "WARNING: tensor '%s' data not found\n", name.c_str());
            return false;
        }
        auto& info = files[file_idx].header[name];
        *out_dtype = info["dtype"].get<std::string>();
        *out_num_elements = bytes / dtype_size(*out_dtype);
        if (expected_elements > 0 && *out_num_elements != expected_elements) {
            fprintf(stderr, "WARNING: '%s' expected %zu elements, got %zu\n",
                    name.c_str(), expected_elements, *out_num_elements);
        }
        return true;
    };

    // Upload as FP16 (BF16->FP16 conversion if needed)
    auto upload_tensor = [&](const std::string& name, size_t expected_elements) -> half* {
        const void* data; size_t n; std::string dtype;
        if (!find_tensor(name, expected_elements, &data, &n, &dtype)) return nullptr;
        if (is_bf16 && dtype == "BF16")
            return alloc_upload_bf16_to_fp16(data, n, &w.total_bytes);
        return alloc_and_upload(data, n * dtype_size(dtype), &w.total_bytes);
    };

    // Upload as FP32 (BF16->FP32; avoids FP16 range/precision loss for norm weights)
    auto upload_tensor_fp32 = [&](const std::string& name, size_t expected_elements) -> float* {
        const void* data; size_t n; std::string dtype;
        if (!find_tensor(name, expected_elements, &data, &n, &dtype)) return nullptr;
        if (is_bf16 && dtype == "BF16")
            return alloc_upload_bf16_to_fp32(data, n, &w.total_bytes);
        // Already FP32 — direct upload
        if (dtype == "F32") {
            float* dst = nullptr;
            size_t bytes = n * sizeof(float);
            CUDA_CHECK(cudaMalloc(&dst, bytes));
            CUDA_CHECK(cudaMemcpy(dst, data, bytes, cudaMemcpyHostToDevice));
            w.total_bytes += bytes;
            return dst;
        }
        // FP16 -> FP32 fallback
        std::vector<float> buf(n);
        const uint16_t* src = (const uint16_t*)data;
        for (size_t i = 0; i < n; i++) buf[i] = __half2float(*(const half*)&src[i]);
        float* dst = nullptr;
        CUDA_CHECK(cudaMalloc(&dst, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dst, buf.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        w.total_bytes += n * sizeof(float);
        return dst;
    };

    // --- Upload global tensors ---
    size_t embed_elems = (size_t)cfg.vocab_size * cfg.hidden_size;
    w.embed_tokens = upload_tensor("model.embed_tokens.weight", embed_elems);
    w.final_norm   = upload_tensor_fp32("model.norm.weight", cfg.hidden_size);

    // --- Upload per-layer tensors ---
    for (int i = 0; i < cfg.num_hidden_layers; i++) {
        auto& L = w.layers[i];
        std::string prefix = "model.layers." + std::to_string(i);

        L.input_layernorm     = upload_tensor_fp32(prefix + ".input_layernorm.weight",
                                                   cfg.hidden_size);
        L.q_proj              = upload_tensor(prefix + ".self_attn.q_proj.weight",
                                              (size_t)cfg.q_dim * cfg.hidden_size);
        L.k_proj              = upload_tensor(prefix + ".self_attn.k_proj.weight",
                                              (size_t)cfg.kv_dim * cfg.hidden_size);
        L.v_proj              = upload_tensor(prefix + ".self_attn.v_proj.weight",
                                              (size_t)cfg.kv_dim * cfg.hidden_size);
        L.o_proj              = upload_tensor(prefix + ".self_attn.o_proj.weight",
                                              (size_t)cfg.hidden_size * cfg.q_dim);
        L.q_norm              = upload_tensor_fp32(prefix + ".self_attn.q_norm.weight",
                                                   cfg.head_dim);
        L.k_norm              = upload_tensor_fp32(prefix + ".self_attn.k_norm.weight",
                                                   cfg.head_dim);
        L.post_attn_layernorm = upload_tensor_fp32(prefix + ".post_attention_layernorm.weight",
                                                   cfg.hidden_size);
        L.gate_proj           = upload_tensor(prefix + ".mlp.gate_proj.weight",
                                              (size_t)cfg.intermediate_size * cfg.hidden_size);
        L.up_proj             = upload_tensor(prefix + ".mlp.up_proj.weight",
                                              (size_t)cfg.intermediate_size * cfg.hidden_size);
        L.down_proj           = upload_tensor(prefix + ".mlp.down_proj.weight",
                                              (size_t)cfg.hidden_size * cfg.intermediate_size);
    }

    printf("[WEIGHTS] Loaded %.2f MB to GPU (%d layers)\n",
           w.total_bytes / (1024.0 * 1024.0), cfg.num_hidden_layers);

    return w;
}

void free_weights(Qwen3Weights& w) {
    // Generic: works for both half* and float*
    auto safe_free = [](auto*& p) {
        if (p) { cudaFree(p); p = nullptr; }
    };

    safe_free(w.embed_tokens);
    safe_free(w.final_norm);

    for (auto& L : w.layers) {
        safe_free(L.input_layernorm);
        safe_free(L.q_proj);
        safe_free(L.k_proj);
        safe_free(L.v_proj);
        safe_free(L.o_proj);
        safe_free(L.q_norm);
        safe_free(L.k_norm);
        safe_free(L.post_attn_layernorm);
        safe_free(L.gate_proj);
        safe_free(L.up_proj);
        safe_free(L.down_proj);
    }
    w.layers.clear();
    w.total_bytes = 0;
}