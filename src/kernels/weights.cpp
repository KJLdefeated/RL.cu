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
// Conversion helpers (BF16 → FP16/FP32, write into caller-provided buffer)
// ============================================================================

static void bf16_to_fp16(half* dst, const void* src_bf16, size_t n) {
    const uint16_t* src = (const uint16_t*)src_bf16;
    for (size_t i = 0; i < n; i++) {
        uint32_t fp32_bits = ((uint32_t)src[i]) << 16;
        float f;
        memcpy(&f, &fp32_bits, sizeof(float));
        dst[i] = __float2half(f);
    }
}

static void bf16_to_fp32(float* dst, const void* src_bf16, size_t n) {
    const uint16_t* src = (const uint16_t*)src_bf16;
    for (size_t i = 0; i < n; i++) {
        uint32_t fp32_bits = ((uint32_t)src[i]) << 16;
        memcpy(&dst[i], &fp32_bits, sizeof(float));
    }
}

static void fp16_to_fp32_cpu(float* dst, const void* src_fp16, size_t n) {
    const uint16_t* src = (const uint16_t*)src_fp16;
    for (size_t i = 0; i < n; i++) dst[i] = __half2float(*(const half*)&src[i]);
}

// ============================================================================
// Main loader — flat buffer pattern
//
// Two bulk GPU allocations:
//   fp16_pool: all FP16 weight tensors (projections + embedding)
//   fp32_pool: all FP32 weight tensors (norms)
//
// Layout matches Qwen3Gradients pool layout for flat optimizer step:
//   FP16: per-layer [q, k, v, o, gate, up, down], then embed
//   FP32: per-layer [input_norm, q_norm, k_norm, post_attn_norm], then final_norm
// ============================================================================

Qwen3Weights load_weights(const std::string& model_dir, const Qwen3Config& cfg) {
    Qwen3Weights w;
    const int L = cfg.num_hidden_layers;
    const int H = cfg.hidden_size;
    w.layers.resize(L);

    // --- Compute flat pool sizes ---
    const size_t q_proj_sz    = (size_t)cfg.q_dim * H;
    const size_t k_proj_sz    = (size_t)cfg.kv_dim * H;
    const size_t v_proj_sz    = (size_t)cfg.kv_dim * H;
    const size_t o_proj_sz    = (size_t)H * cfg.q_dim;
    const size_t gate_proj_sz = (size_t)cfg.intermediate_size * H;
    const size_t up_proj_sz   = (size_t)cfg.intermediate_size * H;
    const size_t down_proj_sz = (size_t)H * cfg.intermediate_size;
    const size_t per_layer_fp16 = q_proj_sz + k_proj_sz + v_proj_sz + o_proj_sz
                                 + gate_proj_sz + up_proj_sz + down_proj_sz;
    const size_t embed_sz = (size_t)cfg.vocab_size * H;
    const size_t total_fp16 = per_layer_fp16 * L + embed_sz;

    const size_t per_layer_fp32 = (size_t)H + cfg.head_dim + cfg.head_dim + H;
    const size_t final_norm_sz = H;
    const size_t total_fp32 = per_layer_fp32 * L + final_norm_sz;

    w.fp16_pool_elems = total_fp16;
    w.fp32_pool_elems = total_fp32;

    // --- Allocate host staging buffers ---
    std::vector<half>  h_fp16(total_fp16);
    std::vector<float> h_fp32(total_fp32);

    // --- Open safetensors files ---
    std::vector<SafetensorsFile> files;
    std::unordered_map<std::string, size_t> tensor_to_file;

    std::string single = model_dir + "/model.safetensors";
    std::string index_path = model_dir + "/model.safetensors.index.json";

    if (std::filesystem::exists(single) && !std::filesystem::exists(index_path)) {
        files.push_back(open_safetensors(single));
        for (auto& [key, val] : files[0].header.items()) {
            if (key == "__metadata__") continue;
            tensor_to_file[key] = 0;
        }
    } else if (std::filesystem::exists(index_path)) {
        std::ifstream idx_f(index_path);
        json idx = json::parse(idx_f);
        auto& weight_map = idx["weight_map"];
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

    bool is_bf16 = (cfg.torch_dtype == "bfloat16");

    // Helper: find tensor data in safetensors
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

    // Convert and write tensor into FP16 staging buffer at given offset
    auto stage_fp16 = [&](const std::string& name, size_t expected, half* dst) {
        const void* data; size_t n; std::string dtype;
        if (!find_tensor(name, expected, &data, &n, &dtype)) return;
        if (is_bf16 && dtype == "BF16") {
            bf16_to_fp16(dst, data, n);
        } else if (dtype == "F16") {
            memcpy(dst, data, n * sizeof(half));
        } else {
            fprintf(stderr, "WARNING: unexpected dtype '%s' for FP16 tensor '%s'\n",
                    dtype.c_str(), name.c_str());
        }
    };

    // Convert and write tensor into FP32 staging buffer at given offset
    auto stage_fp32 = [&](const std::string& name, size_t expected, float* dst) {
        const void* data; size_t n; std::string dtype;
        if (!find_tensor(name, expected, &data, &n, &dtype)) return;
        if (is_bf16 && dtype == "BF16") {
            bf16_to_fp32(dst, data, n);
        } else if (dtype == "F32") {
            memcpy(dst, data, n * sizeof(float));
        } else if (dtype == "F16") {
            fp16_to_fp32_cpu(dst, data, n);
        }
    };

    // --- Stage FP16 tensors into host buffer ---
    // Layout: per-layer [q, k, v, o, gate, up, down], then embed
    half* hp = h_fp16.data();
    for (int i = 0; i < L; i++) {
        std::string prefix = "model.layers." + std::to_string(i);
        stage_fp16(prefix + ".self_attn.q_proj.weight",    q_proj_sz,    hp); hp += q_proj_sz;
        stage_fp16(prefix + ".self_attn.k_proj.weight",    k_proj_sz,    hp); hp += k_proj_sz;
        stage_fp16(prefix + ".self_attn.v_proj.weight",    v_proj_sz,    hp); hp += v_proj_sz;
        stage_fp16(prefix + ".self_attn.o_proj.weight",    o_proj_sz,    hp); hp += o_proj_sz;
        stage_fp16(prefix + ".mlp.gate_proj.weight",       gate_proj_sz, hp); hp += gate_proj_sz;
        stage_fp16(prefix + ".mlp.up_proj.weight",         up_proj_sz,   hp); hp += up_proj_sz;
        stage_fp16(prefix + ".mlp.down_proj.weight",       down_proj_sz, hp); hp += down_proj_sz;
    }
    stage_fp16("model.embed_tokens.weight", embed_sz, hp);

    // --- Stage FP32 tensors into host buffer ---
    // Layout: per-layer [input_norm, q_norm, k_norm, post_attn_norm], then final_norm
    float* fp = h_fp32.data();
    for (int i = 0; i < L; i++) {
        std::string prefix = "model.layers." + std::to_string(i);
        stage_fp32(prefix + ".input_layernorm.weight",          H,           fp); fp += H;
        stage_fp32(prefix + ".self_attn.q_norm.weight",         cfg.head_dim, fp); fp += cfg.head_dim;
        stage_fp32(prefix + ".self_attn.k_norm.weight",         cfg.head_dim, fp); fp += cfg.head_dim;
        stage_fp32(prefix + ".post_attention_layernorm.weight",  H,           fp); fp += H;
    }
    stage_fp32("model.norm.weight", final_norm_sz, fp);

    // --- One cudaMalloc + cudaMemcpy per pool ---
    size_t fp16_bytes = total_fp16 * sizeof(half);
    size_t fp32_bytes = total_fp32 * sizeof(float);

    CUDA_CHECK(cudaMalloc(&w.fp16_pool, fp16_bytes));
    CUDA_CHECK(cudaMalloc(&w.fp32_pool, fp32_bytes));
    CUDA_CHECK(cudaMemcpy(w.fp16_pool, h_fp16.data(), fp16_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w.fp32_pool, h_fp32.data(), fp32_bytes, cudaMemcpyHostToDevice));

    w.total_bytes = fp16_bytes + fp32_bytes;

    // --- Assign convenience pointers (offsets into pools) ---
    half* hp2 = w.fp16_pool;
    for (int i = 0; i < L; i++) {
        auto& lw = w.layers[i];
        lw.q_proj    = hp2; hp2 += q_proj_sz;
        lw.k_proj    = hp2; hp2 += k_proj_sz;
        lw.v_proj    = hp2; hp2 += v_proj_sz;
        lw.o_proj    = hp2; hp2 += o_proj_sz;
        lw.gate_proj = hp2; hp2 += gate_proj_sz;
        lw.up_proj   = hp2; hp2 += up_proj_sz;
        lw.down_proj = hp2; hp2 += down_proj_sz;
    }
    w.embed_tokens = hp2;
    w.lm_head = cfg.tie_word_embeddings ? w.embed_tokens : (hp2 + embed_sz);  // tied

    float* fp2 = w.fp32_pool;
    for (int i = 0; i < L; i++) {
        auto& lw = w.layers[i];
        lw.input_layernorm     = fp2; fp2 += H;
        lw.q_norm              = fp2; fp2 += cfg.head_dim;
        lw.k_norm              = fp2; fp2 += cfg.head_dim;
        lw.post_attn_layernorm = fp2; fp2 += H;
    }
    w.final_norm = fp2;

    printf("[WEIGHTS] Loaded %.2f MB to GPU (%d layers, flat: fp16=%zu fp32=%zu params)\n",
           w.total_bytes / (1024.0 * 1024.0), L, total_fp16, total_fp32);

    return w;
}

void free_weights(Qwen3Weights& w) {
    if (w.fp16_pool) { cudaFree(w.fp16_pool); w.fp16_pool = nullptr; }
    if (w.fp32_pool) { cudaFree(w.fp32_pool); w.fp32_pool = nullptr; }
    // All layer/embed/norm pointers were offsets into pools — nothing else to free.
    w.embed_tokens = nullptr;
    w.lm_head      = nullptr;
    w.final_norm   = nullptr;
    w.layers.clear();
    w.total_bytes = 0;
    w.fp16_pool_elems = 0;
    w.fp32_pool_elems = 0;
}