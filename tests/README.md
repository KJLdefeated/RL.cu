# Tests

## Directory Structure

```
tests/
├── kernels/      # Unit tests for individual CUDA kernels
│   ├── test_rmsnorm.cu
│   ├── test_rmsnorm_backward.cu
│   ├── test_softmax.cu
│   ├── test_swiglu.cu
│   ├── test_swiglu_backward.cu
│   ├── test_attention.cu          # FA2 prefill + paged decode
│   ├── test_attention_backward.cu
│   ├── test_rope.cu
│   ├── test_rope_backward.cu
│   ├── test_embedding.cu
│   ├── test_embedding_backward.cu
│   ├── test_linear.cu
│   ├── test_linear_backward.cu
│   ├── test_fused_norm_linear.cu
│   ├── test_sampler.cu
│   ├── test_kv_cache.cu
│   ├── test_adamw.cu
│   ├── test_fwd_bwd.cu
│   ├── test_dataloader.cpp
│   ├── test_loading_weights.cpp
│   └── test_lr_scheduler.cpp
│
├── models/       # End-to-end model tests (Qwen3, LLMEngine)
│   ├── test_qwen3.cu
│   ├── test_qwen3_forward.cu
│   ├── test_qwen3_backward.cu
│   └── test_llmengine.cu         # 11 integration tests (correctness, throughput)
│
└── training/     # Training loop tests
    ├── train_sft.cu
    └── train_grpo.cu
```

## Building & Running Tests

```bash
# Build and run all tests
make tests

# Build and run a single test
make test_attention
make test_rmsnorm

# Build only (no run)
make build/test_attention

# Debug build
make test_attention BUILD_TYPE=Debug
```

## Profiling with Nsight Systems (nsys)

### Quick Profile

Profile a single test to get a timeline of kernel launches, memory transfers, and CUDA API calls:

```bash
# Build the test first
make build/test_attention

# Profile and generate a .nsys-rep report
nsys profile --stats=true ./build/test_attention
```

### Detailed Kernel Profiling

```bash
# Trace CUDA kernels + cuBLAS with full names
nsys profile \
  --trace=cuda,cublas \
  --stats=true \
  --output=reports/attention_profile \
  ./build/test_attention

# Open the report in Nsight Systems GUI
nsys-ui reports/attention_profile.nsys-rep
```

### Profiling the Full Engine

```bash
# Profile LLMEngine throughput test (most representative of real workload)
nsys profile \
  --trace=cuda,cublas \
  --cuda-graph-trace=node \
  --stats=true \
  --output=reports/engine_profile \
  ./build/test_llmengine

# Profile GRPO training
nsys profile \
  --trace=cuda,cublas \
  --stats=true \
  --output=reports/grpo_profile \
  ./build/train_grpo
```

### Key nsys Flags

| Flag | Description |
|------|-------------|
| `--trace=cuda,cublas` | Capture CUDA runtime + cuBLAS API calls |
| `--cuda-graph-trace=node` | Trace individual kernels inside CUDA graphs |
| `--stats=true` | Print summary statistics to stdout after profiling |
| `--output=<name>` | Output file path (produces `<name>.nsys-rep`) |
| `--capture-range=cudaProfilerApi` | Only profile between `cudaProfilerStart/Stop` calls |
| `-s none` | Skip CPU sampling (faster, GPU-only analysis) |

### Targeted Profiling with cudaProfilerApi

To profile only a specific section of code, add profiler markers:

```cpp
#include <cuda_profiler_api.h>

cudaProfilerStart();
// ... code to profile ...
cudaProfilerStop();
```

Then run with:

```bash
nsys profile --capture-range=cudaProfilerApi ./build/test_attention
```

### Reading nsys Output

The `--stats=true` flag prints tables like:

- **CUDA Kernel Statistics** -- time per kernel, launch count, avg/min/max duration
- **CUDA Memory Operation Statistics** -- HtoD, DtoH, DtoD transfers
- **CUDA API Statistics** -- cudaLaunchKernel, cudaMemcpy, cudaMalloc overhead

Look for:
1. **Hottest kernels** -- which kernels dominate wall time
2. **Gaps between kernels** -- idle GPU time (scheduling overhead, CPU bottlenecks)
3. **Small frequent launches** -- candidates for kernel fusion or CUDA graphs
4. **Memory transfer volume** -- unexpected HtoD/DtoH copies in the critical path
