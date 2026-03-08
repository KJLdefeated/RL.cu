# ──────────────────────────────────────────────────────────────────────────────
# Makefile — GRPO-CUDA kernel tests
#
# Quick-start:
#   make tests              # build + run all three kernel tests
#   make test_rmsnorm       # build + run rmsnorm only
#   make generate_refs      # generate PyTorch reference data (needs torch)
#   make clean              # remove build/
#
# To target a specific GPU architecture instead of auto-detect:
#   make tests ARCH=sm_120  # Blackwell (RTX PRO 6000 / RTX 50xx)  ← this machine
#   make tests ARCH=sm_89   # Ada       (RTX 40xx)
#   make tests ARCH=sm_86   # Ampere    (RTX 30xx)
#   make tests ARCH=sm_80   # A100
# ──────────────────────────────────────────────────────────────────────────────

CUDA_HOME := /usr/local/cuda-12.8
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := sm_120
INCLUDES  := -I include
NVCCFLAGS := -O2 -std=c++17 $(INCLUDES) --gpu-architecture=$(ARCH)

BUILDDIR  := build
PYTHON    := python3

# ── Kernel test binaries ───────────────────────────────────────────────────────

$(BUILDDIR)/test_rmsnorm: src/kernels/rmsnorm.cu tests/test_rmsnorm.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILDDIR)/test_softmax: src/kernels/softmax.cu tests/test_softmax.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILDDIR)/test_swiglu: src/kernels/swiglu.cu tests/test_swiglu.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILDDIR)/test_attention: src/kernels/attention.cu tests/test_attention.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILDDIR)/test_kv_cache: src/model/kv_cache.cu tests/test_kv_cache.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILDDIR)/test_rope: src/kernels/rope.cu tests/test_rope.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILDDIR)/test_embedding: src/kernels/embedding.cu tests/test_embedding.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILDDIR)/test_linear: src/kernels/linear.cu tests/test_linear.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcublas

$(BUILDDIR)/test_sampler: src/kernels/sampler.cu tests/test_sampler.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

QWEN3_SRCS := src/model/qwen3.cu src/model/kv_cache.cu \
              src/kernels/rmsnorm.cu src/kernels/rope.cu src/kernels/attention.cu \
              src/kernels/swiglu.cu src/kernels/embedding.cu src/kernels/linear.cu \
              src/kernels/config.cpp src/kernels/weights.cpp

$(BUILDDIR)/test_qwen3: $(QWEN3_SRCS) tests/test_qwen3.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcublas

$(BUILDDIR)/bench_decode: $(QWEN3_SRCS) tests/bench_decode.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcublas

ENGINE_SRCS := $(QWEN3_SRCS) src/kernels/sampler.cu

$(BUILDDIR)/test_llmengine: $(ENGINE_SRCS) tests/test_llmengine.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcublas

$(BUILDDIR)/test_loading_weights: src/kernels/config.cpp src/kernels/weights.cpp tests/test_loading_weights.cpp | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# ── Run targets ────────────────────────────────────────────────────────────────

.PHONY: test_rmsnorm test_softmax test_swiglu test_attention test_kv_cache test_rope test_embedding test_linear test_sampler test_qwen3 test_loading_weights bench_decode test_llmengine tests generate_refs clean

test_rmsnorm: $(BUILDDIR)/test_rmsnorm
	./$(BUILDDIR)/test_rmsnorm

test_softmax: $(BUILDDIR)/test_softmax
	./$(BUILDDIR)/test_softmax

test_swiglu: $(BUILDDIR)/test_swiglu
	./$(BUILDDIR)/test_swiglu

test_attention: $(BUILDDIR)/test_attention
	./$(BUILDDIR)/test_attention

test_kv_cache: $(BUILDDIR)/test_kv_cache
	./$(BUILDDIR)/test_kv_cache

test_rope: $(BUILDDIR)/test_rope
	./$(BUILDDIR)/test_rope

test_embedding: $(BUILDDIR)/test_embedding
	./$(BUILDDIR)/test_embedding

test_linear: $(BUILDDIR)/test_linear
	./$(BUILDDIR)/test_linear

test_sampler: $(BUILDDIR)/test_sampler
	./$(BUILDDIR)/test_sampler

test_qwen3: $(BUILDDIR)/test_qwen3
	./$(BUILDDIR)/test_qwen3

test_loading_weights: $(BUILDDIR)/test_loading_weights
	./$(BUILDDIR)/test_loading_weights

bench_decode: $(BUILDDIR)/bench_decode
	./$(BUILDDIR)/bench_decode

test_llmengine: $(BUILDDIR)/test_llmengine
	./$(BUILDDIR)/test_llmengine

tests: test_rmsnorm test_softmax test_swiglu test_attention test_kv_cache test_rope test_embedding test_linear test_sampler test_qwen3 test_loading_weights test_llmengine

# ── PyTorch reference generator ───────────────────────────────────────────────

generate_refs:
	$(PYTHON) tests/generate_references.py --outdir tests/reference_data

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	rm -rf $(BUILDDIR)
