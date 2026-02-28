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

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# ── Run targets ────────────────────────────────────────────────────────────────

.PHONY: test_rmsnorm test_softmax test_swiglu test_attention test_kv_cache test_rope test_embedding tests generate_refs clean

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

tests: test_rmsnorm test_softmax test_swiglu test_attention test_kv_cache test_rope test_embedding

# ── PyTorch reference generator ───────────────────────────────────────────────

generate_refs:
	$(PYTHON) tests/generate_references.py --outdir tests/reference_data

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	rm -rf $(BUILDDIR)
