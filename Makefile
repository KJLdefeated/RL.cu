# ──────────────────────────────────────────────────────────────────────────────
# Makefile — thin wrapper around CMake
#
# Usage:
#   make                    # configure + build all targets
#   make test_attention     # build + run a single test
#   make tests              # build + run all kernel tests
#   make train_grpo         # build + run GRPO training (default args)
#   make clean              # remove build/
#
# CMake options (passed through):
#   make ARCH=90            # target sm_90 (default: native auto-detect)
#   make BUILD_TYPE=Debug   # debug build
# ──────────────────────────────────────────────────────────────────────────────

BUILDDIR   := build
BUILD_TYPE ?= Release
ARCH       ?= 120
PYTHON     := python3
CMAKE_ARGS := -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
              -DCMAKE_CUDA_ARCHITECTURES=$(ARCH)

NPROC := $(shell nproc 2>/dev/null || echo 4)

# ── Core build rules ─────────────────────────────────────────────────────────

.PHONY: all configure clean

all: configure
	cmake --build $(BUILDDIR) -j$(NPROC)

configure: $(BUILDDIR)/CMakeCache.txt

$(BUILDDIR)/CMakeCache.txt: CMakeLists.txt
	cmake -B $(BUILDDIR) $(CMAKE_ARGS)

# Build a specific target: make build/<name>
$(BUILDDIR)/%: configure
	cmake --build $(BUILDDIR) --target $* -j$(NPROC)

# ── Test targets (build + run) ────────────────────────────────────────────────

KERNEL_TESTS := test_rmsnorm test_softmax test_swiglu test_attention \
                test_rope test_embedding test_linear test_sampler \
                test_fused_norm_linear test_kv_cache

BACKWARD_TESTS := test_linear_backward test_embedding_backward \
                  test_rmsnorm_backward test_swiglu_backward \
                  test_rope_backward test_attention_backward

MODEL_TESTS := test_qwen3 test_qwen3_forward test_qwen3_backward \
               test_fwd_bwd test_adamw test_llmengine

CPP_TESTS := test_dataloader test_lr_scheduler test_loading_weights

ALL_TESTS := $(KERNEL_TESTS) $(BACKWARD_TESTS) $(MODEL_TESTS) $(CPP_TESTS)

.PHONY: tests $(ALL_TESTS) bench_decode train_sft train_grpo

# Generic: "make test_foo" builds and runs ./build/test_foo
$(ALL_TESTS): %: $(BUILDDIR)/%
	./$(BUILDDIR)/$@

bench_decode: $(BUILDDIR)/bench_decode
	./$(BUILDDIR)/bench_decode

train_sft: $(BUILDDIR)/train_sft
	./$(BUILDDIR)/train_sft

train_grpo: $(BUILDDIR)/train_grpo
	./$(BUILDDIR)/train_grpo

tests: $(KERNEL_TESTS) $(MODEL_TESTS) test_loading_weights

# ── Data preparation ─────────────────────────────────────────────────────────

.PHONY: prepare_sft prepare_grpo

prepare_sft:
	$(PYTHON) python_scripts/prepare_data.py --mode sft --output data/sft_train.bin

prepare_grpo:
	$(PYTHON) python_scripts/prepare_data.py --mode grpo-text \
		--dataset trl-lib/DeepMath-103K --output data/deepmath-103k.jsonl

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -rf $(BUILDDIR)
