cmake_minimum_required(VERSION 3.18)

# Auto-detect CUDA toolkit if not in PATH
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDA_HOME})
    # Common install locations, newest first
    foreach(_ver 12.8 12.6 12.4 12.2 12.0)
        if(EXISTS "/usr/local/cuda-${_ver}/bin/nvcc")
            set(CMAKE_CUDA_COMPILER "/usr/local/cuda-${_ver}/bin/nvcc")
            break()
        endif()
    endforeach()
endif()

project(rl_cu LANGUAGES CXX CUDA)

# ── CUDA & C++ standards ──────────────────────────────────────────────────────
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── GPU architecture ──────────────────────────────────────────────────────────
# Pass via: cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120
# Or via Makefile: make ARCH=120
# CMake 3.24+ supports "native" for auto-detect.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        message(FATAL_ERROR
            "CMake < 3.24 cannot auto-detect GPU arch. "
            "Please specify: cmake -DCMAKE_CUDA_ARCHITECTURES=<arch> "
            "(e.g. 80 for A100, 89 for RTX 4090, 90 for H100, 120 for RTX 5090)")
    endif()
endif()

# ── Find dependencies ─────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── Global compile options ────────────────────────────────────────────────────
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
add_compile_definitions(CUTLASS_ENABLE_TENSOR_CORE_MMA=1)

# ── Include paths ─────────────────────────────────────────────────────────────
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/cutlass/include
)

# ── Kernel library (static) ──────────────────────────────────────────────────
# All kernel .cu files compiled once, linked into every target that needs them.
add_library(kernels STATIC
    src/kernels/rmsnorm.cu
    src/kernels/rope.cu
    src/kernels/attention.cu
    src/kernels/swiglu.cu
    src/kernels/embedding.cu
    src/kernels/linear.cu
    src/kernels/softmax.cu
    src/kernels/sampler.cu
    src/kernels/adamw.cu
    src/kernels/fused_norm_linear.cu
    src/kernels/config.cpp
    src/kernels/weights.cpp
)
target_link_libraries(kernels PUBLIC CUDA::cublas)

# ── Model library (static) ───────────────────────────────────────────────────
add_library(model STATIC
    src/model/qwen3.cu
    src/model/kv_cache.cu
)
target_link_libraries(model PUBLIC kernels)

# ── Helper: add a test/binary that links against kernels only ─────────────────
function(add_kernel_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE kernels)
endfunction()

# ── Helper: add a test/binary that links against model + kernels ──────────────
function(add_model_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE model)
endfunction()

# ── Kernel-only tests ─────────────────────────────────────────────────────────
add_kernel_test(test_rmsnorm           tests/test_rmsnorm.cu)
add_kernel_test(test_softmax           tests/test_softmax.cu)
add_kernel_test(test_swiglu            tests/test_swiglu.cu)
add_kernel_test(test_attention         tests/test_attention.cu)
add_kernel_test(test_rope              tests/test_rope.cu)
add_kernel_test(test_embedding         tests/test_embedding.cu)
add_kernel_test(test_linear            tests/test_linear.cu)
add_kernel_test(test_sampler           tests/test_sampler.cu)
add_kernel_test(test_fused_norm_linear tests/test_fused_norm_linear.cu)
add_kernel_test(test_linear_backward   tests/test_linear_backward.cu)
add_kernel_test(test_embedding_backward tests/test_embedding_backward.cu)
add_kernel_test(test_rmsnorm_backward  tests/test_rmsnorm_backward.cu)
add_kernel_test(test_swiglu_backward   tests/test_swiglu_backward.cu)
add_kernel_test(test_rope_backward     tests/test_rope_backward.cu)
add_kernel_test(test_attention_backward tests/test_attention_backward.cu)

# ── Model-level tests ─────────────────────────────────────────────────────────
# Note: test_qwen3, test_qwen3_forward, bench_decode have stale APIs
# and need updating before they can compile again.
# add_model_test(test_qwen3              tests/test_qwen3.cu)
# add_model_test(test_qwen3_forward      tests/test_qwen3_forward.cu)
# add_model_test(bench_decode            tests/bench_decode.cu)
add_model_test(test_qwen3_backward     tests/test_qwen3_backward.cu)
add_model_test(test_fwd_bwd            tests/test_fwd_bwd.cu)
add_model_test(test_adamw              tests/test_adamw.cu)
add_model_test(test_kv_cache           tests/test_kv_cache.cu)
add_model_test(test_llmengine          tests/test_llmengine.cu)

# ── C++-only tests (no CUDA runtime needed) ──────────────────────────────────
add_executable(test_dataloader tests/test_dataloader.cpp)
target_include_directories(test_dataloader PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_lr_scheduler tests/test_lr_scheduler.cpp)
target_include_directories(test_lr_scheduler PRIVATE ${CMAKE_SOURCE_DIR}/include)

# test_loading_weights is C++ but includes cuda_fp16.h and calls cudaGetErrorString
add_executable(test_loading_weights
    tests/test_loading_weights.cpp
    src/kernels/config.cpp
    src/kernels/weights.cpp
)
target_include_directories(test_loading_weights PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(test_loading_weights PRIVATE CUDA::cudart)

# ── Training binaries ─────────────────────────────────────────────────────────
add_model_test(train_sft               tests/train_sft.cu)
add_model_test(train_grpo              tests/train_grpo.cu)
