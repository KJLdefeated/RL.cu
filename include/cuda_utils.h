#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t e = (call);                                            \
        if (e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                    cudaGetErrorString(e), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t s = (call);                                         \
        if (s != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n",                 \
                    (int)s, __FILE__, __LINE__);                           \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)
#endif
