#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <assert.h>
#include <cfloat>
#include <cublas_v2.h>
#include "utils.h"

namespace attention
{

    void launchFlashAttentionMinimal(const float *Q, const float *K, const float *V, const int batch_size, const int num_head,
                                     const int N, const int d, float *l, float *m, float *O, cudaStream_t stream = 0);

    void launchFlashAttentionKernel_v1(const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
                                       float *__restrict__ O, float *__restrict__ l, float *__restrict__ m,
                                       const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);

    void launchFlashAttentionKernel_v2(const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
                                       float *__restrict__ O, float *__restrict__ l, float *__restrict__ m,
                                       const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);
    void launchAttentionBaseline(const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
                                 float *__restrict__ QK, float *__restrict__ QK_softmax, float *__restrict__ O,
                                 const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);

} // namespace attention
