#include "attention.cuh"
#include "utils.h"
#include "common.cuh"

#include <cublas_v2.h>


static __global__ void softmaxKernel(const float* __restrict__ mat, float* __restrict__ output, const int ncol, const float softmax_scale) {
    float val;
    float vmax = -FLT_MAX;
    float exp_sum = 1e-10f;

#pragma unroll
    for (int i = threadIdx.x; i < ncol; i += blockDim.x) {
        vmax = max(mat[blockIdx.x * ncol + i], vmax);
    }
    __syncthreads();

    vmax = blockAllReduceMax<float>(vmax);

#pragma unroll
    for (int i = threadIdx.x; i < ncol; i += blockDim.x) {
        exp_sum += __expf((mat[blockIdx.x * ncol + i] - vmax) * softmax_scale);
    }
    __syncthreads();

    exp_sum = blockAllReduceSum<float>(exp_sum);

#pragma unroll
    for (int i = threadIdx.x; i < ncol; i += blockDim.x) {
        val = __expf((mat[blockIdx.x * ncol + i] - vmax) * softmax_scale) / exp_sum;
        output[blockIdx.x * ncol + i] = val;
    }
}

void launchSoftmaxKernel(const float* __restrict__ mat, float* __restrict__ output, const int ncol, const int nrow,
    const float softmax_scale, cudaStream_t stream) {
    constexpr int block_size = 256;
    dim3 block(block_size);
    dim3 grid(nrow);
    softmaxKernel << <grid, block, 0, stream >> > (mat, output, ncol, softmax_scale);
}



void launchAttentionBaseline(const float* Q, const float* K, const float* V,
    float* QK, float* QK_softmax, float* O,
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream) {
    const float softmax_scale = 1.0f / sqrtf((float)d);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    CHECK_CUBLAS_STATUS(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, d,
        &alpha,
        K, d, M * d,
        Q, d, N * d,
        &beta,
        QK, M, N * M,
        batch_size * num_head));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    launchSoftmaxKernel(QK, QK_softmax, M, batch_size * num_head * N, softmax_scale, stream);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUBLAS_STATUS(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        d, N, M,
        &alpha,
        V, d, M * d,
        QK_softmax, M, N * M,
        &beta,
        O, d, N * d,
        batch_size * num_head));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
