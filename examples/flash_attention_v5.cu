#include "utils.h"
#include "attention.cuh"

#include <assert.h>
#include <cstdio>
// #include <string>

template <typename T>
void printMatrix(const T *mat, char *s, int height, int width,
                 int end_row, int end_col, int start_row = 0, int start_col = 0)
{
    assert(start_row >= 0 && start_col >= 0 && end_row <= height && end_col <= width);
    printf("\nmatrix %s: width=%d, height=%d, start_row=%d, end_row=%d, start_col=%d, end_col=%d\n",
           s, width, height, start_row, end_row, start_col, end_col);
    for (int i = start_row; i < end_row; i++)
    {
        for (int j = start_col; j < end_col; j++)
        {
            printf("%g\t", static_cast<float>(mat[i * width + j]));
        }
        printf("\n");
    }
}

template <typename T>
void printVec(const T *vec, char *s, int length, int end_id, int start_id = 0)
{
    assert(start_id >= 0 && end_id <= length);
    printf("\nvec %s: length=%d, start_id=%d, end_id=%d\n", s, length, start_id, end_id);
    for (int i = start_id; i < end_id; i++)
    {
        printf("%g\t", static_cast<float>(vec[i]));
    }
    printf("\n");
}

template <typename QKVType, typename OType>
void timingAttn(const QKVType *Q, const QKVType *K, const QKVType *V, const int batch_size, const int num_head,
                const int N, const int M, const int d, OType *O)
{
    constexpr int REPEAT_NUM = 1;
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < REPEAT_NUM; ++i)
    {
        attention::launchFlashAttentionKernel_v5(Q, K, V, O, batch_size, num_head, N, M, d);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("alogrithm: flash attention v5 bz(%d) nh(%d) N(%d) M(%d) d(%d), elapsed_time: %g ms\n",
           batch_size, num_head, N, M, d, elapsed_time / REPEAT_NUM);
}
int main(int argc, char *argv[])
{
    using QKVType = half;
    using OType = half;

    constexpr int batch_size = 32;
    constexpr int num_head = 8;
    constexpr int N = 1024;
    constexpr int M = 1024;
    constexpr int d = 128;

    QKVType *Q = new QKVType[batch_size * num_head * N * d];
    QKVType *K = new QKVType[batch_size * num_head * M * d];
    QKVType *V = new QKVType[batch_size * num_head * M * d];
    OType *O = new OType[batch_size * num_head * N * d];

    // srand(1024);
    // for (int i = 0; i < batch_size * num_head * N * d; ++i)
    // {
    //     Q[i] = (rand() / (RAND_MAX + 1.0f)) * 1.0f - 0.5f;
    //     O[i] = 0.0f;
    // }

    // for (int i = 0; i < batch_size * num_head * M * d; ++i)
    // {
    //     K[i] = (rand() / (RAND_MAX + 1.0f)) * 1.0f - 0.5f;
    //     V[i] = (rand() / (RAND_MAX + 1.0f)) * 1.0f - 0.5f;
    // }

    // for (int i = 0; i < batch_size * num_head * N * d; ++i)
    // {
    //     Q[i] = static_cast<QKVType>(i);
    //     O[i] = 0.0f;
    // }

    // for (int i = 0; i < batch_size * num_head * M * d; ++i)
    // {
    //     K[i] = static_cast<QKVType>(i);
    //     V[i] = static_cast<QKVType>(i);
    // }

    // 初始化Q矩阵
    for (size_t i = 0; i < batch_size * num_head * N * d; ++i)
    {
        // Q[i] = ((i % 200) - 100) * 0.1f; // 方案一
        Q[i] = static_cast<float>(static_cast<int>(i * 41 % 2001) * 0.01f - 10.0f); // 方案二
        O[i] = 0.0f;
    }

    // 初始化K矩阵（使用不同周期）
    for (size_t i = 0; i < batch_size * num_head * M * d; ++i)
    {
        K[i] = static_cast<float>((static_cast<int>(i % 211) - 105) * 0.095f);         // 211是质数
        V[i] = static_cast<float>(static_cast<int>(i * 53 % 1999) * 0.01f - 10.0f); // 503是质数
    }

    printMatrix(Q, (char *)("Matrix Q: "), N, d, 32, 32, 28, 24);
    printMatrix(K, (char *)("Matrix K: "), M, d, 32, 32, 28, 24);
    printMatrix(V, (char *)("Matrix V: "), M, d, 32, 32, 28, 24);
    // printMatrix(Q, (char *)("Matrix Q: "), N, d, 4, 16, 0, 0);
    // printMatrix(K, (char *)("Matrix K: "), M, d, 16, 16, 0, 0);
    // printMatrix(V, (char *)("Matrix V: "), M, d, 4, 16, 0, 0);

    QKVType *d_Q;
    QKVType *d_K;
    QKVType *d_V;
    OType *d_O;
    size_t mem_size = sizeof(QKVType) * batch_size * num_head * (N + 2 * M) * d + sizeof(OType) * batch_size * num_head * N * d;
    printf("requested global memory: %g GB \n", mem_size / 1024.0f / 1024.0f / 1024.0f);

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_Q, mem_size));
    d_K = d_Q + batch_size * num_head * N * d;
    d_V = d_K + batch_size * num_head * M * d;
    d_O = d_V + batch_size * num_head * M * d;

    CHECK_CUDA_ERROR(cudaMemcpy(d_Q, Q, sizeof(QKVType) * batch_size * num_head * N * d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_K, K, sizeof(QKVType) * batch_size * num_head * M * d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, V, sizeof(QKVType) * batch_size * num_head * M * d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_O, O, sizeof(OType) * batch_size * num_head * N * d, cudaMemcpyHostToDevice));

    timingAttn(d_Q, d_K, d_V, batch_size, num_head, N, M, d, d_O);

    CHECK_CUDA_ERROR(cudaMemcpy(O, d_O, sizeof(OType) * batch_size * num_head * N * d, cudaMemcpyDeviceToHost));

    printMatrix(O, (char *)("Matrix output: "), N, d, 32, 32, 28, 24);

    CHECK_CUDA_ERROR(cudaFree(d_Q));
    delete[] Q;
    delete[] K;
    delete[] V;

    return 0;
}
