#include "utils.h"
#include "attention.cuh"

#include <assert.h>
#include <cstdio>
// #include <string>

void printMatrix(const float *mat, char *s, int height, int width,
                 int end_row, int end_col, int start_row = 0, int start_col = 0)
{
    assert(start_row >= 0 && start_col >= 0 && end_row <= height && end_col <= width);
    printf("\nmatrix %s: width=%d, height=%d, start_row=%d, end_row=%d, start_col=%d, end_col=%d\n",
           s, width, height, start_row, end_row, start_col, end_col);
    for (int i = start_row; i < end_row; i++)
    {
        for (int j = start_col; j < end_col; j++)
        {
            printf("%g\t", mat[i * width + j]);
        }
        printf("\n");
    }
}

void printVec(const float *vec, char *s, int length, int end_id, int start_id = 0)
{
    assert(start_id >= 0 && end_id <= length);
    printf("\nvec %s: length=%d, start_id=%d, end_id=%d\n", s, length, start_id, end_id);
    for (int i = start_id; i < end_id; i++)
    {
        printf("%g\t", vec[i]);
    }
    printf("\n");
}

void timingAttn(const float *Q, const float *K, const float *V, const int batch_size, const int num_head,
                const int N, const int M, const int d, float *l, float *m, float *O)
{
    constexpr int REPEAT_NUM = 1;
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < REPEAT_NUM; ++i)
    {
        attention::launchFlashAttentionKernel_v4(Q, K, V, O, batch_size, num_head, N, M, d);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("alogrithm: flash attention v4 bz(%d) nh(%d) N(%d) M(%d) d(%d), elapsed_time: %g ms\n",
           batch_size, num_head, N, M, d, elapsed_time / REPEAT_NUM);
}
int main(int argc, char *argv[])
{
    constexpr int batch_size = 32;
    constexpr int num_head = 8;
    constexpr int N = 1024;
    constexpr int M = 1024;
    constexpr int d = 128;

    float *Q = new float[batch_size * num_head * N * d];
    float *K = new float[batch_size * num_head * M * d];
    float *V = new float[batch_size * num_head * M * d];
    float *l = new float[batch_size * num_head * N];
    float *m = new float[batch_size * num_head * N];
    float *O = new float[batch_size * num_head * N * d];

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
    //     Q[i] = i % 1003 - 500.0f;
    //     O[i] = 0.0f;
    // }

    // for (int i = 0; i < batch_size * num_head * M * d; ++i)
    // {
    //     K[i] = i % 2157 - 1218.1f;
    //     V[i] = i % 191 - 100.9f;
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

    for (int i = 0; i < batch_size * num_head * N; ++i)
    {
        l[i] = 0.0f;
        m[i] = -1e20f;
    }

    printMatrix(Q, (char *)("Matrix Q: "), N, d, 32, 32, 28, 24);
    printMatrix(K, (char *)("Matrix K: "), M, d, 32, 32, 28, 24);
    printMatrix(V, (char *)("Matrix V: "), M, d, 32, 32, 28, 24);

    float *d_Q;
    float *d_K;
    float *d_V;
    float *d_l;
    float *d_m;
    float *d_O;
    size_t mem_size = sizeof(float) * (batch_size * num_head * (N + M) * d * 2 + batch_size * num_head * N * 2);
    printf("requested global memory: %g GB \n", mem_size / 1024.0f / 1024.0f / 1024.0f);

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_Q, mem_size));
    d_K = d_Q + batch_size * num_head * N * d;
    d_V = d_K + batch_size * num_head * M * d;
    d_l = d_V + batch_size * num_head * M * d;
    d_m = d_l + batch_size * num_head * N;
    d_O = d_m + batch_size * num_head * N;

    CHECK_CUDA_ERROR(cudaMemcpy(d_Q, Q, sizeof(float) * batch_size * num_head * N * d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_K, K, sizeof(float) * batch_size * num_head * M * d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, V, sizeof(float) * batch_size * num_head * M * d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_l, l, sizeof(float) * batch_size * num_head * N, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_m, m, sizeof(float) * batch_size * num_head * N, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_O, O, sizeof(float) * batch_size * num_head * N * d, cudaMemcpyHostToDevice));

    timingAttn(d_Q, d_K, d_V, batch_size, num_head, N, M, d, d_l, d_m, d_O);

    CHECK_CUDA_ERROR(cudaMemcpy(O, d_O, sizeof(float) * batch_size * num_head * N * d, cudaMemcpyDeviceToHost));
    // CHECK_CUDA_ERROR(cudaMemcpy(l, d_l, sizeof(float) * batch_size * num_head * N, cudaMemcpyDeviceToHost));
    // CHECK_CUDA_ERROR(cudaMemcpy(m, d_m, sizeof(float) * batch_size * num_head * N, cudaMemcpyDeviceToHost));

    printMatrix(O, (char *)("Matrix output: "), N, d, 32, 32, 28, 24);
    // printVec(l, (char *)("Vec l: "), N, 64, 48);
    // printVec(m, (char *)("Vec m: "), N, 64, 48);

    CHECK_CUDA_ERROR(cudaFree(d_Q));
    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] l;
    delete[] m;
    delete[] O;

    return 0;
}
