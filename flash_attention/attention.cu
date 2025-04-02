#include "attention.cuh"

namespace attention
{
    // namespace
    // {
    //     constexpr int BLOCK_SIZE = 512; // 线程块大小，需根据GPU架构调整
    // }

    /**
     * grid(batch_size, num_head)
     * block(Bc)
     * Q\K\V\O: [batch_size, num_head, N, d]
     * l\m: [batch_size, num_head, N, 1]
     */
    __global__ void flashAttentionMinimal(const float *Q, const float *K, const float *V, const int batch_size, const int num_head,
                                          const int N, const int d,
                                          const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                                          float *l, float *m, float *O)
    {
        int tx = threadIdx.x;
        int bx = blockIdx.x; // batch_id
        int by = blockIdx.y; // head_id

        // Offset into Q,K,V,O,l,m - different for each batch and head
        // a [N, d] mat processed by each block
        int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
        int lm_offset = (bx * gridDim.y * N) + (by * N);          // offset for l and m

        // Define SRAM for Q,K,V,S
        extern __shared__ float sram[];
        int tile_size = Bc * d; // size of Qi, Kj, Vj
        float *Qi = sram;
        float *Kj = &sram[tile_size];
        float *Vj = &sram[tile_size * 2];
        float *S = &sram[tile_size * 3]; // Bc * Br

        for (int j = 0; j < Tc; j++)
        {

            // Load Kj, Vj to SRAM
            for (int x = 0; x < d; x++)
            {
                Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
                Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
            }
            __syncthreads(); // such that the inner loop can use the correct Kj, Vj

            for (int i = 0; i < Tr; i++)
            {

                // Load Qi to SRAM, l and m to registers
                for (int x = 0; x < d; x++)
                {
                    Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
                }
                float row_m_prev = m[lm_offset + (Br * i) + tx];
                float row_l_prev = l[lm_offset + (Br * i) + tx];

                // S = QK^T, row_m = rowmax(S)
                float row_m = -INFINITY;
                for (int y = 0; y < Bc; y++)
                {
                    float sum = 0;
                    for (int x = 0; x < d; x++)
                    {
                        sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                    }
                    sum *= softmax_scale;
                    S[(Bc * tx) + y] = sum;

                    if (sum > row_m)
                        row_m = sum;
                }

                // P = exp(S - row_m), row_l = rowsum(P)
                float row_l = 0;
                for (int y = 0; y < Bc; y++)
                {
                    S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                    row_l += S[(Bc * tx) + y];
                }

                // Compute new m and l
                float row_m_new = max(row_m_prev, row_m);
                float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

                // Write O, l, m to HBM
                for (int x = 0; x < d; x++)
                {
                    float pv = 0; // Pij * Vj
                    for (int y = 0; y < Bc; y++)
                    {
                        pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                    }
                    O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) + (__expf(row_m - row_m_new) * pv));
                }
                m[lm_offset + (Br * i) + tx] = row_m_new;
                l[lm_offset + (Br * i) + tx] = row_l_new;
            }
            __syncthreads(); // otherwise, thread can use the wrong Kj, Vj in inner loop
        }
    }

    void launchFlashAttentionMinimal(const float *Q, const float *K, const float *V, const int batch_size, const int num_head,
                                     const int N, const int d, float *l, float *m, float *O, cudaStream_t stream)
    {
        constexpr int Bc = 2;
        constexpr int Br = 2;
        assert(N % Br == 0);
        assert(N % Bc == 0);
        const int Tr = N / Br;
        const int Tc = N / Bc;
        const float softmax_scale = 1.0f / sqrtf((float)d);

        const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
        int max_sram_size;
        cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
        printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

        dim3 grid_dim(batch_size, num_head); // batch_size x num_heads
        dim3 block_dim(Bc);                  // Bc threads per block

        flashAttentionMinimal<<<grid_dim, block_dim, sram_size, stream>>>(Q, K, V, batch_size, num_head, N, d, Tc, Tr, Bc, Br, softmax_scale, l, m, O);
    }

    template <typename T>
    struct MaxOp
    {
        __device__ __forceinline__ T operator()(const T &a, const T &b) { return max(a, b); }
    };

    template <typename T>
    struct SumOp
    {
        __device__ __forceinline__ T operator()(const T &a, const T &b) { return a + b; }
    };

    template <template <typename> class ReduceOp, typename T>
    __device__ __inline__ T warpAllReduce(T val)
    {
        auto functor = ReduceOp<T>();
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
        {
            val = functor(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
        }
        return val;
    }

    template <typename T>
    __device__ __inline__ T blockAllReduceSum(T val)
    {
        __shared__ T shared[32];
        __shared__ T ret;
        int warp_id = (threadIdx.x >> 5);
        int lane_id = (threadIdx.x & 31);

        val = warpAllReduce<SumOp, T>(val);
        if (lane_id == 0)
        {
            shared[warp_id] = val;
        }
        __syncthreads();

        val = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x] : (T)(0.0f);
        val = warpAllReduce<SumOp, T>(val);
        if (threadIdx.x == 0)
        {
            ret = val;
        }
        __syncthreads();

        return ret;
    }

    template <typename T>
    __device__ __inline__ T blockAllReduceMax(T val)
    {
        __shared__ T shared[32];
        __shared__ T ret;
        int warp_id = (threadIdx.x >> 5);
        int lane_id = (threadIdx.x & 31);

        val = warpAllReduce<MaxOp, T>(val);
        if (lane_id == 0)
        {
            shared[warp_id] = val;
        }
        __syncthreads();

        val = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x] : (T)(-FLT_MAX);
        val = warpAllReduce<MaxOp, T>(val);
        if (threadIdx.x == 0)
        {
            ret = val;
        }
        __syncthreads();

        return ret;
    }

    struct __align__(8) MD_F
    {
        float m; // max val
        float d; // exp sum
    };

    struct MDFOp
    {
        __device__ __forceinline__ MD_F operator()(MD_F &a, MD_F &b)
        {
            MD_F ret;
            ret.m = max(a.m, b.m);
            ret.d = a.d * __expf(a.m - ret.m) + b.d * __expf(b.m - ret.m);
            return ret;
        }
    };

    /**
     * grid( num_head, batch_size )
     * block( BLOCK_SIZE )
     * Q\O: [batch_size, num_head, N, d]
     * K\V: [batch_size, num_head, M, d]
     * l: [batch_size, num_head, N, 1]
     * m: [batch_size, num_head, N, 1]
     */
    template <int Bc>
    __global__ void flashAttentionKernel_v1(const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
                                            float *__restrict__ O, float *__restrict__ l, float *__restrict__ m,
                                            const int N, const int M, const int d, const float softmax_scale)
    {
        const int qo_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N * d;
        const int kv_offset = (blockIdx.y * gridDim.x + blockIdx.x) * M * d;
        const int lm_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N;

        extern __shared__ float s_ptr[];
        float *s_Q = s_ptr;        // [1, d]
        float *s_K = s_Q + d;      // [Bc, d]
        float *s_V = s_K + Bc * d; // [Bc, d]
        float *s_S = s_V + Bc * d; // [1, Bc]

        __shared__ MD_F row_ml_prev;

        // 对 K|V 在 M 维度分组，每组长度为 Bc，共分为 Tc 组
        for (int i = 0; i < M; i += Bc)
        {
            // 加载 [Bc, d] 数据到 s_K 和 s_Vs_V
            for (int j = threadIdx.x; j < Bc * d; j += blockDim.x)
            {
                s_K[j] = K[kv_offset + i * d + j];
                s_V[j] = V[kv_offset + i * d + j];
            }
            __syncthreads();

            // 遍历 Q 的 N 列，每次处理一列
            for (int j = 0; j < N; ++j)
            {
                // 加载 1 列数据到 s_Q
                for (int k = threadIdx.x; k < d; k += blockDim.x)
                {
                    s_Q[k] = Q[qo_offset + j * d + k];
                }
                // 上一个 Bc 组结束时每行的 m 和 l
                if (threadIdx.x == 0)
                {
                    row_ml_prev = {m[lm_offset + j], l[lm_offset + j]};
                }
                __syncthreads();

                // 存储当前第 j 行的 l 和 m
                MD_F row_ml = {-1e20f, 0.0f};
                // 遍历 K^T 的 Bc 列
                for (int k = 0; k < Bc; ++k)
                {
                    MD_F tmp_ml = {0.0f, 1.0f};
                    // 计算 QK^T
                    for (int x = threadIdx.x; x < d; x += blockDim.x)
                    {
                        tmp_ml.m += s_Q[x] * s_K[k * d + x];
                    }
                    tmp_ml.m *= softmax_scale;
                    __syncthreads();

                    // 存储第 j 行的 Q 向量与第 k 列的 s_K 向量的内积, QK^T 矩阵当前第 j 列的值
                    s_S[k] = blockAllReduceSum<float>(tmp_ml.m);
                    tmp_ml.m = s_S[k];
                    row_ml = MDFOp()(row_ml, tmp_ml);
                }
                __syncthreads();

                MD_F row_ml_new = MDFOp()(row_ml_prev, row_ml);

                // 遍历矩阵 O 的 d 维度，O = softmax(QK^T)V
                for (int k = threadIdx.x; k < d; k += blockDim.x)
                {
                    float pv = 0.0f;
                    for (int x = 0; x < Bc; ++x)
                    {
                        pv += __expf(s_S[x] - row_ml.m) * s_V[x * d + k];
                    }
                    // 更新 O 矩阵
                    O[qo_offset + j * d + k] = row_ml_prev.d / row_ml_new.d * __expf(row_ml_prev.m - row_ml_new.m) * O[qo_offset + j * d + k] + __expf(row_ml.m - row_ml_new.m) * pv;
                }

                // 写入当前 Bc 组的 l 和 m
                if (threadIdx.x == 0)
                {
                    l[lm_offset + j] = row_ml_new.d;
                    m[lm_offset + j] = row_ml_new.m;
                }
                __syncthreads();
            }
            __syncthreads();
        }
    }

    void launchFlashAttentionKernel_v1(const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
                                       float *__restrict__ O, float *__restrict__ l, float *__restrict__ m,
                                       const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream)
    {
        constexpr int Bc = 4;
        assert(M % Bc == 0);
        const float softmax_scale = 1.0f / sqrtf((float)d);

        const int sram_size = (d + 2 * Bc * d + Bc) * sizeof(float);
        int max_sram_size;
        cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
        printf("Max shared memory: %g KB, requested shared memory: %g KB \n", max_sram_size / 1024.0f, sram_size / 1024.0f);

        constexpr int block_size = 128;
        dim3 grid_dim(num_head, batch_size);
        dim3 block_dim(block_size);
        flashAttentionKernel_v1<Bc><<<grid_dim, block_dim, sram_size, stream>>>(Q, K, V, O, l, m, N, M, d, softmax_scale);
    }

    /**
     * grid( num_head, batch_size )
     * block( BLOCK_SIZE )
     * Q\O: [batch_size, num_head, N, d]
     * K\V: [batch_size, num_head, M, d]
     * l: [batch_size, num_head, N, 1]
     * m: [batch_size, num_head, N, 1]
     */
    template <int Bc, int Br>
    __global__ void flashAttentionKernel_v2(const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
                                            float *__restrict__ O, float *__restrict__ l, float *__restrict__ m,
                                            const int N, const int M, const int d, const float softmax_scale)
    {
        const int qo_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N * d;
        const int kv_offset = (blockIdx.y * gridDim.x + blockIdx.x) * M * d;
        const int lm_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N;

        extern __shared__ float s_ptr[];
        float *s_Q = s_ptr;        // [Br, d]
        float *s_K = s_Q + Br * d; // [Bc, d]
        float *s_V = s_K + Bc * d; // [Bc, d]
        float *s_S = s_V + Bc * d; // [Br, Bc]

        __shared__ MD_F row_ml_prev[Br];

        const int warp_id = threadIdx.x / 32;
        const int lane_id = threadIdx.x & 31;

        // 对 K|V 在 M 维度分组，每组长度为 Bc，共分为 Tc 组
        for (int i = 0; i < M; i += Bc)
        {
            // 加载 [Bc, d] 数据到 s_K 和 s_Vs_V
            for (int j = threadIdx.x; j < Bc * d; j += blockDim.x)
            {
                s_K[j] = K[kv_offset + i * d + j];
                s_V[j] = V[kv_offset + i * d + j];
            }
            __syncthreads();

            // 遍历 Q 的 N 列，每次处理一列
            for (int j = 0; j < N; j += Br)
            {
                // 加载 Br 行数据到 s_Q
                for (int k = threadIdx.x; k < Br * d; k += blockDim.x)
                {
                    s_Q[k] = Q[qo_offset + j * d + k];
                }
                // 上一个 Bc 组结束时每行的 m 和 l
                if (threadIdx.x < Br)
                {
                    row_ml_prev[threadIdx.x] = {m[lm_offset + j + threadIdx.x], l[lm_offset + j + threadIdx.x]};
                }
                __syncthreads();

                // 存储当前 warp 对应的第 j+warp_id 行的 l 和 m
                MD_F row_ml = {-1e20f, 0.0f};
                // 遍历 K^T 的 Bc 列
                #pragma unroll
                for (int k = 0; k < Bc; ++k)
                {
                    MD_F tmp_ml = {0.0f, 1.0f};
                    // 计算 QK^T
                    for (int x = lane_id; x < d; x += 32)
                    {
                        tmp_ml.m += s_Q[warp_id * d + x] * s_K[k * d + x];
                    }
                    tmp_ml.m *= softmax_scale;
                    __syncwarp();

                    // 存储第 j 行的 Q 向量与第 k 列的 s_K 向量的内积, QK^T 矩阵当前第 j 列的值
                    s_S[warp_id * Bc + k] = warpAllReduce<SumOp, float>(tmp_ml.m);
                    tmp_ml.m = s_S[warp_id * Bc + k];
                    row_ml = MDFOp()(row_ml, tmp_ml);
                }
                __syncthreads();

                MD_F row_ml_new = MDFOp()(row_ml_prev[warp_id], row_ml);

                // 遍历矩阵 O 的 d 维度，O = softmax(QK^T)V
                for (int k = lane_id; k < d; k += 32)
                {
                    float pv = 0.0f;
                    #pragma unroll
                    for (int x = 0; x < Bc; ++x)
                    {
                        pv += __expf(s_S[warp_id * Bc + x] - row_ml.m) * s_V[x * d + k];
                    }
                    // 更新 O 矩阵
                    O[qo_offset + (j + warp_id) * d + k] = row_ml_prev[warp_id].d / row_ml_new.d * __expf(row_ml_prev[warp_id].m - row_ml_new.m) * O[qo_offset + (j + warp_id) * d + k] + __expf(row_ml.m - row_ml_new.m) * pv;
                }

                // 写入当前 Bc 组的 l 和 m
                if (lane_id == 0)
                {
                    l[lm_offset + j + warp_id] = row_ml_new.d;
                    m[lm_offset + j + warp_id] = row_ml_new.m;
                }
                __syncthreads();
            }
            __syncthreads();
        }
    }

    void launchFlashAttentionKernel_v2(const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
                                       float *__restrict__ O, float *__restrict__ l, float *__restrict__ m,
                                       const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream)
    {
        constexpr int Bc = 2;
        constexpr int Br = 4;
        assert(M % Bc == 0);
        const float softmax_scale = 1.0f / sqrtf((float)d);

        const int sram_size = (Br * d + 2 * Bc * d + Br * Bc) * sizeof(float);
        int max_sram_size;
        cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
        printf("Max shared memory: %g KB, requested shared memory: %g KB \n", max_sram_size / 1024.0f, sram_size / 1024.0f);

        dim3 grid_dim(num_head, batch_size);
        dim3 block_dim(Br * 32);
        flashAttentionKernel_v2<Bc, Br><<<grid_dim, block_dim, sram_size, stream>>>(Q, K, V, O, l, m, N, M, d, softmax_scale);
    }

    __global__ void softmaxKernel(const float *__restrict__ mat, float *__restrict__ output, const int ncol, const float softmax_scale)
    {
        float val;
        float vmax = -FLT_MAX;
        float exp_sum = 1e-10f;

#pragma unroll
        for (int i = threadIdx.x; i < ncol; i += blockDim.x)
        {
            vmax = max(mat[blockIdx.x * ncol + i], vmax);
        }
        __syncthreads();

        vmax = blockAllReduceMax<float>(vmax);

#pragma unroll
        for (int i = threadIdx.x; i < ncol; i += blockDim.x)
        {
            exp_sum += __expf((mat[blockIdx.x * ncol + i] - vmax) * softmax_scale);
        }
        __syncthreads();

        exp_sum = blockAllReduceSum<float>(exp_sum);

#pragma unroll
        for (int i = threadIdx.x; i < ncol; i += blockDim.x)
        {
            val = __expf((mat[blockIdx.x * ncol + i] - vmax) * softmax_scale) / exp_sum;
            output[blockIdx.x * ncol + i] = val;
        }
    }

    void launchSoftmaxKernel(const float *__restrict__ mat, float *__restrict__ output, const int ncol, const int nrow, 
        const float softmax_scale, cudaStream_t stream)
    {
        constexpr int block_size = 256;
        dim3 block(block_size);
        dim3 grid(nrow);
        softmaxKernel<<<grid, block, 0, stream>>>(mat, output, ncol, softmax_scale);
    }

    void launchAttentionBaseline(const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
                                 float *__restrict__ QK, float *__restrict__ QK_softmax, float *__restrict__ O,
                                 const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream)
    {
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

} // namespace attention
