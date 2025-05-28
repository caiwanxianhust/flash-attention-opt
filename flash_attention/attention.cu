#include "attention.cuh"

#include <cuda_fp16.h>
#include <assert.h>
#include <cfloat>
#include <cublas_v2.h>
#include "utils.h"
#include <mma.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace attention {
    // namespace
    // {
    //     constexpr int BLOCK_SIZE = 512; // 线程块大小，需根据GPU架构调整
    // }

    namespace cg = cooperative_groups;

    /**
     * grid(batch_size, num_head)
     * block(Bc)
     * Q\K\V\O: [batch_size, num_head, N, d]
     * l\m: [batch_size, num_head, N, 1]
     */
    __global__ void flashAttentionMinimal(const float* Q, const float* K, const float* V, const int batch_size, const int num_head,
        const int N, const int d,
        const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
        float* l, float* m, float* O) {
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
        float* Qi = sram;
        float* Kj = &sram[tile_size];
        float* Vj = &sram[tile_size * 2];
        float* S = &sram[tile_size * 3]; // Bc * Br

        for (int j = 0; j < Tc; j++) {

            // Load Kj, Vj to SRAM
            for (int x = 0; x < d; x++) {
                Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
                Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
            }
            __syncthreads(); // such that the inner loop can use the correct Kj, Vj

            for (int i = 0; i < Tr; i++) {

                // Load Qi to SRAM, l and m to registers
                for (int x = 0; x < d; x++) {
                    Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
                }
                float row_m_prev = m[lm_offset + (Br * i) + tx];
                float row_l_prev = l[lm_offset + (Br * i) + tx];

                // S = QK^T, row_m = rowmax(S)
                float row_m = -INFINITY;
                for (int y = 0; y < Bc; y++) {
                    float sum = 0;
                    for (int x = 0; x < d; x++) {
                        sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                    }
                    sum *= softmax_scale;
                    S[(Bc * tx) + y] = sum;

                    if (sum > row_m)
                        row_m = sum;
                }

                // P = exp(S - row_m), row_l = rowsum(P)
                float row_l = 0;
                for (int y = 0; y < Bc; y++) {
                    S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                    row_l += S[(Bc * tx) + y];
                }

                // Compute new m and l
                float row_m_new = max(row_m_prev, row_m);
                float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

                // Write O, l, m to HBM
                for (int x = 0; x < d; x++) {
                    float pv = 0; // Pij * Vj
                    for (int y = 0; y < Bc; y++) {
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

    void launchFlashAttentionMinimal(const float* Q, const float* K, const float* V, const int batch_size, const int num_head,
        const int N, const int d, float* l, float* m, float* O, cudaStream_t stream) {
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

        flashAttentionMinimal << <grid_dim, block_dim, sram_size, stream >> > (Q, K, V, batch_size, num_head, N, d, Tc, Tr, Bc, Br, softmax_scale, l, m, O);
    }

    struct __align__(8) MD_F {
        float m; // max val
        float d; // exp sum
    };

    struct MDFOp {
        __device__ __forceinline__ MD_F operator()(MD_F& a, MD_F& b) {
            MD_F ret;
            ret.m = max(a.m, b.m);
            ret.d = a.d * __expf(a.m - ret.m) + b.d * __expf(b.m - ret.m);
            return ret;
        }
    };

    template <typename T>
    struct MaxOp {
        __device__ __forceinline__ T operator()(const T& a, const T& b) { return max(a, b); }
    };

    template <typename T>
    struct SumOp {
        __device__ __forceinline__ T operator()(const T& a, const T& b) { return a + b; }
    };

    template <template <typename> class ReduceOp, typename T>
    __device__ __inline__ T warpAllReduce(T val) {
        auto functor = ReduceOp<T>();
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            val = functor(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
        }
        return val;
    }

    __device__ __inline__ MD_F warpAllReduce(MD_F val) {
        float tmp_m;
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            tmp_m = max(val.m, __shfl_xor_sync(0xffffffff, val.m, mask, 32));
            val.d = val.d * __expf(val.m - tmp_m) + __shfl_xor_sync(0xffffffff, val.d, mask, 32) * __expf(__shfl_xor_sync(0xffffffff, val.m, mask, 32) - tmp_m);
            val.m = tmp_m;
        }
        return val;
    }

    template <typename T>
    __device__ __inline__ T blockAllReduceSum(T val) {
        __shared__ T shared[32];
        __shared__ T ret;
        int warp_id = (threadIdx.x >> 5);
        int lane_id = (threadIdx.x & 31);

        val = warpAllReduce<SumOp, T>(val);
        if (lane_id == 0) {
            shared[warp_id] = val;
        }
        __syncthreads();

        val = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x] : (T)(0.0f);
        val = warpAllReduce<SumOp, T>(val);
        if (threadIdx.x == 0) {
            ret = val;
        }
        __syncthreads();

        return ret;
    }

    template <typename T>
    __device__ __inline__ T blockAllReduceMax(T val) {
        __shared__ T shared[32];
        __shared__ T ret;
        int warp_id = (threadIdx.x >> 5);
        int lane_id = (threadIdx.x & 31);

        val = warpAllReduce<MaxOp, T>(val);
        if (lane_id == 0) {
            shared[warp_id] = val;
        }
        __syncthreads();

        val = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x] : (T)(-FLT_MAX);
        val = warpAllReduce<MaxOp, T>(val);
        if (threadIdx.x == 0) {
            ret = val;
        }
        __syncthreads();

        return ret;
    }

    /**
     * grid( num_head, batch_size )
     * block( BLOCK_SIZE )
     * Q\O: [batch_size, num_head, N, d]
     * K\V: [batch_size, num_head, M, d]
     * l: [batch_size, num_head, N, 1]
     * m: [batch_size, num_head, N, 1]
     */
    template <int Bc>
    __global__ void flashAttentionKernel_v1(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
        float* __restrict__ O, float* __restrict__ l, float* __restrict__ m,
        const int N, const int M, const int d, const float softmax_scale) {
        const int qo_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N * d;
        const int kv_offset = (blockIdx.y * gridDim.x + blockIdx.x) * M * d;
        const int lm_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N;

        extern __shared__ float s_ptr[];
        float* s_Q = s_ptr;        // [1, d]
        float* s_K = s_Q + d;      // [Bc, d]
        float* s_V = s_K + Bc * d; // [Bc, d]
        float* s_S = s_V + Bc * d; // [1, Bc]

        __shared__ MD_F row_ml_prev;

        // 对 K|V 在 M 维度分组，每组长度为 Bc，共分为 Tc 组
        for (int i = 0; i < M; i += Bc) {
            // 加载 [Bc, d] 数据到 s_K 和 s_Vs_V
            for (int j = threadIdx.x; j < Bc * d; j += blockDim.x) {
                s_K[j] = K[kv_offset + i * d + j];
                s_V[j] = V[kv_offset + i * d + j];
            }
            __syncthreads();

            // 遍历 Q 的 N 列，每次处理一列
            for (int j = 0; j < N; ++j) {
                // 加载 1 列数据到 s_Q
                for (int k = threadIdx.x; k < d; k += blockDim.x) {
                    s_Q[k] = Q[qo_offset + j * d + k];
                }
                // 上一个 Bc 组结束时每行的 m 和 l
                if (threadIdx.x == 0) {
                    row_ml_prev = { m[lm_offset + j], l[lm_offset + j] };
                }
                __syncthreads();

                // 存储当前第 j 行的 l 和 m
                MD_F row_ml = { -1e20f, 0.0f };
                // 遍历 K^T 的 Bc 列
                for (int k = 0; k < Bc; ++k) {
                    MD_F tmp_ml = { 0.0f, 1.0f };
                    // 计算 QK^T
                    for (int x = threadIdx.x; x < d; x += blockDim.x) {
                        tmp_ml.m += s_Q[x] * s_K[k * d + x];
                    }
                    tmp_ml.m *= softmax_scale;
                    __syncthreads();

                    // 存储第 j 行的 Q 向量与第 k 列的 s_K 向量的内积, QK^T 矩阵当前第 j 列的值
                    tmp_ml.m = blockAllReduceSum<float>(tmp_ml.m);
                    row_ml = MDFOp()(row_ml, tmp_ml);
                    if (threadIdx.x == 0) {
                        s_S[k] = tmp_ml.m;
                    }
                    __syncthreads();
                }
                // __syncthreads();

                MD_F row_ml_new = MDFOp()(row_ml_prev, row_ml);

                // 遍历矩阵 O 的 d 维度，O = softmax(QK^T)V
                for (int k = threadIdx.x; k < d; k += blockDim.x) {
                    float pv = 0.0f;
                    for (int x = 0; x < Bc; ++x) {
                        pv += __expf(s_S[x] - row_ml.m) * s_V[x * d + k];
                    }
                    // 更新 O 矩阵
                    O[qo_offset + j * d + k] = 1.0f / row_ml_new.d * (row_ml_prev.d * __expf(row_ml_prev.m - row_ml_new.m) * O[qo_offset + j * d + k] + __expf(row_ml.m - row_ml_new.m) * pv);
                }

                // 写入当前 Bc 组的 l 和 m
                if (threadIdx.x == 0) {
                    l[lm_offset + j] = row_ml_new.d;
                    m[lm_offset + j] = row_ml_new.m;
                }
                __syncthreads();
            }
            __syncthreads();
        }
    }

    void launchFlashAttentionKernel_v1(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
        float* __restrict__ O, float* __restrict__ l, float* __restrict__ m,
        const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream) {
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
        flashAttentionKernel_v1<Bc> << <grid_dim, block_dim, sram_size, stream >> > (Q, K, V, O, l, m, N, M, d, softmax_scale);
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
    __global__ void flashAttentionKernel_v2(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
        float* __restrict__ O, float* __restrict__ l, float* __restrict__ m,
        const int N, const int M, const int d, const float softmax_scale) {
        const int qo_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N * d;
        const int kv_offset = (blockIdx.y * gridDim.x + blockIdx.x) * M * d;
        const int lm_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N;

        extern __shared__ float s_ptr[];
        float* s_Q = s_ptr;        // [Br, d]
        float* s_K = s_Q + Br * d; // [Bc, d]
        float* s_V = s_K + Bc * d; // [Bc, d]
        float* s_S = s_V + Bc * d; // [Br, Bc]

        __shared__ MD_F row_ml_prev[Br];

        const int warp_id = threadIdx.x / 32;
        const int lane_id = threadIdx.x & 31;

        // 对 K|V 在 M 维度分组，每组长度为 Bc，共分为 Tc 组
        for (int i = 0; i < M; i += Bc) {
            // 加载 [Bc, d] 数据到 s_K 和 s_Vs_V
            for (int j = threadIdx.x; j < Bc * d; j += blockDim.x) {
                s_K[j] = K[kv_offset + i * d + j];
                s_V[j] = V[kv_offset + i * d + j];
            }
            __syncthreads();

            // 遍历 Q 的 N 列，每次处理一列
            for (int j = 0; j < N; j += Br) {
                // 加载 Br 行数据到 s_Q
                for (int k = threadIdx.x; k < Br * d; k += blockDim.x) {
                    s_Q[k] = Q[qo_offset + j * d + k];
                }
                // 上一个 Bc 组结束时每行的 m 和 l
                if (threadIdx.x < Br) {
                    row_ml_prev[threadIdx.x] = { m[lm_offset + j + threadIdx.x], l[lm_offset + j + threadIdx.x] };
                }
                __syncthreads();

                // 存储当前 warp 对应的第 j+warp_id 行的 l 和 m
                MD_F row_ml = { -1e20f, 0.0f };
                // 遍历 K^T 的 Bc 列
#pragma unroll
                for (int k = 0; k < Bc; ++k) {
                    MD_F tmp_ml = { 0.0f, 1.0f };
                    // 计算 QK^T
                    for (int x = lane_id; x < d; x += 32) {
                        tmp_ml.m += s_Q[warp_id * d + x] * s_K[k * d + x];
                    }
                    tmp_ml.m *= softmax_scale;
                    __syncwarp();

                    // 存储第 j 行的 Q 向量与第 k 列的 s_K 向量的内积, QK^T 矩阵当前第 j 列的值
                    tmp_ml.m = warpAllReduce<SumOp, float>(tmp_ml.m);
                    if (lane_id == 0) {
                        s_S[warp_id * Bc + k] = tmp_ml.m;
                    }
                    row_ml = MDFOp()(row_ml, tmp_ml);
                }
                __syncthreads();

                MD_F row_ml_new = MDFOp()(row_ml_prev[warp_id], row_ml);

                // 遍历矩阵 O 的 d 维度，O = softmax(QK^T)V
                for (int k = lane_id; k < d; k += 32) {
                    float pv = 0.0f;
#pragma unroll
                    for (int x = 0; x < Bc; ++x) {
                        pv += __expf(s_S[warp_id * Bc + x] - row_ml.m) * s_V[x * d + k];
                    }
                    // 更新 O 矩阵
                    O[qo_offset + (j + warp_id) * d + k] = 1.0f / row_ml_new.d * (row_ml_prev[warp_id].d * __expf(row_ml_prev[warp_id].m - row_ml_new.m) * O[qo_offset + (j + warp_id) * d + k] + __expf(row_ml.m - row_ml_new.m) * pv);
                }

                // 写入当前 Bc 组的 l 和 m
                if (lane_id == 0) {
                    l[lm_offset + j + warp_id] = row_ml_new.d;
                    m[lm_offset + j + warp_id] = row_ml_new.m;
                }
                __syncthreads();
            }
            // __syncthreads();
        }
    }

    void launchFlashAttentionKernel_v2(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
        float* __restrict__ O, float* __restrict__ l, float* __restrict__ m,
        const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream) {
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
        flashAttentionKernel_v2<Bc, Br> << <grid_dim, block_dim, sram_size, stream >> > (Q, K, V, O, l, m, N, M, d, softmax_scale);
    }

    template <int Br, int Bc, int Bd>
    __device__ void loadQKFromGmemAndConvertToHalf(const float* Q, const float* K, const int d,
        half* s_Q, half* s_K, const int offset_q, const int offset_kv) {
        int row_a, col_a;
        float4 tmp4;

#pragma unroll
        for (int i = (threadIdx.x << 2); i < Br * Bd; i += (blockDim.x << 2)) {
            row_a = i / Bd;
            col_a = i % Bd;
            tmp4 = reinterpret_cast<const float4*>(Q + offset_q + row_a * d + col_a)[0];
            s_Q[row_a * Bd + col_a] = __float2half(tmp4.x);
            s_Q[row_a * Bd + col_a + 1] = __float2half(tmp4.y);
            s_Q[row_a * Bd + col_a + 2] = __float2half(tmp4.z);
            s_Q[row_a * Bd + col_a + 3] = __float2half(tmp4.w);
        }

#pragma unroll
        for (int i = (threadIdx.x << 2); i < Bc * Bd; i += (blockDim.x << 2)) {
            row_a = i / Bd;
            col_a = i % Bd;
            tmp4 = reinterpret_cast<const float4*>(K + offset_kv + row_a * d + col_a)[0];
            s_K[row_a * Bd + col_a] = __float2half(tmp4.x);
            s_K[row_a * Bd + col_a + 1] = __float2half(tmp4.y);
            s_K[row_a * Bd + col_a + 2] = __float2half(tmp4.z);
            s_K[row_a * Bd + col_a + 3] = __float2half(tmp4.w);
        }
    }

    template <int Bc, int Bd>
    __device__ void loadVFromGmemAndConvertToHalf(const float* V, const int d, half* s_V, const int offset_kv) {
        int row_a, col_a;
        float4 tmp4;
#pragma unroll
        for (int i = (threadIdx.x << 2); i < Bc * Bd; i += (blockDim.x << 2)) {
            row_a = i / Bd;
            col_a = i % Bd;
            tmp4 = reinterpret_cast<const float4*>(V + offset_kv + row_a * d + col_a)[0];
            s_V[row_a * Bd + col_a] = __float2half(tmp4.x);
            s_V[row_a * Bd + col_a + 1] = __float2half(tmp4.y);
            s_V[row_a * Bd + col_a + 2] = __float2half(tmp4.z);
            s_V[row_a * Bd + col_a + 3] = __float2half(tmp4.w);
        }
    }

    template <int Bc, int Bd>
    __device__ void loadVFromGmem(const float* V, const int d, float* s_V, const int offset_kv) {
        int row_a, col_a;
        float4 tmp4;
#pragma unroll
        for (int i = (threadIdx.x << 2); i < Bc * Bd; i += (blockDim.x << 2)) {
            row_a = i / Bd;
            col_a = i % Bd;
            tmp4 = reinterpret_cast<const float4*>(V + offset_kv + row_a * d + col_a)[0];
            reinterpret_cast<float4*>(s_V + row_a * Bd + col_a)[0] = tmp4;
        }
    }

    template <int Bd, int Wc, int Wr, typename T1, typename T2, typename T3>
    __device__ void gemmFromSmemByWMMA(const half* __restrict__ s_Q, const half* __restrict__ s_K,
        T1* q_frag, T2* k_frag, T3* acc_frag, const int warp_row, const int warp_col,
        const int WMITERS, const int WNITERS, const int WKITERS) {
        using namespace nvcuda;

#pragma unroll
        for (int wmidx = 0; wmidx < WMITERS; ++wmidx) {
#pragma unroll
            for (int wkidx = 0; wkidx < WKITERS; ++wkidx) {
                int shm_offset = warp_row * Wr * Bd + wmidx * 16 * Bd + wkidx * 16;
                wmma::load_matrix_sync(q_frag[wmidx * WKITERS + wkidx], s_Q + shm_offset, Bd);
            }
        }

#pragma unroll
        for (int wnidx = 0; wnidx < WNITERS; ++wnidx) {
#pragma unroll
            for (int wkidx = 0; wkidx < WKITERS; ++wkidx) {
                int shm_offset = warp_col * Wc * Bd + wnidx * 16 * Bd + wkidx * 16;
                wmma::load_matrix_sync(k_frag[wnidx * WNITERS + wkidx], s_K + shm_offset, Bd);
            }
        }

#pragma unroll
        for (int wmidx = 0; wmidx < WMITERS; ++wmidx) {
#pragma unroll
            for (int wnidx = 0; wnidx < WNITERS; ++wnidx) {
#pragma unroll
                for (int wkidx = 0; wkidx < WKITERS; ++wkidx) {
                    wmma::mma_sync(acc_frag[wmidx * WNITERS + wnidx], q_frag[wmidx * WKITERS + wkidx],
                        k_frag[wnidx * WKITERS + wkidx], acc_frag[wmidx * WNITERS + wnidx]);
                }
            }
        }
    }

    template <int Bd, int Wc, int Wr, typename T1, typename T2, typename T3>
    __device__ void pvGemmFromSmemByWMMA(const half* __restrict__ s_V,
        T1* p_frag, T2* v_frag, T3* c_frag, const int warp_row, const int warp_col,
        const int WMITERS, const int WNITERS, const int WKITERS) {
        using namespace nvcuda;
#pragma unroll
        for (int wnidx = 0; wnidx < WNITERS; ++wnidx) {
#pragma unroll
            for (int wkidx = 0; wkidx < WKITERS; ++wkidx) {
                int shm_offset = warp_col * Wc + wnidx * 16 + wkidx * 16 * Bd;
                wmma::load_matrix_sync(v_frag[wnidx * WNITERS + wkidx], s_V + shm_offset, Bd);
            }
        }

#pragma unroll
        for (int wmidx = 0; wmidx < WMITERS; ++wmidx) {
#pragma unroll
            for (int wnidx = 0; wnidx < WNITERS; ++wnidx) {
#pragma unroll
                for (int wkidx = 0; wkidx < WKITERS; ++wkidx) {
                    wmma::mma_sync(c_frag[wmidx * WNITERS + wnidx], p_frag[wmidx * WKITERS + wkidx],
                        v_frag[wnidx * WKITERS + wkidx], c_frag[wmidx * WNITERS + wnidx]);
                }
            }
        }
    }

    template <int Bc, int Wc, int Wr, typename T>
    __device__ void loadSFromSmemToReg(const half* __restrict__ s_S, T* a_frag, const int warp_row, const int warp_col,
        const int WMITERS, const int WNITERS, const int WKITERS) {
        using namespace nvcuda;
#pragma unroll
        for (int wmidx = 0; wmidx < WMITERS; ++wmidx) {
#pragma unroll
            for (int wkidx = 0; wkidx < WKITERS; ++wkidx) {
                int shm_offset = warp_row * Wr * Bc + wmidx * 16 * Bc + wkidx * 16;
                wmma::load_matrix_sync(a_frag[wmidx * WKITERS + wkidx], s_S + shm_offset, Bc);
            }
        }
    }

    template <int Bc, int Wc, int Wr, typename T>
    __device__ void StoreQKGEMMToSmem(float* __restrict__ s_S, T* acc_frag, const int warp_row, const int warp_col,
        const int WMITERS, const int WNITERS, const int WKITERS, const float softmax_scale) {
        using namespace nvcuda;
        // 从 s_S 中取出元素，累加矩阵计算结果，再写入 s_S
#pragma unroll
        for (int wmidx = 0; wmidx < WMITERS; ++wmidx) {
#pragma unroll
            for (int wnidx = 0; wnidx < WNITERS; ++wnidx) {
                int shm_offset = warp_row * Wr * Bc + warp_col * Wc + wmidx * 16 * Bc + wnidx * 16;
#pragma unroll
                for (int idx = 0; idx < acc_frag[wmidx * WNITERS + wnidx].num_elements; ++idx) {
                    acc_frag[wmidx * WNITERS + wnidx].x[idx] *= softmax_scale;
                }
                wmma::store_matrix_sync(s_S + shm_offset, acc_frag[wmidx * WNITERS + wnidx], Bc, wmma::mem_row_major);
            }
        }
    }

    template <int Bd, int Wc, int Wr, typename T>
    __device__ void StoreOGEMMToSmem(float* __restrict__ s_Q, T* acc_frag, const int warp_row, const int warp_col,
        const int WMITERS, const int WNITERS, const int WKITERS) {
        using namespace nvcuda;
#pragma unroll
        for (int wmidx = 0; wmidx < WMITERS; ++wmidx) {
#pragma unroll
            for (int wnidx = 0; wnidx < WNITERS; ++wnidx) {
                int shm_offset = warp_row * Wr * Bd + warp_col * Wc + wmidx * 16 * Bd + wnidx * 16;
                wmma::store_matrix_sync(s_Q + shm_offset, acc_frag[wmidx * WNITERS + wnidx], Bd, wmma::mem_row_major);
            }
        }
    }

    /**
     * grid( num_head, batch_size )
     * block( BLOCK_SIZE )
     * Q\O: [batch_size, num_head, N, d]
     * K\V: [batch_size, num_head, M, d]
     * l: [batch_size, num_head, N, 1]
     * m: [batch_size, num_head, N, 1]
     */
    template <int Bc, int Br, int Wc, int Wr>
    __global__ void flashAttentionKernel_v3(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
        float* __restrict__ O, float* __restrict__ l, float* __restrict__ m,
        const int N, const int M, const int d, const float softmax_scale) {
        using namespace nvcuda;

        // 当前矩阵的偏移量
        const int qo_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N * d;
        const int kv_offset = (blockIdx.y * gridDim.x + blockIdx.x) * M * d;
        const int lm_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N;

        // 让 Bd 等于 Bc 从而使得 QK 矩阵分片[Br, Bc] 与 QKV 矩阵分片[Br, Bd] 形状相同，方便排布
        constexpr int Bd = Bc;

        __shared__ half s_Q_half[Br * Bd];
        __shared__ half s_K_half[Bc * Bd];
        __shared__ half s_V_half[Bc * Bd];
        __shared__ float s_S[Br * Bc];
        __shared__ half s_S_half[Br * Bc];
        __shared__ float s_O[Br * Bd];

        // 前一个 Bc 组的 l 和 m
        __shared__ MD_F row_ml_prev[Br];
        __shared__ MD_F row_ml[Br];
        __shared__ MD_F row_ml_new[Br];

        // block 内 warp 二维分布的 id
        int warp_row = (threadIdx.x >> 5) / (Bc / Wc);
        int warp_col = (threadIdx.x >> 5) % (Bc / Wc);
        int warp_id = (threadIdx.x >> 5);
        int lane_id = (threadIdx.x & 31);

        // 单个 warp 处理层面 M、N、K 方向每个 warp 迭代次数
        constexpr int WMITERS = Wr / 16;
        constexpr int WNITERS = Wc / 16;
        constexpr int WKITERS = Bd / 16;

        using FragAType = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
        using FragBType = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>;
        using FragCFloatType = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
        using FragVType = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
        // 当前 warp 内的矩阵乘法片段
        FragAType a_frag[WMITERS * WKITERS];        // 用于存储矩阵 Q 和 QK 的分片
        FragBType b_frag[WNITERS * WKITERS];        // 用于存储矩阵 K 的分片
        FragCFloatType acc_frag[WMITERS * WNITERS]; // 用于存储矩阵 QK 的分片
        FragVType v_frag[WNITERS * WKITERS];        // 用于存储矩阵 V 的分片

        // 对 K|V 在 M 维度分组，每组长度为 Bc，共分为 Tc 组
        for (int i = 0; i < M; i += Bc) {
            // 对 Q 在 N 维度分组，每组长度为 Br，共分为 Tr 组
            for (int j = 0; j < N; j += Br) {
#pragma unroll
                for (int k = threadIdx.x; k < Br; k += blockDim.x) {
                    // 上一个 Bc 组结束时每行的 m 和 l
                    row_ml_prev[k] = { m[lm_offset + j + k], l[lm_offset + j + k] };
                    row_ml[k] = { -1e20f, 0.0f };
                }
                __syncthreads();
#pragma unroll
                for (int k = 0; k < WMITERS * WNITERS; ++k) {
                    wmma::fill_fragment(acc_frag[k], 0.0f);
                }
                // 计算 QK 矩阵
                for (int k = 0; k < d; k += Bd) {
                    loadQKFromGmemAndConvertToHalf<Br, Bc, Bd>(Q, K, d, s_Q_half, s_K_half, qo_offset + j * d + k, kv_offset + i * d + k);
                    __syncthreads();

                    gemmFromSmemByWMMA<Bd, Wc, Wr, FragAType, FragBType, FragCFloatType>(s_Q_half, s_K_half, a_frag, b_frag, acc_frag,
                        warp_row, warp_col, WMITERS, WNITERS, WKITERS);
                    __syncthreads();
                }
                StoreQKGEMMToSmem<Bc, Wc, Wr, FragCFloatType>(s_S, acc_frag, warp_row, warp_col, WMITERS, WNITERS, WKITERS, softmax_scale);
                __syncthreads();

                // 对 s_S[Br, Bc] 求 softmax，每个 warp 计算一行
                // MD_F row_ml_tmp = {-1e20f, 0.0f};
#pragma unroll
                for (int s = warp_id; s < Br; s += (blockDim.x >> 5)) {
                    MD_F row_ml_tmp = { -1e20f, 0.0f };
                    for (int k = lane_id; k < Bc; k += 32) {
                        MD_F tmp_ml = { s_S[s * Bc + k], 1.0f };
                        row_ml_tmp = MDFOp()(row_ml_tmp, tmp_ml);
                    }
                    __syncwarp();

                    // 得到 s_S[Br, Bc] 每一行的 m 和 l
                    row_ml_tmp = warpAllReduce(row_ml_tmp);
                    if (lane_id == 0) {
                        row_ml[s] = row_ml_tmp;
                        row_ml_new[s] = MDFOp()(row_ml_prev[s], row_ml_tmp);
                    }

                    // 更新 s_S[Br, Bc]
                    for (int k = lane_id; k < Bc; k += 32) {
                        s_S_half[s * Bc + k] = __float2half(__expf(s_S[s * Bc + k] - row_ml_tmp.m));
                    }
                }
                __syncthreads();

                loadSFromSmemToReg<Bc, Wc, Wr, FragAType>(s_S_half, a_frag, warp_row, warp_col, WMITERS, WNITERS, WKITERS);

                // 计算 s_S[Br, Bc] * s_V[Bc, Bd]
                for (int k = 0; k < d; k += Bd) {
                    for (int s = 0; s < WMITERS * WNITERS; ++s) {
                        wmma::fill_fragment(acc_frag[s], 0.0f);
                    }
                    loadVFromGmemAndConvertToHalf<Bc, Bd>(V, d, s_V_half, kv_offset + i * d + k);

                    __syncthreads();
                    pvGemmFromSmemByWMMA<Bd, Wc, Wr, FragAType, FragVType, FragCFloatType>(s_V_half,
                        a_frag, v_frag, acc_frag, warp_row, warp_col, WMITERS, WNITERS, WKITERS);
                    StoreOGEMMToSmem<Bd, Wc, Wr, FragCFloatType>(s_O, acc_frag, warp_row, warp_col, WMITERS, WNITERS, WKITERS);

                    __syncthreads();

                    for (int s = warp_id; s < Br; s += (blockDim.x >> 5)) {
                        for (int t = lane_id; t < Bd; t += 32) {
                            // 更新 O 矩阵
                            O[qo_offset + (j + s) * d + k + t] = 1.0f / row_ml_new[s].d * (row_ml_prev[s].d * __expf(row_ml_prev[s].m - row_ml_new[s].m) * O[qo_offset + (j + s) * d + k + t] + __expf(row_ml[s].m - row_ml_new[s].m) * s_O[s * Bd + t]);
                        }
                    }
                }

                // 写入当前 Bc 组的 l 和 m
#pragma unroll
                for (int k = threadIdx.x; k < Br; k += blockDim.x) {
                    l[lm_offset + j + k] = row_ml_new[k].d;
                    m[lm_offset + j + k] = row_ml_new[k].m;
                }
                __syncthreads();
            }
        }
    }

    void launchFlashAttentionKernel_v3(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
        float* __restrict__ O, float* __restrict__ l, float* __restrict__ m,
        const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream) {
        printf("in fa v3 launchFlashAttentionKernel_v3\n");
        constexpr int Bc = 32;
        constexpr int Br = 64;
        constexpr int Wr = 32;
        constexpr int Wc = 16;
        constexpr int Bd = 32;
        // 让 Bd 等于 Bc 从而使得 QK 矩阵分片[Br, Bc] 与 QKV 矩阵分片[Br, Bd] 形状相同，方便排布
        assert(M % Bc == 0 && N % Br == 0 && d % Bc == 0);
        const float softmax_scale = 1.0f / sqrtf((float)d);

        /**
            __shared__ half s_Q_half[Br * Bd];
            __shared__ half s_K_half[Bc * Bd];
            __shared__ half s_V_half[Bc * Bd];
            __shared__ float s_S[Br * Bc];
            __shared__ half s_S_half[Br * Bc];
            __shared__ float s_O[Br * Bd];

            // 前一个 Bc 组的 l 和 m
            __shared__ MD_F row_ml_prev[Br];
            __shared__ MD_F row_ml[Br];
            __shared__ MD_F row_ml_new[Br];
         */

        const int sram_size = (Br * Bc + Br * Bd) * sizeof(float) + (Br * Bd + 2 * Bc * Bd + Br * Bc) * sizeof(half) + 8 * 3 * Br;
        int max_sram_size;
        cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
        printf("Max shared memory: %g KB, requested shared memory: %g KB \n", max_sram_size / 1024.0f, sram_size / 1024.0f);

        dim3 grid_dim(num_head, batch_size);
        dim3 block_dim(Bc * Br / (Wr * Wc) * 32);
        flashAttentionKernel_v3<Bc, Br, Wc, Wr> << <grid_dim, block_dim, 0, stream >> > (Q, K, V, O, l, m, N, M, d, softmax_scale);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    /**
     * grid( num_head, batch_size )
     * block( BLOCK_SIZE )
     * Q\O: [batch_size, num_head, N, d]
     * K\V: [batch_size, num_head, M, d]
     * l: [batch_size, num_head, N, 1]
     * m: [batch_size, num_head, N, 1]
     */
    template <int Bc, int Br, int Wc, int Wr>
    __global__ void flashAttentionKernel_v4(const float* __restrict__ Q,
        const float* __restrict__ K,
        const float* __restrict__ V,
        float* __restrict__ O,
        const int N, const int M, const int d, const float softmax_scale) {

        using namespace nvcuda;

        // 当前矩阵的偏移量
        const int qo_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N * d;
        const int kv_offset = (blockIdx.y * gridDim.x + blockIdx.x) * M * d;

        // 让 Bd 等于 Bc 从而使得 QK 矩阵分片[Br, Bc] 与 QKV 矩阵分片[Br, Bd] 形状相同，方便排布
        constexpr int Bd = Bc;

        __shared__ half s_Q_half[Br * Bd];
        __shared__ half s_K_half[Bc * Bd];
        __shared__ half s_V_half[Bc * Bd];
        __shared__ float s_S[Br * Bc];
        __shared__ half s_S_half[Br * Bc];
        __shared__ float s_O[Br * Bd];

        // 前一个 Bc 组的 l 和 m
        __shared__ MD_F row_ml_prev[Br];
        __shared__ MD_F row_ml[Br];
        __shared__ MD_F row_ml_new[Br];

        // block 内 warp 二维分布的 id
        int warp_row = (threadIdx.x >> 5) / (Bc / Wc);
        int warp_col = (threadIdx.x >> 5) % (Bc / Wc);
        int warp_id = (threadIdx.x >> 5);
        int lane_id = (threadIdx.x & 31);

        // 单个 warp 处理层面 M、N、K 方向每个 warp 迭代次数
        constexpr int WMITERS = Wr / 16;
        constexpr int WNITERS = Wc / 16;
        constexpr int WKITERS = Bd / 16;

        using FragAType = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
        using FragBType = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>;
        using FragCFloatType = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
        using FragVType = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
        // 当前 warp 内的矩阵乘法片段
        FragAType a_frag[WMITERS * WKITERS];        // 用于存储矩阵 Q 和 QK 的分片
        FragBType b_frag[WNITERS * WKITERS];        // 用于存储矩阵 K 的分片
        FragCFloatType acc_frag[WMITERS * WNITERS]; // 用于存储矩阵 QK 的分片
        FragVType v_frag[WNITERS * WKITERS];        // 用于存储矩阵 V 的分片

        // 对 Q 在 N 维度分组，每组长度为 Br，共分为 Tr 组
        for (int j = 0; j < N; j += Br) {
            // 初始化 row_ml_prev 和 row_ml
#pragma unroll
            for (int k = threadIdx.x; k < Br; k += blockDim.x) {
                row_ml_prev[k] = { -1e20f, 0.0f };
                row_ml[k] = { -1e20f, 0.0f };
            }
            __syncthreads();

            // 对 K|V 在 M 维度分组，每组长度为 Bc，共分为 Tc 组
            for (int i = 0; i < M; i += Bc) {
                // 每组计算 QK 矩阵前先初始化累加矩阵
#pragma unroll
                for (int k = 0; k < WMITERS * WNITERS; ++k) {
                    wmma::fill_fragment(acc_frag[k], 0.0f);
                }

                // 计算 QK 矩阵
                for (int k = 0; k < d; k += Bd) {
                    loadQKFromGmemAndConvertToHalf<Br, Bc, Bd>(Q, K, d, s_Q_half, s_K_half, qo_offset + j * d + k, kv_offset + i * d + k);
                    __syncthreads();

                    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && i == 0 && j == 0 && k == 0) {
                        printf("s_Q_half[0-7]: %f %f %f %f %f %f %f %f\n", __half2float(s_Q_half[0]), __half2float(s_Q_half[1]), 
                            __half2float(s_Q_half[2]), __half2float(s_Q_half[3]), __half2float(s_Q_half[4]), __half2float(s_Q_half[5]), 
                            __half2float(s_Q_half[6]), __half2float(s_Q_half[7]));

                        printf("s_K_half[0-7]: %f %f %f %f %f %f %f %f\n", __half2float(s_K_half[0]), __half2float(s_K_half[1]), 
                            __half2float(s_K_half[2]), __half2float(s_K_half[3]), __half2float(s_K_half[4]), __half2float(s_K_half[5]), 
                            __half2float(s_K_half[6]), __half2float(s_K_half[7]));
                    }

                    gemmFromSmemByWMMA<Bd, Wc, Wr, FragAType, FragBType, FragCFloatType>(s_Q_half, s_K_half, a_frag, b_frag, acc_frag,
                        warp_row, warp_col, WMITERS, WNITERS, WKITERS);
                    __syncthreads();
                }
                StoreQKGEMMToSmem<Bc, Wc, Wr, FragCFloatType>(s_S, acc_frag, warp_row, warp_col, WMITERS, WNITERS, WKITERS, softmax_scale);
                __syncthreads();

                #if 0
                if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && i == 0 && j == 0) {
                    printf("s_S[0-7]: %f %f %f %f %f %f %f %f\n", s_S[0], s_S[1], s_S[2], s_S[3], s_S[4], s_S[5], s_S[6], s_S[7]);
                }
                #endif

                // 对 s_S[Br, Bc] 求 softmax，每个 warp 计算一行
                // MD_F row_ml_tmp = {-1e20f, 0.0f};
#pragma unroll
                for (int s = warp_id; s < Br; s += (blockDim.x >> 5)) {
                    MD_F row_ml_tmp = { -1e20f, 0.0f };
#pragma unroll
                    for (int k = lane_id; k < Bc; k += 32) {
                        MD_F tmp_ml = { s_S[s * Bc + k], 1.0f };
                        row_ml_tmp = MDFOp()(row_ml_tmp, tmp_ml);
                    }
                    __syncwarp();

                    // 得到 s_S[Br, Bc] 每一行的 m 和 l
                    row_ml_tmp = warpAllReduce(row_ml_tmp);
                    if (lane_id == 0) {
                        row_ml[s] = row_ml_tmp;
                        row_ml_new[s] = MDFOp()(row_ml_prev[s], row_ml_tmp);
                    }

                    // 更新 s_S[Br, Bc]
#pragma unroll
                    for (int k = lane_id; k < Bc; k += 32) {
                        s_S_half[s * Bc + k] = __float2half(__expf(s_S[s * Bc + k] - row_ml_tmp.m));
                    }
                }
                __syncthreads();
                #if 0
                if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && i == 0 && j == 0) {
                    uint32_t print_row = 0;
                    printf("s_S[0-7]: %f %f %f %f %f %f %f %f\n", 
                        __half2float(s_S_half[print_row * Bc + 0]), __half2float(s_S_half[print_row * Bc + 1]), 
                        __half2float(s_S_half[print_row * Bc + 2]), __half2float(s_S_half[print_row * Bc + 3]), 
                        __half2float(s_S_half[print_row * Bc + 4]), __half2float(s_S_half[print_row * Bc + 5]), 
                        __half2float(s_S_half[print_row * Bc + 6]), __half2float(s_S_half[print_row * Bc + 7]));
                }
                #endif

                // 将更新好的 s_S 写入寄存器，这里复用 a_frag
                loadSFromSmemToReg<Bc, Wc, Wr, FragAType>(s_S_half, a_frag, warp_row, warp_col, WMITERS, WNITERS, WKITERS);

                // 计算 s_S[Br, Bc] * s_V[Bc, Bd]
                for (int k = 0; k < d; k += Bd) {
                    for (int s = 0; s < WMITERS * WNITERS; ++s) {
                        wmma::fill_fragment(acc_frag[s], 0.0f);
                    }
                    loadVFromGmemAndConvertToHalf<Bc, Bd>(V, d, s_V_half, kv_offset + i * d + k);
                    __syncthreads();

                    pvGemmFromSmemByWMMA<Bd, Wc, Wr, FragAType, FragVType, FragCFloatType>(s_V_half,
                        a_frag, v_frag, acc_frag, warp_row, warp_col, WMITERS, WNITERS, WKITERS);
                    StoreOGEMMToSmem<Bd, Wc, Wr, FragCFloatType>(s_O, acc_frag, warp_row, warp_col, WMITERS, WNITERS, WKITERS);
                    __syncthreads();

                    for (int s = warp_id; s < Br; s += (blockDim.x >> 5)) {
                        for (int t = lane_id; t < Bd; t += 32) {
                            // 更新 O 矩阵
                            O[qo_offset + (j + s) * d + k + t] = 
                            1.0f / row_ml_new[s].d * (row_ml_prev[s].d * __expf(row_ml_prev[s].m - row_ml_new[s].m) * O[qo_offset + (j + s) * d + k + t] + 
                            __expf(row_ml[s].m - row_ml_new[s].m) * s_O[s * Bd + t]);
                        }
                    }
                }

                if (threadIdx.x < Br) {
                    row_ml_prev[threadIdx.x] = row_ml_new[threadIdx.x];
                }
                __syncthreads();

            }
            #if 0
            if (threadIdx.x < 8 && blockIdx.x == 0 && blockIdx.y == 0 && j == 0)
            {
                printf("row=%d row_ml_prev: m(%f) d(%f)\n", threadIdx.x, row_ml_prev[threadIdx.x].m, row_ml_prev[threadIdx.x].d);
            }
            #endif
        }

    }

    void launchFlashAttentionKernel_v4(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
        float* __restrict__ O, 
        const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream) {
        printf("in fa v4 launchFlashAttentionKernel_v4\n");
        constexpr int Bc = 32;
        constexpr int Br = 64;
        constexpr int Wr = 32;
        constexpr int Wc = 16;
        constexpr int Bd = 32;
        // 让 Bd 等于 Bc 从而使得 QK 矩阵分片[Br, Bc] 与 QKV 矩阵分片[Br, Bd] 形状相同，方便排布
        assert(M % Bc == 0 && N % Br == 0 && d % Bc == 0);
        const float softmax_scale = 1.0f / sqrtf((float)d);

        /**
        __shared__ half s_Q_half[Br * Bd];
        __shared__ half s_K_half[Bc * Bd];
        __shared__ half s_V_half[Bc * Bd];
        __shared__ float s_S[Br * Bc];
        __shared__ half s_S_half[Br * Bc];
        __shared__ float s_O[Br * Bd];

        // 前一个 Bc 组的 l 和 m
        __shared__ MD_F row_ml_prev[Br];
        __shared__ MD_F row_ml[Br];
        __shared__ MD_F row_ml_new[Br];
        */

        const int sram_size = (Br * Bc + Br * Bd) * sizeof(float) + (Br * Bd + 2 * Bc * Bd + Br * Bc) * sizeof(half) + 8 * 3 * Br;
        int max_sram_size;
        cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
        printf("Max shared memory: %g KB, requested shared memory: %g KB \n", max_sram_size / 1024.0f, sram_size / 1024.0f);

        dim3 grid_dim(num_head, batch_size);
        dim3 block_dim(Bc * Br / (Wr * Wc) * 32);
        flashAttentionKernel_v4<Bc, Br, Wc, Wr> << <grid_dim, block_dim, 0, stream >> > (Q, K, V, O, N, M, d, softmax_scale);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }


    template <int GroupSize = 16>
    __device__ __forceinline__ MD_F threadGroupAllReduce(MD_F val) {
        float tmp_m;
#pragma unroll
        for (int mask = (GroupSize / 2); mask > 0; mask >>= 1) {
            tmp_m = max(val.m, __shfl_xor_sync(0xffffffff, val.m, mask, GroupSize));
            val.d = val.d * __expf(val.m - tmp_m) + __shfl_xor_sync(0xffffffff, val.d, mask, GroupSize) * __expf(__shfl_xor_sync(0xffffffff, val.m, mask, GroupSize) - tmp_m);
            val.m = tmp_m;
        }
        return val;
    }


    __host__ __device__ __forceinline__ int div_ceil(int a, int b) {
        return (a + b - 1) / b;
    }

#define LDMATRIX_X4(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "l"(__cvta_generic_to_shared(addr)))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "l"(__cvta_generic_to_shared(addr)))
#define MMA_M16N8K16_F16F16F16F16(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" : "=r"(RD0), "=r"(RD1) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])


    /** 每个 block 包含 4 个 warp，每个 warp 单独处理 [Br, d] 的 Q 矩阵分片，4 个 warp 共用 [Br, d] 的 K、V分片
     * grid(div_ceil(N, 4 * Br), num_head, batch_size )
     * block( 128 )
     * Q\O: [batch_size, num_head, N, d]
     * K\V: [batch_size, num_head, M, d]
     */
    template <int Br, int Bc, int Bd>
    __global__ void flashAttentionKernel_v5(const half* __restrict__ Q,
        const half* __restrict__ K,
        const half* __restrict__ V,
        half* __restrict__ O,
        const int N, const int M, const int d, const float softmax_scale) {
        // 划分 warp
        const uint32_t warp_id = threadIdx.x >> 5;
        const uint32_t lane_id = threadIdx.x & 0x1f;

        // 当前 warp 处理的 Q、O 矩阵偏移量
        const uint32_t qo_offset = (blockIdx.z * gridDim.y + blockIdx.y) * N * d + blockIdx.x * 4 * Br * d + warp_id * Br * d;
        const uint32_t kv_offset = (blockIdx.z * gridDim.y + blockIdx.y) * M * d;

        // 共享内存
        extern __shared__ half smem_ptr[];
        half* s_Q = reinterpret_cast<half*>(smem_ptr);
        half* s_K = s_Q + 4 * Br * d;
        half* s_V = s_K + Bc * d;
        half* s_QK = s_V + Bc * d;
        half* s_S = s_QK + 4 * Br * Bc;
        half* s_O = s_S + 4 * Br * Bc;
        MD_F* row_ml_prev = reinterpret_cast<MD_F*>(s_O + 4 * Br * Bd);
        MD_F* row_ml = row_ml_prev + 4 * Br;
        MD_F* row_ml_new = row_ml + 4 * Br;

        // 初始化 ml
        #pragma unroll
        for (int k = lane_id; k < Br; k += 32) {
            row_ml_prev[warp_id * Br + k] = { -1e20f, 0.0f };
            row_ml[warp_id * Br + k] = { -1e20f, 0.0f };
        }

        // load [4 * Br, d] 的 Q 矩阵分片到 s_Q，每个 warp load [Br, d]，每次 load 8 个 half
        for (int i = (lane_id << 3); i < Br * d; i += (32 << 3)) {
            reinterpret_cast<float4 *>(s_Q + warp_id * Br * d + i)[0] = reinterpret_cast<const float4 *>(Q + qo_offset + i)[0];
        }
        __syncwarp();

        #if 0
        if (lane_id == 0 && warp_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
        {
            printf("s_Q[0-15]: %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", __half2float(s_Q[0]), __half2float(s_Q[1]), 
            __half2float(s_Q[2]), __half2float(s_Q[3]), __half2float(s_Q[4]), __half2float(s_Q[5]), __half2float(s_Q[6]), 
            __half2float(s_Q[7]), __half2float(s_Q[8]), __half2float(s_Q[9]), __half2float(s_Q[10]), __half2float(s_Q[11]),
            __half2float(s_Q[12]), __half2float(s_Q[13]), __half2float(s_Q[14]), __half2float(s_Q[15]));
        }
        #endif

        // warp 矩阵乘法的尺寸为 16x16x16，调用两次 mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 指令
        // 所以 3 个矩阵都需要 4 个寄存器
        uint32_t RA[4];
        uint32_t RB[4];

        // 对 K|V 在 M 维度分组，每组长度为 Bc，共分为 Tc 组
        for (int i = 0; i < M; i += Bc) {
            // 初始化矩阵 C 的寄存器
            uint32_t RC[4] = {0, 0, 0, 0};

            // load [Bc, d] 的 K 矩阵分片到 s_K，整个 block 一起 load [Br, d]，每次 load 8 个 half
            for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
                reinterpret_cast<float4 *>(s_K + j)[0] = reinterpret_cast<const float4 *>(K + kv_offset + i * d + j)[0];
                reinterpret_cast<float4 *>(s_V + j)[0] = reinterpret_cast<const float4 *>(V + kv_offset + i * d + j)[0];
            }
            __syncthreads();

            #if 0
            if (lane_id == 0 && warp_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i == 0)
            {
                uint32_t print_row = 2;
                printf("s_K[%d][0-15]: %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", print_row,
                    __half2float(s_K[print_row * d + 0]), __half2float(s_K[print_row * d + 1]), 
                    __half2float(s_K[print_row * d + 2]), __half2float(s_K[print_row * d + 3]), 
                    __half2float(s_K[print_row * d + 4]), __half2float(s_K[print_row * d + 5]), 
                    __half2float(s_K[print_row * d + 6]), __half2float(s_K[print_row * d + 7]), 
                    __half2float(s_K[print_row * d + 8]), __half2float(s_K[print_row * d + 9]), 
                    __half2float(s_K[print_row * d + 10]), __half2float(s_K[print_row * d + 11]),
                    __half2float(s_K[print_row * d + 12]), __half2float(s_K[print_row * d + 13]), 
                    __half2float(s_K[print_row * d + 14]), __half2float(s_K[print_row * d + 15]));
            }
            #endif

            // 计算 QK 矩阵，每次计算尺寸为 16x16x16，
            #pragma unroll
            for (int k = 0; k < d; k += Bd) {
                // 从 s_Q load 16x16 矩阵分片到 RA，使用 ldmatrix.sync.aligned.x4.m8n8.shared.b16 指令
                // warp 内每个线程都需要传入一个地址
                uint32_t saddr = warp_id * Br * d + k + (lane_id % 16) * d + (lane_id / 16) * 8;
                LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_Q + saddr);
                #if 0
                if (lane_id < 12 && warp_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i == 0 && k == 0) {
                    printf("lane_id = %d ldaddr = %d **** RA[0] = %g %g, RA[1] = %g %g, RA[2] = %g %g, RA[3] = %g %g\n",
                        lane_id, saddr,
                        __half2float(reinterpret_cast<half*>(&(RA[0]))[0]),
                        __half2float(reinterpret_cast<half*>(&(RA[0]))[1]),
                        __half2float(reinterpret_cast<half*>(&(RA[1]))[0]),
                        __half2float(reinterpret_cast<half*>(&(RA[1]))[1]),
                        __half2float(reinterpret_cast<half*>(&(RA[2]))[0]),
                        __half2float(reinterpret_cast<half*>(&(RA[2]))[1]),
                        __half2float(reinterpret_cast<half*>(&(RA[3]))[0]),
                        __half2float(reinterpret_cast<half*>(&(RA[3]))[1])
                    );
                }
                #endif

                // 从 s_K（列主序） load 16x16 矩阵分片到 RB，使用 ldmatrix.sync.aligned.x4.m8n8.shared.b16 指令
                // warp 内线程 0-7 加载第一个 8x8 矩阵，线程  8-15 加载第二个 8x8 矩阵，线程 16-23 加载第三个 8x8 矩阵， 线程 24-31 加载第四个 8x8 矩阵
                // 此时可以认为 4 个子矩阵是行主序排布的，子矩阵内部元素列主序排布
                // 子矩阵偏移量 = ((lane_id / 8) % 2) * 8 + (lane_id / 16) * d * 8)
                saddr = k + (lane_id % 8) * d + ((lane_id / 8) % 2) * 8 + (lane_id / 16) * d * 8;
                LDMATRIX_X4(RB[0], RB[1], RB[2], RB[3], s_K + saddr);
                #if 0
                if (lane_id < 32 && warp_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i == 0 && k == 0) {
                    printf("lane_id = %d ldaddr = %d **** RB[0] = %g %g, RB[1] = %g %g, RB[2] = %g %g, RB[3] = %g %g\n", 
                        lane_id, saddr,
                        __half2float(reinterpret_cast<half*>(&(RB[0]))[0]),
                        __half2float(reinterpret_cast<half*>(&(RB[0]))[1]),
                        __half2float(reinterpret_cast<half*>(&(RB[1]))[0]),
                        __half2float(reinterpret_cast<half*>(&(RB[1]))[1]),
                        __half2float(reinterpret_cast<half*>(&(RB[2]))[0]),
                        __half2float(reinterpret_cast<half*>(&(RB[2]))[1]),
                        __half2float(reinterpret_cast<half*>(&(RB[3]))[0]),
                        __half2float(reinterpret_cast<half*>(&(RB[3]))[1])
                    );
                }
                #endif

                MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
                MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[2], RB[3], RC[2], RC[3]);
                __syncwarp();
            }
            // 将矩阵 C 的寄存器变量写入 s_QK，每个 warp 仅负责 [Br, Bc] 分片，sm_90 之前不支持 stmatrix 指令
            // 子矩阵按列主序填充，参照 mma 指令规定的矩阵 C 的元素排布，每次写入 32bit
            uint32_t s_QK_offset = warp_id * Br * Bc;
            uint32_t store_smem_qk_m = lane_id / 4;
            uint32_t store_smem_qk_n = (lane_id % 4) * 2;
            LDST32BITS(s_QK[s_QK_offset + store_smem_qk_m * Bc + store_smem_qk_n]) = LDST32BITS(RC[0]);
            LDST32BITS(s_QK[s_QK_offset + (store_smem_qk_m + 8) * Bc + store_smem_qk_n]) = LDST32BITS(RC[1]);
            LDST32BITS(s_QK[s_QK_offset + store_smem_qk_m * Bc + store_smem_qk_n + 8]) = LDST32BITS(RC[2]);
            LDST32BITS(s_QK[s_QK_offset + (store_smem_qk_m + 8) * Bc + store_smem_qk_n + 8]) = LDST32BITS(RC[3]);
            __syncwarp();

            #if 0
            if (lane_id == 0 && warp_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i == 0)
            {
                printf("s_QK[0-7]: %f %f %f %f %f %f %f %f\n", 
                    __half2float(s_QK[0]) * softmax_scale, __half2float(s_QK[1]) * softmax_scale, 
                    __half2float(s_QK[2]) * softmax_scale, __half2float(s_QK[3]) * softmax_scale, 
                    __half2float(s_QK[4]) * softmax_scale, __half2float(s_QK[5]) * softmax_scale, 
                    __half2float(s_QK[6]) * softmax_scale, __half2float(s_QK[7]) * softmax_scale);
            }
            #endif

            // 对 s_QK 求 softmax，每个 warp 单独计算 [16, 16] 矩阵的 softmax，根据 online-softmax 先计算 m 和 l
            // 一个 warp 每次单独处理两行，在 warp 内的 16 个线程内部做规约，总共需要处理 8 次
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                // 读取 2 行数据到 warp 
                MD_F tmp_ml = {__half2float(s_QK[s_QK_offset + j * 32 + lane_id]) * softmax_scale, 1.0f};
                __syncwarp();
                // 每行数据由 16 个线程组成的 group 持有，内部 reduce
                tmp_ml = threadGroupAllReduce<16>(tmp_ml);
                // 当前线程处理的行索引
                uint32_t current_row = warp_id * Br + j * 2 + (lane_id / 16);
                if ((lane_id % 16) == 0) {
                    row_ml[current_row] = tmp_ml;
                    row_ml_new[current_row] = MDFOp()(row_ml_prev[current_row], tmp_ml);
                }
                __syncwarp();
                s_S[current_row * Bc + (lane_id % 16)] = __float2half(
                    __expf(__half2float(s_QK[current_row * Bc + (lane_id % 16)]) * softmax_scale - row_ml[current_row].m));
                __syncwarp();
            }

            // 从 s_S load 到 RA，使用 ldmatrix.sync.aligned.x4.m8n8.shared.b16 指令
            // warp 内每个线程都需要传入一个地址
            uint32_t warp_offset = warp_id * Br * Bc;
            LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_S + warp_offset + (lane_id % 16) * Bc + (lane_id / 16) * 8);
            #if 0
                if (lane_id < 12 && warp_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i == 0) {
                    printf("lane_id = %d **** RA[0] = %g %g, RA[1] = %g %g, RA[2] = %g %g, RA[3] = %g %g\n",
                        lane_id, 
                        __half2float(reinterpret_cast<half*>(&(RA[0]))[0]),
                        __half2float(reinterpret_cast<half*>(&(RA[0]))[1]),
                        __half2float(reinterpret_cast<half*>(&(RA[1]))[0]),
                        __half2float(reinterpret_cast<half*>(&(RA[1]))[1]),
                        __half2float(reinterpret_cast<half*>(&(RA[2]))[0]),
                        __half2float(reinterpret_cast<half*>(&(RA[2]))[1]),
                        __half2float(reinterpret_cast<half*>(&(RA[3]))[0]),
                        __half2float(reinterpret_cast<half*>(&(RA[3]))[1])
                    );
                }
            #endif

            // 计算 QK 矩阵，每次计算尺寸为 16x16x16，
            #pragma unroll
            for (int k = 0; k < d; k += Bd) {
                // 初始化 RC
                RC[0] = 0;
                RC[1] = 0;
                RC[2] = 0;
                RC[3] = 0;

                // 从 s_V load 16x16 矩阵分片到 RB，使用 ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 指令
                // warp 内每个线程都需要传入一个地址
                LDMATRIX_X4_T(RB[0], RB[1], RB[2], RB[3], s_V + k + (lane_id % 16) * d + (lane_id / 16) * 8);

                MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
                MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[2], RB[3], RC[2], RC[3]);

                // 将矩阵 C 的寄存器变量写入 s_O[4 * Br, Bd]，每个 warp 仅负责 [Br, Bd] 分片，sm_90 之前不支持 stmatrix 指令
                // 子矩阵按列主序填充，参照 mma 指令规定的矩阵 C 的元素排布，每次写入 32bit
                uint32_t s_O_offset = warp_id * Br * Bd;
                uint32_t store_smem_o_m = lane_id / 4;
                uint32_t store_smem_o_n = (lane_id % 4) * 2;
                LDST32BITS(s_O[s_O_offset + store_smem_o_m * Bd + store_smem_o_n]) = LDST32BITS(RC[0]);
                LDST32BITS(s_O[s_O_offset + (store_smem_o_m + 8) * Bd + store_smem_o_n]) = LDST32BITS(RC[1]);
                LDST32BITS(s_O[s_O_offset + store_smem_o_m * Bd + store_smem_o_n + 8]) = LDST32BITS(RC[2]);
                LDST32BITS(s_O[s_O_offset + (store_smem_o_m + 8) * Bd + store_smem_o_n + 8]) = LDST32BITS(RC[3]);
                __syncwarp();

                // 更新 O，每个 warp 每次更新 [16, 16] 分片
                // 一个 warp 每次单独处理两行，在 warp 内的 16 个线程为一组，总共需要处理 8 次
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    // 当前元素在 [16, 16] 矩阵中的行索引
                    uint32_t current_row = j * 2 + (lane_id / 16);
                    // 当前元素在矩阵 O 中的索引
                    uint32_t out_idx = qo_offset + current_row * d + k + (lane_id % 16);
                    // 当前元素在矩阵 s_O[4 * Br, Bd] 中的索引
                    uint32_t s_o_idx = s_O_offset + current_row * Bd + (lane_id % 16);
                    // exp(m_prev-m_new)
                    float exp_sub_prev_new_m = __expf(row_ml_prev[warp_id * Br + current_row].m - row_ml_new[warp_id * Br + current_row].m);
                    // exp(m_cur-m_new)
                    float exp_sub_cur_new_m = __expf(row_ml[warp_id * Br + current_row].m - row_ml_new[warp_id * Br + current_row].m);
                    // 1.0 / l_new
                    float rlf_i = 1.0f / row_ml_new[warp_id * Br + current_row].d;
                    // 更新矩阵 O
                    O[out_idx] = __float2half(rlf_i * (row_ml_prev[warp_id * Br + current_row].d * exp_sub_prev_new_m * __half2float(O[out_idx]) + 
                            exp_sub_cur_new_m * __half2float(s_O[s_o_idx])));
                }
            }

            // 更新 row_ml_new
            if (lane_id < Br) {
                row_ml_prev[warp_id * Br + lane_id] = row_ml_new[warp_id * Br + lane_id];
            }
            __syncthreads();
        }
        #if 0
        if (threadIdx.x < 8 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
        {
            printf("row=%d row_ml_prev: m(%f) d(%f)\n", threadIdx.x, row_ml_prev[threadIdx.x].m, row_ml_prev[threadIdx.x].d);
        }
        #endif
    }


    void launchFlashAttentionKernel_v5(const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
        half* __restrict__ O, const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream) {
        printf("in fa v4 launchFlashAttentionKernel_v4\n");
        constexpr int Bc = 16;
        constexpr int Br = 16;
        // 让 Bd 等于 Bc 从而使得 QK 矩阵分片[Br, Bc] 与 QKV 矩阵分片[Br, Bd] 形状相同，方便排布
        constexpr int Bd = Bc;
        assert(M % Bc == 0 && N % (4 * Br) == 0 && d % Bc == 0);
        const float softmax_scale = 1.0f / sqrtf((float)d);

        /**
        __shared__ half s_Q[4 * Br * d];
        __shared__ half s_K[Bc * d];
        __shared__ half s_V[Bc * d];
        __shared__ half s_QK[4 * Br * Bc];
        __shared__ half s_S[4 * Br * Bc];
        __shared__ half s_O[4 * Br * Bd];

        // 前一个 Bc 组的 l 和 m
        __shared__ MD_F row_ml_prev[4 * Br];
        __shared__ MD_F row_ml[4 * Br];
        __shared__ MD_F row_ml_new[4 * Br];
        */

        const int sram_size = (4 * Br * 3) * sizeof(MD_F) + (4 * Br * d + 2 * Bc * d + 4 * Br * Bc * 2 + 4 * Br * Bd) * sizeof(half);
        int max_sram_size;
        cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
        printf("Max shared memory: %g KB, requested shared memory: %g KB \n", max_sram_size / 1024.0f, sram_size / 1024.0f);

        dim3 grid_dim(div_ceil(N, 4 * Br), num_head, batch_size);
        dim3 block_dim(128);
        flashAttentionKernel_v5<Br, Bc, Bd><<<grid_dim, block_dim, sram_size, stream >>>(Q, K, V, O, N, M, d, softmax_scale);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }


    __global__ void softmaxKernel(const float* __restrict__ mat, float* __restrict__ output, const int ncol, const float softmax_scale) {
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

    void launchAttentionBaseline(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
        float* __restrict__ QK, float* __restrict__ QK_softmax, float* __restrict__ O,
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

} // namespace attention
