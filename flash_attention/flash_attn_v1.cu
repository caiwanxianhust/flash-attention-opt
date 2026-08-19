
#include "attention.cuh"
#include "utils.h"
#include "common.cuh"

#include <assert.h>
#include <mma.h>

/**
     * grid( num_head, batch_size )
     * block( BLOCK_SIZE )
     * Q\O: [batch_size, num_head, N, d]
     * K\V: [batch_size, num_head, M, d]
     * l: [batch_size, num_head, N, 1]
     * m: [batch_size, num_head, N, 1]
     */
template <int Bc>
__global__ void flash_attn_v1_tiling_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
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


/**
 * 基础 Tiling 实现
 * 将 K/V 分块加载到 Shared Memory，外层循环遍历 Q 的每一行，内层循环遍历 K 的分块，逐行计算 Softmax 和 O。
 */
void launch_flash_attn_v1_tiling_kernel(const float* Q, const float* K, const float* V,
    float* O, float* l, float* m,
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream) {
    constexpr int Bc = 4;
    assert(M % Bc == 0);
    const float softmax_scale = 1.0f / sqrtf((float)d);

    const int sram_size = (d + 2 * Bc * d + Bc) * sizeof(float);
#if 0
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %g KB, requested shared memory: %g KB \n", max_sram_size / 1024.0f, sram_size / 1024.0f);
#endif
    constexpr int block_size = 128;
    dim3 grid_dim(num_head, batch_size);
    dim3 block_dim(block_size);
    flash_attn_v1_tiling_kernel<Bc> << <grid_dim, block_dim, sram_size, stream >> > (Q, K, V, O, l, m, N, M, d, softmax_scale);
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
__global__ void flash_attn_v1_tiling_warp_parallel_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
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


/**
 * Warp 级并行优化
 *   将 Q 也进行分块（Br），利用 Warp 内线程协作（warpAllReduce）并行计算点积和 Softmax，一次处理多行 Q。
 */
void launch_flash_attn_v1_tiling_warp_parallel_kernel(const float* Q, const float* K, const float* V,
    float* O, float* l, float* m,
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream) {
    constexpr int Bc = 2;
    constexpr int Br = 4;
    assert(M % Bc == 0);
    const float softmax_scale = 1.0f / sqrtf((float)d);

    const int sram_size = (Br * d + 2 * Bc * d + Br * Bc) * sizeof(float);
#if 0
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %g KB, requested shared memory: %g KB \n", max_sram_size / 1024.0f, sram_size / 1024.0f);
#endif

    dim3 grid_dim(num_head, batch_size);
    dim3 block_dim(Br * 32);
    flash_attn_v1_tiling_warp_parallel_kernel<Bc, Br> << <grid_dim, block_dim, sram_size, stream >> > (Q, K, V, O, l, m, N, M, d, softmax_scale);
}

namespace {

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
__global__ void flash_attn_v1_tiling_D_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
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
                row_ml_tmp = warpAllReduceMDF(row_ml_tmp);
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


/**
 * 在 d 维度上更细粒度的分块计算，同时引入 WMMA API
 *  引入 NVIDIA WMMA API，将 Q/K/V 转换为 half 类型，利用 Tensor Core 进行矩阵乘法（GEMM），并行计算 QK 和 PV。
 */
void launch_flash_attn_v1_tiling_D_kernel(const float* Q, const float* K, const float* V,
    float* O, float* l, float* m,
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream) {
    constexpr int Bc = 32;
    constexpr int Br = 64;
    constexpr int Wr = 32;
    constexpr int Wc = 16;
    constexpr int Bd = Bc; // 让 Bd 等于 Bc 从而使得 QK 矩阵分片[Br, Bc] 与 QKV 矩阵分片[Br, Bd] 形状相同，方便排布

    assert(M % Bc == 0 && N % Br == 0 && d % Bd == 0);
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

#if 0
    const int sram_size = (Br * Bc + Br * Bd) * sizeof(float) + (Br * Bd + 2 * Bc * Bd + Br * Bc) * sizeof(half) + 8 * 3 * Br;
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %g KB, requested shared memory: %g KB \n", max_sram_size / 1024.0f, sram_size / 1024.0f);
#endif

    dim3 grid_dim(num_head, batch_size);
    dim3 block_dim(Bc * Br / (Wr * Wc) * 32);
    flash_attn_v1_tiling_D_kernel<Bc, Br, Wc, Wr> << <grid_dim, block_dim, 0, stream >> > (Q, K, V, O, l, m, N, M, d, softmax_scale);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


