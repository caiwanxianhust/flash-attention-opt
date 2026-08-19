
#include "attention.cuh"
#include "utils.h"
#include "common.cuh"

#include <cuda_fp16.h>
#include <assert.h>
#include <cfloat>
#include <mma.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>


namespace cg = cooperative_groups;

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
     */
template <int Bc, int Br, int Wc, int Wr>
__global__ void flash_attn_v2_tiling_kernel(const float* __restrict__ Q,
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
                row_ml_tmp = warpAllReduceMDF(row_ml_tmp);
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
        if (threadIdx.x < 8 && blockIdx.x == 0 && blockIdx.y == 0 && j == 0) {
            printf("row=%d row_ml_prev: m(%f) d(%f)\n", threadIdx.x, row_ml_prev[threadIdx.x].m, row_ml_prev[threadIdx.x].d);
        }
#endif
    }

}


/**
 * 循环重排与流水线优化
 * 调整循环顺序（先遍历 Q 分块，再遍历 K 分块），在 K 的循环内部累加 O，减少了 O 的重复读写。
 * 通过循环交换减少了 Global Memory 的访问量，更好地利用了寄存器缓存中间结果，进一步提升了带宽利用率。
 */
void launch_flash_attn_v2_tiling_kernel(const float* Q, const float* K, const float* V, float* O,
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream) {
    constexpr int Bc = 32;
    constexpr int Br = 64;
    constexpr int Wr = 32;
    constexpr int Wc = 16;
    constexpr int Bd = Bc;  // 让 Bd 等于 Bc 从而使得 QK 矩阵分片[Br, Bc] 与 QKV 矩阵分片[Br, Bd] 形状相同，方便排布

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
    flash_attn_v2_tiling_kernel<Bc, Br, Wc, Wr> << <grid_dim, block_dim, 0, stream >> > (Q, K, V, O, N, M, d, softmax_scale);
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
__global__ void flash_attn_v2_mma_kernel(const half* __restrict__ Q,
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
        reinterpret_cast<float4*>(s_Q + warp_id * Br * d + i)[0] = reinterpret_cast<const float4*>(Q + qo_offset + i)[0];
    }
    __syncwarp();

#if 0
    if (lane_id == 0 && warp_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
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
        uint32_t RC[4] = { 0, 0, 0, 0 };

        // load [Bc, d] 的 K 矩阵分片到 s_K，整个 block 一起 load [Br, d]，每次 load 8 个 half
        for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
            reinterpret_cast<float4*>(s_K + j)[0] = reinterpret_cast<const float4*>(K + kv_offset + i * d + j)[0];
            reinterpret_cast<float4*>(s_V + j)[0] = reinterpret_cast<const float4*>(V + kv_offset + i * d + j)[0];
        }
        __syncthreads();

#if 0
        if (lane_id == 0 && warp_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i == 0) {
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
        uint32_t store_smem_qk_m = lane_id / 4;
        uint32_t store_smem_qk_n = (lane_id % 4) * 2;
        LDST32BITS(s_QK[warp_id * Br * Bc + store_smem_qk_m * Bc + store_smem_qk_n]) = LDST32BITS(RC[0]);
        LDST32BITS(s_QK[warp_id * Br * Bc + (store_smem_qk_m + 8) * Bc + store_smem_qk_n]) = LDST32BITS(RC[1]);
        LDST32BITS(s_QK[warp_id * Br * Bc + store_smem_qk_m * Bc + store_smem_qk_n + 8]) = LDST32BITS(RC[2]);
        LDST32BITS(s_QK[warp_id * Br * Bc + (store_smem_qk_m + 8) * Bc + store_smem_qk_n + 8]) = LDST32BITS(RC[3]);
        __syncwarp();

#if 0
        if (lane_id == 0 && warp_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i == 0) {
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
            MD_F tmp_ml = { __half2float(s_QK[warp_id * Br * Bc + j * 32 + lane_id]) * softmax_scale, 1.0f };
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

        // 计算 QKV 矩阵，每次计算尺寸为 16x16x16，
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
            uint32_t store_smem_o_m = lane_id / 4;
            uint32_t store_smem_o_n = (lane_id % 4) * 2;
            LDST32BITS(s_O[warp_id * Br * Bd + store_smem_o_m * Bd + store_smem_o_n]) = LDST32BITS(RC[0]);
            LDST32BITS(s_O[warp_id * Br * Bd + (store_smem_o_m + 8) * Bd + store_smem_o_n]) = LDST32BITS(RC[1]);
            LDST32BITS(s_O[warp_id * Br * Bd + store_smem_o_m * Bd + store_smem_o_n + 8]) = LDST32BITS(RC[2]);
            LDST32BITS(s_O[warp_id * Br * Bd + (store_smem_o_m + 8) * Bd + store_smem_o_n + 8]) = LDST32BITS(RC[3]);
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
                uint32_t s_o_idx = warp_id * Br * Bd + current_row * Bd + (lane_id % 16);
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

        // 更新 row_ml_prev
        if (lane_id < Br) {
            row_ml_prev[warp_id * Br + lane_id] = row_ml_new[warp_id * Br + lane_id];
        }
        __syncthreads();
    }
#if 0
    if (threadIdx.x < 8 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("row=%d row_ml_prev: m(%f) d(%f)\n", threadIdx.x, row_ml_prev[threadIdx.x].m, row_ml_prev[threadIdx.x].d);
    }
#endif
}


/**
 * 使用 ldmatrix 和 mma 指令
 */
void launch_flash_attn_v2_mma_kernel(const half* Q, const half* K, const half* V,
    half* O, const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream) {
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
#if 0
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %g KB, requested shared memory: %g KB \n", max_sram_size / 1024.0f, sram_size / 1024.0f);
#endif
    dim3 grid_dim(div_ceil(N, 4 * Br), num_head, batch_size);
    dim3 block_dim(128);
    flash_attn_v2_mma_kernel<Br, Bc, Bd> << <grid_dim, block_dim, sram_size, stream >> > (Q, K, V, O, N, M, d, softmax_scale);
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


/**
     * \tparam S: SShift, right shift the addr for swizzling
     * \tparam B: BShift, bits to be swizzled
     * \tparam M: MBase, bits keep the same
     */
template <uint32_t B, uint32_t M, uint32_t S>
__device__ __forceinline__ uint32_t swizzle(uint32_t addr) {
    constexpr auto Bmask = ((1 << B) - 1) << M;
    return ((addr >> S) & Bmask) ^ addr;
}

/** 在 v5 基础上加入 swizzle 机制
 * grid(div_ceil(N, 4 * Br), num_head, batch_size )
 * block( 128 )
 * Q\O: [batch_size, num_head, N, d]
 * K\V: [batch_size, num_head, M, d]
 */
template <uint32_t Br, uint32_t Bc, uint32_t Bd>
__global__ void flash_attn_v2_mma_swizzle_kernel(const half* __restrict__ Q,
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
    // s_Q 的宽度是 d，当大于 64 的时候 swizzle_B 应该取 3，当前按 d = 128 考虑
    for (int i = (lane_id << 3); i < Br * d; i += (32 << 3)) {
        uint32_t dst_addr = swizzle<3, 3, 4>(i);
        reinterpret_cast<float4*>(s_Q + warp_id * Br * d + dst_addr)[0] = reinterpret_cast<const float4*>(Q + qo_offset + i)[0];
    }
    __syncwarp();

#if 0
    if (lane_id == 0 && warp_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        uint32_t print_row = 2;
        printf("s_Q[%d][0-15]: %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", print_row,
            __half2float(s_Q[print_row * d + 0]), __half2float(s_Q[print_row * d + 1]),
            __half2float(s_Q[print_row * d + 2]), __half2float(s_Q[print_row * d + 3]),
            __half2float(s_Q[print_row * d + 4]), __half2float(s_Q[print_row * d + 5]),
            __half2float(s_Q[print_row * d + 6]), __half2float(s_Q[print_row * d + 7]),
            __half2float(s_Q[print_row * d + 8]), __half2float(s_Q[print_row * d + 9]),
            __half2float(s_Q[print_row * d + 10]), __half2float(s_Q[print_row * d + 11]),
            __half2float(s_Q[print_row * d + 12]), __half2float(s_Q[print_row * d + 13]),
            __half2float(s_Q[print_row * d + 14]), __half2float(s_Q[print_row * d + 15]));
    }
#endif

    // warp 矩阵乘法的尺寸为 16x16x16，调用两次 mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 指令
    // 所以 3 个矩阵都需要 4 个寄存器
    uint32_t RA[4];
    uint32_t RB[4];

    // 对 K|V 在 M 维度分组，每组长度为 Bc，共分为 Tc 组
    for (int i = 0; i < M; i += Bc) {
        // 初始化矩阵 C 的寄存器
        uint32_t RC[4] = { 0, 0, 0, 0 };

        // load [Bc, d] 的 K 矩阵分片到 s_K，整个 block 一起 load [Br, d]，每次 load 8 个 half
        // s_K s_V 的宽度是 d，当大于 64 的时候 swizzle_B 应该取 3，当前按 d = 128 考虑
        for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
            uint32_t dst_addr = swizzle<3, 3, 4>(j);
            reinterpret_cast<float4*>(s_K + dst_addr)[0] = reinterpret_cast<const float4*>(K + kv_offset + i * d + j)[0];
            reinterpret_cast<float4*>(s_V + dst_addr)[0] = reinterpret_cast<const float4*>(V + kv_offset + i * d + j)[0];
        }
        __syncthreads();

#if 0
        if (lane_id == 0 && warp_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i == 0) {
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
            uint32_t src_addr = k + (lane_id % 16) * d + (lane_id / 16) * 8;
            uint32_t dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_Q + warp_id * Br * d + dst_addr);
#if 0
            if (lane_id < 12 && warp_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i == 0 && k == 0) {
                printf("lane_id = %d dst_addr = %d **** RA[0] = %g %g, RA[1] = %g %g, RA[2] = %g %g, RA[3] = %g %g\n",
                    lane_id, dst_addr,
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
            src_addr = k + (lane_id % 8) * d + ((lane_id / 8) % 2) * 8 + (lane_id / 16) * d * 8;
            dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4(RB[0], RB[1], RB[2], RB[3], s_K + dst_addr);
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
        // s_QK 宽度为 Bc，等于 16，此时 swizzle_B = 1
        uint32_t store_smem_qk_m = lane_id / 4;
        uint32_t store_smem_qk_n = (lane_id % 4) * 2;
        uint32_t dst_addr_c0 = swizzle<1, 3, 3>(store_smem_qk_m * Bc + store_smem_qk_n);
        uint32_t dst_addr_c1 = swizzle<1, 3, 3>((store_smem_qk_m + 8) * Bc + store_smem_qk_n);
        uint32_t dst_addr_c2 = swizzle<1, 3, 3>(store_smem_qk_m * Bc + store_smem_qk_n + 8);
        uint32_t dst_addr_c3 = swizzle<1, 3, 3>((store_smem_qk_m + 8) * Bc + store_smem_qk_n + 8);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c0]) = LDST32BITS(RC[0]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c1]) = LDST32BITS(RC[1]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c2]) = LDST32BITS(RC[2]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c3]) = LDST32BITS(RC[3]);
        __syncwarp();

#if 0
        if (lane_id == 0 && warp_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i == 0) {
            printf("s_QK[0-7]: %f %f %f %f %f %f %f %f\n",
                __half2float(s_QK[0]) * softmax_scale, __half2float(s_QK[1]) * softmax_scale,
                __half2float(s_QK[2]) * softmax_scale, __half2float(s_QK[3]) * softmax_scale,
                __half2float(s_QK[4]) * softmax_scale, __half2float(s_QK[5]) * softmax_scale,
                __half2float(s_QK[6]) * softmax_scale, __half2float(s_QK[7]) * softmax_scale);
        }
#endif

        // 对 s_QK 求 softmax，每个 warp 单独计算 [16, 16] 矩阵的 softmax，根据 online-softmax 先计算 m 和 l
        // 一个 warp 每次单独处理两行，在 warp 内的 16 个线程内部做规约，总共需要处理 8 次
        // 由于 s_QK 的宽度为 Bc 即 16，通过 swizzle<1, 3, 3> 映射后，目的地址与源地址在相同行，所以不影响 softmax 求 m 和 l
#pragma unroll
        for (int j = 0; j < 8; j++) {
            // 读取 2 行数据到 warp 
            MD_F tmp_ml = { __half2float(s_QK[warp_id * Br * Bc + j * 32 + lane_id]) * softmax_scale, 1.0f };
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
        // s_S 布局与 s_QK 一致，所以通过 swizzle<1, 3, 3> 映射
        uint32_t warp_offset = warp_id * Br * Bc;
        uint32_t dst_addr = swizzle<1, 3, 3>((lane_id % 16) * Bc + (lane_id / 16) * 8);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_S + warp_offset + dst_addr);
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
        for (int k = 0; k < d; k += Bd) {
            // 初始化 RC
            RC[0] = 0;
            RC[1] = 0;
            RC[2] = 0;
            RC[3] = 0;

            // 从 s_V load 16x16 矩阵分片到 RB，使用 ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 指令
            // warp 内每个线程都需要传入一个地址
            uint32_t src_addr = k + (lane_id % 16) * d + (lane_id / 16) * 8;
            uint32_t dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4_T(RB[0], RB[1], RB[2], RB[3], s_V + dst_addr);

            MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
            MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[2], RB[3], RC[2], RC[3]);

            // 将矩阵 C 的寄存器变量写入 s_O[4 * Br, Bd]，每个 warp 仅负责 [Br, Bd] 分片，sm_90 之前不支持 stmatrix 指令
            // 子矩阵按列主序填充，参照 mma 指令规定的矩阵 C 的元素排布，每次写入 32bit
            // s_O 宽度为 Bd，等于 16，此时 swizzle_B = 1
            uint32_t store_smem_o_m = lane_id / 4;
            uint32_t store_smem_o_n = (lane_id % 4) * 2;
            uint32_t dst_addr_c0 = swizzle<1, 3, 3>(store_smem_o_m * Bd + store_smem_o_n);
            uint32_t dst_addr_c1 = swizzle<1, 3, 3>((store_smem_o_m + 8) * Bd + store_smem_o_n);
            uint32_t dst_addr_c2 = swizzle<1, 3, 3>(store_smem_o_m * Bd + store_smem_o_n + 8);
            uint32_t dst_addr_c3 = swizzle<1, 3, 3>((store_smem_o_m + 8) * Bd + store_smem_o_n + 8);
            LDST32BITS(s_O[warp_id * Br * Bd + dst_addr_c0]) = LDST32BITS(RC[0]);
            LDST32BITS(s_O[warp_id * Br * Bd + dst_addr_c1]) = LDST32BITS(RC[1]);
            LDST32BITS(s_O[warp_id * Br * Bd + dst_addr_c2]) = LDST32BITS(RC[2]);
            LDST32BITS(s_O[warp_id * Br * Bd + dst_addr_c3]) = LDST32BITS(RC[3]);
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
                uint32_t s_o_idx = warp_id * Br * Bd + swizzle<1, 3, 3>(current_row * Bd + (lane_id % 16));
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
    if (threadIdx.x < 8 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("row=%d row_ml_prev: m(%f) d(%f)\n", threadIdx.x, row_ml_prev[threadIdx.x].m, row_ml_prev[threadIdx.x].d);
    }
#endif
}


void launch_flash_attn_v2_mma_swizzle_kernel(const half* Q, const half* K, const half* V,
    half* O, const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream) {
    constexpr uint32_t Bc = 16;
    constexpr uint32_t Br = 16;
    // 让 Bd 等于 Bc 从而使得 QK 矩阵分片[Br, Bc] 与 QKV 矩阵分片[Br, Bd] 形状相同，方便排布
    constexpr uint32_t Bd = Bc;
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
#if 0
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %g KB, requested shared memory: %g KB \n", max_sram_size / 1024.0f, sram_size / 1024.0f);
#endif
    dim3 grid_dim(div_ceil(N, 4 * Br), num_head, batch_size);
    dim3 block_dim(128);
    flash_attn_v2_mma_swizzle_kernel<Br, Bc, Bd> << <grid_dim, block_dim, sram_size, stream >> > (Q, K, V, O, N, M, d, softmax_scale);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}



// ==================== 双缓冲新增:cp.async 辅助宏 ====================
// sm_80+ 才支持 cp.async(你是 sm_89,没问题)
#define CP_ASYNC_16(smem_ptr, global_ptr) \
    do { \
        unsigned int _smem_addr = (unsigned int) __cvta_generic_to_shared(smem_ptr); \
        asm volatile( \
            "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" \
            : \
            : "r"(_smem_addr), "l"(global_ptr) \
        ); \
    } while(0)

// 提交当前所有 cp.async 为一个 group
#define CP_ASYNC_COMMIT() asm volatile("cp.async.commit_group;\n")

// 等待最近 N 个 group 之前的 group 全部完成
#define CP_ASYNC_WAIT_GROUP(N) \
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N))




/**
 * K/V 双缓冲版 flash attention v2
 * 在 v5(mma + swizzle)基础上加 K/V 双缓冲(cp.async)
 *
 * grid(div_ceil(N, 4 * Br), num_head, batch_size)
 * block(128)
 * Q/O: [batch_size, num_head, N, d]
 * K/V: [batch_size, num_head, M, d]
 *
 * 双缓冲:用 s_K_buf[2], s_V_buf[2] 轮转
 *   - 当前 iter 用 s_K_buf[iter&1], s_V_buf[iter&1]
 *   - 同时后台预取下一组 K/V 到 s_K_buf[(iter+1)&1], s_V_buf[(iter+1)&1]
 */
template <uint32_t Br, uint32_t Bc, uint32_t Bd>
__global__ void flash_attn_v2_mma_swizzle_doublebuffer_kernel(const half* __restrict__ Q,
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

    // ============ 双缓冲 shared memory 布局 ============
    extern __shared__ half smem_ptr[];
    half* s_Q = reinterpret_cast<half*>(smem_ptr);

    // ★ 双份 K/V buffer
    half* s_K_buf[2];
    half* s_V_buf[2];
    s_K_buf[0] = s_Q + 4 * Br * d;
    s_V_buf[0] = s_K_buf[0] + Bc * d;
    s_K_buf[1] = s_V_buf[0] + Bc * d;
    s_V_buf[1] = s_K_buf[1] + Bc * d;

    half* s_QK = s_V_buf[1] + Bc * d;
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

    // load [4 * Br, d] 的 Q 矩阵分片到 s_Q,每个 warp load [Br, d],每次 load 8 个 half
    // s_Q 的宽度是 d,当大于 64 的时候 swizzle_B 应该取 3,当前按 d = 128 考虑
    for (int i = (lane_id << 3); i < Br * d; i += (32 << 3)) {
        uint32_t dst_addr = swizzle<3, 3, 4>(i);
        reinterpret_cast<float4*>(s_Q + warp_id * Br * d + dst_addr)[0] = reinterpret_cast<const float4*>(Q + qo_offset + i)[0];
    }
    __syncwarp();


    // warp 矩阵乘法的尺寸为 16x16x16,调用两次 mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 指令
    // 所以 3 个矩阵都需要 4 个寄存器
    uint32_t RA[4];
    uint32_t RB[4];

    // 总迭代次数
    const int num_iters = M / Bc;

    // ============ 预加载第 0 组 K/V 到 buffer 0 ============
    {
        half* sK = s_K_buf[0];
        half* sV = s_V_buf[0];
        for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
            uint32_t dst_addr = swizzle<3, 3, 4>(j);
            CP_ASYNC_16(sK + dst_addr, K + kv_offset + 0 * d + j);
            CP_ASYNC_16(sV + dst_addr, V + kv_offset + 0 * d + j);
        }
        CP_ASYNC_COMMIT(); // group 0
    }

    // ============ 主循环(双缓冲) ============
    for (int iter = 0; iter < num_iters; ++iter) {
        const int curr_buf = iter & 1;        // 当前用的 buffer
        const int next_buf = curr_buf ^ 1;    // 下一个 buffer
        const int kv_off_next = (iter + 1) * Bc;

        // ★ 预取下一组 K/V(到 next_buf),不阻塞当前计算
        //    仅在 iter+1 < num_iters 时执行
        if (iter + 1 < num_iters) {
            half* sK = s_K_buf[next_buf];
            half* sV = s_V_buf[next_buf];
            for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
                uint32_t dst_addr = swizzle<3, 3, 4>(j);
                CP_ASYNC_16(sK + dst_addr, K + kv_offset + kv_off_next * d + j);
                CP_ASYNC_16(sV + dst_addr, V + kv_offset + kv_off_next * d + j);
            }
            CP_ASYNC_COMMIT();   // 新 group
        }

        // ★ 等待当前 buffer 的 K/V 加载完成
        //    pipeline 里至少有一个未完成 group(iter+1 的预取)
        //    wait_group 1 等所有比最新 group 早的 group 完成(即当前 curr_buf 的加载)
        CP_ASYNC_WAIT_GROUP(1);
        __syncthreads();

        // ★ 用 curr_buf 的 K/V 做计算(以下逻辑完全照搬原版,只把 s_K/s_V 改成带 buffer 下标)
        half* s_K = s_K_buf[curr_buf];
        half* s_V = s_V_buf[curr_buf];

        // 初始化矩阵 C 的寄存器
        uint32_t RC[4] = { 0, 0, 0, 0 };

        // 计算 QK 矩阵,每次计算尺寸为 16x16x16,
#pragma unroll
        for (int k = 0; k < d; k += Bd) {
            // 从 s_Q load 16x16 矩阵分片到 RA,使用 ldmatrix.sync.aligned.x4.m8n8.shared.b16 指令
            // warp 内每个线程都需要传入一个地址
            uint32_t src_addr = k + (lane_id % 16) * d + (lane_id / 16) * 8;
            uint32_t dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_Q + warp_id * Br * d + dst_addr);

            // 从 s_K(列主序) load 16x16 矩阵分片到 RB,使用 ldmatrix.sync.aligned.x4.m8n8.shared.b16 指令
            src_addr = k + (lane_id % 8) * d + ((lane_id / 8) % 2) * 8 + (lane_id / 16) * d * 8;
            dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4(RB[0], RB[1], RB[2], RB[3], s_K + dst_addr);

            MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
            MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[2], RB[3], RC[2], RC[3]);
            // __syncwarp();
        }
        // 将矩阵 C 的寄存器变量写入 s_QK,每个 warp 仅负责 [Br, Bc] 分片,sm_90 之前不支持 stmatrix 指令
        uint32_t store_smem_qk_m = lane_id / 4;
        uint32_t store_smem_qk_n = (lane_id % 4) * 2;
        uint32_t dst_addr_c0 = swizzle<1, 3, 3>(store_smem_qk_m * Bc + store_smem_qk_n);
        uint32_t dst_addr_c1 = swizzle<1, 3, 3>((store_smem_qk_m + 8) * Bc + store_smem_qk_n);
        uint32_t dst_addr_c2 = swizzle<1, 3, 3>(store_smem_qk_m * Bc + store_smem_qk_n + 8);
        uint32_t dst_addr_c3 = swizzle<1, 3, 3>((store_smem_qk_m + 8) * Bc + store_smem_qk_n + 8);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c0]) = LDST32BITS(RC[0]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c1]) = LDST32BITS(RC[1]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c2]) = LDST32BITS(RC[2]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c3]) = LDST32BITS(RC[3]);
        __syncwarp();


        // 对 s_QK 求 softmax,每个 warp 单独计算 [16, 16] 矩阵的 softmax
#pragma unroll
        for (int j = 0; j < 8; j++) {
            // 读取 2 行数据到 warp
            MD_F tmp_ml = { __half2float(s_QK[warp_id * Br * Bc + j * 32 + lane_id]) * softmax_scale, 1.0f };
            __syncwarp();
            // 每行数据由 16 个线程组成的 group 持有,内部 reduce
            tmp_ml = threadGroupAllReduce<16>(tmp_ml);
            // 当前线程处理的行索引
            uint32_t current_row = warp_id * Br + j * 2 + (lane_id / 16);

            s_S[current_row * Bc + (lane_id % 16)] = __float2half(
                __expf(__half2float(s_QK[current_row * Bc + (lane_id % 16)]) * softmax_scale - tmp_ml.m));

            if ((lane_id % 16) == 0) {
                row_ml[current_row] = tmp_ml;
                row_ml_new[current_row] = MDFOp()(row_ml_prev[current_row], tmp_ml);
            }
            __syncwarp();
        }

        // 从 s_S load 到 RA
        uint32_t warp_offset = warp_id * Br * Bc;
        uint32_t dst_addr = swizzle<1, 3, 3>((lane_id % 16) * Bc + (lane_id / 16) * 8);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_S + warp_offset + dst_addr);

        // 计算 PV 矩阵,每次计算尺寸为 16x16x16,
        for (int k = 0; k < d; k += Bd) {
            // 初始化 RC
            RC[0] = 0;
            RC[1] = 0;
            RC[2] = 0;
            RC[3] = 0;

            // 从 s_V load 16x16 矩阵分片到 RB,使用 ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 指令
            uint32_t src_addr = k + (lane_id % 16) * d + (lane_id / 16) * 8;
            uint32_t dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4_T(RB[0], RB[1], RB[2], RB[3], s_V + dst_addr);

            MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
            MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[2], RB[3], RC[2], RC[3]);

            // 将矩阵 C 的寄存器变量写入 s_O[4 * Br, Bd]
            uint32_t store_smem_o_m = lane_id / 4;
            uint32_t store_smem_o_n = (lane_id % 4) * 2;
            uint32_t dst_addr_c0 = swizzle<1, 3, 3>(store_smem_o_m * Bd + store_smem_o_n);
            uint32_t dst_addr_c1 = swizzle<1, 3, 3>((store_smem_o_m + 8) * Bd + store_smem_o_n);
            uint32_t dst_addr_c2 = swizzle<1, 3, 3>(store_smem_o_m * Bd + store_smem_o_n + 8);
            uint32_t dst_addr_c3 = swizzle<1, 3, 3>((store_smem_o_m + 8) * Bd + store_smem_o_n + 8);
            LDST32BITS(s_O[warp_id * Br * Bd + dst_addr_c0]) = LDST32BITS(RC[0]);
            LDST32BITS(s_O[warp_id * Br * Bd + dst_addr_c1]) = LDST32BITS(RC[1]);
            LDST32BITS(s_O[warp_id * Br * Bd + dst_addr_c2]) = LDST32BITS(RC[2]);
            LDST32BITS(s_O[warp_id * Br * Bd + dst_addr_c3]) = LDST32BITS(RC[3]);
            __syncwarp();

            // 更新 O,每个 warp 每次更新 [16, 16] 分片
#pragma unroll
            for (int j = 0; j < 8; j++) {
                uint32_t current_row = j * 2 + (lane_id / 16);
                uint32_t out_idx = qo_offset + current_row * d + k + (lane_id % 16);
                uint32_t s_o_idx = warp_id * Br * Bd + swizzle<1, 3, 3>(current_row * Bd + (lane_id % 16));
                float exp_sub_prev_new_m = __expf(row_ml_prev[warp_id * Br + current_row].m - row_ml_new[warp_id * Br + current_row].m);
                float exp_sub_cur_new_m = __expf(row_ml[warp_id * Br + current_row].m - row_ml_new[warp_id * Br + current_row].m);
                float rlf_i = 1.0f / row_ml_new[warp_id * Br + current_row].d;
                O[out_idx] = __float2half(rlf_i * (row_ml_prev[warp_id * Br + current_row].d * exp_sub_prev_new_m * __half2float(O[out_idx]) +
                    exp_sub_cur_new_m * __half2float(s_O[s_o_idx])));
            }
        }

        // 更新 row_ml_prev
        if (lane_id < Br) {
            row_ml_prev[warp_id * Br + lane_id] = row_ml_new[warp_id * Br + lane_id];
        }
        __syncthreads();
    }

}


void launch_flash_attn_v2_mma_swizzle_doublebuffer_kernel(const half* Q, const half* K, const half* V,
    half* O, const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream) {
    constexpr uint32_t Bc = 16;
    constexpr uint32_t Br = 16;
    // 让 Bd 等于 Bc 从而使得 QK 矩阵分片[Br, Bc] 与 QKV 矩阵分片[Br, Bd] 形状相同,方便排布
    constexpr uint32_t Bd = Bc;
    assert(M % Bc == 0 && N % (4 * Br) == 0 && d % Bc == 0);
    const float softmax_scale = 1.0f / sqrtf((float)d);

    /**
    __shared__ half s_Q[4 * Br * d];
    __shared__ half s_K[2 * Bc * d];   // ★ 双缓冲
    __shared__ half s_V[2 * Bc * d];   // ★ 双缓冲
    __shared__ half s_QK[4 * Br * Bc];
    __shared__ half s_S[4 * Br * Bc];
    __shared__ half s_O[4 * Br * Bd];

    // 前一个 Bc 组的 l 和 m
    __shared__ MD_F row_ml_prev[4 * Br];
    __shared__ MD_F row_ml[4 * Br];
    __shared__ MD_F row_ml_new[4 * Br];
    */
    // ★ 双缓冲后,多 1 份 K + 1 份 V
    const int sram_size = (4 * Br * 3) * sizeof(MD_F) + (4 * Br * d + 4 * Bc * d + 4 * Br * Bc * 2 + 4 * Br * Bd) * sizeof(half);

    dim3 grid_dim(div_ceil(N, 4 * Br), num_head, batch_size);
    dim3 block_dim(128);
    flash_attn_v2_mma_swizzle_doublebuffer_kernel<Br, Bc, Bd> << <grid_dim, block_dim, sram_size, stream >> > (Q, K, V, O, N, M, d, softmax_scale);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


/**
 * 多缓冲版 flash attention v2(N_BUF 由模板参数控制)
 *
 * grid(div_ceil(N, 4 * Br), num_head, batch_size)
 * block(128)
 *
 * 模板参数:
 *   Br, Bc, Bd: tile 尺寸
 *   N_BUF:    buffer 数量(2 = 双缓冲, 3 = 三缓冲, 4 = 四缓冲, ...)
 *
 * 使用方法:
 *   flash_attn_v2_mma_swizzle_multistage_kernel<16, 16, 16, 2>  // 双缓冲
 *   flash_attn_v2_mma_swizzle_multistage_kernel<16, 16, 16, 3>  // 三缓冲
 *   flash_attn_v2_mma_swizzle_multistage_kernel<16, 16, 16, 4>  // 四缓冲
 */
template <uint32_t Br, uint32_t Bc, uint32_t Bd, uint32_t N_BUF>
__global__ void flash_attn_v2_mma_swizzle_multistage_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int N, const int M, const int d, const float softmax_scale) {
    static_assert(N_BUF >= 2 && N_BUF <= 8, "N_BUF must be in [2, 8]");

    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t lane_id = threadIdx.x & 0x1f;

    const uint32_t qo_offset = (blockIdx.z * gridDim.y + blockIdx.y) * N * d + blockIdx.x * 4 * Br * d + warp_id * Br * d;
    const uint32_t kv_offset = (blockIdx.z * gridDim.y + blockIdx.y) * M * d;

    // ============ Shared memory 布局(N_BUF 份 K/V) ============
    extern __shared__ half smem_ptr[];
    half* s_Q = reinterpret_cast<half*>(smem_ptr);

    half* s_K_buf[N_BUF];
    half* s_V_buf[N_BUF];
    half* sK_ptr = s_Q + 4 * Br * d;
    #pragma unroll
    for (int i = 0; i < N_BUF; ++i) {
        s_K_buf[i] = sK_ptr;
        sK_ptr += Bc * d;
        s_V_buf[i] = sK_ptr;
        sK_ptr += Bc * d;
    }

    half* s_QK = sK_ptr;
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

    // load Q 到 s_Q
    for (int i = (lane_id << 3); i < Br * d; i += (32 << 3)) {
        uint32_t dst_addr = swizzle<3, 3, 4>(i);
        reinterpret_cast<float4*>(s_Q + warp_id * Br * d + dst_addr)[0] = reinterpret_cast<const float4*>(Q + qo_offset + i)[0];
    }
    __syncwarp();

    uint32_t RA[4];
    uint32_t RB[4];

    const int num_iters = M / Bc;

    // ============ 启动阶段:填充前 N_BUF-1 个 buffer 的预取 ============
    // 第 0 组立刻 commit(group 0)
    // 第 1~N_BUF-2 组也立即 commit(让 pipeline 尽可能早启动)
    const int preload = (num_iters < (int)N_BUF - 1) ? num_iters : (int)N_BUF - 1;
    for (int b = 0; b < preload; ++b) {
        int buf_id = b;
        int kv_off = b * Bc;
        half* sK = s_K_buf[buf_id];
        half* sV = s_V_buf[buf_id];
        for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
            uint32_t dst_addr = swizzle<3, 3, 4>(j);
            CP_ASYNC_16(sK + dst_addr, K + kv_offset + kv_off * d + j);
            CP_ASYNC_16(sV + dst_addr, V + kv_offset + kv_off * d + j);
        }
        CP_ASYNC_COMMIT();
    }

    // ============ 主循环 ============
    for (int iter = 0; iter < num_iters; ++iter) {
        const int curr_buf = iter % N_BUF;
        const int next_buf = (iter + 1) % N_BUF;
        const int next_iter = iter + 1;
        const int next_kv_off = next_iter * Bc;

        // ★ 预取下一组(只要还有数据)
        if (next_iter < num_iters) {
            half* sK = s_K_buf[next_buf];
            half* sV = s_V_buf[next_buf];
            for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
                uint32_t dst_addr = swizzle<3, 3, 4>(j);
                CP_ASYNC_16(sK + dst_addr, K + kv_offset + next_kv_off * d + j);
                CP_ASYNC_16(sV + dst_addr, V + kv_offset + next_kv_off * d + j);
            }
            CP_ASYNC_COMMIT();
        }

        // ★ 等待当前 buffer 的 K/V
        //    pipeline 里通常有 N_BUF-1 个未完成 group
        //    wait_group (N_BUF-1) 等所有更早的 group 完成
        CP_ASYNC_WAIT_GROUP(N_BUF - 1);
        __syncthreads();

        // ★ 用 curr_buf 算 QK + softmax + PV + 更新 O
        half* s_K = s_K_buf[curr_buf];
        half* s_V = s_V_buf[curr_buf];

        uint32_t RC[4] = { 0, 0, 0, 0 };

        // ---- 计算 QK^T ----
#pragma unroll
        for (int k = 0; k < d; k += Bd) {
            uint32_t src_addr = k + (lane_id % 16) * d + (lane_id / 16) * 8;
            uint32_t dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_Q + warp_id * Br * d + dst_addr);

            src_addr = k + (lane_id % 8) * d + ((lane_id / 8) % 2) * 8 + (lane_id / 16) * d * 8;
            dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4(RB[0], RB[1], RB[2], RB[3], s_K + dst_addr);

            MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
            MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[2], RB[3], RC[2], RC[3]);
            __syncwarp();
        }

        // ---- 写 s_QK ----
        uint32_t store_smem_qk_m = lane_id / 4;
        uint32_t store_smem_qk_n = (lane_id % 4) * 2;
        uint32_t dst_addr_c0 = swizzle<1, 3, 3>(store_smem_qk_m * Bc + store_smem_qk_n);
        uint32_t dst_addr_c1 = swizzle<1, 3, 3>((store_smem_qk_m + 8) * Bc + store_smem_qk_n);
        uint32_t dst_addr_c2 = swizzle<1, 3, 3>(store_smem_qk_m * Bc + store_smem_qk_n + 8);
        uint32_t dst_addr_c3 = swizzle<1, 3, 3>((store_smem_qk_m + 8) * Bc + store_smem_qk_n + 8);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c0]) = LDST32BITS(RC[0]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c1]) = LDST32BITS(RC[1]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c2]) = LDST32BITS(RC[2]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c3]) = LDST32BITS(RC[3]);
        __syncwarp();

        // ---- Softmax ----
#pragma unroll
        for (int j = 0; j < 8; j++) {
            MD_F tmp_ml = { __half2float(s_QK[warp_id * Br * Bc + j * 32 + lane_id]) * softmax_scale, 1.0f };
            __syncwarp();
            tmp_ml = threadGroupAllReduce<16>(tmp_ml);
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

        // ---- 从 s_S load 到 RA ----
        uint32_t warp_offset = warp_id * Br * Bc;
        uint32_t dst_addr = swizzle<1, 3, 3>((lane_id % 16) * Bc + (lane_id / 16) * 8);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_S + warp_offset + dst_addr);

        // ---- PV ----
        for (int k = 0; k < d; k += Bd) {
            RC[0] = 0; RC[1] = 0; RC[2] = 0; RC[3] = 0;

            uint32_t src_addr = k + (lane_id % 16) * d + (lane_id / 16) * 8;
            uint32_t dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4_T(RB[0], RB[1], RB[2], RB[3], s_V + dst_addr);

            MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
            MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[2], RB[3], RC[2], RC[3]);

            // 写 s_O
            uint32_t store_smem_o_m = lane_id / 4;
            uint32_t store_smem_o_n = (lane_id % 4) * 2;
            uint32_t dst_addr_c0 = swizzle<1, 3, 3>(store_smem_o_m * Bd + store_smem_o_n);
            uint32_t dst_addr_c1 = swizzle<1, 3, 3>((store_smem_o_m + 8) * Bd + store_smem_o_n);
            uint32_t dst_addr_c2 = swizzle<1, 3, 3>(store_smem_o_m * Bd + store_smem_o_n + 8);
            uint32_t dst_addr_c3 = swizzle<1, 3, 3>((store_smem_o_m + 8) * Bd + store_smem_o_n + 8);
            LDST32BITS(s_O[warp_id * Br * Bd + dst_addr_c0]) = LDST32BITS(RC[0]);
            LDST32BITS(s_O[warp_id * Br * Bd + dst_addr_c1]) = LDST32BITS(RC[1]);
            LDST32BITS(s_O[warp_id * Br * Bd + dst_addr_c2]) = LDST32BITS(RC[2]);
            LDST32BITS(s_O[warp_id * Br * Bd + dst_addr_c3]) = LDST32BITS(RC[3]);
            __syncwarp();

            // 更新 O 到 global memory
#pragma unroll
            for (int j = 0; j < 8; j++) {
                uint32_t current_row = j * 2 + (lane_id / 16);
                uint32_t out_idx = qo_offset + current_row * d + k + (lane_id % 16);
                uint32_t s_o_idx = warp_id * Br * Bd + swizzle<1, 3, 3>(current_row * Bd + (lane_id % 16));
                float exp_sub_prev_new_m = __expf(row_ml_prev[warp_id * Br + current_row].m - row_ml_new[warp_id * Br + current_row].m);
                float exp_sub_cur_new_m = __expf(row_ml[warp_id * Br + current_row].m - row_ml_new[warp_id * Br + current_row].m);
                float rlf_i = 1.0f / row_ml_new[warp_id * Br + current_row].d;
                O[out_idx] = __float2half(rlf_i * (row_ml_prev[warp_id * Br + current_row].d * exp_sub_prev_new_m * __half2float(O[out_idx]) +
                    exp_sub_cur_new_m * __half2float(s_O[s_o_idx])));
            }
        }

        // 更新 row_ml_prev
        if (lane_id < Br) {
            row_ml_prev[warp_id * Br + lane_id] = row_ml_new[warp_id * Br + lane_id];
        }
        __syncthreads();
    }
}


// ==================== Launch 函数 ====================
void launch_flash_attn_v2_mma_swizzle_multibuf_kernel(
    const half* Q, const half* K, const half* V, half* O,
    const int batch_size, const int num_head,
    const int N, const int M, const int d,
    cudaStream_t stream) {
    
    // 配置参数
    constexpr uint32_t N_BUF = 3;      // 缓冲区数量
    constexpr uint32_t Bc = 16;
    constexpr uint32_t Br = 16;
    constexpr uint32_t Bd = Bc;
    assert(M % Bc == 0 && N % (4 * Br) == 0 && d % Bc == 0);
    const float softmax_scale = 1.0f / sqrtf((float)d);

    // N_BUF 份 K + N_BUF 份 V
    const int sram_size =
        (4 * Br * 3) * sizeof(MD_F)
        + (4 * Br * d
           + 2 * N_BUF * Bc * d
           + 4 * Br * Bc * 2
           + 4 * Br * Bd) * sizeof(half);

    dim3 grid_dim(div_ceil(N, 4 * Br), num_head, batch_size);
    dim3 block_dim(128);
    flash_attn_v2_mma_swizzle_multistage_kernel<Br, Bc, Bd, N_BUF><<<grid_dim, block_dim, sram_size, stream>>>(Q, K, V, O, N, M, d, softmax_scale);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


/**
 * 双缓冲 + O 累加放 register 版 flash attention v2
 *
 * 在 v5(mma + swizzle)基础上加:
 *   - K/V 双缓冲(cp.async)
 *   - O 累加直接用 RC 寄存器,不再写回 s_O(省 smem + 少 LDS)
 *
 * grid(div_ceil(N, 4 * Br), num_head, batch_size)
 * block(128)
 *
 * 关键点(与原版区别):
 *   - 没有 s_O 数组
 *   - RC_pv 计算后立即用于 O 更新(不再中转 s_O)
 *   - 每个 lane 持有 4 个 half2,通过 mma 的输出 row/col 映射直接更新 O 的对应位置
 */
template <uint32_t Br, uint32_t Bc, uint32_t Bd>
__global__ void flash_attn_v2_mma_swizzle_doublebuffer_accops_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int N, const int M, const int d, const float softmax_scale) {

    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t lane_id = threadIdx.x & 0x1f;

    const uint32_t qo_offset = (blockIdx.z * gridDim.y + blockIdx.y) * N * d + blockIdx.x * 4 * Br * d + warp_id * Br * d;
    const uint32_t kv_offset = (blockIdx.z * gridDim.y + blockIdx.y) * M * d;

    // ============ Shared memory 布局(2 缓冲 K/V,无 s_O)============
    extern __shared__ half smem_ptr[];
    half* s_Q = reinterpret_cast<half*>(smem_ptr);

    // 双 K/V buffer
    half* s_K_buf[2];
    half* s_V_buf[2];
    s_K_buf[0] = s_Q + 4 * Br * d;
    s_V_buf[0] = s_K_buf[0] + Bc * d;
    s_K_buf[1] = s_V_buf[0] + Bc * d;
    s_V_buf[1] = s_K_buf[1] + Bc * d;

    half* s_QK = s_V_buf[1] + Bc * d;
    half* s_S = s_QK + 4 * Br * Bc;
    // ★ 注意:没有 s_O!省 4*Br*Bd half = 1024 half = 2 KB

    MD_F* row_ml_prev = reinterpret_cast<MD_F*>(s_S + 4 * Br * Bc);
    MD_F* row_ml = row_ml_prev + 4 * Br;
    MD_F* row_ml_new = row_ml + 4 * Br;

    // 初始化 ml
#pragma unroll
    for (int k = lane_id; k < Br; k += 32) {
        row_ml_prev[warp_id * Br + k] = { -1e20f, 0.0f };
        row_ml[warp_id * Br + k] = { -1e20f, 0.0f };
    }

    // load Q
    for (int i = (lane_id << 3); i < Br * d; i += (32 << 3)) {
        uint32_t dst_addr = swizzle<3, 3, 4>(i);
        reinterpret_cast<float4*>(s_Q + warp_id * Br * d + dst_addr)[0] =
            reinterpret_cast<const float4*>(Q + qo_offset + i)[0];
    }
    __syncwarp();

    uint32_t RA[4];
    uint32_t RB[4];

    const int num_iters = M / Bc;

    // 预加载第 0 组 K/V
    {
        half* sK = s_K_buf[0];
        half* sV = s_V_buf[0];
        for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
            uint32_t dst_addr = swizzle<3, 3, 4>(j);
            CP_ASYNC_16(sK + dst_addr, K + kv_offset + 0 * d + j);
            CP_ASYNC_16(sV + dst_addr, V + kv_offset + 0 * d + j);
        }
        CP_ASYNC_COMMIT();
    }

    // ============ 主循环 ============
    for (int iter = 0; iter < num_iters; ++iter) {
        const int curr_buf = iter & 1;
        const int next_buf = curr_buf ^ 1;
        const int next_iter = iter + 1;
        const int next_kv_off = next_iter * Bc;

        // 预取下一组 K/V
        if (next_iter < num_iters) {
            half* sK = s_K_buf[next_buf];
            half* sV = s_V_buf[next_buf];
            for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
                uint32_t dst_addr = swizzle<3, 3, 4>(j);
                CP_ASYNC_16(sK + dst_addr, K + kv_offset + next_kv_off * d + j);
                CP_ASYNC_16(sV + dst_addr, V + kv_offset + next_kv_off * d + j);
            }
            CP_ASYNC_COMMIT();
        }

        // 等待当前 buffer
        CP_ASYNC_WAIT_GROUP(1);
        __syncthreads();

        half* s_K = s_K_buf[curr_buf];
        half* s_V = s_V_buf[curr_buf];

        // ============ 计算 QK^T ============
        uint32_t RC[4] = { 0, 0, 0, 0 };
#pragma unroll
        for (int k = 0; k < d; k += Bd) {
            uint32_t src_addr = k + (lane_id % 16) * d + (lane_id / 16) * 8;
            uint32_t dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_Q + warp_id * Br * d + dst_addr);

            src_addr = k + (lane_id % 8) * d + ((lane_id / 8) % 2) * 8 + (lane_id / 16) * d * 8;
            dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4(RB[0], RB[1], RB[2], RB[3], s_K + dst_addr);

            MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
            MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[2], RB[3], RC[2], RC[3]);
            // __syncwarp();
        }

        // 写 s_QK
        uint32_t store_smem_qk_m = lane_id / 4;
        uint32_t store_smem_qk_n = (lane_id % 4) * 2;
        uint32_t dst_addr_c0 = swizzle<1, 3, 3>(store_smem_qk_m * Bc + store_smem_qk_n);
        uint32_t dst_addr_c1 = swizzle<1, 3, 3>((store_smem_qk_m + 8) * Bc + store_smem_qk_n);
        uint32_t dst_addr_c2 = swizzle<1, 3, 3>(store_smem_qk_m * Bc + store_smem_qk_n + 8);
        uint32_t dst_addr_c3 = swizzle<1, 3, 3>((store_smem_qk_m + 8) * Bc + store_smem_qk_n + 8);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c0]) = LDST32BITS(RC[0]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c1]) = LDST32BITS(RC[1]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c2]) = LDST32BITS(RC[2]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c3]) = LDST32BITS(RC[3]);
        __syncwarp();

        // ============ Softmax ============
#pragma unroll
        for (int j = 0; j < 8; j++) {
            MD_F tmp_ml = { __half2float(s_QK[warp_id * Br * Bc + j * 32 + lane_id]) * softmax_scale, 1.0f };
            __syncwarp();
            tmp_ml = threadGroupAllReduce<16>(tmp_ml);
            uint32_t current_row = warp_id * Br + j * 2 + (lane_id / 16);
            
            s_S[current_row * Bc + (lane_id % 16)] = __float2half(
                __expf(__half2float(s_QK[current_row * Bc + (lane_id % 16)]) * softmax_scale - tmp_ml.m));

            if ((lane_id % 16) == 0) {
                row_ml[current_row] = tmp_ml;
                row_ml_new[current_row] = MDFOp()(row_ml_prev[current_row], tmp_ml);
            }
            __syncwarp();
        }

        // 从 s_S load 到 RA
        uint32_t warp_offset = warp_id * Br * Bc;
        uint32_t dst_addr_s = swizzle<1, 3, 3>((lane_id % 16) * Bc + (lane_id / 16) * 8);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_S + warp_offset + dst_addr_s);

        // ============ 计算 PV + ★★★ O 累加优化(无 s_O) ============
        //
        // mma m16n8k16 输出布局(每个 lane t 持 4 个 half2):
        //   - RC_pv[0]:row = t/4,     col = (t%4)*2..+1         (来自 mma 1 的 RD0)
        //   - RC_pv[1]:row = t/4 + 8, col = (t%4)*2..+1         (来自 mma 1 的 RD1)
        //   - RC_pv[2]:row = t/4,     col = (t%4)*2 + 8..+9    (来自 mma 2 的 RD0)
        //   - RC_pv[3]:row = t/4 + 8, col = (t%4)*2 + 8..+9    (来自 mma 2 的 RD1)
        //
        // Br=Bd=16 → 2 次 mma(每次 16×8)拼成 16×16,4 个 RC_pv 各持 2 个 half = 1 个 half2
        // 每个 RC_pv[t] 对应 O 中一对 (col, col+1) 的 2 个元素
        // 4 个 RC_pv 覆盖 4 对 = 8 个元素,每 lane 持有 8 个 half

        for (int k = 0; k < d; k += Bd) {
            uint32_t RC_pv[4] = { 0, 0, 0, 0 };

            // 从 s_V load(trans)
            uint32_t src_addr = k + (lane_id % 16) * d + (lane_id / 16) * 8;
            uint32_t dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4_T(RB[0], RB[1], RB[2], RB[3], s_V + dst_addr);

            // mma 1:RB[0..1] → RC_pv[0..1](col 0~7)
            MMA_M16N8K16_F16F16F16F16(RC_pv[0], RC_pv[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], 0, 0);
            // mma 2:RB[2..3] → RC_pv[2..3](col 8~15)
            MMA_M16N8K16_F16F16F16F16(RC_pv[2], RC_pv[3], RA[0], RA[1], RA[2], RA[3], RB[2], RB[3], 0, 0);

            // ★★★ O 累加核心:基于 RC_pv 直接更新 O(global memory)★★★
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                // t=0: row_low,  col_low
                // t=1: row_high, col_low
                // t=2: row_low,  col_high
                // t=3: row_high, col_high
                uint32_t local_row = (t & 1) ? 8 : 0;          // 0 或 8
                uint32_t local_col = (t >= 2) ? 8 : 0;          // 0 或 8
                uint32_t row_in_warp = (lane_id / 4) + local_row;
                uint32_t col_in_bd = (lane_id % 4) * 2 + local_col;
                // 注意:qo_offset 已经包含了 warp_id * Br * d
                // 所以这里用 row_in_warp * d,而不是 (warp_id * Br + row_in_warp) * d
                uint32_t global_col = k + col_in_bd;

                // 读取 global O(一次 2 个 half)
                half2 prev_O = *(half2*)(&O[qo_offset + row_in_warp * d + global_col]);

                // 读取 softmax 状态(同一行的 m/l 都是共享的)
                float m_prev = row_ml_prev[warp_id * Br + row_in_warp].m;
                float m_cur = row_ml[warp_id * Br + row_in_warp].m;
                float m_new = row_ml_new[warp_id * Br + row_in_warp].m;
                float l_prev = row_ml_prev[warp_id * Br + row_in_warp].d;
                float l_new = row_ml_new[warp_id * Br + row_in_warp].d;

                // online softmax 更新系数
                float exp_prev = __expf(m_prev - m_new);
                float exp_cur = __expf(m_cur - m_new);
                float rlf = 1.0f / l_new;

                // 当前 mma 输出的 half2 值(2 个 half)
                half2 cur = LDST32BITS(RC_pv[t]);

                // 算 2 对 new O 值
                float p0 = __half2float(__low2half(prev_O));
                float p1 = __half2float(__high2half(prev_O));
                float c0 = __half2float(__low2half(cur));
                float c1 = __half2float(__high2half(cur));

                float n0 = rlf * (l_prev * exp_prev * p0 + exp_cur * c0);
                float n1 = rlf * (l_prev * exp_prev * p1 + exp_cur * c1);

                // 写回 global O(2 个 half)
                *(half2*)(&O[qo_offset + row_in_warp * d + global_col]) = __floats2half2_rn(n0, n1);
            }
        }

        // 更新 row_ml_prev
        if (lane_id < Br) {
            row_ml_prev[warp_id * Br + lane_id] = row_ml_new[warp_id * Br + lane_id];
        }
        __syncthreads();
    }
}


// ==================== Launch 函数 ====================
void launch_flash_attn_v2_mma_swizzle_doublebuffer_accops_kernel(
    const half* Q, const half* K, const half* V,
    half* O,
    const int batch_size, const int num_head,
    const int N, const int M, const int d,
    cudaStream_t stream) {
    constexpr uint32_t Bc = 16;
    constexpr uint32_t Br = 16;
    constexpr uint32_t Bd = Bc;
    assert(M % Bc == 0 && N % (4 * Br) == 0 && d % Bc == 0);
    const float softmax_scale = 1.0f / sqrtf((float)d);

    // ★ 没有 s_O,比双缓冲还少 2 KB
    // s_Q:4*Br*d
    // s_K_buf[2]:2*Bc*d
    // s_V_buf[2]:2*Bc*d
    // s_QK:4*Br*Bc
    // s_S:4*Br*Bc
    // (无 s_O)
    // ml 数组:3 * 4*Br * sizeof(MD_F)
    const int sram_size =
        (4 * Br * 3) * sizeof(MD_F)
        + (4 * Br * d
           + 4 * Bc * d
           + 4 * Br * Bc * 2) * sizeof(half);

    dim3 grid_dim(div_ceil(N, 4 * Br), num_head, batch_size);
    dim3 block_dim(128);
    flash_attn_v2_mma_swizzle_doublebuffer_accops_kernel<Br, Bc, Bd>
        <<<grid_dim, block_dim, sram_size, stream>>>(Q, K, V, O, N, M, d, softmax_scale);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


/**
 * Shared-Global 双缓冲版 flash attention v2
 *
 * 在 fa_doublebuffer.cu 基础上加:
 *   - K/V 已经是双缓冲(cp.async)
 *   - 额外:ldmatrix RA/RB 的"寄存器双缓冲"
 *     - 维护 RA_curr, RA_next, RB_curr, RB_next
 *     - 当前 mma 用 curr, 同时 ldmatrix 下一组到 next
 *     - 下一 k 迭代时交换 curr/next
 *
 * 风险:
 *   - ldmatrix 本身是 async 的,可能不显著加速
 *   - 多 8 个 register 的压力(可能 spill)
 *   - 同步逻辑更复杂
 *
 * grid(div_ceil(N, 4 * Br), num_head, batch_size)
 * block(128)
 */
template <uint32_t Br, uint32_t Bc, uint32_t Bd>
__global__ void flash_attn_v2_mma_swizzle_doublebuffer_sg_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int N, const int M, const int d, const float softmax_scale) {

    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t lane_id = threadIdx.x & 0x1f;

    const uint32_t qo_offset = (blockIdx.z * gridDim.y + blockIdx.y) * N * d + blockIdx.x * 4 * Br * d + warp_id * Br * d;
    const uint32_t kv_offset = (blockIdx.z * gridDim.y + blockIdx.y) * M * d;

    // ============ Shared memory 布局(K/V 双缓冲 + 常规 s_QK, s_S, s_O)============
    extern __shared__ half smem_ptr[];
    half* s_Q = reinterpret_cast<half*>(smem_ptr);

    half* s_K_buf[2];
    half* s_V_buf[2];
    s_K_buf[0] = s_Q + 4 * Br * d;
    s_V_buf[0] = s_K_buf[0] + Bc * d;
    s_K_buf[1] = s_V_buf[0] + Bc * d;
    s_V_buf[1] = s_K_buf[1] + Bc * d;

    half* s_QK = s_V_buf[1] + Bc * d;
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

    // load Q
    for (int i = (lane_id << 3); i < Br * d; i += (32 << 3)) {
        uint32_t dst_addr = swizzle<3, 3, 4>(i);
        reinterpret_cast<float4*>(s_Q + warp_id * Br * d + dst_addr)[0] =
            reinterpret_cast<const float4*>(Q + qo_offset + i)[0];
    }
    __syncwarp();

    // ★ SG 双缓冲:2 套 RA,RB 寄存器
    //   curr: 当前 mma 用的
    //   next: 下一 k 迭代预取用的
    uint32_t RA_curr[4];
    uint32_t RA_next[4];
    uint32_t RB_curr[4];
    uint32_t RB_next[4];

    const int num_iters = M / Bc;

    // 预加载第 0 组 K/V
    {
        half* sK = s_K_buf[0];
        half* sV = s_V_buf[0];
        for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
            uint32_t dst_addr = swizzle<3, 3, 4>(j);
            CP_ASYNC_16(sK + dst_addr, K + kv_offset + 0 * d + j);
            CP_ASYNC_16(sV + dst_addr, V + kv_offset + 0 * d + j);
        }
        CP_ASYNC_COMMIT();
    }

    // ============ 主循环 ============
    for (int iter = 0; iter < num_iters; ++iter) {
        const int curr_buf = iter & 1;
        const int next_buf = curr_buf ^ 1;
        const int next_iter = iter + 1;
        const int next_kv_off = next_iter * Bc;

        // 预取下一组 K/V(cp.async 双缓冲)
        if (next_iter < num_iters) {
            half* sK = s_K_buf[next_buf];
            half* sV = s_V_buf[next_buf];
            for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
                uint32_t dst_addr = swizzle<3, 3, 4>(j);
                CP_ASYNC_16(sK + dst_addr, K + kv_offset + next_kv_off * d + j);
                CP_ASYNC_16(sV + dst_addr, V + kv_offset + next_kv_off * d + j);
            }
            CP_ASYNC_COMMIT();
        }

        // 等待当前 buffer
        CP_ASYNC_WAIT_GROUP(1);
        __syncthreads();

        half* s_K = s_K_buf[curr_buf];
        half* s_V = s_V_buf[curr_buf];

        // ============ 计算 QK^T (带 SG 双缓冲) ============
        uint32_t RC[4] = { 0, 0, 0, 0 };

        // ★★★ 预取第一组到 next ★★★
        {
            // ldmatrix RA_next[0..3] (Q 的第 0 列块)
            uint32_t src_addr = (lane_id % 16) * d + (lane_id / 16) * 8;
            uint32_t dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4(RA_next[0], RA_next[1], RA_next[2], RA_next[3], s_Q + warp_id * Br * d + dst_addr);

            // ldmatrix RB_next[0..3] (K 的第 0 列块)
            src_addr = (lane_id % 8) * d + ((lane_id / 8) % 2) * 8 + (lane_id / 16) * d * 8;
            dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4(RB_next[0], RB_next[1], RB_next[2], RB_next[3], s_K + dst_addr);
        }

#pragma unroll
        for (int k = 0; k < d; k += Bd) {
            // ★ 切换:next → curr
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                RA_curr[r] = RA_next[r];
                RB_curr[r] = RB_next[r];
            }

            // ★ 预取下一组(如果还有)
            if (k + Bd < d) {
                uint32_t src_addr = (k + Bd) + (lane_id % 16) * d + (lane_id / 16) * 8;
                uint32_t dst_addr = swizzle<3, 3, 4>(src_addr);
                LDMATRIX_X4(RA_next[0], RA_next[1], RA_next[2], RA_next[3], s_Q + warp_id * Br * d + dst_addr);

                src_addr = (k + Bd) + (lane_id % 8) * d + ((lane_id / 8) % 2) * 8 + (lane_id / 16) * d * 8;
                dst_addr = swizzle<3, 3, 4>(src_addr);
                LDMATRIX_X4(RB_next[0], RB_next[1], RB_next[2], RB_next[3], s_K + dst_addr);
            }

            // 用 curr 算 mma
            MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA_curr[0], RA_curr[1], RA_curr[2], RA_curr[3], RB_curr[0], RB_curr[1], RC[0], RC[1]);
            MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA_curr[0], RA_curr[1], RA_curr[2], RA_curr[3], RB_curr[2], RB_curr[3], RC[2], RC[3]);
            // __syncwarp();
        }

        // 写 s_QK
        uint32_t store_smem_qk_m = lane_id / 4;
        uint32_t store_smem_qk_n = (lane_id % 4) * 2;
        uint32_t dst_addr_c0 = swizzle<1, 3, 3>(store_smem_qk_m * Bc + store_smem_qk_n);
        uint32_t dst_addr_c1 = swizzle<1, 3, 3>((store_smem_qk_m + 8) * Bc + store_smem_qk_n);
        uint32_t dst_addr_c2 = swizzle<1, 3, 3>(store_smem_qk_m * Bc + store_smem_qk_n + 8);
        uint32_t dst_addr_c3 = swizzle<1, 3, 3>((store_smem_qk_m + 8) * Bc + store_smem_qk_n + 8);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c0]) = LDST32BITS(RC[0]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c1]) = LDST32BITS(RC[1]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c2]) = LDST32BITS(RC[2]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c3]) = LDST32BITS(RC[3]);
        __syncwarp();

        // ============ Softmax ============
#pragma unroll
        for (int j = 0; j < 8; j++) {
            MD_F tmp_ml = { __half2float(s_QK[warp_id * Br * Bc + j * 32 + lane_id]) * softmax_scale, 1.0f };
            __syncwarp();
            tmp_ml = threadGroupAllReduce<16>(tmp_ml);
            uint32_t current_row = warp_id * Br + j * 2 + (lane_id / 16);
            s_S[current_row * Bc + (lane_id % 16)] = __float2half(
                __expf(__half2float(s_QK[current_row * Bc + (lane_id % 16)]) * softmax_scale - tmp_ml.m));

            if ((lane_id % 16) == 0) {
                row_ml[current_row] = tmp_ml;
                row_ml_new[current_row] = MDFOp()(row_ml_prev[current_row], tmp_ml);
            }
            __syncwarp();
        }

        // 从 s_S load 到 RA_curr
        uint32_t warp_offset = warp_id * Br * Bc;
        uint32_t dst_addr_s = swizzle<1, 3, 3>((lane_id % 16) * Bc + (lane_id / 16) * 8);
        LDMATRIX_X4(RA_curr[0], RA_curr[1], RA_curr[2], RA_curr[3], s_S + warp_offset + dst_addr_s);

        // ============ 计算 PV(常规路径) ============
        for (int k = 0; k < d; k += Bd) {
            uint32_t RC_pv[4] = { 0, 0, 0, 0 };

            uint32_t src_addr = k + (lane_id % 16) * d + (lane_id / 16) * 8;
            uint32_t dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4_T(RB_curr[0], RB_curr[1], RB_curr[2], RB_curr[3], s_V + dst_addr);

            MMA_M16N8K16_F16F16F16F16(RC_pv[0], RC_pv[1], RA_curr[0], RA_curr[1], RA_curr[2], RA_curr[3], RB_curr[0], RB_curr[1], 0, 0);
            MMA_M16N8K16_F16F16F16F16(RC_pv[2], RC_pv[3], RA_curr[0], RA_curr[1], RA_curr[2], RA_curr[3], RB_curr[2], RB_curr[3], 0, 0);

            // 写 s_O
            uint32_t store_smem_o_m = lane_id / 4;
            uint32_t store_smem_o_n = (lane_id % 4) * 2;
            uint32_t dst_addr_c0 = swizzle<1, 3, 3>(store_smem_o_m * Bd + store_smem_o_n);
            uint32_t dst_addr_c1 = swizzle<1, 3, 3>((store_smem_o_m + 8) * Bd + store_smem_o_n);
            uint32_t dst_addr_c2 = swizzle<1, 3, 3>(store_smem_o_m * Bd + store_smem_o_n + 8);
            uint32_t dst_addr_c3 = swizzle<1, 3, 3>((store_smem_o_m + 8) * Bd + store_smem_o_n + 8);
            LDST32BITS(s_O[warp_id * Br * Bd + dst_addr_c0]) = LDST32BITS(RC_pv[0]);
            LDST32BITS(s_O[warp_id * Br * Bd + dst_addr_c1]) = LDST32BITS(RC_pv[1]);
            LDST32BITS(s_O[warp_id * Br * Bd + dst_addr_c2]) = LDST32BITS(RC_pv[2]);
            LDST32BITS(s_O[warp_id * Br * Bd + dst_addr_c3]) = LDST32BITS(RC_pv[3]);
            __syncwarp();

            // 更新 O 到 global memory
#pragma unroll
            for (int j = 0; j < 8; j++) {
                uint32_t current_row = j * 2 + (lane_id / 16);
                uint32_t out_idx = qo_offset + current_row * d + k + (lane_id % 16);
                uint32_t s_o_idx = warp_id * Br * Bd + swizzle<1, 3, 3>(current_row * Bd + (lane_id % 16));
                float exp_sub_prev_new_m = __expf(row_ml_prev[warp_id * Br + current_row].m - row_ml_new[warp_id * Br + current_row].m);
                float exp_sub_cur_new_m = __expf(row_ml[warp_id * Br + current_row].m - row_ml_new[warp_id * Br + current_row].m);
                float rlf_i = 1.0f / row_ml_new[warp_id * Br + current_row].d;
                O[out_idx] = __float2half(rlf_i * (row_ml_prev[warp_id * Br + current_row].d * exp_sub_prev_new_m * __half2float(O[out_idx]) +
                    exp_sub_cur_new_m * __half2float(s_O[s_o_idx])));
            }
        }

        // 更新 row_ml_prev
        if (lane_id < Br) {
            row_ml_prev[warp_id * Br + lane_id] = row_ml_new[warp_id * Br + lane_id];
        }
        __syncthreads();
    }
}


// ==================== Launch 函数 ====================
void launch_flash_attn_v2_mma_swizzle_doublebuffer_sg_kernel(
    const half* Q, const half* K, const half* V,
    half* O,
    const int batch_size, const int num_head,
    const int N, const int M, const int d,
    cudaStream_t stream) {
    constexpr uint32_t Bc = 16;
    constexpr uint32_t Br = 16;
    constexpr uint32_t Bd = Bc;
    assert(M % Bc == 0 && N % (4 * Br) == 0 && d % Bc == 0);
    const float softmax_scale = 1.0f / sqrtf((float)d);

    const int sram_size =
        (4 * Br * 3) * sizeof(MD_F)
        + (4 * Br * d
           + 2 * 2 * Bc * d
           + 4 * Br * Bc * 2
           + 4 * Br * Bd) * sizeof(half);

    dim3 grid_dim(div_ceil(N, 4 * Br), num_head, batch_size);
    dim3 block_dim(128);
    flash_attn_v2_mma_swizzle_doublebuffer_sg_kernel<Br, Bc, Bd>
        <<<grid_dim, block_dim, sram_size, stream>>>(Q, K, V, O, N, M, d, softmax_scale);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}



/**
 * 双缓冲 + O 累加 + SG 双缓冲版 flash attention v2
 *
 * 在 flash_attn_v2_mma_swizzle_doublebuffer_accops_kernel 基础上加:
 *   - K/V 双缓冲(cp.async)
 *   - O 累加放 register(无 s_O)
 *   - SG 双缓冲:ldmatrix RA/RB 的"寄存器双缓冲"
 *     - 在 QK^T 阶段维护 RA_curr/RA_next/RB_curr/RB_next
 *     - 当前 mma 用 curr, 同时 ldmatrix 下一组到 next
 *     - 下一 k 迭代时交换 curr/next
 *
 * 风险:
 *   - SG 双缓冲在之前的 fa_doublebuffer_sg_kernel 中实测无效(ldmatrix 本身已经 hardware-managed)
 *   - 叠加 O 累加可能 register pressure 更大
 *   - 但:换角度想,SG 可能隐藏 ldmatrix 延迟,让 mma pipeline 更密
 *
 * grid(div_ceil(N, 4 * Br), num_head, batch_size)
 * block(128)
 */
template <uint32_t Br, uint32_t Bc, uint32_t Bd>
__global__ void flash_attn_v2_mma_swizzle_doublebuffer_accops_sg_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int N, const int M, const int d, const float softmax_scale) {

    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t lane_id = threadIdx.x & 0x1f;

    const uint32_t qo_offset = (blockIdx.z * gridDim.y + blockIdx.y) * N * d + blockIdx.x * 4 * Br * d + warp_id * Br * d;
    const uint32_t kv_offset = (blockIdx.z * gridDim.y + blockIdx.y) * M * d;

    // ============ Shared memory 布局(2 缓冲 K/V,无 s_O)============
    extern __shared__ half smem_ptr[];
    half* s_Q = reinterpret_cast<half*>(smem_ptr);

    // 双 K/V buffer
    half* s_K_buf[2];
    half* s_V_buf[2];
    s_K_buf[0] = s_Q + 4 * Br * d;
    s_V_buf[0] = s_K_buf[0] + Bc * d;
    s_K_buf[1] = s_V_buf[0] + Bc * d;
    s_V_buf[1] = s_K_buf[1] + Bc * d;

    half* s_QK = s_V_buf[1] + Bc * d;
    half* s_S = s_QK + 4 * Br * Bc;
    // ★ 没有 s_O

    MD_F* row_ml_prev = reinterpret_cast<MD_F*>(s_S + 4 * Br * Bc);
    MD_F* row_ml = row_ml_prev + 4 * Br;
    MD_F* row_ml_new = row_ml + 4 * Br;

    // 初始化 ml
#pragma unroll
    for (int k = lane_id; k < Br; k += 32) {
        row_ml_prev[warp_id * Br + k] = { -1e20f, 0.0f };
        row_ml[warp_id * Br + k] = { -1e20f, 0.0f };
    }

    // load Q
    for (int i = (lane_id << 3); i < Br * d; i += (32 << 3)) {
        uint32_t dst_addr = swizzle<3, 3, 4>(i);
        reinterpret_cast<float4*>(s_Q + warp_id * Br * d + dst_addr)[0] =
            reinterpret_cast<const float4*>(Q + qo_offset + i)[0];
    }
    __syncwarp();

    // ★ SG 双缓冲:2 套 RA,RB 寄存器(只在 QK^T 阶段使用)
    uint32_t RA_curr[4];
    uint32_t RA_next[4];
    uint32_t RB_curr[4];
    uint32_t RB_next[4];

    const int num_iters = M / Bc;

    // 预加载第 0 组 K/V
    {
        half* sK = s_K_buf[0];
        half* sV = s_V_buf[0];
        for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
            uint32_t dst_addr = swizzle<3, 3, 4>(j);
            CP_ASYNC_16(sK + dst_addr, K + kv_offset + 0 * d + j);
            CP_ASYNC_16(sV + dst_addr, V + kv_offset + 0 * d + j);
        }
        CP_ASYNC_COMMIT();
    }

    // ============ 主循环 ============
    for (int iter = 0; iter < num_iters; ++iter) {
        const int curr_buf = iter & 1;
        const int next_buf = curr_buf ^ 1;
        const int next_iter = iter + 1;
        const int next_kv_off = next_iter * Bc;

        // 预取下一组 K/V
        if (next_iter < num_iters) {
            half* sK = s_K_buf[next_buf];
            half* sV = s_V_buf[next_buf];
            for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
                uint32_t dst_addr = swizzle<3, 3, 4>(j);
                CP_ASYNC_16(sK + dst_addr, K + kv_offset + next_kv_off * d + j);
                CP_ASYNC_16(sV + dst_addr, V + kv_offset + next_kv_off * d + j);
            }
            CP_ASYNC_COMMIT();
        }

        // 等待当前 buffer
        CP_ASYNC_WAIT_GROUP(1);
        __syncthreads();

        half* s_K = s_K_buf[curr_buf];
        half* s_V = s_V_buf[curr_buf];

        // ============ 计算 QK^T (SG 双缓冲版)============
        uint32_t RC[4] = { 0, 0, 0, 0 };

        // ★ 预取第一组到 RA_next / RB_next
        {
            // Q 的 lane base
            uint32_t src_addr = (lane_id % 16) * d + (lane_id / 16) * 8;
            uint32_t dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4(RA_next[0], RA_next[1], RA_next[2], RA_next[3], s_Q + warp_id * Br * d + dst_addr);

            // K 的 lane base
            src_addr = (lane_id % 8) * d + ((lane_id / 8) % 2) * 8 + (lane_id / 16) * d * 8;
            dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4(RB_next[0], RB_next[1], RB_next[2], RB_next[3], s_K + dst_addr);
        }

        // k 维度迭代,每次 Bd=16
        // d=64 → 4 次迭代,d=128 → 8 次迭代
#pragma unroll
        for (int k = 0; k < d; k += Bd) {
            // ★ 切换:next → curr(8 个 register 拷贝,编译器可并行)
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                RA_curr[r] = RA_next[r];
                RB_curr[r] = RB_next[r];
            }

            // ★ 预取下一组到 next(如果还有)
            if (k + Bd < d) {
                // Q 的下一组
                uint32_t src_addr = (k + Bd) + (lane_id % 16) * d + (lane_id / 16) * 8;
                uint32_t dst_addr = swizzle<3, 3, 4>(src_addr);
                LDMATRIX_X4(RA_next[0], RA_next[1], RA_next[2], RA_next[3], s_Q + warp_id * Br * d + dst_addr);

                // K 的下一组
                src_addr = (k + Bd) + (lane_id % 8) * d + ((lane_id / 8) % 2) * 8 + (lane_id / 16) * d * 8;
                dst_addr = swizzle<3, 3, 4>(src_addr);
                LDMATRIX_X4(RB_next[0], RB_next[1], RB_next[2], RB_next[3], s_K + dst_addr);
            }

            // ★ 用 curr 算 mma
            MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA_curr[0], RA_curr[1], RA_curr[2], RA_curr[3], RB_curr[0], RB_curr[1], RC[0], RC[1]);
            MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA_curr[0], RA_curr[1], RA_curr[2], RA_curr[3], RB_curr[2], RB_curr[3], RC[2], RC[3]);
        }

        // 写 s_QK
        uint32_t store_smem_qk_m = lane_id / 4;
        uint32_t store_smem_qk_n = (lane_id % 4) * 2;
        uint32_t dst_addr_c0 = swizzle<1, 3, 3>(store_smem_qk_m * Bc + store_smem_qk_n);
        uint32_t dst_addr_c1 = swizzle<1, 3, 3>((store_smem_qk_m + 8) * Bc + store_smem_qk_n);
        uint32_t dst_addr_c2 = swizzle<1, 3, 3>(store_smem_qk_m * Bc + store_smem_qk_n + 8);
        uint32_t dst_addr_c3 = swizzle<1, 3, 3>((store_smem_qk_m + 8) * Bc + store_smem_qk_n + 8);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c0]) = LDST32BITS(RC[0]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c1]) = LDST32BITS(RC[1]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c2]) = LDST32BITS(RC[2]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c3]) = LDST32BITS(RC[3]);
        __syncwarp();

        // ============ Softmax ============
#pragma unroll
        for (int j = 0; j < 8; j++) {
            MD_F tmp_ml = { __half2float(s_QK[warp_id * Br * Bc + j * 32 + lane_id]) * softmax_scale, 1.0f };
            __syncwarp();
            tmp_ml = threadGroupAllReduce<16>(tmp_ml);
            uint32_t current_row = warp_id * Br + j * 2 + (lane_id / 16);

            // 用 tmp_ml.m 直接算 s_S(省 1 次 sync)
            s_S[current_row * Bc + (lane_id % 16)] = __float2half(
                __expf(__half2float(s_QK[current_row * Bc + (lane_id % 16)]) * softmax_scale - tmp_ml.m));

            if ((lane_id % 16) == 0) {
                row_ml[current_row] = tmp_ml;
                row_ml_new[current_row] = MDFOp()(row_ml_prev[current_row], tmp_ml);
            }
            __syncwarp();
        }

        // 从 s_S load 到 RA_curr(直接用,不需要双缓冲因为 PV 只读一次)
        // 用 RA_curr 节省 4 个 register,反正这里没 SG
        uint32_t warp_offset = warp_id * Br * Bc;
        uint32_t dst_addr_s = swizzle<1, 3, 3>((lane_id % 16) * Bc + (lane_id / 16) * 8);
        LDMATRIX_X4(RA_curr[0], RA_curr[1], RA_curr[2], RA_curr[3], s_S + warp_offset + dst_addr_s);

        // ============ 计算 PV + O 累加(无 s_O)============
        for (int k = 0; k < d; k += Bd) {
            uint32_t RC_pv[4] = { 0, 0, 0, 0 };

            // 从 s_V load(trans)
            uint32_t src_addr = k + (lane_id % 16) * d + (lane_id / 16) * 8;
            uint32_t dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4_T(RB_curr[0], RB_curr[1], RB_curr[2], RB_curr[3], s_V + dst_addr);

            // mma 1:RB[0..1] → RC_pv[0..1](col 0~7)
            MMA_M16N8K16_F16F16F16F16(RC_pv[0], RC_pv[1], RA_curr[0], RA_curr[1], RA_curr[2], RA_curr[3], RB_curr[0], RB_curr[1], 0, 0);
            // mma 2:RB[2..3] → RC_pv[2..3](col 8~15)
            MMA_M16N8K16_F16F16F16F16(RC_pv[2], RC_pv[3], RA_curr[0], RA_curr[1], RA_curr[2], RA_curr[3], RB_curr[2], RB_curr[3], 0, 0);

            // ★★★ O 累加核心:基于 RC_pv 直接更新 O(global memory)★★★
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                uint32_t local_row = (t & 1) ? 8 : 0;
                uint32_t local_col = (t >= 2) ? 8 : 0;
                uint32_t row_in_warp = (lane_id / 4) + local_row;
                uint32_t col_in_bd = (lane_id % 4) * 2 + local_col;
                uint32_t global_col = k + col_in_bd;

                half2 prev_O = *(half2*)(&O[qo_offset + row_in_warp * d + global_col]);

                float m_prev = row_ml_prev[warp_id * Br + row_in_warp].m;
                float m_cur = row_ml[warp_id * Br + row_in_warp].m;
                float m_new = row_ml_new[warp_id * Br + row_in_warp].m;
                float l_prev = row_ml_prev[warp_id * Br + row_in_warp].d;
                float l_new = row_ml_new[warp_id * Br + row_in_warp].d;

                float exp_prev = __expf(m_prev - m_new);
                float exp_cur = __expf(m_cur - m_new);
                float rlf = 1.0f / l_new;

                half2 cur = LDST32BITS(RC_pv[t]);

                float p0 = __half2float(__low2half(prev_O));
                float p1 = __half2float(__high2half(prev_O));
                float c0 = __half2float(__low2half(cur));
                float c1 = __half2float(__high2half(cur));

                float n0 = rlf * (l_prev * exp_prev * p0 + exp_cur * c0);
                float n1 = rlf * (l_prev * exp_prev * p1 + exp_cur * c1);

                *(half2*)(&O[qo_offset + row_in_warp * d + global_col]) = __floats2half2_rn(n0, n1);
            }
        }

        // 更新 row_ml_prev
        if (lane_id < Br) {
            row_ml_prev[warp_id * Br + lane_id] = row_ml_new[warp_id * Br + lane_id];
        }
        __syncthreads();
    }
}


// ==================== Launch 函数 ====================
void launch_flash_attn_v2_mma_swizzle_doublebuffer_accops_sg_kernel(
    const half* Q, const half* K, const half* V,
    half* O,
    const int batch_size, const int num_head,
    const int N, const int M, const int d,
    cudaStream_t stream) {
    constexpr uint32_t Bc = 16;
    constexpr uint32_t Br = 16;
    constexpr uint32_t Bd = Bc;
    assert(M % Bc == 0 && N % (4 * Br) == 0 && d % Bc == 0);
    const float softmax_scale = 1.0f / sqrtf((float)d);

    // sram 与 accops 版相同(无 s_O)
    const int sram_size =
        (4 * Br * 3) * sizeof(MD_F)
        + (4 * Br * d
           + 4 * Bc * d
           + 4 * Br * Bc * 2) * sizeof(half);

    dim3 grid_dim(div_ceil(N, 4 * Br), num_head, batch_size);
    dim3 block_dim(128);
    flash_attn_v2_mma_swizzle_doublebuffer_accops_sg_kernel<Br, Bc, Bd>
        <<<grid_dim, block_dim, sram_size, stream>>>(Q, K, V, O, N, M, d, softmax_scale);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


/**
 * 双缓冲 + O 暂存 smem + 最后一次写 O 版 flash attention v2
 *
 * 核心思路:
 *   - 引入 s_O 暂存 O(消除 main iter 内的 global O 读写)
 *   - 每次 main iter 开始:
 *     * 第一次:写 0 到 s_O
 *     * 否则:LDG 读 global O → s_O
 *   - PV 阶段:每次 k_seg 都从 s_O 读 prev_O,写 s_O(不写 global)
 *   - main iter 结束:STG s_O → global O
 *
 * 性能优势:
 *   - 消除 main iter 内的 global O 流量
 *   - 主循环内:4 个 t × 0 LDG + 0 STG(只写 sram)
 *   - 主循环外:1 LDG(读 O) + 1 STG(写 O)per lane
 *   - 减少约 75% O 流量
 *
 * sram 预算(Br=Bc=Bd=16):
 *   d=64: s_O = 4 KB, 总 15.5 KB < 46 KB ✓
 *   d=128: s_O = 8 KB, 总 19.5 KB < 46 KB ✓
 *
 * grid(div_ceil(N, 4 * Br), num_head, batch_size)
 * block(128)
 */
template <uint32_t Br, uint32_t Bc, uint32_t Bd>
__global__ void flash_attn_v2_mma_swizzle_doublebuffer_smemO_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int N, const int M, const int d, const float softmax_scale) {

    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t lane_id = threadIdx.x & 0x1f;

    const uint32_t qo_offset = (blockIdx.z * gridDim.y + blockIdx.y) * N * d + blockIdx.x * 4 * Br * d + warp_id * Br * d;
    const uint32_t kv_offset = (blockIdx.z * gridDim.y + blockIdx.y) * M * d;

    // ============ Shared memory 布局(2 缓冲 K/V + s_O)============
    extern __shared__ half smem_ptr[];
    half* s_Q = reinterpret_cast<half*>(smem_ptr);

    half* s_K_buf[2];
    half* s_V_buf[2];
    s_K_buf[0] = s_Q + 4 * Br * d;
    s_V_buf[0] = s_K_buf[0] + Bc * d;
    s_K_buf[1] = s_V_buf[0] + Bc * d;
    s_V_buf[1] = s_K_buf[1] + Bc * d;

    half* s_QK = s_V_buf[1] + Bc * d;
    half* s_S = s_QK + 4 * Br * Bc;

    // ★ 新增 s_O 暂存
    half* s_O = s_S + 4 * Br * Bc;

    MD_F* row_ml_prev = reinterpret_cast<MD_F*>(s_O + 4 * Br * d);
    MD_F* row_ml = row_ml_prev + 4 * Br;
    MD_F* row_ml_new = row_ml + 4 * Br;

    // 初始化 ml
#pragma unroll
    for (int k = lane_id; k < Br; k += 32) {
        row_ml_prev[warp_id * Br + k] = { -1e20f, 0.0f };
        row_ml[warp_id * Br + k] = { -1e20f, 0.0f };
    }

    // load Q
    for (int i = (lane_id << 3); i < Br * d; i += (32 << 3)) {
        uint32_t dst_addr = swizzle<3, 3, 4>(i);
        reinterpret_cast<float4*>(s_Q + warp_id * Br * d + dst_addr)[0] =
            reinterpret_cast<const float4*>(Q + qo_offset + i)[0];
    }

    // init s_O by 0
    for (int i = (lane_id << 3); i < Br * d; i += (32 << 3)) {
        *reinterpret_cast<float4*>(s_O + warp_id * Br * d + i) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __syncwarp();

    // SG 双缓冲变量(本版本不用 SG,但保留 RA/RB 复用结构)
    uint32_t RA_curr[4];
    uint32_t RB_curr[4];

    const int num_iters = M / Bc;

    // 预加载第 0 组 K/V
    {
        half* sK = s_K_buf[0];
        half* sV = s_V_buf[0];
        for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
            uint32_t dst_addr = swizzle<3, 3, 4>(j);
            CP_ASYNC_16(sK + dst_addr, K + kv_offset + 0 * d + j);
            CP_ASYNC_16(sV + dst_addr, V + kv_offset + 0 * d + j);
        }
        CP_ASYNC_COMMIT();
    }

    // ============ 主循环 ============
    for (int iter = 0; iter < num_iters; ++iter) {
        const int curr_buf = iter & 1;
        const int next_buf = curr_buf ^ 1;
        const int next_iter = iter + 1;
        const int next_kv_off = next_iter * Bc;

        // 预取下一组 K/V
        if (next_iter < num_iters) {
            half* sK = s_K_buf[next_buf];
            half* sV = s_V_buf[next_buf];
            for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
                uint32_t dst_addr = swizzle<3, 3, 4>(j);
                CP_ASYNC_16(sK + dst_addr, K + kv_offset + next_kv_off * d + j);
                CP_ASYNC_16(sV + dst_addr, V + kv_offset + next_kv_off * d + j);
            }
            CP_ASYNC_COMMIT();
        }

        CP_ASYNC_WAIT_GROUP(1);
        __syncthreads();

        half* s_K = s_K_buf[curr_buf];
        half* s_V = s_V_buf[curr_buf];

        // ============ 计算 QK^T (无 SG 双缓冲)============
        uint32_t RC[4] = { 0, 0, 0, 0 };

        {
            uint32_t src_addr = (lane_id % 16) * d + (lane_id / 16) * 8;
            uint32_t dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4(RA_curr[0], RA_curr[1], RA_curr[2], RA_curr[3], s_Q + warp_id * Br * d + dst_addr);

            src_addr = (lane_id % 8) * d + ((lane_id / 8) % 2) * 8 + (lane_id / 16) * d * 8;
            dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4(RB_curr[0], RB_curr[1], RB_curr[2], RB_curr[3], s_K + dst_addr);
        }

#pragma unroll
        for (int k = 0; k < d; k += Bd) {
            MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA_curr[0], RA_curr[1], RA_curr[2], RA_curr[3], RB_curr[0], RB_curr[1], RC[0], RC[1]);
            MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA_curr[0], RA_curr[1], RA_curr[2], RA_curr[3], RB_curr[2], RB_curr[3], RC[2], RC[3]);

            if (k + Bd < d) {
                uint32_t src_addr = (k + Bd) + (lane_id % 16) * d + (lane_id / 16) * 8;
                uint32_t dst_addr = swizzle<3, 3, 4>(src_addr);
                LDMATRIX_X4(RA_curr[0], RA_curr[1], RA_curr[2], RA_curr[3], s_Q + warp_id * Br * d + dst_addr);

                src_addr = (k + Bd) + (lane_id % 8) * d + ((lane_id / 8) % 2) * 8 + (lane_id / 16) * d * 8;
                dst_addr = swizzle<3, 3, 4>(src_addr);
                LDMATRIX_X4(RB_curr[0], RB_curr[1], RB_curr[2], RB_curr[3], s_K + dst_addr);
            }
        }

        // 写 s_QK
        uint32_t store_smem_qk_m = lane_id / 4;
        uint32_t store_smem_qk_n = (lane_id % 4) * 2;
        uint32_t dst_addr_c0 = swizzle<1, 3, 3>(store_smem_qk_m * Bc + store_smem_qk_n);
        uint32_t dst_addr_c1 = swizzle<1, 3, 3>((store_smem_qk_m + 8) * Bc + store_smem_qk_n);
        uint32_t dst_addr_c2 = swizzle<1, 3, 3>(store_smem_qk_m * Bc + store_smem_qk_n + 8);
        uint32_t dst_addr_c3 = swizzle<1, 3, 3>((store_smem_qk_m + 8) * Bc + store_smem_qk_n + 8);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c0]) = LDST32BITS(RC[0]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c1]) = LDST32BITS(RC[1]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c2]) = LDST32BITS(RC[2]);
        LDST32BITS(s_QK[warp_id * Br * Bc + dst_addr_c3]) = LDST32BITS(RC[3]);
        __syncwarp();

        // ============ Softmax ============
#pragma unroll
        for (int j = 0; j < 8; j++) {
            MD_F tmp_ml = { __half2float(s_QK[warp_id * Br * Bc + j * 32 + lane_id]) * softmax_scale, 1.0f };
            __syncwarp();
            tmp_ml = threadGroupAllReduce<16>(tmp_ml);
            uint32_t current_row = warp_id * Br + j * 2 + (lane_id / 16);

            s_S[current_row * Bc + (lane_id % 16)] = __float2half(
                __expf(__half2float(s_QK[current_row * Bc + (lane_id % 16)]) * softmax_scale - tmp_ml.m));

            if ((lane_id % 16) == 0) {
                row_ml[current_row] = tmp_ml;
                row_ml_new[current_row] = MDFOp()(row_ml_prev[current_row], tmp_ml);
            }
            __syncwarp();
        }

        // 从 s_S load 到 RA_curr
        uint32_t warp_offset = warp_id * Br * Bc;
        uint32_t dst_addr_s = swizzle<1, 3, 3>((lane_id % 16) * Bc + (lane_id / 16) * 8);
        LDMATRIX_X4(RA_curr[0], RA_curr[1], RA_curr[2], RA_curr[3], s_S + warp_offset + dst_addr_s);

        // ============ 计算 PV + s_O 更新(不写 global)============
        // 关键:每次 k_seg 从 s_O 读,写 s_O(不写 global)
#pragma unroll
        for (int k = 0; k < d; k += Bd) {
            uint32_t RC_pv[4] = { 0, 0, 0, 0 };

            // 从 s_V load(trans)
            uint32_t src_addr = k + (lane_id % 16) * d + (lane_id / 16) * 8;
            uint32_t dst_addr = swizzle<3, 3, 4>(src_addr);
            LDMATRIX_X4_T(RB_curr[0], RB_curr[1], RB_curr[2], RB_curr[3], s_V + dst_addr);

            // mma
            MMA_M16N8K16_F16F16F16F16(RC_pv[0], RC_pv[1], RA_curr[0], RA_curr[1], RA_curr[2], RA_curr[3], RB_curr[0], RB_curr[1], 0, 0);
            MMA_M16N8K16_F16F16F16F16(RC_pv[2], RC_pv[3], RA_curr[0], RA_curr[1], RA_curr[2], RA_curr[3], RB_curr[2], RB_curr[3], 0, 0);

            // ★ 4 t 循环:从 s_O 读,写 s_O(不写 global)
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                uint32_t local_row = (t & 1) ? 8 : 0;
                uint32_t local_col = (t >= 2) ? 8 : 0;
                uint32_t row_in_warp = (lane_id / 4) + local_row;
                uint32_t col_in_bd = (lane_id % 4) * 2 + local_col;
                uint32_t s_col = k + col_in_bd;  // 在 s_O 中的 col 索引

                // ★ 从 s_O 读 prev_O(sram,快)
                uint32_t s_addr = swizzle<3, 3, 4>(row_in_warp * d + s_col);
                half2 prev_O = *(half2*)(&s_O[warp_id * Br * d + s_addr]);

                // 读 m, l
                float m_prev = row_ml_prev[warp_id * Br + row_in_warp].m;
                float m_cur = row_ml[warp_id * Br + row_in_warp].m;
                float m_new = row_ml_new[warp_id * Br + row_in_warp].m;
                float l_prev = row_ml_prev[warp_id * Br + row_in_warp].d;
                float l_new = row_ml_new[warp_id * Br + row_in_warp].d;

                float exp_prev = __expf(m_prev - m_new);
                float exp_cur = __expf(m_cur - m_new);
                float rlf = 1.0f / l_new;

                half2 cur = LDST32BITS(RC_pv[t]);
                float p0 = __half2float(__low2half(prev_O));
                float p1 = __half2float(__high2half(prev_O));
                float c0 = __half2float(__low2half(cur));
                float c1 = __half2float(__high2half(cur));

                float n0 = rlf * (l_prev * exp_prev * p0 + exp_cur * c0);
                float n1 = rlf * (l_prev * exp_prev * p1 + exp_cur * c1);

                // ★ 写 s_O(不写 global!)
                *(half2*)(&s_O[warp_id * Br * d + s_addr]) = __floats2half2_rn(n0, n1);
            }
        }

        // 更新 row_ml_prev
        if (lane_id < Br) {
            row_ml_prev[warp_id * Br + lane_id] = row_ml_new[warp_id * Br + lane_id];
        }
        __syncthreads();
    }

    // ★ 所有 main iter 结束后:写 1 次 s_O → global O
    for (int i = (lane_id << 3); i < Br * d; i += (32 << 3)) {
        uint32_t dst_addr = swizzle<3, 3, 4>(i);
        *reinterpret_cast<float4*>(O + qo_offset + i) =
            *reinterpret_cast<const float4*>(s_O + warp_id * Br * d + dst_addr);
    }
}


// ==================== Launch 函数 ====================
void launch_flash_attn_v2_mma_swizzle_doublebuffer_smemO_kernel(
    const half* Q, const half* K, const half* V,
    half* O,
    const int batch_size, const int num_head,
    const int N, const int M, const int d,
    cudaStream_t stream) {
    constexpr uint32_t Bc = 16;
    constexpr uint32_t Br = 16;
    constexpr uint32_t Bd = Bc;
    assert(M % Bc == 0 && N % (4 * Br) == 0 && d % Bc == 0);
    const float softmax_scale = 1.0f / sqrtf((float)d);

    // sram 包含 s_O
    const int sram_size =
        (4 * Br * 3) * sizeof(MD_F)
        + (4 * Br * d           // s_Q
           + 4 * Bc * d         // s_K_buf[2] + s_V_buf[2] = 4*Bc*d
           + 4 * Br * Bc * 2    // s_QK + s_S
           + 4 * Br * d)        // ★ s_O
          * sizeof(half);

    // 1. 先查当前 kernel 的 MaxDynamicSharedMemorySize(可能已设过)
    int current_sram_size = 0;
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&current_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    // printf("sram_size: %d, current_sram_size: %d\n", sram_size, current_sram_size);

    // 2. 只在 sram_size > current_sram_size 时,才 setAttribute
    if (sram_size > current_sram_size) {
        CHECK_CUDA_ERROR(cudaFuncSetAttribute(
            flash_attn_v2_mma_swizzle_doublebuffer_smemO_kernel<Br, Bc, Bd>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, sram_size));
    }

    dim3 grid_dim(div_ceil(N, 4 * Br), num_head, batch_size);
    dim3 block_dim(128);
    flash_attn_v2_mma_swizzle_doublebuffer_smemO_kernel<Br, Bc, Bd>
        <<<grid_dim, block_dim, sram_size, stream>>>(Q, K, V, O, N, M, d, softmax_scale);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


