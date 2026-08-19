#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


void launchAttentionBaseline(const float* Q, const float* K, const float* V,
        float* QK, float* QK_softmax, float* O,
        const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);

void launchFlashAttentionMinimal(const float* Q, const float* K, const float* V, const int batch_size, const int num_head,
        const int N, const int d, float* l, float* m, float* O, cudaStream_t stream = 0);



/**
 * 基础 Tiling 实现
 * 将 K/V 分块加载到 Shared Memory，外层循环遍历 Q 的每一行，内层循环遍历 K 的分块，逐行计算 Softmax 和 O。
 */
void launch_flash_attn_v1_tiling_kernel(const float* Q, const float* K, const float* V,
    float* O, float* l, float* m,
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);

/**
 * Warp 级并行优化
 *   将 Q 也进行分块（Br），利用 Warp 内线程协作（warpAllReduce）并行计算点积和 Softmax，一次处理多行 Q。
 */
void launch_flash_attn_v1_tiling_warp_parallel_kernel(const float* Q, const float* K, const float* V,
    float* O, float* l, float* m,
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);

/**
 * 在 d 维度上更细粒度的分块计算，同时引入 WMMA API
 *  引入 NVIDIA WMMA API，将 Q/K/V 转换为 half 类型，利用 Tensor Core 进行矩阵乘法（GEMM），并行计算 QK 和 PV。
 */
void launch_flash_attn_v1_tiling_D_kernel(const float* Q, const float* K, const float* V,
    float* O, float* l, float* m,
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);


/**
 * 循环重排与流水线优化
 * 调整循环顺序（先遍历 Q 分块，再遍历 K 分块），在 K 的循环内部累加 O，减少了 O 的重复读写。
 * 通过循环交换减少了 Global Memory 的访问量，更好地利用了寄存器缓存中间结果，进一步提升了带宽利用率。
 */
void launch_flash_attn_v2_tiling_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ O,
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);


/**
 * 使用 ldmatrix 和 mma 指令
 */
void launch_flash_attn_v2_mma_kernel(const half* Q, const half* K, const half* V,
    half* O, const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);


/**
 * ldmatrix mma swizzle
 */
void launch_flash_attn_v2_mma_swizzle_kernel(const half* Q, const half* K, const half* V,
    half* O, const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);

/**
 * K/V 双缓冲版 flash attention v2
 * 在 (mma + swizzle)基础上加 K/V 双缓冲(cp.async)
 *
 * 双缓冲:用 s_K_buf[2], s_V_buf[2] 轮转
 *   - 当前 iter 用 s_K_buf[iter&1], s_V_buf[iter&1]
 *   - 同时后台预取下一组 K/V 到 s_K_buf[(iter+1)&1], s_V_buf[(iter+1)&1]
 */
void launch_flash_attn_v2_mma_swizzle_doublebuffer_kernel(const half* Q, const half* K, const half* V,
    half* O, const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);


/**
 * 多缓冲版 flash attention v2(N_BUF 由模板参数控制)
 */
void launch_flash_attn_v2_mma_swizzle_multibuf_kernel(const half* Q, const half* K, const half* V, half* O,
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0); 

/**
 * 双缓冲 + O 累加放 register 版 flash attention v2
 *
 * 在 v5(mma + swizzle)基础上加:
 *   - K/V 双缓冲(cp.async)
 *   - O 累加直接用 RC 寄存器,不再写回 s_O(省 smem + 少 LDS)
 *
 * 关键点(与原版区别):
 *   - 没有 s_O 数组
 *   - RC_pv 计算后立即用于 O 更新(不再中转 s_O)
 *   - 每个 lane 持有 4 个 half2,通过 mma 的输出 row/col 映射直接更新 O 的对应位置
 */
void launch_flash_attn_v2_mma_swizzle_doublebuffer_accops_kernel(const half* Q, const half* K, const half* V, half* O,
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0); 


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
 */
void launch_flash_attn_v2_mma_swizzle_doublebuffer_sg_kernel(const half* Q, const half* K, const half* V, half* O,
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);


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
 */
void launch_flash_attn_v2_mma_swizzle_doublebuffer_accops_sg_kernel(const half* Q, const half* K, const half* V, half* O,
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);


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
 */
void launch_flash_attn_v2_mma_swizzle_doublebuffer_smemO_kernel(const half* Q, const half* K, const half* V, half* O,
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);


