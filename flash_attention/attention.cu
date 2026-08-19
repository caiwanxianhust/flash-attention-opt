#include "attention.cuh"
#include <stdio.h>



// v1 对外接口
void flash_attention_v1(const float* Q, const float* K, const float* V, float* O,
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream) 
{
    // v1 版本需要额外的 l, m 空间
    float *d_l;
    float *d_m;
    size_t mem_size = sizeof(float) * batch_size * num_head * N * 2;

    (cudaMallocAsync((void **)&d_l, mem_size, stream));
    d_m = d_l + batch_size * num_head * N;

    launch_flash_attn_v1_tiling_D_kernel(Q, K, V, O, d_l, d_m, batch_size, num_head, N, M, d, stream);

    (cudaFreeAsync(d_l, stream));
}



// v2 对外接口
void flash_attention_v2(const half* Q, const half* K, const half* V, half* O, 
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream) 
{
    // printf("flash_attention_v2\n");
    // launch_flash_attn_v2_mma_kernel(Q, K, V, O, batch_size, num_head, N, M, d, stream);
    // launch_flash_attn_v2_mma_swizzle_kernel(Q, K, V, O, batch_size, num_head, N, M, d, stream);
    // launch_flash_attn_v2_mma_swizzle_doublebuffer_kernel(Q, K, V, O, batch_size, num_head, N, M, d, stream);
    // launch_flash_attn_v2_mma_swizzle_multibuf_kernel(Q, K, V, O, batch_size, num_head, N, M, d, stream);                 
    // launch_flash_attn_v2_mma_swizzle_doublebuffer_accops_kernel(Q, K, V, O, batch_size, num_head, N, M, d, stream);     
    // launch_flash_attn_v2_mma_swizzle_doublebuffer_sg_kernel(Q, K, V, O, batch_size, num_head, N, M, d, stream);         
    // launch_flash_attn_v2_mma_swizzle_doublebuffer_accops_sg_kernel(Q, K, V, O, batch_size, num_head, N, M, d, stream);
    launch_flash_attn_v2_mma_swizzle_doublebuffer_smemO_kernel(Q, K, V, O, batch_size, num_head, N, M, d, stream);
}


// 唯一对外接口
void flash_attention_fp16_cuda(const half* Q, const half* K, const half* V, half* O, 
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream) 
{
    // printf("flash_attention_fp16_cuda\n");
    flash_attention_v2(Q, K, V, O, batch_size, num_head, N, M, d, stream);
}

// 唯一对外接口
void flash_attention_fp32_cuda(const float* Q, const float* K, const float* V, float* O,
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream) 
{
    // printf("flash_attention_fp32_cuda\n");
    flash_attention_v1(Q, K, V, O, batch_size, num_head, N, M, d, stream);
}


