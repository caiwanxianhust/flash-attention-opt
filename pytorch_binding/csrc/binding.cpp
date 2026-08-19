#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <stdio.h>


// 唯一对外接口
void flash_attention_fp32_cuda(const float* Q, const float* K, const float* V, float* O,
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream);


// 唯一对外接口
void flash_attention_fp16_cuda(const half* Q, const half* K, const half* V, half* O, 
    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream);


/**
 * 绑定接口
 * Q [batch_size, num_heads, N, d]
 * K [batch_size, num_heads, M, d]
 * V [batch_size, num_heads, M, d]
 */
torch::Tensor flash_attention(const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V)
{ 
    // printf("binfing.cpp flash_attention\n");
    TORCH_CHECK(Q.ndimension() == 4, "Input Q must be 4D");
    TORCH_CHECK(K.ndimension() == 4, "Input K must be 4D");
    TORCH_CHECK(V.ndimension() == 4, "Input V must be 4D");
    TORCH_CHECK(Q.device().is_cuda(), "Input Q must be on CUDA");
    TORCH_CHECK(K.device().is_cuda(), "Input K must be on CUDA");
    TORCH_CHECK(V.device().is_cuda(), "Input V must be on CUDA");

    int batch_size = Q.size(0);
    int num_head = Q.size(1);
    int N = Q.size(2);
    int M = K.size(2);
    int d = Q.size(3);

    auto options = Q.options();
    torch::Tensor output = torch::zeros_like(Q, options);
    auto stream = c10::cuda::getCurrentCUDAStream();
    
    TORCH_CHECK((Q.dtype() == K.dtype()) && (Q.dtype() == V.dtype()), "Input Q K V must be same type");

    if (Q.dtype() == torch::kFloat16)
    {
        flash_attention_fp16_cuda(
            reinterpret_cast<half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<half*>(K.data_ptr<at::Half>()),
            reinterpret_cast<half*>(V.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            batch_size, num_head, N, M, d,
            stream.stream());
    }
    else if (Q.dtype() == torch::kFloat32)
    {
        flash_attention_fp32_cuda(
            reinterpret_cast<float*>(Q.data_ptr<float>()),
            reinterpret_cast<float*>(K.data_ptr<float>()),
            reinterpret_cast<float*>(V.data_ptr<float>()),
            reinterpret_cast<float*>(output.data_ptr<float>()),
            batch_size, num_head, N, M, d,
            stream.stream());
    }

    return output;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention", &flash_attention, "flash_attention (CUDA)");
}