# Flash Attention 算子的简单实现

**写在前面**：孩子出生这两个月，笔者经历了人生最疲惫的时光，也真切体会到为人父母的不易，真是一把屎一把尿把娃拉扯大。喂奶、换尿布、哭闹哄睡，周而复始，所幸笔者情绪稳定，父爱仍在。直到清明时节孩子满两个月，作息终于规律了些，偶尔能抽出些时间整理之前的代码。本文代码早在之前就已经写好，由于近期比较忙，没有时间继续优化，先写一篇文章记录一下。

## 1 Flash Attention 简介
### 1.1 传统Attention的困境：当 $O(N^2)$ 复杂度遭遇长文本
作为人工智能领域近十年最具突破性的算法范式，Transformer模型自 2015 年提出以来已深刻重塑了深度学习的技术版图。其核心的 Self-Attention 机制虽然在建模长程依赖关系上表现出色，但 $O(N^2)$ 的时间复杂度（$N$ 为序列长度）在当今大模型动辄处理数万 token 的长文本场景中，逐渐暴露出两大系统性瓶颈：

- **显存容量瓶颈**：标准的 Self-Attention 机制计算复杂度为 $O(N^2)$，也就是说中间结果矩阵（如 QK^T、softmax 结果）的形状大小随 $N^2$ 增长，导致长序列训练或推理过程中很容易显存溢出。
- **内存带宽限制‌**：中间结果矩阵通常会临时存储到显存（即 HBM）中，因此计算时需要频繁从 HBM 读写中间矩阵，而 HBM 在 GPU 内存体系中具有大容量高延迟的性质，从而造成严重的内存访问延迟。

### 1.2 Flash Attention 的破局：用算法革新突破硬件边界

面对大模型上下文窗口指数级扩张的产业需求，为了解决 Transformer 模型面临的问题，‌斯坦福大学和纽约州立大学布法罗分校的研究者共同提出了 FlashAttention 算法，无需损失任何精度即可加速 Self-Attention 计算，并显著减少内存占用，代价是增加了计算复杂度。‌FlashAttention 的核心原理是将输入 Q、K、V 分块，并保证每个块能够在 SRAM 上完成注意力操作，并将结果更新回 HBM，从而降低对 HBM 的读写操作。总之，FlashAttention 将传统 Attention 的显存复杂度从 $O(N^2)$ 降至 $O(N)$，而代价仅是少量重复计算的额外 FLOPs。这种用计算换带宽的设计思路，完美契合了现代 GPU 显存带宽增长滞后于算力增长的硬件特性，让 Transformer 模型在长文本处理领域真正突破性能桎梏。

### 1.3 Flash Attention 的计算思路

在理解 Flash Attention 计算原理之前，先来了解一下 Online Softmax 计算思路，笔者在上一篇文章中对其进行了详细介绍。
> [【CUDA编程】online softmax 的 CUDA 实现](https://mp.weixin.qq.com/s/icKqqDfFBU2vVABexl1njw)

Flash Attention 的核心策略在于，沿着序列长度维度对 Q、K、V 进行分块处理。每次仅计算一个分块的最大值及指数和，并基于当前与前一分块的计算结果，动态调整指标，逐步修正，直至处理完所有分块。这一机制确保了仅需一次 HBM 访问，即可完成整个 Attention 计算，即实现 one-pass Attention。

对于给定序列，不妨记前 $N$ 个元素的最大值 $x_{max}$ 记为 $m_{N}$，前 $N$ 个元素的 $\sum e^{x_j - x_{max}}$ 记为 $d_{N}$，序列索引从 1 开始，则有递推公式：

- 最大值更新

$$
m_i = max(m_{i-1}, x_i)
$$

- 指数和更新

$$
\begin{split}
    d_{i} &= \sum _{j=1}^{i} e^{x_j - m_{i}} \\
            &= \sum _{j=1}^{i-1} e^{x_j - m_{i}} + e^{x_i - m_{i}} \\
            &= d_{i-1} e^{m_{i-1} - m_{i}} + e^{x_{i} - m_{i}}
\end{split}  
$$

- Attention Out 更新‌

$$
\begin{split}
    O_{i} &= \sum _{j=1}^{i} \frac{e^{x_j - m_{i}}}{d_{i}} V[j, :] \\
            &= \sum _{j=1}^{i-1} \frac{e^{x_j - m_{i}}}{d_{i}} V[j, :] + \frac{e^{x_i - m_{i}}}{d_{i}} V[i, :] \\
            &= \sum _{j=1}^{i-1} \frac{e^{x_j - m_{i-1}}}{d_{i-1}} \frac{e^{x_j - m_{i}}}{e^{x_j - m_{i-1}}} \frac{d_{i-1}}{d_i} V[j, :] + \frac{e^{x_i - m_{i}}}{d_{i}} V[i, :] \\
            &= (\sum _{j=1}^{i-1} \frac{e^{x_j - m_{i-1}}}{d_{i-1}} V[j, :]) \frac{e^{x_j - m_{i}}}{e^{x_j - m_{i-1}}} \frac{d_{i-1}}{d_i} V[j, :] + \frac{e^{x_i - m_{i}}}{d_{i}} V[i, :] \\
            &= O_{i-1} \frac{d_{i-1}}{d_i} e^{m_{i-1} - m_i} + \frac{e^{x_i - m_{i}}}{d_{i}} V[i, :] \\
\end{split}  
$$

从上述公式可以看出，我们需要在计算分块过程中维护两个基本中间结果变量 $m$ 和 $d$，这一点与 Online Softmax 一致，毕竟 Attention 就是在 Softmax 的基础上右乘一个 V 矩阵。而最终当前分块的 Attention Out（$O_{i}$）则跟上一个分块的 $O_{i-1}$、$m_{i-1}$、$d_{i-1}$ 有关，而 $O$、$m$、$d$ 三个变量占用的内存空间与分块大小有关，$m$、$d$ 作为需要维护的中间结果变量相比与传统 Attention 的 $QK^T$ 与 $Softmax(QK^T)$ 矩阵从 $O(N^2)$ 降低到了 $O(N)$，极大地降低了显存占用量且可以轻松放入 GPU shared memory 中参与计算，从而显著提升计算效率。

## 2 Attention Baseline
在介绍 Flash Attention 算子的 CUDA 实现之前，我们不妨回顾一下传统 Attention 的实现细节，具体计算公式如下，不再详细赘述。

$$
Attention(Q,K,V) = Softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

从公式来看，主要分为三个步骤，矩阵乘法 $QK^T$、$Softmax(\frac{QK^T}{\sqrt{d_k}})$、矩阵乘法 $Softmax(\frac{QK^T}{\sqrt{d_k}})V$。其中两次矩阵乘法我们直接调用 cuBLAS 库完成，Softmax 通过标准的 3-pass-safe-softmax Kernel 完成，具体代码如下：

```cpp
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
```

关于这个 Baseline 代码，唯一需要注意的就是在调用 cuBALS api 时的参数设置问题。

- cuBALS api 要求矩阵的内存分布是列主序，而我们通常从 CPU 端传输到 GPU 端的数据是行主序，要想矩阵乘法的计算结果按行主序输出，通常我们要利用矩阵乘法转置公式 $(AB)^T = B^T A^T$，即要计算 `[N, d]` 与 `[d, M]` 的乘法即 `[N, M, d]`，实际应按 `[M, N, d]` 传入。
- K 矩阵通常是 `[M, d]` 的存储形式，所以要在计算时对其进行转置。具体参数设置思路，可以参考笔者另外两篇文章：[【CUDA编程】cuBLAS 库中矩阵乘法参数设置问题](https://mp.weixin.qq.com/s/MvTaIBfVW3gcwQtV2VjMTw)、[【CUDA编程】cuBLAS中的列主序问题](https://mp.weixin.qq.com/s/pwFem1cCB-cS3kZvFKSLQw)

## 3 Flash Attention Minimal 实现思路
Flash Attention Minimal 是 github 上的一个开源项目，提供了一个 Flash Attention 的 CUDA 实现，并提供了 C++ 和 PyTorch 接口，获得了 769 个 star，笔者也是无意间见群里有人讨论，才去阅读了一下源码，这里先对这个项目源码进行解析，源码地址如下，有兴趣的读者可以阅读。
> https://github.com/tspeterkim/flash-attention-minimal

```cpp
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
```
**注意**：由于笔者要在自己的 CUDA C++ 环境上跑起来，不想使用 Pytorch 接口，所以上面的 Kernel 启动函数是笔者参照源码自己写的，调整了 `Bc` 和 `Br` 的值，至于为什么会调整后面会介绍。

Flash Attention Minimal 的 Kernel 解决的是形如 `[batch_size, num_head, N, d]` 形状的 Q、K、V 矩阵的 Self-Attention 计算任务，因为是 Self-Attention，所以 Q、K、V 矩阵形状一致，序列长度都为 N。执行配置是每个 block 处理一个 head 的结果，每个 head 是独立的，相当于单独处理 `[N, d]` 形状的 Q、K、V 矩阵的 Self-Attention。

Flash Attention Minimal 将 Q 矩阵分为 `Tr` 个分块每个分块形状为 `[Br, d]`，将 K、V 矩阵分为 `Tc` 个分块每个分块形状为 `[Bc, d]`，并将每个 block 的线程数量设置为 `Bc`，具体 `Bc` 取值多少呢？源码中 `Bc` 和 `Br` 取值 `32`，也就是一个 block 只有 32 个线程，每个线程单独处理一行计算，这显然不是一个明智的选择，会导致 SM 占用率严重不足从而影响性能。关于 block_size 设置思路笔者在之前的一篇文章有过介绍，有兴趣的读者可以移步（[【CUDA编程】OneFlow Element-Wise 算子源码解读](https://mp.weixin.qq.com/s/tEUg_b5qH066qvMZJp88vQ)）。

```cpp
// TODO: determine Bc, Br dynamically
const int Bc = 32; const int Br = 32;
```

现在说到为什么笔者要在自定义的 kernel 启动函数 `launchFlashAttentionMinimal` 中将 `Bc` 和 `Br` 的值调整为 `2`，这也是无奈之举。按照 Flash Attention Minimal 的思路，以隐藏层维度 `d = 1024` 为例，需要 `(3 * 32 * 1024 + 32 * 32) * 4B = 388 KB` 共享内存，这是一个巨大的容量，而在笔者的 GPU 上单个 block 所能利用的共享内存最大只有 `48 KB`，所以根本跑不起来，只好调整小一些。为什么源码作者能跑起来？因为源码的测试用例里面 `d = 64`，这个跟当前大模型工程中的实际隐藏层维度差太远了，不太现实。

```cpp
// Calculate SRAM size needed per block
const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
int max_sram_size;
cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);
```

现在来看一下 Kernel 代码，首先是两层循环，第一层循环在 KV 的序列维度上移动，每次加载形状为 `[Bc, d]` 的分块到共享内存 `Kj` 和 `Vj` 中存储用于后续计算，这里源码用了一个比较 navie 的方式即每个线程单独处理自己的一行元素，在当前行上循环 `d` 次，这会带来两个严重问题：
  
- 对于全局内存来说，同一 warp 内每个线程每次访问的元素间隔了一行（即 `d` 个元素），这回导致非合并访问，非合并访问会使得 L2 缓存命中率降低，甚至每个线程可能都需要单独的内存访问事务从全局内存中加载元素，这将极大降低全局内存访问性能。
- 对于共享内存来说，同一 warp 内每个线程每次访问的元素也间隔了一行（即 `d` 个元素），而 `d` 通常是 `32` 的倍数，这将导致严重的 bank conflict，极大地降低共享内存访问性能。

第二层循环是在 Q 地序列维度上移动，每次加载形状为 `[Br, d]` 的分块到共享内存 `Qi`，然后从全局内存中加载上一个分块的 `m`、`l`（相当于前面公式里的 $d$） 变量记为 `row_m_prev` 和 `row_l_prev`。随后在 `Bc` 维度上循环，每个线程固定取 `Qi` 上一行数据，分别与 `Bc` 个向量进行内积，然后乘以 `softmax_scale`（相当于 $1/\sqrt{d}$），将计算结果存入共享内存变量 `S[Br, Bc]` 中，同时在 `Bc` 维度上记录最大值 `row_m` 用于后续 Softmax 计算。这个环节对于 `Qi`、`Kj`、`S` 来说都有严重的 bank conflict 问题。

接着又在 `Bc` 维度进行一次循环求 `row_l` 和 Softmax 分子，除了 bank conflict 问题之外，其实这里求 `row_l` 和 Softmax 分子的操作完全可以通过 online softmax 思路融合到上一次在 `Bc` 维度的遍历之中，没有必要再循环一次。

获取当前线程对应的 `row_m` 和 `row_l` 之后，结合之前的 `row_m_prev` 和 `row_l_prev` 计算出最新的 `m`、`l` 记为 `row_m_new` 和 `row_l_new`。

然后计算 Attention Out，先在 `d` 维度上循环，然后在 `Bc` 维度上循环，每个线程负责一行 `S` 向量与 `Bc` 行 `Vj` 向量的内积，然后在 `d` 维度上利用 Flash Attention 公式对 `O` 进行更新。最后将 `row_m_new` 和 `row_l_new` 更新到全局内存。

整个代码看下来源码存在如下问题：
- 严重的非合并访存。
- 严重的共享内存 bank conflict。
- SM 占用率偏低，并行程度不高。
- thread 之间完全没有数据交互。
- 只能计算 QKV 形状相同的 Self-Attention。
- 计算思路比较简单，没有合理利用 GPU 并行特性，导致最终计算效率极低。

## 4 Flash Attention v1：每次循环处理一行计算任务
考虑到 Flash Attention Minimal 存在诸多的性能问题，笔者考虑自己对 Flash Attention 进行 CUDA 实现，并逐步迭代优化。

第一版我们还是一次加载 `Bc` 行 K\V 矩阵元素和 `1` 行 Q 矩阵元素到共享内存，在 Q 矩阵序列维度的一次循环中，我们利用一个 block 内所有线程只计算一行结果。这样的话我们总共需要 `(d + 2 * Bc * d + Bc) * sizeof(float)` 的共享容量。这里 `Bc` 我们设置为 `4`，在隐藏层维度 `d` 为 `1024` 时总共需要约 `36 KB` 共享内存，如果 `d` 取更大的值，则 `Bc` 要做相应缩减。`block_size` 设置为 `128`，与其他参数解耦。

```cpp
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
                tmp_ml.m = blockAllReduceSum<float>(tmp_ml.m);
                row_ml = MDFOp()(row_ml, tmp_ml);
                if (threadIdx.x == 0) { s_S[k] = tmp_ml.m; }
                __syncthreads();
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
                O[qo_offset + j * d + k] = 1.0f / row_ml_new.d * (row_ml_prev.d * __expf(row_ml_prev.m - row_ml_new.m) * O[qo_offset + j * d + k] + __expf(row_ml.m - row_ml_new.m) * pv);
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
```

首先也是两层循环，第一层循环在 KV 的序列维度上移动，每次加载形状为 `[Bc, d]` 的分块到共享内存 `s_K` 和 `s_V` 中存储用于后续计算，这里我们加载的时候充分考虑到共享内存的 bank conflict 和全局内存的合并访问，加载完成后进行 block 内同步。

第二层循环是在 Q 的序列维度上移动，每次加载形状为 `[1, d]` 的分块到共享内存 `s_Q`，然后从全局内存中加载上一个分块的 `m`、`l`（相当于前面公式里的 $d$） 变量记为 `row_ml_prev`（这里我们定义了一个结构体 `MD_F`，存储 `m`、`l` 两个 float 变量）。

在 `Bc` 维度上循环，每次计算 `s_Q` 与 `s_K` 的一行向量的内积，block 内线程共同参与计算，利用 block 内规约求和算出结果存入 `tmp_ml.m` 中并将内积结果存入 `s_S`，然后根据 `m` 和 `l` 的更新公式不断循环更新，获取这 `Bc` 个结果的最终 `m` 和 `l` 存入 `row_ml` 中。

获取当前线程对应的 `row_ml` 之后，结合之前的 `row_ml_prev` 计算出最新的 `m`、`l` 记为 `row_ml_new`。

然后计算 Attention Out，先在 `d` 维度上循环，由 block 内线程共同完成，然后在 `Bc` 维度上循环，每个线程每次负责 `s_S` 向量与 `1` 行 `s_V` 列向量的内积，然后在 `d` 维度上利用 Flash Attention 公式对 `O` 进行更新。最后将 `row_ml_new` 更新到全局内存。

## 5 Flash Attention v2：每次循环处理 Br 行计算任务
前面的 Kernel 每次循环只能计算一行结果，针对序列长度较长，而隐藏层维度较小的场景，性能可能偏低，那么我们是否可以改写一下一次性计算多行呢？当然可以，在不改变整体计算思路的情况下，只需要满足共享内存容量不溢出即可。

```cpp
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
                tmp_ml.m = warpAllReduce<SumOp, float>(tmp_ml.m);
                if (lane_id == 0)
                {
                    s_S[warp_id * Bc + k] = tmp_ml.m;
                }
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
                O[qo_offset + (j + warp_id) * d + k] = 1.0f / row_ml_new.d * (row_ml_prev[warp_id].d * __expf(row_ml_prev[warp_id].m - row_ml_new.m) * O[qo_offset + (j + warp_id) * d + k] + __expf(row_ml.m - row_ml_new.m) * pv);
            }

            // 写入当前 Bc 组的 l 和 m
            if (lane_id == 0)
            {
                l[lm_offset + j + warp_id] = row_ml_new.d;
                m[lm_offset + j + warp_id] = row_ml_new.m;
            }
            __syncthreads();
        }
        // __syncthreads();
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
```

这次我们一次加载 `Bc` 行 K\V 矩阵元素和 `Br` 行 Q 矩阵元素到共享内存，在 Q 矩阵序列维度的一次循环中，我们利用一个 block 内所有线程计算 `Br` 行结果。这样的话我们总共需要 `(Br * d + 2 * Bc * d + Br * Bc) * sizeof(float)` 的共享容量。这里 `Bc` 我们设置为 `2`，`Br` 设置为 `4`，在隐藏层维度 `d` 为 `1024` 时总共需要约 `32 KB` 共享内存，如果 `d` 取更大的值，则 `Bc` 或 `Br` 要做相应缩减。`block_size` 设置为 `Br * 32`，即每个 warp 处理一行计算，合理利用 warp 内规约机制。

首先还是两层循环，加载 Q\K\V 矩阵元素到共享内存，保证基本的全局内存合并访问和共享内存不出现 bank conflict。从全局内存中加载上一个分块的 `m`、`l`（相当于前面公式里的 $d$） 变量记为 `row_ml_prev`，并同步 block。

在 `Bc` 维度上循环，每个 warp 每次计算 `s_Q` 的一行向量 与 `s_K` 的一行向量的内积，warp 内线程共同参与计算，利用 warp 内规约求和算出结果存入 `tmp_ml.m` 中并将内积结果存入 `s_S`，然后根据 `m` 和 `l` 的更新公式不断循环更新，获取这 `Bc` 个结果的最终 `m` 和 `l` 存入 `row_ml` 中。与上一个 Kernel 不同的是，本 Kernel 的计算基本单位不再是 block 而是会细粒度到 warp。

获取当前线程对应的 `row_ml` 之后，结合之前的 `row_ml_prev` 计算出最新的 `m`、`l` 记为 `row_ml_new`。

然后计算 Attention Out，先在 `d` 维度上循环，由 warp 内线程共同完成，然后在 `Bc` 维度上循环，每个线程每次负责 `1` 行 `s_S` 向量与 `1` 行 `s_V` 列向量的内积，然后在 `d` 维度上利用 Flash Attention 公式对 `O` 进行更新。最后将 `row_ml_new` 更新到全局内存。

## 6 Flash Attention v3：在 d 维度上更细粒度的分块计算
在前面两个 Kernel 中，每次都需要将 Q\K\V 矩阵的整行元素加载到共享内存，当隐藏层维度较大时，`Br` 和 `Bc` 受限于共享内存的大小都只能取较小的值，甚至与无法将整行元素加载进去。为了摆脱共享内存对隐藏层维度 `d` 的限制，我们可以一次性只加载 `Bd` 列元素，循环加载最终完成整个 `d` 维度的计算，这样的话 `Br` 和 `Bc` 也可以取一个较大的值从而有更高的并行性。

具体地，我们把启动参数设置如下：
```cpp
void launchFlashAttentionKernel_v3(const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
                                    float *__restrict__ O, float *__restrict__ l, float *__restrict__ m,
                                    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream)
{
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
    flashAttentionKernel_v3<Bc, Br, Wc, Wr><<<grid_dim, block_dim, 0, stream>>>(Q, K, V, O, l, m, N, M, d, softmax_scale);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
```
在隐藏层维度 `d` 为 `1024` 时总共需要约 `29.5 KB` 共享内存，如果 `d` 取更大的值，则 `Bc` 或 `Br` 不受影响。`block_size` 设置为 `Bc * Br / (Wr * Wc) * 32`，将计算任务具体到每个 warp，合理利用 warp 内规约机制。其中 `Bc`、`Br` 和 `Bd` 表示单个 block 每次处理的矩阵大小，而 `Wc`、`Wr` 则表示单个 warp 处理的矩阵大小。

在 Kernel 内部由于我们使用 tensor core 的 WMMA api 计算矩阵乘法，所以首先引入 `nvcuda` 命名空间，为了让前后两个矩阵乘法（$QK^T$ 和 $Softmax(\frac{QK^T}{\sqrt{d_k}})V$）复用一些内存空间，我们让 `Bd` 等于 `Bc`，这样的话两个矩阵乘法的形状参数将完全一致，很多参数和变量可以复用。此外，在共享内存中还定义了一些变量：

- `s_Q_half[Br * Bd]`：加载 Q 矩阵分片
- `s_K_half[Bc * Bd]`：加载 K 矩阵分片
- `s_V_half[Bc * Bd]`：加载 V 矩阵分片
- `s_S[Br * Bc]`：存储 $QK^T$ 结果
- `s_S_half[Br * Bc]`：存储 softmax 结果并作为左矩阵参与第二个矩阵乘法
- `s_O[Br * Bd]`：存储矩阵乘法 $Softmax(\frac{QK^T}{\sqrt{d_k}})V$ 的结果
- `row_ml_prev[Br]`：存储截至前一个分片的所有行的 `m` 和 `l`
- `row_ml[Br]`：存储当前分片处理的所有行的 `m` 和 `l`
- `row_ml_new[Br]`：存储截至当前分片的所有行的 `m` 和 `l`

要说明的是，为什么出现了这么多 half 类型，是因为使用了 WMMA api 中 `half * half -> float` 模板，在 Ampere 架构中 tensor core 其实也支持 `tf32 * tf32 -> float`，但是 `tf32` 转 `float` 还需要专门处理，所以干脆就不用 `tf32 * tf32 -> float` 了。

由于我们的计算任务是要具体分配到 warp 级别的，所以 Kernel 内定义了一些 warp 相关的索引：

- `warp_col`：当前 warp 负责计算的 gemm 矩阵分片 `[Wr, Wc]` 在 block 负责的矩阵分片 `[Br, Bc]` 中的列索引
- `warp_row`：当前 warp 负责计算的 gemm 矩阵分片 `[Wr, Wc]` 在 block 负责的矩阵分片 `[Br, Bc]` 中的行索引
- `warp_id`：当前 warp 在 block 中的索引
- `lane_id`：当前 thread 在 warp 中的索引
- `WMITERS`：由于 WMMA API 限制，一个 warp 单次不能完成 `[Wr, Wc]` 的 gemm 运算，需要进行循环处理，在行上的循环次数
- `WNITERS`：同上，在列上的循环次数
- `WKITERS`：同上，在 `Bd` 维度上的循环次数

为了加深理解，笔者简单绘制了示意图如下：
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qRgKAHCG09zD2iamk08Ft1Giacd0PhkNwK7DjjqddLw31Cmwb8VvWcW6r6icmEopdbv5DPk0ylMtzZw/640?wx_fmt=png&amp;from=appmsg)

然后对 gemm 计算所需的类模板、参数、变量进行定义，初始化了几个类模板：`FragAType`、`FragBType`、`FragCFloatType`、`FragVType`，分别标识矩阵 Q、K、QK^T、V 的分片的输入类型，基于以上类型定义了当前 warp 内的矩阵乘法 fragment：`a_frag`、`b_frag`、`acc_frag`、`v_frag`。这里需要注意的是，K 矩阵是以其转置矩阵 `K^T` 的形式参与计算的，所以可以将其指定为 `K^T` 矩阵的列主序形式，而 V 矩阵虽然也是右矩阵，但是其不需要转置，所以不能复用 `b_frag`。

还是两层循环，对 K|V 在 `M` 维度分组，每组长度为 `Bc`，共分为 `Tc` 组，对 Q 在 `N` 维度分组，每组长度为 `Br`，共分为 `Tr` 组。从全局内存中加载上一个分块的 `m`、`l`（相当于前面公式里的 $d$）存储到 `row_ml_prev` 中，并初始化 `row_ml` 后同步 block。

在计算 $QK^T$ gemm 时由于需要在 `d` 维度上循环累加，所以要提前将累加矩阵分片初始化为 `0.0f`，然后开始以 `Bd` 为步长循环。先从全局内存中加载数据到共享内存变量 `s_Q_half`、`s_K_half`，这里笔者通过一个设备函数 `loadQKFromGmemAndConvertToHalf` 实现，在加载的同时顺便做了 `float -> half` 的转换，加载完成后同步 block，在加载时适当的利用了向量化加载从全局内存一次读取 4 个 float，这里其实也可以将 half 向量化写入共享内存，由于提升不明显，为了可读性就单个数据写入吧，唯一需要注意的点就是数据索引的计算，具体代码如下：

```cpp
template <int Br, int Bc, int Bd>
__device__ void loadQKFromGmemAndConvertToHalf(const float *Q, const float *K, const int d,
                                                half *s_Q, half *s_K, const int offset_q, const int offset_kv)
{
    int row_a, col_a;
    float4 tmp4;

#pragma unroll
    for (int i = (threadIdx.x << 2); i < Br * Bd; i += (blockDim.x << 2))
    {
        row_a = i / Bd;
        col_a = i % Bd;
        tmp4 = reinterpret_cast<const float4 *>(Q + offset_q + row_a * d + col_a)[0];
        s_Q[row_a * Bd + col_a] = __float2half(tmp4.x);
        s_Q[row_a * Bd + col_a + 1] = __float2half(tmp4.y);
        s_Q[row_a * Bd + col_a + 2] = __float2half(tmp4.z);
        s_Q[row_a * Bd + col_a + 3] = __float2half(tmp4.w);
    }

#pragma unroll
    for (int i = (threadIdx.x << 2); i < Bc * Bd; i += (blockDim.x << 2))
    {
        row_a = i / Bd;
        col_a = i % Bd;
        tmp4 = reinterpret_cast<const float4 *>(K + offset_kv + row_a * d + col_a)[0];
        s_K[row_a * Bd + col_a] = __float2half(tmp4.x);
        s_K[row_a * Bd + col_a + 1] = __float2half(tmp4.y);
        s_K[row_a * Bd + col_a + 2] = __float2half(tmp4.z);
        s_K[row_a * Bd + col_a + 3] = __float2half(tmp4.w);
    }
}
```
数据加载到共享内存之后下一步就是在各个 warp 内将数据进一步加载到 WMMA API 中定义的 fragment，随后调用 tensor core 计算 gemm，最后同步 block，从而可以重新填充下一轮的 `s_Q_half`、`s_K_half`。这里笔者通过一个设备函数 `gemmFromSmemByWMMA` 实现，唯一需要注意的就是加载数据的时候要以 warp 为主体计算索引，具体代码如下：

```cpp
template <int Bd, int Wc, int Wr, typename T1, typename T2, typename T3>
__device__ void gemmFromSmemByWMMA(const half *__restrict__ s_Q, const half *__restrict__ s_K,
                                    T1 *q_frag, T2 *k_frag, T3 *acc_frag, const int warp_row, const int warp_col,
                                    const int WMITERS, const int WNITERS, const int WKITERS)
{
    using namespace nvcuda;

    for (int wmidx = 0; wmidx < WMITERS; ++wmidx)
    {
        for (int wkidx = 0; wkidx < WKITERS; ++wkidx)
        {
            int shm_offset = warp_row * Wr * Bd + wmidx * 16 * Bd + wkidx * 16;
            wmma::load_matrix_sync(q_frag[wmidx * WKITERS + wkidx], s_Q + shm_offset, Bd);
        }
    }

    for (int wnidx = 0; wnidx < WNITERS; ++wnidx)
    {
        for (int wkidx = 0; wkidx < WKITERS; ++wkidx)
        {
            int shm_offset = warp_col * Wc * Bd + wnidx * 16 * Bd + wkidx * 16;
            wmma::load_matrix_sync(k_frag[wnidx * WNITERS + wkidx], s_K + shm_offset, Bd);
        }
    }

    for (int wmidx = 0; wmidx < WMITERS; ++wmidx)
    {
        for (int wnidx = 0; wnidx < WNITERS; ++wnidx)
        {
            for (int wkidx = 0; wkidx < WKITERS; ++wkidx)
            {
                wmma::mma_sync(acc_frag[wmidx * WNITERS + wnidx], q_frag[wmidx * WKITERS + wkidx],
                                k_frag[wnidx * WKITERS + wkidx], acc_frag[wmidx * WNITERS + wnidx]);
            }
        }
    }
}
```

完成 gemm 后，`acc_frag` 中存储的已经是当前 warp 负责处理的形如 `[Br, d] * [Bc, d] -> [Br, Bc]` 的计算结果的一部分，还需要把各个 warp 中的数据存储到 `s_S` 中获得完整的 $QK^T$ 结果，另外还可以顺便除以 `softmax_scale` 规整一下，笔者通过设备函数 `StoreQKGEMMToSmem` 实现，该函数逻辑比较简单，结合上面的示意图计算好索引即可，具体代码如下：

```cpp
template <int Bc, int Wc, int Wr, typename T>
__device__ void StoreQKGEMMToSmem(float *__restrict__ s_S, T *acc_frag, const int warp_row, const int warp_col,
                                    const int WMITERS, const int WNITERS, const int WKITERS, const float softmax_scale)
{
    using namespace nvcuda;
    // 从 s_S 中取出元素，累加矩阵计算结果，再写入 s_S
    for (int wmidx = 0; wmidx < WMITERS; ++wmidx)
    {
        for (int wnidx = 0; wnidx < WNITERS; ++wnidx)
        {
            int shm_offset = warp_row * Wr * Bc + warp_col * Wc + wmidx * 16 * Bc + wnidx * 16;
            for (int idx = 0; idx < acc_frag[wmidx * WNITERS + wnidx].num_elements; ++idx)
            {
                acc_frag[wmidx * WNITERS + wnidx].x[idx] *= softmax_scale;
            }
            wmma::store_matrix_sync(s_S + shm_offset, acc_frag[wmidx * WNITERS + wnidx], Bc, wmma::mem_row_major);
        }
    }
}
```

获取 $\frac{QK^T}{\sqrt{d}}$ 后，需要结合 online softmax 计算 `m` 和 `l`。这里我们让每个 warp 单次只处理一行数据，由于 `Br` 可能不会正好等于 warp 个数，所以需要加一个循环，每个 warp 可能会处理多行。然后 warp 内每个线程分别读取 `s_S` 中对应元素，循环计算出当前线程处理的元素的整体 `m` 和 `l`，warp 内同步。通过自定义的束内同步函数 `warpAllReduce` 计算出当前行的 `m` 和 `l` 并存入 `row_ml`，顺便再更新 `row_ml_new`。最后为了下一个 gemm 的计算，需要再将 `s_S` 中的 float 元素转为 half，顺便再减去当前行的 `m`，写入 `s_S_half`。

```cpp
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

__device__ __inline__ MD_F warpAllReduce(MD_F val)
{
    float tmp_m;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        tmp_m = max(val.m, __shfl_xor_sync(0xffffffff, val.m, mask, 32));
        val.d = val.d * __expf(val.m - tmp_m) + __shfl_xor_sync(0xffffffff, val.d, mask, 32) * __expf(__shfl_xor_sync(0xffffffff, val.m, mask, 32) - tmp_m);
        val.m = tmp_m;
    }
    return val;
}
```

得到 $Softmax(\frac{QK^T}{\sqrt{d}})$ 和对应的 `m` 和 `l` 后，即可开始第二个 gemm 的计算，第二个 gemm 不同于第一个 gemm，由于左矩阵 `[Br, Bc]` 是固定的，所以只需要右矩阵在 `d` 方向上滑动，且每次的乘法结果 `[Br, Bd]` 无需累加，所以在整体计算前可以先把 `[Br, Bc]`（即 `s_S_half`）写入 `a_frag` 中，无需每次循环再加载一次，另外再每次循环开始时，由于计算结果无需累加，所以要对累加矩阵 `c_frag` 置零。

循环开始后，先从全局内存中加载数据到共享内存变量 `s_V_half`，这里笔者通过一个设备函数 `loadVFromGmemAndConvertToHalf` 实现，在加载的同时顺便做了 `float -> half` 的转换，加载完成后同步 block，在加载时适当的利用了向量化加载从全局内存一次读取 4 个 float，唯一需要注意的点就是数据索引的计算，具体代码如下：

```cpp
template <int Bc, int Bd>
__device__ void loadVFromGmemAndConvertToHalf(const float *V, const int d, half *s_V, const int offset_kv)
{
    int row_a, col_a;
    float4 tmp4;
#pragma unroll
    for (int i = (threadIdx.x << 2); i < Bc * Bd; i += (blockDim.x << 2))
    {
        row_a = i / Bd;
        col_a = i % Bd;
        tmp4 = reinterpret_cast<const float4 *>(V + offset_kv + row_a * d + col_a)[0];
        s_V[row_a * Bd + col_a] = __float2half(tmp4.x);
        s_V[row_a * Bd + col_a + 1] = __float2half(tmp4.y);
        s_V[row_a * Bd + col_a + 2] = __float2half(tmp4.z);
        s_V[row_a * Bd + col_a + 3] = __float2half(tmp4.w);
    }
}
```

数据加载到共享内存之后下一步就是在各个 warp 内将 `s_V_half` 进一步加载到 `v_frag`，随后调用 tensor core 计算 gemm，最后同步 block，从而可以重新填充下一轮的 `s_V_half`。这里笔者通过一个设备函数 `pvGemmFromSmemByWMMA` 实现，唯一需要注意的就是加载数据的时候要以 warp 为主体计算索引，具体代码如下：

```cpp
template <int Bd, int Wc, int Wr, typename T1, typename T2, typename T3>
__device__ void pvGemmFromSmemByWMMA(const half *__restrict__ s_V,
    T1 *p_frag, T2 *v_frag, T3 *c_frag, const int warp_row, const int warp_col,
    const int WMITERS, const int WNITERS, const int WKITERS)
{
    using namespace nvcuda;
    #pragma unroll
    for (int wnidx = 0; wnidx < WNITERS; ++wnidx)
    {
        #pragma unroll
        for (int wkidx = 0; wkidx < WKITERS; ++wkidx)
        {
            int shm_offset = warp_col * Wc + wnidx * 16 + wkidx * 16 * Bd;
            wmma::load_matrix_sync(v_frag[wnidx * WNITERS + wkidx], s_V + shm_offset, Bd);
        }
    }

    #pragma unroll
    for (int wmidx = 0; wmidx < WMITERS; ++wmidx)
    {
        #pragma unroll
        for (int wnidx = 0; wnidx < WNITERS; ++wnidx)
        {
            #pragma unroll
            for (int wkidx = 0; wkidx < WKITERS; ++wkidx)
            {
                wmma::mma_sync(c_frag[wmidx * WNITERS + wnidx], p_frag[wmidx * WKITERS + wkidx],
                                v_frag[wnidx * WKITERS + wkidx], c_frag[wmidx * WNITERS + wnidx]);
            }
        }
    }
}
```

完成 gemm 后，`acc_frag` 中存储的已经是当前 warp 负责处理的形如 `[Br, Bc] * [Bc, Bd] -> [Br, Bd]` 的计算结果的一部分，还需要把各个 warp 中的数据存储到 `s_O` 中获得完整的 $Softmax(\frac{QK^T}{\sqrt{d}}) V$ 结果，笔者通过设备函数 `StoreOGEMMToSmem` 实现，该函数逻辑比较简单，结合上面的示意图计算索引即可，具体代码如下：

```cpp
template <int Bd, int Wc, int Wr, typename T>
__device__ void StoreOGEMMToSmem(float *__restrict__ s_Q, T *acc_frag, const int warp_row, const int warp_col,
                                    const int WMITERS, const int WNITERS, const int WKITERS)
{
    using namespace nvcuda;
    #pragma unroll
    for (int wmidx = 0; wmidx < WMITERS; ++wmidx)
    {
        #pragma unroll
        for (int wnidx = 0; wnidx < WNITERS; ++wnidx)
        {
            int shm_offset = warp_row * Wr * Bd + warp_col * Wc + wmidx * 16 * Bd + wnidx * 16;
            wmma::store_matrix_sync(s_Q + shm_offset, acc_frag[wmidx * WNITERS + wnidx], Bd, wmma::mem_row_major);
        }
    }
}
```

此时当前分片内的中间变量都已经计算完毕，剩下的就是在 `d` 维度上更新结果矩阵 O 即可。我们还是采用单个 warp 单次只处理一行的策略，基于以下公式更新 O 矩阵。

$$
O_{i} = O_{i-1} \frac{d_{i-1}}{d_i} e^{m_{i-1} - m_i} + \frac{e^{x_i - m_{i}}}{d_{i}} V[i, :] 
$$

沿 `d` 维度更新完结果矩阵 O 以后，当前分片的计算任务就完成了，最后将前面计算得到的 `row_ml_new` 更新到全局内存中以备下一轮分片计算时取用即可，完整的 Kernel 主函数代码如下：

```cpp
/**
    * grid( num_head, batch_size )
    * block( BLOCK_SIZE )
    * Q\O: [batch_size, num_head, N, d]
    * K\V: [batch_size, num_head, M, d]
    * l: [batch_size, num_head, N, 1]
    * m: [batch_size, num_head, N, 1]
    */
template <int Bc, int Br, int Wc, int Wr>
__global__ void flashAttentionKernel_v3(const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
                                        float *__restrict__ O, float *__restrict__ l, float *__restrict__ m,
                                        const int N, const int M, const int d, const float softmax_scale)
{
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
    for (int i = 0; i < M; i += Bc)
    {
        // 对 Q 在 N 维度分组，每组长度为 Br，共分为 Tr 组
        for (int j = 0; j < N; j += Br)
        {
#pragma unroll
            for (int k = threadIdx.x; k < Br; k += blockDim.x)
            {
                // 上一个 Bc 组结束时每行的 m 和 l
                row_ml_prev[k] = {m[lm_offset + j + k], l[lm_offset + j + k]};
                row_ml[k] = {-1e20f, 0.0f};
            }
            __syncthreads();
#pragma unroll
            for (int k = 0; k < WMITERS * WNITERS; ++k)
            {
                wmma::fill_fragment(acc_frag[k], 0.0f);
            }
            // 计算 QK 矩阵
            for (int k = 0; k < d; k += Bd)
            {
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
            for (int s = warp_id; s < Br; s += (blockDim.x >> 5))
            {
                MD_F row_ml_tmp = {-1e20f, 0.0f};
                for (int k = lane_id; k < Bc; k += 32)
                {
                    MD_F tmp_ml = {s_S[s * Bc + k], 1.0f};
                    row_ml_tmp = MDFOp()(row_ml_tmp, tmp_ml);
                }
                __syncwarp();

                // 得到 s_S[Br, Bc] 每一行的 m 和 l
                row_ml_tmp = warpAllReduce(row_ml_tmp);
                if (lane_id == 0)
                {
                    row_ml[s] = row_ml_tmp;
                    row_ml_new[s] = MDFOp()(row_ml_prev[s], row_ml_tmp);
                }

                // 更新 s_S[Br, Bc]
                for (int k = lane_id; k < Bc; k += 32)
                {
                    s_S_half[s * Bc + k] = __float2half(__expf(s_S[s * Bc + k] - row_ml_tmp.m));
                }
            }
            __syncthreads();

            loadSFromSmemToReg<Bc, Wc, Wr, FragAType>(s_S_half, a_frag, warp_row, warp_col, WMITERS, WNITERS, WKITERS);

            // 计算 s_S[Br, Bc] * s_V[Bc, Bd]
            for (int k = 0; k < d; k += Bd)
            {
                for (int s = 0; s < WMITERS * WNITERS; ++s)
                {
                    wmma::fill_fragment(acc_frag[s], 0.0f);
                }
                loadVFromGmemAndConvertToHalf<Bc, Bd>(V, d, s_V_half, kv_offset + i * d + k);

                __syncthreads();
                pvGemmFromSmemByWMMA<Bd, Wc, Wr, FragAType, FragVType, FragCFloatType>(s_V_half,
                                                                                        a_frag, v_frag, acc_frag, warp_row, warp_col, WMITERS, WNITERS, WKITERS);
                StoreOGEMMToSmem<Bd, Wc, Wr, FragCFloatType>(s_O, acc_frag, warp_row, warp_col, WMITERS, WNITERS, WKITERS);

                __syncthreads();

                for (int s = warp_id; s < Br; s += (blockDim.x >> 5))
                {
                    for (int t = lane_id; t < Bd; t += 32)
                    {
                        // 更新 O 矩阵
                        O[qo_offset + (j + s) * d + k + t] = 1.0f / row_ml_new[s].d * (row_ml_prev[s].d * __expf(row_ml_prev[s].m - row_ml_new[s].m) * O[qo_offset + (j + s) * d + k + t] + __expf(row_ml[s].m - row_ml_new[s].m) * s_O[s * Bd + t]);
                    }
                }
            }

// 写入当前 Bc 组的 l 和 m
#pragma unroll
            for (int k = threadIdx.x; k < Br; k += blockDim.x)
            {
                l[lm_offset + j + k] = row_ml_new[k].d;
                m[lm_offset + j + k] = row_ml_new[k].m;
            }
            __syncthreads();
        }
    }
}
```

## 7 性能对比
在 NVIDIA RTX 4060 上的测试数据（单位：ms），注意表中的 v1、v2、v3 并不是 Flash Attention 源项目的版本，而是笔者自己实现的 CUDA Kernel 的优化版本，不能混为一谈。

|`[bs, nh, N, M, d]`|Baseline|Minimal|v1|v2|v3|
|:---:|:---:|:---:|:---:|:---:|:---:|
|[32, 8, 256, 256, 256]|46.7941|741.067|104.727|127.366|12.4245|
|[32, 8, 256, 256, 1024]|56.871|9544.46|618.695|542.985|48.8425|
|[32, 8, 1024, 1024, 256]|92.6972|11343.9|1524.32|2026.06|167.241|
|[32, 8, 1024, 1024, 1024]|232.265|153121|10134.3|8636.17|707.792|

可见，整体来看：
- 序列长度较小时，v3 Kernel 性能是最好的，当序列长度扩大后，逐渐落后于 Baseline；
- v3 相比于 Flash Attention Minimal 性能提升巨大，且随着数据量的扩大甚至会有数百倍的差距。
- 当然，Baseline 性能比较好的原因，主要在于 cuBLAS 库底层针对不同 GEMM 尺寸会有自适应匹配最优 Kernel 机制，总能取得近乎最优的性能，这一点自定义 Kernel 很难超越，更不用说还没经过参数调优；
- v3 版本其实还有不少可以优化的点，比如 gemm 计算优化，如，加入双缓冲机制，load_matrix_sync 时进一步优化 bank conflict。再比如，进一步优化计算思路，Flash Attention 原作者其实更新了不少版本，从计算原理上有所优化，笔者还没有时间进一步阅读新的版本。
- 虽然 Baseline 性能比较好，但是其中间变量所占的全局内存（即显存）也不可忽视，相比于 Flash Attention 几乎多了 `bs * nh * N * M * 2 * sizeof(float)` 大小，在 `[32, 8, 1024, 1024, 1024]` 尺寸场景下，显存多占用了 2 GB，相当于 Flash Attention 的 50%，且 Flash Attention 的显存占用不受序列长度制约，这一点是非常重要的。

## 8 小结
本文针对 Flash Attention 提出背景和计算原理进行了介绍，针对开源项目 Flash Attention Minimal 实现思路进行了解析和点评，并逐步优化了 Flash Attention 算子，给出了 3 个版本的 CUDA Kernel 并对实现思路进行详细介绍，本文源代码地址如下，有兴趣的读者可以阅读。
> https://github.com/caiwanxianhust/flash-attention-opt