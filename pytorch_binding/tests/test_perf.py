"""性能对比:flash_attn vs 朴素 PyTorch attention"""
import torch
import torch.nn.functional as F
import time
from flash_attn import flash_attention

# ============================================================
# 朴素 baseline(标准 PyTorch 实现)
# ============================================================
def naive_attention(Q, K, V):
    D = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)

def pytorch_attention(Q, K, V):
    """PyTorch 2.0+ 内置的 SDPA,带 flash 加速"""
    return F.scaled_dot_product_attention(Q, K, V)

# ============================================================
# CUDA 计时工具
# ============================================================
def cuda_benchmark(fn, *args, warmup=10, iters=100):
    """CUDA 事件计时,比 time.time() 精确"""
    # warmup(让 GPU 进稳态)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # 正式计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # ms/iter

# ============================================================
# 单个 case 测速
# ============================================================
def bench_case(B, H, N, M, D, dtype=torch.float16):
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    K = torch.randn(B, H, M, D, device="cuda", dtype=dtype)
    V = torch.randn(B, H, M, D, device="cuda", dtype=dtype)

    t_naive = cuda_benchmark(naive_attention, Q, K, V)
    t_sdpa = cuda_benchmark(pytorch_attention, Q, K, V)
    t_flash = cuda_benchmark(flash_attention, Q, K, V)

    speedup_vs_naive = t_naive / t_flash
    speedup_vs_sdpa = t_sdpa / t_flash

    print(
        f"[B={B:>2} H={H:>2} N={N:>4} M={M:>4} D={D:>3}] "
        f"naive={t_naive:>7.3f}ms  "
        f"sdpa={t_sdpa:>7.3f}ms  "
        f"flash={t_flash:>7.3f}ms  "
        f"| vs naive: {speedup_vs_naive:>5.2f}x  "
        f"vs sdpa: {speedup_vs_sdpa:>5.2f}x"
    )
    return t_naive, t_sdpa, t_flash

# ============================================================
# 多种规模扫一遍
# ============================================================
def main():
    print("=" * 100)
    print(f"{'Config':<35} {'naive':>10} {'sdpa':>10} {'flash':>10} {'vs naive':>12} {'vs sdpa':>12}")
    print("=" * 100)

    # 典型 attention 规模
    cases = [
        # (B, H, N,  M,  D)
        (1,  8, 128,  128,  64),    # 小
        (1,  8, 512,  512,  64),    # 中
        (1,  8, 1024, 1024, 64),    # 大
        (4,  8, 512,  512,  64),    # 多 batch
        (2, 16, 512,  512,  64),    # 多 head
        (1, 12, 1024, 1024, 64),    # 长序列
        (1,  8, 512,  512, 128),    # 大 head_dim
        (2,  8, 2048, 2048, 64),    # 超长序列(看显存占用)
        (2,  8, 2048, 8192, 64),    # 超长序列(看显存占用)
        (2,  8, 8192, 2048, 64),    # 超长序列(看显存占用)
    ]

    for B, H, N, M, D in cases:
        bench_case(B, H, N, M, D)

    print("=" * 100)

if __name__ == "__main__":
    main()
