"""
profile_perf_trace.py - 多次调用,触发 nsys timeline profile

不带 NVTX,适合 nsys 看时间线和 kernel 分布。
"""

import torch

from flash_attn import flash_attention

def main():
    # 多组配置,看不同规模下的表现
    configs = [
        # (B, H, N, D)
        (1, 8, 128, 64),
        (1, 8, 512, 64),
        (1, 8, 1024, 64),
        (1, 8, 2048, 64),
    ]

    for batch_size, num_heads, seq_len, head_dim in configs:
        print(f"\n--- B={batch_size} H={num_heads} N={seq_len} D={head_dim} ---")

        q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                        device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                        device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                        device="cuda", dtype=torch.float16)

        # warmup
        for _ in range(20):
            flash_attention(q, k, v)
        torch.cuda.synchronize()

        # profile50 次
        torch.cuda.synchronize()
        for _ in range(50):
            flash_attention(q, k, v)
        torch.cuda.synchronize()

        print(f"  done")

if __name__ == "__main__":
    main()
