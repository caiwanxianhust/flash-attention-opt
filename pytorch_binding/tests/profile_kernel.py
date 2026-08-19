"""
profile_kernel.py - 单次 kernel 调用,触发 ncu profile

用 NVTX 标记 flash_attention 调用范围,这样 ncu 能精确 profile。
"""

import torch
import nvtx

from flash_attn import flash_attention

def main():
    batch_size = 2
    num_heads = 8
    seq_len = 2048
    head_dim = 64

    print(f"Config: B={batch_size} H={num_heads} N={seq_len} D={head_dim}")

    # 准备输入
    q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    device="cuda", dtype=torch.float16)

    # warmup
    print("Warmup (20 iterations)...")
    for _ in range(20):
        flash_attention(q, k, v)
    torch.cuda.synchronize()

    # profile 区间
    print("Starting profiled iteration...")
    torch.cuda.synchronize()
    with nvtx.annotate("FLASH_PROFILE", color="red"):
        out = flash_attention(q, k, v)
    torch.cuda.synchronize()

    print(f"Done. Output shape: {out.shape}, dtype: {out.dtype}")

if __name__ == "__main__":
    main()
