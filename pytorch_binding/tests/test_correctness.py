"""验证 flash_attn 与朴素 PyTorch attention 的数值一致性"""
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func





def compare_tensors(tensor1, tensor2, atol=1e-3, rtol=1e-5, max_print=10):
    """
    比较两个张量，并打印所有不匹配元素的信息
    
    Args:
        tensor1, tensor2: 要比较的张量
        atol: 绝对容差
        rtol: 相对容差
        max_print: 最多打印多少个不匹配项
    
    Returns:
        bool: 如果所有元素都匹配则返回True
    """
    # 检查形状是否一致
    if tensor1.shape != tensor2.shape:
        print(f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
        return False
        
    # 计算差异
    close = torch.isclose(tensor1, tensor2, atol=atol, rtol=rtol)
    
    if close.all():
        print("All tests passed!")
        return True
    else:
        # 找到不匹配的位置
        not_close = ~close
        mismatch_indices = torch.nonzero(not_close)
        
        print(f"Found {mismatch_indices.shape[0]} mismatched elements out of {tensor1.numel()} total elements")
        print("\nFirst {} mismatches:".format(min(max_print, mismatch_indices.shape[0])))
        print("Index\t\tComputed\tExpected\tDifference")
        print("-" * 60)
        
        for i, idx in enumerate(mismatch_indices[:max_print]):
            idx_tuple = tuple(idx.tolist())
            computed_val = tensor1[idx_tuple].item()
            expected_val = tensor2[idx_tuple].item()
            diff = abs(computed_val - expected_val)
            
            print(f"{str(idx_tuple):15} {computed_val:10.6f} \t{expected_val:10.6f} \t{diff:10.6f}")
            
        # 显示最大差异
        abs_diff = torch.abs(tensor1 - tensor2)
        max_diff_idx = torch.argmax(abs_diff)
        max_diff_val = abs_diff.flatten()[max_diff_idx].item()
        
        # 将扁平索引转换为多维索引
        max_diff_multi_idx = torch.unravel_index(max_diff_idx, tensor1.shape) if hasattr(torch, 'unravel_index') else \
                            tuple((max_diff_idx // tensor1.stride(dim)) % tensor1.size(dim) for dim in range(tensor1.dim()))
        
        print(f"\nMaximum difference: {max_diff_val:.6f}")
        print(f"At index: {max_diff_multi_idx}")
        print(f"Computed value: {tensor1[max_diff_multi_idx].item():.6f}")
        print(f"Expected value: {tensor2[max_diff_multi_idx].item():.6f}")
        
        return False




# ============================================================
# 参考实现(朴素 attention,用 PyTorch 内置算子写)
# ============================================================
def naive_attention(Q, K, V):
    """
    朴素 attention,作为 correctness 参考

    Args:
        Q: [B, H, N, D] fp16/fp32 CUDA tensor
        K: [B, H, M, D]
        V: [B, H, M, D]

    Returns:
        O: [B, H, N, D]
    """
    D = Q.shape[-1]
    scale = 1.0 / (D ** 0.5)

    # [B, H, N, D] @ [B, H, D, M] -> [B, H, N, M]
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # softmax over M dimension
    attn = F.softmax(scores, dim=-1)

    # [B, H, N, M] @ [B, H, M, D] -> [B, H, N, D]
    out = torch.matmul(attn, V)
    return out

# ============================================================
# 测试用例
# ============================================================
def _run_test(B, H, N, M, D, dtype=torch.float16, atol=0.05, rtol=0.05):
    """跑一次 correctness 测试"""
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    K = torch.randn(B, H, M, D, device="cuda", dtype=dtype)
    V = torch.randn(B, H, M, D, device="cuda", dtype=dtype)

    # 朴素参考实现(用 fp32 算,降低参考误差)
    ref = naive_attention(Q.float(), K.float(), V.float()).to(dtype)

    # 调你的 binding
    out = flash_attn_func(Q, K, V)

    # 校验
    compare_tensors(out, ref, atol, rtol)
    diff = (out.float() - ref.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel = mean_abs / (ref.float().abs().mean().item() + 1e-6)

    print(f"[B={B} H={H} N={N} M={M} D={D}] "
          f"max_abs={max_abs:.4f}, mean_abs={mean_abs:.4f}, rel={rel:.4f}")

    assert out.shape == ref.shape, \
        f"shape mismatch: out={out.shape}, ref={ref.shape}"
    assert out.dtype == ref.dtype, \
        f"dtype mismatch: out={out.dtype}, ref={ref.dtype}"
    assert max_abs < atol, \
        f"数值误差过大: max_abs={max_abs:.4f} > {atol}"

def test_small():
    """最小规模 sanity check"""
    _run_test(B=1, H=1, N=64, M=16, D=16)

def test_typical():
    """典型规模:小 batch,中等序列"""
    _run_test(B=2, H=4, N=128, M=128, D=64)

def test_long_sequence():
    """长序列"""
    _run_test(B=1, H=2, N=512, M=512, D=64)

def test_asymmetric():
    """Q 和 K/V 序列长度不同(N != M)"""
    _run_test(B=1, H=2, N=128, M=256, D=64)

def test_d64():
    """head_dim = 64"""
    _run_test(B=2, H=4, N=128, M=128, D=64)

def test_d128():
    """head_dim = 128"""
    _run_test(B=2, H=4, N=128, M=128, D=128)

# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FlashAttention Correctness Tests")
    print("=" * 60)

    test_small()
    test_typical()
    test_long_sequence()
    test_asymmetric()
    test_d64()
    test_d128()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


# houlai@houlai:~/codespace/flash_attention/pytorch_binding$ python3 -m  tests.test_correctness