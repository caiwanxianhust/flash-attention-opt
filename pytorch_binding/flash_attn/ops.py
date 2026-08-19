"""flash_attn.ops - 高层 Python API"""
from .utils import _check_inputs, _ensure_contiguous
from ._C import flash_attention


def flash_attn_func(q, k, v):
    """FlashAttention forward.

    Args:
        q: [B, H, N, D]  CUDA tensor
        k: [B, H, M, D]  CUDA tensor
        v: [B, H, M, D]  CUDA tensor

    Returns:
        [B, H, N, D]  CUDA tensor
    """
    _check_inputs(q, k, v)
    q, k, v = _ensure_contiguous(q, k, v)
    return flash_attention(q, k, v)

__all__ = ["flash_attn_func", "flash_attention"]
