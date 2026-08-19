"""flash_attn - PyTorch binding for FlashAttention (CUDA fp16)"""
from ._C import flash_attention
from .ops import flash_attn_func

__all__ = ["flash_attn_func", "flash_attention"]
__version__ = "0.1.0"
