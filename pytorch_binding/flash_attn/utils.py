"""flash_attn.utils - tensor 检查和转换"""
import torch

def _check_inputs(q, k, v):
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("Q/K/V must be CUDA tensors")
    if not (q.dim() == k.dim() == v.dim() == 4):
        raise ValueError(
            f"Q/K/V must be 4-D [B, H, N/M, D], "
            f"got Q={tuple(q.shape)}, K={tuple(k.shape)}, V={tuple(v.shape)}"
        )
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[3] != k.shape[3]:
        raise ValueError(
            f"Q/K/V batch/head/dim mismatch: "
            f"Q={tuple(q.shape)}, K={tuple(k.shape)}, V={tuple(v.shape)}"
        )

def _ensure_contiguous(q, k, v):
    return (
        q.contiguous() if not q.is_contiguous() else q,
        k.contiguous() if not k.is_contiguous() else k,
        v.contiguous() if not v.is_contiguous() else v,
    )

