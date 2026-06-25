"""
Unified decode attention dispatch (CUDA backend).

Usage:
    from attention import decode_attention
    out = decode_attention(q, kv_data, kv_indptr, kv_indices, kv_last_page_len, split_kv=4)
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _get_cuda_ext():
    _spec = importlib.util.spec_from_file_location(
        "_cuda_build",
        Path(__file__).resolve().parent / "cuda_" / "build.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod.get_cuda_ext()


def decode_attention(
    q: torch.Tensor,
    kv_data: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    split_kv: int = 1,
) -> torch.Tensor:
    """
    Decode attention over a paged KV cache.

    Args:
        q                : query tensor (batch, num_q_heads, head_dim), fp16
        kv_data          : paged KV cache in FlashInfer NHD layout, fp16
        kv_indptr        : CSR row pointer into kv_indices
        kv_indices       : physical page indices per slot
        kv_last_page_len : valid tokens in the last page of each sequence
        split_kv         : KV partitions per (batch, q_head).
                           1 -> naive single-CTA mode.
                           >1 -> split-KV two-pass mode.

    Returns:
        output tensor (batch, num_q_heads, head_dim), same dtype as q.
    """
    ext = _get_cuda_ext()
    if split_kv == 1:
        return ext.decode_attention_naive(
            q, kv_data, kv_indptr, kv_indices, kv_last_page_len
        )
    else:
        return ext.decode_attention_split_kv(
            q, kv_data, kv_indptr, kv_indices, kv_last_page_len, split_kv
        )
