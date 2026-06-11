"""KV layout utilities and FlashInfer-compatible paged tensor synthesis."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class PagedKVLayout:
    page_size: int
    head_dim: int
    num_kv_heads: int
    dtype: torch.dtype = torch.float16

    def validate(self, kv_data, kv_indptr, kv_indices, kv_last_page_len):
        """Raise ValueError on any layout invariant violation."""
        assert kv_data.dim() == 5, "kv_data must be 5-D"
        assert kv_data.shape[1] == 2, "kv_data dim-1 must be 2 (K/V)"
        assert kv_data.shape[2] == self.page_size
        assert kv_data.shape[3] == self.num_kv_heads
        assert kv_data.shape[4] == self.head_dim
        assert kv_indptr[-1] == kv_indices.shape[0], (
            f"kv_indptr[-1]={kv_indptr[-1].item()} != "
            f"kv_indices.shape[0]={kv_indices.shape[0]}"
        )
        assert (kv_last_page_len > 0).all(), "kv_last_page_len must be > 0"
        assert (kv_last_page_len <= self.page_size).all(), (
            "kv_last_page_len must be <= page_size"
        )


def build_block_table(
    batch: int,
    context_length: int,
    page_size: int,
    device: str | torch.device = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs kv_indptr, kv_indices, kv_last_page_len for a uniform context.

    All sequences share the same context_length.  Pages are never aliased:
    sequence b owns physical pages [b*K_p, (b+1)*K_p).

    Returns:
        kv_indptr       : (batch + 1,) int32
        kv_indices      : (batch * pages_per_seq,) int32
        kv_last_page_len: (batch,) int32
    """
    pages_per_seq = math.ceil(context_length / page_size)
    last_len = context_length - (pages_per_seq - 1) * page_size
    if last_len == 0:
        last_len = page_size

    kv_indptr = torch.arange(
        0, batch + 1, dtype=torch.int32, device=device
    ) * pages_per_seq

    kv_indices = torch.arange(
        0, batch * pages_per_seq, dtype=torch.int32, device=device
    )

    kv_last_page_len = torch.full(
        (batch,), last_len, dtype=torch.int32, device=device
    )

    return kv_indptr, kv_indices, kv_last_page_len


def synthesize(
    batch: int,
    context_length: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int = 16,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cuda",
    seed: int = 42,
) -> Tuple[
    torch.Tensor,  # Q
    torch.Tensor,  # kv_data
    torch.Tensor,  # kv_indptr
    torch.Tensor,  # kv_indices
    torch.Tensor,  # kv_last_page_len
    PagedKVLayout,
]:
    """
    Allocates random Q, K, V tensors in FlashInfer's NHD paged layout.

    kv_data shape: (num_pages, 2, page_size, num_kv_heads, head_dim)

    Returns (Q, kv_data, kv_indptr, kv_indices, kv_last_page_len, layout).
    """
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads})"
        )

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    q = torch.randn(batch, num_q_heads, head_dim,
                    dtype=dtype, device=device, generator=gen)

    kv_indptr, kv_indices, kv_last_page_len = build_block_table(
        batch, context_length, page_size, device
    )

    total_pages = int(kv_indices.shape[0])
    kv_data = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=dtype, device=device, generator=gen,
    )

    layout = PagedKVLayout(
        page_size=page_size,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
    )

    return q, kv_data, kv_indptr, kv_indices, kv_last_page_len, layout
