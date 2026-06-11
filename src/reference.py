"""Pure PyTorch fp32 reference attention and paged-to-dense gathering."""
from __future__ import annotations

from typing import Tuple

import torch


def gather_paged_to_dense(
    kv_data: torch.Tensor,          # (num_pages, 2, page_size, num_kv_heads, head_dim)
    kv_indptr: torch.Tensor,        # (batch + 1,) int32
    kv_indices: torch.Tensor,       # (total_pages,) int32
    kv_last_page_len: torch.Tensor, # (batch,) int32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstructs dense fp32 K, V from the paged layout.

    Returns:
        K: (batch, context_length, num_kv_heads, head_dim) float32  (CPU)
        V: (batch, context_length, num_kv_heads, head_dim) float32  (CPU)
    """
    _, _, page_size, num_kv_heads, head_dim = kv_data.shape
    batch = kv_indptr.shape[0] - 1

    kv_data_cpu = kv_data.cpu().float()
    kv_indptr_cpu = kv_indptr.cpu()
    kv_indices_cpu = kv_indices.cpu()
    kv_last_page_len_cpu = kv_last_page_len.cpu()

    # Compute per-sequence lengths to determine max length
    seq_lens = []
    for b in range(batch):
        start = int(kv_indptr_cpu[b].item())
        end = int(kv_indptr_cpu[b + 1].item())
        n_pages = end - start
        last_len = int(kv_last_page_len_cpu[b].item())
        seq_len = max(0, (n_pages - 1) * page_size + last_len)
        seq_lens.append(seq_len)

    max_len = max(seq_lens) if seq_lens else 0

    K = torch.zeros(batch, max_len, num_kv_heads, head_dim, dtype=torch.float32)
    V = torch.zeros(batch, max_len, num_kv_heads, head_dim, dtype=torch.float32)

    for b in range(batch):
        start = int(kv_indptr_cpu[b].item())
        end = int(kv_indptr_cpu[b + 1].item())
        last_len = int(kv_last_page_len_cpu[b].item())
        pos = 0
        for ptr in range(start, end):
            page = int(kv_indices_cpu[ptr].item())
            is_last = ptr == end - 1
            plen = last_len if is_last else page_size
            K[b, pos : pos + plen] = kv_data_cpu[page, 0, :plen]
            V[b, pos : pos + plen] = kv_data_cpu[page, 1, :plen]
            pos += plen

    return K, V


def reference_attention(
    Q: torch.Tensor,  # (batch, num_q_heads, head_dim)
    K: torch.Tensor,  # (batch, context_length, num_kv_heads, head_dim) — may be CPU
    V: torch.Tensor,  # (batch, context_length, num_kv_heads, head_dim) — may be CPU
) -> torch.Tensor:
    """
    Pure PyTorch fp32 decode attention with GQA support.

    Returns (batch, num_q_heads, head_dim) float32.
    """
    batch, num_q_heads, head_dim = Q.shape
    _, context_length, num_kv_heads, _ = K.shape
    group_size = num_q_heads // num_kv_heads

    # Promote to fp32 on the same device as Q
    dev = Q.device
    Q32 = Q.float()                    # (B, H_q, D)
    K32 = K.float().to(dev)            # (B, S, H_kv, D)
    V32 = V.float().to(dev)            # (B, S, H_kv, D)

    # GQA: expand KV heads to match Q heads
    # (B, S, H_kv, D) -> (B, S, H_q, D)
    K32 = K32.repeat_interleave(group_size, dim=2)
    V32 = V32.repeat_interleave(group_size, dim=2)

    # Reshape for batched matmul: (B, H_q, 1, D) x (B, H_q, D, S)
    Q32 = Q32.unsqueeze(2)                             # (B, H_q, 1, D)
    K32 = K32.permute(0, 2, 3, 1)                     # (B, H_q, D, S)
    V32 = V32.permute(0, 2, 1, 3)                     # (B, H_q, S, D)

    scale = head_dim ** -0.5
    scores = torch.matmul(Q32 * scale, K32)            # (B, H_q, 1, S)
    weights = torch.softmax(scores, dim=-1)            # (B, H_q, 1, S)
    out = torch.matmul(weights, V32).squeeze(2)        # (B, H_q, D)

    return out


def reference_attention_paged(
    Q: torch.Tensor,
    kv_data: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> torch.Tensor:
    """Convenience wrapper: gather paged KV, then run reference attention."""
    K, V = gather_paged_to_dense(kv_data, kv_indptr, kv_indices, kv_last_page_len)
    return reference_attention(Q, K, V)
