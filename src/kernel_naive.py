"""
Naive single-block Triton decode attention kernel.

Grid: (batch, num_q_heads).
Each CTA reads the entire KV for its (batch, head) pair, accumulating with
online softmax.  No split across the KV dimension.

GQA is handled by kv_head_idx = q_head_idx // GROUP_SIZE; KV is never
replicated in memory.

Accumulator and softmax statistics are fp32 throughout.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _naive_decode_kernel(
    # ── Input pointers ───────────────────────────────────────────────────
    Q_ptr,            # (B, H_q, D)          fp16
    KV_ptr,           # (N_p, 2, P, H_kv, D) fp16
    KV_indptr_ptr,    # (B+1,)                int32
    KV_indices_ptr,   # (total_pages,)        int32
    KV_lastlen_ptr,   # (B,)                  int32
    # ── Output pointer ───────────────────────────────────────────────────
    O_ptr,            # (B, H_q, D)           fp16
    # ── Strides for Q ────────────────────────────────────────────────────
    stride_qb, stride_qh, stride_qd,
    # ── Strides for kv_data (N_p, 2, P, H_kv, D) ────────────────────────
    stride_kvp, stride_kvr, stride_kvs, stride_kvh, stride_kvd,
    # ── Strides for O ────────────────────────────────────────────────────
    stride_ob, stride_oh, stride_od,
    # ── Scalar runtime params ────────────────────────────────────────────
    head_dim,
    scale,
    # ── Compile-time constants ───────────────────────────────────────────
    BLOCK_D: tl.constexpr,   # next_power_of_2(head_dim)
    PAGE_SIZE: tl.constexpr, # tokens per page
    GROUP_SIZE: tl.constexpr, # H_q / H_kv
):
    batch_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // GROUP_SIZE

    # ── Load Q ─────────────────────────────────────────────────────────
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    q_base = Q_ptr + batch_idx * stride_qb + q_head_idx * stride_qh
    q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
    q = q * scale

    # ── Page range for this sequence ───────────────────────────────────
    start_ptr = tl.load(KV_indptr_ptr + batch_idx)
    end_ptr   = tl.load(KV_indptr_ptr + batch_idx + 1)
    last_len  = tl.load(KV_lastlen_ptr + batch_idx)
    num_pages = end_ptr - start_ptr

    # ── Online softmax state ───────────────────────────────────────────
    m_i  = -1e9   # running row-max
    l_i  = 0.0    # running sum-of-exp
    acc  = tl.zeros([BLOCK_D], dtype=tl.float32)

    offs_ps = tl.arange(0, PAGE_SIZE)

    # ── Main loop over pages ───────────────────────────────────────────
    for p in range(0, num_pages):
        page_idx   = tl.load(KV_indices_ptr + start_ptr + p)
        is_last    = p == num_pages - 1
        valid_toks = tl.where(is_last, last_len, PAGE_SIZE)
        tok_mask   = offs_ps < valid_toks

        # K block: [PAGE_SIZE, BLOCK_D]
        k_base = (KV_ptr
                  + page_idx * stride_kvp
                  + 0 * stride_kvr
                  + kv_head_idx * stride_kvh)
        k_ptrs = (k_base
                  + offs_ps[:, None] * stride_kvs
                  + offs_d[None, :] * stride_kvd)
        k = tl.load(k_ptrs,
                    mask=tok_mask[:, None] & d_mask[None, :],
                    other=0.0).to(tl.float32)

        # QK dot product: [PAGE_SIZE]
        qk = tl.sum(q[None, :] * k, axis=1)
        qk = tl.where(tok_mask, qk, -1e9)

        # Block-level softmax stats
        m_block = tl.max(qk, axis=0)
        p_exp   = tl.exp(qk - m_block)           # [PAGE_SIZE]
        l_block = tl.sum(p_exp, axis=0)

        # V block: [PAGE_SIZE, BLOCK_D]
        v_base = (KV_ptr
                  + page_idx * stride_kvp
                  + 1 * stride_kvr
                  + kv_head_idx * stride_kvh)
        v_ptrs = (v_base
                  + offs_ps[:, None] * stride_kvs
                  + offs_d[None, :] * stride_kvd)
        v = tl.load(v_ptrs,
                    mask=tok_mask[:, None] & d_mask[None, :],
                    other=0.0).to(tl.float32)

        # Partial accumulator: [BLOCK_D] = Σ p_exp[i] * v[i,:]
        acc_block = tl.sum(p_exp[:, None] * v, axis=0)

        # Merge into global running state
        m_new = tl.maximum(m_i, m_block)
        alpha  = tl.exp(m_i - m_new)
        beta   = tl.exp(m_block - m_new)
        l_i    = alpha * l_i + beta * l_block
        acc    = alpha * acc  + beta * acc_block
        m_i    = m_new

    # ── Normalise and store ────────────────────────────────────────────
    acc = acc / l_i

    o_base = O_ptr + batch_idx * stride_ob + q_head_idx * stride_oh
    # Triton casts fp32 -> fp16 implicitly when the pointer type is fp16.
    tl.store(o_base + offs_d * stride_od, acc, mask=d_mask)


# ── Python wrapper ─────────────────────────────────────────────────────────


def decode_attention_naive(
    q: torch.Tensor,            # (B, H_q, D)
    kv_data: torch.Tensor,      # (N_p, 2, P, H_kv, D)
    kv_indptr: torch.Tensor,    # (B+1,) int32
    kv_indices: torch.Tensor,   # (total_pages,) int32
    kv_last_page_len: torch.Tensor,  # (B,) int32
) -> torch.Tensor:
    """
    Naive single-block Triton decode attention.

    Reads the full KV sequence inside one CTA per (batch, q_head) pair.
    Compatible with FlashInfer's NHD paged KV layout.
    """
    batch, num_q_heads, head_dim = q.shape
    _, _, page_size, num_kv_heads, _ = kv_data.shape
    group_size = num_q_heads // num_kv_heads

    out = torch.empty_like(q)
    scale = float(head_dim ** -0.5)
    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (batch, num_q_heads)

    _naive_decode_kernel[grid](
        q, kv_data, kv_indptr, kv_indices, kv_last_page_len, out,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2),
        # KV strides
        kv_data.stride(0), kv_data.stride(1), kv_data.stride(2),
        kv_data.stride(3), kv_data.stride(4),
        # O strides
        out.stride(0), out.stride(1), out.stride(2),
        # Scalars
        head_dim=head_dim,
        scale=scale,
        # Constexpr
        BLOCK_D=BLOCK_D,
        PAGE_SIZE=page_size,
        GROUP_SIZE=group_size,
    )

    return out
