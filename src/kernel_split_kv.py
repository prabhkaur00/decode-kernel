"""
Split-KV Triton decode attention kernel.

Two-pass design:

Pass 1 — Partition kernel
  Grid: (batch, num_q_heads, SPLIT_KV)
  Each CTA owns a contiguous slice of KV pages for its (batch, q_head) and
  writes partial (m_i, l_i, O_i) to fp32 scratch buffers.

Pass 2 — Reduction kernel
  Grid: (batch, num_q_heads)
  Each CTA reads SPLIT_KV partial results and merges with the online softmax
  identity:

      m   = max_i  m_i
      l   = Σ_i  exp(m_i − m) · l_i
      O   = (1/l) · Σ_i  exp(m_i − m) · O_i

The merge state (m_i, l_i, O_i) is fp32 regardless of the input dtype.
Storing it in fp16 is the most common source of silent numerical error when
SPLIT_KV is large; the error scales with the number of splits.

GQA is handled via kv_head_idx = q_head_idx // GROUP_SIZE.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


# ── Pass 1: partition kernel ───────────────────────────────────────────────


@triton.jit
def _partition_kernel(
    # ── Input pointers ───────────────────────────────────────────────────
    Q_ptr,
    KV_ptr,
    KV_indptr_ptr,
    KV_indices_ptr,
    KV_lastlen_ptr,
    # ── Scratch output pointers (fp32) ────────────────────────────────────
    Partial_O_ptr,   # (B, H_q, SPLIT_KV, D)
    Partial_m_ptr,   # (B, H_q, SPLIT_KV)
    Partial_l_ptr,   # (B, H_q, SPLIT_KV)
    # ── Strides for Q ────────────────────────────────────────────────────
    stride_qb, stride_qh, stride_qd,
    # ── Strides for kv_data ──────────────────────────────────────────────
    stride_kvp, stride_kvr, stride_kvs, stride_kvh, stride_kvd,
    # ── Strides for partial_O ────────────────────────────────────────────
    stride_pob, stride_poh, stride_pos, stride_pod,
    # ── Strides for partial_m and partial_l (same shape) ────────────────
    stride_pmb, stride_pmh, stride_pms,
    # ── Scalar runtime params ────────────────────────────────────────────
    head_dim,
    scale,
    # ── Compile-time constants ───────────────────────────────────────────
    SPLIT_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    batch_idx  = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    split_idx  = tl.program_id(2)
    kv_head_idx = q_head_idx // GROUP_SIZE

    # ── Q ──────────────────────────────────────────────────────────────
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    q_base = Q_ptr + batch_idx * stride_qb + q_head_idx * stride_qh
    q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
    q = q * scale

    # ── Page range ─────────────────────────────────────────────────────
    seq_start  = tl.load(KV_indptr_ptr + batch_idx)
    seq_end    = tl.load(KV_indptr_ptr + batch_idx + 1)
    last_len   = tl.load(KV_lastlen_ptr + batch_idx)
    num_pages  = seq_end - seq_start

    # Assign pages to this split: [split_start, split_end)
    pages_per_split = (num_pages + SPLIT_KV - 1) // SPLIT_KV
    split_start = split_idx * pages_per_split
    split_end   = tl.minimum(split_start + pages_per_split, num_pages)
    my_num_pages = split_end - split_start

    # ── Online softmax accumulators (fp32) ─────────────────────────────
    m_i  = -1e9
    l_i  = 0.0
    acc  = tl.zeros([BLOCK_D], dtype=tl.float32)

    offs_ps = tl.arange(0, PAGE_SIZE)

    for p in range(0, my_num_pages):
        global_page_offset = split_start + p
        page_idx   = tl.load(KV_indices_ptr + seq_start + global_page_offset)
        is_last    = global_page_offset == num_pages - 1
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

        # QK scores: [PAGE_SIZE]
        qk = tl.sum(q[None, :] * k, axis=1)
        qk = tl.where(tok_mask, qk, -1e9)

        m_block = tl.max(qk, axis=0)
        p_exp   = tl.exp(qk - m_block)
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

        acc_block = tl.sum(p_exp[:, None] * v, axis=0)

        m_new = tl.maximum(m_i, m_block)
        alpha  = tl.exp(m_i - m_new)
        beta   = tl.exp(m_block - m_new)
        l_i    = alpha * l_i + beta * l_block
        acc    = alpha * acc  + beta * acc_block
        m_i    = m_new

    # ── Write partial results to scratch (fp32) ────────────────────────
    po_base = (Partial_O_ptr
               + batch_idx * stride_pob
               + q_head_idx * stride_poh
               + split_idx * stride_pos)
    pm_ptr  = (Partial_m_ptr
               + batch_idx * stride_pmb
               + q_head_idx * stride_pmh
               + split_idx * stride_pms)
    pl_ptr  = (Partial_l_ptr
               + batch_idx * stride_pmb
               + q_head_idx * stride_pmh
               + split_idx * stride_pms)

    tl.store(po_base + offs_d * stride_pod, acc, mask=d_mask)
    tl.store(pm_ptr, m_i)
    tl.store(pl_ptr, l_i)


# ── Pass 2: reduction kernel ───────────────────────────────────────────────


@triton.jit
def _reduce_kernel(
    # ── Scratch inputs (fp32) ────────────────────────────────────────────
    Partial_O_ptr,
    Partial_m_ptr,
    Partial_l_ptr,
    # ── Final output ─────────────────────────────────────────────────────
    O_ptr,
    # ── Strides for partial_O ────────────────────────────────────────────
    stride_pob, stride_poh, stride_pos, stride_pod,
    # ── Strides for partial_m / partial_l ────────────────────────────────
    stride_pmb, stride_pmh, stride_pms,
    # ── Strides for O ────────────────────────────────────────────────────
    stride_ob, stride_oh, stride_od,
    # ── Scalar params ────────────────────────────────────────────────────
    head_dim,
    # ── Compile-time constants ───────────────────────────────────────────
    SPLIT_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx  = tl.program_id(0)
    q_head_idx = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    # ── Find global row-max across all splits ──────────────────────────
    m_global = -1e9
    for s in tl.static_range(SPLIT_KV):
        m_s = tl.load(
            Partial_m_ptr
            + batch_idx * stride_pmb
            + q_head_idx * stride_pmh
            + s * stride_pms
        )
        m_global = tl.maximum(m_global, m_s)

    # ── Accumulate weighted partial results ────────────────────────────
    l_global   = 0.0
    acc_global = tl.zeros([BLOCK_D], dtype=tl.float32)

    for s in tl.static_range(SPLIT_KV):
        pm_ptr = (Partial_m_ptr
                  + batch_idx * stride_pmb
                  + q_head_idx * stride_pmh
                  + s * stride_pms)
        pl_ptr = (Partial_l_ptr
                  + batch_idx * stride_pmb
                  + q_head_idx * stride_pmh
                  + s * stride_pms)
        po_base = (Partial_O_ptr
                   + batch_idx * stride_pob
                   + q_head_idx * stride_poh
                   + s * stride_pos)

        m_s = tl.load(pm_ptr)
        l_s = tl.load(pl_ptr)
        o_s = tl.load(po_base + offs_d * stride_pod, mask=d_mask, other=0.0)

        w = tl.exp(m_s - m_global)
        l_global   += w * l_s
        acc_global += w * o_s

    # ── Normalise ──────────────────────────────────────────────────────
    acc_global = acc_global / l_global

    o_base = O_ptr + batch_idx * stride_ob + q_head_idx * stride_oh
    tl.store(o_base + offs_d * stride_od, acc_global, mask=d_mask)


# ── Python wrapper ─────────────────────────────────────────────────────────


# Cache pre-allocated scratch buffers to avoid re-allocation across timing
# iterations.  Key: (batch, num_q_heads, split_kv, head_dim, device).
_scratch_cache: dict = {}


def decode_attention_split_kv(
    q: torch.Tensor,
    kv_data: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    split_kv: int = 4,
    _scratch: Optional[dict] = None,
) -> torch.Tensor:
    """
    Split-KV Triton decode attention.

    The KV sequence for each (batch, q_head) is partitioned into `split_kv`
    contiguous slices processed in parallel.  A second reduction pass merges
    the partial softmax results.

    Scratch buffers (partial_O, partial_m, partial_l) are allocated in fp32.

    Args:
        split_kv: number of KV partitions.  1 ≡ naive single-CTA mode.
        _scratch: optional dict for pre-allocated scratch buffers (used by
                  the benchmark harness to avoid allocation in the hot path).
    """
    batch, num_q_heads, head_dim = q.shape
    _, _, page_size, num_kv_heads, _ = kv_data.shape
    group_size = num_q_heads // num_kv_heads

    out = torch.empty_like(q)
    scale = float(head_dim ** -0.5)
    BLOCK_D = triton.next_power_of_2(head_dim)

    # Allocate or reuse scratch buffers
    scratch_key = (batch, num_q_heads, split_kv, head_dim, str(q.device))
    if _scratch is not None and scratch_key in _scratch:
        partial_o, partial_m, partial_l = _scratch[scratch_key]
    else:
        partial_o = torch.empty(
            batch, num_q_heads, split_kv, head_dim,
            dtype=torch.float32, device=q.device
        )
        partial_m = torch.empty(
            batch, num_q_heads, split_kv,
            dtype=torch.float32, device=q.device
        )
        partial_l = torch.empty_like(partial_m)
        if _scratch is not None:
            _scratch[scratch_key] = (partial_o, partial_m, partial_l)

    # ── Pass 1: partition ──────────────────────────────────────────────
    grid_part = (batch, num_q_heads, split_kv)
    _partition_kernel[grid_part](
        q, kv_data, kv_indptr, kv_indices, kv_last_page_len,
        partial_o, partial_m, partial_l,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2),
        # KV strides
        kv_data.stride(0), kv_data.stride(1), kv_data.stride(2),
        kv_data.stride(3), kv_data.stride(4),
        # partial_O strides
        partial_o.stride(0), partial_o.stride(1),
        partial_o.stride(2), partial_o.stride(3),
        # partial_m / partial_l strides
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        # scalars
        head_dim=head_dim,
        scale=scale,
        # constexpr
        SPLIT_KV=split_kv,
        BLOCK_D=BLOCK_D,
        PAGE_SIZE=page_size,
        GROUP_SIZE=group_size,
    )

    # ── Pass 2: reduction ──────────────────────────────────────────────
    grid_red = (batch, num_q_heads)
    _reduce_kernel[grid_red](
        partial_o, partial_m, partial_l, out,
        # partial_O strides
        partial_o.stride(0), partial_o.stride(1),
        partial_o.stride(2), partial_o.stride(3),
        # partial_m / partial_l strides
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        # O strides
        out.stride(0), out.stride(1), out.stride(2),
        # scalar
        head_dim=head_dim,
        # constexpr
        SPLIT_KV=split_kv,
        BLOCK_D=BLOCK_D,
    )

    return out
