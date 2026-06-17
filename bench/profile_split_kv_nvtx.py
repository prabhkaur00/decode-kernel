"""
NVTX-annotated profiling script for the split-KV decode attention kernel.

Kernel logic is unchanged — NVTX ranges wrap each of the two Triton kernel
dispatches (partition pass and reduction pass) separately so nsys shows them
as distinct labelled bands on the GPU timeline.

Usage:
    nsys profile --trace=cuda,nvtx -o results/split_kv_nvtx \
        python bench/profile_split_kv_nvtx.py

    # To vary split_kv:
    python bench/profile_split_kv_nvtx.py --split-kv 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from layout import synthesize
# Import the two Triton kernels directly so we can place NVTX ranges around
# each pass independently.  The wrapper logic below is identical to
# decode_attention_split_kv() in kernel_split_kv.py.
from kernel_split_kv import _partition_kernel, _reduce_kernel

BATCH        = 1
CTX          = 8192
PAGE_SIZE    = 16
NUM_Q_HEADS  = 32
NUM_KV_HEADS = 8
HEAD_DIM     = 128
DTYPE        = torch.float16
DEVICE       = "cuda"
WARMUP       = 5
TIMED        = 10


def _run_split_kv_nvtx(
    q: torch.Tensor,
    kv_data: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    split_kv: int,
    out: torch.Tensor,
    partial_o: torch.Tensor,
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    iteration: Optional[int] = None,
) -> None:
    """
    Identical dispatch to decode_attention_split_kv(), split into two
    labelled NVTX ranges so each pass is visible separately in nsys.
    Pre-allocated scratch buffers are passed in to keep the hot path clean.
    """
    batch, num_q_heads, head_dim = q.shape
    _, _, page_size, num_kv_heads, _ = kv_data.shape
    group_size = num_q_heads // num_kv_heads
    scale = float(head_dim ** -0.5)
    BLOCK_D = triton.next_power_of_2(head_dim)

    iter_label = "" if iteration is None else f"_iter{iteration}"

    # ── Pass 1: partition ─────────────────────────────────────────────────
    torch.cuda.nvtx.range_push(f"split_kv_partition{iter_label}")
    grid_part = (batch, num_q_heads, split_kv)
    _partition_kernel[grid_part](
        q, kv_data, kv_indptr, kv_indices, kv_last_page_len,
        partial_o, partial_m, partial_l,
        q.stride(0), q.stride(1), q.stride(2),
        kv_data.stride(0), kv_data.stride(1), kv_data.stride(2),
        kv_data.stride(3), kv_data.stride(4),
        partial_o.stride(0), partial_o.stride(1),
        partial_o.stride(2), partial_o.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        head_dim=head_dim,
        scale=scale,
        SPLIT_KV=split_kv,
        BLOCK_D=BLOCK_D,
        PAGE_SIZE=page_size,
        GROUP_SIZE=group_size,
    )
    torch.cuda.nvtx.range_pop()

    # ── Pass 2: reduction ─────────────────────────────────────────────────
    torch.cuda.nvtx.range_push(f"split_kv_reduce{iter_label}")
    grid_red = (batch, num_q_heads)
    _reduce_kernel[grid_red](
        partial_o, partial_m, partial_l, out,
        partial_o.stride(0), partial_o.stride(1),
        partial_o.stride(2), partial_o.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        head_dim=head_dim,
        SPLIT_KV=split_kv,
        BLOCK_D=BLOCK_D,
    )
    torch.cuda.nvtx.range_pop()


def main(split_kv: int):
    q, kv_data, kv_indptr, kv_indices, kv_last_page_len, _ = synthesize(
        batch=BATCH,
        context_length=CTX,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        dtype=DTYPE,
        device=DEVICE,
    )

    batch, num_q_heads, head_dim = q.shape
    out       = torch.empty_like(q)
    partial_o = torch.empty(batch, num_q_heads, split_kv, head_dim,
                            dtype=torch.float32, device=DEVICE)
    partial_m = torch.empty(batch, num_q_heads, split_kv,
                            dtype=torch.float32, device=DEVICE)
    partial_l = torch.empty_like(partial_m)

    # ── Warmup: trigger Triton JIT outside the profiled region ────────────
    torch.cuda.nvtx.range_push("warmup")
    for _ in range(WARMUP):
        _run_split_kv_nvtx(
            q, kv_data, kv_indptr, kv_indices, kv_last_page_len,
            split_kv, out, partial_o, partial_m, partial_l,
        )
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # ── Timed / profiled iterations ───────────────────────────────────────
    torch.cuda.nvtx.range_push("profile_region")
    for i in range(TIMED):
        _run_split_kv_nvtx(
            q, kv_data, kv_indptr, kv_indices, kv_last_page_len,
            split_kv, out, partial_o, partial_m, partial_l,
            iteration=i,
        )
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    parser = argparse.ArgumentParser()
    parser.add_argument("--split-kv", type=int, default=8,
                        help="Number of KV partitions (default: 8)")
    args = parser.parse_args()
    main(args.split_kv)
