"""
NVTX-annotated profiling script for the naive decode attention kernel.

Kernel logic is unchanged — NVTX ranges are added only around Python-level
kernel dispatches so nsys shows labelled bands on the GPU timeline.

Usage:
    nsys profile --trace=cuda,nvtx -o results/naive_nvtx \
        python bench/profile_naive_nvtx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from layout import synthesize
from kernel_naive import decode_attention_naive

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


def main():
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

    # ── Warmup: trigger Triton JIT outside the profiled region ────────────
    torch.cuda.nvtx.range_push("warmup")
    for _ in range(WARMUP):
        decode_attention_naive(q, kv_data, kv_indptr, kv_indices, kv_last_page_len)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # ── Timed / profiled iterations ───────────────────────────────────────
    torch.cuda.nvtx.range_push("profile_region")
    for i in range(TIMED):
        torch.cuda.nvtx.range_push(f"naive_decode_iter{i}")
        decode_attention_naive(q, kv_data, kv_indptr, kv_indices, kv_last_page_len)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    main()
