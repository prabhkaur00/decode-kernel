"""
NVTX-annotated profiling script for the naive CUDA decode attention kernel.

Kernel logic is unchanged — NVTX ranges wrap Python-level extension calls.

Usage:
    nsys profile --trace=cuda,nvtx --force-overwrite=true \
        -o results/cuda_naive_nvtx \
        python bench/profile_cuda_naive_nvtx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from layout import synthesize
from cuda.build import get_naive_ext

BATCH        = 1
CTX          = 8192
NUM_Q_HEADS  = 32
NUM_KV_HEADS = 8
HEAD_DIM     = 128
PAGE_SIZE    = 16
DTYPE        = torch.float16
DEVICE       = "cuda"
WARMUP       = 5
TIMED        = 20


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    ext = get_naive_ext(verbose=False)

    q, kv_data, kv_indptr, kv_indices, kv_last_page_len, _ = synthesize(
        batch=BATCH, context_length=CTX,
        num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM, page_size=PAGE_SIZE,
        dtype=DTYPE, device=DEVICE,
    )

    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"Q   : {tuple(q.shape)}  KV: {tuple(kv_data.shape)}")

    # ── Warmup: JIT compile + GPU warm-up outside the profiled region ─────
    nvtx.range_push("warmup")
    for _ in range(WARMUP):
        ext.decode_attention_naive(q, kv_data, kv_indptr, kv_indices, kv_last_page_len)
    torch.cuda.synchronize()
    nvtx.range_pop()

    # ── Profiled iterations ───────────────────────────────────────────────
    nvtx.range_push("naive_profile_region")
    for i in range(TIMED):
        nvtx.range_push(f"naive_iter_{i}")
        ext.decode_attention_naive(q, kv_data, kv_indptr, kv_indices, kv_last_page_len)
        torch.cuda.synchronize()
        nvtx.range_pop()
    nvtx.range_pop()

    print("done")


if __name__ == "__main__":
    main()
