"""
NVTX-annotated profiling script for the split-KV CUDA decode attention kernel.

Two layers of NVTX ranges:
  - Python layer (torch.cuda.nvtx): outer per-iteration range "splitkv_iter_N"
  - C++ layer (nvToolsExt inside split_kv_kernel.cu):
      "cuda_split_kv_partition" — partition kernel launch
      "cuda_split_kv_reduce"    — reduction kernel launch

In nsys you will see all three layers stacked on the CPU timeline, with
decode_attn_partition_kernel and decode_attn_reduce_kernel visible as
separate blocks on the GPU CUDA kernel row below them.

Usage:
    nsys profile --trace=cuda,nvtx --force-overwrite=true \
        -o results/cuda_split_kv_nvtx \
        python bench/profile_cuda_split_kv_nvtx.py

    # To vary split_kv:
    python bench/profile_cuda_split_kv_nvtx.py --split-kv 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from layout import synthesize
from cuda.build import get_split_kv_ext

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


def main(split_kv: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    ext = get_split_kv_ext(verbose=False)

    q, kv_data, kv_indptr, kv_indices, kv_last_page_len, _ = synthesize(
        batch=BATCH, context_length=CTX,
        num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM, page_size=PAGE_SIZE,
        dtype=DTYPE, device=DEVICE,
    )

    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"Q        : {tuple(q.shape)}  KV: {tuple(kv_data.shape)}")
    print(f"split_kv : {split_kv}")

    # ── Warmup: JIT compile + GPU warm-up outside the profiled region ─────
    nvtx.range_push("warmup")
    for _ in range(WARMUP):
        ext.decode_attention_split_kv(
            q, kv_data, kv_indptr, kv_indices, kv_last_page_len, split_kv
        )
    torch.cuda.synchronize()
    nvtx.range_pop()

    # ── Profiled iterations ───────────────────────────────────────────────
    # Each Python-level range "splitkv_iter_N" encloses two C++ NVTX ranges
    # ("cuda_split_kv_partition" and "cuda_split_kv_reduce") that are pushed
    # inside the extension before each kernel launch.
    nvtx.range_push("split_kv_profile_region")
    for i in range(TIMED):
        nvtx.range_push(f"splitkv_iter_{i}")
        ext.decode_attention_split_kv(
            q, kv_data, kv_indptr, kv_indices, kv_last_page_len, split_kv
        )
        torch.cuda.synchronize()
        nvtx.range_pop()
    nvtx.range_pop()

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-kv", type=int, default=8,
                        help="Number of KV partitions (default: 8)")
    args = parser.parse_args()
    main(args.split_kv)
