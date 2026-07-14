"""
Minimal single-call driver for Nsight Compute (ncu) profiling.

Unlike profile_cuda_split_kv_nvtx.py, this makes exactly one call to the
kernel so `--launch-skip`/`--launch-count` map directly onto the kernels
launched by that call (for split-kv variants: decode_attn_partition_kernel
then decode_attn_reduce_kernel; for naive: a single kernel) with no warmup
calls ahead of them to account for in the skip count.

Shape and kernel are both CLI args so you can point this at whichever
(kernel, batch, ctx, split_kv) regime results/microbench_cuda_*.csv flags
as interesting, without editing the file — e.g. the worst bw_pct_of_peak
row vs. the best one, or comparing v1 against the pipelined variant.

Usage (from repo root, on a GPU runtime):
    # worst case from the CSV: batch=1, split_kv=1, ~0.4% of peak bw
    ncu --set full --kernel-name regex:decode \
        --launch-skip 0 --launch-count 2 \
        -o decode_ncu_worst -f \
        python bench/mini_driver.py --kernel split_kv --batch 1 --ctx 8192 --split-kv 1

    # best case from the CSV: batch=16, split_kv=16, ~13.75% of peak bw
    ncu --set full --kernel-name regex:decode \
        --launch-skip 0 --launch-count 2 \
        -o decode_ncu_best -f \
        python bench/mini_driver.py --kernel split_kv --batch 16 --ctx 65536 --split-kv 16

    # v2.5 kernel (v2 + QK scores in shared mem instead of registers), same shape
    ncu --set full --kernel-name regex:decode \
        --launch-skip 0 --launch-count 2 \
        -o decode_ncu_v2_5 -f \
        python bench/mini_driver.py --kernel split_kv_v2_5 --batch 16 --ctx 65536 --split-kv 16

    # pipelined kernel, same shape (requires SM 80+)
    ncu --set full --kernel-name regex:decode \
        --launch-skip 0 --launch-count 2 \
        -o decode_ncu_pipelined -f \
        python bench/mini_driver.py --kernel split_kv_pipelined --batch 16 --ctx 65536 --split-kv 16

    # v3.5 kernel (v3 pipelining + v2 group fusion), same shape (requires SM 80+)
    ncu --set full --kernel-name regex:decode \
        --launch-skip 0 --launch-count 2 \
        -o decode_ncu_v3_5 -f \
        python bench/mini_driver.py --kernel split_kv_v3_5 --batch 16 --ctx 65536 --split-kv 16

    # v4 kernel (pipelined + reduced register pressure), same shape (requires SM 80+)
    ncu --set full --kernel-name regex:decode \
        --launch-skip 0 --launch-count 2 \
        -o decode_ncu_v4 -f \
        python bench/mini_driver.py --kernel split_kv_v4 --batch 16 --ctx 65536 --split-kv 16

    # naive kernel launches a single "decode" kernel, so use --launch-count 1
    ncu --set full --kernel-name regex:decode \
        --launch-skip 0 --launch-count 1 \
        -o decode_ncu_naive -f \
        python bench/mini_driver.py --kernel naive --batch 1 --ctx 8192
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from layout import synthesize
from cuda_.build import (
    get_naive_ext,
    get_split_kv_ext,
    get_split_kv_v2_ext,
    get_split_kv_v2_5_ext,
    get_split_kv_pipelined_ext,
    get_split_kv_v3_5_ext,
    get_split_kv_v4_ext,
)

NUM_Q_HEADS  = 32
NUM_KV_HEADS = 8
HEAD_DIM     = 128
PAGE_SIZE    = 16
DTYPE        = torch.float16
DEVICE       = "cuda"

# name -> (loader, takes_split_kv). `loader` returns an extension whose
# call is `.decode_attention_naive(...)` (no split_kv) or
# `.decode_attention_split_kv(..., split_kv)` (all split-kv variants share
# that Python-facing method name regardless of which .cu file backs it).
KERNELS = {
    "naive":              (get_naive_ext,              False),
    "split_kv":           (get_split_kv_ext,            True),
    "split_kv_v2":        (get_split_kv_v2_ext,         True),
    "split_kv_v2_5":      (get_split_kv_v2_5_ext,       True),
    "split_kv_pipelined": (get_split_kv_pipelined_ext,  True),
    "split_kv_v3_5":      (get_split_kv_v3_5_ext,       True),
    "split_kv_v4":        (get_split_kv_v4_ext,         True),
}


def main(kernel: str, batch: int, ctx: int, split_kv: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    if kernel not in KERNELS:
        raise ValueError(f"Unknown --kernel {kernel!r}; choices: {list(KERNELS)}")
    loader, takes_split_kv = KERNELS[kernel]

    # Build (and JIT-compile) the extension *before* ncu's kernel-name
    # filter is live for any launches — compilation itself launches no
    # CUDA kernels, so this doesn't consume any of the skip/count budget.
    ext = loader(verbose=False)

    q, kv_data, kv_indptr, kv_indices, kv_last_page_len, _ = synthesize(
        batch=batch, context_length=ctx,
        num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM, page_size=PAGE_SIZE,
        dtype=DTYPE, device=DEVICE,
    )

    if takes_split_kv:
        ext.decode_attention_split_kv(
            q, kv_data, kv_indptr, kv_indices, kv_last_page_len, split_kv
        )
    else:
        ext.decode_attention_naive(
            q, kv_data, kv_indptr, kv_indices, kv_last_page_len
        )
    torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=list(KERNELS), default="split_kv",
                        help="Which kernel to launch (default: split_kv)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--split-kv", type=int, default=8,
                        help="Ignored for --kernel naive")
    args = parser.parse_args()
    main(args.kernel, args.batch, args.ctx, args.split_kv)
