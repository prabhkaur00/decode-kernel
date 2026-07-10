"""
Minimal single-call driver for Nsight Compute (ncu) profiling.

Unlike profile_cuda_split_kv_nvtx.py, this makes exactly one call to the
kernel so `--launch-skip`/`--launch-count` map directly onto the two
kernels launched by that call (decode_attn_partition_kernel, then
decode_attn_reduce_kernel) with no warmup calls ahead of them to account
for in the skip count.

Shape is a CLI arg so you can point this at whichever (batch, ctx, split_kv)
regime results/microbench_cuda_*.csv flags as interesting, without editing
the file — e.g. the worst bw_pct_of_peak row vs. the best one.

Usage (from repo root, on a GPU runtime):
    # worst case from the CSV: batch=1, split_kv=1, ~0.4% of peak bw
    ncu --set full --kernel-name regex:decode \
        --launch-skip 0 --launch-count 2 \
        -o decode_ncu_worst -f \
        python bench/mini_driver.py --batch 1 --ctx 8192 --split-kv 1

    # best case from the CSV: batch=16, split_kv=16, ~13.75% of peak bw
    ncu --set full --kernel-name regex:decode \
        --launch-skip 0 --launch-count 2 \
        -o decode_ncu_best -f \
        python bench/mini_driver.py --batch 16 --ctx 65536 --split-kv 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from layout import synthesize
from cuda_.build import get_split_kv_ext

NUM_Q_HEADS  = 32
NUM_KV_HEADS = 8
HEAD_DIM     = 128
PAGE_SIZE    = 16
DTYPE        = torch.float16
DEVICE       = "cuda"


def main(batch: int, ctx: int, split_kv: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    # Build (and JIT-compile) the extension *before* ncu's kernel-name
    # filter is live for any launches — compilation itself launches no
    # CUDA kernels, so this doesn't consume any of the skip/count budget.
    ext = get_split_kv_ext(verbose=False)

    q, kv_data, kv_indptr, kv_indices, kv_last_page_len, _ = synthesize(
        batch=batch, context_length=ctx,
        num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM, page_size=PAGE_SIZE,
        dtype=DTYPE, device=DEVICE,
    )

    ext.decode_attention_split_kv(
        q, kv_data, kv_indptr, kv_indices, kv_last_page_len, split_kv
    )
    torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--split-kv", type=int, default=8)
    args = parser.parse_args()
    main(args.batch, args.ctx, args.split_kv)
