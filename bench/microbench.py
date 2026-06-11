"""
Microbenchmark harness for decode attention implementations.

Sweeps (implementation, context_length, batch_size, split_kv) and writes a CSV
with columns:
    implementation, context_length, batch_size, split_kv,
    latency_ms_p50, latency_ms_p95,
    achieved_bw_gb_s, peak_bw_gb_s, bw_pct_of_peak,
    max_abs_err_vs_fp32

Usage:
    python bench/microbench.py                      # defaults
    python bench/microbench.py --out results/bench.csv
    python bench/microbench.py --quick              # small sweep for CI
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

# Allow running from repo root or bench/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from layout import synthesize
from reference import reference_attention_paged
from kernel_naive import decode_attention_naive
from kernel_split_kv import decode_attention_split_kv

try:
    import flashinfer
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    print("WARNING: flashinfer not available; flashinfer rows will be skipped")


# ── Constants ─────────────────────────────────────────────────────────────

NUM_Q_HEADS = 32
NUM_KV_HEADS = 8          # GQA group size = 4  (Llama 3 / Mistral style)
HEAD_DIM = 128
PAGE_SIZE = 16
DTYPE = torch.float16
DEVICE = "cuda"

WARMUP_ITERS = 20
TIMED_ITERS  = 100

# L2 cache flush buffer (256 MB > A100 L2 of 40 MB)
_L2_FLUSH_BYTES = 256 * 1024 * 1024
_l2_flush_buf: torch.Tensor | None = None


def _get_flush_buf() -> torch.Tensor:
    global _l2_flush_buf
    if _l2_flush_buf is None:
        _l2_flush_buf = torch.empty(
            _L2_FLUSH_BYTES // 4, dtype=torch.float32, device=DEVICE
        )
    return _l2_flush_buf


def flush_l2() -> None:
    """Write through a 256 MB buffer to evict the L2 cache."""
    _get_flush_buf().fill_(0.0)
    torch.cuda.synchronize()


# ── Peak bandwidth ─────────────────────────────────────────────────────────

def get_peak_bw_gb_s() -> float:
    """Returns theoretical peak HBM bandwidth in GB/s for the current GPU."""
    props = torch.cuda.get_device_properties(0)
    name = props.name.lower()
    # Known values; fall back to a BW estimate from memory clock + bus width
    known = {
        "a100": 2000.0,
        "h100": 3350.0,
        "v100": 900.0,
        "a10":  600.0,
        "t4":   300.0,
    }
    for key, bw in known.items():
        if key in name:
            return bw
    # Generic estimate: memory_clock_rate (kHz) * bus_width (bits) * 2 (DDR)
    bw_bytes_s = (
        props.memory_clock_rate * 1e3  # Hz
        * props.memory_bus_width / 8    # bytes per transfer
        * 2                             # DDR
    )
    return bw_bytes_s / 1e9


# ── Bandwidth accounting ───────────────────────────────────────────────────

def attention_bytes(
    batch: int,
    context_length: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    elem_bytes: int = 2,
) -> int:
    """Minimum bytes touched for one decode attention call (memory-bound model)."""
    q_bytes  = batch * num_q_heads * head_dim * elem_bytes   # Q read
    kv_bytes = batch * context_length * num_kv_heads * head_dim * elem_bytes * 2  # K+V read
    o_bytes  = batch * num_q_heads * head_dim * elem_bytes   # O write
    return q_bytes + kv_bytes + o_bytes


# ── FlashInfer wrapper ─────────────────────────────────────────────────────

_fi_workspace: torch.Tensor | None = None


def _get_fi_workspace() -> torch.Tensor:
    global _fi_workspace
    if _fi_workspace is None:
        _fi_workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE
        )
    return _fi_workspace


def flashinfer_decode(q, kv_data, kv_indptr, kv_indices, kv_last_page_len):
    """Runs only the FlashInfer .run() phase (plan phase excluded)."""
    _, _, page_size, num_kv_heads, head_dim = kv_data.shape
    _, num_q_heads, _ = q.shape
    workspace = _get_fi_workspace()

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    # begin_forward = plan phase; not timed
    wrapper.begin_forward(
        kv_indptr, kv_indices, kv_last_page_len,
        num_q_heads, num_kv_heads, head_dim,
        page_size, data_type=q.dtype,
    )
    return wrapper.forward(q, kv_data)


# ── Timing helper ──────────────────────────────────────────────────────────

def time_kernel(fn, n_warmup: int = WARMUP_ITERS, n_timed: int = TIMED_ITERS):
    """
    Returns sorted list of latencies in ms.
    Uses CUDA events for accurate GPU timing.
    L2 cache is flushed before each timed iteration.
    """
    # Warmup (no flush)
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    latencies = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

    for _ in range(n_timed):
        flush_l2()
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        latencies.append(start_ev.elapsed_time(end_ev))

    return sorted(latencies)


# ── Correctness check ──────────────────────────────────────────────────────

def check_correctness(output: torch.Tensor, ref: torch.Tensor, label: str):
    """Returns max_abs_err; prints a warning if tolerances are exceeded."""
    err = (output.float() - ref.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    if max_err > 1e-2:
        print(f"  [WARN] {label}: max_abs_err={max_err:.4e} > 1e-2")
    if mean_err > 1e-3:
        print(f"  [WARN] {label}: mean_abs_err={mean_err:.4e} > 1e-3")
    return max_err


# ── Main sweep ─────────────────────────────────────────────────────────────

def run_sweep(
    context_lengths: List[int],
    batch_sizes: List[int],
    split_kvs: List[int],
    implementations: List[str],
    out_path: str,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for benchmarking")

    peak_bw = get_peak_bw_gb_s()
    print(f"Peak HBM bandwidth: {peak_bw:.0f} GB/s")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    rows = []
    scratch = {}  # shared scratch buffers across calls

    for ctx in context_lengths:
        for batch in batch_sizes:
            print(f"ctx={ctx:>6}  batch={batch:>3}", flush=True)

            # Synthesize tensors once per (ctx, batch)
            q, kv_data, kv_indptr, kv_indices, kv_last_page_len, _ = synthesize(
                batch=batch,
                context_length=ctx,
                num_q_heads=NUM_Q_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                head_dim=HEAD_DIM,
                page_size=PAGE_SIZE,
                dtype=DTYPE,
                device=DEVICE,
            )

            # fp32 reference (computed once, on CPU for correctness baseline)
            ref = reference_attention_paged(
                q, kv_data, kv_indptr, kv_indices, kv_last_page_len
            ).to(DEVICE)

            total_bytes = attention_bytes(
                batch, ctx, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, elem_bytes=2
            )

            for impl in implementations:
                for skv in split_kvs:
                    # Skip split_kv > 1 for implementations that don't use it
                    if impl != "split_kv_triton" and skv != 1:
                        continue
                    if impl == "flashinfer" and not FLASHINFER_AVAILABLE:
                        continue

                    # Build the callable
                    if impl == "flashinfer":
                        fi_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                            _get_fi_workspace(), "NHD"
                        )
                        fi_wrapper.begin_forward(
                            kv_indptr, kv_indices, kv_last_page_len,
                            NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
                            PAGE_SIZE, data_type=q.dtype,
                        )
                        fn = lambda: fi_wrapper.forward(q, kv_data)
                    elif impl == "naive_triton":
                        fn = lambda: decode_attention_naive(
                            q, kv_data, kv_indptr, kv_indices, kv_last_page_len
                        )
                    elif impl == "split_kv_triton":
                        fn = lambda skv_=skv: decode_attention_split_kv(
                            q, kv_data, kv_indptr, kv_indices, kv_last_page_len,
                            split_kv=skv_, _scratch=scratch,
                        )
                    else:
                        raise ValueError(f"Unknown implementation: {impl}")

                    # Correctness check (before timing)
                    with torch.no_grad():
                        out = fn()
                    max_err = check_correctness(out, ref, f"{impl}/skv={skv}")

                    # Timing
                    latencies = time_kernel(fn)
                    p50 = np.percentile(latencies, 50)
                    p95 = np.percentile(latencies, 95)
                    bw  = (total_bytes / 1e9) / (p50 / 1e3)  # GB/s

                    label = f"  {impl:<20} split={skv:>2}  "
                    label += f"p50={p50:.3f}ms  bw={bw:.0f}GB/s  max_err={max_err:.1e}"
                    print(label)

                    rows.append({
                        "implementation": impl,
                        "context_length": ctx,
                        "batch_size": batch,
                        "split_kv": skv,
                        "latency_ms_p50": round(p50, 4),
                        "latency_ms_p95": round(p95, 4),
                        "achieved_bw_gb_s": round(bw, 2),
                        "peak_bw_gb_s": round(peak_bw, 2),
                        "bw_pct_of_peak": round(100.0 * bw / peak_bw, 2),
                        "max_abs_err_vs_fp32": round(max_err, 6),
                    })
            print()

    df = pd.DataFrame(rows)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Results written to {out_file}")
    return df


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Decode attention microbenchmark")
    parser.add_argument("--out", default="results/microbench.csv")
    parser.add_argument("--quick", action="store_true",
                        help="Small sweep (2 ctx, 1 batch, 3 splits) for quick validation")
    return parser.parse_args()


FULL_CONTEXT_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536]
FULL_BATCH_SIZES     = [1, 4, 16]
FULL_SPLIT_KVS       = [1, 2, 4, 8, 16]
FULL_IMPLS           = ["flashinfer", "naive_triton", "split_kv_triton"]

QUICK_CONTEXT_LENGTHS = [2048, 8192]
QUICK_BATCH_SIZES     = [1]
QUICK_SPLIT_KVS       = [1, 4, 8]
QUICK_IMPLS           = ["naive_triton", "split_kv_triton"]


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        ctxs, batches, splits, impls = (
            QUICK_CONTEXT_LENGTHS, QUICK_BATCH_SIZES,
            QUICK_SPLIT_KVS, QUICK_IMPLS,
        )
    else:
        ctxs, batches, splits, impls = (
            FULL_CONTEXT_LENGTHS, FULL_BATCH_SIZES,
            FULL_SPLIT_KVS, FULL_IMPLS,
        )
    run_sweep(ctxs, batches, splits, impls, args.out)
