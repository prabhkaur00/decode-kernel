"""
CUDA-backend microbenchmark for decode attention.

Exercises the CUDA C++ kernels in src/cuda_/.
The CUDA extension is JIT-compiled on first call (~30-60 s); subsequent
runs use the on-disk cache.

Usage:
    python bench/microbench_cuda.py                          # v1 (default)
    python bench/microbench_cuda.py --version v2             # v2 KV-head-centric
    python bench/microbench_cuda.py --version v2 --quick
    python bench/microbench_cuda.py --out results/my_run.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import importlib.util

import numpy as np
import pandas as pd
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from layout import synthesize
from reference import reference_attention_paged

# Load src/cuda_/build.py by file path to avoid conflict with the top-level
# 'cuda' namespace package installed by cuda-python in newer environments.
_spec = importlib.util.spec_from_file_location("_cuda_build", _SRC / "cuda_" / "build.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Registry: version string -> builder function for the split_kv extension.
# The naive extension is shared across all versions.
_SPLIT_KV_BUILDERS = {
    "v1": _mod.get_split_kv_ext,
    "v2": _mod.get_split_kv_v2_ext,
}
get_naive_ext = _mod.get_naive_ext

try:
    import flashinfer
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    print("WARNING: flashinfer not available; flashinfer rows will be skipped")


# ── Constants ─────────────────────────────────────────────────────────────

NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 16
DTYPE = torch.float16
DEVICE = "cuda"

WARMUP_ITERS = 20
TIMED_ITERS  = 100

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
    _get_flush_buf().fill_(0.0)
    torch.cuda.synchronize()


# ── Peak bandwidth ─────────────────────────────────────────────────────────

def get_peak_bw_gb_s() -> float:
    props = torch.cuda.get_device_properties(0)
    name = props.name.lower()
    known = {
        "a100": 2000.0, "h100": 3350.0, "v100": 900.0, "a10": 600.0, "t4": 300.0,
    }
    for key, bw in known.items():
        if key in name:
            return bw
    bw_bytes_s = (
        props.memory_clock_rate * 1e3
        * props.memory_bus_width / 8
        * 2
    )
    return bw_bytes_s / 1e9


# ── Bandwidth accounting ───────────────────────────────────────────────────

def attention_bytes(
    batch: int, context_length: int,
    num_q_heads: int, num_kv_heads: int, head_dim: int, elem_bytes: int = 2,
) -> int:
    q_bytes  = batch * num_q_heads  * head_dim * elem_bytes
    kv_bytes = batch * context_length * num_kv_heads * head_dim * elem_bytes * 2
    o_bytes  = batch * num_q_heads  * head_dim * elem_bytes
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


# ── Timing helper ──────────────────────────────────────────────────────────

def time_kernel(fn, n_warmup: int = WARMUP_ITERS, n_timed: int = TIMED_ITERS):
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

def check_correctness(output: torch.Tensor, ref, label: str) -> float:
    if ref is None:
        return float("nan")
    err = (output.float() - ref.float()).abs()
    max_err = err.max().item()
    if max_err > 1e-2:
        print(f"  [WARN] {label}: max_abs_err={max_err:.4e} > 1e-2")
    return max_err


# ── Main sweep ─────────────────────────────────────────────────────────────

def run_sweep(
    context_lengths: List[int],
    batch_sizes: List[int],
    split_kvs: List[int],
    implementations: List[str],
    out_path: str,
    version: str,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for benchmarking")

    naive_ext = get_naive_ext()
    split_kv_ext = _SPLIT_KV_BUILDERS[version]()

    peak_bw = get_peak_bw_gb_s()
    print(f"Peak HBM bandwidth: {peak_bw:.0f} GB/s")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kernel version: {version}\n")

    rows = []

    for ctx in context_lengths:
        for batch in batch_sizes:
            print(f"ctx={ctx:>6}  batch={batch:>3}", flush=True)

            q, kv_data, kv_indptr, kv_indices, kv_last_page_len, _ = synthesize(
                batch=batch, context_length=ctx,
                num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
                head_dim=HEAD_DIM, page_size=PAGE_SIZE, dtype=DTYPE, device=DEVICE,
            )

            try:
                ref = reference_attention_paged(
                    q.cpu(), kv_data.cpu(),
                    kv_indptr.cpu(), kv_indices.cpu(), kv_last_page_len.cpu(),
                ).to(DEVICE)
            except (torch.OutOfMemoryError, RuntimeError):
                print(
                    f"  [WARN] reference OOM for ctx={ctx} batch={batch};"
                    " skipping correctness check"
                )
                ref = None

            total_bytes = attention_bytes(
                batch, ctx, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, elem_bytes=2
            )

            for impl in implementations:
                for skv in split_kvs:
                    if impl != "cuda_split_kv" and skv != 1:
                        continue
                    if impl == "flashinfer" and not FLASHINFER_AVAILABLE:
                        continue

                    if impl == "flashinfer":
                        fi_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                            _get_fi_workspace(), "NHD"
                        )
                        if hasattr(fi_wrapper, "plan"):
                            fi_wrapper.plan(
                                kv_indptr, kv_indices, kv_last_page_len,
                                NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
                                PAGE_SIZE, data_type=q.dtype,
                            )
                            fn = lambda: fi_wrapper.run(q, kv_data)
                        else:
                            fi_wrapper.begin_forward(
                                kv_indptr, kv_indices, kv_last_page_len,
                                NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
                                PAGE_SIZE, data_type=q.dtype,
                            )
                            fn = lambda: fi_wrapper.forward(q, kv_data)
                    elif impl == "cuda_naive":
                        fn = lambda: naive_ext.decode_attention_naive(
                            q, kv_data, kv_indptr, kv_indices, kv_last_page_len
                        )
                    elif impl == "cuda_split_kv":
                        fn = lambda skv_=skv: split_kv_ext.decode_attention_split_kv(
                            q, kv_data, kv_indptr, kv_indices, kv_last_page_len, skv_
                        )
                    else:
                        raise ValueError(f"Unknown implementation: {impl}")

                    with torch.no_grad():
                        out = fn()
                    max_err = check_correctness(out, ref, f"{impl}/skv={skv}")

                    latencies = time_kernel(fn)
                    p50 = np.percentile(latencies, 50)
                    p95 = np.percentile(latencies, 95)
                    bw  = (total_bytes / 1e9) / (p50 / 1e3)

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
                        "max_abs_err_vs_fp32": (
                            round(max_err, 6) if not np.isnan(max_err) else float("nan")
                        ),
                    })
            print()

    df = pd.DataFrame(rows)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Results written to {out_file}")
    return df


# ── CLI ────────────────────────────────────────────────────────────────────

AVAILABLE_VERSIONS = sorted(_SPLIT_KV_BUILDERS.keys())

def parse_args():
    parser = argparse.ArgumentParser(description="CUDA decode attention microbenchmark")
    parser.add_argument("--version", default="v1", choices=AVAILABLE_VERSIONS,
                        help=f"Split-KV kernel version ({', '.join(AVAILABLE_VERSIONS)})")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: results/microbench_cuda_<version>.csv)")
    parser.add_argument("--quick", action="store_true",
                        help="Small sweep for quick validation")
    parser.add_argument("--split-kv-only", action="store_true",
                        help="Only benchmark the split-KV kernel (skip naive and flashinfer)")
    return parser.parse_args()


FULL_CONTEXT_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536]
FULL_BATCH_SIZES     = [1, 4, 16]
FULL_SPLIT_KVS       = [1, 2, 4, 8, 16]
FULL_IMPLS           = ["flashinfer", "cuda_naive", "cuda_split_kv"]

QUICK_CONTEXT_LENGTHS = [2048, 8192]
QUICK_BATCH_SIZES     = [1]
QUICK_SPLIT_KVS       = [1, 4, 8]
QUICK_IMPLS           = ["cuda_naive", "cuda_split_kv"]


if __name__ == "__main__":
    args = parse_args()
    out_path = args.out or f"results/microbench_cuda_{args.version}.csv"
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
    if args.split_kv_only:
        impls = ["cuda_split_kv"]
    run_sweep(ctxs, batches, splits, impls, out_path, version=args.version)
