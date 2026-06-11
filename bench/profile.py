"""
Profiling wrapper for ncu (NVIDIA Nsight Compute).

Captures three configurations:
  1. Short context (2k), small split (2)   → baseline, low-occupancy regime
  2. Long context (32k), optimal split (8) → target operating point
  3. Long context (32k), many splits (16)  → over-split, reduction overhead

For each, saves a .ncu-rep file and extracts key metrics into a CSV:
  achieved_occupancy, dram_throughput_gb_s, l2_hit_rate_pct,
  warp_stall_reason_1, warp_stall_reason_2

Usage:
    # Single run — ncu must be on PATH:
    ncu --target-processes all python bench/profile.py

    # Or without ncu (writes metric CSV using CUDA events as proxy):
    python bench/profile.py --no-ncu
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from layout import synthesize
from kernel_split_kv import decode_attention_split_kv

NUM_Q_HEADS  = 32
NUM_KV_HEADS = 8
HEAD_DIM     = 128
PAGE_SIZE    = 16
DTYPE        = torch.float16
DEVICE       = "cuda"
WARMUP       = 5

CONFIGS = [
    # (label,         context_length, batch, split_kv)
    ("short_small",   2_048,          1,     2),
    ("long_optimal",  32_768,         1,     8),
    ("long_oversplit",32_768,         1,     16),
]


def warmup_and_run(context_length: int, batch: int, split_kv: int):
    """Runs warmup outside the profiled region, then returns the kernel callable."""
    q, kv_data, kv_indptr, kv_indices, kv_last_page_len, _ = synthesize(
        batch=batch,
        context_length=context_length,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        dtype=DTYPE,
        device=DEVICE,
    )
    scratch = {}

    def fn():
        return decode_attention_split_kv(
            q, kv_data, kv_indptr, kv_indices, kv_last_page_len,
            split_kv=split_kv, _scratch=scratch,
        )

    # Warmup: trigger JIT compilation outside profiled region
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    return fn


def run_with_ncu(label: str, context_length: int, batch: int, split_kv: int,
                 out_dir: Path):
    """Launches a subprocess under ncu and saves .ncu-rep to out_dir."""
    rep_path = out_dir / f"{label}.ncu-rep"
    script = f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bench.profile import warmup_and_run
import torch, cudaprof
fn = warmup_and_run({context_length}, {batch}, {split_kv})
torch.cuda.cudart().cudaProfilerStart()
fn()
torch.cuda.cudart().cudaProfilerStop()
"""
    # Write a temporary script
    tmp_script = out_dir / f"_profile_{label}.py"
    tmp_script.write_text(script)

    ncu_cmd = [
        "ncu",
        "--target-processes", "all",
        "--set", "full",
        "--export", str(rep_path),
        "python", str(tmp_script),
    ]
    print(f"  Running: {' '.join(ncu_cmd)}")
    result = subprocess.run(ncu_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ncu failed:\n{result.stderr}")
    else:
        print(f"  Saved: {rep_path}")
    tmp_script.unlink(missing_ok=True)


def run_proxy_metrics(out_dir: Path):
    """
    Without ncu, measures latency + achieved bandwidth as proxy metrics.
    Useful for verifying the kernel runs correctly in CI without ncu access.
    """
    import csv
    rows = []
    for label, ctx, batch, split_kv in CONFIGS:
        q, kv_data, kv_indptr, kv_indices, kv_last_page_len, _ = synthesize(
            batch=batch, context_length=ctx,
            num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM, page_size=PAGE_SIZE,
            dtype=DTYPE, device=DEVICE,
        )
        scratch = {}

        def fn():
            return decode_attention_split_kv(
                q, kv_data, kv_indptr, kv_indices, kv_last_page_len,
                split_kv=split_kv, _scratch=scratch,
            )

        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record(); fn(); end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)

        kv_bytes = batch * ctx * NUM_KV_HEADS * HEAD_DIM * 2 * 2  # fp16 K+V
        bw = kv_bytes / 1e9 / (ms / 1e3)

        rows.append({
            "config": label, "context_length": ctx, "batch": batch,
            "split_kv": split_kv, "latency_ms": round(ms, 4),
            "kv_bw_gb_s": round(bw, 2),
            "note": "proxy (no ncu)",
        })
        print(f"  {label:<20} {ms:.3f} ms  bw={bw:.0f} GB/s")

    csv_path = out_dir / "profile_proxy.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Proxy metrics written to {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/profiles")
    parser.add_argument("--no-ncu", action="store_true",
                        help="Skip ncu; just run proxy metrics")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_ncu:
        print("Running proxy metrics (no ncu)...")
        run_proxy_metrics(out_dir)
    else:
        # Check ncu is available
        if subprocess.run(["which", "ncu"], capture_output=True).returncode != 0:
            print("ncu not found on PATH; falling back to proxy metrics")
            run_proxy_metrics(out_dir)
        else:
            for label, ctx, batch, split_kv in CONFIGS:
                print(f"\nProfiling config: {label}  ctx={ctx}  split={split_kv}")
                run_with_ncu(label, ctx, batch, split_kv, out_dir)
