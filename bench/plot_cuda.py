"""
Plotting script for CUDA decode attention benchmarks.

Generates 6 plots comparing FlashInfer, Split-KV v1, and Split-KV v2
from two CSVs (one per kernel version).

Usage:
    python bench/plot_cuda.py
    python bench/plot_cuda.py --v1 results/microbench_cuda_v1.csv \
                              --v2 results/microbench_cuda_v2.csv \
                              --out results/plots_cuda/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})

COLORS = {
    "flashinfer":       "#2196F3",
    "cuda_split_kv_v1": "#F44336",
    "cuda_split_kv_v2": "#4CAF50",
}
LABELS = {
    "flashinfer":       "FlashInfer",
    "cuda_split_kv_v1": "CUDA Split-KV v1",
    "cuda_split_kv_v2": "CUDA Split-KV v2",
}


def load_and_merge(v1_path: str, v2_path: str) -> pd.DataFrame:
    """Load v1 and v2 CSVs, relabel split_kv rows, drop cuda_naive, merge."""
    df1 = pd.read_csv(v1_path)
    df2 = pd.read_csv(v2_path)

    # Keep flashinfer rows from v1 only (v2 doesn't have them)
    fi = df1[df1["implementation"] == "flashinfer"].copy()

    # Relabel split_kv rows per version
    skv1 = df1[df1["implementation"] == "cuda_split_kv"].copy()
    skv1["implementation"] = "cuda_split_kv_v1"

    skv2 = df2[df2["implementation"] == "cuda_split_kv"].copy()
    skv2["implementation"] = "cuda_split_kv_v2"

    df = pd.concat([fi, skv1, skv2], ignore_index=True)
    print(f"Merged: {len(df)} rows")
    print(f"Implementations: {sorted(df['implementation'].unique())}")
    return df


# ── Plot 1: latency vs SPLIT_KV ───────────────────────────────────────────

def plot1(df: pd.DataFrame, out_dir: Path):
    ctx_lengths = sorted(df["context_length"].unique())
    batch_sizes = sorted(df["batch_size"].unique())

    fig, axes = plt.subplots(
        len(batch_sizes), len(ctx_lengths),
        figsize=(4.5 * len(ctx_lengths), 3.5 * len(batch_sizes)),
        sharey=False, squeeze=False,
    )

    for row, batch in enumerate(batch_sizes):
        for col, ctx in enumerate(ctx_lengths):
            ax = axes[row][col]
            sub = df[(df["context_length"] == ctx) & (df["batch_size"] == batch)]

            for impl, color in COLORS.items():
                rows = sub[sub["implementation"] == impl]
                if rows.empty:
                    continue
                lbl = LABELS[impl]
                if impl == "flashinfer":
                    ax.axhline(rows["latency_ms_p50"].values[0],
                               color=color, linestyle="--", linewidth=1.5, label=lbl)
                else:
                    r = rows.sort_values("split_kv")
                    ax.plot(r["split_kv"], r["latency_ms_p50"],
                            marker="o", color=color, label=lbl)

            ax.set_title(f"ctx={ctx//1024}k  B={batch}", fontsize=9)
            ax.set_xlabel("SPLIT_KV", fontsize=8)
            ax.set_ylabel("Latency (ms)", fontsize=8)
            split_ticks = sorted(df[df["implementation"] != "flashinfer"]["split_kv"].unique())
            ax.set_xticks(split_ticks)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6)

    fig.suptitle("Plot 1: Latency vs SPLIT_KV  (A100)", fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "plot1_latency_vs_split.png")


# ── Plot 2: latency vs context length ─────────────────────────────────────

def plot2(df: pd.DataFrame, out_dir: Path):
    best = (df.groupby(["implementation", "context_length", "batch_size"])
              ["latency_ms_p50"].min().reset_index())

    batch_sizes = sorted(best["batch_size"].unique())
    fig, axes = plt.subplots(1, len(batch_sizes),
                             figsize=(5 * len(batch_sizes), 4), sharey=False)
    if len(batch_sizes) == 1:
        axes = [axes]

    for ax, batch in zip(axes, batch_sizes):
        sub = best[best["batch_size"] == batch]
        for impl, color in COLORS.items():
            rows = sub[sub["implementation"] == impl].sort_values("context_length")
            if rows.empty:
                continue
            ax.plot(rows["context_length"] / 1024, rows["latency_ms_p50"],
                    marker="o", color=color, label=LABELS[impl])
        ax.set_title(f"batch={batch}")
        ax.set_xlabel("Context length (k tokens)")
        ax.set_ylabel("Latency (ms)")
        ax.legend(fontsize=9)

    fig.suptitle("Plot 2: Latency vs Context Length  (optimal SPLIT_KV, A100)",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "plot2_latency_vs_ctx.png")


# ── Plot 3: achieved BW % of peak ─────────────────────────────────────────

def plot3(df: pd.DataFrame, out_dir: Path):
    best = (df.groupby(["implementation", "context_length"])
              ["bw_pct_of_peak"].max().reset_index())

    fig, ax = plt.subplots(figsize=(8, 4))
    for impl, color in COLORS.items():
        rows = best[best["implementation"] == impl].sort_values("context_length")
        if rows.empty:
            continue
        ax.plot(rows["context_length"] / 1024, rows["bw_pct_of_peak"],
                marker="o", color=color, label=LABELS[impl])

    ax.axhline(100, color="black", linestyle="--", alpha=0.4, label="Peak BW")
    ax.set_xlabel("Context length (k tokens)")
    ax.set_ylabel("% of peak HBM bandwidth")
    ax.set_ylim(0, 110)
    ax.legend()
    fig.suptitle("Plot 3: Achieved Bandwidth as % of Peak  (A100 = 2000 GB/s)",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "plot3_bw_pct.png")


# ── Plot 4: roofline ───────────────────────────────────────────────────────

def plot4(df: pd.DataFrame, out_dir: Path,
          peak_compute_tflops: float = 312.0,
          peak_bw_gb_s: float = 2000.0,
          num_q_heads: int = 32, num_kv_heads: int = 8):
    fig, ax = plt.subplots(figsize=(7, 5))

    ai_range     = np.logspace(-2, 3, 500)
    compute_ceil = np.full_like(ai_range, peak_compute_tflops)
    bw_ceil      = ai_range * peak_bw_gb_s / 1e3
    ridge        = peak_compute_tflops / (peak_bw_gb_s / 1e3)

    ax.loglog(ai_range, np.minimum(compute_ceil, bw_ceil),
              "k--", linewidth=2, label="Roofline")
    ax.axvline(ridge, color="gray", linestyle=":", alpha=0.5,
               label=f"Ridge point ({ridge:.1f} FLOPs/B)")

    group_size = num_q_heads // num_kv_heads
    ai = group_size / 2.0
    for impl, color in COLORS.items():
        rows = df[df["implementation"] == impl]
        if rows.empty:
            continue
        tflops = rows["achieved_bw_gb_s"] / 1e3 * ai
        ax.scatter([ai] * len(tflops), tflops,
                   color=color, label=LABELS[impl], alpha=0.7, s=40)

    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)")
    ax.set_ylabel("Performance (TFLOPs/s)")
    ax.legend(fontsize=9)
    fig.suptitle(f"Plot 4: Roofline  (A100 fp16: {peak_compute_tflops} TFLOPs, "
                 f"{peak_bw_gb_s:.0f} GB/s)", fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "plot4_roofline.png")


# ── Plot 5: speedup heatmap (v2 best vs v1 best) ──────────────────────────

def plot5(df: pd.DataFrame, out_dir: Path):
    best_v1 = (df[df["implementation"] == "cuda_split_kv_v1"]
               .groupby(["context_length", "batch_size"])["latency_ms_p50"]
               .min().reset_index()
               .rename(columns={"latency_ms_p50": "lat_v1"}))

    best_v2 = (df[df["implementation"] == "cuda_split_kv_v2"]
               .groupby(["context_length", "batch_size"])["latency_ms_p50"]
               .min().reset_index()
               .rename(columns={"latency_ms_p50": "lat_v2"}))

    merged = best_v1.merge(best_v2, on=["context_length", "batch_size"])
    merged["speedup"] = merged["lat_v1"] / merged["lat_v2"]

    ctx_vals   = sorted(merged["context_length"].unique())
    batch_vals = sorted(merged["batch_size"].unique())
    matrix = np.zeros((len(ctx_vals), len(batch_vals)))
    for i, ctx in enumerate(ctx_vals):
        for j, batch in enumerate(batch_vals):
            row = merged[(merged["context_length"] == ctx) &
                         (merged["batch_size"] == batch)]
            if not row.empty:
                matrix[i, j] = row["speedup"].values[0]

    vmin = min(matrix.min(), 0.99)
    vmax = max(matrix.max(), 1.01)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                   vmin=vmin - 0.05, vmax=vmax + 0.05)
    ax.set_xticks(range(len(batch_vals)))
    ax.set_xticklabels([f"B={b}" for b in batch_vals])
    ax.set_yticks(range(len(ctx_vals)))
    ax.set_yticklabels([f"{c//1024}k" for c in ctx_vals])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Context length")
    plt.colorbar(im, ax=ax, label="Speedup (v1 / v2)")
    for i in range(len(ctx_vals)):
        for j in range(len(batch_vals)):
            ax.text(j, i, f"{matrix[i,j]:.2f}x",
                    ha="center", va="center", fontsize=8, color="black")
    fig.suptitle("Plot 5: Speedup — Split-KV v2 (best) vs v1 (best)  (A100)",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "plot5_speedup_heatmap.png")


# ── Plot 6: gap to FlashInfer ──────────────────────────────────────────────

def plot6(df: pd.DataFrame, out_dir: Path):
    fi = (df[df["implementation"] == "flashinfer"]
          .groupby(["context_length", "batch_size"])["latency_ms_p50"]
          .mean().reset_index()
          .rename(columns={"latency_ms_p50": "lat_fi"}))

    batch_sizes = sorted(fi["batch_size"].unique())
    fig, axes = plt.subplots(1, len(batch_sizes),
                             figsize=(5 * len(batch_sizes), 4), sharey=True)
    if len(batch_sizes) == 1:
        axes = [axes]

    bar_impls = [("cuda_split_kv_v1", COLORS["cuda_split_kv_v1"]),
                 ("cuda_split_kv_v2", COLORS["cuda_split_kv_v2"])]

    for ax, batch in zip(axes, batch_sizes):
        sub_fi = fi[fi["batch_size"] == batch].sort_values("context_length")
        ctx_labels = [f"{c//1024}k" for c in sub_fi["context_length"]]
        x = np.arange(len(sub_fi))
        width = 0.35

        for offset, (impl, color) in enumerate(bar_impls):
            best = (df[df["implementation"] == impl]
                    .groupby(["context_length", "batch_size"])["latency_ms_p50"]
                    .min().reset_index()
                    .rename(columns={"latency_ms_p50": "lat_split"}))
            merged = sub_fi.merge(best, on=["context_length", "batch_size"])
            merged["ratio"] = merged["lat_split"] / merged["lat_fi"]
            merged = merged.sort_values("context_length")

            bars = ax.bar(x + (offset - 0.5) * width, merged["ratio"],
                          width, color=color, edgecolor="black", linewidth=0.5,
                          label=LABELS[impl])
            for rect, val in zip(bars, merged["ratio"]):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 0.02,
                        f"{val:.1f}×", ha="center", va="bottom", fontsize=7)

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2,
                   label="Parity with FlashInfer")
        ax.set_xticks(x)
        ax.set_xticklabels(ctx_labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Context length")
        ax.set_ylabel("Latency ratio (Split-KV / FlashInfer)")
        ax.set_title(f"batch={batch}")
        ax.legend(fontsize=7)

    fig.suptitle("Plot 6: Gap to FlashInfer  (A100, optimal SPLIT_KV)",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "plot6_gap_to_flashinfer.png")


# ── Helpers ────────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--v1", default="microbench_cudav1.csv",
                   help="Path to v1 CSV (contains flashinfer + cuda_naive + cuda_split_kv)")
    p.add_argument("--v2", default="results/microbench_cuda_v2.csv",
                   help="Path to v2 CSV (contains cuda_split_kv)")
    p.add_argument("--out", default="results/plots_cuda")
    p.add_argument("--peak-compute", type=float, default=312.0,
                   help="GPU peak FP16 TFLOPs (default: 312 for A100)")
    p.add_argument("--peak-bw", type=float, default=2000.0,
                   help="GPU peak HBM bandwidth GB/s (default: 2000 for A100)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(args.v1, args.v2)
    print(f"Context lengths: {sorted(df['context_length'].unique())}")
    print(f"Batch sizes:     {sorted(df['batch_size'].unique())}\n")

    plot1(df, out_dir)
    plot2(df, out_dir)
    plot3(df, out_dir)
    plot4(df, out_dir, peak_compute_tflops=args.peak_compute, peak_bw_gb_s=args.peak_bw)
    plot5(df, out_dir)
    plot6(df, out_dir)

    print(f"\nAll plots saved to {out_dir}/")
