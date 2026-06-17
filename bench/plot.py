"""
Generates all 6 plots from the microbench CSV.

Usage:
    python bench/plot.py --csv results/microbench.csv --out results/plots/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

IMPL_COLORS = {
    "flashinfer":      "#2196F3",
    "naive_triton":    "#F44336",
    "split_kv_triton": "#4CAF50",
    "cuda_naive":      "#F44336",
    "cuda_split_kv":   "#4CAF50",
}
IMPL_LABELS = {
    "flashinfer":      "FlashInfer",
    "naive_triton":    "Naive Triton",
    "split_kv_triton": "Split KV Triton",
    "cuda_naive":      "Naive CUDA",
    "cuda_split_kv":   "Split KV CUDA",
}
_FALLBACK_COLORS = ["#FF9800", "#9C27B0", "#00BCD4", "#795548"]

def _impl_color(impl: str) -> str:
    if impl in IMPL_COLORS:
        return IMPL_COLORS[impl]
    idx = abs(hash(impl)) % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[idx]

def _impl_label(impl: str) -> str:
    return IMPL_LABELS.get(impl, impl)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


# ── Plot 1: latency vs SPLIT_KV ───────────────────────────────────────────

def plot1_latency_vs_split(df: pd.DataFrame, out_dir: Path):
    ctx_lengths = sorted(df["context_length"].unique())
    n = len(ctx_lengths)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, ctx in zip(axes, ctx_lengths):
        sub = df[df["context_length"] == ctx]

        for impl in sorted(sub["implementation"].unique()):
            rows = sub[sub["implementation"] == impl]
            if rows.empty:
                continue
            c   = _impl_color(impl)
            lbl = _impl_label(impl)
            if impl == "flashinfer":
                lat = rows["latency_ms_p50"].mean()
                ax.axhline(lat, color=c, linestyle="--", label=lbl)
            elif "naive" in impl:
                lat = rows["latency_ms_p50"].values[0]
                ax.axhline(lat, color=c, linestyle=":", label=lbl)
            else:
                rows_s = rows.sort_values("split_kv")
                ax.plot(rows_s["split_kv"], rows_s["latency_ms_p50"],
                        marker="o", color=c, label=lbl)

        ax.set_title(f"ctx={ctx//1024}k")
        ax.set_xlabel("SPLIT_KV")
        ax.set_ylabel("Latency (ms)")
        ax.set_xticks(sorted(df["split_kv"].unique()))
        ax.legend(fontsize=9)

    fig.suptitle("Plot 1: Latency vs SPLIT_KV", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "plot1_latency_vs_split.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 2: latency vs context length ─────────────────────────────────────

def plot2_latency_vs_ctx(df: pd.DataFrame, out_dir: Path):
    # For each (impl, ctx), take the minimum latency over SPLIT_KV (optimal)
    best = (df.groupby(["implementation", "context_length", "batch_size"])
              ["latency_ms_p50"].min().reset_index())

    batch_sizes = sorted(best["batch_size"].unique())
    fig, axes = plt.subplots(1, len(batch_sizes),
                              figsize=(5 * len(batch_sizes), 4), sharey=False)
    if len(batch_sizes) == 1:
        axes = [axes]

    for ax, batch in zip(axes, batch_sizes):
        sub = best[best["batch_size"] == batch]
        for impl in sorted(sub["implementation"].unique()):
            rows = sub[sub["implementation"] == impl].sort_values("context_length")
            if rows.empty:
                continue
            ax.plot(rows["context_length"] / 1024, rows["latency_ms_p50"],
                    marker="o", color=_impl_color(impl), label=_impl_label(impl))
        ax.set_title(f"batch={batch}")
        ax.set_xlabel("Context length (k tokens)")
        ax.set_ylabel("Latency (ms)")
        ax.legend(fontsize=9)

    fig.suptitle("Plot 2: Latency vs Context Length (at optimal SPLIT_KV)",
                 fontweight="bold")
    fig.tight_layout()
    path = out_dir / "plot2_latency_vs_ctx.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 3: achieved BW % of peak ─────────────────────────────────────────

def plot3_bw_pct(df: pd.DataFrame, out_dir: Path):
    best = (df.groupby(["implementation", "context_length"])
              ["bw_pct_of_peak"].max().reset_index())

    fig, ax = plt.subplots(figsize=(8, 4))
    for impl in sorted(best["implementation"].unique()):
        rows = best[best["implementation"] == impl].sort_values("context_length")
        if rows.empty:
            continue
        ax.plot(rows["context_length"] / 1024, rows["bw_pct_of_peak"],
                marker="o", color=_impl_color(impl), label=_impl_label(impl))

    ax.axhline(100, color="black", linestyle="--", alpha=0.4, label="Peak BW")
    ax.set_xlabel("Context length (k tokens)")
    ax.set_ylabel("% of peak HBM bandwidth")
    ax.set_ylim(0, 110)
    ax.legend()
    fig.suptitle("Plot 3: Achieved Bandwidth as % of Peak", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "plot3_bw_pct.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 4: roofline ───────────────────────────────────────────────────────

def plot4_roofline(df: pd.DataFrame, out_dir: Path,
                   peak_compute_tflops: float = 312.0,
                   peak_bw_gb_s: float = 2000.0,
                   num_q_heads: int = 32, num_kv_heads: int = 8,
                   head_dim: int = 128):
    fig, ax = plt.subplots(figsize=(7, 5))

    # Roofline ceilings
    ai_range = np.logspace(-2, 3, 500)
    compute_ceil = np.full_like(ai_range, peak_compute_tflops)
    bw_ceil      = ai_range * peak_bw_gb_s / 1e3  # TB/s → TFLOPs at 1 FLOP/byte

    ridge_point = peak_compute_tflops / (peak_bw_gb_s / 1e3)
    ax.loglog(ai_range, np.minimum(compute_ceil, bw_ceil),
              "k--", linewidth=2, label="Roofline")
    ax.axvline(ridge_point, color="gray", linestyle=":", alpha=0.5)

    # Data points
    group_size = num_q_heads // num_kv_heads
    for impl in sorted(df["implementation"].unique()):
        rows = df[df["implementation"] == impl]
        if rows.empty:
            continue
        ai     = group_size / 2.0
        tflops = rows["achieved_bw_gb_s"] / 1e3 * ai
        ax.scatter([ai] * len(tflops), tflops,
                   color=_impl_color(impl), label=_impl_label(impl),
                   alpha=0.7, s=40)

    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)")
    ax.set_ylabel("Performance (TFLOPs/s)")
    ax.legend()
    fig.suptitle("Plot 4: Roofline — Decode Attention", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "plot4_roofline.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 5: speedup heatmap ────────────────────────────────────────────────

def plot5_speedup_heatmap(df: pd.DataFrame, out_dir: Path):
    # Auto-detect naive and split implementations so this works with both
    # Triton ("naive_triton", "split_kv_triton") and CUDA ("cuda_naive", "cuda_split_kv") CSVs.
    impls = df["implementation"].unique()
    naive_impls = [i for i in impls if "naive" in i]
    split_impls = [i for i in impls if "split" in i]
    if not naive_impls or not split_impls:
        print(f"plot5: need a naive and a split-KV impl; found {list(impls)} — skipping")
        return
    naive_impl = naive_impls[0]
    split_impl = split_impls[0]

    naive = df[df["implementation"] == naive_impl][
        ["context_length", "batch_size", "latency_ms_p50"]
    ].rename(columns={"latency_ms_p50": "lat_naive"})

    best_split = (df[df["implementation"] == split_impl]
                  .groupby(["context_length", "batch_size"])["latency_ms_p50"]
                  .min().reset_index()
                  .rename(columns={"latency_ms_p50": "lat_split"}))

    merged = naive.merge(best_split, on=["context_length", "batch_size"])
    if merged.empty:
        print("plot5: no overlapping (context_length, batch_size) rows — skipping")
        return
    merged["speedup"] = merged["lat_naive"] / merged["lat_split"]

    ctx_vals   = sorted(merged["context_length"].unique())
    batch_vals = sorted(merged["batch_size"].unique())
    matrix = np.zeros((len(ctx_vals), len(batch_vals)))
    for i, ctx in enumerate(ctx_vals):
        for j, batch in enumerate(batch_vals):
            row = merged[(merged["context_length"] == ctx) &
                         (merged["batch_size"] == batch)]
            if not row.empty:
                matrix[i, j] = row["speedup"].values[0]

    vmax = matrix.max() if matrix.size > 0 and matrix.max() > 1.0 else 2.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                   vmin=1.0, vmax=vmax + 0.1)
    ax.set_xticks(range(len(batch_vals)))
    ax.set_xticklabels([f"B={b}" for b in batch_vals])
    ax.set_yticks(range(len(ctx_vals)))
    ax.set_yticklabels([f"{c//1024}k" for c in ctx_vals])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Context length")
    plt.colorbar(im, ax=ax, label="Speedup vs Naive")
    for i in range(len(ctx_vals)):
        for j in range(len(batch_vals)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black")
    fig.suptitle("Plot 5: Speedup Heatmap — Split KV / Naive", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "plot5_speedup_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 6: gap to FlashInfer ──────────────────────────────────────────────

def plot6_gap_to_flashinfer(df: pd.DataFrame, out_dir: Path):
    fi = df[df["implementation"] == "flashinfer"].groupby("context_length")[
        "latency_ms_p50"
    ].mean().reset_index().rename(columns={"latency_ms_p50": "lat_fi"})

    split_impls = [i for i in df["implementation"].unique() if "split" in i]
    if fi.empty or not split_impls:
        print("  Plot 6: skipped (no FlashInfer or split-KV rows in CSV)")
        return
    split_impl = split_impls[0]

    best_split = (df[df["implementation"] == split_impl]
                  .groupby("context_length")["latency_ms_p50"]
                  .min().reset_index()
                  .rename(columns={"latency_ms_p50": "lat_split"}))

    merged = fi.merge(best_split, on="context_length")
    if merged.empty:
        print("  Plot 6: skipped (no overlapping context_length rows)")
        return
    merged["ratio"] = merged["lat_split"] / merged["lat_fi"]
    merged = merged.sort_values("context_length")

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(merged))
    ax.bar(x, merged["ratio"], color=_impl_color(split_impl),
                  edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="black", linestyle="--", label="Parity with FlashInfer")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{c//1024}k" for c in merged["context_length"]])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Latency ratio (Split KV / FlashInfer)")
    ax.legend()
    fig.suptitle("Plot 6: Gap to FlashInfer at Optimal SPLIT_KV",
                 fontweight="bold")
    fig.tight_layout()
    path = out_dir / "plot6_gap_to_flashinfer.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/microbench.csv")
    parser.add_argument("--out", default="results/plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    plot1_latency_vs_split(df, out_dir)
    plot2_latency_vs_ctx(df, out_dir)
    plot3_bw_pct(df, out_dir)
    plot4_roofline(df, out_dir)
    plot5_speedup_heatmap(df, out_dir)
    plot6_gap_to_flashinfer(df, out_dir)

    print(f"\nAll plots saved to {out_dir}/")
