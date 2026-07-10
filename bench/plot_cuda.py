"""
Plotting script for CUDA decode attention benchmarks.

Generates 6 plots comparing FlashInfer against any number of CUDA kernel
variants (naive, split-KV v1, v2, pipelined, ...), each supplied as its own
microbench CSV.

Usage:
    # old default (v1 vs v2), unchanged
    python bench/plot_cuda.py

    # any set of kernels, named however you like
    python bench/plot_cuda.py \
        --csv v1=results/microbench_cuda_v1.csv \
        --csv v2=results/microbench_cuda_v2.csv \
        --csv pipelined=results/microbench_cuda_pipelined.csv \
        --baseline v1 --compare pipelined \
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

# Colors/labels for implementations that aren't a versioned split-kv kernel —
# these are the same across every CSV, so they're assigned once and never
# come out of the rotating PALETTE below.
FIXED_COLORS = {"flashinfer": "#2196F3", "cuda_naive": "#9E9E9E"}
FIXED_LABELS = {"flashinfer": "FlashInfer", "cuda_naive": "CUDA Naive"}

# Rotating palette for however many cuda_split_kv_<name> variants show up.
PALETTE = ["#F44336", "#4CAF50", "#FF9800", "#9C27B0",
           "#00BCD4", "#795548", "#E91E63", "#3F51B5"]

# Populated by build_style() once the CSVs are loaded; module-level so every
# plot function below can keep referencing bare COLORS/LABELS as before.
COLORS: dict[str, str] = dict(FIXED_COLORS)
LABELS: dict[str, str] = dict(FIXED_LABELS)


def build_style(implementations) -> tuple[dict[str, str], dict[str, str]]:
    """Assign a stable color + label to every implementation name present."""
    colors, labels = dict(FIXED_COLORS), dict(FIXED_LABELS)
    split_kv_impls = sorted(i for i in implementations if i not in FIXED_COLORS)
    for impl, color in zip(split_kv_impls, PALETTE):
        colors[impl] = color
        name = impl.removeprefix("cuda_split_kv_")
        labels[impl] = f"CUDA Split-KV {name}"
    if len(split_kv_impls) > len(PALETTE):
        print(f"WARNING: {len(split_kv_impls)} split-kv variants but only "
              f"{len(PALETTE)} palette colors; some will repeat.")
    return colors, labels


def load_and_merge(kernels: dict[str, str]) -> pd.DataFrame:
    """Load one CSV per kernel name, relabel split_kv rows per name, merge.

    `kernels` maps an arbitrary display name (e.g. "v1", "pipelined") to a
    microbench_cuda.py CSV path. Each CSV's "cuda_split_kv" rows are
    relabeled to "cuda_split_kv_<name>" so they don't collide across kernels.
    flashinfer/cuda_naive rows aren't versioned, so only the first CSV that
    contains them contributes those rows.
    """
    dfs = []
    have_flashinfer = have_naive = False
    for name, path in kernels.items():
        d = pd.read_csv(path)

        if not have_flashinfer:
            fi = d[d["implementation"] == "flashinfer"]
            if not fi.empty:
                dfs.append(fi.copy())
                have_flashinfer = True

        if not have_naive:
            naive = d[d["implementation"] == "cuda_naive"]
            if not naive.empty:
                dfs.append(naive.copy())
                have_naive = True

        skv = d[d["implementation"] == "cuda_split_kv"].copy()
        if not skv.empty:
            skv["implementation"] = f"cuda_split_kv_{name}"
            dfs.append(skv)

    df = pd.concat(dfs, ignore_index=True)
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


# ── Plot 5: speedup heatmap (compare kernel best vs baseline kernel best) ──

def plot5(df: pd.DataFrame, out_dir: Path, baseline: str, compare: str):
    """Speedup heatmap of `compare` vs `baseline`, both kernel names as
    passed to --csv (e.g. baseline="v1", compare="pipelined")."""
    impl_base = f"cuda_split_kv_{baseline}"
    impl_cmp  = f"cuda_split_kv_{compare}"

    best_base = (df[df["implementation"] == impl_base]
               .groupby(["context_length", "batch_size"])["latency_ms_p50"]
               .min().reset_index()
               .rename(columns={"latency_ms_p50": "lat_base"}))

    best_cmp = (df[df["implementation"] == impl_cmp]
               .groupby(["context_length", "batch_size"])["latency_ms_p50"]
               .min().reset_index()
               .rename(columns={"latency_ms_p50": "lat_cmp"}))

    merged = best_base.merge(best_cmp, on=["context_length", "batch_size"])
    merged["speedup"] = merged["lat_base"] / merged["lat_cmp"]

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
    plt.colorbar(im, ax=ax, label=f"Speedup ({baseline} / {compare})")
    for i in range(len(ctx_vals)):
        for j in range(len(batch_vals)):
            ax.text(j, i, f"{matrix[i,j]:.2f}x",
                    ha="center", va="center", fontsize=8, color="black")
    fig.suptitle(f"Plot 5: Speedup — {LABELS[impl_cmp]} (best) vs "
                 f"{LABELS[impl_base]} (best)  (A100)",
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

    bar_impls = [(impl, color) for impl, color in COLORS.items()
                 if impl not in FIXED_COLORS]

    for ax, batch in zip(axes, batch_sizes):
        sub_fi = fi[fi["batch_size"] == batch].sort_values("context_length")
        ctx_labels = [f"{c//1024}k" for c in sub_fi["context_length"]]
        x = np.arange(len(sub_fi))
        width = 0.8 / max(len(bar_impls), 1)

        for offset, (impl, color) in enumerate(bar_impls):
            best = (df[df["implementation"] == impl]
                    .groupby(["context_length", "batch_size"])["latency_ms_p50"]
                    .min().reset_index()
                    .rename(columns={"latency_ms_p50": "lat_split"}))
            merged = sub_fi.merge(best, on=["context_length", "batch_size"])
            merged["ratio"] = merged["lat_split"] / merged["lat_fi"]
            merged = merged.sort_values("context_length")

            center_offset = offset - (len(bar_impls) - 1) / 2
            bars = ax.bar(x + center_offset * width, merged["ratio"],
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
    p.add_argument("--v1", default=None,
                   help="Shortcut for --csv v1=<path> (back-compat)")
    p.add_argument("--v2", default=None,
                   help="Shortcut for --csv v2=<path> (back-compat)")
    p.add_argument("--csv", action="append", default=None, metavar="NAME=PATH",
                   help="Kernel CSV to include, as name=path. Repeatable, e.g. "
                        "--csv v1=results/microbench_cuda_v1.csv "
                        "--csv pipelined=results/microbench_cuda_pipelined.csv. "
                        "Each CSV's cuda_split_kv rows are relabeled "
                        "cuda_split_kv_<name>; flashinfer/cuda_naive rows are "
                        "taken from whichever CSV has them first.")
    p.add_argument("--out", default="results/plots_cuda")
    p.add_argument("--peak-compute", type=float, default=312.0,
                   help="GPU peak FP16 TFLOPs (default: 312 for A100)")
    p.add_argument("--peak-bw", type=float, default=2000.0,
                   help="GPU peak HBM bandwidth GB/s (default: 2000 for A100)")
    p.add_argument("--baseline", default=None,
                   help="Kernel name (as given to --csv) used as the "
                        "denominator in Plot 5's speedup heatmap. Default: "
                        "first kernel name in sorted order.")
    p.add_argument("--compare", default=None,
                   help="Kernel name compared against --baseline in Plot 5. "
                        "Default: second kernel name in sorted order.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    kernels = {}
    if args.v1:
        kernels["v1"] = args.v1
    if args.v2:
        kernels["v2"] = args.v2
    for item in args.csv or []:
        if "=" not in item:
            raise SystemExit(f"--csv must be NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        kernels[name] = path
    if not kernels:
        kernels = {"v1": "results/microbench_cuda_v1.csv",
                   "v2": "results/microbench_cuda_v2.csv"}

    df = load_and_merge(kernels)
    COLORS, LABELS = build_style(df["implementation"].unique())
    print(f"Context lengths: {sorted(df['context_length'].unique())}")
    print(f"Batch sizes:     {sorted(df['batch_size'].unique())}\n")

    plot1(df, out_dir)
    plot2(df, out_dir)
    plot3(df, out_dir)
    plot4(df, out_dir, peak_compute_tflops=args.peak_compute, peak_bw_gb_s=args.peak_bw)

    split_kv_names = sorted(kernels.keys())
    baseline = args.baseline or (split_kv_names[0] if split_kv_names else None)
    compare  = args.compare  or (split_kv_names[1] if len(split_kv_names) > 1 else None)
    if baseline and compare and baseline != compare:
        plot5(df, out_dir, baseline, compare)
    else:
        print(f"Skipping Plot 5 (speedup heatmap): need two distinct kernel "
              f"names, got baseline={baseline!r} compare={compare!r}. "
              f"Pass --baseline/--compare explicitly if you have 3+ kernels.")

    plot6(df, out_dir)

    print(f"\nAll plots saved to {out_dir}/")
