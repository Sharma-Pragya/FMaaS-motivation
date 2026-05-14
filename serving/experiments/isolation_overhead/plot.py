#!/usr/bin/env python3
"""isolation_overhead/plot.py — Service-time bar chart for the paper.

X-axis: backbones. Y-axis: server-side service time (ms). One bar group per
isolation mode (none / shared / process). Bar height = mean service time;
black error caps mark the p99.

Usage:
    python experiments/isolation_overhead/plot.py \
        [--exp-dir experiments/isolation_overhead/results] \
        [--out-dir experiments/isolation_overhead/plots]   \
        [--metric avg|p50|p95|p99]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paper style (mirrors sharing_benefit/tsfm/plot.py)
# ---------------------------------------------------------------------------

MODE_ORDER  = ["none", "shared", "process"]
MODE_LABELS = {
    "none":    "No Isolation",
    "shared":  "FMVisor (Shared)",
    "process": "Process Isolation",
}
MODE_COLORS = {
    "none":    "#6B9AC4",  # muted blue
    "shared":  "#E06C75",  # pink-red
    "process": "#888888",  # mid gray
}


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "black",
        "axes.labelcolor":   "black",
        "axes.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "grid.color":        "#cccccc",
        "grid.linestyle":    ":",
        "grid.linewidth":    0.4,
        "grid.alpha":        1.0,
        "xtick.color":       "black",
        "ytick.color":       "black",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size":  2.5,
        "ytick.major.size":  2.5,
        "text.color":        "black",
        "font.family":       "sans-serif",
        "font.size":         7,
        "axes.titlesize":    7.5,
        "axes.labelsize":    7,
        "xtick.labelsize":   6.5,
        "ytick.labelsize":   6.5,
        "legend.fontsize":   6.5,
        "lines.linewidth":   1.0,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "savefig.facecolor": "white",
        "savefig.bbox":      "tight",
    })


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRIC_COL = {
    "avg": "avg_svc_ms",
    "p50": "p50_svc_ms",
    "p95": "p95_svc_ms",
    "p99": "p99_svc_ms",
}
METRIC_LABEL = {
    "avg": "Mean",
    "p50": "Median",
    "p95": "p95",
    "p99": "p99",
}


def plot_service_time(df: pd.DataFrame, out_path: Path, metric: str = "avg") -> None:
    metric_col = METRIC_COL[metric]
    err_col    = "p99_svc_ms"  # p99 cap for visual context (skipped when metric == p99)

    # Stable backbone ordering: first appearance in the CSV
    backbones = list(dict.fromkeys(df["backbone"].tolist()))
    modes     = [m for m in MODE_ORDER if m in df["isolation_mode"].unique()]

    x        = np.arange(len(backbones))
    n_modes  = len(modes)
    width    = 0.8 / max(n_modes, 1)

    # Sized for a single-column paper figure
    fig, ax = plt.subplots(figsize=(max(3.4, 0.9 * len(backbones) * n_modes + 1.2), 2.4))

    for i, mode in enumerate(modes):
        heights = []
        errs    = []
        for bb in backbones:
            row = df[(df["isolation_mode"] == mode) & (df["backbone"] == bb)]
            if row.empty:
                heights.append(0.0)
                errs.append(0.0)
                continue
            h = float(row[metric_col].iloc[0])
            heights.append(h)
            if metric != "p99":
                p99 = float(row[err_col].iloc[0])
                errs.append(max(0.0, p99 - h))
            else:
                errs.append(0.0)

        offset = (i - (n_modes - 1) / 2) * width
        bars = ax.bar(
            x + offset, heights, width,
            color=MODE_COLORS[mode],
            edgecolor="black", linewidth=0.5,
            label=MODE_LABELS[mode],
            yerr=errs if any(e > 0 for e in errs) else None,
            error_kw={"ecolor": "black", "elinewidth": 0.6, "capsize": 2.0, "capthick": 0.6},
        )
        for bar, val in zip(bars, heights):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=5.5,
                    color="black",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(backbones, rotation=0)
    ax.set_xlabel("Backbone")
    ylabel = f"Service Time ({METRIC_LABEL[metric]}, ms)"
    if metric != "p99":
        ylabel += "  —  caps: p99"
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncol=min(n_modes, 3),
              handlelength=1.5, columnspacing=1.2)

    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    print(f"[Plot] Saved: {out_path}  (+ .pdf)")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Plot isolation_overhead service-time bars")
    parser.add_argument("--exp-dir", type=Path,
                        default=Path(__file__).parent / "results")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).parent / "plots")
    parser.add_argument("--metric", choices=list(METRIC_COL.keys()), default="avg",
                        help="Which service-time statistic to use for bar height (default: avg)")
    args = parser.parse_args()

    csv_path = args.exp_dir / "summary.csv"
    if not csv_path.exists():
        print(f"[ERR] summary.csv not found at: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    required = {"isolation_mode", "backbone", METRIC_COL[args.metric], "p99_svc_ms"}
    missing  = required - set(df.columns)
    if missing:
        print(f"[ERR] summary.csv missing columns: {missing}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    out_path = args.out_dir / f"service_time_by_backbone_{args.metric}.png"
    plot_service_time(df, out_path, metric=args.metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
