#!/usr/bin/env python3
"""isolation_overhead/plot.py — Service-time bar chart for the paper.

X-axis: backbones. Y-axis: server-side service time (ms). One bar group per
isolation mode (ST / FMVisor / Process). Bar height = chosen statistic
(mean by default); black error caps mark the p99 tail above the bar height
(when the bar is not already p99).

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
# Paper style — aligned with serving/experiments/fair_share/tsfm/plot.py
# (fonts, ticks, grid, PDF embedding) for consistency across paper figures.
# ---------------------------------------------------------------------------

MODE_ORDER  = ["none", "shared", "process"]
MODE_LABELS = {
    "none":    "ST",
    "shared":  "FMVisor",
    "process": "Process",
}
MODE_COLORS = {
    "none":    "#6B9AC4",  # muted blue
    "shared":  "#E06C75",  # pink-red
    "process": "#888888",  # mid gray
}
# Diagonal hatching on FMVisor for grayscale / light-color distinction
MODE_HATCH = {
    "none":    None,
    "shared":  "//",
    "process": None,
}

# Figure typography (pt) — large for projection / printed poster-scale PDFs
LABEL_FS = 24
TICK_FS = 22
LEGEND_FS = 22
BAR_VALUE_FS = 17

# Pretty backbone names for x-axis tick labels
BACKBONE_LABELS = {
    "momentlarge": "MOMENT-Large",
    "papageip": "Papageip",
    "dinobase": "DINOv2-Base",
    "swinlarge":"Swin-Large"

}

# Fixed y-axis top (ms); ticks end at 40 for this figure scale
Y_AXIS_MAX = 40.0


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     "black",
        "axes.labelcolor":    "black",
        "axes.linewidth":     0.7,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.color":         "#cccccc",
        "grid.linestyle":     ":",
        "grid.linewidth":     0.5,
        "grid.alpha":         1.0,
        "xtick.color":        "black",
        "ytick.color":        "black",
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.width":  0.9,
        "ytick.major.width":  0.9,
        "xtick.major.size":   4.5,
        "ytick.major.size":   4.5,
        "text.color":         "black",
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          18,
        "axes.titlesize":     20,
        "axes.labelsize":     20,
        "xtick.labelsize":    18,
        "ytick.labelsize":    18,
        "legend.fontsize":    18,
        "legend.frameon":     False,
        "legend.loc":         "upper center",
        "lines.linewidth":    1.6,
        "hatch.linewidth":    0.85,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.facecolor":  "white",
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
    })


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    print(f"[Plot] saved {out_path.with_suffix('.pdf')}")


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

    # Wide/tall enough that 18–20 pt text does not crowd
    fig_w = max(7.0, 1.2 * len(backbones) * max(n_modes, 1) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, 3.8))

    bar_groups: list[tuple[object, list[float], list[float]]] = []

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
        hatch = MODE_HATCH.get(mode)
        bars = ax.bar(
            x + offset, heights, width,
            color=MODE_COLORS[mode],
            edgecolor="black",
            linewidth=0.6,
            hatch=hatch,
            label=MODE_LABELS[mode],
            yerr=errs if any(e > 0 for e in errs) else None,
            error_kw={
                "ecolor": "black",
                "elinewidth": 0.8,
                "capsize": 4.5,
                "capthick": 1.0,
            },
        )
        bar_groups.append((bars, heights, errs))

    ax.set_xticks(x)
    xtick_labels = [BACKBONE_LABELS.get(bb.lower(), bb) for bb in backbones]
    ax.set_xticklabels(xtick_labels, rotation=0)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS, width=0.9, length=4.5)
    if metric == "avg":
        ax.set_ylabel("Service time (ms)", fontsize=LABEL_FS)
    else:
        ax.set_ylabel(f"{METRIC_LABEL[metric]} service time (ms)", fontsize=LABEL_FS)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.margins(x=0.02)
    ymax = max(
        float(df[metric_col].max()),
        float(df[err_col].max()) if metric != "p99" else float(df[metric_col].max()),
    )
    # Gap above the error-bar cap (mean + Δ to p99), not above the bar top, so
    # labels do not sit on the interval whiskers.
    label_pad = ymax * 0.045

    for bars, heights, errs in bar_groups:
        for bar, val, err in zip(bars, heights, errs):
            if val <= 0:
                continue
            y_txt = val + err + label_pad
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_txt,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=BAR_VALUE_FS,
                fontweight="semibold",
                color="black",
            )

    ylim_hi = Y_AXIS_MAX
    ax.set_ylim(0, ylim_hi)
    ax.set_yticks(np.arange(0, int(Y_AXIS_MAX) + 1, 10))

    ax.legend(
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FS,
        handlelength=1.8,
        columnspacing=1.5,
        borderaxespad=0.4,
    )

    fig.tight_layout()
    save_figure(fig, out_path)
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
