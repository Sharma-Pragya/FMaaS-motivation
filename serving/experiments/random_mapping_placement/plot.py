#!/usr/bin/env python3
"""Plot placement results with confidence intervals.

Usage:
    python plot.py [output_dir] [--mode fixed-n|admission] [--output plot.pdf]

Examples:
    # Plot specific run
    python plot.py outputs/run_20260513_120000 --mode admission
    
    # Plot all runs in outputs/
    python plot.py --mode admission
    
    # Save to file
    python plot.py outputs/run_20260513_120000 --output results.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

# Palette aligned with sharing_benefit / end_to_end_realworld_mix paper plots.
CONDITION_COLORS = {
    "fmaas":      "#E06C75",  # FMVisor – pink-red
    "no_sharing": "#888888",  # BE      – mid grey
}
CONDITION_LABELS = {
    "fmaas":      "FMVisor",
    "no_sharing": "BE",
}
# Plot / legend order: BE (baseline) first, then FMVisor
CONDITION_ORDER = ["no_sharing", "fmaas"]
REGIME_ORDER = {"low": 0, "medium": 1, "high": 2}
CONDITION_HATCH = {
    "no_sharing": None,
    "fmaas":      "//",
}


def _paper_style() -> None:
    """Publication-ready rcParams matching the other paper plots."""
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     "black",
        "axes.labelcolor":    "black",
        "axes.linewidth":     0.9,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.color":         "#cccccc",
        "grid.linestyle":     ":",
        "grid.linewidth":     0.6,
        "grid.alpha":         1.0,
        "xtick.color":        "black",
        "ytick.color":        "black",
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.width":  0.9,
        "ytick.major.width":  0.9,
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "text.color":         "black",
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          20,
        "axes.titlesize":     22,
        "axes.labelsize":     22,
        "xtick.labelsize":    20,
        "ytick.labelsize":    20,
        "legend.fontsize":    20,
        "legend.frameon":     False,
        "lines.linewidth":    1.8,
        "hatch.linewidth":    0.85,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.facecolor":  "white",
    })


def _format_count(v: float) -> str:
    """Compact tick / annotation label: 1947 -> '1.9k', 312 -> '312'."""
    if not np.isfinite(v) or v <= 0:
        return ""
    if v >= 1000:
        return f"{v / 1000:.1f}k".replace(".0k", "k")
    return f"{v:.0f}"


def plot_placement_results(summary_path: Path, mode: str,
                           output_path: Path | None = None) -> None:
    """Plot placement results with 95 % confidence intervals — paper-ready,
    single-column-wide layout with the legend above the axes."""

    with open(summary_path, "r") as f:
        data = json.load(f)

    scenarios = data["scenarios"]

    regimes = sorted({s["regime"] for s in scenarios},
                     key=lambda r: REGIME_ORDER.get(r, 999))

    metric_key = "admitted_before_failure" if mode == "admission" else "placed_count"

    placed_counts = {regime: {} for regime in regimes}
    for scenario in scenarios:
        regime = scenario["regime"]
        for cond_key, cond_data in scenario["conditions"].items():
            if cond_key not in CONDITION_LABELS:
                continue
            if metric_key not in cond_data:
                print(f"Warning: {metric_key} not found in {cond_key} data")
                continue
            m = cond_data[metric_key]
            placed_counts[regime][cond_key] = {
                "mean":    m["mean"],
                "ci_low":  m["ci95_low"],
                "ci_high": m["ci95_high"],
                "std":     m["std"],
            }

    _paper_style()

    # Wide-aspect single-column layout: ~ full text-column width with room
    # for the bigger fonts.
    fig, ax = plt.subplots(figsize=(7.0, 2.6))

    x_positions = np.arange(len(regimes))
    width = 0.36

    all_means: list[float] = []
    for i, cond in enumerate(CONDITION_ORDER):
        means: list[float] = []
        err_lo: list[float] = []
        err_hi: list[float] = []
        for regime in regimes:
            dp = placed_counts[regime].get(cond)
            if dp is None:
                means.append(np.nan)
                err_lo.append(0.0)
                err_hi.append(0.0)
            else:
                means.append(dp["mean"])
                err_lo.append(max(dp["mean"] - dp["ci_low"], 0.0))
                err_hi.append(max(dp["ci_high"] - dp["mean"], 0.0))

        offset = width * (i - 0.5)
        xs = x_positions + offset
        hatch = CONDITION_HATCH.get(cond)
        bars = ax.bar(
            xs, means, width,
            label=CONDITION_LABELS[cond],
            color=CONDITION_COLORS[cond],
            edgecolor="black", linewidth=0.7,
            hatch=hatch,
            yerr=[err_lo, err_hi],
            capsize=4,
            error_kw={"linewidth": 1.2, "ecolor": "black"},
            zorder=2,
        )

        # Value label above each bar (just above the CI whisker) so the
        # actual counts are readable even on a log axis.
        for xi, m, eh in zip(xs, means, err_hi):
            if not np.isfinite(m) or m <= 0:
                continue
            label_y = (m + eh) * 1.15
            ax.text(xi, label_y, _format_count(m),
                    ha="center", va="bottom",
                    fontsize=16, color="black")
            all_means.append(m)

    ax.set_ylabel("# Tasks Placed", labelpad=6)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([r.capitalize() for r in regimes])

    ax.set_yscale("log")
    if all_means:
        ymin = min(all_means) * 0.4
        ymax = max(all_means) * 5.0
        # Snap limits onto integer decades for a clean look.
        lo_exp = int(np.floor(np.log10(max(ymin, 1e-6))))
        hi_exp = int(np.ceil(np.log10(max(ymax, 1e-6))))
        ax.set_ylim(10 ** lo_exp, 10 ** hi_exp)
        # Every full decade in range (e.g. 10^2, 10^3, …); LogLocator numticks
        # can omit an end tick like 10^2.
        major_ticks = [10 ** k for k in range(lo_exp, hi_exp + 1)]
        ax.yaxis.set_major_locator(mticker.FixedLocator(major_ticks))
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0,
                               subs=np.arange(2, 10) * 0.1,
                               numticks=12))
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax.grid(axis="y", which="major", zorder=0)
    ax.grid(axis="y", which="minor", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.08)

    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 0.85),
        ncol=len(CONDITION_ORDER), frameon=False,
        handlelength=2.0, columnspacing=2.0, handletextpad=0.5,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(top=0.96, bottom=0.22, left=0.12, right=0.98)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', nargs='?', default=None,
                       help='Path to experiment output directory. If not provided, plots all runs in outputs/')
    parser.add_argument('--mode', choices=['fixed-n', 'admission'], 
                       default='admission',
                       help='Experiment mode (default: admission)')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output plot path (e.g., plot.pdf, plot.png). Only used with specific output_dir')
    args = parser.parse_args()
    
    # Determine which directories to process
    if args.output_dir:
        output_dirs = [Path(args.output_dir)]
    else:
        # Find all run directories
        if not DEFAULT_OUTPUT_ROOT.exists():
            print(f"Error: outputs directory not found at {DEFAULT_OUTPUT_ROOT}")
            return 1
        output_dirs = sorted([d for d in DEFAULT_OUTPUT_ROOT.iterdir() if d.is_dir()])
        if not output_dirs:
            print(f"Error: No run directories found in {DEFAULT_OUTPUT_ROOT}")
            return 1
        print(f"Found {len(output_dirs)} run directories")
    
    # Process each directory
    for output_dir in output_dirs:
        if args.mode == 'admission':
            summary_file = output_dir / 'admission_aggregate_summary.json'
        else:
            summary_file = output_dir / 'aggregate_summary.json'
        
        if not summary_file.exists():
            print(f"Skipping {output_dir.name}: Summary file not found")
            continue
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = output_dir / f'placement_results_{args.mode}.pdf'
        
        print(f"Processing {output_dir.name}...")
        plot_placement_results(summary_file, args.mode, Path(output_path))
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
