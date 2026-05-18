#!/usr/bin/env python3
"""Plot Motivation Experiment #2 results.

Dual y-axis figure:
  Left  axis  (bars)  — GPU memory (GB)
  Right axis  (lines) — Throughput (req/s)
  X-axis              — Number of tasks
  Two series          — task_sharing vs deployment_sharing
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D

SERVING_DIR = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Load summary.csv
# ---------------------------------------------------------------------------

def load_summary(path: Path) -> Dict[str, Dict[int, Dict]]:
    """Returns {strategy: {n_tasks: row_dict}}."""
    data: Dict[str, Dict[int, Dict]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            strategy = row["strategy"]
            n_tasks = int(row["n_tasks"])
            data.setdefault(strategy, {})[n_tasks] = {
                "gpu_mem_mb": float(row["avg_gpu_mem_mb"]),
                "throughput_rps": float(row["throughput_rps"]),
                "avg_latency_ms": float(row["avg_latency_ms"]),
            }
    return data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

STRATEGY_LABELS = {
    "task_sharing": "Instance-per-Task",
    "deploy_sharing": "Deployment Sharing",
}

# Missing rows for this strategy are treated as OOM (annotate, no bar).
OOM_STRATEGY = "task_sharing"
DEPLOY_STRATEGY = "deploy_sharing"
# Hatch pattern on bars (Deployment Sharing only); lines stay solid colors.
BAR_HATCHES = {
    "task_sharing": "",
    "deploy_sharing": "//",
}
# Colors / line geometry aligned with serving/experiments/RTVSntask/tpc/plot.py
# (no_sharing_tpc vs sharing palette).
COLORS = {
    "task_sharing": "#6B9AC4",
    "deploy_sharing": "#E06C75",
}

LINE_MARKERS = {
    "task_sharing": "^",
    "deploy_sharing": "o",
}

LINE_STYLES = {
    "task_sharing": "--",
    "deploy_sharing": "-",
}

# Same numbers as RTVSntask condition line plots (COND_LINE_*).
COND_LINE_MARKERSIZE = 6
COND_LINE_MARKEREDGEWIDTH = 0.25
COND_LINE_MARKER_EDGECOLOR = "black"
COND_LINE_LINEWIDTH = 0.9
BAR_EDGEWIDTH = 0.4

# OOM annotation (missing Instance-per-task row).
OOM_FONT_SIZE = 13

# Explicit sizes so PDFs stay readable (rcParams alone can look small after layout).
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 12

# Wide layout — dual-axis + large fonts need horizontal room.
FIGSIZE_MEM_THROUGHPUT = (7.8, 3.05)
FIGSIZE_SINGLE_METRIC = (7.2, 2.95)


def _bar_group_x_positions(n_groups: int, n_strategies: int, bar_width: float) -> np.ndarray:
    """X centers for each #Tasks group; spacing keeps bar pairs from crowding."""
    inner = bar_width * n_strategies + 0.24
    between_groups = 0.32
    step = inner + between_groups
    return np.arange(n_groups, dtype=float) * step


def mb_to_gb(value_mb: float) -> float:
    return value_mb / 1024.0


def nice_axis_upper(values: List[float], steps=(1, 2, 2.5, 5, 10), headroom: float = 1.0) -> float:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return 1.0

    max_val = max(finite) * headroom
    if max_val <= 0:
        return 1.0

    exponent = np.floor(np.log10(max_val))
    base = 10 ** exponent
    for step in steps:
        candidate = step * base
        if candidate >= max_val:
            return float(candidate)
    return float(steps[0] * 10 * base)


def strategy_handles(strategies: List[str]) -> List[mpatches.Patch]:
    return [
        mpatches.Patch(
            facecolor=COLORS.get(strategy, "gray"),
            edgecolor="black",
            linewidth=BAR_EDGEWIDTH,
            hatch=BAR_HATCHES.get(strategy, ""),
            label=STRATEGY_LABELS.get(strategy, strategy),
        )
        for strategy in strategies
    ]


def throughput_scale(values: List[float], mode: str) -> List[float]:
    if mode == "none":
        return values
    if mode == "max":
        finite = [v for v in values if np.isfinite(v)]
        if not finite:
            return values
        denom = max(finite)
        if denom <= 0:
            return values
        return [v / denom if np.isfinite(v) else v for v in values]
    raise ValueError(f"Unknown throughput normalization mode: {mode}")


def apply_paper_style() -> None:
    """Paper-style axes with readable font sizes for motivation figures."""
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     "black",
        "axes.labelcolor":    "black",
        "axes.linewidth":     0.6,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.color":         "#cccccc",
        "grid.linestyle":     ":",
        "grid.linewidth":     0.4,
        "grid.alpha":         1.0,
        "xtick.color":        "black",
        "ytick.color":        "black",
        "xtick.major.width":  0.5,
        "ytick.major.width":  0.5,
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "text.color":         "black",
        "font.family":        "sans-serif",
        "font.size":          13,
        "axes.titlesize":     14,
        "axes.labelsize":     14,
        "xtick.labelsize":    12,
        "ytick.labelsize":    12,
        "legend.fontsize":    12,
        "lines.linewidth":    1.2,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.facecolor":  "white",
        "savefig.bbox":       "tight",
    })


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[Plot] Saved: {out_path}")


def make_plot(data: Dict, out_path: Path, strategies: List[str], all_n: List[int], normalize_throughput: str) -> None:
    fig, ax_mem = plt.subplots(figsize=FIGSIZE_MEM_THROUGHPUT)
    ax_thr = ax_mem.twinx()

    n_strategies = len(strategies)
    bar_width = 0.38
    x_positions = _bar_group_x_positions(len(all_n), n_strategies, bar_width)

    mem_values_all = []
    thr_values_all = []
    oom_indices: List[int] = []

    for si, strategy in enumerate(strategies):
        strategy_data = data.get(strategy, {})
        offset = (si - (n_strategies - 1) / 2) * bar_width
        xs_b: List[float] = []
        hs_b: List[float] = []
        for i, n in enumerate(all_n):
            row = strategy_data.get(n)
            if row is None:
                if strategy == OOM_STRATEGY:
                    oom_indices.append(i)
                continue
            v_gb = mb_to_gb(float(row["gpu_mem_mb"]))
            mem_values_all.append(v_gb)
            xs_b.append(float(x_positions[i] + offset))
            hs_b.append(v_gb)
        if xs_b:
            ax_mem.bar(
                xs_b,
                hs_b,
                width=bar_width,
                color=COLORS.get(strategy, "gray"),
                edgecolor="black",
                linewidth=BAR_EDGEWIDTH,
                hatch=BAR_HATCHES.get(strategy, ""),
                zorder=2,
            )

    for strategy in strategies:
        strategy_data = data.get(strategy, {})
        thr_vals = [strategy_data.get(n, {}).get("throughput_rps", float("nan")) for n in all_n]
        thr_vals = throughput_scale(thr_vals, normalize_throughput)
        thr_values_all.extend(thr_vals)
        ax_thr.plot(
            x_positions,
            thr_vals,
            color=COLORS.get(strategy, "gray"),
            linestyle=LINE_STYLES.get(strategy, "-"),
            marker=LINE_MARKERS.get(strategy, "o"),
            markersize=COND_LINE_MARKERSIZE,
            markerfacecolor=COLORS.get(strategy, "gray"),
            markeredgecolor=COND_LINE_MARKER_EDGECOLOR,
            markeredgewidth=COND_LINE_MARKEREDGEWIDTH,
            linewidth=COND_LINE_LINEWIDTH,
            zorder=3,
        )

    ax_mem.set_xlabel("#Tasks", fontsize=AXIS_LABEL_FONTSIZE)
    ax_mem.set_ylabel("GPU Memory (GB)", color="black", fontsize=AXIS_LABEL_FONTSIZE)
    thr_ylabel = "Throughput (req/s)" if normalize_throughput == "none" else "Normalized Throughput"
    ax_thr.set_ylabel(thr_ylabel, color="black", fontsize=AXIS_LABEL_FONTSIZE)

    ax_mem.set_xticks(x_positions)
    ax_mem.set_xticklabels([str(n) for n in all_n])
    ax_mem.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax_thr.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)

    mem_upper = nice_axis_upper(mem_values_all)
    if normalize_throughput == "none":
        # Keep req/s axis tight; avoid jumping to coarse bounds like 200 unnecessarily.
        thr_upper = nice_axis_upper(thr_values_all, steps=(1, 1.2, 1.5, 1.6, 2, 2.5, 3, 5, 10), headroom=1.05)
    else:
        thr_upper = nice_axis_upper(thr_values_all)
    ax_mem.set_ylim(0, mem_upper)
    ax_thr.set_ylim(0, thr_upper)

    # Fewer y-ticks (dual-axis plots get busy otherwise).
    y_bins = 4
    ax_mem.yaxis.set_major_locator(ticker.MaxNLocator(nbins=y_bins, min_n_ticks=3))
    ax_thr.yaxis.set_major_locator(ticker.MaxNLocator(nbins=y_bins, min_n_ticks=3))
    ax_mem.minorticks_off()
    ax_thr.minorticks_off()

    ax_mem.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    if normalize_throughput == "none":
        ax_thr.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}" if thr_upper >= 20 else f"{v:.1f}"))
    else:
        ax_thr.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))

    ax_mem.grid(axis="both", zorder=0)
    ax_mem.set_axisbelow(True)
    half_group = (n_strategies * bar_width) / 2 + 0.14
    ax_mem.set_xlim(float(x_positions[0] - half_group - 0.08), float(x_positions[-1] + half_group + 0.08))

    ax_thr.spines["right"].set_visible(True)
    ax_thr.spines["right"].set_color("black")
    ax_thr.spines["right"].set_linewidth(0.6)

    for i in oom_indices:
        if OOM_STRATEGY not in strategies:
            continue
        si_ipt = strategies.index(OOM_STRATEGY)
        offset_ipt = (si_ipt - (n_strategies - 1) / 2) * bar_width
        # Center of missing Instance-per-task slot — avoids sitting on Deployment Sharing bar.
        x_oom = float(x_positions[i] + offset_ipt)
        n = all_n[i]
        row_ds = data.get(DEPLOY_STRATEGY, {}).get(n) if DEPLOY_STRATEGY in strategies else None
        if row_ds is not None:
            h_bar = mb_to_gb(float(row_ds["gpu_mem_mb"]))
            # Sit clearly above the deploy bar (not on the x-axis).
            y_oom = float(max(h_bar + mem_upper * 0.06, mem_upper * 0.11))
        else:
            y_oom = float(mem_upper * 0.11)
        ax_mem.text(
            x_oom,
            y_oom,
            "OOM",
            rotation=90,
            ha="center",
            va="center",
            color=COLORS[OOM_STRATEGY],
            fontsize=OOM_FONT_SIZE,
            fontweight="bold",
            zorder=5,
            clip_on=False,
        )

    legend_handles = strategy_handles(strategies) + [
        mpatches.Patch(
            facecolor="white",
            edgecolor="black",
            linewidth=BAR_EDGEWIDTH,
            label="GPU Memory",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=1.4,
            marker="",
            markersize=0,
            label="Throughput",
        ),
    ]
    ax_mem.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        bbox_transform=ax_mem.transAxes,
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        columnspacing=1.15,
        handletextpad=0.5,
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.5)
    fig.subplots_adjust(top=0.84, right=0.88)
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default=os.environ.get("EXP_DIR", "experiments/motivation1/tsfm/results"))
    parser.add_argument("--strategies", default="task_sharing,deploy_sharing")
    parser.add_argument("--normalize-throughput", choices=["none", "max"], default="none")
    args = parser.parse_args()

    result_root = (SERVING_DIR / args.exp_dir).resolve()
    summary_path = result_root / "summary.csv"

    if not summary_path.exists():
        print(f"[Error] summary.csv not found at {summary_path}")
        return 1

    strategies = [s.strip() for s in args.strategies.split(",")]
    data = load_summary(summary_path)

    all_n = sorted({n for strategy_data in data.values() for n in strategy_data})
    if not all_n:
        print("[Error] No data rows found in summary.csv")
        return 1

    print(f"[Plot] strategies={strategies}  n_tasks={all_n}")

    out_dir = result_root
    make_plot(data, out_dir / "motivation1_memory_throughput.pdf", strategies, all_n, args.normalize_throughput)
    _plot_single(data, out_dir / "motivation1_memory.pdf", strategies, all_n, metric="gpu_mem_mb", ylabel="GPU Memory (GB)")
    thr_ylabel = "Throughput (req/s)" if args.normalize_throughput == "none" else "Normalized Throughput"
    _plot_single(
        data,
        out_dir / "motivation1_throughput.pdf",
        strategies,
        all_n,
        metric="throughput_rps",
        ylabel=thr_ylabel,
        normalize_throughput=args.normalize_throughput,
    )
    _plot_single(data, out_dir / "motivation1_latency.pdf", strategies, all_n, metric="avg_latency_ms", ylabel="Avg Latency (ms)")
    return 0


def _plot_single(
    data: Dict,
    out_path: Path,
    strategies: List[str],
    all_n: List[int],
    metric: str,
    ylabel: str,
    normalize_throughput: str = "none",
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_METRIC)
    n_strategies = len(strategies)
    bar_width = 0.38
    x_positions = _bar_group_x_positions(len(all_n), n_strategies, bar_width)

    all_values = []
    oom_indices: List[int] = []

    for si, strategy in enumerate(strategies):
        strategy_data = data.get(strategy, {})
        offset = (si - (n_strategies - 1) / 2) * bar_width
        xs_b: List[float] = []
        hs_b: List[float] = []
        for i, n in enumerate(all_n):
            row = strategy_data.get(n)
            if row is None:
                if strategy == OOM_STRATEGY:
                    oom_indices.append(i)
                continue
            if metric == "gpu_mem_mb":
                v = mb_to_gb(float(row["gpu_mem_mb"]))
            else:
                v = float(row[metric])
            xs_b.append(float(x_positions[i] + offset))
            hs_b.append(v)

        if metric == "throughput_rps":
            hs_b = throughput_scale(hs_b, normalize_throughput)

        for v in hs_b:
            if np.isfinite(v):
                all_values.append(v)

        if xs_b:
            ax.bar(
                xs_b,
                hs_b,
                width=bar_width,
                color=COLORS.get(strategy, "gray"),
                edgecolor="black",
                linewidth=BAR_EDGEWIDTH,
                hatch=BAR_HATCHES.get(strategy, ""),
                zorder=2,
            )

    y_upper = nice_axis_upper(all_values)
    ax.set_xlabel("#Tasks", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(n) for n in all_n])
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_ylim(0, y_upper)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, min_n_ticks=3))
    ax.minorticks_off()

    for i in oom_indices:
        if OOM_STRATEGY not in strategies:
            continue
        si_ipt = strategies.index(OOM_STRATEGY)
        offset_ipt = (si_ipt - (n_strategies - 1) / 2) * bar_width
        x_oom = float(x_positions[i] + offset_ipt)
        n = all_n[i]
        row_ds = data.get(DEPLOY_STRATEGY, {}).get(n) if DEPLOY_STRATEGY in strategies else None
        if row_ds is not None:
            if metric == "gpu_mem_mb":
                h = mb_to_gb(float(row_ds["gpu_mem_mb"]))
            elif metric == "throughput_rps":
                h = throughput_scale([float(row_ds["throughput_rps"])], normalize_throughput)[0]
            else:
                h = float(row_ds["avg_latency_ms"])
            y_oom = (
                float(max(h + y_upper * 0.06, y_upper * 0.11))
                if np.isfinite(h) and h > 0
                else float(y_upper * 0.11)
            )
        else:
            y_oom = float(y_upper * 0.11)
        ax.text(
            x_oom,
            y_oom,
            "OOM",
            rotation=90,
            ha="center",
            va="center",
            color=COLORS[OOM_STRATEGY],
            fontsize=OOM_FONT_SIZE,
            fontweight="bold",
            zorder=5,
            clip_on=False,
        )

    if metric == "gpu_mem_mb":
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    elif metric == "avg_latency_ms":
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    elif metric == "throughput_rps" and normalize_throughput != "none":
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    else:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}" if y_upper >= 20 else f"{v:.1f}"))

    ax.grid(axis="both", zorder=0)
    ax.set_axisbelow(True)
    half_group = (n_strategies * bar_width) / 2 + 0.14
    ax.set_xlim(float(x_positions[0] - half_group - 0.08), float(x_positions[-1] + half_group + 0.08))
    ax.legend(
        handles=strategy_handles(strategies),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        bbox_transform=ax.transAxes,
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.5,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.5)
    fig.subplots_adjust(top=0.84)
    save_figure(fig, out_path)
    plt.close(fig)


if __name__ == "__main__":
    apply_paper_style()
    raise SystemExit(main())
