#!/usr/bin/env python3
"""motivation2/plot.py — Plots for the sharing-benefit experiment.

Produces:
  1. motivation2_latency_cdf.png      — CDF of per-request latency, one line per condition
  2. motivation2_throughput_cdf.png   — CDF of instantaneous throughput (1000/latency_ms)
  3. motivation2_summary_bars.png     — bar chart: p99 latency per condition per task

Usage:
    python experiments/motivation2/plot.py [--exp-dir experiments/motivation2/results]
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

PAPER_PALETTE = {
    "blue":       "#8FB7CF",
    "peach":      "#E8B298",
    "sage":       "#A9C7B5",
    "gold":       "#E7C98B",
    "lavender":   "#C7BEDF",
    "rose":       "#D9A6B3",
    "charcoal":   "#2F3640",
    "slate":      "#5C6773",
    "grid":       "#D9DEE5",
    "background": "#FAFBFC",
}

# Three logical series after averaging single tasks together
SERIES_ORDER  = ["single", "no_sharing", "sharing"]
SERIES_COLORS = {
    "single":     "#6B9AC4",   # muted blue  — matches microbenchmark C_BB tone
    "no_sharing": "#888888",   # mid gray
    "sharing":    "#E06C75",   # pink-red    — matches microbenchmark C_TASK
}
SERIES_LABELS = {
    "single":     "Single Task",
    "no_sharing": "Without Sharing",
    "sharing":    "With Sharing",
}
SERIES_LINESTYLE = {
    "single":     "--",
    "no_sharing": "-.",
    "sharing":    "-",
}

# Raw condition names still used for data loading
CONDITION_ORDER = ["single_ecgclass", "single_gestureclass", "no_sharing", "sharing"]


def apply_paper_style() -> None:
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
        "xtick.major.size":   2.5,
        "ytick.major.size":   2.5,
        "text.color":         "black",
        "font.family":        "sans-serif",
        "font.size":          7,
        "axes.titlesize":     7.5,
        "axes.labelsize":     7,
        "xtick.labelsize":    6.5,
        "ytick.labelsize":    6.5,
        "legend.fontsize":    6.5,
        "lines.linewidth":    1.2,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.facecolor":  "white",
        "savefig.bbox":       "tight",
    })


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[Plot] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_condition_latencies(
    result_root: Path,
    cond: str,
    warmup_secs: float,
    warmup_requests: int,
) -> List[float]:
    lat_file = result_root / cond / "latencies.csv"
    if not lat_file.exists():
        print(f"[Warn] {lat_file} not found, skipping {cond}")
        return []
    with lat_file.open() as f:
        reader = csv.DictReader(f)
        has_elapsed = "elapsed_sec" in (reader.fieldnames or [])
        rows = list(reader)
    if has_elapsed:
        kept = [float(r["latency_ms"]) for r in rows if float(r["elapsed_sec"]) > warmup_secs]
        dropped = len(rows) - len(kept)
    else:
        per_task: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            per_task[r["task"]].append(float(r["latency_ms"]))
        kept, dropped = [], 0
        for task, lats in per_task.items():
            dropped += min(warmup_requests, len(lats))
            kept.extend(lats[warmup_requests:])
    print(f"[Plot] {cond}: dropped {dropped}, kept {len(kept)}")
    return kept


def load_series_latencies(
    result_root: Path,
    warmup_secs: float = 10.0,
    warmup_requests: int = 180,
) -> Dict[str, List[float]]:
    """Returns {series: [latency_ms]} where series in SERIES_ORDER.

    single_ecgclass + single_gestureclass are averaged into one 'single' series.
    """
    raw: Dict[str, List[float]] = {}
    for cond in CONDITION_ORDER:
        lats = _read_condition_latencies(result_root, cond, warmup_secs, warmup_requests)
        if lats:
            raw[cond] = lats

    series: Dict[str, List[float]] = {}

    # Average single-task conditions: pool all latencies then subsample to equal size
    single_lats = []
    for cond in ("single_ecgclass", "single_gestureclass"):
        single_lats.extend(raw.get(cond, []))
    if single_lats:
        series["single"] = single_lats

    for cond in ("no_sharing", "sharing"):
        if cond in raw:
            series[cond] = raw[cond]

    return series


def load_task_results(result_root: Path) -> Dict[str, List[Dict]]:
    """Returns {condition: [task_row, ...]} from each condition's task_results.csv."""
    data: Dict[str, List[Dict]] = {}
    for cond in CONDITION_ORDER:
        path = result_root / cond / "task_results.csv"
        if not path.exists():
            continue
        with path.open() as f:
            data[cond] = list(csv.DictReader(f))
    return data


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _nice_upper(val: float) -> float:
    """Round up to nearest multiple of 10."""
    if val <= 0:
        return 10.0
    return float(np.ceil(val / 10.0) * 10)


def _plot_cdf_on_ax(ax: plt.Axes, series: Dict[str, List[float]], metric: str = "latency") -> None:
    all_vals: List[float] = []
    for s in SERIES_ORDER:
        lats = series.get(s)
        if not lats:
            continue
        arr = np.array(lats)
        if metric == "throughput":
            arr = 1000.0 / arr
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        ax.plot(sorted_arr, cdf,
                color=SERIES_COLORS[s],
                linestyle=SERIES_LINESTYLE[s],
                linewidth=1.0,
                label=SERIES_LABELS[s])
        all_vals.extend(sorted_arr.tolist())

    x_max = _nice_upper(float(np.max(all_vals)) if all_vals else 1.0)
    ax.set_ylim(0, 1.05)
    # Pick n_ticks so labels are round integers: try 3,4,5 ticks, pick first where x_max/(n-1) is integer
    for n_ticks in (3, 4, 5, 6):
        step = x_max / (n_ticks - 1)
        if step == int(step):
            break
    xticks = np.linspace(0, x_max, n_ticks)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlim(0, x_max)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.grid(axis="both", zorder=0)
    ax.set_axisbelow(True)


def _legend_handles() -> List:
    return [
        plt.Line2D([0], [0],
                   color=SERIES_COLORS[s],
                   linestyle=SERIES_LINESTYLE[s],
                   linewidth=1.0,
                   label=SERIES_LABELS[s])
        for s in SERIES_ORDER
    ]


# ---------------------------------------------------------------------------
# Plot 1: Latency CDF (single RPS)
# ---------------------------------------------------------------------------

def plot_latency_cdf(series: Dict[str, List[float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 1.3))
    _plot_cdf_on_ax(ax, series, metric="latency")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.legend(handles=_legend_handles(), frameon=False, loc="lower right",
              ncol=1, handlelength=1.5, handletextpad=0.3)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Real throughput CDF (completions per second, 1s bins)
# ---------------------------------------------------------------------------

def load_series_throughput(
    result_root: Path,
    warmup_secs: float = 10.0,
    warmup_requests: int = 180,
) -> Dict[str, List[float]]:
    """Returns {series: [throughput_per_second]} — completions in each 1s bin.

    completion_time = elapsed_sec + latency_ms / 1000
    Bins completions into 1s windows, returns list of per-second counts.
    Single task conditions are pooled then halved (per-task average).
    """
    def _completion_times(result_root: Path, cond: str) -> List[float]:
        lat_file = result_root / cond / "latencies.csv"
        if not lat_file.exists():
            return []
        completions = []
        with lat_file.open() as f:
            reader = csv.DictReader(f)
            has_elapsed = "elapsed_sec" in (reader.fieldnames or [])
            for row in reader:
                elapsed = float(row["elapsed_sec"]) if has_elapsed else warmup_secs + 1
                if elapsed <= warmup_secs:
                    continue
                completion = elapsed + float(row["latency_ms"]) / 1000.0
                completions.append(completion)
        return completions

    def _bin_throughput(completions: List[float]) -> List[float]:
        if not completions:
            return []
        t_min, t_max = min(completions), max(completions)
        bins = np.arange(t_min, t_max + 1, 1.0)
        counts, _ = np.histogram(completions, bins=bins)
        return counts.tolist()

    series: Dict[str, List[float]] = {}

    # Pool single task conditions — each runs independently so average per-task
    # Each condition has 1 task, so pool completions and divide by 2 (2 conditions)
    single_completions = (
        _completion_times(result_root, "single_ecgclass") +
        _completion_times(result_root, "single_gestureclass")
    )
    if single_completions:
        counts = _bin_throughput(single_completions)
        series["single"] = [c / 2.0 for c in counts]  # per-task average (2 tasks)

    for cond in ("no_sharing", "sharing"):
        completions = _completion_times(result_root, cond)
        if completions:
            counts = _bin_throughput(completions)
            series[cond] = [c / 2.0 for c in counts]  # per-task average (2 tasks)

    return series


def plot_throughput_cdf(throughput: Dict[str, List[float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 1.3))

    all_vals: List[float] = []
    for s in SERIES_ORDER:
        vals = throughput.get(s)
        if not vals:
            continue
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf,
                color=SERIES_COLORS[s],
                linestyle=SERIES_LINESTYLE[s],
                linewidth=1.0,
                label=SERIES_LABELS[s])
        all_vals.extend(sorted_vals.tolist())

    if all_vals:
        x_max = _nice_upper(float(np.max(all_vals)))
        for n_ticks in (3, 4, 5, 6):
            step = x_max / (n_ticks - 1)
            if step == int(step):
                break
        ax.set_xticks(np.linspace(0, x_max, n_ticks))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xlim(0, x_max)

    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.grid(axis="both", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Throughput (req/s)")
    ax.set_ylabel("CDF")
    ax.legend(handles=_legend_handles(), frameon=False, loc="lower right",
              ncol=1, handlelength=1.5, handletextpad=0.3)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: P99 latency bars (averaged single task)
# ---------------------------------------------------------------------------

def plot_summary_bars(task_results: Dict[str, List[Dict]], out_path: Path) -> None:
    # Build series p99: average single tasks, keep no_sharing/sharing as-is
    p99: Dict[str, float] = {}
    single_p99s = []
    for cond in ("single_ecgclass", "single_gestureclass"):
        rows = task_results.get(cond, [])
        vals = [float(r["p99_latency_ms"]) for r in rows if "p99_latency_ms" in r]
        single_p99s.extend(vals)
    if single_p99s:
        p99["single"] = float(np.mean(single_p99s))
    for cond in ("no_sharing", "sharing"):
        rows = task_results.get(cond, [])
        vals = [float(r["p99_latency_ms"]) for r in rows if "p99_latency_ms" in r]
        if vals:
            p99[cond] = float(np.mean(vals))

    series = [s for s in SERIES_ORDER if s in p99]
    if not series:
        print("[Warn] No task_results data, skipping bar chart")
        return

    fig, ax = plt.subplots(figsize=(2.5, 1.3))
    x = np.arange(len(series))
    bars = ax.bar(x, [p99[s] for s in series],
                  width=0.5,
                  color=[SERIES_COLORS[s] for s in series],
                  edgecolor="black", linewidth=0.4, zorder=2)
    for bar, s in zip(bars, series):
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                f"{v:.0f}", ha="center", va="bottom", fontsize=4.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SERIES_LABELS[s] for s in series], rotation=15, ha="right")
    ax.set_ylabel("P99 Latency (ms)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Sweep CDF — 3 subplots in one row, one per RPS level
# ---------------------------------------------------------------------------

def _plot_throughput_cdf_on_ax(ax: plt.Axes, series: Dict[str, List[float]]) -> None:
    """Plot throughput CDF on ax. series = {name: [per_second_counts]}."""
    all_vals: List[float] = []
    for s in SERIES_ORDER:
        vals = series.get(s)
        if not vals:
            continue
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf,
                color=SERIES_COLORS[s],
                linestyle=SERIES_LINESTYLE[s],
                linewidth=1.0,
                label=SERIES_LABELS[s])
        all_vals.extend(sorted_vals.tolist())
    if all_vals:
        x_max = _nice_upper(float(np.max(all_vals)))
        for n_ticks in (3, 4, 5, 6):
            step = x_max / (n_ticks - 1)
            if step == int(step):
                break
        ax.set_xticks(np.linspace(0, x_max, n_ticks))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.grid(axis="both", zorder=0)
    ax.set_axisbelow(True)

def plot_sweep_cdf(
    all_series: Dict[int, Dict[str, List[float]]],
    rps_list: List[int],
    out_path: Path,
    metric: str = "latency",
) -> None:
    n = len(rps_list)
    # 3 panels × 1.1" = 3.3" total ≈ one IEEE/ACM column; height 1.6" keeps it compact
    fig, axes = plt.subplots(1, n, figsize=(1.1 * n, 1.3), sharey=True, sharex=False)
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.10)

    for ax, rps in zip(axes, rps_list):
        if metric == "latency":
            _plot_cdf_on_ax(ax, all_series.get(rps, {}), metric="latency")
            xlabel = "Latency (ms)"
        else:
            # throughput: all_series already contains per-second counts
            _plot_throughput_cdf_on_ax(ax, all_series.get(rps, {}))
            xlabel = "Throughput (req/s)"
        ax.set_title(f"{rps} req/s", pad=2)
        ax.set_xlabel(xlabel)
        if ax is not axes[0]:
            ax.tick_params(axis="y", left=False)

    axes[0].set_ylabel("CDF")

    fig.legend(
        handles=_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=len(SERIES_ORDER),   # single row
        frameon=False,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    fig.tight_layout(rect=(0, 0, 1, 1.0))
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir",         default=os.environ.get("EXP_DIR", "experiments/sharing_benefit/results"))
    parser.add_argument("--rps-sweep",       default="20,40,60",
                        help="Comma-separated RPS values to plot (must match run.sh sweep)")
    parser.add_argument("--warmup-secs",     type=float, default=10.0)
    parser.add_argument("--warmup-requests", type=int,   default=180,
                        help="Fallback warmup drop when no elapsed_sec column")
    args = parser.parse_args()

    result_root = (SERVING_DIR / args.exp_dir).resolve()
    if not result_root.exists():
        print(f"[Error] Results directory not found: {result_root}")
        return 1

    apply_paper_style()

    rps_list = [int(r.strip()) for r in args.rps_sweep.split(",")]

    out_dir = Path(f"{result_root}/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_series:     Dict[int, Dict[str, List[float]]] = {}  # latency
    all_throughput: Dict[int, Dict[str, List[float]]] = {}  # per-second counts
    all_task_results: Dict[int, Dict] = {}

    for rps in rps_list:
        rps_root = result_root / f"rps_{rps}"
        if not rps_root.exists():
            print(f"[Warn] {rps_root} not found, skipping rps={rps}")
            continue
        s    = load_series_latencies(rps_root,   args.warmup_secs, args.warmup_requests)
        tput = load_series_throughput(rps_root,  args.warmup_secs, args.warmup_requests)
        tres = load_task_results(rps_root)
        if s:
            all_series[rps] = s
            plot_latency_cdf(s, out_dir / f"motivation2_latency_cdf_rps{rps}.pdf")
        if tput:
            all_throughput[rps] = tput
            plot_throughput_cdf(tput, out_dir / f"motivation2_throughput_cdf_rps{rps}.pdf")
        if tres:
            all_task_results[rps] = tres
            plot_summary_bars(tres, out_dir / f"motivation2_summary_bars_rps{rps}.pdf")

    if not all_series and not all_task_results:
        print("[Error] No result data found. Run run.sh first.")
        return 1

    rps_with_data = [r for r in rps_list if r in all_series]
    if len(rps_with_data) > 1:
        plot_sweep_cdf(all_series,     rps_with_data, out_dir / "motivation2_sweep_latency_cdf.pdf")
        plot_sweep_cdf(all_throughput, rps_with_data, out_dir / "motivation2_sweep_throughput_cdf.pdf",
                       metric="throughput")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
