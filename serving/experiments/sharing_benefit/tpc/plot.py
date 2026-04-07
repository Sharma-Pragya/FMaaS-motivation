#!/usr/bin/env python3
"""sharing_benefit/tpc/plot.py — Plots for the sharing-benefit + TPC experiment.

Produces per-RPS:
  1. tpc_sharing_latency_cdf_rps{N}.pdf      — CDF of per-request latency
  2. tpc_sharing_throughput_cdf_rps{N}.pdf   — CDF of per-second throughput
  3. tpc_sharing_summary_bars_rps{N}.pdf     — bar chart: p99 latency per condition
  4. tpc_sharing_mean_service_time_rps{N}.pdf — bar chart: mean server-only service time per condition
  5. tpc_sharing_mean_batch_size_rps{N}.pdf  — bar chart: mean observed batch size per condition

And if multiple RPS levels:
  6. tpc_sharing_sweep_latency_cdf.pdf       — multi-panel latency CDF
  7. tpc_sharing_sweep_throughput_cdf.pdf    — multi-panel throughput CDF

Usage:
    python experiments/sharing_benefit/tpc/plot.py [--exp-dir experiments/sharing_benefit/tpc/results]
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

SERVING_DIR = Path(__file__).resolve().parents[3]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

SERIES_ORDER  = ["single", "no_sharing_tpc", "no_sharing", "sharing"]
SERIES_COLORS = {
    "single":         "#A9C7B5",   # sage green
    "no_sharing_tpc": "#6B9AC4",   # muted blue
    "no_sharing":     "#888888",   # mid gray
    "sharing":        "#E06C75",   # pink-red
}
SERIES_LABELS = {
    "single":         "Single Task",
    "no_sharing_tpc": "No Sharing (TPC)",
    "no_sharing":     "No Sharing (2 Servers)",
    "sharing":        "Sharing (STFQ)",
}
SERIES_LINESTYLE = {
    "single":         ":",
    "no_sharing_tpc": "--",
    "no_sharing":     "-.",
    "sharing":        "-",
}

CONDITION_ORDER = ["single_ecgclass", "single_gestureclass", "no_sharing_tpc", "no_sharing", "sharing"]


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
    """Returns {series: [latency_ms]} where single_ecgclass + single_gestureclass
    are pooled into one 'single' series."""
    raw: Dict[str, List[float]] = {}
    for cond in CONDITION_ORDER:
        lats = _read_condition_latencies(result_root, cond, warmup_secs, warmup_requests)
        if lats:
            raw[cond] = lats

    series: Dict[str, List[float]] = {}

    # Pool single-task conditions
    single_lats = []
    for cond in ("single_ecgclass", "single_gestureclass"):
        single_lats.extend(raw.get(cond, []))
    if single_lats:
        series["single"] = single_lats

    for cond in ("no_sharing_tpc", "no_sharing", "sharing"):
        if cond in raw:
            series[cond] = raw[cond]

    return series


def load_task_results(result_root: Path) -> Dict[str, List[Dict]]:
    data: Dict[str, List[Dict]] = {}
    for cond in CONDITION_ORDER:
        path = result_root / cond / "task_results.csv"
        if not path.exists():
            continue
        with path.open() as f:
            data[cond] = list(csv.DictReader(f))
    return data


def load_series_throughput(
    result_root: Path,
    warmup_secs: float = 10.0,
) -> Dict[str, List[float]]:
    """Returns {series: [per_second_counts]} — completions binned into 1s windows."""
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

    # Pool single task conditions
    single_completions = (
        _completion_times(result_root, "single_ecgclass") +
        _completion_times(result_root, "single_gestureclass")
    )
    if single_completions:
        counts = _bin_throughput(single_completions)
        series["single"] = [c / 2.0 for c in counts]  # per-task average

    for cond in ("no_sharing_tpc", "no_sharing", "sharing"):
        completions = _completion_times(result_root, cond)
        if completions:
            counts = _bin_throughput(completions)
            series[cond] = [c / 2.0 for c in counts]  # per-task average (2 tasks)
    return series


def _observed_batch_stats(rows: List[Dict]) -> Dict[str, float]:
    grouped: Dict[str, int] = defaultdict(int)
    for row in rows:
        key = row.get("server_start_ns", "")
        if key:
            grouped[key] += 1
    batch_sizes = np.array(list(grouped.values()), dtype=float) if grouped else np.array([], dtype=float)
    if len(batch_sizes) == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(batch_sizes.mean()),
        "p95": float(np.percentile(batch_sizes, 95)),
        "max": float(batch_sizes.max()),
    }


def load_batch_sizes(
    result_root: Path,
    warmup_secs: float = 10.0,
) -> Dict[str, float]:
    raw: Dict[str, float] = {}
    for cond in CONDITION_ORDER:
        lat_file = result_root / cond / "latencies.csv"
        if not lat_file.exists():
            continue
        with lat_file.open() as f:
            reader = csv.DictReader(f)
            if "server_start_ns" not in (reader.fieldnames or []):
                continue
            rows = [r for r in reader if float(r.get("elapsed_sec", 0)) > warmup_secs]
        if rows:
            raw[cond] = _observed_batch_stats(rows)["mean"]

    series: Dict[str, float] = {}
    single_vals = [raw[c] for c in ("single_ecgclass", "single_gestureclass") if c in raw]
    if single_vals:
        series["single"] = float(np.mean(single_vals))
    for cond in ("no_sharing_tpc", "no_sharing", "sharing"):
        if cond in raw:
            series[cond] = raw[cond]
    return series


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _nice_upper(val: float) -> float:
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
    if metric == "latency":
        x_max = 100.0
    ax.set_ylim(0, 1.05)
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


def _plot_throughput_cdf_on_ax(ax: plt.Axes, series: Dict[str, List[float]]) -> None:
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


def _present_series(series: Dict[str, object]) -> List[str]:
    return [s for s in SERIES_ORDER if s in series and series[s]]


def _legend_handles(series_keys: List[str] | None = None) -> List:
    keys = series_keys if series_keys is not None else SERIES_ORDER
    return [
        plt.Line2D([0], [0],
                   color=SERIES_COLORS[s],
                   linestyle=SERIES_LINESTYLE[s],
                   linewidth=1.0,
                   label=SERIES_LABELS[s])
        for s in keys
    ]


# ---------------------------------------------------------------------------
# Plot 1: Latency CDF (single RPS)
# ---------------------------------------------------------------------------

def plot_latency_cdf(series: Dict[str, List[float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 1.3))
    _plot_cdf_on_ax(ax, series, metric="latency")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.legend(handles=_legend_handles(_present_series(series)), frameon=False, loc="lower right",
              ncol=1, handlelength=1.5, handletextpad=0.3)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Throughput CDF
# ---------------------------------------------------------------------------

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
    ax.legend(handles=_legend_handles(_present_series(throughput)), frameon=False, loc="lower right",
              ncol=1, handlelength=1.5, handletextpad=0.3)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: P99 latency bars
# ---------------------------------------------------------------------------

def plot_summary_bars(task_results: Dict[str, List[Dict]], out_path: Path) -> None:
    p99: Dict[str, float] = {}

    # Pool single tasks
    single_p99s = []
    for cond in ("single_ecgclass", "single_gestureclass"):
        rows = task_results.get(cond, [])
        single_p99s.extend(float(r["p99_latency_ms"]) for r in rows if "p99_latency_ms" in r)
    if single_p99s:
        p99["single"] = float(np.mean(single_p99s))

    for cond in ("no_sharing_tpc", "no_sharing", "sharing"):
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
# Plot 4: Mean service time bars
# ---------------------------------------------------------------------------

def plot_mean_service_time_bars(task_results: Dict[str, List[Dict]], out_path: Path) -> None:
    mean_service_time: Dict[str, float] = {}

    single_means = []
    for cond in ("single_ecgclass", "single_gestureclass"):
        rows = task_results.get(cond, [])
        single_means.extend(float(r["avg_server_exec_ms"]) for r in rows if "avg_server_exec_ms" in r)
    if single_means:
        mean_service_time["single"] = float(np.mean(single_means))

    for cond in ("no_sharing_tpc", "no_sharing", "sharing"):
        rows = task_results.get(cond, [])
        vals = [float(r["avg_server_exec_ms"]) for r in rows if "avg_server_exec_ms" in r]
        if vals:
            mean_service_time[cond] = float(np.mean(vals))

    series = [s for s in SERIES_ORDER if s in mean_service_time]
    if not series:
        print("[Warn] No server-only service time data, skipping mean service time chart")
        return

    fig, ax = plt.subplots(figsize=(2.5, 1.3))
    x = np.arange(len(series))
    bars = ax.bar(x, [mean_service_time[s] for s in series],
                  width=0.5,
                  color=[SERIES_COLORS[s] for s in series],
                  edgecolor="black", linewidth=0.4, zorder=2)
    for bar, s in zip(bars, series):
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                f"{v:.1f}", ha="center", va="bottom", fontsize=4.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SERIES_LABELS[s] for s in series], rotation=15, ha="right")
    ax.set_ylabel("Mean Server Exec Time (ms)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=_legend_handles(series), frameon=False, loc="upper right",
              ncol=1, handlelength=1.5, handletextpad=0.3)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: Mean batch size bars
# ---------------------------------------------------------------------------

def plot_mean_batch_size_bars(batch_sizes: Dict[str, float], out_path: Path) -> None:
    series = [s for s in SERIES_ORDER if s in batch_sizes]
    if not series:
        print("[Warn] No batch size data, skipping mean batch size chart")
        return

    fig, ax = plt.subplots(figsize=(2.5, 1.3))
    x = np.arange(len(series))
    bars = ax.bar(x, [batch_sizes[s] for s in series],
                  width=0.5,
                  color=[SERIES_COLORS[s] for s in series],
                  edgecolor="black", linewidth=0.4, zorder=2)
    for bar, s in zip(bars, series):
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=4.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SERIES_LABELS[s] for s in series], rotation=15, ha="right")
    ax.set_ylabel("Mean Batch Size")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6: Sweep CDF — multi-panel
# ---------------------------------------------------------------------------

def plot_sweep_cdf(
    all_series: Dict[int, Dict[str, List[float]]],
    rps_list: List[int],
    out_path: Path,
    metric: str = "latency",
) -> None:
    n = len(rps_list)
    fig, axes = plt.subplots(1, n, figsize=(1.1 * n, 1.3), sharey=True, sharex=False)
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.10)

    for ax, rps in zip(axes, rps_list):
        if metric == "latency":
            _plot_cdf_on_ax(ax, all_series.get(rps, {}), metric="latency")
            xlabel = "Latency (ms)"
        else:
            _plot_throughput_cdf_on_ax(ax, all_series.get(rps, {}))
            xlabel = "Throughput (req/s)"
        ax.set_title(f"{rps} req/s", pad=2)
        ax.set_xlabel(xlabel)
        if ax is not axes[0]:
            ax.tick_params(axis="y", left=False)

    axes[0].set_ylabel("CDF")

    legend_series = [s for s in SERIES_ORDER if any(all_series.get(rps, {}).get(s) for rps in rps_list)]

    fig.legend(
        handles=_legend_handles(legend_series),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=max(1, len(legend_series)),
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
    parser.add_argument("--exp-dir",         default=os.environ.get("EXP_DIR", "experiments/sharing_benefit/tpc/results"))
    parser.add_argument("--rps-sweep",       default="20",
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

    all_series:     Dict[int, Dict[str, List[float]]] = {}
    all_throughput: Dict[int, Dict[str, List[float]]] = {}
    all_task_results: Dict[int, Dict] = {}
    all_batch_sizes: Dict[int, Dict[str, float]] = {}

    for rps in rps_list:
        rps_root = result_root / f"rps_{rps}"
        if not rps_root.exists():
            print(f"[Warn] {rps_root} not found, skipping rps={rps}")
            continue
        s    = load_series_latencies(rps_root, args.warmup_secs, args.warmup_requests)
        tput = load_series_throughput(rps_root, args.warmup_secs)
        tres = load_task_results(rps_root)
        bs   = load_batch_sizes(rps_root, args.warmup_secs)
        if s:
            all_series[rps] = s
            plot_latency_cdf(s, out_dir / f"tpc_sharing_latency_cdf_rps{rps}.pdf")
        if tput:
            all_throughput[rps] = tput
            plot_throughput_cdf(tput, out_dir / f"tpc_sharing_throughput_cdf_rps{rps}.pdf")
        if tres:
            all_task_results[rps] = tres
            plot_summary_bars(tres, out_dir / f"tpc_sharing_summary_bars_rps{rps}.pdf")
            plot_mean_service_time_bars(tres, out_dir / f"tpc_sharing_mean_service_time_rps{rps}.pdf")
        if bs:
            all_batch_sizes[rps] = bs
            plot_mean_batch_size_bars(bs, out_dir / f"tpc_sharing_mean_batch_size_rps{rps}.pdf")

    if not all_series and not all_task_results:
        print("[Error] No result data found. Run run.sh first.")
        return 1

    rps_with_data = [r for r in rps_list if r in all_series]
    if len(rps_with_data) > 1:
        plot_sweep_cdf(all_series,     rps_with_data, out_dir / "tpc_sharing_sweep_latency_cdf.pdf")
        plot_sweep_cdf(all_throughput, rps_with_data, out_dir / "tpc_sharing_sweep_throughput_cdf.pdf",
                       metric="throughput")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
