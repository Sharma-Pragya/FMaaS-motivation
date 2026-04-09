#!/usr/bin/env python3
"""sharing_benefit/tpc/plot.py — Plots for the sharing-benefit + TPC experiment.

Handles two directory structures automatically:

  OLD (RPS sweep only):
    <exp-dir>/rps_{rps}/{condition}/latencies.csv

  NEW (num-tasks × RPS sweep):
    <exp-dir>/ntasks_{n}/rps_{rps}/{condition}/latencies.csv

Per (ntasks, rps) produces:
  tpc_sharing_latency_cdf_rps{N}.pdf
  tpc_sharing_throughput_cdf_rps{N}.pdf
  tpc_sharing_summary_bars_rps{N}.pdf
  tpc_sharing_mean_service_time_rps{N}.pdf
  tpc_sharing_mean_batch_size_rps{N}.pdf
  tpc_per_task_latency_rps{N}.pdf
  tpc_per_task_service_time_rps{N}.pdf

Sweep plots (when multiple RPS values):
  tpc_sharing_sweep_latency_cdf.pdf
  tpc_sharing_sweep_throughput_cdf.pdf

NEW — ntasks sweep (when multiple ntasks values):
  tpc_ntasks_mean_latency.pdf
    x-axis: number of tasks, y-axis: mean response time
    one line per condition; one subplot per RPS if multiple RPS values

Usage:
    python experiments/sharing_benefit/tpc/plot.py \\
        [--exp-dir experiments/sharing_benefit/tpc/results_momentbase] \\
        [--rps-sweep 25] \\
        [--num-tasks-sweep 2,4,6,8,10]
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

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

SERIES_ORDER  = ["single", "no_sharing_tpc", "no_sharing_mps", "no_sharing", "sharing"]
SERIES_COLORS = {
    "single":         "#A9C7B5",   # sage green
    "no_sharing_tpc": "#6B9AC4",   # muted blue
    "no_sharing_mps": "#F0A500",   # amber/orange
    "no_sharing":     "#888888",   # mid gray
    "sharing":        "#E06C75",   # pink-red
}
SERIES_LABELS = {
    "single":         "Single Task",
    "no_sharing_tpc": "No Sharing (TPC)",
    "no_sharing_mps": "No Sharing (MPS)",
    "no_sharing":     "No Sharing (N Servers)",
    "sharing":        "FMVisor",
}
SERIES_LINESTYLE = {
    "single":         ":",
    "no_sharing_tpc": "--",
    "no_sharing_mps": (0, (3, 1, 1, 1)),  # dash-dot-dot
    "no_sharing":     "-.",
    "sharing":        "-",
}
SERIES_MARKER = {
    "single":         "s",
    "no_sharing_tpc": "^",
    "no_sharing_mps": "P",   # plus-filled
    "no_sharing":     "D",
    "sharing":        "o",
}

SINGLE_CONDITIONS_BY_TASK_SET = {
    "tsfm":   ["single_ecgclass", "single_gestureclass"],
    "vision": ["single_nyudepth", "single_vocseg"],
}

# Set in main() — used only by legacy per-dir loaders
SINGLE_CONDITIONS: List[str] = SINGLE_CONDITIONS_BY_TASK_SET["tsfm"]
CONDITION_ORDER:   List[str] = SINGLE_CONDITIONS + ["no_sharing_tpc", "no_sharing_mps", "no_sharing", "sharing"]


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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[Plot] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Directory structure helpers
# ---------------------------------------------------------------------------

def detect_single_conditions(rps_root: Path) -> List[str]:
    """Return sorted list of single_* conditions available for rps_root.

    Checks three places in order:
      1. Real dirs or symlinks directly under rps_root (ntasks_N/rps_R/single_*)
      2. Shared singles/ dir two levels up  (…/singles/rps_R/single_*)
      3. Shared singles/ dir one level up   (…/singles/rps_R/single_*)  [flat layout]
    """
    found: set = set()

    # 1. Direct children (follows symlinks via Path.is_symlink check)
    if rps_root.exists():
        for d in rps_root.iterdir():
            if d.name.startswith("single_") and (d.is_dir() or d.is_symlink()):
                found.add(d.name)

    # 2 & 3. singles/ fallback directories
    rps_name = rps_root.name
    for singles_base in (
        rps_root.parent.parent / "singles" / rps_name,  # ntasks layout
        rps_root.parent / "singles" / rps_name,          # flat layout
    ):
        if singles_base.exists():
            for d in singles_base.iterdir():
                if d.name.startswith("single_") and d.is_dir():
                    found.add(d.name)

    return sorted(found)


def detect_ntasks_list(result_root: Path) -> List[int]:
    """Return sorted list of ntasks values from ntasks_* subdirs."""
    vals = []
    for d in result_root.iterdir():
        if d.is_dir() and d.name.startswith("ntasks_"):
            try:
                vals.append(int(d.name.split("_", 1)[1]))
            except ValueError:
                pass
    return sorted(vals)


def detect_rps_list(parent: Path) -> List[int]:
    """Return sorted list of rps values from rps_* subdirs under parent."""
    vals = []
    for d in parent.iterdir():
        if d.is_dir() and d.name.startswith("rps_"):
            try:
                vals.append(int(d.name.split("_", 1)[1]))
            except ValueError:
                pass
    return sorted(vals)


# ---------------------------------------------------------------------------
# Data loading (existing per-dir loaders — use SINGLE_CONDITIONS / CONDITION_ORDER globals)
# ---------------------------------------------------------------------------

def _singles_fallback(result_root: Path, cond: str) -> Path:
    """Return the latencies.csv path in the shared singles/ dir.
    result_root may be …/ntasks_{n}/rps_{rps}/ or …/rps_{rps}/."""
    rps_name = result_root.name  # e.g. "rps_5"
    # ntasks layout: go up two levels (ntasks_N → RESULTS_BASE)
    candidate = result_root.parent.parent / "singles" / rps_name / cond / "latencies.csv"
    if candidate.exists():
        return candidate
    # flat layout: go up one level (RESULTS_BASE)
    return result_root.parent / "singles" / rps_name / cond / "latencies.csv"


def _read_condition_latencies(
    result_root: Path,
    cond: str,
    warmup_secs: float,
    warmup_requests: int,
) -> List[float]:
    lat_file = result_root / cond / "latencies.csv"
    if not lat_file.exists() and cond.startswith("single_"):
        lat_file = _singles_fallback(result_root, cond)
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
    single_conds = detect_single_conditions(result_root) or SINGLE_CONDITIONS
    cond_order   = single_conds + ["no_sharing_tpc", "no_sharing_mps", "no_sharing", "sharing"]
    raw: Dict[str, List[float]] = {}
    for cond in cond_order:
        lats = _read_condition_latencies(result_root, cond, warmup_secs, warmup_requests)
        if lats:
            raw[cond] = lats
    series: Dict[str, List[float]] = {}
    single_lats = []
    for cond in single_conds:
        single_lats.extend(raw.get(cond, []))
    if single_lats:
        series["single"] = single_lats
    for cond in ("no_sharing_tpc", "no_sharing_mps", "no_sharing", "sharing"):
        if cond in raw:
            series[cond] = raw[cond]
    return series


def _resolve_cond_file(result_root: Path, cond: str, filename: str) -> Path:
    """Return the path to filename under cond, falling back to singles/ for single_* conds."""
    p = result_root / cond / filename
    if not p.exists() and cond.startswith("single_"):
        fallback = _singles_fallback(result_root, cond).parent / filename
        if fallback.exists():
            return fallback
    return p


def load_task_results(result_root: Path) -> Dict[str, List[Dict]]:
    single_conds = detect_single_conditions(result_root) or SINGLE_CONDITIONS
    cond_order   = single_conds + ["no_sharing_tpc", "no_sharing_mps", "no_sharing", "sharing"]
    data: Dict[str, List[Dict]] = {}
    for cond in cond_order:
        path = _resolve_cond_file(result_root, cond, "task_results.csv")
        if not path.exists():
            continue
        with path.open() as f:
            data[cond] = list(csv.DictReader(f))
    return data


def load_series_throughput(
    result_root: Path,
    warmup_secs: float = 10.0,
) -> Dict[str, List[float]]:
    single_conds = detect_single_conditions(result_root) or SINGLE_CONDITIONS
    n_tasks      = max(len(single_conds), 1)

    def _completion_times(cond: str) -> List[float]:
        lat_file = _resolve_cond_file(result_root, cond, "latencies.csv")
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
                completions.append(elapsed + float(row["latency_ms"]) / 1000.0)
        return completions

    def _bin_throughput(completions: List[float]) -> List[float]:
        if not completions:
            return []
        t_min, t_max = min(completions), max(completions)
        bins = np.arange(t_min, t_max + 1, 1.0)
        counts, _ = np.histogram(completions, bins=bins)
        return counts.tolist()

    series: Dict[str, List[float]] = {}
    single_completions = []
    for cond in single_conds:
        single_completions += _completion_times(cond)
    if single_completions:
        counts = _bin_throughput(single_completions)
        series["single"] = [c / n_tasks for c in counts]

    for cond in ("no_sharing_tpc", "no_sharing_mps", "no_sharing", "sharing"):
        completions = _completion_times(cond)
        if completions:
            counts = _bin_throughput(completions)
            series[cond] = [c / n_tasks for c in counts]
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
        "p95":  float(np.percentile(batch_sizes, 95)),
        "max":  float(batch_sizes.max()),
    }


def load_batch_sizes(
    result_root: Path,
    warmup_secs: float = 10.0,
) -> Dict[str, float]:
    single_conds = detect_single_conditions(result_root) or SINGLE_CONDITIONS
    cond_order   = single_conds + ["no_sharing_tpc", "no_sharing_mps", "no_sharing", "sharing"]
    raw: Dict[str, float] = {}
    for cond in cond_order:
        lat_file = _resolve_cond_file(result_root, cond, "latencies.csv")
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
    single_vals = [raw[c] for c in single_conds if c in raw]
    if single_vals:
        series["single"] = float(np.mean(single_vals))
    for cond in ("no_sharing_tpc", "no_sharing_mps", "no_sharing", "sharing"):
        if cond in raw:
            series[cond] = raw[cond]
    return series


# ---------------------------------------------------------------------------
# New loader: mean latency per (ntasks, rps) for the ntasks sweep plot
# ---------------------------------------------------------------------------

def load_p99_for_dir(rps_root: Path) -> Dict[str, float]:
    """Read task_results.csv files and return {series: mean_p99_latency_ms}.
    Pools single_* conditions into 'single'; other series are direct."""
    result: Dict[str, float] = {}

    single_conds = detect_single_conditions(rps_root) or SINGLE_CONDITIONS
    single_p99s: List[float] = []
    for cond in single_conds:
        path = _resolve_cond_file(rps_root, cond, "task_results.csv")
        if not path.exists():
            continue
        with path.open() as f:
            for row in csv.DictReader(f):
                if "p99_latency_ms" in row:
                    single_p99s.append(float(row["p99_latency_ms"]))
    if single_p99s:
        result["single"] = float(np.mean(single_p99s))

    for cond in ("no_sharing_tpc", "no_sharing_mps", "no_sharing", "sharing"):
        path = rps_root / cond / "task_results.csv"
        if not path.exists():
            continue
        vals: List[float] = []
        with path.open() as f:
            for row in csv.DictReader(f):
                if "p99_latency_ms" in row:
                    vals.append(float(row["p99_latency_ms"]))
        if vals:
            result[cond] = float(np.mean(vals))

    return result


def load_mean_latency_for_dir(rps_root: Path) -> Dict[str, float]:
    """Read task_results.csv files in rps_root and return {series: mean_avg_latency_ms}.
    Pools all single_* conditions into 'single'; other series are direct."""
    result: Dict[str, float] = {}

    # Pool single conditions (auto-discovered from symlinks/dirs + singles/ fallback)
    single_conds = detect_single_conditions(rps_root) or SINGLE_CONDITIONS
    single_lats: List[float] = []
    for cond in single_conds:
        path = _resolve_cond_file(rps_root, cond, "task_results.csv")
        if not path.exists():
            continue
        with path.open() as f:
            for row in csv.DictReader(f):
                if "avg_latency_ms" in row:
                    single_lats.append(float(row["avg_latency_ms"]))
    if single_lats:
        result["single"] = float(np.mean(single_lats))

    for cond in ("no_sharing_tpc", "no_sharing_mps", "no_sharing", "sharing"):
        path = rps_root / cond / "task_results.csv"
        if not path.exists():
            continue
        vals: List[float] = []
        with path.open() as f:
            for row in csv.DictReader(f):
                if "avg_latency_ms" in row:
                    vals.append(float(row["avg_latency_ms"]))
        if vals:
            result[cond] = float(np.mean(vals))

    return result


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _nice_upper(val: float) -> float:
    if val <= 0:
        return 10.0
    return float(np.ceil(val / 10.0) * 10)


def _set_nice_ylim(ax: plt.Axes, headroom: float = 1.25) -> None:
    """Set y upper limit to a round number, with tick landing exactly on it."""
    current_max = ax.get_ylim()[1]
    top = _nice_upper(current_max * headroom)
    ax.set_ylim(0, top)
    # Place ticks at even intervals ending exactly on top
    step = _nice_upper(top / 5)
    if step <= 0:
        step = 10.0
    ax.yaxis.set_major_locator(ticker.MultipleLocator(step))


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
    for n_ticks in (3, 4, 5, 6):
        step = x_max / (n_ticks - 1)
        if step == int(step):
            break

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    if metric == "latency":
        ax.set_xlim(0,100)
        xticks = np.linspace(0, 100, n_ticks)
        ax.set_xticks(xticks)
    else:
        ax.set_xlim(0, x_max)
        xticks = np.linspace(0, x_max, n_ticks)
        ax.set_xticks(xticks)
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


def _legend_handles(series_keys: Optional[List[str]] = None) -> List:
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
    single_conds = [k for k in task_results if k.startswith("single_")]
    p99: Dict[str, float] = {}
    single_p99s = []
    for cond in single_conds:
        rows = task_results.get(cond, [])
        single_p99s.extend(float(r["p99_latency_ms"]) for r in rows if "p99_latency_ms" in r)
    if single_p99s:
        p99["single"] = float(np.mean(single_p99s))
    for cond in ("no_sharing_tpc", "no_sharing_mps", "no_sharing", "sharing"):
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
                  width=1.0,
                  color=[SERIES_COLORS[s] for s in series],
                  edgecolor="black", linewidth=0.4, zorder=2)
    for bar, s in zip(bars, series):
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                f"{v:.0f}", ha="center", va="bottom", fontsize=4.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SERIES_LABELS[s] for s in series], rotation=15, ha="right")
    ax.set_ylabel("P99 Latency (ms)")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    _set_nice_ylim(ax)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Mean service time bars
# ---------------------------------------------------------------------------

def plot_mean_service_time_bars(task_results: Dict[str, List[Dict]], out_path: Path) -> None:
    single_conds = [k for k in task_results if k.startswith("single_")]
    mean_service_time: Dict[str, float] = {}
    single_means = []
    for cond in single_conds:
        rows = task_results.get(cond, [])
        single_means.extend(float(r["avg_server_exec_ms"]) for r in rows if "avg_server_exec_ms" in r)
    if single_means:
        mean_service_time["single"] = float(np.mean(single_means))
    for cond in ("no_sharing_tpc", "no_sharing_mps", "no_sharing", "sharing"):
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
                  width=1.0,
                  color=[SERIES_COLORS[s] for s in series],
                  edgecolor="black", linewidth=0.4, zorder=2)
    for bar, s in zip(bars, series):
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                f"{v:.1f}", ha="center", va="bottom", fontsize=4.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SERIES_LABELS[s] for s in series], rotation=15, ha="right")
    ax.set_ylabel("Mean Server Exec Time (ms)")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    _set_nice_ylim(ax)
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
                  width=1.0,
                  color=[SERIES_COLORS[s] for s in series],
                  edgecolor="black", linewidth=0.4, zorder=2)
    for bar, s in zip(bars, series):
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=4.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SERIES_LABELS[s] for s in series], rotation=15, ha="right")
    ax.set_ylabel("Mean Batch Size")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    _set_nice_ylim(ax)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6: Sweep CDF — multi-panel (RPS sweep)
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
# Plot 7: ntasks sweep — mean response time vs number of tasks
#   data: {rps: {ntasks: {series: mean_latency_ms}}}
#   One subplot per RPS value; one line per condition.
# ---------------------------------------------------------------------------

def plot_ntasks_mean_latency(
    data: Dict[int, Dict[int, Dict[str, float]]],
    rps_list: List[int],
    ntasks_list: List[int],
    out_path: Path,
) -> None:
    # Filter to RPS values that actually have data
    rps_with_data = [r for r in rps_list if any(data.get(r, {}).values())]
    if not rps_with_data:
        print("[Warn] No ntasks sweep data, skipping ntasks mean latency plot")
        return

    n_panels = len(rps_with_data)
    fig_w = max(2.8 * n_panels, 3.3)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 2.0),
                             sharey=False, squeeze=False)
    axes = axes[0]  # flatten to 1-D

    # Collect all series that appear across all panels
    present: List[str] = []
    for s in SERIES_ORDER:
        if any(
            s in data.get(rps, {}).get(n, {})
            for rps in rps_with_data
            for n in ntasks_list
        ):
            present.append(s)

    for ax, rps in zip(axes, rps_with_data):
        rps_data = data.get(rps, {})
        for s in present:
            xs, ys = [], []
            for n in ntasks_list:
                v = rps_data.get(n, {}).get(s)
                if v is not None:
                    xs.append(n)
                    ys.append(v)
            if xs:
                ax.plot(
                    xs, ys,
                    color=SERIES_COLORS[s],
                    linestyle=SERIES_LINESTYLE[s],
                    marker=SERIES_MARKER[s],
                    markersize=3.5,
                    linewidth=1.2,
                    label=SERIES_LABELS[s],
                )

        ax.set_title(f"RPS = {rps}", pad=3)
        ax.set_xlabel("Number of Tasks")
        ax.set_xticks(ntasks_list)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.grid(axis="both", zorder=0)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.set_ylabel("Mean Response Time (ms)")

    fig.legend(
        handles=_legend_handles(present),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=max(1, len(present)),
        frameon=False,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    fig.tight_layout(rect=(0, 0, 1, 1.0))
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: P99 latency vs RPS — grouped bar chart, one bar-group per RPS
# ---------------------------------------------------------------------------

def plot_p99_vs_rps(
    p99_data: Dict[int, Dict[str, float]],
    rps_list: List[int],
    out_path: Path,
    ntasks: Optional[int] = None,
) -> None:
    """Grouped bar chart: x=RPS, y=P99 latency (ms), grouped bars per condition."""
    rps_with_data = [r for r in rps_list if p99_data.get(r)]
    if not rps_with_data:
        print("[Warn] No P99 data, skipping P99 vs RPS plot")
        return

    present = [s for s in SERIES_ORDER
                if any(s in p99_data.get(r, {}) for r in rps_with_data)]
    if not present:
        return

    n_groups = len(rps_with_data)
    n_bars   = len(present)
    group_w  = 0.8                  # total width occupied by all bars in one group
    bar_w    = group_w / n_bars     # each bar gets an equal slice — bars touch
    fig_w    = max(1.8 * n_groups, 3.5)

    fig, ax = plt.subplots(figsize=(fig_w, 2.4))

    x = np.arange(n_groups)
    # Offsets so bars are centred on x with no gap between them
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_w

    for offset, series in zip(offsets, present):
        vals = [p99_data.get(r, {}).get(series, float("nan")) for r in rps_with_data]
        bars = ax.bar(
            x + offset, vals,
            width=bar_w,
            color=SERIES_COLORS[series],
            edgecolor="black", linewidth=0.4,
            label=SERIES_LABELS[series],
            zorder=2,
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v * 1.01,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=4.0, rotation=90)

    title = "P99 Latency vs Request Rate"
    if ntasks is not None:
        title += f"  (ntasks={ntasks})"
    ax.set_title(title, pad=3)
    ax.set_xlabel("Request Rate (RPS per task)")
    ax.set_ylabel("P99 Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rps_with_data])
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    _set_nice_ylim(ax)
    ax.legend(
        handles=_legend_handles(present),
        loc="upper left",
        fontsize=5,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.3,
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# New loader: per-task mean latency broken out by condition
# ---------------------------------------------------------------------------

def _active_tasks_for_dir(rps_root: Path) -> List[str]:
    """Determine the active task set for this rps_root by reading whichever
    multi-task condition exists first (sharing > no_sharing > no_sharing_tpc).
    Falls back to all single_* dirs if none found."""
    for cond in ("sharing", "no_sharing", "no_sharing_mps", "no_sharing_tpc"):
        path = rps_root / cond / "task_results.csv"
        if path.exists():
            tasks = []
            with path.open() as f:
                for row in csv.DictReader(f):
                    if "task" in row:
                        tasks.append(row["task"])
            if tasks:
                return tasks
    # Fallback: infer from single_* dirs present locally (not singles/)
    local = sorted(
        d.name[len("single_"):] for d in rps_root.iterdir()
        if d.name.startswith("single_") and (d.is_dir() or d.is_symlink())
    ) if rps_root.exists() else []
    return local


def _load_per_task_metric(rps_root: Path, column: str) -> Dict[str, Dict[str, float]]:
    """Generic loader: return {condition_series: {task: value}} for any task_results column."""
    result: Dict[str, Dict[str, float]] = {}
    active_tasks = _active_tasks_for_dir(rps_root)

    for task in active_tasks:
        cond = f"single_{task}"
        path = _resolve_cond_file(rps_root, cond, "task_results.csv")
        if not path.exists():
            continue
        with path.open() as f:
            for row in csv.DictReader(f):
                if column in row and "task" in row:
                    result.setdefault("single", {})[row["task"]] = float(row[column])

    for cond in ("no_sharing_tpc", "no_sharing_mps", "no_sharing", "sharing"):
        path = _resolve_cond_file(rps_root, cond, "task_results.csv")
        if not path.exists():
            continue
        with path.open() as f:
            for row in csv.DictReader(f):
                if column in row and "task" in row:
                    result.setdefault(cond, {})[row["task"]] = float(row[column])

    return result


def load_per_task_latencies(rps_root: Path) -> Dict[str, Dict[str, float]]:
    """Return {condition_series: {task: avg_latency_ms}}."""
    return _load_per_task_metric(rps_root, "avg_latency_ms")


def load_per_task_service_time(rps_root: Path) -> Dict[str, Dict[str, float]]:
    """Return {condition_series: {task: avg_server_exec_ms}} (server-side execution only)."""
    return _load_per_task_metric(rps_root, "avg_server_exec_ms")


# ---------------------------------------------------------------------------
# Plot: per-task mean response time — one subplot per condition
# ---------------------------------------------------------------------------

def plot_per_task_latency(
    per_task: Dict[str, Dict[str, float]],
    out_path: Path,
) -> None:
    """Bar chart: x=task, y=mean response time (ms).
    One subplot per condition (series) that has data."""
    conditions = [s for s in SERIES_ORDER if s in per_task and per_task[s]]
    if not conditions:
        print("[Warn] No per-task latency data, skipping per-task plot")
        return

    # Union of tasks across all conditions, preserving canonical order
    all_tasks_set: set = set()
    for d in per_task.values():
        all_tasks_set.update(d.keys())
    # Canonical tsfm order first, then any extras alphabetically
    canonical = [
        "ecgclass", "gestureclass", "heartrate", "diasbp", "sysbp",
        "etth1fore", "weatherfore", "trafficfore", "eclfore", "exchangefore",
        "nyudepth", "vocseg",
    ]
    tasks = [t for t in canonical if t in all_tasks_set] + \
            sorted(all_tasks_set - set(canonical))

    n_panels = len(conditions)
    fig_w = max(1.8 * n_panels, 3.5)
    # sharey=False: each condition can have very different latency ranges
    # (e.g. no_sharing_tpc may be 100x larger than sharing)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 2.2),
                             sharey=False, squeeze=False)
    axes = axes[0]

    x = np.arange(len(tasks))
    bar_w = 1.0  # bars touch

    for ax, cond in zip(axes, conditions):
        task_lats = per_task[cond]
        vals = [task_lats.get(t, float("nan")) for t in tasks]
        bars = ax.bar(x, vals, width=bar_w,
                      color=SERIES_COLORS[cond], edgecolor="black", linewidth=0.4, zorder=2)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=4.5)
        ax.set_title(SERIES_LABELS[cond], pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=5.5)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        _set_nice_ylim(ax)
        if ax is axes[0]:
            ax.set_ylabel("Mean Response Time (ms)")

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_per_task_service_time(
    per_task: Dict[str, Dict[str, float]],
    out_path: Path,
) -> None:
    """Bar chart: x=task, y=mean server execution time (ms).
    One subplot per condition — same layout as plot_per_task_latency."""
    conditions = [s for s in SERIES_ORDER if s in per_task and per_task[s]]
    if not conditions:
        print("[Warn] No per-task service time data, skipping per-task service time plot")
        return

    all_tasks_set: set = set()
    for d in per_task.values():
        all_tasks_set.update(d.keys())
    canonical = [
        "ecgclass", "gestureclass", "heartrate", "diasbp", "sysbp",
        "etth1fore", "weatherfore", "trafficfore", "eclfore", "exchangefore",
        "nyudepth", "vocseg",
    ]
    tasks = [t for t in canonical if t in all_tasks_set] + \
            sorted(all_tasks_set - set(canonical))

    n_panels = len(conditions)
    fig_w = max(1.8 * n_panels, 3.5)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 2.2),
                             sharey=False, squeeze=False)
    axes = axes[0]

    x = np.arange(len(tasks))
    bar_w = 1.0  # bars touch

    for ax, cond in zip(axes, conditions):
        task_vals = per_task[cond]
        vals = [task_vals.get(t, float("nan")) for t in tasks]
        bars = ax.bar(x, vals, width=bar_w,
                      color=SERIES_COLORS[cond], edgecolor="black", linewidth=0.4, zorder=2)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=4.5)
        ax.set_title(SERIES_LABELS[cond], pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=5.5)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        _set_nice_ylim(ax)
        if ax is axes[0]:
            ax.set_ylabel("Mean Service Time (ms)")

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-dir helper: run all per-(ntasks, rps) plots
# ---------------------------------------------------------------------------

def _run_per_dir_plots(
    rps_root: Path,
    out_dir: Path,
    rps: int,
    warmup_secs: float,
    warmup_requests: int,
) -> tuple:
    """Run per-(ntasks, rps) plots; return (series, throughput) for sweep plots."""
    s    = load_series_latencies(rps_root, warmup_secs, warmup_requests)
    tput = load_series_throughput(rps_root, warmup_secs)
    tres = load_task_results(rps_root)
    bs   = load_batch_sizes(rps_root, warmup_secs)
    if s:
        plot_latency_cdf(s, out_dir / f"tpc_sharing_latency_cdf_rps{rps}.pdf")
    if tput:
        plot_throughput_cdf(tput, out_dir / f"tpc_sharing_throughput_cdf_rps{rps}.pdf")
    if tres:
        plot_summary_bars(tres, out_dir / f"tpc_sharing_summary_bars_rps{rps}.pdf")
        plot_mean_service_time_bars(tres, out_dir / f"tpc_sharing_mean_service_time_rps{rps}.pdf")
    if bs:
        plot_mean_batch_size_bars(bs, out_dir / f"tpc_sharing_mean_batch_size_rps{rps}.pdf")
    per_task = load_per_task_latencies(rps_root)
    if per_task:
        plot_per_task_latency(per_task, out_dir / f"tpc_per_task_latency_rps{rps}.pdf")
    per_task_st = load_per_task_service_time(rps_root)
    if per_task_st:
        plot_per_task_service_time(per_task_st, out_dir / f"tpc_per_task_service_time_rps{rps}.pdf")
    return s, tput


# ---------------------------------------------------------------------------
# TPC count sweep loader + plot
#   tpc_sweep/rps_{rps}/tpc{N}_{task}/task_results.csv
#   x-axis: n_tpcs, y-axis: mean response time, one line per task
#   one subplot per RPS if multiple RPS values
# ---------------------------------------------------------------------------

def load_tpc_sweep_data(result_root: Path) -> Dict[int, Dict[int, Dict[str, float]]]:
    """Return {rps: {n_tpcs: {task: avg_latency_ms}}}.

    Scans result_root/tpc_sweep/rps_{rps}/tpc{N}_{task}/task_results.csv
    """
    tpc_sweep_root = result_root / "tpc_sweep"
    if not tpc_sweep_root.exists():
        return {}

    data: Dict[int, Dict[int, Dict[str, float]]] = {}
    for rps_dir in sorted(tpc_sweep_root.iterdir()):
        if not rps_dir.name.startswith("rps_"):
            continue
        try:
            rps = int(rps_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        for cond_dir in sorted(rps_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            # Expect name like tpc2_ecgclass
            parts = cond_dir.name.split("_", 1)
            if len(parts) != 2 or not parts[0].startswith("tpc"):
                continue
            try:
                n_tpcs = int(parts[0][3:])
            except ValueError:
                continue
            task = parts[1]
            path = cond_dir / "task_results.csv"
            if not path.exists():
                continue
            with path.open() as f:
                for row in csv.DictReader(f):
                    if "avg_latency_ms" in row:
                        data.setdefault(rps, {}).setdefault(n_tpcs, {})[task] = \
                            float(row["avg_latency_ms"])
    return data


def plot_tpc_count_sweep(
    data: Dict[int, Dict[int, Dict[str, float]]],
    out_path: Path,
) -> None:
    """Line plot: x=n_tpcs, y=mean response time, one line per task.
    One subplot per RPS value."""
    rps_list = sorted(data.keys())
    if not rps_list:
        print("[Warn] No tpc_sweep data, skipping tpc count sweep plot")
        return

    # Collect all tasks and all n_tpcs values
    all_tasks: set = set()
    all_n_tpcs: set = set()
    for rps_data in data.values():
        for n, task_lats in rps_data.items():
            all_n_tpcs.add(n)
            all_tasks.update(task_lats.keys())
    n_tpcs_list = sorted(all_n_tpcs)
    tasks = sorted(all_tasks)

    # Color each task distinctly
    cmap = plt.cm.get_cmap("tab10", len(tasks))
    task_colors = {t: cmap(i) for i, t in enumerate(tasks)}
    task_markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "p"]

    n_panels = len(rps_list)
    fig_w = max(2.8 * n_panels, 3.3)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 2.2),
                             sharey=False, squeeze=False)
    axes = axes[0]

    for ax, rps in zip(axes, rps_list):
        rps_data = data[rps]
        for i, task in enumerate(tasks):
            xs, ys = [], []
            for n in n_tpcs_list:
                v = rps_data.get(n, {}).get(task)
                if v is not None:
                    xs.append(n)
                    ys.append(v)
            if xs:
                ax.plot(xs, ys,
                        color=task_colors[task],
                        marker=task_markers[i % len(task_markers)],
                        markersize=3.5,
                        linewidth=1.2,
                        label=task)
        ax.set_title(f"RPS = {rps}", pad=3)
        ax.set_xlabel("Number of TPCs")
        ax.set_xticks(n_tpcs_list)
        ax.grid(axis="both", zorder=0)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.set_ylabel("Mean Response Time (ms)")

    # Shared legend
    handles = [
        plt.Line2D([0], [0], color=task_colors[t],
                   marker=task_markers[i % len(task_markers)],
                   markersize=3.5, linewidth=1.2, label=t)
        for i, t in enumerate(tasks)
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.12),
               ncol=max(1, len(tasks)),
               frameon=False, handlelength=1.5,
               columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 1.0))
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    global SINGLE_CONDITIONS, CONDITION_ORDER

    parser = argparse.ArgumentParser()
    parser.add_argument("--task-set",        default=os.environ.get("TASK_SET", "tsfm"),
                        choices=["tsfm", "vision"])
    parser.add_argument("--exp-dir",         default=os.environ.get("EXP_DIR",
                        "experiments/sharing_benefit/tpc/results_tsfm"))
    parser.add_argument("--rps-sweep",       default=None,
                        help="Comma-separated RPS values (auto-detected if omitted)")
    parser.add_argument("--num-tasks-sweep", default=None,
                        help="Comma-separated ntasks values (auto-detected if omitted; "
                             "ignored for old rps_* directory structure)")
    parser.add_argument("--warmup-secs",     type=float, default=10.0)
    parser.add_argument("--warmup-requests", type=int,   default=180,
                        help="Fallback warmup drop when no elapsed_sec column")
    args = parser.parse_args()

    SINGLE_CONDITIONS = SINGLE_CONDITIONS_BY_TASK_SET[args.task_set]
    CONDITION_ORDER   = SINGLE_CONDITIONS + ["no_sharing_tpc", "no_sharing_mps", "no_sharing", "sharing"]

    result_root = (SERVING_DIR / args.exp_dir).resolve()
    if not result_root.exists():
        print(f"[Error] Results directory not found: {result_root}")
        return 1

    apply_paper_style()

    out_dir = result_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Detect directory structure
    # -----------------------------------------------------------------------
    ntasks_list_auto = detect_ntasks_list(result_root)
    use_new_structure = len(ntasks_list_auto) > 0
    has_tpc_sweep = (result_root / "tpc_sweep").exists()

    if use_new_structure:
        # NEW: ntasks_*/rps_*/ layout
        ntasks_list = (
            [int(x.strip()) for x in args.num_tasks_sweep.split(",")]
            if args.num_tasks_sweep
            else ntasks_list_auto
        )
        # Collect rps from all ntasks dirs (union)
        rps_set: set = set()
        for n in ntasks_list:
            ntasks_dir = result_root / f"ntasks_{n}"
            if ntasks_dir.exists():
                rps_set.update(detect_rps_list(ntasks_dir))
        rps_list = (
            sorted(int(x.strip()) for x in args.rps_sweep.split(","))
            if args.rps_sweep
            else sorted(rps_set)
        )

        print(f"[Plot] New structure: ntasks={ntasks_list}  rps={rps_list}")

        # Per (ntasks, rps): standard CDF / bar plots
        all_series_by_rps:     Dict[int, Dict[str, List[float]]] = {}
        all_throughput_by_rps: Dict[int, Dict[str, List[float]]] = {}
        # ntasks sweep data: {rps: {ntasks: {series: mean_lat}}}
        ntasks_sweep_data: Dict[int, Dict[int, Dict[str, float]]] = {}
        # p99 data per ntasks: {ntasks: {rps: {series: p99_ms}}}
        p99_by_ntasks: Dict[int, Dict[int, Dict[str, float]]] = {}

        for n in ntasks_list:
            for rps in rps_list:
                rps_root = result_root / f"ntasks_{n}" / f"rps_{rps}"
                if not rps_root.exists():
                    print(f"[Warn] {rps_root} not found, skipping ntasks={n} rps={rps}")
                    continue

                sub_out = out_dir / f"ntasks_{n}"
                s, tput = _run_per_dir_plots(rps_root, sub_out, rps,
                                             args.warmup_secs, args.warmup_requests)

                # Accumulate for sweep plots (keyed by rps, across ntasks)
                if s:
                    all_series_by_rps.setdefault(rps, {})
                    # merge — extend series from this ntasks point
                    for k, v in s.items():
                        all_series_by_rps[rps].setdefault(k, []).extend(v)
                if tput:
                    all_throughput_by_rps.setdefault(rps, {})
                    for k, v in tput.items():
                        all_throughput_by_rps[rps].setdefault(k, []).extend(v)

                # ntasks sweep data
                mean_lats = load_mean_latency_for_dir(rps_root)
                if mean_lats:
                    ntasks_sweep_data.setdefault(rps, {})[n] = mean_lats

                # p99 data
                p99 = load_p99_for_dir(rps_root)
                if p99:
                    p99_by_ntasks.setdefault(n, {})[rps] = p99

        # RPS sweep CDFs (if multiple rps)
        rps_with_series = [r for r in rps_list if r in all_series_by_rps]
        if len(rps_with_series) > 1:
            plot_sweep_cdf(all_series_by_rps, rps_with_series,
                           out_dir / "tpc_sharing_sweep_latency_cdf.pdf")
            plot_sweep_cdf(all_throughput_by_rps, rps_with_series,
                           out_dir / "tpc_sharing_sweep_throughput_cdf.pdf",
                           metric="throughput")

        # NEW: ntasks sweep plot
        if len(ntasks_list) > 0:
            plot_ntasks_mean_latency(
                ntasks_sweep_data, rps_list, ntasks_list,
                out_dir / "tpc_ntasks_mean_latency.pdf",
            )

        # P99 vs RPS grouped bar chart — one plot per ntasks value
        for n, p99_data in p99_by_ntasks.items():
            plot_p99_vs_rps(
                p99_data, rps_list,
                out_dir / f"ntasks_{n}" / f"tpc_p99_vs_rps_ntasks{n}.pdf",
                ntasks=n,
            )

    else:
        # OLD: rps_*/ layout (no ntasks dimension)
        rps_list = (
            [int(r.strip()) for r in args.rps_sweep.split(",")]
            if args.rps_sweep
            else detect_rps_list(result_root)
        )
        if not rps_list:
            if has_tpc_sweep:
                rps_list = []  # nothing to plot here, tpc_sweep handled below
            else:
                print("[Error] No rps_* directories found and --rps-sweep not specified.")
                return 1

        print(f"[Plot] Old structure: rps={rps_list}")

        all_series:     Dict[int, Dict[str, List[float]]] = {}
        all_throughput: Dict[int, Dict[str, List[float]]] = {}

        old_p99_by_rps: Dict[int, Dict[str, float]] = {}
        for rps in rps_list:
            rps_root = result_root / f"rps_{rps}"
            if not rps_root.exists():
                print(f"[Warn] {rps_root} not found, skipping rps={rps}")
                continue
            s, tput = _run_per_dir_plots(rps_root, out_dir, rps,
                                         args.warmup_secs, args.warmup_requests)
            if s:
                all_series[rps] = s
            if tput:
                all_throughput[rps] = tput
            p99 = load_p99_for_dir(rps_root)
            if p99:
                old_p99_by_rps[rps] = p99

        rps_with_data = [r for r in rps_list if r in all_series]
        if len(rps_with_data) > 1:
            plot_sweep_cdf(all_series,     rps_with_data,
                           out_dir / "tpc_sharing_sweep_latency_cdf.pdf")
            plot_sweep_cdf(all_throughput, rps_with_data,
                           out_dir / "tpc_sharing_sweep_throughput_cdf.pdf",
                           metric="throughput")

        if old_p99_by_rps:
            plot_p99_vs_rps(old_p99_by_rps, rps_list,
                            out_dir / "tpc_p99_vs_rps.pdf")

        if not all_series and not has_tpc_sweep:
            print("[Error] No result data found. Run run.sh first.")
            return 1

    # TPC count sweep plot — always run if tpc_sweep/ exists, regardless of structure
    tpc_sweep_data = load_tpc_sweep_data(result_root)
    if tpc_sweep_data:
        plot_tpc_count_sweep(tpc_sweep_data,
                             out_dir / "tpc_count_sweep_latency.pdf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
