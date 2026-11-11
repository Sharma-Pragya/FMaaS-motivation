#!/usr/bin/env python3
"""sharing_benefit/llm/plot.py — Plots for the LLM sharing-benefit experiment.

Directory layout (matches sharing_benefit/tpc):
  <exp-dir>/singles/rps_{rps}/single_{task}/latencies.csv
  <exp-dir>/ntasks_{n}/rps_{rps}/{condition}/latencies.csv
  <exp-dir>/ntasks_{n}/rps_{rps}/single_{task}/   (symlink into singles/)

Per (ntasks, rps) produces:
  llm_sharing_latency_cdf_rps{N}.pdf
  llm_sharing_summary_bars_rps{N}.pdf       (P99 latency)
  llm_sharing_mean_latency_rps{N}.pdf
  llm_per_task_latency_rps{N}.pdf

Across RPS (when multiple rps values):
  llm_sharing_sweep_latency_cdf.pdf
  llm_p99_vs_rps.pdf

Across ntasks (when multiple ntasks values):
  llm_ntasks_mean_latency.pdf
  llm_ntasks_p99_latency.pdf

Usage:
    python experiments/sharing_benefit/llm/plot.py \\
        --exp-dir experiments/sharing_benefit/llm/results
"""
from __future__ import annotations

import argparse
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
from matplotlib.patches import Patch
import numpy as np


# ---------------------------------------------------------------------------
# Styles (kept consistent with sharing_benefit/tpc)
# ---------------------------------------------------------------------------

SERIES_ORDER  = ["single", "no_sharing_tpc", "no_sharing", "sharing"]
SERIES_COLORS = {
    "single":         "#A9C7B5",
    "no_sharing_tpc": "#6B9AC4",
    "no_sharing":     "#888888",
    "sharing":        "#E06C75",
}
SERIES_LABELS = {
    "single":         "ST",
    "no_sharing_tpc": "SP",
    "no_sharing":     "BE",
    "sharing":        "FMVisor",
}
SERIES_LINESTYLE = {
    "single":         ":",
    "no_sharing_tpc": "--",
    "no_sharing":     "-.",
    "sharing":        "-",
}
SERIES_MARKER = {
    "single":         "s",
    "no_sharing_tpc": "^",
    "no_sharing":     "D",
    "sharing":        "o",
}


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


def save_figure(fig: plt.Figure, out_path: Path, *, pad_inches: float = 0.05) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=pad_inches)
    print(f"[Plot] Saved: {out_path}")


def _nice_upper(val: float) -> float:
    if val <= 0:
        return 1.0
    exp = 10 ** np.floor(np.log10(val))
    for mult in (1.0, 1.5, 2.0, 2.5, 5.0, 7.5, 10.0):
        candidate = mult * exp
        if candidate >= val:
            return float(candidate)
    return float(10.0 * exp)


# ---------------------------------------------------------------------------
# Directory structure helpers
# ---------------------------------------------------------------------------

def detect_ntasks_list(result_root: Path) -> List[int]:
    vals = []
    if not result_root.exists():
        return vals
    for d in result_root.iterdir():
        if d.is_dir() and d.name.startswith("ntasks_"):
            try:
                vals.append(int(d.name.split("_", 1)[1]))
            except ValueError:
                pass
    return sorted(vals)


def detect_rps_list(parent: Path) -> List[int]:
    vals = []
    if not parent.exists():
        return vals
    for d in parent.iterdir():
        if d.is_dir() and d.name.startswith("rps_"):
            try:
                vals.append(int(d.name.split("_", 1)[1]))
            except ValueError:
                pass
    return sorted(vals)


def _singles_fallback(rps_root: Path, cond: str) -> Path:
    rps_name = rps_root.name
    candidate = rps_root.parent.parent / "singles" / rps_name / cond / "latencies.csv"
    if candidate.exists():
        return candidate
    return rps_root.parent / "singles" / rps_name / cond / "latencies.csv"


def _resolve_cond_file(rps_root: Path, cond: str, fname: str) -> Path:
    p = rps_root / cond / fname
    if not p.exists() and cond.startswith("single_"):
        fb = _singles_fallback(rps_root, cond).parent / fname
        if fb.exists():
            return fb
    return p


def detect_single_conditions(rps_root: Path) -> List[str]:
    found: set = set()
    if rps_root.exists():
        for d in rps_root.iterdir():
            if d.name.startswith("single_") and (d.is_dir() or d.is_symlink()):
                found.add(d.name)
    rps_name = rps_root.name
    for singles_base in (
        rps_root.parent.parent / "singles" / rps_name,
        rps_root.parent / "singles" / rps_name,
    ):
        if singles_base.exists():
            for d in singles_base.iterdir():
                if d.name.startswith("single_") and d.is_dir():
                    found.add(d.name)
    return sorted(found)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _read_latencies(path: Path, warmup_secs: float) -> List[float]:
    if not path.exists():
        return []
    with path.open() as f:
        reader = csv.DictReader(f)
        return [
            float(r["latency_ms"]) for r in reader
            if "elapsed_sec" in r and float(r["elapsed_sec"]) > warmup_secs
        ]


def load_series_latencies(rps_root: Path, warmup_secs: float = 10.0) -> Dict[str, List[float]]:
    """Return {series: [latency_ms, ...]} pooled by series. Singles get merged."""
    single_conds = detect_single_conditions(rps_root)
    series: Dict[str, List[float]] = {}

    single_lats: List[float] = []
    for cond in single_conds:
        single_lats.extend(_read_latencies(_resolve_cond_file(rps_root, cond, "latencies.csv"),
                                          warmup_secs))
    if single_lats:
        series["single"] = single_lats

    for cond in ("no_sharing_tpc", "no_sharing", "sharing"):
        lats = _read_latencies(rps_root / cond / "latencies.csv", warmup_secs)
        if lats:
            series[cond] = lats
    return series


def load_task_results(rps_root: Path) -> Dict[str, List[Dict]]:
    data: Dict[str, List[Dict]] = {}
    single_conds = detect_single_conditions(rps_root)
    for cond in single_conds + ["no_sharing_tpc", "no_sharing", "sharing"]:
        path = _resolve_cond_file(rps_root, cond, "task_results.csv")
        if not path.exists():
            continue
        with path.open() as f:
            data[cond] = list(csv.DictReader(f))
    return data


def _aggregate_stat(task_results: Dict[str, List[Dict]], column: str) -> Dict[str, float]:
    """Pool single_* into 'single', return {series: mean over tasks}."""
    out: Dict[str, float] = {}
    single_vals: List[float] = []
    for cond, rows in task_results.items():
        if not cond.startswith("single_"):
            continue
        single_vals.extend(float(r[column]) for r in rows if column in r)
    if single_vals:
        out["single"] = float(np.mean(single_vals))
    for cond in ("no_sharing_tpc", "no_sharing", "sharing"):
        rows = task_results.get(cond, [])
        vals = [float(r[column]) for r in rows if column in r]
        if vals:
            out[cond] = float(np.mean(vals))
    return out


def _active_tasks_for_dir(rps_root: Path) -> List[str]:
    for cond in ("sharing", "no_sharing", "no_sharing_tpc"):
        path = rps_root / cond / "task_results.csv"
        if path.exists():
            with path.open() as f:
                tasks = [r["task"] for r in csv.DictReader(f) if "task" in r]
            if tasks:
                return tasks
    if rps_root.exists():
        return sorted(
            d.name[len("single_"):] for d in rps_root.iterdir()
            if d.name.startswith("single_") and (d.is_dir() or d.is_symlink())
        )
    return []


def load_per_task_metric(rps_root: Path, column: str) -> Dict[str, Dict[str, float]]:
    """{condition_series: {task: value}}."""
    result: Dict[str, Dict[str, float]] = {}
    for task in _active_tasks_for_dir(rps_root):
        path = _resolve_cond_file(rps_root, f"single_{task}", "task_results.csv")
        if not path.exists():
            continue
        with path.open() as f:
            for row in csv.DictReader(f):
                if column in row and "task" in row:
                    result.setdefault("single", {})[row["task"]] = float(row[column])
    for cond in ("no_sharing_tpc", "no_sharing", "sharing"):
        path = _resolve_cond_file(rps_root, cond, "task_results.csv")
        if not path.exists():
            continue
        with path.open() as f:
            for row in csv.DictReader(f):
                if column in row and "task" in row:
                    result.setdefault(cond, {})[row["task"]] = float(row[column])
    return result


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _present_series(series: Dict[str, object]) -> List[str]:
    return [s for s in SERIES_ORDER if s in series and series[s]]


def _legend_handles(series_keys: List[str]) -> List:
    return [
        plt.Line2D([0], [0],
                   color=SERIES_COLORS[s],
                   linestyle=SERIES_LINESTYLE[s],
                   linewidth=1.0,
                   label=SERIES_LABELS[s])
        for s in series_keys
    ]


def _bar_legend_handles(series_keys: List[str]) -> List:
    return [
        Patch(facecolor=SERIES_COLORS[s], edgecolor="black", linewidth=0.4,
              label=SERIES_LABELS[s])
        for s in series_keys
    ]


# ---------------------------------------------------------------------------
# Plot: Latency CDF
# ---------------------------------------------------------------------------

def _plot_cdf_on_ax(ax: plt.Axes, series: Dict[str, List[float]],
                    x_upper: Optional[float] = None) -> None:
    all_vals: List[float] = []
    for s in SERIES_ORDER:
        lats = series.get(s)
        if not lats:
            continue
        arr = np.sort(np.array(lats))
        cdf = np.arange(1, len(arr) + 1) / len(arr)
        ax.plot(arr, cdf,
                color=SERIES_COLORS[s], linestyle=SERIES_LINESTYLE[s],
                linewidth=1.0, label=SERIES_LABELS[s])
        all_vals.extend(arr.tolist())
    x_max = x_upper if x_upper is not None else _nice_upper(
        float(np.max(all_vals)) * 1.05 if all_vals else 1.0
    )
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 5))
    ax.grid(axis="both", zorder=0)
    ax.set_axisbelow(True)


def plot_latency_cdf(series: Dict[str, List[float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(2.8, 1.6))
    _plot_cdf_on_ax(ax, series)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.legend(handles=_legend_handles(_present_series(series)),
              frameon=False, loc="lower right",
              handlelength=1.2, handletextpad=0.3, labelspacing=0.2,
              fontsize=6)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_sweep_latency_cdf(all_series: Dict[int, Dict[str, List[float]]],
                           rps_list: List[int], out_path: Path) -> None:
    rps_with_data = [r for r in rps_list if all_series.get(r)]
    if not rps_with_data:
        return
    # Shared x-bound across panels
    all_vals: List[float] = []
    for r in rps_with_data:
        for v in all_series[r].values():
            all_vals.extend(v)
    x_upper = _nice_upper(float(np.max(all_vals)) * 1.05) if all_vals else 1.0

    n = len(rps_with_data)
    fig, axes = plt.subplots(1, n, figsize=(1.6 * n, 1.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, rps_with_data):
        _plot_cdf_on_ax(ax, all_series[r], x_upper=x_upper)
        ax.set_title(f"{r} req/s", pad=2)
        ax.set_xlabel("Latency (ms)")
    axes[0].set_ylabel("CDF")
    legend_series = [s for s in SERIES_ORDER
                     if any(all_series.get(r, {}).get(s) for r in rps_with_data)]
    fig.legend(handles=_legend_handles(legend_series),
               loc="upper center", bbox_to_anchor=(0.5, 1.08),
               ncol=max(1, len(legend_series)), frameon=False,
               handlelength=1.5, columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: bar charts (P99, mean)
# ---------------------------------------------------------------------------

def _plot_bar_summary(stat: Dict[str, float], out_path: Path,
                      ylabel: str, fmt: str = "{:.0f}") -> None:
    series = [s for s in SERIES_ORDER if s in stat]
    if not series:
        return
    fig, ax = plt.subplots(figsize=(2.5, 1.5))
    x = np.arange(len(series))
    bars = ax.bar(x, [stat[s] for s in series], width=0.8,
                  color=[SERIES_COLORS[s] for s in series],
                  edgecolor="black", linewidth=0.4, zorder=2)
    for bar, s in zip(bars, series):
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                fmt.format(v), ha="center", va="bottom", fontsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([SERIES_LABELS[s] for s in series], rotation=0)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ymax = max(stat[s] for s in series)
    ax.set_ylim(0, _nice_upper(ymax * 1.25))
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_p99_bars(task_results: Dict[str, List[Dict]], out_path: Path) -> None:
    p99 = _aggregate_stat(task_results, "p99_latency_ms")
    if p99:
        _plot_bar_summary(p99, out_path, ylabel="P99 Latency (ms)")


def plot_mean_bars(task_results: Dict[str, List[Dict]], out_path: Path) -> None:
    mean = _aggregate_stat(task_results, "avg_latency_ms")
    if mean:
        _plot_bar_summary(mean, out_path, ylabel="Mean Latency (ms)")


# ---------------------------------------------------------------------------
# Plot: per-task latency (one panel per condition)
# ---------------------------------------------------------------------------

def plot_per_task_latency(per_task: Dict[str, Dict[str, float]], out_path: Path,
                          ylabel: str = "Mean Latency (ms)") -> None:
    conditions = [s for s in SERIES_ORDER if s in per_task and per_task[s]]
    if not conditions:
        return
    all_tasks: set = set()
    for d in per_task.values():
        all_tasks.update(d.keys())
    tasks = sorted(all_tasks)

    n_panels = len(conditions)
    fig_w = max(1.8 * n_panels, 3.5)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 1.8),
                             sharey=False, squeeze=False)
    axes = axes[0]
    x = np.arange(len(tasks))
    for ax, cond in zip(axes, conditions):
        vals = [per_task[cond].get(t, float("nan")) for t in tasks]
        bars = ax.bar(x, vals, width=0.8,
                      color=SERIES_COLORS[cond],
                      edgecolor="black", linewidth=0.4, zorder=2)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=5)
        ax.set_title(SERIES_LABELS[cond], pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=25, ha="right", fontsize=6)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        finite = [v for v in vals if not np.isnan(v)]
        if finite:
            ax.set_ylim(0, _nice_upper(max(finite) * 1.25))
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: P99 vs RPS (line, one line per condition)
# ---------------------------------------------------------------------------

def plot_stat_vs_rps(stat_data: Dict[int, Dict[str, float]],
                     rps_list: List[int], out_path: Path,
                     ylabel: str) -> None:
    rps_with_data = [r for r in rps_list if stat_data.get(r)]
    if not rps_with_data:
        return
    present = [s for s in SERIES_ORDER
               if any(s in stat_data.get(r, {}) for r in rps_with_data)]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(2.4, 1.7))
    for s in present:
        ys = [stat_data.get(r, {}).get(s, float("nan")) for r in rps_with_data]
        ax.plot(rps_with_data, ys,
                color=SERIES_COLORS[s], linestyle=SERIES_LINESTYLE[s],
                marker=SERIES_MARKER[s], markersize=4, linewidth=1.0,
                label=SERIES_LABELS[s])
    ax.set_xlabel("RPS/task")
    ax.set_ylabel(ylabel)
    ax.set_xticks(rps_with_data)
    ax.grid(axis="both", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6, loc="best",
              handlelength=1.2, handletextpad=0.3, labelspacing=0.2)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: ntasks sweep — stat vs ntasks
# ---------------------------------------------------------------------------

def plot_stat_vs_ntasks(data: Dict[int, Dict[int, Dict[str, float]]],
                        rps_list: List[int], ntasks_list: List[int],
                        out_path: Path, ylabel: str) -> None:
    """data: {rps: {ntasks: {series: stat}}} — one panel per RPS."""
    rps_with_data = [r for r in rps_list if any(data.get(r, {}).values())]
    if not rps_with_data:
        return
    present: List[str] = []
    for s in SERIES_ORDER:
        if any(s in data.get(r, {}).get(n, {})
               for r in rps_with_data for n in ntasks_list):
            present.append(s)
    if not present:
        return

    n_panels = len(rps_with_data)
    fig_w = max(2.6 * n_panels, 3.0)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 1.8),
                             sharey=False, squeeze=False)
    axes = axes[0]
    for ax, rps in zip(axes, rps_with_data):
        rps_data = data.get(rps, {})
        for s in present:
            xs, ys = [], []
            for n in ntasks_list:
                v = rps_data.get(n, {}).get(s)
                if v is not None:
                    xs.append(n); ys.append(v)
            if xs:
                ax.plot(xs, ys,
                        color=SERIES_COLORS[s], linestyle=SERIES_LINESTYLE[s],
                        marker=SERIES_MARKER[s], markersize=4, linewidth=1.0,
                        label=SERIES_LABELS[s])
        ax.set_title(f"RPS = {rps}", pad=3)
        ax.set_xlabel("Number of Tasks")
        ax.set_xticks(ntasks_list)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.grid(axis="both", zorder=0)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
    axes[0].legend(handles=_legend_handles(present),
                   loc="best", fontsize=6, frameon=False,
                   handlelength=1.2, handletextpad=0.3, labelspacing=0.2)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default=os.environ.get(
        "EXP_DIR", "experiments/sharing_benefit/llm/results"))
    parser.add_argument("--rps-sweep", default=None,
                        help="Comma-separated RPS values (auto-detected if omitted)")
    parser.add_argument("--num-tasks-sweep", default=None,
                        help="Comma-separated ntasks values (auto-detected if omitted)")
    parser.add_argument("--warmup-secs", type=float, default=10.0)
    args = parser.parse_args()

    result_root = (SERVING_DIR / args.exp_dir).resolve()
    if not result_root.exists():
        print(f"[Error] Results directory not found: {result_root}")
        return 1

    apply_paper_style()
    out_dir = result_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    ntasks_list = (
        [int(x.strip()) for x in args.num_tasks_sweep.split(",")]
        if args.num_tasks_sweep else detect_ntasks_list(result_root)
    )
    if not ntasks_list:
        print(f"[Error] No ntasks_*/ subdirs under {result_root}")
        return 1

    rps_set: set = set()
    for n in ntasks_list:
        rps_set.update(detect_rps_list(result_root / f"ntasks_{n}"))
    rps_list = (
        sorted(int(x.strip()) for x in args.rps_sweep.split(","))
        if args.rps_sweep else sorted(rps_set)
    )
    print(f"[Plot] ntasks={ntasks_list}  rps={rps_list}")

    # Per-(ntasks, rps) plots + aggregate buffers
    p99_by_ntasks_rps:  Dict[int, Dict[int, Dict[str, float]]] = {}
    mean_by_ntasks_rps: Dict[int, Dict[int, Dict[str, float]]] = {}

    for n in ntasks_list:
        per_ntasks_dir = out_dir / f"ntasks_{n}"
        per_ntasks_dir.mkdir(exist_ok=True)
        p99_by_rps:  Dict[int, Dict[str, float]] = {}
        mean_by_rps: Dict[int, Dict[str, float]] = {}
        series_by_rps: Dict[int, Dict[str, List[float]]] = {}

        for rps in rps_list:
            rps_root = result_root / f"ntasks_{n}" / f"rps_{rps}"
            if not rps_root.exists():
                continue

            series = load_series_latencies(rps_root, args.warmup_secs)
            tres   = load_task_results(rps_root)

            if series:
                plot_latency_cdf(series,
                                 per_ntasks_dir / f"llm_sharing_latency_cdf_rps{rps}.pdf")
                series_by_rps[rps] = series
            if tres:
                plot_p99_bars(tres,
                              per_ntasks_dir / f"llm_sharing_summary_bars_rps{rps}.pdf")
                plot_mean_bars(tres,
                               per_ntasks_dir / f"llm_sharing_mean_latency_rps{rps}.pdf")
                p99_by_rps[rps]  = _aggregate_stat(tres, "p99_latency_ms")
                mean_by_rps[rps] = _aggregate_stat(tres, "avg_latency_ms")

            per_task = load_per_task_metric(rps_root, "avg_latency_ms")
            if per_task:
                plot_per_task_latency(per_task,
                                      per_ntasks_dir / f"llm_per_task_latency_rps{rps}.pdf")

        # RPS sweep plots within this ntasks
        if len(series_by_rps) > 1:
            plot_sweep_latency_cdf(series_by_rps, sorted(series_by_rps),
                                   per_ntasks_dir / "llm_sharing_sweep_latency_cdf.pdf")
        if len(p99_by_rps) > 1:
            plot_stat_vs_rps(p99_by_rps, sorted(p99_by_rps),
                             per_ntasks_dir / "llm_p99_vs_rps.pdf",
                             ylabel="P99 Latency (ms)")
            plot_stat_vs_rps(mean_by_rps, sorted(mean_by_rps),
                             per_ntasks_dir / "llm_mean_vs_rps.pdf",
                             ylabel="Mean Latency (ms)")

        # Buffer for ntasks-sweep plots ({rps: {ntasks: {series: stat}}})
        for rps, d in p99_by_rps.items():
            p99_by_ntasks_rps.setdefault(rps, {})[n] = d
        for rps, d in mean_by_rps.items():
            mean_by_ntasks_rps.setdefault(rps, {})[n] = d

    # ntasks sweep plots (only if user actually swept ntasks)
    if len(ntasks_list) > 1:
        plot_stat_vs_ntasks(p99_by_ntasks_rps, sorted(p99_by_ntasks_rps), ntasks_list,
                            out_dir / "llm_ntasks_p99_latency.pdf",
                            ylabel="P99 Latency (ms)")
        plot_stat_vs_ntasks(mean_by_ntasks_rps, sorted(mean_by_ntasks_rps), ntasks_list,
                            out_dir / "llm_ntasks_mean_latency.pdf",
                            ylabel="Mean Latency (ms)")

    print(f"[Plot] Done. Output in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
