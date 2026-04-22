#!/usr/bin/env python3
"""batch_size_vs_rps/plot.py — Plot observed batch size, latency, and
throughput vs. request rate for different batch waiting times.

Reads summary.json / latencies.csv / batch_sizes.csv produced by run.py and generates:
  plots/batch_size_vs_rps.pdf        — x=req rate, y=mean observed batch size
  plots/latency_p99_vs_rps.pdf       — x=req rate, y=p99 latency (ms)
  plots/latency_avg_vs_rps.pdf       — x=req rate, y=avg latency (ms)
  plots/throughput_vs_rps.pdf        — x=req rate, y=measured throughput (req/s)
  plots/batch_size_distribution.pdf  — PMF of batch sizes, one panel per RPS
  plots/workload_trace.pdf           — Poisson arrival rate over time, one line per RPS

Line plots have one line per batch waiting time (legend).

Usage:
    python experiments/batch_size_vs_rps/plot.py \
        [--exp-dir  experiments/batch_size_vs_rps/results] \
        [--rps-sweep  1,2,5,10,20] \
        [--wait-sweep 0,10,50,100,200]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

WAIT_COLORS = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # gray
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


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
        "lines.markersize":   3.5,
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
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _result_dir(result_root: Path, wait_ms: float, rps: float) -> Optional[Path]:
    """Return result directory for (wait_ms, rps), or None if missing."""
    for w_str in (f"{wait_ms:.0f}", str(wait_ms)):
        for r_str in (f"{rps:.0f}", str(rps)):
            p = result_root / f"wait_{w_str}" / f"rps_{r_str}"
            if p.exists():
                return p
    return None


def load_summary(result_root: Path, wait_ms: float, rps: float) -> Optional[dict]:
    d = _result_dir(result_root, wait_ms, rps)
    if d is None:
        return None
    p = d / "summary.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def load_latencies(result_root: Path, wait_ms: float, rps: float,
                   warmup_secs: float = 10.0) -> List[float]:
    d = _result_dir(result_root, wait_ms, rps)
    if d is None:
        return []
    p = d / "latencies.csv"
    if not p.exists():
        return []
    lats = []
    with p.open() as f:
        for row in csv.DictReader(f):
            if float(row["send_time_rel"]) > warmup_secs:
                lats.append(float(row["latency_ms"]))
    return lats


def load_batch_sizes(result_root: Path, wait_ms: float, rps: float) -> List[int]:
    """Return list of observed_batch_size values from batch_sizes.csv."""
    d = _result_dir(result_root, wait_ms, rps)
    if d is None:
        return []
    p = d / "batch_sizes.csv"
    if not p.exists():
        return []
    sizes = []
    with p.open() as f:
        for row in csv.DictReader(f):
            sizes.append(int(row["observed_batch_size"]))
    return sizes


def load_send_times(result_root: Path, wait_ms: float, rps: float) -> List[float]:
    """Return all scheduled send_time_rel values from send_times.csv.

    This reflects every request that was fired, including those that timed out
    or were dropped — giving the true Poisson arrival trace.
    Falls back to latencies.csv if send_times.csv is absent (old results).
    """
    d = _result_dir(result_root, wait_ms, rps)
    if d is None:
        return []
    for filename in ("send_times.csv", "latencies.csv"):
        p = d / filename
        if p.exists():
            times = []
            with p.open() as f:
                for row in csv.DictReader(f):
                    times.append(float(row["send_time_rel"]))
            return times
    return []


_LOG_BATCH_RE = re.compile(r"Prepared batch_size=(\d+)")


def _logs_dir(result_root: Path) -> Optional[Path]:
    for candidate in (result_root / "logs", result_root.parent / "logs"):
        if candidate.exists():
            return candidate
    return None


def load_log_batch_sizes(result_root: Path, wait_ms: float, rps: float) -> List[int]:
    """Return prepared batch sizes parsed from device logs."""
    logs_dir = _logs_dir(result_root)
    if logs_dir is None:
        return []

    candidates = [
        logs_dir / f"device_wait{wait_ms:.0f}_rps{rps:.0f}.log",
        logs_dir / f"device_wait{int(wait_ms)}_rps{int(rps)}.log",
    ]
    log_path = next((p for p in candidates if p.exists()), None)
    if log_path is None:
        return []

    sizes: List[int] = []
    with log_path.open(errors="ignore") as f:
        for line in f:
            m = _LOG_BATCH_RE.search(line)
            if m:
                sizes.append(int(m.group(1)))
    return sizes


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

MetricTable = Dict[float, Dict[float, float]]   # {wait_ms: {rps: value}}


def build_tables(
    result_root: Path,
    rps_list: List[float],
    wait_list: List[float],
    warmup_secs: float,
) -> tuple[MetricTable, MetricTable, MetricTable, MetricTable, MetricTable]:
    """Returns (observed_bs, log_prepared_bs, p99_latency, avg_latency, throughput) tables."""
    bs_tbl:    MetricTable = {}
    log_bs_tbl: MetricTable = {}
    p99_tbl:   MetricTable = {}
    avg_tbl:   MetricTable = {}
    tput_tbl:  MetricTable = {}

    for wait_ms in wait_list:
        for rps in rps_list:
            s = load_summary(result_root, wait_ms, rps)
            if s is None:
                print(f"[Warn] Missing summary: wait={wait_ms}ms  rps={rps}")
                continue

            mean_bs  = float(s.get("mean_batch_size", 0))
            log_sizes = load_log_batch_sizes(result_root, wait_ms, rps)
            log_mean_bs = float(np.mean(log_sizes)) if log_sizes else 0.0
            duration = float(s.get("duration_s", 1))
            n_meas   = int(s.get("n_requests_measured", 0))

            # Latencies from CSV for accurate percentiles
            lats = load_latencies(result_root, wait_ms, rps, warmup_secs)
            if lats:
                p99  = float(np.percentile(lats, 99))
                avg  = float(np.mean(lats))
            else:
                p99  = float(s.get("p99_latency_ms", 0))
                avg  = float(s.get("avg_latency_ms", 0))

            eff_dur = max(duration - warmup_secs, 1.0)
            tput    = n_meas / eff_dur if n_meas else 0.0

            bs_tbl.setdefault(wait_ms,  {})[rps] = mean_bs
            log_bs_tbl.setdefault(wait_ms, {})[rps] = log_mean_bs
            p99_tbl.setdefault(wait_ms, {})[rps] = p99
            avg_tbl.setdefault(wait_ms, {})[rps] = avg
            tput_tbl.setdefault(wait_ms, {})[rps] = tput

            print(f"[Load] wait={wait_ms}ms  rps={rps:5.0f}  "
                  f"mean_bs={mean_bs:.2f}  log_bs={log_mean_bs:.2f}  p99={p99:.1f}ms  tput={tput:.2f}")

    return bs_tbl, log_bs_tbl, p99_tbl, avg_tbl, tput_tbl


# ---------------------------------------------------------------------------
# Generic line-plot helper
# ---------------------------------------------------------------------------

def _line_plot(
    table: MetricTable,
    rps_list: List[float],
    ylabel: str,
    out_path: Path,
    y_bottom: float = 0.0,
    legend_loc: str = "upper left",
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.2))

    wait_list_present = sorted(table.keys())
    for i, wait_ms in enumerate(wait_list_present):
        rps_vals = sorted(table[wait_ms].keys())
        y_vals   = [table[wait_ms][r] for r in rps_vals]
        color    = WAIT_COLORS[i % len(WAIT_COLORS)]
        marker   = MARKERS[i % len(MARKERS)]
        label    = f"{wait_ms:.0f} ms" if wait_ms > 0 else "0 ms (no wait)"
        ax.plot(rps_vals, y_vals,
                color=color, marker=marker,
                linewidth=1.2, markersize=3.5,
                label=label)

    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=y_bottom)
    ax.grid(axis="both", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(sorted(rps_list))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))

    ax.legend(
        title="Batch wait",
        title_fontsize=6,
        frameon=False,
        loc=legend_loc,
        handlelength=1.5,
        handletextpad=0.3,
        labelspacing=0.3,
    )

    fig.tight_layout()
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Plot: batch size distribution (PMF), one subplot per RPS
# ---------------------------------------------------------------------------

def plot_batch_size_distribution(
    result_root: Path,
    rps_list: List[float],
    wait_list: List[float],
    out_path: Path,
    source: str = "observed",
) -> None:
    """Bar-chart PMF of observed batch sizes.

    Layout: one subplot per RPS value, bars grouped by wait_ms.
    x = batch size (integer), y = fraction of batches.
    """
    rps_with_data = []
    for rps in sorted(rps_list):
        for wait_ms in wait_list:
            sizes = (
                load_log_batch_sizes(result_root, wait_ms, rps)
                if source == "log"
                else load_batch_sizes(result_root, wait_ms, rps)
            )
            if sizes:
                rps_with_data.append(rps)
                break
    if not rps_with_data:
        print("[Warn] No batch_sizes.csv data found — skipping distribution plot")
        return

    n = len(rps_with_data)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.2), sharey=False)
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.35)

    wait_list_sorted = sorted(wait_list)
    bar_width = 0.8 / max(len(wait_list_sorted), 1)

    for ax, rps in zip(axes, rps_with_data):
        all_sizes: List[int] = []
        series_data: dict[float, dict[int, float]] = {}

        for wait_ms in wait_list_sorted:
            sizes = (
                load_log_batch_sizes(result_root, wait_ms, rps)
                if source == "log"
                else load_batch_sizes(result_root, wait_ms, rps)
            )
            if not sizes:
                continue
            all_sizes.extend(sizes)
            counts: dict[int, int] = defaultdict(int)
            for s in sizes:
                counts[s] += 1
            total = len(sizes)
            series_data[wait_ms] = {k: v / total for k, v in counts.items()}

        if not all_sizes:
            continue

        max_size = max(all_sizes)
        x_vals = list(range(1, max_size + 1))

        for i, wait_ms in enumerate(wait_list_sorted):
            pmf = series_data.get(wait_ms)
            if pmf is None:
                continue
            color  = WAIT_COLORS[i % len(WAIT_COLORS)]
            label  = f"{wait_ms:.0f} ms" if wait_ms > 0 else "0 ms"
            offset = (i - (len(wait_list_sorted) - 1) / 2) * bar_width
            heights = [pmf.get(x, 0.0) for x in x_vals]
            ax.bar([x + offset for x in x_vals], heights,
                   width=bar_width, color=color, label=label,
                   edgecolor="none", zorder=2)

        ax.set_title(f"{rps:.0f} req/s", pad=2)
        ax.set_xlabel("Batch Size")
        ax.set_xticks(x_vals)
        ax.set_xlim(0.5, max_size + 0.5)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.set_ylabel("Fraction of Batches")

    # Shared legend above subplots
    handles = [
        plt.Rectangle((0, 0), 1, 1,
                       color=WAIT_COLORS[i % len(WAIT_COLORS)],
                       label=f"{w:.0f} ms" if w > 0 else "0 ms")
        for i, w in enumerate(wait_list_sorted)
        if any(
            (load_log_batch_sizes(result_root, w, r) if source == "log" else load_batch_sizes(result_root, w, r))
            for r in rps_with_data
        )
    ]
    fig.legend(handles=handles, title="Batch wait", title_fontsize=6,
               loc="upper center", bbox_to_anchor=(0.5, 1.08),
               ncol=len(handles), frameon=False,
               handlelength=1.0, handletextpad=0.3, columnspacing=0.8)

    fig.tight_layout(rect=(0, 0, 1, 1.0))
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Plot: workload trace (arrivals per second over time)
# ---------------------------------------------------------------------------

def plot_workload_trace(
    result_root: Path,
    rps_list: List[float],
    wait_list: List[float],
    out_path: Path,
    bin_size_s: float = 1.0,
) -> None:
    """Arrival rate trace over the experiment duration.

    Uses one representative wait_ms per RPS (the first one that has data).
    x = elapsed time (s), y = arrivals in each bin (req/s).
    One line per RPS value.
    """
    fig, ax = plt.subplots(figsize=(4.5, 2.0))

    plotted = False
    for i, rps in enumerate(sorted(rps_list)):
        send_times: List[float] = []
        for wait_ms in sorted(wait_list):
            send_times = load_send_times(result_root, wait_ms, rps)
            if send_times:
                break
        if not send_times:
            continue

        t_max = max(send_times)
        bins  = np.arange(0, t_max + bin_size_s, bin_size_s)
        counts, edges = np.histogram(send_times, bins=bins)
        bin_centers   = (edges[:-1] + edges[1:]) / 2

        color = WAIT_COLORS[i % len(WAIT_COLORS)]
        ax.plot(bin_centers, counts / bin_size_s,
                color=color, linewidth=0.9, alpha=0.85,
                label=f"{rps:.0f} req/s")
        plotted = True

    if not plotted:
        print("[Warn] No latencies.csv data found — skipping workload trace plot")
        plt.close(fig)
        return

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Arrival Rate (req/s)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(axis="both", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(title="Target rate", title_fontsize=6, frameon=False,
              loc="upper right", handlelength=1.5,
              handletextpad=0.3, labelspacing=0.3)

    fig.tight_layout()
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir",     default=os.environ.get(
                            "EXP_DIR", "experiments/batch_size_vs_rps/results"))
    parser.add_argument("--rps-sweep",   default="100,200,300,400,500",
                        help="Comma-separated RPS values used in run.sh")
    parser.add_argument("--wait-sweep",  default="0,5",
                        help="Comma-separated batch wait times (ms) used in run.sh")
    parser.add_argument("--warmup-secs", type=float, default=10.0)
    args = parser.parse_args()

    result_root = (SERVING_DIR / args.exp_dir).resolve()
    if not result_root.exists():
        print(f"[Error] Results directory not found: {result_root}")
        return 1

    apply_paper_style()

    rps_list  = [float(x.strip()) for x in args.rps_sweep.split(",")]
    wait_list = [float(x.strip()) for x in args.wait_sweep.split(",")]

    bs_tbl, log_bs_tbl, p99_tbl, avg_tbl, tput_tbl = build_tables(
        result_root, rps_list, wait_list, args.warmup_secs)

    if not bs_tbl:
        print("[Error] No data found. Run run.sh first.")
        return 1

    out_dir = result_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Observed batch size
    _line_plot(bs_tbl, rps_list,
               ylabel="Mean Observed Batch Size",
               out_path=out_dir / "batch_size_vs_rps.pdf",
               legend_loc="upper left")
    _line_plot(bs_tbl, rps_list,
               ylabel="Mean Observed Batch Size",
               out_path=out_dir / "batch_size_vs_rps.png",
               legend_loc="upper left")

    # 1b. Prepared batch size from logs
    if any(log_bs_tbl.values()):
        _line_plot(log_bs_tbl, rps_list,
                   ylabel="Mean Prepared Batch Size (Logs)",
                   out_path=out_dir / "prepared_batch_size_vs_rps.pdf",
                   legend_loc="upper left")
        _line_plot(log_bs_tbl, rps_list,
                   ylabel="Mean Prepared Batch Size (Logs)",
                   out_path=out_dir / "prepared_batch_size_vs_rps.png",
                   legend_loc="upper left")

    # 2. P99 latency
    _line_plot(p99_tbl, rps_list,
               ylabel="P99 Latency (ms)",
               out_path=out_dir / "latency_p99_vs_rps.pdf",
               legend_loc="upper left")
    _line_plot(p99_tbl, rps_list,
               ylabel="P99 Latency (ms)",
               out_path=out_dir / "latency_p99_vs_rps.png",
               legend_loc="upper left")

    # 3. Average latency
    _line_plot(avg_tbl, rps_list,
               ylabel="Avg Latency (ms)",
               out_path=out_dir / "latency_avg_vs_rps.pdf",
               legend_loc="upper left")
    _line_plot(avg_tbl, rps_list,
               ylabel="Avg Latency (ms)",
               out_path=out_dir / "latency_avg_vs_rps.png",
               legend_loc="upper left")

    # 4. Throughput
    _line_plot(tput_tbl, rps_list,
               ylabel="Throughput (req/s)",
               out_path=out_dir / "throughput_vs_rps.pdf",
               legend_loc="upper left")
    _line_plot(tput_tbl, rps_list,
               ylabel="Throughput (req/s)",
               out_path=out_dir / "throughput_vs_rps.png",
               legend_loc="upper left")

    # 5. Batch size distribution (PMF per RPS, bars per wait_ms)
    plot_batch_size_distribution(
        result_root, rps_list, wait_list,
        out_dir / "batch_size_distribution.pdf")
    plot_batch_size_distribution(
        result_root, rps_list, wait_list,
        out_dir / "batch_size_distribution.png")

    if any(load_log_batch_sizes(result_root, w, r) for w in wait_list for r in rps_list):
        plot_batch_size_distribution(
            result_root, rps_list, wait_list,
            out_dir / "prepared_batch_size_distribution.pdf",
            source="log")
        plot_batch_size_distribution(
            result_root, rps_list, wait_list,
            out_dir / "prepared_batch_size_distribution.png",
            source="log")

    # 6. Workload trace (arrival rate over time per RPS)
    plot_workload_trace(
        result_root, rps_list, wait_list,
        out_dir / "workload_trace.pdf")
    plot_workload_trace(
        result_root, rps_list, wait_list,
        out_dir / "workload_trace.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
