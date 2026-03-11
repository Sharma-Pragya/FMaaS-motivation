#!/usr/bin/env python3
"""Plot Motivation Experiment #2 results.

Dual y-axis figure:
  Left  axis  (bars)  — GPU memory (MB)
  Right axis  (lines) — Throughput (req/s)
  X-axis              — Number of tasks
  Two series          — task_sharing (blue) vs deploy_sharing (orange)
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SERVING_DIR = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Load summary.csv
# ---------------------------------------------------------------------------

def load_summary(path: Path) -> Dict[str, Dict[int, Dict]]:
    """Returns {strategy: {n_tasks: row_dict}}."""
    data: Dict[str, Dict[int, Dict]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            s = row["strategy"]
            n = int(row["n_tasks"])
            data.setdefault(s, {})[n] = {
                "gpu_mem_mb":     float(row["gpu_mem_mb"]),
                "throughput_rps": float(row["throughput_rps"]),
                "avg_latency_ms": float(row["avg_latency_ms"]),
            }
    return data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

STRATEGY_LABELS = {
    "task_sharing":   "Task Sharing (1 model/task)",
    "deploy_sharing": "Deploy Sharing (shared backbone)",
}

COLORS = {
    "task_sharing":   "#4C72B0",   # blue
    "deploy_sharing": "#DD8452",   # orange
}

BAR_ALPHA  = 0.75
LINE_ALPHA = 0.95


def make_plot(data: Dict, out_path: Path, strategies: List[str], all_n: List[int]):
    fig, ax_mem = plt.subplots(figsize=(10, 5.5))
    ax_thr = ax_mem.twinx()

    n_strategies = len(strategies)
    bar_width     = 0.35
    group_gap     = bar_width * n_strategies + 0.15   # total width per x-tick group
    x_positions   = np.arange(len(all_n)) * (group_gap + 0.1)

    # ── Bars (memory) ──────────────────────────────────────────────────────
    for si, strategy in enumerate(strategies):
        sd = data.get(strategy, {})
        mem_vals = [sd.get(n, {}).get("gpu_mem_mb", 0.0) for n in all_n]

        offset = (si - (n_strategies - 1) / 2) * bar_width
        bars = ax_mem.bar(
            x_positions + offset,
            mem_vals,
            width=bar_width,
            label=STRATEGY_LABELS.get(strategy, strategy),
            color=COLORS.get(strategy, "gray"),
            alpha=BAR_ALPHA,
            zorder=2,
        )

    # ── Lines (throughput) ─────────────────────────────────────────────────
    for strategy in strategies:
        sd = data.get(strategy, {})
        thr_vals = [sd.get(n, {}).get("throughput_rps", float("nan")) for n in all_n]

        ax_thr.plot(
            x_positions,
            thr_vals,
            color=COLORS.get(strategy, "gray"),
            linewidth=2.2,
            linestyle="--",
            marker="o",
            markersize=6,
            alpha=LINE_ALPHA,
            zorder=3,
            label=f"{STRATEGY_LABELS.get(strategy, strategy)} (throughput)",
        )

    # ── Axes labels & formatting ───────────────────────────────────────────
    ax_mem.set_xlabel("Number of Tasks", fontsize=13)
    ax_mem.set_ylabel("GPU Memory (MB)", fontsize=13, color="#333333")
    ax_thr.set_ylabel("Throughput (req/s)", fontsize=13, color="#333333")

    ax_mem.set_xticks(x_positions)
    ax_mem.set_xticklabels([str(n) for n in all_n], fontsize=11)
    ax_mem.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.1f}k" if v >= 1000 else f"{v:.0f}"))
    ax_thr.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))

    ax_mem.set_ylim(bottom=0)
    ax_thr.set_ylim(bottom=0)
    ax_mem.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    # ── Combined legend ────────────────────────────────────────────────────
    handles_mem, labels_mem = ax_mem.get_legend_handles_labels()
    handles_thr, labels_thr = ax_thr.get_legend_handles_labels()
    ax_mem.legend(
        handles_mem + handles_thr,
        labels_mem + labels_thr,
        loc="upper left",
        fontsize=9,
        framealpha=0.85,
    )

    ax_mem.set_title(
        "Task Sharing vs. Deploy Sharing — GPU Memory (bars) & Throughput (lines)",
        fontsize=13,
        pad=12,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default=os.environ.get("EXP_DIR", "experiments/motivation2/results"))
    parser.add_argument("--strategies", default="task_sharing,deploy_sharing")
    args = parser.parse_args()

    result_root = (SERVING_DIR / args.exp_dir).resolve()
    summary_path = result_root / "summary.csv"

    if not summary_path.exists():
        print(f"[Error] summary.csv not found at {summary_path}")
        return 1

    strategies = [s.strip() for s in args.strategies.split(",")]
    data = load_summary(summary_path)

    # Union of all n_tasks values present, sorted ascending
    all_n = sorted({n for sd in data.values() for n in sd})
    if not all_n:
        print("[Error] No data rows found in summary.csv")
        return 1

    print(f"[Plot] strategies={strategies}  n_tasks={all_n}")

    out_dir = result_root
    make_plot(data, out_dir / "motivation2_memory_throughput.png", strategies, all_n)

    # Also save individual PNGs for memory and throughput separately
    _plot_single(data, out_dir / "motivation2_memory.png",     strategies, all_n, metric="gpu_mem_mb",     ylabel="GPU Memory (MB)",     title="GPU Memory vs. Number of Tasks")
    _plot_single(data, out_dir / "motivation2_throughput.png", strategies, all_n, metric="throughput_rps", ylabel="Throughput (req/s)",   title="Throughput vs. Number of Tasks")
    _plot_single(data, out_dir / "motivation2_latency.png",    strategies, all_n, metric="avg_latency_ms", ylabel="Avg Latency (ms)",     title="Avg Latency vs. Number of Tasks")

    return 0


def _plot_single(data, out_path, strategies, all_n, metric, ylabel, title):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n_strategies = len(strategies)
    bar_width = 0.35
    group_gap = bar_width * n_strategies + 0.15
    x_positions = np.arange(len(all_n)) * (group_gap + 0.1)

    for si, strategy in enumerate(strategies):
        sd = data.get(strategy, {})
        vals = [sd.get(n, {}).get(metric, float("nan")) for n in all_n]
        offset = (si - (n_strategies - 1) / 2) * bar_width
        ax.bar(
            x_positions + offset, vals,
            width=bar_width,
            label=STRATEGY_LABELS.get(strategy, strategy),
            color=COLORS.get(strategy, "gray"),
            alpha=BAR_ALPHA,
            zorder=2,
        )

    ax.set_xlabel("Number of Tasks", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(n) for n in all_n], fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
