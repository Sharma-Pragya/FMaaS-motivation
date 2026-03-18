#!/usr/bin/env python3
"""Plot motivation2/tsfm results.

Produces two figures:
  1. latency_vs_rps.png  — p50 latency vs target_rps, one line per n_tasks
                           (aggregated across tasks via median)
  2. tail_vs_rps.png     — p99 latency vs target_rps, one line per n_tasks

Run from serving/:
    python experiments/motivation2/tsfm/plot.py
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SERVING_DIR = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Style (matches motivation1)
# ---------------------------------------------------------------------------
PAPER_PALETTE = {
    "charcoal": "#2F3640",
    "slate":    "#5C6773",
    "grid":     "#D9DEE5",
    "background": "#FAFBFC",
}

N_TASK_COLORS = {1: "#4878CF", 2: "#6ACC65", 4: "#D65F5F"}
N_TASK_MARKERS = {1: "o", 2: "s", 4: "^"}
N_TASK_LINES   = {1: "-", 2: "--", 4: "-."}


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": PAPER_PALETTE["background"],
        "axes.edgecolor": PAPER_PALETTE["slate"],
        "axes.labelcolor": PAPER_PALETTE["charcoal"],
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": PAPER_PALETTE["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.8,
        "xtick.color": PAPER_PALETTE["charcoal"],
        "ytick.color": PAPER_PALETTE["charcoal"],
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[Plot] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_summary(path: Path):
    """Return {n_tasks: {target_rps: [row, ...]}}."""
    data: Dict[int, Dict[float, List[dict]]] = defaultdict(lambda: defaultdict(list))
    with path.open() as f:
        for row in csv.DictReader(f):
            n = int(row["n_tasks"])
            rps = float(row["target_rps"])
            data[n][rps].append({k: float(v) if k not in ("backbone", "task") else v
                                  for k, v in row.items()})
    return data


def aggregate(rows: List[dict], metric: str) -> float:
    """Median across tasks for a given metric."""
    vals = [r[metric] for r in rows]
    return float(np.median(vals))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def plot_latency(data, metric: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for n_tasks in sorted(data.keys()):
        rps_data = data[n_tasks]
        rps_vals = sorted(rps_data.keys())
        lat_vals = [aggregate(rps_data[r], metric) for r in rps_vals]

        ax.plot(
            rps_vals, lat_vals,
            color=N_TASK_COLORS.get(n_tasks, "gray"),
            linestyle=N_TASK_LINES.get(n_tasks, "-"),
            marker=N_TASK_MARKERS.get(n_tasks, "o"),
            linewidth=2.2,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.5,
            markeredgecolor=N_TASK_COLORS.get(n_tasks, "gray"),
            label=f"{n_tasks} task{'s' if n_tasks > 1 else ''}",
            zorder=3,
        )

    ax.set_xlabel("Target RPS (req/s)", fontsize=12, fontweight="semibold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="semibold")
    ax.set_title(title, fontsize=12, fontweight="semibold", pad=8)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=False)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_input_vs_output(data, out_path: Path) -> None:
    """Scatter: input_rps vs output_rps — should be on diagonal if no drops."""
    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    all_vals = []
    for n_tasks in sorted(data.keys()):
        for rps, rows in data[n_tasks].items():
            for row in rows:
                inp = row["input_rps"]
                out = row["output_rps"]
                all_vals.append(max(inp, out))
                ax.scatter(
                    inp, out,
                    color=N_TASK_COLORS.get(n_tasks, "gray"),
                    marker=N_TASK_MARKERS.get(n_tasks, "o"),
                    s=45, zorder=3, alpha=0.85,
                    label=f"{n_tasks} task{'s' if n_tasks > 1 else ''}",
                )

    # Diagonal reference line
    mx = max(all_vals) * 1.05 if all_vals else 40
    ax.plot([0, mx], [0, mx], color=PAPER_PALETTE["slate"],
            linestyle="--", linewidth=1.2, zorder=1, label="ideal")

    ax.set_xlabel("Input RPS (req/s)", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Output RPS (req/s)", fontsize=12, fontweight="semibold")
    ax.set_title("TSFM: Input vs Output Throughput", fontsize=12, fontweight="semibold", pad=8)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), fontsize=10, frameon=False)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_p50_p99_bar(data, n_tasks: int, out_path: Path) -> None:
    """Grouped bar: p50 vs p99 per task at a fixed n_tasks, across RPS."""
    rps_data = data.get(n_tasks)
    if not rps_data:
        return

    # Collect all tasks
    all_tasks = sorted({row["task"] for rows in rps_data.values() for row in rows})
    rps_vals = sorted(rps_data.keys())

    n_tasks_count = len(all_tasks)
    task_colors = plt.cm.tab10(np.linspace(0, 0.6, n_tasks_count))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    for ax_idx, (metric, label) in enumerate([("p50_lat_ms", "P50 Latency (ms)"),
                                               ("p99_lat_ms", "P99 Latency (ms)")]):
        ax = axes[ax_idx]
        bar_width = 0.7 / n_tasks_count
        x = np.arange(len(rps_vals))

        for ti, task in enumerate(all_tasks):
            vals = []
            for rps in rps_vals:
                task_rows = [r for r in rps_data[rps] if r["task"] == task]
                vals.append(task_rows[0][metric] if task_rows else 0.0)
            offset = (ti - (n_tasks_count - 1) / 2) * bar_width
            ax.bar(x + offset, vals, width=bar_width,
                   color=task_colors[ti], label=task, alpha=0.88,
                   edgecolor=PAPER_PALETTE["charcoal"], linewidth=0.7)

        ax.set_xlabel("Target RPS (req/s)", fontsize=11, fontweight="semibold")
        ax.set_ylabel(label, fontsize=11, fontweight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(r)) for r in rps_vals], fontsize=10)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        if ax_idx == 0:
            ax.legend(fontsize=9, frameon=False)

    fig.suptitle(f"TSFM latency per task — {n_tasks} task{'s' if n_tasks > 1 else ''} sharing backbone",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default=os.environ.get("EXP_DIR",
                        "experiments/motivation2/tsfm/results"))
    args = parser.parse_args()

    result_root = (SERVING_DIR / args.exp_dir).resolve()
    summary_path = result_root / "summary.csv"
    plot_dir = result_root.parent / "plots"

    if not summary_path.exists():
        print(f"[Error] Not found: {summary_path}")
        return 1

    data = load_summary(summary_path)
    print(f"[Plot] n_tasks={sorted(data.keys())}")

    # 1. p50 latency vs RPS
    plot_latency(data, "p50_lat_ms", "P50 Latency (ms)",
                 "TSFM: Median Latency vs RPS",
                 plot_dir / "tsfm_p50_vs_rps.png")

    # 2. p99 latency vs RPS
    plot_latency(data, "p99_lat_ms", "P99 Latency (ms)",
                 "TSFM: Tail Latency (P99) vs RPS",
                 plot_dir / "tsfm_p99_vs_rps.png")

    # 3. Input vs output RPS
    plot_input_vs_output(data, plot_dir / "tsfm_input_vs_output.png")

    # 4. Per-task latency breakdown for n_tasks=4
    for n in sorted(data.keys()):
        if n > 1:
            plot_p50_p99_bar(data, n, plot_dir / f"tsfm_per_task_n{n}.png")

    return 0


if __name__ == "__main__":
    apply_paper_style()
    raise SystemExit(main())
