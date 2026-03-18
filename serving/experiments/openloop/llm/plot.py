#!/usr/bin/env python3
"""Plot motivation2/llm results.

Produces:
  1. llm_p50_vs_rps.png       — median p50 latency vs RPS, one line per n_tasks
  2. llm_p99_vs_rps.png       — median p99 latency vs RPS, one line per n_tasks
  3. llm_input_vs_output.png  — input vs output RPS scatter
  4. llm_per_task_n{N}.png    — grouped bar: p50 vs p99 per task for n_tasks > 1

Run from serving/:
    python experiments/motivation2/llm/plot.py
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SERVING_DIR = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PAPER_PALETTE = {
    "charcoal":   "#2F3640",
    "slate":      "#5C6773",
    "grid":       "#D9DEE5",
    "background": "#FAFBFC",
}

N_TASK_COLORS  = {1: "#4878CF", 2: "#6ACC65", 4: "#D65F5F"}
N_TASK_MARKERS = {1: "o", 2: "s", 4: "^"}
N_TASK_LINES   = {1: "-", 2: "--", 4: "-."}

# Task display names + colors for per-task bar charts
TASK_PALETTE = {
    "ag_news":   "#4878CF",
    "sst2":      "#6ACC65",
    "conll2003": "#D65F5F",
    "squad":     "#B47CC7",
}


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
# Data
# ---------------------------------------------------------------------------
def load_summary(path: Path) -> Dict[int, Dict[float, List[dict]]]:
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
    return float(np.median([r[metric] for r in rows]))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_latency(data, metric: str, ylabel: str, title: str, out_path: Path,
                 ms_to_s: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    scale = 1 / 1000 if ms_to_s else 1.0
    unit = "s" if ms_to_s else "ms"

    for n_tasks in sorted(data.keys()):
        rps_data = data[n_tasks]
        rps_vals = sorted(rps_data.keys())
        lat_vals = [aggregate(rps_data[r], metric) * scale for r in rps_vals]

        ax.plot(
            rps_vals, lat_vals,
            color=N_TASK_COLORS.get(n_tasks, "gray"),
            linestyle=N_TASK_LINES.get(n_tasks, "-"),
            marker=N_TASK_MARKERS.get(n_tasks, "o"),
            linewidth=2.2, markersize=6,
            markerfacecolor="white", markeredgewidth=1.5,
            markeredgecolor=N_TASK_COLORS.get(n_tasks, "gray"),
            label=f"{n_tasks} task{'s' if n_tasks > 1 else ''}",
            zorder=3,
        )

    ax.set_xlabel("Target RPS (req/s)", fontsize=12, fontweight="semibold")
    ax.set_ylabel(f"{ylabel} ({unit})", fontsize=12, fontweight="semibold")
    ax.set_title(title, fontsize=12, fontweight="semibold", pad=8)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=False)

    fmt = (lambda v, _: f"{v:.1f}") if ms_to_s else (lambda v, _: f"{v:.0f}")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_input_vs_output(data, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    all_vals = []

    for n_tasks in sorted(data.keys()):
        plotted = False
        for rps, rows in data[n_tasks].items():
            for row in rows:
                inp, out = row["input_rps"], row["output_rps"]
                all_vals += [inp, out]
                ax.scatter(inp, out,
                           color=N_TASK_COLORS.get(n_tasks, "gray"),
                           marker=N_TASK_MARKERS.get(n_tasks, "o"),
                           s=45, zorder=3, alpha=0.85,
                           label=f"{n_tasks} task{'s' if n_tasks > 1 else ''}" if not plotted else "")
                plotted = True

    mx = max(all_vals) * 1.05 if all_vals else 40
    ax.plot([0, mx], [0, mx], color=PAPER_PALETTE["slate"],
            linestyle="--", linewidth=1.2, zorder=1, label="ideal")

    ax.set_xlabel("Input RPS (req/s)", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Output RPS (req/s)", fontsize=12, fontweight="semibold")
    ax.set_title("LLM: Input vs Output Throughput", fontsize=12, fontweight="semibold", pad=8)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    seen: dict = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), fontsize=10, frameon=False)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_per_task_latency(data, n_tasks: int, out_path: Path) -> None:
    """Side-by-side p50/p99 bars per task across RPS — highlights task unfairness."""
    rps_data = data.get(n_tasks)
    if not rps_data:
        return

    all_tasks = sorted({row["task"] for rows in rps_data.values() for row in rows})
    rps_vals = sorted(rps_data.keys())
    n_t = len(all_tasks)
    bar_w = 0.7 / n_t
    x = np.arange(len(rps_vals))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax_idx, (metric, label) in enumerate([
        ("p50_lat_ms", "P50 Latency (ms)"),
        ("p99_lat_ms", "P99 Latency (ms)"),
    ]):
        ax = axes[ax_idx]
        for ti, task in enumerate(all_tasks):
            color = TASK_PALETTE.get(task, f"C{ti}")
            vals = []
            for rps in rps_vals:
                task_rows = [r for r in rps_data[rps] if r["task"] == task]
                vals.append(task_rows[0][metric] if task_rows else 0.0)
            offset = (ti - (n_t - 1) / 2) * bar_w
            ax.bar(x + offset, vals, width=bar_w, color=color, label=task,
                   alpha=0.88, edgecolor=PAPER_PALETTE["charcoal"], linewidth=0.7)

        ax.set_xlabel("Target RPS (req/s)", fontsize=11, fontweight="semibold")
        ax.set_ylabel(label, fontsize=11, fontweight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(r)) for r in rps_vals], fontsize=10)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        if ax_idx == 0:
            ax.legend(fontsize=9, frameon=False)

    fig.suptitle(
        f"LLM latency per task — {n_tasks} tasks sharing backbone  "
        f"(heterogeneous output lengths → unfairness)",
        fontsize=11, fontweight="semibold",
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_saturation_heatmap(data, metric: str, label: str, out_path: Path) -> None:
    """Heatmap: rows=n_tasks, cols=rps, cell=median latency — shows where system saturates."""
    n_task_list = sorted(data.keys())
    all_rps = sorted({r for nd in data.values() for r in nd.keys()})

    matrix = np.full((len(n_task_list), len(all_rps)), np.nan)
    for i, n in enumerate(n_task_list):
        for j, rps in enumerate(all_rps):
            if rps in data[n]:
                matrix[i, j] = aggregate(data[n][rps], metric) / 1000  # → seconds

    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(len(all_rps)))
    ax.set_xticklabels([str(int(r)) for r in all_rps], fontsize=10)
    ax.set_yticks(range(len(n_task_list)))
    ax.set_yticklabels([f"{n} task{'s' if n > 1 else ''}" for n in n_task_list], fontsize=10)
    ax.set_xlabel("Target RPS (req/s)", fontsize=11, fontweight="semibold")
    ax.set_title(f"LLM {label} (s) — saturation heatmap", fontsize=11, fontweight="semibold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(f"{label} (s)", fontsize=10)

    # Annotate cells
    for i in range(len(n_task_list)):
        for j in range(len(all_rps)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=8, color="black" if v < matrix[~np.isnan(matrix)].max() * 0.6 else "white")

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
                        "experiments/motivation2/llm/results"))
    args = parser.parse_args()

    result_root = (SERVING_DIR / args.exp_dir).resolve()
    summary_path = result_root / "summary.csv"
    plot_dir = result_root.parent / "plots"

    if not summary_path.exists():
        print(f"[Error] Not found: {summary_path}")
        return 1

    data = load_summary(summary_path)
    print(f"[Plot] n_tasks={sorted(data.keys())}")

    # Decide whether to use seconds (LLM latencies can be in the seconds range)
    all_p99 = [aggregate(rows, "p99_lat_ms")
               for nd in data.values() for rows in nd.values()]
    use_seconds = max(all_p99) > 2000

    # 1. p50 latency vs RPS
    plot_latency(data, "p50_lat_ms", "P50 Latency",
                 "LLM: Median Latency vs RPS",
                 plot_dir / "llm_p50_vs_rps.png", ms_to_s=use_seconds)

    # 2. p99 latency vs RPS
    plot_latency(data, "p99_lat_ms", "P99 Latency",
                 "LLM: Tail Latency (P99) vs RPS",
                 plot_dir / "llm_p99_vs_rps.png", ms_to_s=use_seconds)

    # 3. Input vs output RPS
    plot_input_vs_output(data, plot_dir / "llm_input_vs_output.png")

    # 4. Per-task latency for multi-task configs
    for n in sorted(data.keys()):
        if n > 1:
            plot_per_task_latency(data, n, plot_dir / f"llm_per_task_n{n}.png")

    # 5. Saturation heatmap
    plot_saturation_heatmap(data, "p50_lat_ms", "P50",
                            plot_dir / "llm_saturation_heatmap.png")

    return 0


if __name__ == "__main__":
    apply_paper_style()
    raise SystemExit(main())
