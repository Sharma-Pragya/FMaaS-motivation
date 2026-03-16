#!/usr/bin/env python3
"""Plot Motivation Experiment #1 (LLM) results.

Dual y-axis figure:
  Left  axis  (stacked bars) — GPU memory (GB)
      - model memory
      - extra static peak memory (= static_peak - model)
  Right axis  (lines)        — Throughput (req/s)
  X-axis                     — Number of tasks
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


def load_summary(path: Path) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Return {strategy: {n_tasks: metric_dict}}."""
    data: Dict[str, Dict[int, Dict[str, float]]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            strategy = row["strategy"]
            n_tasks = int(row["n_tasks"])
            model_mem_mb = float(row["model_memory_mb"])
            static_peak_mb = float(row["static_peak_gpu_mem_mb"])
            extra_static_mb = max(static_peak_mb - model_mem_mb, 0.0)
            data.setdefault(strategy, {})[n_tasks] = {
                "model_memory_mb": model_mem_mb,
                "extra_static_mb": extra_static_mb,
                "throughput_rps": float(row["throughput_rps"]),
                "avg_latency_ms": float(row["avg_latency_ms"]),
            }
    return data


STRATEGY_LABELS = {
    "task_sharing": "Task Sharing",
    "deploy_sharing": "Deploy Sharing",
}

PAPER_PALETTE = {
    "blue": "#8FB7CF",
    "peach": "#E8B298",
    "charcoal": "#2F3640",
    "slate": "#5C6773",
    "grid": "#D9DEE5",
    "background": "#FAFBFC",
}

COLORS = {
    "task_sharing": PAPER_PALETTE["blue"],
    "deploy_sharing": PAPER_PALETTE["peach"],
}

LINE_COLORS = {
    "task_sharing": "#2E5E7E",
    "deploy_sharing": "#A85C3A",
}

HATCHES = {
    "task_sharing": "",
    "deploy_sharing": "//",
}

LINE_MARKERS = {
    "task_sharing": "o",
    "deploy_sharing": "s",
}

LINE_STYLES = {
    "task_sharing": "-",
    "deploy_sharing": "--",
}

BAR_ALPHA = 0.94
LINE_ALPHA = 0.98
BAR_EDGEWIDTH = 0.9
EXTRA_STATIC_COLOR = "#C5CBD4"
EXTRA_STATIC_ALPHA = 0.28


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


def strategy_handles(strategies: List[str]) -> List[mpatches.Patch]:
    return [
        mpatches.Patch(
            facecolor=COLORS.get(strategy, "gray"),
            edgecolor=PAPER_PALETTE["charcoal"],
            hatch=HATCHES.get(strategy, ""),
            label=STRATEGY_LABELS.get(strategy, strategy),
        )
        for strategy in strategies
    ]


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
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[Plot] Saved: {out_path}")
    print(f"[Plot] Saved: {out_path.with_suffix('.pdf')}")


def make_plot(
    data: Dict[str, Dict[int, Dict[str, float]]],
    out_path: Path,
    strategies: List[str],
    all_n: List[int],
    normalize_throughput: str,
) -> None:
    fig, ax_mem = plt.subplots(figsize=(8.8, 5.0))
    ax_thr = ax_mem.twinx()

    n_strategies = len(strategies)
    bar_width = 0.34
    group_gap = bar_width * n_strategies + 0.18
    x_positions = np.arange(len(all_n)) * (group_gap + 0.08)

    mem_values_all: List[float] = []
    thr_values_all: List[float] = []

    for si, strategy in enumerate(strategies):
        strategy_data = data.get(strategy, {})
        model_vals = [mb_to_gb(strategy_data.get(n, {}).get("model_memory_mb", 0.0)) for n in all_n]
        extra_vals = [mb_to_gb(strategy_data.get(n, {}).get("extra_static_mb", 0.0)) for n in all_n]
        mem_values_all.extend([m + e for m, e in zip(model_vals, extra_vals)])

        offset = (si - (n_strategies - 1) / 2) * bar_width
        x = x_positions + offset

        ax_mem.bar(
            x,
            model_vals,
            width=bar_width,
            color=COLORS.get(strategy, "gray"),
            edgecolor=PAPER_PALETTE["charcoal"],
            linewidth=BAR_EDGEWIDTH,
            hatch=HATCHES.get(strategy, ""),
            alpha=BAR_ALPHA,
            zorder=2,
        )
        ax_mem.bar(
            x,
            extra_vals,
            bottom=model_vals,
            width=bar_width,
            color=EXTRA_STATIC_COLOR,
            edgecolor=PAPER_PALETTE["slate"],
            linewidth=0.8,
            hatch="..",
            alpha=EXTRA_STATIC_ALPHA,
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
            color=LINE_COLORS.get(strategy, PAPER_PALETTE["charcoal"]),
            linewidth=2.8,
            linestyle=LINE_STYLES.get(strategy, "-"),
            marker=LINE_MARKERS.get(strategy, "o"),
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            markeredgecolor=LINE_COLORS.get(strategy, PAPER_PALETTE["charcoal"]),
            alpha=LINE_ALPHA,
            zorder=4,
        )

    ax_mem.set_xlabel("Number of Tasks", fontsize=13, fontweight="semibold")
    ax_mem.set_ylabel("GPU Memory (GB)", fontsize=13, color=PAPER_PALETTE["charcoal"], fontweight="semibold")
    thr_ylabel = "Throughput (req/s)" if normalize_throughput == "none" else "Normalized Throughput"
    ax_thr.set_ylabel(thr_ylabel, fontsize=13, color=PAPER_PALETTE["charcoal"], fontweight="semibold")

    ax_mem.set_xticks(x_positions)
    ax_mem.set_xticklabels([str(n) for n in all_n], fontsize=11)
    ax_mem.tick_params(axis="y", labelsize=11)
    ax_thr.tick_params(axis="y", labelsize=11)

    # End both y-axes at clean rounded values.
    mem_upper = nice_axis_upper(mem_values_all, steps=(1, 1.2, 1.5, 1.6, 2, 2.5, 3, 5, 10), headroom=1.01)
    thr_upper = nice_axis_upper(thr_values_all, steps=(1, 1.2, 1.5, 1.6, 2, 2.5, 3, 5, 10), headroom=1.01)
    ax_mem.set_ylim(0, mem_upper)
    ax_thr.set_ylim(0, thr_upper)

    ax_mem.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    if normalize_throughput == "none":
        ax_thr.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}" if thr_upper >= 20 else f"{v:.1f}"))
    else:
        ax_thr.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))

    ax_mem.grid(axis="y", zorder=0)
    ax_mem.set_axisbelow(True)
    ax_thr.spines["right"].set_visible(True)
    ax_thr.spines["right"].set_color(PAPER_PALETTE["slate"])
    ax_thr.spines["right"].set_linewidth(0.9)

    legend_handles = strategy_handles(strategies) + [
        mpatches.Patch(
            facecolor=EXTRA_STATIC_COLOR,
            edgecolor=PAPER_PALETTE["slate"],
            alpha=EXTRA_STATIC_ALPHA,
            hatch="..",
            label="KV Cache Reserve",
        ),
        mpatches.Patch(
            facecolor="white",
            edgecolor=PAPER_PALETTE["charcoal"],
            label="GPU Memory",
        ),
        Line2D(
            [0], [0],
            color=PAPER_PALETTE["charcoal"],
            linewidth=2.3,
            marker="o",
            label="Throughput",
        ),
    ]
    ax_mem.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.002),
        ncol=len(legend_handles),
        fontsize=8.8,
        frameon=False,
        columnspacing=0.7,
        handletextpad=0.4,
        borderaxespad=0.05,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default=os.environ.get("EXP_DIR", "experiments/motivation1/llm/results"))
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
    make_plot(data, result_root / "motivation1_llm_memory_throughput.png", strategies, all_n, args.normalize_throughput)
    return 0


if __name__ == "__main__":
    apply_paper_style()
    raise SystemExit(main())
