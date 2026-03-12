#!/usr/bin/env python3
"""Alternative LLM visualization: small multiples (no twin axis).

Panel 1: Stacked GPU memory (GB)
  - model_memory_mb
  - kv cache reserve = static_peak_gpu_mem_mb - model_memory_mb
Panel 2: Throughput (req/s)
Shared x-axis: number of tasks.
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

SERVING_DIR = Path(__file__).resolve().parents[3]

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

LINE_COLORS = {
    "task_sharing": "#2E5E7E",
    "deploy_sharing": "#A85C3A",
}

EXTRA_STATIC_COLOR = "#C5CBD4"
EXTRA_STATIC_ALPHA = 0.28
BAR_ALPHA = 0.94
BAR_EDGEWIDTH = 0.9


def mb_to_gb(value_mb: float) -> float:
    return value_mb / 1024.0


def nice_axis_upper(values: List[float], steps=(1, 1.2, 1.5, 1.6, 2, 2.5, 3, 5, 10), headroom: float = 1.01) -> float:
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


def load_summary(path: Path) -> Dict[str, Dict[int, Dict[str, float]]]:
    data: Dict[str, Dict[int, Dict[str, float]]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            strategy = row["strategy"]
            n_tasks = int(row["n_tasks"])
            model_mem_mb = float(row["model_memory_mb"])
            static_peak_mb = float(row["static_peak_gpu_mem_mb"])
            extra_static_mb = max(static_peak_mb - model_mem_mb, 0.0)
            data.setdefault(strategy, {})[n_tasks] = {
                "model_memory_gb": mb_to_gb(model_mem_mb),
                "kv_cache_reserve_gb": mb_to_gb(extra_static_mb),
                "throughput_rps": float(row["throughput_rps"]),
            }
    return data


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


def make_plot(data: Dict[str, Dict[int, Dict[str, float]]], strategies: List[str], all_n: List[int], out_path: Path) -> None:
    fig, (ax_mem, ax_thr) = plt.subplots(
        2, 1, figsize=(8.8, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.35, 1.0]}
    )

    n_strategies = len(strategies)
    bar_width = 0.34
    group_gap = bar_width * n_strategies + 0.18
    x_positions = np.arange(len(all_n)) * (group_gap + 0.08)

    mem_totals: List[float] = []
    thr_values_all: List[float] = []

    for si, strategy in enumerate(strategies):
        sd = data.get(strategy, {})
        model_vals = [sd.get(n, {}).get("model_memory_gb", 0.0) for n in all_n]
        reserve_vals = [sd.get(n, {}).get("kv_cache_reserve_gb", 0.0) for n in all_n]
        thr_vals = [sd.get(n, {}).get("throughput_rps", float("nan")) for n in all_n]

        mem_totals.extend([m + r for m, r in zip(model_vals, reserve_vals)])
        thr_values_all.extend(thr_vals)

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
            label=STRATEGY_LABELS.get(strategy, strategy),
        )
        ax_mem.bar(
            x,
            reserve_vals,
            width=bar_width,
            bottom=model_vals,
            color=EXTRA_STATIC_COLOR,
            edgecolor=PAPER_PALETTE["slate"],
            linewidth=0.8,
            hatch="..",
            alpha=EXTRA_STATIC_ALPHA,
            zorder=2,
        )

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
            zorder=3,
        )

    ax_mem.set_ylabel("GPU Memory (GB)", fontsize=12, fontweight="semibold")
    ax_thr.set_ylabel("Throughput (req/s)", fontsize=12, fontweight="semibold")
    ax_thr.set_xlabel("Number of Tasks", fontsize=12, fontweight="semibold")

    ax_thr.set_xticks(x_positions)
    ax_thr.set_xticklabels([str(n) for n in all_n], fontsize=11)

    ax_mem.set_ylim(0, nice_axis_upper(mem_totals))
    ax_thr.set_ylim(0, nice_axis_upper(thr_values_all))
    ax_mem.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax_thr.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}" if v >= 20 else f"{v:.1f}"))

    ax_mem.grid(axis="y", zorder=0)
    ax_thr.grid(axis="y", zorder=0)
    ax_mem.set_axisbelow(True)
    ax_thr.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(
            facecolor=COLORS.get(strategy, "gray"),
            edgecolor=PAPER_PALETTE["charcoal"],
            hatch=HATCHES.get(strategy, ""),
            label=STRATEGY_LABELS.get(strategy, strategy),
        )
        for strategy in strategies
    ] + [
        mpatches.Patch(
            facecolor=EXTRA_STATIC_COLOR,
            edgecolor=PAPER_PALETTE["slate"],
            alpha=EXTRA_STATIC_ALPHA,
            hatch="..",
            label="KV Cache Reserve",
        ),
    ]
    ax_mem.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=9.2,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.5,
        borderaxespad=0.05,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default=os.environ.get("EXP_DIR", "experiments/motivation1/llm/results"))
    parser.add_argument("--strategies", default="task_sharing,deploy_sharing")
    args = parser.parse_args()

    result_root = (SERVING_DIR / args.exp_dir).resolve()
    summary_path = result_root / "summary.csv"
    if not summary_path.exists():
        print(f"[Error] summary.csv not found at {summary_path}")
        return 1

    data = load_summary(summary_path)
    strategies = [s.strip() for s in args.strategies.split(",")]
    all_n = sorted({n for sd in data.values() for n in sd})
    if not all_n:
        print("[Error] No data rows found in summary.csv")
        return 1

    out_path = result_root / "motivation1_llm_small_multiples.png"
    make_plot(data, strategies, all_n, out_path)
    return 0


if __name__ == "__main__":
    apply_paper_style()
    raise SystemExit(main())
