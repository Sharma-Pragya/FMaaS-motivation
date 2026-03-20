#!/usr/bin/env python3
"""noisy_neighbor/llm — Plotting script.

Produces:
  1. llm_victim_latency_vs_aggressor_rps.png
       Victim p50 + p99 as aggressor RPS increases.
  2. llm_victim_vs_aggressor_comparison.png
       Side-by-side: victim latency vs aggressor latency across aggressor RPS.

Run from serving/:
    python experiments/noisy_neighbor/llm/plot.py
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SERVING_DIR = Path(__file__).resolve().parents[3]

PAPER_PALETTE = {
    "charcoal":   "#2F3640",
    "slate":      "#5C6773",
    "grid":       "#D9DEE5",
    "background": "#FAFBFC",
}

VICTIM_COLOR    = "#4878CF"
AGGRESSOR_COLOR = "#D65F5F"


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   PAPER_PALETTE["background"],
        "axes.edgecolor":   PAPER_PALETTE["slate"],
        "axes.labelcolor":  PAPER_PALETTE["charcoal"],
        "axes.linewidth":   0.9,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "grid.color":       PAPER_PALETTE["grid"],
        "grid.linestyle":   "--",
        "grid.linewidth":   0.7,
        "grid.alpha":       0.8,
        "xtick.color":      PAPER_PALETTE["charcoal"],
        "ytick.color":      PAPER_PALETTE["charcoal"],
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
        "savefig.facecolor": "white",
        "savefig.bbox":     "tight",
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
    """Returns {aggressor_rps: {role: row}}, meta."""
    data: Dict[float, Dict[str, dict]] = defaultdict(dict)
    meta = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            agg_rps = float(row["aggressor_rps"])
            role    = row["role"]
            data[agg_rps][role] = {
                k: float(v) if k not in ("backbone","victim_task","aggressor_task","role","task") else v
                for k, v in row.items()
            }
            meta["victim_task"]    = row["victim_task"]
            meta["aggressor_task"] = row["aggressor_task"]
            meta["victim_rps"]     = float(row["victim_rps"])
            meta["backbone"]       = row["backbone"]
    return data, meta


def _scale(val_ms: float, use_seconds: bool) -> float:
    return val_ms / 1000.0 if use_seconds else val_ms


def _ylabel(use_seconds: bool) -> str:
    return "Latency (s)" if use_seconds else "Latency (ms)"


def _fmt(use_seconds: bool):
    if use_seconds:
        return lambda v, _: f"{v:.1f}"
    return lambda v, _: f"{v:.0f}"


# ---------------------------------------------------------------------------
# Plot 1: victim latency vs aggressor RPS
# ---------------------------------------------------------------------------

def plot_victim_latency(data, meta, out_path: Path) -> None:
    agg_rps_vals = sorted(data.keys())

    # Auto-detect scale
    max_p99 = max(data[r]["victim"]["p99_lat_ms"] for r in agg_rps_vals)
    use_s   = max_p99 > 2000

    victim_p50 = [_scale(data[r]["victim"]["p50_lat_ms"], use_s) for r in agg_rps_vals]
    victim_p99 = [_scale(data[r]["victim"]["p99_lat_ms"], use_s) for r in agg_rps_vals]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    ax.plot(agg_rps_vals, victim_p50,
            color=VICTIM_COLOR, linewidth=2.2, marker="o", markersize=6,
            markerfacecolor="white", markeredgewidth=1.5, markeredgecolor=VICTIM_COLOR,
            label="Victim P50", zorder=3)
    ax.plot(agg_rps_vals, victim_p99,
            color=VICTIM_COLOR, linewidth=2.2, marker="o", markersize=6,
            markerfacecolor="white", markeredgewidth=1.5, markeredgecolor=VICTIM_COLOR,
            linestyle="--", alpha=0.7,
            label="Victim P99", zorder=3)

    ax.fill_between(agg_rps_vals, victim_p50, victim_p99,
                    color=VICTIM_COLOR, alpha=0.10, zorder=1)

    if 0.0 in data:
        baseline = _scale(data[0.0]["victim"]["p50_lat_ms"], use_s)
        ax.axhline(baseline, color=PAPER_PALETTE["slate"], linewidth=1.0,
                   linestyle=":", zorder=0,
                   label=f"Baseline P50 ({baseline:.1f}{'s' if use_s else 'ms'})")

    ax.set_xlabel("Aggressor RPS (req/s)", fontsize=12, fontweight="semibold")
    ax.set_ylabel(_ylabel(use_s), fontsize=12, fontweight="semibold")
    ax.set_title(
        f"LLM Noisy Neighbor — victim: {meta['victim_task']} @ {meta['victim_rps']:.0f} rps  "
        f"|  aggressor: {meta['aggressor_task']}",
        fontsize=11, fontweight="semibold", pad=8,
    )
    ax.set_xticks(agg_rps_vals)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=False)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt(use_s)))

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: victim vs aggressor side-by-side
# ---------------------------------------------------------------------------

def plot_victim_vs_aggressor(data, meta, out_path: Path) -> None:
    agg_rps_vals = sorted(r for r in data.keys() if r > 0)

    # Auto-detect scale per role separately
    max_v = max(data[r]["victim"]["p99_lat_ms"]     for r in agg_rps_vals)
    max_a = max(data[r]["aggressor"]["p99_lat_ms"]  for r in agg_rps_vals)
    use_s_v = max_v > 2000
    use_s_a = max_a > 2000

    x     = np.arange(len(agg_rps_vals))
    bar_w = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    for ax_idx, (role, color, use_s) in enumerate([
        ("victim",    VICTIM_COLOR,    use_s_v),
        ("aggressor", AGGRESSOR_COLOR, use_s_a),
    ]):
        ax     = axes[ax_idx]
        p50    = [_scale(data[r][role]["p50_lat_ms"], use_s) for r in agg_rps_vals]
        p99    = [_scale(data[r][role]["p99_lat_ms"], use_s) for r in agg_rps_vals]
        label  = f"Victim ({meta['victim_task']})" if role == "victim" else f"Aggressor ({meta['aggressor_task']})"

        ax.bar(x - bar_w/2, p50, width=bar_w, color=color, alpha=0.88,
               edgecolor=PAPER_PALETTE["charcoal"], linewidth=0.7, label="P50")
        ax.bar(x + bar_w/2, p99, width=bar_w, color=color, alpha=0.45,
               edgecolor=PAPER_PALETTE["charcoal"], linewidth=0.7, label="P99", hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels([str(int(r)) for r in agg_rps_vals], fontsize=10)
        ax.set_xlabel("Aggressor RPS (req/s)", fontsize=11, fontweight="semibold")
        ax.set_ylabel(_ylabel(use_s), fontsize=11, fontweight="semibold")
        ax.set_title(label, fontsize=11, fontweight="semibold")
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, frameon=False)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt(use_s)))

    fig.suptitle(
        f"LLM Noisy Neighbor — backbone: {meta['backbone']}  "
        f"victim fixed @ {meta['victim_rps']:.0f} rps",
        fontsize=11, fontweight="semibold",
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: slowdown factor — victim p50 normalized to baseline
# ---------------------------------------------------------------------------

def plot_slowdown(data, meta, out_path: Path) -> None:
    if 0.0 not in data:
        print("[Plot] No baseline (aggressor_rps=0) — skipping slowdown plot")
        return

    baseline_p50 = data[0.0]["victim"]["p50_lat_ms"]
    agg_rps_vals = sorted(r for r in data.keys() if r > 0)
    slowdown     = [data[r]["victim"]["p50_lat_ms"] / baseline_p50 for r in agg_rps_vals]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    ax.bar(range(len(agg_rps_vals)), slowdown,
           color=VICTIM_COLOR, alpha=0.88,
           edgecolor=PAPER_PALETTE["charcoal"], linewidth=0.7)
    ax.axhline(1.0, color=PAPER_PALETTE["slate"], linewidth=1.2,
               linestyle="--", zorder=0, label="No interference (1×)")

    # Annotate bars
    for i, v in enumerate(slowdown):
        ax.text(i, v + 0.02, f"{v:.2f}×", ha="center", va="bottom",
                fontsize=9, color=PAPER_PALETTE["charcoal"])

    ax.set_xticks(range(len(agg_rps_vals)))
    ax.set_xticklabels([str(int(r)) for r in agg_rps_vals], fontsize=10)
    ax.set_xlabel("Aggressor RPS (req/s)", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Victim P50 Slowdown (×)", fontsize=12, fontweight="semibold")
    ax.set_title(
        f"LLM Victim Slowdown — {meta['victim_task']} degraded by {meta['aggressor_task']}",
        fontsize=11, fontweight="semibold", pad=8,
    )
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=False)

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
                        "experiments/noisy_neighbor/llm/results"))
    args = parser.parse_args()

    result_root  = (SERVING_DIR / args.exp_dir).resolve()
    summary_path = result_root / "summary.csv"
    plot_dir     = result_root / "plots"

    if not summary_path.exists():
        print(f"[Error] Not found: {summary_path}")
        return 1

    data, meta = load_summary(summary_path)
    print(f"[Plot] victim={meta['victim_task']}  aggressor={meta['aggressor_task']}  "
          f"aggressor_rps={sorted(data.keys())}")

    plot_victim_latency(data, meta,
                        plot_dir / "llm_victim_latency_vs_aggressor_rps.png")
    plot_victim_vs_aggressor(data, meta,
                             plot_dir / "llm_victim_vs_aggressor.png")
    plot_slowdown(data, meta,
                  plot_dir / "llm_victim_slowdown.png")

    return 0


if __name__ == "__main__":
    apply_paper_style()
    raise SystemExit(main())
