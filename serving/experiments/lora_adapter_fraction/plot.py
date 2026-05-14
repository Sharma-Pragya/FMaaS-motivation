#!/usr/bin/env python3
"""lora_adapter_fraction/plot.py — Paper-style plot of throughput vs K.

x-axis: number of LoRA-adapted tasks K (out of fixed N total)
y-axis: aggregate closed-loop throughput (req/s)

Companion latency-vs-K plot also produced. Reads results/K{K}/summary.json
files written by run.py / run.sh.

Usage (from serving/):
    python experiments/lora_adapter_fraction/plot.py \
        [--exp-dir experiments/lora_adapter_fraction/results]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SERIES_COLOR = "#E06C75"   # pink-red, matches sharing accent in sibling experiments


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


def _nice_upper(val: float, step: float = 10.0) -> float:
    if val <= 0:
        return step
    return float(np.ceil(val / step) * step)


def load_sweep(exp_dir: Path) -> List[Tuple[int, int, float, float, float, float, float, float]]:
    """[(K, N, tput, avg_all, p95_all, p99_all, avg_adapted, avg_plain), ...]."""
    rows = []
    for sub in sorted(exp_dir.iterdir() if exp_dir.exists() else []):
        if not (sub.is_dir() and sub.name.startswith("K")):
            continue
        try:
            k = int(sub.name[1:])
        except ValueError:
            continue
        summary = sub / "summary.json"
        if not summary.exists():
            print(f"[Warn] {summary} missing — skipping K={k}")
            continue
        with summary.open() as f:
            d = json.load(f)
        rows.append((
            k,
            int(d.get("num_tasks") or 0),
            float(d.get("aggregate_throughput_rps") or 0.0),
            float(d.get("avg_latency_ms_all") or 0.0),
            float(d.get("p95_latency_ms_all") or 0.0),
            float(d.get("p99_latency_ms_all") or 0.0),
            float(d.get("avg_latency_ms_adapted") or 0.0),
            float(d.get("avg_latency_ms_plain")   or 0.0),
        ))
    rows.sort(key=lambda r: r[0])
    return rows


def plot_throughput(rows, out_path: Path, *, backbone: str, base_task: str, n_total: int) -> None:
    ks   = [r[0] for r in rows]
    tput = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    ax.plot(
        ks, tput,
        color=SERIES_COLOR, marker="o", markersize=4,
        markerfacecolor="white", markeredgecolor=SERIES_COLOR, markeredgewidth=1.0,
        linewidth=1.4,
        label=f"{backbone} ({base_task}), N={n_total}",
    )
    ax.set_xlabel(f"# LoRA-adapted tasks $K$ (of $N={n_total}$)")
    ax.set_ylabel("Aggregate throughput (req/s)")
    ax.set_xticks(ks)
    ymax = _nice_upper(max(tput) * 1.10, step=10.0) if tput else 10.0
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {out_path}")


def plot_latency(rows, out_path: Path, *, backbone: str, base_task: str, n_total: int) -> None:
    ks  = [r[0] for r in rows]
    avg = [r[3] for r in rows]
    p99 = [r[5] for r in rows]
    ad  = [r[6] for r in rows]
    pl  = [r[7] for r in rows]

    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    ax.plot(ks, avg, color="#6B9AC4", marker="o", markersize=4, label="avg (all)",
            markerfacecolor="white", markeredgecolor="#6B9AC4", markeredgewidth=1.0)
    ax.plot(ks, p99, color="#E06C75", marker="^", markersize=4, label="p99 (all)",
            markerfacecolor="white", markeredgecolor="#E06C75", markeredgewidth=1.0)
    ax.plot(ks, ad,  color="#E7C98B", marker="s", markersize=4, label="avg (adapted)",
            markerfacecolor="white", markeredgecolor="#E7C98B", markeredgewidth=1.0,
            linestyle="--")
    ax.plot(ks, pl,  color="#888888", marker="d", markersize=4, label="avg (plain)",
            markerfacecolor="white", markeredgecolor="#888888", markeredgewidth=1.0,
            linestyle="--")
    ax.set_xlabel(f"# LoRA-adapted tasks $K$ (of $N={n_total}$)")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(ks)
    flat = [v for v in (*avg, *p99, *ad, *pl) if v > 0]
    ymax = _nice_upper(max(flat) * 1.10, step=50.0) if flat else 50.0
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y")
    ax.legend(loc="best", frameon=False,
              title=f"{backbone} ({base_task})", title_fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir",   default="experiments/lora_adapter_fraction/results")
    p.add_argument("--backbone",  default="momentlarge")
    p.add_argument("--base-task", default="ecgclass")
    args = p.parse_args()

    exp_dir = (SERVING_DIR / args.exp_dir).resolve()
    if not exp_dir.exists():
        print(f"[Error] {exp_dir} not found")
        return 1
    rows = load_sweep(exp_dir)
    if not rows:
        print(f"[Error] No K*/summary.json found under {exp_dir}")
        return 1

    # Pull labels from the first available summary.
    first = exp_dir / f"K{rows[0][0]}" / "summary.json"
    if first.exists():
        with first.open() as f:
            d = json.load(f)
        args.backbone  = d.get("backbone")  or args.backbone
        args.base_task = d.get("base_task") or args.base_task
    n_total = rows[0][1] or 0

    apply_paper_style()
    plot_throughput(rows, exp_dir / "throughput_vs_adapted.pdf",
                    backbone=args.backbone, base_task=args.base_task, n_total=n_total)
    plot_throughput(rows, exp_dir / "throughput_vs_adapted.png",
                    backbone=args.backbone, base_task=args.base_task, n_total=n_total)
    plot_latency(rows, exp_dir / "latency_vs_adapted.pdf",
                 backbone=args.backbone, base_task=args.base_task, n_total=n_total)
    plot_latency(rows, exp_dir / "latency_vs_adapted.png",
                 backbone=args.backbone, base_task=args.base_task, n_total=n_total)

    print()
    print(f"{'K':>3}  {'tput(rps)':>10}  {'avg(ms)':>9}  {'p99(ms)':>9}  "
          f"{'avg_ad':>8}  {'avg_pl':>8}")
    for k, _n, tp, a, _p95, p99, aad, apl in rows:
        print(f"{k:>3}  {tp:>10.2f}  {a:>9.2f}  {p99:>9.2f}  "
              f"{aad:>8.2f}  {apl:>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
