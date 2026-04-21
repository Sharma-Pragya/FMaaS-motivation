#!/usr/bin/env python3
"""colocation/plot.py — Per-RPS box plot of end-to-end response time.

X-axis: Standalone, Co-located
Y-axis: end-to-end latency (ms)
Two boxes at each x position (one per model), legend = model.

Input layout (produced by run.sh):
  <exp-dir>/rps_<R>/single_ecgclass/latencies.csv
  <exp-dir>/rps_<R>/single_nyudepth/latencies.csv
  <exp-dir>/rps_<R>/no_sharing/latencies.csv

Produces one PDF per RPS:
  <exp-dir>/colocation_latency_box_rps<R>.pdf
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TASKS = ["ecgclass", "nyudepth"]
TASK_LABEL = {"ecgclass": "Moment",
              "nyudepth": "DinoV2"}
TASK_COLOR = {"ecgclass": "#6B9AC4", "nyudepth": "#E06C75"}

X_LABELS = ["Standalone", "Co-located"]
COND_FOR_X = {
    "Standalone": {"ecgclass": "single_ecgclass", "nyudepth": "single_nyudepth"},
    "Co-located": {"ecgclass": "no_sharing",      "nyudepth": "no_sharing"},
}

WARMUP_SECS = 10.0


def load_latencies(csv_path: Path, task: str) -> List[float]:
    if not csv_path.exists():
        return []
    vals = []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            if row["task"] != task:
                continue
            try:
                if float(row["elapsed_sec"]) <= WARMUP_SECS:
                    continue
                vals.append(float(row["latency_ms"]))
            except (KeyError, ValueError):
                continue
    return vals


def plot_rps(rps_dir: Path, out_path: Path) -> None:
    data: Dict[str, Dict[str, List[float]]] = {x: {} for x in X_LABELS}
    for x in X_LABELS:
        for task in TASKS:
            cond = COND_FOR_X[x][task]
            lats = load_latencies(rps_dir / cond / "latencies.csv", task)
            data[x][task] = lats

    fig, ax = plt.subplots(figsize=(6, 4))
    n_x = len(X_LABELS)
    n_tasks = len(TASKS)
    width = 0.35
    positions = np.arange(n_x)

    print(f"[plot] {rps_dir.name} — 75th percentile (ms):")
    for i, task in enumerate(TASKS):
        offset = (i - (n_tasks - 1) / 2) * width
        box_positions = positions + offset
        box_data = [data[x][task] if data[x][task] else [np.nan] for x in X_LABELS]
        for x_label, vals in zip(X_LABELS, box_data):
            p75 = float(np.nanpercentile(vals, 75)) if vals else float("nan")
            print(f"  {task:9s} {x_label:11s} p75={p75:.2f}")
        bp = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(TASK_COLOR[task])
            patch.set_alpha(0.75)
        for median in bp["medians"]:
            median.set_color("black")
        # legend proxy
        ax.plot([], [], color=TASK_COLOR[task], linewidth=8,
                alpha=0.75, label=TASK_LABEL[task])

    ax.set_xticks(positions)
    ax.set_xticklabels(X_LABELS, fontsize=16)
    ax.tick_params(axis='y', labelsize=14) 
    ax.set_ylim(0, 250)
    ax.set_ylabel("Processing Latency (ms)",fontsize=16)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="best", frameon=False, fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir",
                        default="experiments/colocation/results",
                        help="Results base (contains rps_<R> subdirs)")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.exists():
        print(f"ERROR: {exp_dir} not found")
        return 1

    rps_dirs = sorted(d for d in exp_dir.iterdir()
                      if d.is_dir() and d.name.startswith("rps_"))
    if not rps_dirs:
        print(f"ERROR: no rps_* subdirs under {exp_dir}")
        return 1

    for rps_dir in rps_dirs:
        rps = rps_dir.name[len("rps_"):]
        out = exp_dir / f"colocation_latency_box_rps{rps}.pdf"
        plot_rps(rps_dir, out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
