#!/usr/bin/env python3
"""Plot calibration curve: max throughput vs TPC count, per task.

Reads results/calibration/calibration.csv (written by calibrate.sh) and
produces calibration.pdf showing delivered RPS as a function of TPC count.
A linear reference line (anchored at the smallest TPC point) is drawn for
comparison — sub-linear curves indicate fixed-overhead-dominated regions
where partition ratio ≠ throughput ratio.

Run from serving/:
    python experiments/fair_share/tsfm_victim_sweep/calibrate_plot.py
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERVING_DIR = Path(__file__).resolve().parents[3]


COLORS = {"ecgclass": "#3B7DC4", "gestureclass": "#E8A24B"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="experiments/fair_share/tsfm_victim_sweep/results/calibration/calibration.csv")
    ap.add_argument("--out", default="experiments/fair_share/tsfm_victim_sweep/results/calibration/calibration.png")
    args = ap.parse_args()

    csv_path = (SERVING_DIR / args.csv).resolve()
    if not csv_path.exists():
        print(f"[Error] {csv_path} not found")
        return 1

    by_task: dict = defaultdict(list)  # task -> [(tpc, rps), ...]
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            by_task[r["task"]].append((int(r["tpc_count"]),
                                        float(r["delivered_rps"])))
    for t in by_task:
        by_task[t].sort()

    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.0, 2.0))

    for task, points in by_task.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = COLORS.get(task, "black")
        ax.plot(xs, ys, marker="o", markersize=3.5,
                color=color, linewidth=1.0, label=task)

        # Linear reference anchored at the smallest TPC count.
        if len(xs) >= 2:
            x0, y0 = xs[0], ys[0]
            ref_y = [y0 * x / x0 for x in xs]
            ax.plot(xs, ref_y, linestyle=":", linewidth=0.7,
                    color=color, label=f"{task} (linear ref)", alpha=0.6)

    ax.set_xlabel("TPC count")
    ax.set_ylabel("Delivered RPS (saturated)")
    ax.grid(axis="y", linewidth=0.4, color="#cccccc", linestyle=":")
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)
    fig.tight_layout(pad=0.3)

    out_path = (SERVING_DIR / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[Plot] saved {out_path.with_suffix('.pdf')}")

    print("\nSummary (delivered RPS):")
    for task, points in by_task.items():
        line = "  " + task + ": " + ", ".join(f"{n}→{r:.1f}" for n, r in points)
        print(line)
        if len(points) >= 2:
            x0, y0 = points[0]
            xN, yN = points[-1]
            actual_ratio = yN / y0
            tpc_ratio = xN / x0
            eff = actual_ratio / tpc_ratio
            print(f"    {xN}/{x0} TPCs: actual {actual_ratio:.2f}× vs linear {tpc_ratio:.0f}× → {eff*100:.0f}% scaling efficiency")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
