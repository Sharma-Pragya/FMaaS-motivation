#!/usr/bin/env python3
"""Plot closed-loop interference calibration.

X-axis: T2 TPC count.
Y-axis: delivered RPS (closed-loop, saturated) for T1 (fixed TPCs) and T2.

Run from serving/:
    python experiments/fair_share/tsfm_victim_sweep/calibrate2_plot.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERVING_DIR = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="experiments/fair_share/tsfm_victim_sweep/results/calibration2/calibration2.csv")
    ap.add_argument("--out", default="experiments/fair_share/tsfm_victim_sweep/results/calibration2/calibration2.png")
    args = ap.parse_args()

    csv_path = (SERVING_DIR / args.csv).resolve()
    if not csv_path.exists():
        print(f"[Error] {csv_path} not found")
        return 1

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        print(f"[Error] {csv_path} empty")
        return 1
    rows.sort(key=lambda r: int(r["t2_tpcs"]))
    xs    = [int(r["t2_tpcs"])     for r in rows]
    t1ys  = [float(r["t1_rps"])    for r in rows]
    t2ys  = [float(r["t2_rps"])    for r in rows]
    t1_task = rows[0]["t1_task"]
    t2_task = rows[-1]["t2_task"] or "T2"
    t1_tpcs = rows[0]["t1_tpcs"]

    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    ax.plot(xs, t1ys, marker="o", markersize=3.5, color="#3B7DC4",
            linewidth=1.0, label=f"{t1_task} ({t1_tpcs} TPCs, fixed)")
    ax.plot(xs, t2ys, marker="s", markersize=3.5, color="#E8A24B",
            linewidth=1.0, label=f"{t2_task} (varying TPCs)")

    ax.set_xlabel(f"{t2_task} TPC count")
    ax.set_ylabel("Delivered RPS (closed-loop)")
    ax.set_xticks(xs)
    ax.grid(axis="y", linewidth=0.4, color="#cccccc", linestyle=":")
    ax.legend(frameon=False, loc="best", handlelength=1.6)
    fig.tight_layout(pad=0.3)

    out = (SERVING_DIR / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[Plot] saved {out.with_suffix('.pdf')}")

    print("\nSummary:")
    for r in rows:
        n = int(r["t2_tpcs"])
        t1 = float(r["t1_rps"]); t2 = float(r["t2_rps"])
        if n == 0:
            base_t1 = t1
            print(f"  T2={n:2d} TPCs  →  T1={t1:.2f} RPS  (T1 alone)")
        else:
            drop = (t1 / base_t1 - 1.0) * 100 if base_t1 > 0 else 0.0
            print(f"  T2={n:2d} TPCs  →  T1={t1:.2f} RPS ({drop:+.1f}% vs alone)  T2={t2:.2f} RPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
