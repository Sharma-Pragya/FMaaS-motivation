#!/usr/bin/env python3
"""lora_ntasks_throughput/plot.py — Paper-style plot of throughput vs N.

Reads each results/N{N}/summary.json (produced by run.py / run.sh) and
plots:
    x-axis: number of LoRA-adapted tasks (N)
    y-axis: aggregate closed-loop throughput (req/s)

Style mirrors experiments/sharing_benefit/tsfm/plot.py.

Usage (from serving/):
    python experiments/lora_ntasks_throughput/plot.py \
        [--exp-dir experiments/lora_ntasks_throughput/results]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Paper style (matches experiments/sharing_benefit/tsfm/plot.py)
# ---------------------------------------------------------------------------

SERIES_COLORS = {
    "sharing":            "#E06C75",   # pink-red  — FMVisor (Adapter)
    "sharing_no_adapter": "#4C9AC4",   # teal-blue — FMVisor (No Adapter)
    "no_sharing":         "#888888",   # mid gray  — BE (Adapter)
}
SERIES_LABELS = {
    "sharing":            "FMVisor (Adapter)",
    "sharing_no_adapter": "FMVisor (No Adapter)",
    "no_sharing":         "BE (Adapter)",
}
SERIES_MARKER = {
    "sharing":            "o",
    "sharing_no_adapter": "^",
    "no_sharing":         "s",
}
SERIES_LINESTYLE = {
    "sharing":            "-",
    "sharing_no_adapter": "--",
    "no_sharing":         "-.",
}
# Plot order also fixes the legend order.
SERIES_ORDER = ["sharing_no_adapter", "no_sharing", "sharing"]


def apply_paper_style() -> None:
    """Publication-ready rcParams matching the other paper plots in this repo."""
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     "black",
        "axes.labelcolor":    "black",
        "axes.linewidth":     0.9,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.color":         "#cccccc",
        "grid.linestyle":     ":",
        "grid.linewidth":     0.6,
        "grid.alpha":         1.0,
        "xtick.color":        "black",
        "ytick.color":        "black",
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.width":  0.9,
        "ytick.major.width":  0.9,
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "text.color":         "black",
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          17,
        "axes.titlesize":     18,
        "axes.labelsize":     18,
        "xtick.labelsize":    16,
        "ytick.labelsize":    16,
        "legend.fontsize":    14,
        "legend.frameon":     False,
        "lines.linewidth":    2.4,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.facecolor":  "white",
    })


_NICE_STEPS = (1.0, 2.0, 2.5, 5.0, 10.0)


def _nice_top_and_step(value: float, target_ticks: int = 5,
                        min_ticks: int = 4, max_ticks: int = 7):
    """Return (top, step) where `top = ceil(value/step)*step` lands on a tick
    and the resulting tick count is in [min_ticks, max_ticks]. Among valid
    candidates, the one with the smallest overshoot is chosen."""
    if value <= 0 or not np.isfinite(value):
        return 1.0, 1.0
    magnitude = 10.0 ** np.floor(np.log10(value))
    candidates = []
    for mag in (magnitude * 0.1, magnitude, magnitude * 10):
        for s in _NICE_STEPS:
            step = s * mag
            if step <= 0:
                continue
            n = int(np.ceil(value / step - 1e-9))
            if n < 1:
                continue
            top = n * step
            candidates.append(((top - value) / value, step, top, n))
    if not candidates:
        return value, value / 4.0
    valid = [c for c in candidates if min_ticks - 1 <= c[3] <= max_ticks - 1]
    if not valid:
        valid = candidates
    valid.sort(key=lambda c: (c[0], abs(c[3] - (target_ticks - 1))))
    _, step, top, _ = valid[0]
    return top, step


def _set_y_endpoint(ax, value: float, target_ticks: int = 5) -> float:
    top, step = _nice_top_and_step(value, target_ticks=target_ticks)
    ax.set_ylim(0, top)
    n = int(round(top / step))
    ax.set_yticks([i * step for i in range(n + 1)])
    return top


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_mode_rows(mode_dir: Path) -> List[Tuple[int, float, float, float, float]]:
    rows = []
    if not mode_dir.exists():
        return rows
    for sub in sorted(mode_dir.iterdir()):
        if not (sub.is_dir() and sub.name.startswith("N")):
            continue
        try:
            n = int(sub.name[1:])
        except ValueError:
            continue
        summary = sub / "summary.json"
        if not summary.exists():
            print(f"[Warn] {summary} missing — skipping N={n}")
            continue
        with summary.open() as f:
            d = json.load(f)
        rows.append((
            n,
            float(d.get("aggregate_throughput_rps") or 0.0),
            float(d.get("avg_latency_ms_all") or 0.0),
            float(d.get("p95_latency_ms_all") or 0.0),
            float(d.get("p99_latency_ms_all") or 0.0),
        ))
    rows.sort(key=lambda r: r[0])
    return rows


def load_sweep(exp_dir: Path) -> Dict[str, List[Tuple[int, float, float, float, float]]]:
    """Returns {mode: [(N, tput, avg, p95, p99), ...]} for each mode subdir
    present. Falls back to treating exp_dir itself as the mode dir if no
    {sharing, no_sharing} subdirs are found (back-compat with old runs)."""
    out: Dict[str, List] = {}
    for mode in SERIES_ORDER:
        rows = _load_mode_rows(exp_dir / mode)
        if rows:
            out[mode] = rows
    if not out:
        legacy = _load_mode_rows(exp_dir)
        if legacy:
            out["sharing"] = legacy
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _draw_series(ax, by_mode: Dict[str, List], y_idx: int
                 ) -> Tuple[List[int], List[float]]:
    """Plot one curve per mode in SERIES_ORDER. Returns the unioned (xs, ys)
    for downstream axis-limit calculations."""
    all_ns: List[int] = []
    all_y:  List[float] = []
    for mode in SERIES_ORDER:
        if mode not in by_mode:
            continue
        rows = by_mode[mode]
        ns = [r[0] for r in rows]
        ys = [r[y_idx] for r in rows]
        all_ns.extend(ns)
        all_y.extend(ys)
        color = SERIES_COLORS[mode]
        ax.plot(
            ns, ys,
            color=color,
            marker=SERIES_MARKER[mode],
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.4,
            linestyle=SERIES_LINESTYLE[mode],
            linewidth=2.0,
            label=SERIES_LABELS[mode],
        )
    return all_ns, all_y


def _legend_on_top(ax) -> None:
    """Single-row legend just above the axes — keeps the data area clean.
    The labels are long, so we use a tight column spacing and short handles
    so the row stays inside the figure width."""
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=len(SERIES_ORDER), frameon=False,
        handlelength=1.4, columnspacing=0.9, handletextpad=0.4,
        borderaxespad=0.0,
    )


def plot_throughput(by_mode: Dict[str, List], out_path: Path,
                    *, backbone: str, base_task: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    all_ns, all_y = _draw_series(ax, by_mode, y_idx=1)

    ax.set_xlabel("#Tasks")
    ax.set_ylabel("Throughput (req/s)")
    if all_ns:
        ax.set_xticks(sorted(set(all_ns)))
    if all_y:
        _set_y_endpoint(ax, max(all_y) * 1.08, target_ticks=5)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    _legend_on_top(ax)

    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.13, right=0.98)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[Plot] Saved: {out_path}")


def plot_latency(by_mode: Dict[str, List], out_path: Path,
                 *, backbone: str, base_task: str, percentile: str = "p99") -> None:
    """Side-by-side latency comparison; one line per mode at the chosen percentile."""
    idx = {"avg": 2, "p95": 3, "p99": 4}[percentile]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    all_ns, all_y = _draw_series(ax, by_mode, y_idx=idx)

    ax.set_xlabel("#Tasks")
    pretty = {"avg": "Average", "p95": "P95", "p99": "P99"}[percentile]
    ax.set_ylabel(f"{pretty} latency (ms)")
    if all_ns:
        ax.set_xticks(sorted(set(all_ns)))
    if all_y:
        _set_y_endpoint(ax, max(all_y) * 1.08, target_ticks=5)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    _legend_on_top(ax)

    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.13, right=0.98)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[Plot] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", default="experiments/lora_ntasks_throughput/results")
    p.add_argument("--backbone",  default="momentlarge",
                   help="Label only; data is read from summary.json.")
    p.add_argument("--base-task", default="ecgclass",
                   help="Label only; data is read from summary.json.")
    args = p.parse_args()

    exp_dir = (SERVING_DIR / args.exp_dir).resolve()
    if not exp_dir.exists():
        print(f"[Error] {exp_dir} not found")
        return 1

    by_mode = load_sweep(exp_dir)
    if not by_mode:
        print(f"[Error] No <mode>/N*/summary.json found under {exp_dir}")
        return 1

    # Infer backbone / base_task from the first summary we can find.
    for mode, rows in by_mode.items():
        first_summary = exp_dir / mode / f"N{rows[0][0]}" / "summary.json"
        if not first_summary.exists():
            first_summary = exp_dir / f"N{rows[0][0]}" / "summary.json"  # legacy
        if first_summary.exists():
            with first_summary.open() as f:
                d = json.load(f)
            args.backbone  = d.get("backbone")  or args.backbone
            args.base_task = d.get("base_task") or args.base_task
            break

    apply_paper_style()
    plot_throughput(by_mode, exp_dir / "throughput_vs_ntasks.pdf",
                    backbone=args.backbone, base_task=args.base_task)
    plot_throughput(by_mode, exp_dir / "throughput_vs_ntasks.png",
                    backbone=args.backbone, base_task=args.base_task)
    plot_latency(by_mode, exp_dir / "latency_p99_vs_ntasks.pdf",
                 backbone=args.backbone, base_task=args.base_task, percentile="p99")
    plot_latency(by_mode, exp_dir / "latency_p99_vs_ntasks.png",
                 backbone=args.backbone, base_task=args.base_task, percentile="p99")

    # Print combined table for the paper.
    print()
    print(f"{'mode':>12}  {'N':>4}  {'tput(rps)':>10}  {'avg(ms)':>9}  {'p95(ms)':>9}  {'p99(ms)':>9}")
    for mode in SERIES_ORDER:
        if mode not in by_mode:
            continue
        for n, tp, a, p95, p99 in by_mode[mode]:
            print(f"{mode:>12}  {n:>4}  {tp:>10.2f}  {a:>9.2f}  {p95:>9.2f}  {p99:>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
