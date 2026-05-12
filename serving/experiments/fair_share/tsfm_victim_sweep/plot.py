#!/usr/bin/env python3
"""fair_share/tsfm_victim_sweep — paper-ready sweep plots.

X-axis: victim RPS. One line per method.

Figures:
  fairness.pdf            — Jain's (weighted) fairness index per method.
  system_throughput.pdf   — system throughput (T_v + T_a) per method.
  per_task_throughput.pdf — 2-panel: victim & aggressor delivered RPS,
                            with offered-load dashed reference.

Run from serving/:
    python experiments/fair_share/tsfm_victim_sweep/plot.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERVING_DIR = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Style — matches ../tsfm/plot.py
# ---------------------------------------------------------------------------

COLORS = {
    "fcfs":           "#D68910",
    "no_sharing":     "#888888",
    "no_sharing_tpc": "#6B9AC4",
    "bfq_2_1":        "#E06C75",
}
LABELS = {
    "fcfs":           "Shared-BE",
    "no_sharing":     "BE",
    "no_sharing_tpc": "SP",
    "bfq_2_1":        "FMVisor",
}
LINESTYLES = {
    "fcfs":           "-.",
    "no_sharing":     (0, (3, 1, 1, 1)),
    "no_sharing_tpc": "--",
    "bfq_2_1":        "-",
}
MARKERS = {
    "fcfs":           "s",
    "no_sharing":     "D",
    "no_sharing_tpc": "^",
    "bfq_2_1":        "o",
}
METHOD_ORDER  = ["fcfs", "no_sharing", "no_sharing_tpc", "bfq_2_1"]
# All methods are scored against the operator's intended 2:1 (victim:aggressor)
# target. Methods without a priority knob (fcfs, no_sharing) will naturally
# score lower under contention — that's the comparison.
TARGET_W_V = 2.0
TARGET_W_A = 1.0


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
        "lines.linewidth":    1.0,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.facecolor":  "white",
        "savefig.bbox":       "tight",
    })


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[Plot] saved {out_path.with_suffix('.pdf')}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_records(d: Path, task: str) -> List[Tuple[float, float]]:
    p = d / "latencies.csv"
    if not p.exists():
        return []
    out = []
    with p.open() as f:
        for r in csv.DictReader(f):
            if r.get("task") == task:
                out.append((float(r["elapsed_sec"]), float(r["latency_ms"])))
    return out


def _read_meta(d: Path) -> dict:
    p = d / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _binned_completions(recs: List[Tuple[float, float]],
                        t_start: float, t_end: float,
                        bin_s: float) -> np.ndarray:
    n_bins = max(int(np.ceil((t_end - t_start) / bin_s)), 1)
    counts = np.zeros(n_bins, dtype=float)
    for send_t, lat_ms in recs:
        done = send_t + lat_ms / 1000.0
        if t_start <= done < t_end:
            idx = int((done - t_start) / bin_s)
            if 0 <= idx < n_bins:
                counts[idx] += 1.0
    return counts / bin_s  # rate per second


SATISFIED_TOL = 0.95  # T_i >= SATISFIED_TOL * offered_i counts as "fully satisfied"


def _weighted_maxmin_ideal(d_a: float, d_b: float,
                           w_a: float, w_b: float,
                           capacity: float) -> Tuple[float, float]:
    """Weighted max-min fair allocation for 2 flows.

    Process the flow with smaller demand-per-weight first: if its demand fits
    within its weighted share of capacity, give it full demand and let the
    other flow reclaim the leftover. Otherwise both saturate at their
    weighted shares.
    """
    if capacity <= 0:
        return 0.0, 0.0
    if d_a / max(w_a, 1e-12) <= d_b / max(w_b, 1e-12):
        base_a = w_a * capacity / (w_a + w_b)
        if d_a <= base_a:
            ideal_a = d_a
            ideal_b = min(d_b, capacity - d_a)
        else:
            ideal_a = base_a
            ideal_b = capacity - base_a
    else:
        base_b = w_b * capacity / (w_a + w_b)
        if d_b <= base_b:
            ideal_b = d_b
            ideal_a = min(d_a, capacity - d_b)
        else:
            ideal_b = base_b
            ideal_a = capacity - base_b
    return ideal_a, ideal_b


def minmax_fairness(
    a_recs: List[Tuple[float, float]],
    b_recs: List[Tuple[float, float]],
    offered_a: float, offered_b: float,
    w_a: float, w_b: float,
    t_start: float, t_end: float,
    bin_s: float = 1.0,  # kept for API compat; unused
) -> float:
    """Hybrid fairness over the full post-warmup window.

    Aggregate over [t_start, t_end):
        T_i = completions / window_duration
    Step 1 (satisfaction shortcut):
        if T_a >= τ * offered_a AND T_b >= τ * offered_b:  → f = 1
    Step 2 (weighted max-min ratio):
        C = T_a + T_b   (observed capacity for this method/run)
        (ideal_a, ideal_b) = weighted max-min(demands=(offered_a, offered_b),
                                              weights=(w_a, w_b), capacity=C)
        r_i = min(T_i / ideal_i, 1.0)   # over-delivery doesn't penalize
        f   = min(r_a, r_b) / max(r_a, r_b)

    Range [0, 1]; 1 = method delivered the operator's intended split.
    Aggregating over the whole window (rather than per-bin) eliminates
    Poisson-noise artifacts at low offered rates.
    """
    dur = t_end - t_start
    if dur <= 0 or offered_a <= 0 or offered_b <= 0:
        return float("nan")
    n_a = sum(1 for s, l in a_recs if t_start <= (s + l / 1000.0) < t_end)
    n_b = sum(1 for s, l in b_recs if t_start <= (s + l / 1000.0) < t_end)
    T_a = n_a / dur
    T_b = n_b / dur

    if T_a >= SATISFIED_TOL * offered_a and T_b >= SATISFIED_TOL * offered_b:
        return 1.0

    cap = T_a + T_b
    if cap <= 0 or w_a <= 0 or w_b <= 0:
        return float("nan")
    ideal_a, ideal_b = _weighted_maxmin_ideal(
        offered_a, offered_b, w_a, w_b, cap)
    if ideal_a <= 0 or ideal_b <= 0:
        return float("nan")
    r_a = min(T_a / ideal_a, 1.0)
    r_b = min(T_b / ideal_b, 1.0)
    if r_a <= 0 and r_b <= 0:
        return float("nan")
    return min(r_a, r_b) / max(r_a, r_b)


def mean_throughput(recs: List[Tuple[float, float]],
                    t_start: float, t_end: float) -> float:
    if t_end <= t_start:
        return 0.0
    n = sum(1 for s, l in recs if t_start <= (s + l / 1000.0) < t_end)
    return n / (t_end - t_start)


# ---------------------------------------------------------------------------
# Sweep discovery
# ---------------------------------------------------------------------------

def discover_sweep(base: Path) -> List[Tuple[float, Path]]:
    """Return [(victim_rps, dir), ...] sorted by victim_rps."""
    out = []
    for d in base.iterdir():
        m = re.match(r"^victim_(\d+(?:\.\d+)?)$", d.name)
        if d.is_dir() and m:
            out.append((float(m.group(1)), d))
    out.sort(key=lambda x: x[0])
    return out


def _nice_ceil(value: float) -> float:
    if value <= 0 or not np.isfinite(value):
        return 1.0
    magnitude = 10.0 ** np.floor(np.log10(value))
    fraction = value / magnitude
    for cap in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if fraction <= cap + 1e-9:
            return cap * magnitude
    return 10.0 * magnitude


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _present_methods(victim_dirs: List[Tuple[float, Path]]) -> List[str]:
    seen = set()
    for _, d in victim_dirs:
        for m in METHOD_ORDER:
            if (d / m).exists():
                seen.add(m)
    return [m for m in METHOD_ORDER if m in seen]


def plot_fairness_sweep(
    victim_dirs: List[Tuple[float, Path]],
    victim_task: str, aggressor_task: str,
    out_path: Path, bin_s: float = 1.0,
) -> None:
    methods = _present_methods(victim_dirs)
    fig, ax = plt.subplots(figsize=(3.0, 1.9))
    xs = [v for v, _ in victim_dirs]

    for m in methods:
        ys = []
        for v, d in victim_dirs:
            md = d / m
            if not md.exists():
                ys.append(np.nan)
                continue
            meta = _read_meta(md)
            warmup = float(meta.get("warmup_secs", 2.0))
            t_end  = float(meta.get("duration_s", 10.0))
            t_start = min(warmup, max(0.0, t_end - bin_s))
            offered_v = float(meta.get("victim_rps", v))
            offered_a = float(meta.get("aggressor_rps", 0.0))
            a = _load_records(md, victim_task)
            b = _load_records(md, aggressor_task)
            ys.append(minmax_fairness(
                a, b, offered_v, offered_a,
                TARGET_W_V, TARGET_W_A,
                t_start, t_end, bin_s=bin_s,
            ))
        ax.plot(xs, ys,
                color=COLORS[m], linestyle=LINESTYLES[m], marker=MARKERS[m],
                markersize=3.0, linewidth=1.0, label=LABELS[m], zorder=3)

    ax.set_xlabel("Client A RPS")
    ax.set_ylabel(r"Fairness")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="black", linewidth=0.4, linestyle=":", zorder=2)
    ax.set_xticks(xs)
    ax.grid(axis="y", linewidth=0.4)
    ax.legend(loc="lower left", frameon=False, ncol=2,
              handlelength=1.6, columnspacing=0.9)
    fig.tight_layout(pad=0.3)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_system_throughput_sweep(
    victim_dirs: List[Tuple[float, Path]],
    victim_task: str, aggressor_task: str,
    out_path: Path,
) -> None:
    methods = _present_methods(victim_dirs)
    fig, ax = plt.subplots(figsize=(3.0, 1.9))
    xs = [v for v, _ in victim_dirs]
    ymax = 0.0

    for m in methods:
        ys = []
        for v, d in victim_dirs:
            md = d / m
            if not md.exists():
                ys.append(np.nan)
                continue
            meta = _read_meta(md)
            warmup = float(meta.get("warmup_secs", 2.0))
            t_end  = float(meta.get("duration_s", 10.0))
            a = _load_records(md, victim_task)
            b = _load_records(md, aggressor_task)
            ys.append(mean_throughput(a, warmup, t_end)
                      + mean_throughput(b, warmup, t_end))
        ax.plot(xs, ys,
                color=COLORS[m], linestyle=LINESTYLES[m], marker=MARKERS[m],
                markersize=3.0, linewidth=1.0, label=LABELS[m], zorder=3)
        valid = [y for y in ys if np.isfinite(y)]
        if valid:
            ymax = max(ymax, max(valid))

    ax.set_xlabel("Victim RPS")
    ax.set_ylabel("System throughput (req/s)")
    ax.set_ylim(0, _nice_ceil(ymax * 1.10) if ymax > 0 else 1.0)
    ax.set_xticks(xs)
    ax.grid(axis="y", linewidth=0.4)
    ax.legend(loc="lower right", frameon=False, ncol=2,
              handlelength=1.6, columnspacing=0.9)
    fig.tight_layout(pad=0.3)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_per_task_throughput_sweep(
    victim_dirs: List[Tuple[float, Path]],
    victim_task: str, aggressor_task: str,
    aggressor_rps: float,
    out_path: Path,
) -> None:
    methods = _present_methods(victim_dirs)
    fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(3.0, 3.0), sharex=True)
    xs = [v for v, _ in victim_dirs]

    panels = [
        (ax_v, victim_task,    "Victim",    [v for v, _ in victim_dirs]),
        (ax_a, aggressor_task, "Aggressor", [aggressor_rps] * len(victim_dirs)),
    ]
    panel_max = 0.0
    for ax, task, label, offered in panels:
        for m in methods:
            ys = []
            for v, d in victim_dirs:
                md = d / m
                if not md.exists():
                    ys.append(np.nan)
                    continue
                meta = _read_meta(md)
                warmup = float(meta.get("warmup_secs", 2.0))
                t_end  = float(meta.get("duration_s", 10.0))
                recs = _load_records(md, task)
                ys.append(mean_throughput(recs, warmup, t_end))
            ax.plot(xs, ys,
                    color=COLORS[m], linestyle=LINESTYLES[m], marker=MARKERS[m],
                    markersize=3.0, linewidth=1.0, label=LABELS[m], zorder=3)
            valid = [y for y in ys if np.isfinite(y)]
            if valid:
                panel_max = max(panel_max, max(valid))
        ax.plot(xs, offered, color="black", linestyle=":",
                linewidth=0.8, label="Offered", zorder=2)
        panel_max = max(panel_max, max(offered))
        ax.set_ylabel("Throughput (req/s)")
        ax.text(0.02, 0.93, label, transform=ax.transAxes,
                fontsize=6.5, va="top", ha="left", color="black")
        ax.grid(axis="y", linewidth=0.4)

    y_nice = _nice_ceil(panel_max * 1.10) if panel_max > 0 else 1.0
    for ax in (ax_v, ax_a):
        ax.set_ylim(0, y_nice)
    ax_a.set_xlabel("Victim RPS")
    ax_a.set_xticks(xs)

    handles, leg_labels = ax_v.get_legend_handles_labels()
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.92))
    fig.legend(handles, leg_labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=len(handles),
               frameon=False, handlelength=1.6, columnspacing=0.9)
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-base",
                    default="experiments/fair_share/tsfm_victim_sweep/results")
    ap.add_argument("--plot-dir", default=None)
    ap.add_argument("--victim-task",    default="ecgclass")
    ap.add_argument("--aggressor-task", default="gestureclass")
    ap.add_argument("--aggressor-rps",  type=float, default=50.0)
    ap.add_argument("--bin-size-s",     type=float, default=2.0)
    args = ap.parse_args()

    apply_paper_style()

    base = (SERVING_DIR / args.results_base).resolve()
    if not base.exists():
        print(f"[Error] results dir not found: {base}")
        return 1

    victim_dirs = discover_sweep(base)
    if not victim_dirs:
        print(f"[Error] no victim_<rps> subdirs under {base}")
        return 1
    print(f"[Plot] sweep points: {[v for v, _ in victim_dirs]}")

    plot_dir = (SERVING_DIR / args.plot_dir).resolve() if args.plot_dir \
               else base / "plots"

    plot_fairness_sweep(
        victim_dirs, args.victim_task, args.aggressor_task,
        plot_dir / "fairness.png", bin_s=args.bin_size_s,
    )
    plot_system_throughput_sweep(
        victim_dirs, args.victim_task, args.aggressor_task,
        plot_dir / "system_throughput.png",
    )
    plot_per_task_throughput_sweep(
        victim_dirs, args.victim_task, args.aggressor_task,
        args.aggressor_rps, plot_dir / "per_task_throughput.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
