#!/usr/bin/env python3
"""RTVSntask/vision/plot_capacity.py — tasks served per GPU under a latency SLO.

One bar group per hardware target, one bar per method. Bar height is the largest
number of concurrent tasks the method sustains while every task stays under the
SLO on mean latency.

Reads results_vision_{device}_swinlarge/ only.

Each run's offered load is taken from its own run_config.json (phase_duration_s,
rps_per_task) rather than assumed, because sweeps on different GPUs were run with
different durations — assuming one value makes a shorter run look like it dropped
half its requests.

A configuration counts as PASSING only if both hold:
  * worst-task mean latency <= threshold, and
  * delivered requests >= MIN_DELIVERY of those offered.

The delivery gate is not optional. Once servers start hitting CUDA OOM they shed
requests, and the client drops failed RPCs on the floor, so the surviving samples
are biased fast — latency alone reports a *lower* number as the system degrades.

Capacity is the largest n whose whole prefix passes, which keeps those post-
collapse dips from being read as headroom.

Usage:
    python experiments/RTVSntask/vision/plot_capacity.py
    python experiments/RTVSntask/vision/plot_capacity.py --devices a10,l4,l40 --rps 5
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot import (
    SERIES_COLORS,
    SERIES_LABELS,
    apply_paper_style,
    save_figure,
    save_figure_exact,
)

VISION_DIR = Path(__file__).resolve().parent

# Bars are drawn in this order within each hardware group.
METHODS = ["no_sharing_tpc", "no_sharing", "sharing"]

# Pretty names for the x axis; devices not listed fall back to upper().
DEVICE_LABELS = {"a10": "A10", "l4": "L4", "l40": "L40S", "t4": "T4"}

# Printed size of one capacity panel. Two sit side by side in a single column,
# so each occupies just under half of \columnwidth. The figure is generated at
# this exact size and included with no \includegraphics width key, which is what
# keeps its text at the same printed pt as every other figure in the paper.
CAPACITY_FIG_W = 1.63
CAPACITY_FIG_H = 1.20

# Fraction of offered load that must actually complete for a run to count.
MIN_DELIVERY = 0.99

# save_results() trims this many warmup seconds before computing task_results.csv,
# so the offered-request denominator has to skip them too.
WARMUP_SECS = 10.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def result_roots(device: str) -> List[Path]:
    """Directories to search for one device, highest precedence first."""
    base = VISION_DIR / f"results_vision_{device}_swinlarge"
    return [base] if base.is_dir() else []


_TRACE_CACHE: Dict[Path, Optional[float]] = {}


def trace_offered(run_dir: Path) -> Optional[float]:
    """Requests the arrival trace actually schedules after the warmup trim.

    run.py caches trace.json per ntasks/rps directory and reuses it across every
    condition there, consulting --duration only when generating a new one. So a
    run invoked with PHASE_DURATION=600 replays a 300 s trace if one already
    exists, while its run_config.json still records 600. The trace is what the
    client really sent, so it — not the config — is the honest denominator.
    """
    path = run_dir.parent / "trace.json"
    if path not in _TRACE_CACHE:
        try:
            trace = json.loads(path.read_text())
            _TRACE_CACHE[path] = float(sum(
                sum(1 for t in times if t > WARMUP_SECS) for times in trace.values()))
        except (OSError, ValueError, TypeError, AttributeError):
            _TRACE_CACHE[path] = None
    return _TRACE_CACHE[path]


def offered_load(run_dir: Path, n: int, rps: float, duration: float) -> float:
    """Requests the client should have sent, after the warmup trim."""
    exact = trace_offered(run_dir)
    if exact:
        return exact
    # No trace on disk — fall back to the config, then to the CLI default.
    cfg = run_dir / "run_config.json"
    if cfg.is_file():
        try:
            c = json.loads(cfg.read_text())
            duration = float(c.get("phase_duration_s") or duration)
            rps = float(c.get("rps_per_task") or rps)
        except (ValueError, TypeError):
            pass
    return n * rps * (duration - WARMUP_SECS)


def read_run(run_dir: Path, n: int, rps: float, duration: float) -> Optional[Tuple[float, float]]:
    """Return (worst_task_mean_latency_ms, delivery_fraction), or None if no data.

    A run marked infeasible never produced an open-loop phase, so it has no
    latency to report — that is a failure, not missing data, and the caller
    distinguishes the two by checking for the marker itself.
    """
    results = run_dir / "task_results.csv"
    if not results.is_file():
        return None
    with results.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    delivered = sum(int(r["n_requests"]) for r in rows)
    offered = offered_load(run_dir, n, rps, duration)
    worst = max(float(r["avg_latency_ms"]) for r in rows)
    return worst, (delivered / offered if offered else 0.0)


def scan_device(device: str, rps: float, duration: float) -> Dict[str, Dict[int, Optional[Tuple[float, float]]]]:
    """{condition: {ntasks: (worst_mean_ms, delivery) or None-if-infeasible}}."""
    out: Dict[str, Dict[int, Optional[Tuple[float, float]]]] = {m: {} for m in METHODS}
    for root in result_roots(device):
        for ntasks_dir in root.glob("ntasks_*"):
            m = re.fullmatch(r"ntasks_(\d+)", ntasks_dir.name)
            if not m:
                continue
            n = int(m.group(1))
            for cond in METHODS:
                if n in out[cond]:      # already filled by a higher-precedence root
                    continue
                run_dir = ntasks_dir / f"rps_{int(rps)}" / cond
                if not run_dir.is_dir():
                    continue
                if (run_dir / "infeasible.json").is_file():
                    out[cond][n] = None
                    continue
                rec = read_run(run_dir, n, rps, duration)
                if rec is not None:
                    out[cond][n] = rec
    return out


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------

def capacity(points: Dict[int, Optional[Tuple[float, float]]], threshold: float) -> Tuple[int, bool]:
    """Largest n whose entire prefix passes.

    Returns (capacity, exact). exact is True only when n = capacity+1 was
    actually run and failed, i.e. the crossing was measured. It is False when
    the number is merely a lower bound — either every tested n passed, or the
    next tested n jumped over the crossing and left a gap.
    """
    best = 0
    for n in sorted(points):
        rec = points[n]
        # rec is None for an infeasible marker: the run never happened, which
        # is a failure at that n, not missing data.
        failed = rec is None or rec[0] > threshold or rec[1] < MIN_DELIVERY
        if failed:
            return best, n == best + 1
        best = n
    return best, False


# ---------------------------------------------------------------------------
# Limiting factor
# ---------------------------------------------------------------------------

# How the first failing configuration fails tells you which resource ran out.
LIMIT_COLORS = {
    "compute": "#E8A33D",   # amber
    "memory":  "#5B8FB9",   # blue
    "both":    "#9B7EBD",   # violet
    "none":    "#DDDDDD",   # never failed within the sweep
    "no data": "#F5F5F5",
}
LIMIT_LABELS = {
    "compute": "Compute",
    "memory":  "Memory",
    "both":    "Both",
    "none":    "not\nreached",
    "no data": "—",
}


def limiting_factor(points: Dict[int, Optional[Tuple[float, float]]],
                    threshold: float) -> Tuple[str, Optional[int], str]:
    """Classify why scaling stops, from the nature of the first failure.

    Dropping requests means the serving processes hit CUDA OOM — either at
    deploy time (infeasible marker) or under load, where failed RPCs are
    swallowed by the client. Delivering everything but too slowly is compute.
    """
    if not points:
        return "no data", None, ""
    for n in sorted(points):
        rec = points[n]
        if rec is None:
            return "memory", n, "backbone copies did not fit (infeasible)"
        worst, delivery = rec
        lat_fail = worst > threshold
        drop_fail = delivery < MIN_DELIVERY
        if lat_fail and drop_fail:
            return "both", n, f"{worst:.0f} ms and only {delivery*100:.0f}% delivered"
        if drop_fail:
            return "memory", n, f"only {delivery*100:.0f}% delivered ({worst:.0f} ms)"
        if lat_fail:
            return "compute", n, f"{worst:.0f} ms at {delivery*100:.0f}% delivered"
    return "none", None, "no tested n failed"


def plot_limiting_factor(limits: Dict[str, Dict[str, Tuple[str, Optional[int], str]]],
                         out_path: Path) -> None:
    """Grid of methods x hardware, coloured by which resource ran out first."""
    devices = list(limits)
    fig, ax = plt.subplots(figsize=(3.2, 1.6))

    for col, dev in enumerate(devices):
        for row, method in enumerate(METHODS):
            kind, n, _ = limits[dev][method]
            ax.add_patch(plt.Rectangle((col, row), 1, 1,
                                       facecolor=LIMIT_COLORS[kind],
                                       edgecolor="white", linewidth=1.2))
            text = LIMIT_LABELS[kind]
            if n is not None:
                text += f"\n(n={n})"
            ax.text(col + 0.5, row + 0.5, text, ha="center", va="center",
                    fontsize=6, color="black", linespacing=1.15)

    ax.set_xlim(0, len(devices))
    ax.set_ylim(0, len(METHODS))
    ax.set_xticks([c + 0.5 for c in range(len(devices))])
    ax.set_xticklabels([DEVICE_LABELS.get(d, d.upper()) for d in devices])
    ax.set_yticks([r + 0.5 for r in range(len(METHODS))])
    ax.set_yticklabels([SERIES_LABELS[m] for m in METHODS])
    ax.tick_params(length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_capacity(caps: Dict[str, Dict[str, Tuple[int, bool]]], out_path: Path,
                  threshold: float) -> None:
    devices = list(caps)
    x = np.arange(len(devices), dtype=float)
    # Group span sets the bar pitch, and the pitch is what keeps the rotated
    # value labels apart: 0.68 gives ~5.0pt of pitch against ~4.2pt of digit
    # ink at 6pt. The 0.85 below leaves a hairline between bars of one
    # device; the 0.32 left over separates one device from the next.
    width = 0.68 / len(METHODS)

    fig, ax = plt.subplots(figsize=(CAPACITY_FIG_W, CAPACITY_FIG_H))

    for i, method in enumerate(METHODS):
        offset = (i - (len(METHODS) - 1) / 2) * width
        vals = [caps[d][method][0] for d in devices]
        # Not exact -> the crossing was never measured; annotate as a lower bound.
        lower_bound = [not caps[d][method][1] for d in devices]
        bars = ax.bar(x + offset, vals, width * 0.85,
                      color=SERIES_COLORS[method], label=SERIES_LABELS[method],
                      edgecolor="black", linewidth=0.4, zorder=3)
        for bar, v, u in zip(bars, vals, lower_bound):
            if v == 0:
                continue
            # No measured crossing above this n — mark it as a lower bound.
            # 12 bars across 1.6in leaves ~0.1in per bar, so horizontal value
            # labels would collide; rotating them keeps every bar annotated.
            ax.annotate(f"{v}≥" if u else f"{v}",
                        (bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 1.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([DEVICE_LABELS.get(d, d.upper()) for d in devices])
    ax.set_ylabel(f"#Tasks \u2264 {threshold:.0f} ms")
    ax.set_axisbelow(True)
    ax.grid(axis="y", zorder=0)
    # Bars rise left to right across every device, so the upper left of the axes
    # is empty. Putting the legend there as a single column reclaims the strip the
    # old above-the-axes legend consumed, and the headroom only has to clear the
    # tallest bar rather than the legend as well.
    ax.set_ylim(0, max(1, max(c[0] for d in caps.values() for c in d.values())) * 1.32)
    ax.legend(frameon=False, ncol=2, loc="upper left", borderaxespad=0.2,
              labelspacing=0.1, handlelength=1.0, handletextpad=0.3,
              columnspacing=0.7, borderpad=0.0)

    fig.tight_layout(pad=0.2)
    save_figure_exact(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", default="a10,l4,l40",
                    help="comma-separated device keys (default: a10,l4,l40)")
    ap.add_argument("--rps", type=float, default=5, help="per-task request rate (default: 5)")
    ap.add_argument("--threshold", type=float, default=200.0,
                    help="mean-latency SLO in ms (default: 200)")
    ap.add_argument("--duration", type=float, default=600.0,
                    help="fallback phase duration when run_config.json is missing")
    ap.add_argument("--out", default=None, help="output pdf (default: <first device>/plots/)")
    args = ap.parse_args()

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    caps: Dict[str, Dict[str, Tuple[int, bool]]] = {}
    limits: Dict[str, Dict[str, Tuple[str, Optional[int], str]]] = {}
    for dev in devices:
        roots = result_roots(dev)
        if not roots:
            print(f"[Plot] WARNING: no results directory for '{dev}' — skipping")
            continue
        points = scan_device(dev, args.rps, args.duration)
        caps[dev] = {m: capacity(points[m], args.threshold) for m in METHODS}
        limits[dev] = {m: limiting_factor(points[m], args.threshold) for m in METHODS}

        print(f"\n=== {DEVICE_LABELS.get(dev, dev.upper())}  (rps={args.rps:g}, "
              f"SLO={args.threshold:g} ms)  roots: {', '.join(r.name for r in roots)}")
        for m in METHODS:
            cap, exact = caps[dev][m]
            detail = []
            for n in sorted(points[m]):
                rec = points[m][n]
                detail.append(f"{n}:INF" if rec is None
                              else f"{n}:{rec[0]:.0f}ms/{rec[1]*100:.0f}%")
            # A capacity is only a measured crossing if n+1 was actually run.
            # Otherwise the bar just marks where the sweep stopped.
            above = [x for x in sorted(points[m]) if x > cap]
            if exact:
                note = ""
            elif not points[m]:
                note = " (no data)"
            elif not above:
                note = " (>= sweep max, not saturated)"
            else:
                note = f" (>= : n={cap+1} untested, next tested n={above[0]})"
            kind, fail_n, why = limits[dev][m]
            limit_txt = f"limited by {kind}" + (f" at n={fail_n}: {why}" if fail_n else "")
            print(f"  {SERIES_LABELS[m]:<10} capacity={cap}{note}  [{limit_txt}]")
            print(f"    {' '.join(detail) if detail else '(no data)'}")

    if not caps:
        print("[Plot] ERROR: no data found for any device", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else (
        result_roots(devices[0])[0] / "plots" / "capacity_vs_hardware.pdf")
    plot_capacity(caps, out, args.threshold)
    plot_limiting_factor(limits, out.with_name("limiting_factor.pdf"))
    return 0


if __name__ == "__main__":
    apply_paper_style()
    raise SystemExit(main())
