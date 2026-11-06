#!/usr/bin/env python3
"""MemoryVSCompute/plot.py — capacity and limiting-factor plots, one pair per backbone.

Self-contained: everything this needs is defined here. It deliberately does not
import from RTVSntask/vision, because the two experiments no longer agree on
what counts as a passing run — this one requires *complete* delivery
(MIN_DELIVERY = 1.0), while the RTVSntask sweeps allow 1% loss. Sharing the
module would mean one experiment's threshold silently redefining the other's
published numbers. The style constants below are copied rather than imported for
the same reason, and must stay in lockstep with RTVSntask/vision/plot.py so the
bars keep the same colors and labels as every other figure in the paper.

WHY DELIVERY MUST BE COMPLETE
    A run passes only if worst-task mean latency <= threshold AND every offered
    request completed. The delivery gate is not optional: once serving processes
    hit CUDA OOM they shed requests, and the client drops the failed RPCs on the
    floor, so the surviving samples are biased fast — latency alone reports a
    *lower* number as the system degrades. On the A10, no_sharing reports
    164/142/113/75 ms at n = 15/16/17/18 while delivery falls
    100% -> 94% -> 73% -> 33%.

    The gate is set at 1.0 rather than 0.99 because a single OOM is a failure:
    the deployment did not serve the load it was given. Measurement noise cannot
    trip it — the denominator comes from the arrival trace the client actually
    replayed, and across 109 runs the delivered fraction is either exactly 1.0
    (to within 3e-5, an off-by-one at the warmup-trim boundary) or below 0.97.
    The one configuration this reclassifies versus a 0.99 gate is momentlarge on
    the L40S under no_sharing, which delivered 99.4% at n=23: an early OOM that
    the looser gate passed.

Reads:
    results/<backbone>/<device>/ntasks_<N>/rps_<R>/<condition>/task_results.csv

Writes (always into results/plots/):
    results/plots/capacity_vs_hardware_<backbone>.pdf
    results/plots/limiting_factor_<backbone>.pdf

    The limiting-factor grid is a diagnostic, not a paper figure — its content is
    reported in prose in the evaluation. It is still written because it is the
    evidence behind those sentences.

Usage:
    python experiments/MemoryVSCompute/plot.py
    python experiments/MemoryVSCompute/plot.py --backbones swinlarge --threshold 150
    python experiments/MemoryVSCompute/plot.py --min-delivery 0.99   # old, looser gate
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

# Bars are drawn in this order within each hardware group.
METHODS = ["no_sharing_tpc", "no_sharing", "sharing"]

# Copied from RTVSntask/vision/plot.py — keep in lockstep.
SERIES_COLORS = {
    "no_sharing_tpc": "#6B9AC4",   # muted blue
    "no_sharing":     "#888888",   # mid gray
    "sharing":        "#E06C75",   # pink-red
}
SERIES_LABELS = {
    "no_sharing_tpc": "SP",
    "no_sharing":     "BE",
    "sharing":        "FMVisor",
}

# Pretty names for the x axis; devices not listed fall back to upper().
DEVICE_LABELS = {"a10": "A10G", "l4": "L4", "l40": "L40S", "t4": "T4"}

# Weakest to strongest, so the x axis reads as increasing capability.
# Devices not listed are appended alphabetically.
DEVICE_ORDER = ["t4", "a2", "l4", "a10", "l40"]

# Printed size of one capacity panel. Two sit side by side in a single column,
# so each occupies just under half of \columnwidth. The figure is generated at
# this exact size and included with no \includegraphics width key, which is what
# keeps its text at the same printed pt as every other figure in the paper.
CAPACITY_FIG_W = 1.63
CAPACITY_FIG_H = 1.20

# Fraction of offered load that must complete for a run to count. See the module
# docstring — 1.0 means one dropped request fails the configuration.
MIN_DELIVERY = 1.0

# save_results() trims this many warmup seconds before computing task_results.csv,
# so the offered-request denominator has to skip them too.
WARMUP_SECS = 10.0

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


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

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
        "font.size":           8,
        "axes.titlesize":      8,
        "axes.labelsize":      8,
        "xtick.labelsize":     7.5,
        "ytick.labelsize":     7.5,
        "legend.fontsize":     7.5,
        "lines.linewidth":    1.2,
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
    print(f"[Plot] Saved: {out_path}")


def save_figure_exact(fig: plt.Figure, out_path: Path) -> None:
    """Save at exactly the figure's figsize, for panels typeset at a fixed width.

    bbox_inches="tight" crops to the drawn content, so the file comes out wider
    than figsize by however much the tick labels happen to need. Including such a
    file at a fixed \\includegraphics width then rescales the fonts by a factor
    that differs per panel. tight_layout() has already fitted the labels inside
    the canvas, so saving the canvas as-is keeps printed pt == rcParams pt.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(out_path, dpi=300)
    print(f"[Plot] Saved: {out_path} ({fig.get_size_inches()[0]:.2f}in exact)")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

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


def read_run(run_dir: Path, n: int, rps: float, duration: float
             ) -> Optional[Tuple[float, float]]:
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


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_backbones(results_dir: Path) -> List[str]:
    """Backbone directories under results/, excluding the plots output dir."""
    if not results_dir.is_dir():
        return []
    return sorted(d.name for d in results_dir.iterdir()
                  if d.is_dir() and d.name != "plots")


def discover_devices(backbone_dir: Path) -> List[str]:
    """Device directories for one backbone, ordered by rough capability."""
    found = [d.name for d in backbone_dir.iterdir() if d.is_dir()]
    known = [d for d in DEVICE_ORDER if d in found]
    return known + sorted(set(found) - set(known))


def scan(device_dir: Path, rps: float, duration: float
         ) -> Dict[str, Dict[int, Optional[Tuple[float, float]]]]:
    """{condition: {ntasks: (worst_mean_ms, delivery) or None-if-infeasible}}."""
    out: Dict[str, Dict[int, Optional[Tuple[float, float]]]] = {m: {} for m in METHODS}
    for ntasks_dir in device_dir.glob("ntasks_*"):
        m = re.fullmatch(r"ntasks_(\d+)", ntasks_dir.name)
        if not m:
            continue
        n = int(m.group(1))
        for cond in METHODS:
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

def failed(rec: Optional[Tuple[float, float]], threshold: float,
           min_delivery: float) -> bool:
    """Did this configuration miss the SLO or fail to serve the whole load?

    rec is None for an infeasible marker: the run never happened, which is a
    failure at that n, not missing data.

    Delivery is compared with a small tolerance because the denominator is a
    count of trace entries past the warmup cut, and a request scheduled within
    microseconds of that cut can land on either side of it. The slack is ~1e-4
    of the offered load — three orders of magnitude below the smallest real
    shortfall observed (0.6%), so it cannot mask a dropped request.
    """
    if rec is None:
        return True
    worst, delivery = rec
    return worst > threshold or delivery < min_delivery - 1e-4


def capacity(points: Dict[int, Optional[Tuple[float, float]]], threshold: float,
             min_delivery: float = MIN_DELIVERY) -> Tuple[int, bool]:
    """Largest n whose entire prefix passes.

    Returns (capacity, exact). exact is True only when n = capacity+1 was
    actually run and failed, i.e. the crossing was measured. It is False when
    the number is merely a lower bound — either every tested n passed, or the
    next tested n jumped over the crossing and left a gap.

    Taking the prefix rather than the largest passing n keeps post-collapse dips
    (see the docstring's A10 example) from being read as headroom.
    """
    best = 0
    for n in sorted(points):
        if failed(points[n], threshold, min_delivery):
            return best, n == best + 1
        best = n
    return best, False


def limiting_factor(points: Dict[int, Optional[Tuple[float, float]]], threshold: float,
                    min_delivery: float = MIN_DELIVERY
                    ) -> Tuple[str, Optional[int], str]:
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
        drop_fail = delivery < min_delivery - 1e-4
        if lat_fail and drop_fail:
            return "both", n, f"{worst:.0f} ms and only {delivery*100:.1f}% delivered"
        if drop_fail:
            return "memory", n, f"only {delivery*100:.1f}% delivered ({worst:.0f} ms)"
        if lat_fail:
            return "compute", n, f"{worst:.0f} ms at {delivery*100:.0f}% delivered"
    return "none", None, "no tested n failed"


# ---------------------------------------------------------------------------
# Plots
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
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", default=None,
                    help="comma-separated backbones (default: every one under results/)")
    ap.add_argument("--devices", default=None,
                    help="comma-separated devices (default: every one found per backbone)")
    ap.add_argument("--rps", type=float, default=5, help="per-task request rate (default: 5)")
    ap.add_argument("--threshold", type=float, default=200.0,
                    help="mean-latency SLO in ms (default: 200)")
    ap.add_argument("--min-delivery", type=float, default=MIN_DELIVERY,
                    help="fraction of offered load that must complete "
                         f"(default: {MIN_DELIVERY:g} — every request)")
    ap.add_argument("--duration", type=float, default=600.0,
                    help="fallback phase duration when neither trace nor config is readable")
    args = ap.parse_args()

    backbones = ([b.strip() for b in args.backbones.split(",") if b.strip()]
                 if args.backbones else discover_backbones(RESULTS_DIR))
    if not backbones:
        print(f"[Plot] ERROR: no backbone directories under {RESULTS_DIR}", file=sys.stderr)
        return 1

    device_filter = ([d.strip() for d in args.devices.split(",") if d.strip()]
                     if args.devices else None)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    wrote = 0

    for backbone in backbones:
        backbone_dir = RESULTS_DIR / backbone
        if not backbone_dir.is_dir():
            print(f"[Plot] WARNING: no results for backbone '{backbone}' — skipping")
            continue

        devices = discover_devices(backbone_dir)
        if device_filter:
            devices = [d for d in devices if d in device_filter]

        caps: Dict[str, Dict[str, Tuple[int, bool]]] = {}
        limits: Dict[str, Dict[str, Tuple[str, Optional[int], str]]] = {}

        print(f"\n################ {backbone}  (rps={args.rps:g}, "
              f"SLO={args.threshold:g} ms, delivery>={args.min_delivery*100:g}%) "
              f"################")
        for dev in devices:
            points = scan(backbone_dir / dev, args.rps, args.duration)
            if not any(points[m] for m in METHODS):
                print(f"  {DEVICE_LABELS.get(dev, dev.upper())}: "
                      f"no runs at rps={args.rps:g} — skipping")
                continue
            caps[dev] = {m: capacity(points[m], args.threshold, args.min_delivery)
                         for m in METHODS}
            limits[dev] = {m: limiting_factor(points[m], args.threshold, args.min_delivery)
                           for m in METHODS}

            print(f"\n  === {DEVICE_LABELS.get(dev, dev.upper())} ===")
            for m in METHODS:
                cap, exact = caps[dev][m]
                kind, fail_n, why = limits[dev][m]
                above = [x for x in sorted(points[m]) if x > cap]
                if exact:
                    note = ""
                elif not points[m]:
                    note = " (no data)"
                elif not above:
                    note = " (>= sweep max, not saturated)"
                else:
                    note = f" (>= : n={cap+1} untested, next tested n={above[0]})"
                limit_txt = f"limited by {kind}" + (f" at n={fail_n}: {why}" if fail_n else "")
                print(f"    {SERIES_LABELS[m]:<10} capacity={cap}{note}  [{limit_txt}]")
                detail = [f"{n}:INF" if points[m][n] is None
                          else f"{n}:{points[m][n][0]:.0f}ms/{points[m][n][1]*100:.1f}%"
                          for n in sorted(points[m])]
                print(f"      {' '.join(detail) if detail else '(no data)'}")

        if not caps:
            print(f"[Plot] no plottable data for '{backbone}' — skipping figures")
            continue

        plot_capacity(caps, PLOTS_DIR / f"capacity_vs_hardware_{backbone}.pdf", args.threshold)
        plot_limiting_factor(limits, PLOTS_DIR / f"limiting_factor_{backbone}.pdf")
        wrote += 2

    if not wrote:
        print("[Plot] ERROR: no figures produced", file=sys.stderr)
        return 1
    print(f"\n[Plot] {wrote} figure(s) in {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    apply_paper_style()
    raise SystemExit(main())
