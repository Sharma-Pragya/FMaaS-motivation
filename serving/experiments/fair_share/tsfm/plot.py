#!/usr/bin/env python3
"""fair_share/tsfm — minimal paper-ready plots.

For each operator weight ratio (1:1, 2:1, 3:1) we generate three figures:
  fairness_<tag>.pdf   — twin-axis bar chart over phase 2 only:
                         left  axis: fairness = (T_A/w_A) / (T_B/w_B)
                         right axis: system throughput = T_A + T_B (req/s)
  throughput_<tag>.pdf — two-panel throughput timeseries (top: A, bottom: B)
  latency_<tag>.pdf    — two-panel latency timeseries (top: A, bottom: B)

Each figure compares one BFQ variant against the weight-agnostic baselines
(Shared-BE, No-Sharing (BE), No-Sharing (TPC, SP)). Color/font style matches
serving/experiments/sharing_benefit/tpc/plot.py.

Run from serving/:
    python experiments/fair_share/tsfm/plot.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERVING_DIR = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Style — matches sharing_benefit/tpc/plot.py (paper-ready, small figures)
# ---------------------------------------------------------------------------

COLORS = {
    "fcfs":           "#D68910",   # darker amber — Shared-BE (visible on white)
    "no_sharing":     "#888888",   # mid gray — BE
    "no_sharing_tpc": "#6B9AC4",   # muted blue — SP (TPC)
    "bfq":            "#E06C75",   # pink-red — FMVisor (any weight)
}
LABELS = {
    "fcfs":           "Shared-BE",
    "no_sharing":     "BE",
    "no_sharing_tpc": "SP",
    "bfq":            "FMVisor",
}
LINESTYLES = {
    "fcfs":           "-.",
    "no_sharing":     (0, (3, 1, 1, 1)),
    "no_sharing_tpc": "--",
    "bfq":            "-",
}

# Plot order (left-to-right on bar charts; top-to-bottom in legends)
METHOD_ORDER = ["fcfs", "no_sharing", "no_sharing_tpc", "bfq"]


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
# Data loading
# ---------------------------------------------------------------------------

def _read_meta(results_dir: Path) -> dict:
    p = results_dir / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _load_records(results_dir: Path, task: str
                  ) -> List[Tuple[float, float]]:
    """Return list of (send_time_s, latency_ms) for `task`."""
    p = results_dir / "latencies.csv"
    if not p.exists():
        return []
    out: List[Tuple[float, float]] = []
    with p.open() as f:
        for r in csv.DictReader(f):
            if r.get("task") != task:
                continue
            out.append((float(r["elapsed_sec"]), float(r["latency_ms"])))
    return out


def _completions_in_window(recs: List[Tuple[float, float]],
                            t_start: float, t_end: float) -> int:
    """Count records whose completion time (send + latency) is in [t_start, t_end)."""
    n = 0
    for send_t, lat_ms in recs:
        done = send_t + lat_ms / 1000.0
        if t_start <= done < t_end:
            n += 1
    return n


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


def _bin_rate(times: np.ndarray, t_max: float, bin_s: float = 1.0
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Counts/sec in fixed bins."""
    n_bins = int(np.ceil(t_max / bin_s))
    counts = np.zeros(n_bins, dtype=float)
    for t in times:
        idx = int(t / bin_s)
        if 0 <= idx < n_bins:
            counts[idx] += 1.0
    centers = (np.arange(n_bins) + 0.5) * bin_s
    return centers, counts / bin_s


def _bin_mean(times: np.ndarray, vals: np.ndarray, t_max: float,
              bin_s: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Mean of `vals` in fixed bins (NaN where empty)."""
    n_bins = int(np.ceil(t_max / bin_s))
    sums   = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=float)
    for t, v in zip(times, vals):
        idx = int(t / bin_s)
        if 0 <= idx < n_bins:
            sums[idx]   += v
            counts[idx] += 1.0
    means = np.full(n_bins, np.nan, dtype=float)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]
    return (np.arange(n_bins) + 0.5) * bin_s, means


def _phase2_window(meta: dict) -> Tuple[float, float]:
    bounds = meta.get("phase_boundaries_s", [])
    if len(bounds) >= 2:
        return float(bounds[0]), float(bounds[1])
    return 0.0, 0.0


def _add_phase_lines(ax: plt.Axes, meta: dict, t_max: float) -> None:
    for b in meta.get("phase_boundaries_s", [])[:-1]:
        if b < t_max:
            ax.axvline(b, color="black", linewidth=0.4, linestyle=":", zorder=2)


def _nice_ceil(value: float) -> float:
    """Round `value` UP to a 'nice' axis limit.

    Nice set covers integer multipliers commonly seen on paper plots:
    1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10 (each scaled by 10^k). Without 3 in
    the set, 30 would round up to 50; with 3 it stays at 30.
    """
    if value <= 0 or not np.isfinite(value):
        return 1.0
    magnitude = 10.0 ** np.floor(np.log10(value))
    fraction = value / magnitude
    for cap in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if fraction <= cap + 1e-9:
            return cap * magnitude
    return 10.0 * magnitude


def _set_axis_ylim_nice(ax: plt.Axes, data_max: float, headroom: float = 1.05) -> float:
    """Set ax y-limit to a nice ceiling above data_max; return the chosen limit."""
    if data_max <= 0 or not np.isfinite(data_max):
        ax.set_ylim(0, 1)
        return 1.0
    nice = _nice_ceil(data_max * headroom)
    ax.set_ylim(0, nice)
    return nice


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _offered_rate_from_trace(trace_path: Path, task: str,
                             t_start: float, t_end: float) -> float:
    """Mean offered RPS for `task` over [t_start, t_end), from trace.json sends."""
    if not trace_path.exists() or t_end <= t_start:
        return 0.0
    raw = json.loads(trace_path.read_text())
    times = raw.get(task, [])
    n = sum(1 for t in times if t_start <= float(t) < t_end)
    return n / (t_end - t_start)


def plot_fairness_summary(
    method_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_task: str,
    weight_a: float,
    weight_b: float,
    meta: dict,
    out_path: Path,
    bin_s: float = 1.0,
) -> None:
    """Twin-axis bars over phase 2: left=fairness, right=system throughput.

    Fairness = hybrid satisfaction + weighted max-min ratio (see
    `minmax_fairness`); 1 = method delivered the operator's intended split.
    """
    p_start, p_end = _phase2_window(meta)
    p_dur = max(p_end - p_start, 1e-6)

    methods = [m for m in METHOD_ORDER if m in method_dirs]
    if not methods:
        return

    # Offered loads in phase 2 come from the shared trace.json.
    base = next(iter(method_dirs.values())).parent
    trace_path = base / "trace.json"
    offered_a = _offered_rate_from_trace(trace_path, victim_task,    p_start, p_end)
    offered_b = _offered_rate_from_trace(trace_path, aggressor_task, p_start, p_end)

    fairness: List[float] = []
    sys_rps:  List[float] = []
    for m in methods:
        a_recs = _load_records(method_dirs[m], victim_task)
        b_recs = _load_records(method_dirs[m], aggressor_task)
        f = minmax_fairness(a_recs, b_recs,
                            offered_a, offered_b,
                            weight_a, weight_b,
                            p_start, p_end, bin_s=bin_s)
        fairness.append(f if np.isfinite(f) else 0.0)
        T_A = _completions_in_window(a_recs, p_start, p_end) / p_dur
        T_B = _completions_in_window(b_recs, p_start, p_end) / p_dur
        sys_rps.append(T_A + T_B)

    labels = [LABELS[m] for m in methods]
    bar_w  = 0.36
    x      = np.arange(len(methods))

    fig, ax_left = plt.subplots(figsize=(2.7, 1.55))
    ax_right     = ax_left.twinx()
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(True)

    FAIR_COLOR = "#3B7DC4"
    TPUT_COLOR = "#E8A24B"

    ax_left.bar(x - bar_w / 2, fairness, width=bar_w,
                color=FAIR_COLOR, edgecolor="black",
                linewidth=0.4, label="Fairness", zorder=3)
    ax_right.bar(x + bar_w / 2, sys_rps, width=bar_w,
                 color=TPUT_COLOR, edgecolor="black",
                 linewidth=0.4, label="System tput", zorder=3)

    ax_left.set_ylabel(r"Fairness", color=FAIR_COLOR)
    ax_right.set_ylabel("System throughput (req/s)", color=TPUT_COLOR)
    ax_left.tick_params(axis="y", colors=FAIR_COLOR)
    ax_right.tick_params(axis="y", colors=TPUT_COLOR)

    ymax_fair = 1.0
    ymax_tput = _nice_ceil(max(sys_rps, default=1.0) * 1.02)
    ax_left.set_ylim(0, ymax_fair)
    ax_right.set_ylim(0, ymax_tput)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, rotation=18, ha="right")
    ax_left.grid(axis="y", linewidth=0.4)

    for xi, v in zip(x, fairness):
        if v > 0:
            ax_left.text(xi - bar_w / 2, v + ymax_fair * 0.02, f"{v:.2f}",
                         ha="center", va="bottom", fontsize=5.5)
    for xi, v in zip(x, sys_rps):
        if v > 0:
            ax_right.text(xi + bar_w / 2, v + ymax_tput * 0.02, f"{v:.0f}",
                          ha="center", va="bottom", fontsize=5.5)

    fig.tight_layout(pad=0.3)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_throughput_timeseries(
    method_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_task: str,
    meta: dict,
    out_path: Path,
    weight_a: float = 1.0,
    weight_b: float = 1.0,
    bin_s: float = 1.0,
) -> None:
    """Two-panel throughput-vs-time: top=Client A, bottom=Client B.
    Throughput binned by completion time (req/s actually delivered).
    Offered load from trace.json drawn as a thin black dotted line.
    """
    methods = [m for m in METHOD_ORDER if m in method_dirs]
    if not methods:
        return

    bounds = meta.get("phase_boundaries_s", [])
    t_max  = float(bounds[-1]) if bounds else 30.0

    # Load offered load from trace.json (shared across all methods).
    # Fall back to send times from the first method's CSV if trace not found.
    import json as _json
    offered: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    base = next(iter(method_dirs.values())).parent
    trace_path = base / "trace.json"
    if trace_path.exists():
        raw = _json.loads(trace_path.read_text())
        for task, times in raw.items():
            t_arr = np.array(times, dtype=float)
            t_arr = t_arr[t_arr <= t_max]
            if t_arr.size:
                offered[task] = _bin_rate(t_arr, t_max, bin_s=bin_s)
    else:
        first_dir = next(iter(method_dirs.values()))
        for task in (victim_task, aggressor_task):
            recs = _load_records(first_dir, task)
            if recs:
                sends = np.array([s for s, _ in recs])
                offered[task] = _bin_rate(sends, t_max, bin_s=bin_s)

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(2.8, 2.4), sharex=True)
    panels = [(ax_a, victim_task,    f"Client A (w={weight_a:g})"),
              (ax_b, aggressor_task, f"Client B (w={weight_b:g})")]

    panel_max = 0.0
    for ax, task, panel_label in panels:
        for m in methods:
            recs = _load_records(method_dirs[m], task)
            if not recs:
                continue
            done = np.array([s + l / 1000.0 for s, l in recs])
            centers, rps = _bin_rate(done, t_max, bin_s=bin_s)
            ax.plot(centers, rps,
                    color=COLORS[m], linestyle=LINESTYLES[m],
                    linewidth=1.0, label=LABELS[m], zorder=3)
            if rps.size:
                panel_max = max(panel_max, float(rps.max()))
        if task in offered:
            centers_o, rps_o = offered[task]
            ax.plot(centers_o, rps_o, color="black", linestyle=":",
                    linewidth=0.8, label="Offered load", zorder=2)
            panel_max = max(panel_max, float(rps_o.max()))
        _add_phase_lines(ax, meta, t_max)
        ax.set_ylabel("Throughput (req/sec)")
        ax.text(0.02, 0.93, panel_label, transform=ax.transAxes,
                fontsize=6.5, va="top", ha="left", color="black")
        ax.grid(axis="y", linewidth=0.4)

    # Shared y-limit (nice number) so both panels share scale.
    y_nice = _nice_ceil(panel_max * 1.10) if panel_max > 0 else 1.0
    x_nice = _nice_ceil(t_max)
    for ax, _, _ in panels:
        ax.set_xlim(0, x_nice)
        ax.set_ylim(0, y_nice)

    ax_b.set_xlabel("Time (s)")
    # De-duplicate legend entries (both panels add "Offered load").
    handles, leg_labels = ax_a.get_legend_handles_labels()
    seen: Dict[str, int] = {}
    dedup_h, dedup_l = [], []
    for h, l in zip(handles, leg_labels):
        if l not in seen:
            seen[l] = 1
            dedup_h.append(h)
            dedup_l.append(l)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.90))
    fig.legend(dedup_h, dedup_l, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=len(dedup_h),
               frameon=False, handlelength=1.6, columnspacing=0.9)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_latency_timeseries(
    method_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_task: str,
    meta: dict,
    out_path: Path,
    weight_a: float = 1.0,
    weight_b: float = 1.0,
    bin_s: float = 1.0,
) -> None:
    """Two-panel mean-latency-vs-time: top=Client A, bottom=Client B."""
    methods = [m for m in METHOD_ORDER if m in method_dirs]
    if not methods:
        return

    bounds = meta.get("phase_boundaries_s", [])
    t_max  = float(bounds[-1]) if bounds else 30.0

    # Decide units (ms vs s) based on max latency seen across all data.
    cached: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    all_max = 0.0
    for m in methods:
        for task in (victim_task, aggressor_task):
            recs = _load_records(method_dirs[m], task)
            cached[(m, task)] = recs
            if recs:
                all_max = max(all_max, max(l for _, l in recs))
    scale = 1.0 / 1000.0 if all_max > 2000 else 1.0
    unit  = "s" if scale < 1 else "ms"

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(2.8, 2.4), sharex=True)
    panels = [(ax_a, victim_task,    f"Client A (w={weight_a:g})"),
              (ax_b, aggressor_task, f"Client B (w={weight_b:g})")]

    panel_max = 0.0
    for ax, task, panel_label in panels:
        for m in methods:
            recs = cached[(m, task)]
            if not recs:
                continue
            send_times = np.array([s for s, _ in recs])
            lats       = np.array([l * scale for _, l in recs])
            centers, mean_lat = _bin_mean(send_times, lats, t_max, bin_s=bin_s)
            ax.plot(centers, mean_lat,
                    color=COLORS[m], linestyle=LINESTYLES[m],
                    linewidth=1.0, label=LABELS[m], zorder=3)
            valid = mean_lat[~np.isnan(mean_lat)]
            if valid.size:
                panel_max = max(panel_max, float(valid.max()))
        _add_phase_lines(ax, meta, t_max)
        ax.set_ylabel(f"Latency ({unit})")
        ax.text(0.02, 0.93, panel_label, transform=ax.transAxes,
                fontsize=6.5, va="top", ha="left", color="black")
        ax.grid(axis="y", linewidth=0.4)

    y_nice = _nice_ceil(panel_max * 1.10) if panel_max > 0 else 1.0
    x_nice = _nice_ceil(t_max)
    for ax, _, _ in panels:
        ax.set_xlim(0, x_nice)
        ax.set_ylim(0, y_nice)

    ax_b.set_xlabel("Time (s)")
    handles, leg_labels = ax_a.get_legend_handles_labels()
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.90))
    fig.legend(handles, leg_labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=len(handles),
               frameon=False, handlelength=1.6, columnspacing=0.9)
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Per-scenario: (BFQ dir, TPC dir, w_A, w_B, filename tag)
# The TPC dir uses a proportional split matching the weight ratio.
WEIGHT_SCENARIOS = [
    ("bfq_1_1", "no_sharing_tpc_1_1", 1.0, 1.0, "1to1"),
    ("bfq_2_1", "no_sharing_tpc_2_1", 2.0, 1.0, "2to1"),
    ("bfq_3_1", "no_sharing_tpc_3_1", 3.0, 1.0, "3to1"),
]

# Weight-agnostic baselines shared across all scenarios.
# no_sharing_tpc is resolved per-scenario via WEIGHT_SCENARIOS.
BASELINES = ["fcfs", "no_sharing"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-base", default="experiments/fair_share/tsfm/results")
    ap.add_argument("--plot-dir",     default=None,
                    help="Output dir (default: <results-base>/plots)")
    ap.add_argument("--victim-task",    default="ecgclass")
    ap.add_argument("--aggressor-task", default="gestureclass")
    ap.add_argument("--bin-size-s",     type=float, default=2.0)
    args = ap.parse_args()

    apply_paper_style()

    base = (SERVING_DIR / args.results_base).resolve()
    if not base.exists():
        print(f"[Error] results dir not found: {base}")
        return 1
    plot_dir = (SERVING_DIR / args.plot_dir).resolve() if args.plot_dir \
               else base / "plots"

    # Read meta.json from the first available result dir for phase boundaries.
    meta: dict = {}
    for d in base.iterdir():
        if d.is_dir() and (d / "meta.json").exists():
            meta = _read_meta(d)
            if meta:
                break
    if not meta:
        print(f"[Error] no meta.json found under {base}")
        return 1

    for bfq_name, tpc_name, w_a, w_b, tag in WEIGHT_SCENARIOS:
        bfq_dir = base / bfq_name
        if not bfq_dir.exists():
            print(f"[Skip] {bfq_name}: dir not found")
            continue

        method_dirs: Dict[str, Path] = {"bfq": bfq_dir}
        for b in BASELINES:
            d = base / b
            if d.exists():
                method_dirs[b] = d
        tpc_dir = base / tpc_name
        if tpc_dir.exists():
            method_dirs["no_sharing_tpc"] = tpc_dir
        else:
            print(f"[Skip] {tpc_name}: dir not found — TPC bars omitted for {tag}")

        plot_fairness_summary(
            method_dirs, args.victim_task, args.aggressor_task,
            w_a, w_b, meta,
            plot_dir / f"fairness_{tag}.png",
            bin_s=args.bin_size_s,
        )
        plot_throughput_timeseries(
            method_dirs, args.victim_task, args.aggressor_task,
            meta, plot_dir / f"throughput_{tag}.png",
            weight_a=w_a, weight_b=w_b,
            bin_s=args.bin_size_s,
        )
        plot_latency_timeseries(
            method_dirs, args.victim_task, args.aggressor_task,
            meta, plot_dir / f"latency_{tag}.png",
            weight_a=w_a, weight_b=w_b,
            bin_s=args.bin_size_s,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
