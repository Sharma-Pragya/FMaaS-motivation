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
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERVING_DIR = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Style — publication ready
# Palette mirrors serving/experiments/sharing_benefit/{tpc,vision}/plot.py so the
# fair-share and sharing-benefit figures form a visually consistent set in the
# paper.
# ---------------------------------------------------------------------------

COLORS = {
    "fcfs":           "#F0A500",   # amber/orange — S-BE  (alt. sharing baseline)
    "no_sharing":     "#888888",   # mid gray     — BE
    "no_sharing_tpc": "#6B9AC4",   # muted blue   — SP (TPC)
    "stfq":           "#A9C7B5",   # sage green   — S-STFQ (alt. fair sharing)
    "bfq":            "#E06C75",   # pink-red     — FMVisor (proposed)
}

LABELS = {
    "fcfs":           "S-BE",
    "no_sharing":     "BE",
    "no_sharing_tpc": "SP",
    "stfq":           "S-STFQ",
    "bfq":            "FMVisor",
}

LINESTYLES = {
    "fcfs":           (0, (3, 1, 1, 1)),  # dash-dot-dot (matches MPS slot)
    "no_sharing":     "-.",
    "no_sharing_tpc": "--",
    "stfq":           ":",
    "bfq":            "-",
}

MARKERS = {
    "fcfs":           "P",  # plus-filled
    "no_sharing":     "D",  # diamond
    "no_sharing_tpc": "^",  # triangle up
    "stfq":           "s",  # square
    "bfq":            "o",  # circle
}

# Plot order (left-to-right on bar charts; top-to-bottom in legends)
METHOD_ORDER = ["no_sharing", "no_sharing_tpc", "fcfs", "stfq", "bfq"]


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     "black",
        "axes.labelcolor":    "black",
        "axes.linewidth":     0.7,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.color":         "#cccccc",
        "grid.linestyle":     ":",
        "grid.linewidth":     0.5,
        "grid.alpha":         1.0,
        "xtick.color":        "black",
        "ytick.color":        "black",
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.width":  0.7,
        "ytick.major.width":  0.7,
        "xtick.major.size":   3.0,
        "ytick.major.size":   3.0,
        "text.color":         "black",
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        # Larger sizes for academic-paper readability
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "legend.frameon":     False,
        "legend.loc":         "upper center",
        "lines.linewidth":    1.6,
        "lines.markersize":   5,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.facecolor":  "white",
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
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


# Matches lines like: "  Phase 2 (60s): A @ 500 rps, B @ 60 rps"
_CONFIG_PHASE_RE = re.compile(
    r"Phase\s*\d+\s*\([^)]*\):\s*A\s*@\s*([\d.]+)\s*rps,\s*B\s*@\s*([\d.]+)\s*rps",
    re.IGNORECASE,
)


def _parse_offered_loads_from_config(config_path: Path
                                     ) -> Tuple[List[float], List[float]]:
    """Parse per-phase offered RPS for client A and B from results/config.txt.

    Returns (a_rps_per_phase, b_rps_per_phase) — empty lists if file missing
    or malformed.
    """
    if not config_path.exists():
        return [], []
    a_rps: List[float] = []
    b_rps: List[float] = []
    for line in config_path.read_text().splitlines():
        m = _CONFIG_PHASE_RE.search(line)
        if m:
            a_rps.append(float(m.group(1)))
            b_rps.append(float(m.group(2)))
    return a_rps, b_rps


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
    for cap in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
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
    show_left_y: bool = True,
    show_right_y: bool = True,
    tput_ymax: float | None = None,
    row_layout: bool = False,
) -> None:
    """Twin-axis bars over phase 2: left=fairness, right=system throughput.

    For multi-panel row layouts (e.g. fairness plots placed side-by-side in a
    paper), pass ``show_left_y`` / ``show_right_y`` to hide redundant y-axes
    on inner panels, ``tput_ymax`` to share the throughput y-limit across
    panels so bar heights are directly comparable, and ``row_layout=True``
    to use fixed figure margins so every panel has identical outer size
    regardless of which y-axis labels are visible.
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
    bar_w  = 0.38
    x      = np.arange(len(methods))

    # Default sizing pairs visually with ``plot_throughput_timeseries`` so
    # the two figures can be dropped side-by-side in an overleaf
    # two-subfigure layout without awkward height mismatches.
    fig, ax_left = plt.subplots(
        figsize=(3.8, 2.6) if row_layout else (4.0, 3.4))
    ax_right     = ax_left.twinx()
    ax_right.spines["top"].set_visible(False)
    # The right spine stays visible regardless of label visibility — when
    # ``show_right_y`` is False we still want the axis line + tick marks to
    # appear; we just suppress the numeric labels and the axis title below.
    ax_right.spines["right"].set_visible(True)

    # Twin-axis bar colors — chosen for contrast with the method palette
    # used in the timeseries plots (no overlap with COLORS values).
    FAIR_COLOR = "#34495E"   # dark slate blue
    TPUT_COLOR = "#E67E22"   # warm carrot orange

    # Local font sizes — bumped above the global rcParams so the fairness
    # summary stays readable when reduced to ~one-third page width in a
    # multi-panel paper row.
    AXIS_LABEL_FS = 16
    TICK_LABEL_FS = 14
    ANNOT_FS      = 11

    ax_left.bar(x - bar_w / 2, fairness, width=bar_w,
                color=FAIR_COLOR, edgecolor="black",
                linewidth=0.6, label="Fairness", zorder=3)
    ax_right.bar(x + bar_w / 2, sys_rps, width=bar_w,
                 color=TPUT_COLOR, edgecolor="black",
                 linewidth=0.6, label="Throughput", zorder=3)

    # When in a row layout we still draw the y-axis labels and tick labels
    # on the "hidden" sides, but with alpha=0, so they reserve the same
    # horizontal space and ``tight_layout`` + tight-bbox cropping produce
    # identical outer dimensions across all panels in the row.
    hidden_alpha = 0.0 if row_layout else 1.0

    if show_left_y:
        ax_left.set_ylabel("Fairness", color=FAIR_COLOR,
                           fontweight='bold', fontsize=AXIS_LABEL_FS)
        ax_left.tick_params(axis="y", colors=FAIR_COLOR,
                            labelsize=TICK_LABEL_FS)
    else:
        # Keep the spine and tick marks (so the axis stays visible), just drop
        # the numeric tick labels and the axis title. In row layouts we draw
        # them invisibly to reserve the same space as the labeled panels.
        ax_left.set_ylabel("Fairness", color=FAIR_COLOR,
                           fontweight='bold', fontsize=AXIS_LABEL_FS,
                           alpha=hidden_alpha)
        ax_left.tick_params(axis="y", colors=FAIR_COLOR,
                            labelsize=TICK_LABEL_FS,
                            labelcolor=(0, 0, 0, hidden_alpha))

    if show_right_y:
        ax_right.set_ylabel("Throughput (rps)", color=TPUT_COLOR,
                            fontweight='bold', fontsize=AXIS_LABEL_FS)
        ax_right.tick_params(axis="y", colors=TPUT_COLOR,
                             labelsize=TICK_LABEL_FS)
    else:
        ax_right.set_ylabel("Throughput (rps)", color=TPUT_COLOR,
                            fontweight='bold', fontsize=AXIS_LABEL_FS,
                            alpha=hidden_alpha)
        ax_right.tick_params(axis="y", colors=TPUT_COLOR,
                             labelsize=TICK_LABEL_FS,
                             labelcolor=(0, 0, 0, hidden_alpha))

    # y-limits: fairness is nominally [0,1]; throughput uses a nice ceiling of
    # the observed max. A little extra headroom above bar tops keeps rotated
    # value labels inside the panel when the PDF is scaled in Overleaf.
    sys_rps_top = _nice_ceil(max(sys_rps, default=1.0))
    ymax_fair = 1.0
    ymax_tput = tput_ymax if tput_ymax is not None else sys_rps_top
    # Headroom above bar tops so rotated value labels (offset in points) do
    # not paint outside the axes after tight_layout / PDF embedding.
    fair_y_top = min(ymax_fair * 1.08, 1.12)
    tput_y_top = ymax_tput * 1.06
    ax_left.set_ylim(0, fair_y_top)
    ax_right.set_ylim(0, tput_y_top)
    ax_left.set_yticks(np.linspace(0, ymax_fair, 5))
    ax_right.set_yticks(np.linspace(0, sys_rps_top, 5))
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, rotation=30, ha="right",
                            fontsize=TICK_LABEL_FS)
    ax_left.grid(axis="y", zorder=0)

    # Vertical labels: anchor at bar top, nudge upward in *points* so
    # ``ylim`` can stay at the logical max without empty data space above.
    for xi, v in zip(x, fairness):
        if v > 0:
            ax_left.annotate(
                f"{v:.2f}",
                xy=(xi - bar_w / 2, v),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=ANNOT_FS, rotation=90, color=FAIR_COLOR,
            )
    for xi, v in zip(x, sys_rps):
        if v > 0:
            ax_right.annotate(
                f"{v:.0f}",
                xy=(xi + bar_w / 2, v),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=ANNOT_FS, rotation=90, color=TPUT_COLOR,
            )

    fig.tight_layout(pad=0.2)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_fairness_row(
    scenarios: List[Tuple[str, Dict[str, Path], float, float]],
    victim_task: str,
    aggressor_task: str,
    meta: dict,
    out_path: Path,
    bin_s: float = 1.0,
) -> None:
    """Combined single-figure row of fairness summary panels.

    Produces one figure with N twin-axis bar subplots side-by-side, sharing
    both y-scales. The leftmost panel carries the "Fairness" y-axis labels
    and ticks, the rightmost carries the "Throughput (rps)" labels and ticks
    on its right side, and inner panels show only bars + value annotations.
    Each panel is titled with its weight ratio (e.g. "wA:wB = 1:1"), making
    this figure self-contained — ready to drop into an overleaf `figure`
    environment with a single caption.

    `scenarios` is a list of ``(tag, method_dirs, w_a, w_b)`` tuples in the
    order they should appear left-to-right.
    """
    if not scenarios:
        return

    p_start, p_end = _phase2_window(meta)
    p_dur = max(p_end - p_start, 1e-6)

    # Pre-compute fairness and per-method throughput for every panel so we
    # can pick a shared throughput y-limit that fits all bars.
    panel_data: List[Tuple[str, float, float, List[str], List[float], List[float]]] = []
    sys_rps_max = 0.0
    for tag, method_dirs, w_a, w_b in scenarios:
        methods = [m for m in METHOD_ORDER if m in method_dirs]
        if not methods:
            continue
        base = next(iter(method_dirs.values())).parent
        trace_path = base / "trace.json"
        offered_a = _offered_rate_from_trace(trace_path, victim_task,    p_start, p_end)
        offered_b = _offered_rate_from_trace(trace_path, aggressor_task, p_start, p_end)

        fairness: List[float] = []
        sys_rps:  List[float] = []
        for m in methods:
            a_recs = _load_records(method_dirs[m], victim_task)
            b_recs = _load_records(method_dirs[m], aggressor_task)
            f = minmax_fairness(a_recs, b_recs, offered_a, offered_b,
                                w_a, w_b, p_start, p_end, bin_s=bin_s)
            fairness.append(f if np.isfinite(f) else 0.0)
            T_A = _completions_in_window(a_recs, p_start, p_end) / p_dur
            T_B = _completions_in_window(b_recs, p_start, p_end) / p_dur
            sys_rps.append(T_A + T_B)

        labels = [LABELS[m] for m in methods]
        panel_data.append((tag, w_a, w_b, labels, fairness, sys_rps))
        if sys_rps:
            sys_rps_max = max(sys_rps_max, max(sys_rps))

    if not panel_data:
        return

    # Shared y-limits; small headroom above bar tops for annotations (see
    # ``plot_fairness_summary``).
    sys_rps_top = _nice_ceil(sys_rps_max) if sys_rps_max > 0 else 1.0
    ymax_fair = 1.0
    ymax_tput = sys_rps_top

    FAIR_COLOR    = "#34495E"
    TPUT_COLOR    = "#E67E22"
    AXIS_LABEL_FS = 16
    TICK_LABEL_FS = 14
    ANNOT_FS      = 10
    TITLE_FS      = 15
    bar_w         = 0.38

    n = len(panel_data)
    # First panel reserves space for the left y-axis label, last panel for
    # the right y-axis label; inner panels are narrower.
    panel_inch = 2.3
    edge_pad   = 0.6  # extra width on the labeled outer panels
    fig_w = panel_inch * n + 2 * edge_pad
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 2.7), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    if n == 1:
        axes = [axes]

    for i, (ax_left, (tag, w_a, w_b, labels, fairness, sys_rps)) in enumerate(
            zip(axes, panel_data)):
        x = np.arange(len(labels))
        ax_right = ax_left.twinx()

        ax_left.spines["top"].set_visible(False)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(True)

        is_first = i == 0
        is_last  = i == n - 1

        ax_left.bar(x - bar_w / 2, fairness, width=bar_w,
                    color=FAIR_COLOR, edgecolor="black",
                    linewidth=0.6, zorder=3)
        ax_right.bar(x + bar_w / 2, sys_rps, width=bar_w,
                     color=TPUT_COLOR, edgecolor="black",
                     linewidth=0.6, zorder=3)

        fair_y_top = min(ymax_fair * 1.08, 1.12)
        tput_y_top = ymax_tput * 1.06
        ax_left.set_ylim(0, fair_y_top)
        ax_right.set_ylim(0, tput_y_top)
        ax_left.set_yticks(np.linspace(0, ymax_fair, 5))
        ax_right.set_yticks(np.linspace(0, sys_rps_top, 5))
        ax_left.set_xticks(x)
        ax_left.set_xticklabels(labels, rotation=30, ha="right",
                                fontsize=TICK_LABEL_FS)
        ax_left.grid(axis="y", zorder=0)

        # Per-panel weight-ratio title.
        ratio_str = (f"{w_a:g}:{w_b:g}")
        ax_left.set_title(rf"$w_A:w_B = {ratio_str}$",
                          fontsize=TITLE_FS, pad=10)

        # Left y-axis: label + ticks only on the first panel.
        if is_first:
            ax_left.set_ylabel("Fairness", color=FAIR_COLOR,
                               fontweight='bold', fontsize=AXIS_LABEL_FS)
            ax_left.tick_params(axis="y", colors=FAIR_COLOR,
                                labelsize=TICK_LABEL_FS)
        else:
            ax_left.tick_params(axis="y", colors=FAIR_COLOR,
                                labelsize=TICK_LABEL_FS, labelleft=False)

        # Right y-axis: label + ticks only on the last panel.
        if is_last:
            ax_right.set_ylabel("Throughput (rps)", color=TPUT_COLOR,
                                fontweight='bold', fontsize=AXIS_LABEL_FS)
            ax_right.tick_params(axis="y", colors=TPUT_COLOR,
                                 labelsize=TICK_LABEL_FS)
        else:
            ax_right.tick_params(axis="y", colors=TPUT_COLOR,
                                 labelsize=TICK_LABEL_FS, labelright=False)

        # Vertical labels: point offset from bar top (no extra ylim headroom).
        for xi, v in zip(x, fairness):
            if v > 0:
                ax_left.annotate(
                    f"{v:.2f}",
                    xy=(xi - bar_w / 2, v),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=ANNOT_FS, rotation=90, color=FAIR_COLOR,
                )
        for xi, v in zip(x, sys_rps):
            if v > 0:
                ax_right.annotate(
                    f"{v:.0f}",
                    xy=(xi + bar_w / 2, v),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=ANNOT_FS, rotation=90, color=TPUT_COLOR,
                )

    fig.tight_layout(pad=0.3, w_pad=0.6)
    save_figure(fig, out_path)
    plt.close(fig)


def _annotate_offered_loads(ax: plt.Axes,
                             boundaries: List[float],
                             rps_list: List[float],
                             t_max: float,
                             fontsize: float | None = None) -> None:
    """Render per-phase offered-load labels just above the axes.

    Places one centered "<rate> rps" label per phase in axes-fraction coords
    at y slightly above 1.0. This sits outside the data area so it never
    collides with in-panel annotations such as the Client A/B label.

    ``t_max`` should be the x-axis upper limit currently displayed (not the
    raw phase end), so that the fractional positions line up with the actual
    visible time range — important when the x-axis has been extended beyond
    the last phase boundary (e.g. via ``x_max_s``).
    """
    if not boundaries or not rps_list:
        return
    fs = fontsize if fontsize is not None else plt.rcParams["xtick.labelsize"]
    bounds = [0.0] + [float(b) for b in boundaries]
    n = min(len(bounds) - 1, len(rps_list))
    for i in range(n):
        t_s, t_e = bounds[i], bounds[i + 1]
        if t_s >= t_max:
            break
        t_e = min(t_e, t_max)
        mid = (t_s + t_e) / 2.0
        x_frac = mid / t_max if t_max > 0 else 0.5
        ax.text(x_frac, 1.04, f"{rps_list[i]:.0f} rps",
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=fs,
                color="#333333", fontweight="bold", zorder=10,
                clip_on=False)


def plot_throughput_timeseries(
    method_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_task: str,
    meta: dict,
    out_path: Path,
    weight_a: float = 1.0,
    weight_b: float = 1.0,
    bin_s: float = 1.0,
    x_max_s: float | None = None,
) -> None:
    """Two-panel throughput-vs-time.

    ``x_max_s`` (optional) forces the x-axis upper limit to a fixed value
    in seconds — useful when comparing experiments with different phase
    durations and you want a consistent on-page width per second of data.
    Defaults to ``None`` which falls back to a nice ceiling above the last
    phase boundary.
    """
    # Only show SP, S-STFQ, and FMVisor as requested
    ALLOWED_METHODS = ["no_sharing_tpc", "stfq", "bfq"]
    methods = [m for m in METHOD_ORDER if m in method_dirs and m in ALLOWED_METHODS]
    if not methods:
        return

    bounds = meta.get("phase_boundaries_s", [])
    t_max  = float(bounds[-1]) if bounds else 30.0

    # Local font sizes — bumped above the global rcParams so labels, tick
    # values, the legend, the in-panel Client label and the offered-load
    # annotations all stay legible when the figure is reduced to
    # column-width in a paper.
    AXIS_LABEL_FS  = 16
    TICK_LABEL_FS  = 14
    LEGEND_FS      = 14
    PANEL_LABEL_FS = 14
    OFFERED_FS     = 13

    # Per-phase offered RPS comes from the experiment's config.txt (one file
    # per results-base, shared across all method runs). Falls back to meta.json
    # entries when config.txt is unavailable.
    base = next(iter(method_dirs.values())).parent
    config_path = base / "config.txt"
    cfg_a_rps, cfg_b_rps = _parse_offered_loads_from_config(config_path)
    if not cfg_a_rps:
        cfg_a_rps = meta.get("victim_rps_phases", [])
    if not cfg_b_rps:
        cfg_b_rps = meta.get("aggressor_rps_phases", [])

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(4.8, 4.2), sharex=True)
    panels = [(ax_a, victim_task,    f"Client A (w={weight_a:g})", cfg_a_rps),
              (ax_b, aggressor_task, f"Client B (w={weight_b:g})", cfg_b_rps)]

    # Decide the displayed x-axis upper limit up-front so offered-load
    # annotations position correctly relative to the visible time range.
    x_lim = float(x_max_s) if x_max_s is not None else _nice_ceil(t_max)

    panel_max = 0.0
    for ax, task, panel_label, rps_list in panels:
        for m in methods:
            recs = _load_records(method_dirs[m], task)
            if not recs:
                continue
            done = np.array([s + l / 1000.0 for s, l in recs])
            centers, rps = _bin_rate(done, t_max, bin_s=bin_s)

            ax.plot(centers, rps,
                    color=COLORS[m], linestyle=LINESTYLES[m],
                    marker=MARKERS[m], markevery=max(1, len(centers)//8),
                    markersize=plt.rcParams["lines.markersize"],
                    linewidth=plt.rcParams["lines.linewidth"],
                    label=LABELS[m], zorder=3)
            if rps.size:
                panel_max = max(panel_max, float(rps.max()))

        # Per-phase offered-load annotation, placed above the axes so it never
        # collides with the in-panel Client label or the data.
        _annotate_offered_loads(ax, meta.get("phase_boundaries_s", []),
                                rps_list, x_lim, fontsize=OFFERED_FS)

        _add_phase_lines(ax, meta, x_lim)
        # Single shared y-axis label (added via fig.supylabel below) is
        # cleaner than per-panel labels at this font size — the rotated
        # 16pt label is taller than each panel.
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FS)
        # Client label stays inside the panel, top-right, well below the
        # offered-load annotations which now live above the axes.
        ax.text(0.98, 0.92, panel_label, transform=ax.transAxes,
                fontweight='bold', va="top", ha="right",
                fontsize=PANEL_LABEL_FS,
                bbox=dict(facecolor='white', alpha=0.85,
                          edgecolor='none', pad=1.5))
        ax.grid(axis="y")

    y_nice = _nice_ceil(panel_max * 1.05) if panel_max > 0 else 1.0
    for ax, _, _, _ in panels:
        ax.set_xlim(0, x_lim)
        ax.set_ylim(0, y_nice)

    ax_b.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FS)

    handles, leg_labels = ax_a.get_legend_handles_labels()
    dedup_h, dedup_l = [], []
    seen = set()
    for h, l in zip(handles, leg_labels):
        if l not in seen:
            seen.add(l)
            dedup_h.append(h)
            dedup_l.append(l)

    # Reserve space at the top for the legend and the per-phase offered-load
    # annotations that sit above each panel. The extra left margin makes
    # room for the shared rotated y-axis label. ``h_pad`` controls the gap
    # between Client A and Client B — kept just large enough to fit the
    # offered-load labels above panel B without crowding panel A.
    fig.tight_layout(rect=(0.05, 0, 1, 0.90), pad=0.3, h_pad=1.4)
    # Single shared y-axis label spanning both panels — done via fig.text
    # rather than fig.supylabel for compatibility with older matplotlib.
    fig.text(0.01, 0.45, "Throughput (RPS)",
             fontsize=AXIS_LABEL_FS, fontweight='bold',
             rotation=90, va="center", ha="left")
    fig.legend(dedup_h, dedup_l, loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=len(dedup_h),
               frameon=False, handlelength=2.0, columnspacing=1.0,
               fontsize=LEGEND_FS)

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
    """Two-panel mean-latency-vs-time."""
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

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.8, 3.6), sharex=True)
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
                    marker=MARKERS[m], markevery=max(1, len(centers)//8),
                    markersize=plt.rcParams["lines.markersize"],
                    linewidth=plt.rcParams["lines.linewidth"],
                    label=LABELS[m], zorder=3)
            valid = mean_lat[~np.isnan(mean_lat)]
            if valid.size:
                panel_max = max(panel_max, float(valid.max()))

        _add_phase_lines(ax, meta, t_max)
        ax.set_ylabel(f"Latency ({unit})")
        ax.text(0.02, 0.95, panel_label, transform=ax.transAxes,
                fontweight='bold', va="top", ha="left",
                bbox=dict(facecolor='white', alpha=0.85,
                          edgecolor='none', pad=1.5))
        ax.grid(axis="y")

    y_nice = _nice_ceil(panel_max * 1.1) if panel_max > 0 else 1.0
    x_nice = _nice_ceil(t_max)
    for ax, _, _ in panels:
        ax.set_xlim(0, x_nice)
        ax.set_ylim(0, y_nice)

    ax_b.set_xlabel("Time (s)")

    handles, leg_labels = ax_a.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.3, h_pad=1.2)
    fig.legend(handles, leg_labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=len(handles),
               frameon=False, handlelength=2.0, columnspacing=1.0)

    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Per-scenario: (BFQ dir, TPC dir, w_A, w_B, filename tag)
# The TPC dir uses a proportional split matching the weight ratio.
WEIGHT_SCENARIOS = [
    ("bfq_1_1", "no_sharing_tpc_1_1", "stfq_1_1", 1.0, 1.0, "1to1"),
    ("bfq_2_1", "no_sharing_tpc_2_1", "stfq_1_2", 2.0, 1.0, "2to1"),
    ("bfq_3_1", "no_sharing_tpc_3_1", "stfq_1_3", 3.0, 1.0, "3to1"),
]

# Weight-agnostic baselines shared across all scenarios.
# no_sharing_tpc is resolved per-scenario via WEIGHT_SCENARIOS.
BASELINES = ["fcfs", "no_sharing"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-base", default="experiments/fair_share/tsfm/results_noisy_neighbour")
    ap.add_argument("--plot-dir",     default=None,
                    help="Output dir (default: <results-base>/plots)")
    ap.add_argument("--victim-task",    default="ecgclass")
    ap.add_argument("--aggressor-task", default="gestureclass")
    ap.add_argument("--bin-size-s",     type=float, default=2.0)
    ap.add_argument("--x-max-s",        type=float, default=180.0,
                    help="Force the throughput/latency timeseries x-axis "
                         "upper limit (seconds). Use 0 or a negative value "
                         "to auto-fit the last phase boundary.")
    args = ap.parse_args()
    x_max_s = args.x_max_s if args.x_max_s and args.x_max_s > 0 else None

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

    # Resolve per-scenario method dirs once so we can optionally pre-compute
    # shared y-limits before drawing (used for the results_t4 row layout).
    scenario_method_dirs: List[Tuple[str, str, float, float, Dict[str, Path]]] = []
    for bfq_name, tpc_name, stfq_name, w_a, w_b, tag in WEIGHT_SCENARIOS:
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

        stfq_dir = base / stfq_name
        if stfq_dir.exists():
            method_dirs["stfq"] = stfq_dir
        else:
            print(f"[Skip] {stfq_name}: dir not found — STFQ bars omitted for {tag}")

        scenario_method_dirs.append((tag, bfq_name, w_a, w_b, method_dirs))

    # In results_t4 we additionally produce one combined fairness figure
    # (with a single caption-friendly layout) so it can be dropped into
    # overleaf without the awkward whitespace of three separately-saved
    # PDFs glued side-by-side. Individual per-scenario fairness PDFs are
    # still emitted for any other use.
    is_row_layout = base.name == "results_t4"

    if is_row_layout:
        plot_fairness_row(
            [(tag, mds, w_a, w_b)
             for (tag, _bfq, w_a, w_b, mds) in scenario_method_dirs],
            args.victim_task, args.aggressor_task, meta,
            plot_dir / "fairness_row.png",
            bin_s=args.bin_size_s,
        )

    for tag, bfq_name, w_a, w_b, method_dirs in scenario_method_dirs:
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
            x_max_s=x_max_s,
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
