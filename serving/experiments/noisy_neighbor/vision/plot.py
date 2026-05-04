#!/usr/bin/env python3
"""noisy_neighbor/vision — Time-series plots.

Produces:
  1. One plot per scheduler — victim + aggressor latency over time,
     with vertical lines at each phase transition.
  2. One combined plot — victim latency only, all schedulers overlaid.
  3. Throughput over time — all schedulers overlaid.
  4. Batch composition — victim slot share per phase per policy.
  5. Per-phase p50 summary bar chart.

Run from serving/:
    python experiments/noisy_neighbor/vision/plot.py \
        --results-base experiments/noisy_neighbor/vision/results \
        --plot-dir     experiments/noisy_neighbor/vision/plots

Select which methods to plot:
    --methods fcfs,stfq,bfq
    --methods fcfs,bfq,no_sharing,no_sharing_tpc

Limit to a subset of phases:
    --num-phases 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SERVING_DIR = Path(__file__).resolve().parents[3]

PALETTE = {
    "charcoal":   "#2F3640",
    "slate":      "#5C6773",
    "grid":       "#D9DEE5",
    "background": "#FAFBFC",
}

POLICIES: Dict[str, Dict] = {
    "fcfs":           {"color": "#6B9AC4", "label": "FCFS",            "ls": "-"},
    "stfq":           {"color": "#E8B298", "label": "STFQ",            "ls": (0, (4, 1, 1, 1))},
    "bfq":            {"color": "#E06C75", "label": "BFQ",             "ls": "--"},
    "bfq_aggr":       {"color": "#B0455A", "label": "BFQ (aggr-priority)", "ls": (0, (1, 1))},
    "no_sharing":     {"color": "#7BA591", "label": "No-Sharing",      "ls": (0, (3, 1, 1, 1, 1, 1))},
    "no_sharing_tpc": {"color": "#9B7BB8", "label": "No-Sharing (TPC)", "ls": (0, (5, 2))},
}

# Draw order (bottom → top)
POLICY_ORDER = ["fcfs", "stfq", "bfq", "bfq_aggr", "no_sharing", "no_sharing_tpc"]

_FALLBACK_COLORS = ["#C7BEDF", "#E7C98B", "#D9A6B3", "#A9C7B5", "#8FB7CF"]
_FALLBACK_LS     = [(0, (2, 1)), (0, (6, 2)), (0, (3, 2, 1, 2)), "-.", "--"]

VICTIM_COLOR    = "#6B9AC4"
AGGRESSOR_COLOR = "#E06C75"

Record = Tuple[float, float]  # (send_time_s, latency_ms)


def _set_clean_ticks(ax: plt.Axes, xdata_max: float, ydata_max: float, n_y: int = 4) -> Tuple[float, float]:
    """Snap axis limits to nice round numbers, set equally-spaced ticks. Returns (xlim, ylim)."""
    def _ticks_and_limit(data_max: float, n: int = 5) -> Tuple[np.ndarray, float]:
        step_raw = data_max / n
        magnitude = 10 ** np.floor(np.log10(max(step_raw, 1e-9)))
        nice = [1, 2, 2.5, 5, 10]
        step = magnitude * min(nice, key=lambda s: abs(s - step_raw / magnitude))
        nice_limit = np.ceil(data_max / step) * step
        ticks = np.round(np.arange(0, nice_limit + step * 0.01, step), 10)
        return ticks, float(nice_limit)

    xt, xlim_nice = _ticks_and_limit(xdata_max, n=5)
    yt, ylim_nice = _ticks_and_limit(ydata_max, n=n_y)
    ax.set_xlim(0, xlim_nice)
    ax.set_ylim(0, ylim_nice)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    return xlim_nice, ylim_nice


def _policy_cfg(policy: str, idx: int = 0) -> Dict:
    """Return style dict for policy, generating one if not in POLICIES registry."""
    if policy in POLICIES:
        return POLICIES[policy]
    return {
        "color": _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)],
        "label": policy,
        "ls":    _FALLBACK_LS[idx % len(_FALLBACK_LS)],
    }


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     "black",
        "axes.labelcolor":    "black",
        "axes.linewidth":     0.6,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          False,
        "xtick.color":        "black",
        "ytick.color":        "black",
        "xtick.major.width":  0.5,
        "ytick.major.width":  0.5,
        "xtick.major.size":   2.5,
        "ytick.major.size":   2.5,
        "text.color":         "black",
        "font.family":        "sans-serif",
        "font.size":          10,
        "axes.titlesize":     10,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
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
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[Plot] Saved: {out_path}")


def load_task(results_dir: Path, task: str,
              max_time: Optional[float] = None) -> Tuple[List[Record], dict]:
    meta_path = results_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    path = results_dir / f"{task}_timeseries.csv"
    if not path.exists():
        return [], meta
    recs: List[Record] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            t = float(row["send_time_s"])
            if max_time is not None and t > max_time:
                continue
            recs.append((t, float(row["latency_ms"])))
    return recs, meta


def _smooth(times: List[float], lats: List[float],
            window_s: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    if not times:
        return np.array([]), np.array([])
    t = np.array(times)
    l = np.array(lats)
    sm = np.array([
        float(np.median(l[(t >= ti - window_s / 2) & (t <= ti + window_s / 2)]))
        for ti in t
    ])
    return t, sm


def _add_phase_annotations(
    ax: plt.Axes,
    phase_boundaries: List[float],
    aggressor_rps_phases: List[float],
    xlim_max: float,
    ylim_max: float,
) -> None:
    """Add vertical lines at phase transitions (no background shading)."""
    for bnd in phase_boundaries[:-1]:  # skip last (end of experiment)
        ax.axvline(bnd, color=PALETTE["charcoal"], linewidth=1.2,
                   linestyle=":", zorder=4)


# ---------------------------------------------------------------------------
# Per-scheduler plot: victim + aggressor on same axes
# ---------------------------------------------------------------------------

def plot_scheduler(
    results_dir: Path,
    victim_task: str,
    aggressor_task: str,
    scheduler_label: str,
    out_path: Path,
    max_time: Optional[float] = None,
) -> None:
    victim_recs,   meta = load_task(results_dir, victim_task, max_time)
    aggressor_recs, _   = load_task(results_dir, aggressor_task, max_time)

    if not victim_recs and not aggressor_recs:
        print(f"[Info] No data in {results_dir} — skipping")
        return

    phase_boundaries     = meta.get("phase_boundaries_s", [])
    aggressor_rps_phases = meta.get("aggressor_rps_phases", [])
    n_phases = len(phase_boundaries)
    if max_time is not None:
        phase_boundaries     = [b for b in phase_boundaries if b <= max_time]
        aggressor_rps_phases = aggressor_rps_phases[:len(phase_boundaries)]

    all_lats = [r[1] for r in victim_recs] + [r[1] for r in aggressor_recs]
    scale    = 1 / 1000 if max(all_lats, default=0) > 2000 else 1.0
    unit     = "s" if scale < 1 else "ms"

    xlim_max = max_time if max_time is not None else (
        phase_boundaries[-1] if phase_boundaries else max(
            max((r[0] for r in victim_recs),    default=0),
            max((r[0] for r in aggressor_recs), default=0),
        )
    )

    v_bins = a_bins = None
    if victim_recs:
        vt = np.array([r[0] for r in victim_recs])
        vl = np.array([r[1] * scale for r in victim_recs])
        v_bins = _bin_latency(vt, vl, xlim_max)
    if aggressor_recs:
        at = np.array([r[0] for r in aggressor_recs])
        al = np.array([r[1] * scale for r in aggressor_recs])
        a_bins = _bin_latency(at, al, xlim_max)

    ylim_cap = 200 * scale
    fig, axes = plt.subplots(2, 1, figsize=(2.8, 2.4), sharex=True)
    ax_v, ax_a = axes

    # --- Victim panel ---
    if v_bins is not None:
        ax_v.plot(v_bins[0], v_bins[1], color=VICTIM_COLOR, linewidth=1.2, zorder=3)
    ylim_v = min(float(np.nanmax(v_bins[1])) if v_bins is not None else 1.0, ylim_cap)
    xlim_max, _ = _set_clean_ticks(ax_v, xlim_max, ylim_v, n_y=4)
    if phase_boundaries:
        _add_phase_annotations(ax_v, phase_boundaries, aggressor_rps_phases, xlim_max, ylim_v)
    ax_v.set_ylabel(f"Latency ({unit})")
    ax_v.set_title(f"Victim ({victim_task})", pad=2)

    # --- Aggressor panel ---
    if a_bins is not None:
        ax_a.plot(a_bins[0], a_bins[1], color=AGGRESSOR_COLOR, linewidth=1.2, zorder=3)
    ylim_a = min(float(np.nanmax(a_bins[1])) if a_bins is not None else 1.0, ylim_cap)
    _set_clean_ticks(ax_a, xlim_max, ylim_a, n_y=4)
    if phase_boundaries:
        _add_phase_annotations(ax_a, phase_boundaries, aggressor_rps_phases, xlim_max, ylim_a)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel(f"Latency ({unit})")
    ax_a.set_title(f"Aggressor ({aggressor_task})", pad=2)

    fig.tight_layout(pad=0.4)
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined plot: victim latency, all schedulers overlaid
# ---------------------------------------------------------------------------

def plot_all_policies(
    policy_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_task: str,
    out_path: Path,
    max_time: Optional[float] = None,
) -> None:
    v_data: Dict[str, List[Record]] = {}
    a_data: Dict[str, List[Record]] = {}
    meta_ref: dict = {}

    for policy, d in policy_dirs.items():
        if not d.exists():
            print(f"[Info] {policy} results not found at {d} — skipping")
            continue
        v_recs, meta = load_task(d, victim_task,    max_time)
        a_recs, _    = load_task(d, aggressor_task, max_time)
        if v_recs:
            v_data[policy] = v_recs
            meta_ref = meta
        if a_recs:
            a_data[policy] = a_recs

    if not v_data and not a_data:
        print("[Error] No data found for any policy.")
        return

    all_lats = [r[1] for recs in list(v_data.values()) + list(a_data.values()) for r in recs]
    scale    = 1 / 1000 if max(all_lats) > 2000 else 1.0
    unit     = "s" if scale < 1 else "ms"

    phase_boundaries     = meta_ref.get("phase_boundaries_s", [])
    aggressor_rps_phases = meta_ref.get("aggressor_rps_phases", [])
    if max_time is not None:
        phase_boundaries     = [b for b in phase_boundaries if b <= max_time]
        aggressor_rps_phases = aggressor_rps_phases[:len(phase_boundaries)]
    xlim_max = max_time if max_time is not None else (
        phase_boundaries[-1] if phase_boundaries else
        max(r[0] for recs in list(v_data.values()) + list(a_data.values()) for r in recs)
    )

    def _bin_data(data: Dict[str, List[Record]]):
        binned = {}
        for policy, recs in data.items():
            times = np.array([r[0] for r in recs])
            lats  = np.array([r[1] * scale for r in recs])
            binned[policy] = _bin_latency(times, lats, xlim_max)
        return binned

    v_binned = _bin_data(v_data)
    a_binned = _bin_data(a_data)
    ylim_cap = 200 * scale

    fig, axes = plt.subplots(2, 1, figsize=(2.8, 2.4), sharex=True)
    ax_v, ax_a = axes

    for idx, (policy, (centers, means)) in enumerate(v_binned.items()):
        cfg = _policy_cfg(policy, idx)
        ax_v.plot(centers, means, color=cfg["color"],
                  linestyle=cfg["ls"], linewidth=1.2, zorder=3, label=cfg["label"])

    ylim_v = min(max((float(np.nanmax(m)) for _, m in v_binned.values()), default=1.0), ylim_cap)
    xlim_max, _ = _set_clean_ticks(ax_v, xlim_max, ylim_v, n_y=4)
    if phase_boundaries:
        _add_phase_annotations(ax_v, phase_boundaries, aggressor_rps_phases, xlim_max, ylim_v)
    ax_v.set_ylabel(f"Latency ({unit})")
    ax_v.text(0.02, 0.96, "Victim", transform=ax_v.transAxes,
              fontsize=6.5, va="top", ha="left", color=PALETTE["charcoal"])

    for idx, (policy, (centers, means)) in enumerate(a_binned.items()):
        cfg = _policy_cfg(policy, idx)
        ax_a.plot(centers, means, color=cfg["color"],
                  linestyle=cfg["ls"], linewidth=1.2, zorder=3, label=cfg["label"])

    ylim_a = min(max((float(np.nanmax(m)) for _, m in a_binned.values()), default=1.0), ylim_cap)
    _set_clean_ticks(ax_a, xlim_max, ylim_a, n_y=4)
    if phase_boundaries:
        _add_phase_annotations(ax_a, phase_boundaries, aggressor_rps_phases, xlim_max, ylim_a)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel(f"Latency ({unit})")
    ax_a.text(0.02, 0.96, "Aggressor", transform=ax_a.transAxes,
              fontsize=6.5, va="top", ha="left", color=PALETTE["charcoal"])

    handles, labels = ax_v.get_legend_handles_labels()
    fig.tight_layout(pad=0.4)
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                     ncol=len(handles), frameon=False, handlelength=1.2, columnspacing=0.8)
    fig.subplots_adjust(top=1.0 - (leg.get_window_extent(fig.canvas.get_renderer()).height
                                   / fig.get_window_extent().height) - 0.02)
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Batch composition plot: victim_frac per phase, all policies side-by-side
# ---------------------------------------------------------------------------

def _load_batch_composition(
    log_path: Path,
    victim_task: str,
    phase_boundaries_s: List[float],
) -> Optional[List[float]]:
    """Parse device log and return mean victim_frac per phase."""
    import ast
    import re

    n_phases = len(phase_boundaries_s)
    if not log_path.exists() or n_phases == 0:
        return None

    text = log_path.read_text()
    prepared_tasks  = re.findall(r"Prepared batch.*?tasks=(\[.*?\])", text)
    finished_starts = [int(t) for t in re.findall(r"Finished batch.*?start=(\d+)", text)]

    if not prepared_tasks or not finished_starts:
        return None

    if len(prepared_tasks) != len(finished_starts):
        # Mismatched — fall back to equal-count split
        n = len(prepared_tasks)
        phase_size = max(1, n // n_phases)
        result = []
        for i in range(n_phases):
            chunk = prepared_tasks[i * phase_size: (i + 1) * phase_size if i < n_phases - 1 else n]
            fracs = [ast.literal_eval(t).count(victim_task) / len(ast.literal_eval(t)) for t in chunk]
            result.append(float(np.mean(fracs)) if fracs else 0.0)
        return result

    t0_ns   = finished_starts[0]
    times_s = [(t - t0_ns) / 1e9 for t in finished_starts]

    phase_fracs: List[List[float]] = [[] for _ in range(n_phases)]
    for task_str, t in zip(prepared_tasks, times_s):
        tasks = ast.literal_eval(task_str)
        frac  = tasks.count(victim_task) / len(tasks)
        bucket = n_phases - 1
        for k, bnd in enumerate(phase_boundaries_s):
            if t < bnd:
                bucket = k
                break
        phase_fracs[bucket].append(frac)

    return [float(np.mean(f)) if f else 0.0 for f in phase_fracs]


def plot_batch_composition(
    logs_dir: Path,
    policy_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_rps_phases: List[float],
    phase_boundaries_s: List[float],
    out_path: Path,
) -> None:
    """Bar chart: mean victim fraction per batch, grouped by phase, one bar per policy."""
    phase_labels = [f"agg={int(r)}rps" for r in aggressor_rps_phases]
    n_phases = len(phase_labels)

    composition: Dict[str, List[float]] = {}
    for idx, policy in enumerate(policy_dirs):
        log_path = logs_dir / f"device_{policy}.log"
        fracs = _load_batch_composition(log_path, victim_task, phase_boundaries_s)
        if fracs is not None:
            composition[policy] = fracs

    if not composition:
        print("[Plot] No batch logs found — skipping composition plot")
        return

    n_policies  = len(composition)
    x           = np.arange(n_phases)
    total_width = 0.7
    w           = total_width / n_policies

    fig, ax = plt.subplots(figsize=(2.8, 1.5))

    for i, (policy, fracs) in enumerate(composition.items()):
        cfg    = _policy_cfg(policy, i)
        offset = (i - n_policies / 2 + 0.5) * w
        bars   = ax.bar(x + offset, fracs, width=w,
                        color=cfg["color"], alpha=0.85,
                        edgecolor="black", linewidth=0.4,
                        label=cfg["label"])
        for bar, v in zip(bars, fracs):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=4.5,
                        color=PALETTE["charcoal"])

    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels)
    ax.set_xlabel("Aggressor Load Phase")
    ax.set_ylabel("Victim Slot Share")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False, handlelength=1.2)
    fig.tight_layout(pad=0.4)
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary bar chart: per-phase p50 victim latency, all policies
# ---------------------------------------------------------------------------

def plot_phase_summary(
    policy_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_rps_phases: List[float],
    out_path: Path,
) -> None:
    """Grouped bar chart: victim p50 latency per phase for each policy."""
    phase_labels = [f"agg={int(r)}rps" for r in aggressor_rps_phases]
    n_phases     = len(phase_labels)

    p50s: Dict[str, List[float]] = {}
    for policy, d in policy_dirs.items():
        path = d / f"{victim_task}_timeseries.csv"
        if not path.exists():
            continue
        by_phase: Dict[int, List[float]] = {}
        with path.open() as f:
            for row in csv.DictReader(f):
                p = int(row["phase"])
                by_phase.setdefault(p, []).append(float(row["latency_ms"]))
        vals = []
        for p in range(1, n_phases + 1):
            lats = sorted(by_phase.get(p, [0.0]))
            vals.append(lats[len(lats) // 2])
        p50s[policy] = vals

    if not p50s:
        return

    all_vals = [v for vals in p50s.values() for v in vals]
    scale    = 1 / 1000 if max(all_vals) > 2000 else 1.0
    unit     = "s" if scale < 1 else "ms"

    n_policies  = len(p50s)
    x           = np.arange(n_phases)
    total_width = 0.7
    w           = total_width / n_policies

    fig, ax = plt.subplots(figsize=(2.8, 1.5))

    for i, (policy, vals) in enumerate(p50s.items()):
        cfg    = _policy_cfg(policy, i)
        offset = (i - n_policies / 2 + 0.5) * w
        scaled = [v * scale for v in vals]
        ax.bar(x + offset, scaled, width=w,
               color=cfg["color"], alpha=0.85,
               edgecolor="black", linewidth=0.4,
               label=cfg["label"])

    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels)
    ax.set_xlabel("Aggressor Load Phase")
    ax.set_ylabel(f"P50 Latency ({unit})")
    ax.legend(frameon=False, handlelength=1.2)
    fig.tight_layout(pad=0.4)
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Throughput plot: completed requests/s over time, all policies overlaid
# ---------------------------------------------------------------------------

def _bin_rate(
    times: np.ndarray,
    max_time: float,
    bin_size_s: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Count events in fixed-width bins, returning (bin_centers, req/s)."""
    n_bins  = int(np.ceil(max_time / bin_size_s))
    counts  = np.zeros(n_bins, dtype=float)
    for t in times:
        idx = int(t / bin_size_s)
        if 0 <= idx < n_bins:
            counts[idx] += 1.0
    centers = (np.arange(n_bins) + 0.5) * bin_size_s
    return centers, counts / bin_size_s


def _bin_latency(
    times: np.ndarray,
    lats: np.ndarray,
    max_time: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean latency in exact 1s bins [0,1), [1,2), ..., returning (bin_centers, mean_lat).
    Bins with no requests are NaN so line breaks naturally."""
    n_bins  = int(np.ceil(max_time))
    sums    = np.zeros(n_bins, dtype=float)
    counts  = np.zeros(n_bins, dtype=float)
    for t, l in zip(times, lats):
        idx = int(t)
        if 0 <= idx < n_bins:
            sums[idx]   += l
            counts[idx] += 1.0
    means = np.where(counts > 0, sums / counts, np.nan)
    centers = np.arange(n_bins) + 0.5
    return centers, means


def _compute_throughput(
    recs: List[Record],
    max_time: Optional[float] = None,
    bin_size_s: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Completed req/s in fixed-width bins — uses completion time (send + latency)."""
    if not recs:
        return np.array([]), np.array([])
    times = np.array([r[0] + r[1] / 1000.0 for r in recs])
    end   = max_time if max_time is not None else float(times.max())
    return _bin_rate(times, end, bin_size_s)


def _compute_offered_load(
    recs: List[Record],
    max_time: Optional[float] = None,
    bin_size_s: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Offered load from send times in fixed-width bins."""
    if not recs:
        return np.array([]), np.array([])
    times = np.array([r[0] for r in recs])
    end   = max_time if max_time is not None else float(times.max())
    return _bin_rate(times, end, bin_size_s)


def plot_throughput(
    policy_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_task: str,
    out_path: Path,
    max_time: Optional[float] = None,
    bin_size_s: float = 1.0,
) -> None:
    """Two-panel throughput plot: victim (top) and aggressor (bottom)."""
    victim_data:    Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    aggressor_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    meta_ref: dict = {}

    for policy, d in policy_dirs.items():
        if not d.exists():
            continue
        v_recs, meta = load_task(d, victim_task, max_time)
        a_recs, _    = load_task(d, aggressor_task, max_time)
        if v_recs:
            victim_data[policy]    = _compute_throughput(v_recs, max_time, bin_size_s)
            meta_ref = meta
        if a_recs:
            aggressor_data[policy] = _compute_throughput(a_recs, max_time, bin_size_s)

    if not victim_data and not aggressor_data:
        print("[Plot] No throughput data found — skipping")
        return

    phase_boundaries     = meta_ref.get("phase_boundaries_s", [])
    aggressor_rps_phases = meta_ref.get("aggressor_rps_phases", [])
    xlim_max = max_time if max_time is not None else (
        phase_boundaries[-1] if phase_boundaries else max(
            max((c.max() for c, _ in victim_data.values()),    default=0),
            max((c.max() for c, _ in aggressor_data.values()), default=0),
        )
    )

    # Compute offered load from trace.json if present (deterministic, shared
    # across all scheduler runs), otherwise fall back to first policy's CSV.
    agg_offered: Optional[Tuple[np.ndarray, np.ndarray]] = None
    vic_offered: Optional[Tuple[np.ndarray, np.ndarray]] = None
    trace_path = base / "trace.json" if (base := next(
        (d.parent for d in policy_dirs.values() if d.exists()), None
    )) is not None else None
    if trace_path is not None and trace_path.exists():
        trace = json.loads(trace_path.read_text())
        end = max_time if max_time is not None else xlim_max
        if victim_task in trace:
            sends = [t for t in trace[victim_task] if max_time is None or t <= max_time]
            vic_offered = _compute_offered_load([(t, 0.0) for t in sends], end, bin_size_s)
        if aggressor_task in trace:
            sends = [t for t in trace[aggressor_task] if max_time is None or t <= max_time]
            agg_offered = _compute_offered_load([(t, 0.0) for t in sends], end, bin_size_s)
    else:
        # Fallback: use send times from first available policy's CSV
        for _, d in policy_dirs.items():
            if not d.exists():
                continue
            v_recs_raw, _ = load_task(d, victim_task, max_time)
            a_recs_raw, _ = load_task(d, aggressor_task, max_time)
            if v_recs_raw:
                vic_offered = _compute_offered_load(v_recs_raw, max_time, bin_size_s)
            if a_recs_raw:
                agg_offered = _compute_offered_load(a_recs_raw, max_time, bin_size_s)
            break

    fig, axes = plt.subplots(2, 1, figsize=(2.8, 2.4), sharex=True)

    task_labels = {victim_task: "Victim", aggressor_task: "Aggressor"}
    for ax, task_data, task_name, offered in [
        (axes[0], victim_data,    victim_task,    vic_offered),
        (axes[1], aggressor_data, aggressor_task, agg_offered),
    ]:
        all_rps = [r for _, rps in task_data.values() for r in rps]
        offered_max = float(offered[1].max()) if offered is not None and len(offered[1]) else 0.0
        ylim_max = max(all_rps + [offered_max]) if (all_rps or offered_max) else 1.0

        for idx, (policy, (centers, rps)) in enumerate(task_data.items()):
            cfg = _policy_cfg(policy, idx)
            ax.plot(centers, rps, color=cfg["color"],
                    linestyle=cfg["ls"], label=cfg["label"], zorder=3)

        if offered is not None:
            ax.plot(offered[0], offered[1], color=PALETTE["charcoal"],
                    linewidth=0.8, linestyle=":", label="Offered load", zorder=5)

        for bnd in phase_boundaries[:-1]:
            ax.axvline(bnd, color=PALETTE["charcoal"], linewidth=0.8, linestyle=":", zorder=4)

        _set_clean_ticks(ax, xlim_max, ylim_max)
        ax.set_ylabel("Req/s")
        ax.text(0.02, 0.96, task_labels[task_name], transform=ax.transAxes,
                fontsize=6.5, va="top", ha="left", color=PALETTE["charcoal"])

    axes[1].set_xlabel("Time (s)")
    # Single shared legend above the figure with 4 columns
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(pad=0.4)
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                     ncol=4, frameon=False, handlelength=1.2, columnspacing=0.8)
    # fig.subplots_adjust(top=1.0 - (leg.get_window_extent(fig.canvas.get_renderer()).height
    #                                / fig.get_window_extent().height) - 0.04)
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Throughput attainment bar chart: served / offered, per task, per method
# ---------------------------------------------------------------------------

VICTIM_BAR_COLOR    = "#6B9AC4"
AGGRESSOR_BAR_COLOR = "#E06C75"


def _offered_send_times(
    policy_dirs: Dict[str, Path],
    task: str,
    max_time: Optional[float],
) -> Optional[np.ndarray]:
    """Return absolute send times for `task` from trace.json (shared across methods)."""
    base = next((d.parent for d in policy_dirs.values() if d.exists()), None)
    if base is None:
        return None
    trace_path = base / "trace.json"
    if not trace_path.exists():
        return None
    trace = json.loads(trace_path.read_text())
    sends = trace.get(task, [])
    if max_time is not None:
        sends = [t for t in sends if t <= max_time]
    return np.array(sends, dtype=float)


def _bin_attainment(
    sends: np.ndarray,
    completions: np.ndarray,
    max_time: float,
    bin_size_s: float = 1.0,
) -> float:
    """Capped throughput attainment over fixed-width bins.

    Same binning as the timeseries throughput plot. Per-bin ratio is capped
    at 1.0 so a bin where the system catches up on backlog from earlier bins
    can't pull the average above 100%. The final value is the sum of capped
    served requests divided by total offered requests.
    """
    n_bins = int(np.ceil(max_time / bin_size_s))
    if n_bins <= 0:
        return 0.0
    offered = np.zeros(n_bins, dtype=float)
    served  = np.zeros(n_bins, dtype=float)
    for t in sends:
        idx = int(t / bin_size_s)
        if 0 <= idx < n_bins:
            offered[idx] += 1.0
    for t in completions:
        idx = int(t / bin_size_s)
        if 0 <= idx < n_bins:
            served[idx] += 1.0
    total_offered = float(offered.sum())
    if total_offered <= 0:
        return 0.0
    return float(np.minimum(served, offered).sum() / total_offered)


def plot_throughput_attainment(
    policy_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_task: str,
    out_path: Path,
    max_time: Optional[float] = None,
    bin_size_s: float = 1.0,
) -> None:
    """Grouped bar chart: per-bin attainment (served vs offered, capped per bin).

    For each fixed-width bin the served count is capped at the offered count, so a bin
    where the system catches up on backlog from earlier bins doesn't count
    above 100%. The bar is sum_b min(served_b, offered_b) / sum_b offered_b.
    This matches the binned view of the
    timeseries throughput plot.

    X-axis: methods (one group per method, in POLICY_ORDER).
    Bars per group: victim, aggressor.
    """
    methods = [m for m, d in policy_dirs.items() if d.exists()]
    if not methods:
        print("[Plot] No methods available for attainment plot — skipping")
        return

    v_sends = _offered_send_times(policy_dirs, victim_task,    max_time)
    a_sends = _offered_send_times(policy_dirs, aggressor_task, max_time)
    if v_sends is None or a_sends is None or len(v_sends) == 0 or len(a_sends) == 0:
        print("[Plot] Cannot determine offered load (trace.json missing) — skipping attainment plot")
        return

    # Window for binning: cover all offered sends (and any completions that may
    # land just past the last send).
    horizon = max_time if max_time is not None else float(
        max(v_sends.max() if len(v_sends) else 0.0,
            a_sends.max() if len(a_sends) else 0.0) + 1.0
    )

    v_attain: List[float] = []
    a_attain: List[float] = []
    for m in methods:
        d = policy_dirs[m]
        v_recs, _ = load_task(d, victim_task,    max_time)
        a_recs, _ = load_task(d, aggressor_task, max_time)
        # Completion time = send_time + latency
        v_done = np.array([t + lat / 1000.0 for t, lat in v_recs], dtype=float)
        a_done = np.array([t + lat / 1000.0 for t, lat in a_recs], dtype=float)
        v_attain.append(_bin_attainment(v_sends, v_done, horizon, bin_size_s))
        a_attain.append(_bin_attainment(a_sends, a_done, horizon, bin_size_s))

    n           = len(methods)
    x           = np.arange(n)
    total_width = 0.7
    w           = total_width / 2

    fig, ax = plt.subplots(figsize=(3.4, 1.8))

    b_v = ax.bar(x - w/2, v_attain, width=w,
                 color=VICTIM_BAR_COLOR, alpha=0.9,
                 edgecolor="black", linewidth=0.5,
                 label="Victim")
    b_a = ax.bar(x + w/2, a_attain, width=w,
                 color=AGGRESSOR_BAR_COLOR, alpha=0.9,
                 edgecolor="black", linewidth=0.5,
                 label="Aggressor")

    for bars, vals in [(b_v, v_attain), (b_a, a_attain)]:
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        f"{v:.0%}", ha="center", va="bottom",
                        fontsize=5.5, color=PALETTE["charcoal"])

    # Reference line at 100% attainment
    ax.axhline(1.0, color=PALETTE["charcoal"], linewidth=0.6,
               linestyle=":", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([_policy_cfg(m).get("label", m) for m in methods],
                       rotation=20, ha="right")
    ax.set_ylabel("Throughput Attainment")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False, handlelength=1.2, loc="upper right",
              ncol=2, columnspacing=0.8)
    fig.tight_layout(pad=0.4)
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rich-metrics plots (use latencies.csv / task_results.csv from new schema)
# ---------------------------------------------------------------------------

def _load_latencies_csv(results_dir: Path, task: str,
                        max_time: Optional[float] = None) -> List[dict]:
    """Read latencies.csv and return rows for the given task as dicts."""
    path = results_dir / "latencies.csv"
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["task"] != task:
                continue
            t = float(row["elapsed_sec"])
            if max_time is not None and t > max_time:
                continue
            rows.append({k: (int(v) if k == "phase" else
                             float(v) if k not in ("task", "condition") else v)
                         for k, v in row.items()})
    return rows


def plot_latency_components_by_phase(
    policy_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_rps_phases: List[float],
    out_path: Path,
    max_time: Optional[float] = None,
) -> None:
    """Stacked bar: mean exec / queue / overhead per phase for the victim task.

    One group of bars per policy, one bar per phase. Stacks: server_exec_ms
    (GPU time), queue_wait_plus_rpc_ms (queueing), non_server_exec_overhead_ms
    (everything else). Shows how each cost component changes as aggressor ramps.
    """
    phase_labels = [f"agg={int(r)}rps" for r in aggressor_rps_phases]
    n_phases = len(phase_labels)
    if n_phases == 0:
        return

    COMPONENTS = [
        ("server_exec_ms",              "#6B9AC4", "GPU exec"),
        ("queue_wait_plus_rpc_ms",      "#E06C75", "Queue wait"),
        ("non_server_exec_overhead_ms", "#C7BEDF", "Other overhead"),
    ]

    # policy → phase → {component: mean_ms}
    data: Dict[str, List[Dict[str, float]]] = {}
    for policy, d in policy_dirs.items():
        if not d.exists():
            continue
        rows = _load_latencies_csv(d, victim_task, max_time)
        if not rows:
            continue
        by_phase: Dict[int, List[dict]] = {}
        for row in rows:
            p = int(row["phase"])
            by_phase.setdefault(p, []).append(row)
        phase_means = []
        for p in range(1, n_phases + 1):
            recs = by_phase.get(p, [])
            if recs:
                phase_means.append({c: float(np.mean([r[c] for r in recs])) for c, _, _ in COMPONENTS})
            else:
                phase_means.append({c: 0.0 for c, _, _ in COMPONENTS})
        data[policy] = phase_means

    if not data:
        print("[Plot] No latencies.csv found — skipping component plot")
        return

    n_policies  = len(data)
    x           = np.arange(n_phases)
    total_width = 0.7
    w           = total_width / n_policies

    fig, ax = plt.subplots(figsize=(max(3.0, 1.5 * n_phases), 2.0))

    for i, (policy, phase_means) in enumerate(data.items()):
        cfg    = _policy_cfg(policy, i)
        offset = (i - n_policies / 2 + 0.5) * w
        bottoms = np.zeros(n_phases)
        for comp_key, comp_color, comp_label in COMPONENTS:
            heights = np.array([pm[comp_key] for pm in phase_means])
            label = f"{cfg['label']} – {comp_label}" if i == 0 else None
            ax.bar(x + offset, heights, width=w * 0.9,
                   bottom=bottoms, color=comp_color, alpha=0.85,
                   edgecolor="black", linewidth=0.3, label=label)
            bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels)
    ax.set_xlabel("Aggressor Load Phase")
    ax.set_ylabel("Victim Latency (ms)")
    ax.legend(frameon=False, handlelength=1.0, fontsize=7,
              loc="upper left", ncol=1)
    fig.tight_layout(pad=0.4)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_p99_phase_summary(
    policy_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_rps_phases: List[float],
    out_path: Path,
    max_time: Optional[float] = None,
) -> None:
    """Grouped bar chart: victim P99 latency per phase per policy (from latencies.csv)."""
    phase_labels = [f"agg={int(r)}rps" for r in aggressor_rps_phases]
    n_phases = len(phase_labels)
    if n_phases == 0:
        return

    p99s: Dict[str, List[float]] = {}
    for policy, d in policy_dirs.items():
        if not d.exists():
            continue
        rows = _load_latencies_csv(d, victim_task, max_time)
        if not rows:
            continue
        by_phase: Dict[int, List[float]] = {}
        for row in rows:
            by_phase.setdefault(int(row["phase"]), []).append(row["latency_ms"])
        vals = [float(np.percentile(by_phase.get(p, [0.0]), 99))
                for p in range(1, n_phases + 1)]
        p99s[policy] = vals

    if not p99s:
        print("[Plot] No latencies.csv found — skipping P99 phase summary")
        return

    all_vals = [v for vals in p99s.values() for v in vals]
    scale    = 1 / 1000 if max(all_vals) > 2000 else 1.0
    unit     = "s" if scale < 1 else "ms"

    n_policies  = len(p99s)
    x           = np.arange(n_phases)
    total_width = 0.7
    w           = total_width / n_policies

    fig, ax = plt.subplots(figsize=(max(3.0, 1.5 * n_phases), 1.8))

    for i, (policy, vals) in enumerate(p99s.items()):
        cfg    = _policy_cfg(policy, i)
        offset = (i - n_policies / 2 + 0.5) * w
        ax.bar(x + offset, [v * scale for v in vals], width=w,
               color=cfg["color"], alpha=0.85,
               edgecolor="black", linewidth=0.4, label=cfg["label"])

    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels)
    ax.set_xlabel("Aggressor Load Phase")
    ax.set_ylabel(f"P99 Latency ({unit})")
    ax.legend(frameon=False, handlelength=1.2)
    fig.tight_layout(pad=0.4)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_queue_wait_timeseries(
    policy_dirs: Dict[str, Path],
    victim_task: str,
    out_path: Path,
    max_time: Optional[float] = None,
) -> None:
    """Mean queue_wait_plus_rpc_ms over time for the victim task, all policies overlaid."""
    meta_ref: dict = {}
    binned_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for policy, d in policy_dirs.items():
        if not d.exists():
            continue
        rows = _load_latencies_csv(d, victim_task, max_time)
        if not rows:
            continue
        # Also grab meta for phase annotations
        meta_path = d / "meta.json"
        if meta_path.exists() and not meta_ref:
            meta_ref = json.loads(meta_path.read_text())
        times = np.array([r["elapsed_sec"] for r in rows])
        waits = np.array([r["queue_wait_plus_rpc_ms"] for r in rows])
        xlim  = max_time if max_time is not None else float(times.max())
        binned_data[policy] = _bin_latency(times, waits, xlim)

    if not binned_data:
        print("[Plot] No latencies.csv found — skipping queue wait plot")
        return

    phase_boundaries     = meta_ref.get("phase_boundaries_s", [])
    aggressor_rps_phases = meta_ref.get("aggressor_rps_phases", [])
    if max_time is not None:
        phase_boundaries     = [b for b in phase_boundaries if b <= max_time]
        aggressor_rps_phases = aggressor_rps_phases[:len(phase_boundaries)]

    xlim_max = max_time if max_time is not None else (
        phase_boundaries[-1] if phase_boundaries else
        max(float(c.max()) for c, _ in binned_data.values())
    )

    fig, ax = plt.subplots(figsize=(2.8, 1.6))

    for idx, (policy, (centers, means)) in enumerate(binned_data.items()):
        cfg = _policy_cfg(policy, idx)
        ax.plot(centers, means, color=cfg["color"],
                linestyle=cfg["ls"], linewidth=1.2, label=cfg["label"], zorder=3)

    ylim_max = max(float(np.nanmax(m)) for _, m in binned_data.values())
    _set_clean_ticks(ax, xlim_max, ylim_max, n_y=4)
    if phase_boundaries:
        _add_phase_annotations(ax, phase_boundaries, aggressor_rps_phases, xlim_max, ylim_max)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue wait (ms)")
    ax.legend(frameon=False, handlelength=1.2, fontsize=8)
    fig.tight_layout(pad=0.4)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_system_throughput_vs_victim_p95(
    policy_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_task: str,
    out_path: Path,
    max_time: Optional[float] = None,
    bin_size_s: float = 1.0,
) -> None:
    """Scatter: system throughput attainment vs victim P95 latency.

    X-axis uses the same capped binned throughput-attainment definition as
    throughput_attainment.png, weighted by each task's offered request count.
    Y-axis is victim P95 latency over the same time window.
    """
    methods = [m for m, d in policy_dirs.items() if d.exists()]
    if not methods:
        print("[Plot] No methods available for tradeoff plot — skipping")
        return

    v_sends = _offered_send_times(policy_dirs, victim_task,    max_time)
    a_sends = _offered_send_times(policy_dirs, aggressor_task, max_time)
    if v_sends is None or a_sends is None or len(v_sends) == 0 or len(a_sends) == 0:
        print("[Plot] Cannot determine offered load (trace.json missing) — skipping tradeoff plot")
        return

    horizon = max_time if max_time is not None else float(
        max(v_sends.max() if len(v_sends) else 0.0,
            a_sends.max() if len(a_sends) else 0.0) + 1.0
    )

    points = []
    for idx, method in enumerate(methods):
        d = policy_dirs[method]
        v_recs, _ = load_task(d, victim_task,    max_time)
        a_recs, _ = load_task(d, aggressor_task, max_time)
        if not v_recs or not a_recs:
            continue

        v_done = np.array([t + lat / 1000.0 for t, lat in v_recs], dtype=float)
        a_done = np.array([t + lat / 1000.0 for t, lat in a_recs], dtype=float)
        v_attain = _bin_attainment(v_sends, v_done, horizon, bin_size_s)
        a_attain = _bin_attainment(a_sends, a_done, horizon, bin_size_s)
        system_attain = (
            v_attain * len(v_sends) + a_attain * len(a_sends)
        ) / (len(v_sends) + len(a_sends))

        victim_lats = [lat for _, lat in v_recs]
        victim_p95 = float(np.percentile(victim_lats, 95))
        points.append((method, system_attain, victim_p95, idx))

    if not points:
        print("[Plot] No complete data for tradeoff plot — skipping")
        return

    y_vals = [p[2] for p in points]
    scale = 1 / 1000 if max(y_vals) > 2000 else 1.0
    unit = "s" if scale < 1 else "ms"

    fig, ax = plt.subplots(figsize=(2.8, 2.0))

    for method, system_attain, victim_p95, idx in points:
        cfg = _policy_cfg(method, idx)
        x = system_attain * 100
        y = victim_p95 * scale
        ax.scatter(x, y, s=32, color=cfg["color"], edgecolor="black",
                   linewidth=0.5, zorder=3)
        ax.annotate(
            cfg["label"],
            (x, y),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.5,
            color=PALETTE["charcoal"],
        )

    x_min = min(p[1] for p in points) * 100
    x_max = max(p[1] for p in points) * 100
    y_max = max(p[2] for p in points) * scale
    x_pad = max(1.0, (x_max - x_min) * 0.12)
    ax.set_xlim(max(0, x_min - x_pad), min(102, x_max + x_pad))
    _set_clean_ticks(ax, ax.get_xlim()[1], y_max, n_y=4)
    ax.set_xlim(max(0, x_min - x_pad), min(102, x_max + x_pad))
    ax.set_xlabel("System Throughput Attainment (%)")
    ax.set_ylabel(f"Victim P95 Latency ({unit})")
    fig.tight_layout(pad=0.4)
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_latencies_with_phase(results_dir: Path, task: str,
                                max_time: Optional[float] = None
                                ) -> List[Tuple[float, float, int]]:
    """Returns list of (send_time_s, latency_ms, phase) tuples."""
    path = results_dir / "latencies.csv"
    if not path.exists():
        return []
    out: List[Tuple[float, float, int]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("task") != task:
                continue
            t = float(r["elapsed_sec"])
            if max_time is not None and t > max_time:
                continue
            out.append((t, float(r["latency_ms"]), int(r["phase"])))
    return out


def _violation_pct(lats: List[float], slo_ms: float) -> float:
    if not lats:
        return 0.0
    return 100.0 * sum(1 for l in lats if l > slo_ms) / len(lats)


def _load_lat_and_exec_with_phase(results_dir: Path, task: str,
                                   max_time: Optional[float] = None
                                   ) -> List[Tuple[float, float, float, int]]:
    """Returns (send_t, lat_ms, server_exec_ms, phase) tuples."""
    path = results_dir / "latencies.csv"
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for r in csv.DictReader(f):
            if r.get("task") != task:
                continue
            t = float(r["elapsed_sec"])
            if max_time is not None and t > max_time:
                continue
            out.append((t, float(r["latency_ms"]), float(r["server_exec_ms"]),
                        int(r["phase"])))
    return out


def plot_victim_exec_by_phase(
    policy_dirs: Dict[str, Path],
    victim_task: str,
    phase_bounds: List[float],
    out_path: Path,
    max_time: Optional[float] = None,
) -> None:
    """Grouped bar: victim avg server_exec_ms per phase, one group per method.
    Surfaces kernel-level contention: in no_sharing the victim's own kernel
    runs longer in burst phases because the GPU's SMs are occupied by aggressor
    kernels from the co-located process. SLO-independent."""
    if not policy_dirs or not phase_bounds:
        return
    methods = list(policy_dirs.keys())
    num_phases = len(phase_bounds)
    exec_by_method: Dict[str, List[float]] = {}
    for m, d in policy_dirs.items():
        recs = _load_lat_and_exec_with_phase(d, victim_task, max_time=max_time)
        per_phase: Dict[int, List[float]] = {p: [] for p in range(1, num_phases + 1)}
        for _, _, ex, ph in recs:
            if ph in per_phase:
                per_phase[ph].append(ex)
        exec_by_method[m] = [
            float(np.mean(per_phase[p])) if per_phase[p] else 0.0
            for p in range(1, num_phases + 1)
        ]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(num_phases)
    width = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        cfg = _policy_cfg(m, i)
        ax.bar(x + i * width, exec_by_method[m], width,
               label=cfg.get("label", m),
               color=cfg.get("color", "#888"),
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([f"Phase {p+1}" for p in range(num_phases)])
    ax.set_ylabel("Victim avg server_exec (ms)")
    ax.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_victim_latency_quantiles_by_phase(
    policy_dirs: Dict[str, Path],
    victim_task: str,
    phase_bounds: List[float],
    out_path: Path,
    quantile: float = 0.95,
    max_time: Optional[float] = None,
) -> None:
    """Grouped bar: victim p95 (or chosen quantile) latency per phase.
    SLO-independent — captures distribution shift directly."""
    if not policy_dirs or not phase_bounds:
        return
    methods = list(policy_dirs.keys())
    num_phases = len(phase_bounds)
    q_by_method: Dict[str, List[float]] = {}
    for m, d in policy_dirs.items():
        recs = _load_lat_and_exec_with_phase(d, victim_task, max_time=max_time)
        per_phase: Dict[int, List[float]] = {p: [] for p in range(1, num_phases + 1)}
        for _, lat, _, ph in recs:
            if ph in per_phase:
                per_phase[ph].append(lat)
        q_by_method[m] = [
            float(np.percentile(per_phase[p], quantile * 100)) if per_phase[p] else 0.0
            for p in range(1, num_phases + 1)
        ]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(num_phases)
    width = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        cfg = _policy_cfg(m, i)
        ax.bar(x + i * width, q_by_method[m], width,
               label=cfg.get("label", m),
               color=cfg.get("color", "#888"),
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([f"Phase {p+1}" for p in range(num_phases)])
    qlabel = f"p{int(quantile*100)}"
    ax.set_ylabel(f"Victim {qlabel} latency (ms)")
    ax.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_slo_violations_by_phase(
    policy_dirs: Dict[str, Path],
    victim_task: str,
    phase_bounds: List[float],
    out_path: Path,
    slo_ms: float = 100.0,
    max_time: Optional[float] = None,
) -> None:
    """Grouped bar: victim SLO violation rate per phase, one group per method."""
    if not policy_dirs or not phase_bounds:
        return
    methods = list(policy_dirs.keys())
    num_phases = len(phase_bounds)
    rates_by_method: Dict[str, List[float]] = {}
    for m, d in policy_dirs.items():
        recs = _load_latencies_with_phase(d, victim_task, max_time=max_time)
        per_phase: Dict[int, List[float]] = {p: [] for p in range(1, num_phases + 1)}
        for _, lat, ph in recs:
            if ph in per_phase:
                per_phase[ph].append(lat)
        rates_by_method[m] = [_violation_pct(per_phase[p], slo_ms)
                              for p in range(1, num_phases + 1)]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(num_phases)
    width = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        cfg = _policy_cfg(m, i)
        ax.bar(x + i * width, rates_by_method[m], width,
               label=cfg.get("label", m),
               color=cfg.get("color", "#888"),
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([f"Phase {p+1}" for p in range(num_phases)])
    ax.set_ylabel(f"Victim SLO violations (%)\nSLO = {slo_ms:.0f} ms")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_slo_burst_summary(
    policy_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_task: str,
    phase_bounds: List[float],
    out_path: Path,
    slo_ms: float = 100.0,
    burst_phase: Optional[int] = None,
    max_time: Optional[float] = None,
) -> None:
    """Two-panel burst-phase summary:
      Left:  victim SLO violation rate (lower is better — isolation).
      Right: aggressor goodput (completed req/s) during burst (higher is
             better — efficiency at protecting the victim).
    Aggressor goodput surfaces variation that a 100%-violation SLO bar hides.
    """
    if not policy_dirs or not phase_bounds:
        return
    num_phases = len(phase_bounds)
    if burst_phase is None:
        burst_phase = (num_phases // 2) + 1 if num_phases >= 3 else num_phases
    phase_start = 0.0 if burst_phase == 1 else phase_bounds[burst_phase - 2]
    phase_end   = phase_bounds[burst_phase - 1]
    phase_dur   = max(phase_end - phase_start, 1e-6)

    methods = list(policy_dirs.keys())
    victim_viol: List[float] = []
    aggr_goodput: List[float] = []
    for m, d in policy_dirs.items():
        v_lats = [lat for _, lat, ph in _load_latencies_with_phase(d, victim_task, max_time=max_time)
                  if ph == burst_phase]
        victim_viol.append(_violation_pct(v_lats, slo_ms))
        # Aggressor goodput = requests *completed* (send_t + latency) within
        # the burst window, regardless of when they were sent. This captures
        # how much aggressor work the system actually delivered during the
        # burst — varies across schedulers even when offered load is fixed.
        path = d / "latencies.csv"
        completed = 0
        if path.exists():
            with path.open() as f:
                for r in csv.DictReader(f):
                    if r.get("task") != aggressor_task:
                        continue
                    send_t = float(r["elapsed_sec"])
                    lat_s = float(r["latency_ms"]) / 1000.0
                    done_t = send_t + lat_s
                    if phase_start <= done_t < phase_end:
                        completed += 1
        aggr_goodput.append(completed / phase_dur)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(methods))
    colors = [_policy_cfg(m, i).get("color", "#888") for i, m in enumerate(methods)]
    labels = [_policy_cfg(m, i).get("label", m) for i, m in enumerate(methods)]

    axes[0].bar(x, victim_viol, color=colors, edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel(f"Victim SLO violations (%)\nSLO = {slo_ms:.0f} ms")
    axes[0].set_title(f"Isolation — phase {burst_phase} (burst)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", alpha=0.3, linewidth=0.4)
    for xi, v in zip(x, victim_viol):
        axes[0].text(xi, v + 2, f"{v:.0f}%", ha="center", fontsize=8)

    axes[1].bar(x, aggr_goodput, color=colors, edgecolor="black", linewidth=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("Aggressor goodput (req/s)")
    axes[1].set_title(f"Aggressor work — phase {burst_phase}")
    axes[1].grid(axis="y", alpha=0.3, linewidth=0.4)
    for xi, v in zip(x, aggr_goodput):
        axes[1].text(xi, v + max(aggr_goodput) * 0.01, f"{v:.1f}",
                     ha="center", fontsize=8)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _discover_schedulers(base: Path) -> List[str]:
    """Return scheduler names found as subdirectories of base (each has meta.json)."""
    found = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d / "meta.json").exists():
            found.append(d.name)
    return found


def _read_meta(policy_dirs: Dict[str, Path]) -> dict:
    for d in policy_dirs.values():
        meta_path = d / "meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
    return {}


def _resolve_max_time(meta: dict, num_phases: Optional[int]) -> Optional[float]:
    if num_phases is None:
        return None
    boundaries = meta.get("phase_boundaries_s", [])
    if num_phases <= len(boundaries):
        return boundaries[num_phases - 1]
    # Fallback: derive from equal-duration assumption
    if boundaries:
        phase_dur = boundaries[0]  # first boundary = duration of phase 1
        return phase_dur * num_phases
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    default_base = "experiments/noisy_neighbor/vision/results"
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-base",   default=default_base,
                        help="Directory containing per-scheduler subdirectories")
    parser.add_argument("--plot-dir",       default=None,
                        help="Output plot directory (default: <results-base>/plots)")
    parser.add_argument("--victim-task",    default="vocseg")
    parser.add_argument("--aggressor-task", default="nyudepth")
    parser.add_argument("--methods",        default=None,
                        help="Comma-separated list of methods to plot "
                             "(default: auto-discover from results-base). "
                             "Examples: fcfs,stfq,bfq  or  fcfs,bfq,no_sharing,no_sharing_tpc")
    parser.add_argument("--schedulers",     default=None,
                        help="Alias for --methods (kept for backward compatibility)")
    parser.add_argument("--num-phases",     type=int, default=None,
                        help="Limit plot to the first N phases")
    parser.add_argument("--throughput-methods", default="all",
                        help="Comma-separated methods to include in the timeseries "
                             "throughput plot (default: no_sharing,no_sharing_tpc). "
                             "Use 'all' to include every method from --methods.")
    parser.add_argument("--attainment-methods", default=None,
                        help="Comma-separated methods to include in the throughput "
                             "attainment bar chart (default: all from --methods).")
    parser.add_argument("--bin-size-s", type=float, default=1.0,
                        help="Bin width in seconds for throughput and attainment "
                             "plots (default: 1.0; use 0.1 for 100ms bins).")
    parser.add_argument("--slo-ms", type=float, default=100.0,
                        help="SLO threshold (ms) for SLO-violation plots. "
                             "Default 100ms; for vision/dinolarge use 200–300ms.")
    parser.add_argument("--burst-phase", type=int, default=None,
                        help="1-indexed phase to treat as burst for the SLO "
                             "summary plot (default: middle phase).")
    args = parser.parse_args()
    if args.bin_size_s <= 0:
        parser.error("--bin-size-s must be positive")

    apply_paper_style()

    base     = (SERVING_DIR / args.results_base).resolve()
    plot_dir = (
        (SERVING_DIR / args.plot_dir).resolve()
        if args.plot_dir
        else base / "plots"
    )

    # Resolve method list (--methods preferred, --schedulers kept as alias)
    methods_arg = args.methods or args.schedulers
    if methods_arg:
        method_list = [s.strip() for s in methods_arg.split(",") if s.strip()]
    else:
        method_list = _discover_schedulers(base)
        if not method_list:
            print(f"[Error] No method result directories found under {base}")
            return 1
        print(f"[Plot] Auto-discovered methods: {method_list}")

    # Sort by POLICY_ORDER so plots layer in canonical order
    method_list = sorted(method_list,
                         key=lambda s: POLICY_ORDER.index(s) if s in POLICY_ORDER else 999)
    policy_dirs: Dict[str, Path] = {s: base / s for s in method_list}

    # Read meta from the first available scheduler to get phase info
    meta = _read_meta(policy_dirs)

    # Resolve max_time cutoff from --num-phases
    max_time: Optional[float] = _resolve_max_time(meta, args.num_phases)
    if args.num_phases is not None:
        print(f"[Plot] Limiting to first {args.num_phases} phases (t ≤ {max_time:.0f}s)")

    # 1. Per-scheduler plots (victim + aggressor)
    for policy, d in policy_dirs.items():
        if not d.exists():
            print(f"[Info] Skipping {policy} — {d} not found")
            continue
        label = _policy_cfg(policy).get("label", policy)
        plot_scheduler(
            d, args.victim_task, args.aggressor_task,
            label,
            plot_dir / f"{policy}_victim_aggressor.png",
            max_time=max_time,
        )

    # 2. Combined victim-only plot
    plot_all_policies(
        policy_dirs, args.victim_task, args.aggressor_task,
        plot_dir / "latency.png",
        max_time=max_time,
    )

    # 3. Throughput timeseries plot — restricted to a subset of methods
    if args.throughput_methods.strip().lower() == "all":
        tput_methods = list(method_list)
    else:
        requested = [s.strip() for s in args.throughput_methods.split(",") if s.strip()]
        tput_methods = [m for m in requested if m in policy_dirs]
        missing = [m for m in requested if m not in policy_dirs]
        if missing:
            print(f"[Plot] throughput-methods skipped (not in --methods): {missing}")
    if tput_methods:
        tput_dirs = {m: policy_dirs[m] for m in tput_methods}
        plot_throughput(
            tput_dirs, args.victim_task, args.aggressor_task,
            plot_dir / "throughput.png",
            max_time=max_time,
            bin_size_s=args.bin_size_s,
        )
    else:
        print("[Plot] No methods selected for throughput timeseries — skipping")

    # 3b. Throughput attainment bar chart — across all (or selected) methods
    if args.attainment_methods:
        requested = [s.strip() for s in args.attainment_methods.split(",") if s.strip()]
        attain_methods = [m for m in requested if m in policy_dirs]
    else:
        attain_methods = list(method_list)
    if attain_methods:
        attain_dirs = {m: policy_dirs[m] for m in attain_methods}
        plot_throughput_attainment(
            attain_dirs, args.victim_task, args.aggressor_task,
            plot_dir / "throughput_attainment.png",
            max_time=max_time,
            bin_size_s=args.bin_size_s,
        )
        plot_system_throughput_vs_victim_p95(
            attain_dirs, args.victim_task, args.aggressor_task,
            plot_dir / "throughput_vs_victim_p95.png",
            max_time=max_time,
            bin_size_s=args.bin_size_s,
        )

    # 4 & 5 — need aggressor_rps_phases + phase_boundaries from meta
    agg_phases   = meta.get("aggressor_rps_phases", [])
    phase_bounds = meta.get("phase_boundaries_s", [])
    if args.num_phases is not None:
        agg_phases   = agg_phases[:args.num_phases]
        phase_bounds = phase_bounds[:args.num_phases]

    # 4. Batch composition plot
    logs_dir = base / "logs"
    if logs_dir.exists() and agg_phases:
        plot_batch_composition(
            logs_dir, policy_dirs, args.victim_task,
            agg_phases, phase_bounds,
            plot_dir / "batch_composition.png",
        )

    # 5. Per-phase p50 summary bar chart
    if agg_phases:
        plot_phase_summary(
            {p: d for p, d in policy_dirs.items() if d.exists()},
            args.victim_task, agg_phases,
            plot_dir / "phase_summary.png",
        )

    # 6. Latency component stacked bars per phase (uses latencies.csv)
    if agg_phases:
        plot_latency_components_by_phase(
            {p: d for p, d in policy_dirs.items() if d.exists()},
            args.victim_task, agg_phases,
            plot_dir / "latency_components_by_phase.png",
            max_time=max_time,
        )

    # 7. P99 per phase per policy (uses latencies.csv)
    if agg_phases:
        plot_p99_phase_summary(
            {p: d for p, d in policy_dirs.items() if d.exists()},
            args.victim_task, agg_phases,
            plot_dir / "p99_phase_summary.png",
            max_time=max_time,
        )

    # 8. Queue wait over time (uses latencies.csv)
    plot_queue_wait_timeseries(
        {p: d for p, d in policy_dirs.items() if d.exists()},
        args.victim_task,
        plot_dir / "queue_wait_timeseries.png",
        max_time=max_time,
    )

    # 9a. Victim avg server_exec per phase — kernel-contention story (SLO-free)
    if phase_bounds:
        plot_victim_exec_by_phase(
            {p: d for p, d in policy_dirs.items() if d.exists()},
            args.victim_task, phase_bounds,
            plot_dir / "victim_exec_by_phase.png",
            max_time=max_time,
        )
        # 9b. Victim p95 per phase — distribution-shift story (SLO-free)
        plot_victim_latency_quantiles_by_phase(
            {p: d for p, d in policy_dirs.items() if d.exists()},
            args.victim_task, phase_bounds,
            plot_dir / "victim_p95_by_phase.png",
            quantile=0.95, max_time=max_time,
        )

    # 9c. Victim SLO violation rate per phase (isolation story)
    if phase_bounds:
        slo_tag = f"slo{int(args.slo_ms)}ms"
        plot_slo_violations_by_phase(
            {p: d for p, d in policy_dirs.items() if d.exists()},
            args.victim_task, phase_bounds,
            plot_dir / f"slo_violations_by_phase_{slo_tag}.png",
            slo_ms=args.slo_ms, max_time=max_time,
        )
        plot_slo_burst_summary(
            {p: d for p, d in policy_dirs.items() if d.exists()},
            args.victim_task, args.aggressor_task, phase_bounds,
            plot_dir / f"slo_burst_summary_{slo_tag}.png",
            slo_ms=args.slo_ms, burst_phase=args.burst_phase,
            max_time=max_time,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
