#!/usr/bin/env python3
"""noisy_neighbor/tsfm_inaction — Time-series plots.

Produces:
  1. One plot per scheduler — victim + aggressor latency over time,
     with vertical lines at each phase transition.
  2. One combined plot — victim latency only, all schedulers overlaid.
  3. Throughput over time — all schedulers overlaid.
  4. Batch composition — victim slot share per phase per policy.
  5. Per-phase p50 summary bar chart.

Run from serving/:
    python experiments/noisy_neighbor/tsfm_inaction/plot.py \
        --results-base experiments/noisy_neighbor/tsfm_inaction/results \
        --plot-dir     experiments/noisy_neighbor/tsfm_inaction/plots

Select which schedulers to plot:
    --schedulers fifo,wfq,stfq

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
    "fcfs":  {"color": "#6B9AC4", "label": "FCFS",  "ls": "-"},
    "stfq":  {"color": "#E8B298", "label": "STFQ",  "ls": (0, (4, 1, 1, 1))},
    "bfq":   {"color": "#E06C75", "label": "BFQ",   "ls": "--"},
}

# Draw order: FCFS first (bottom) → STFQ → BFQ last (top)
POLICY_ORDER = ["fcfs", "stfq", "bfq"]

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
        ax.axvline(bnd, color=PALETTE["charcoal"], linewidth=0.8,
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

    ylim_cap = 200 * scale
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Count events in exact 1s bins [0,1), [1,2), ..., returning (bin_centers, req/s).
    Each bin covers exactly 1 second so count == req/s directly."""
    n_bins  = int(np.ceil(max_time))
    counts  = np.zeros(n_bins, dtype=float)
    for t in times:
        idx = int(t)
        if 0 <= idx < n_bins:
            counts[idx] += 1.0
    centers = np.arange(n_bins) + 0.5   # centre of each 1s bin
    return centers, counts


def _bin_latency(
    times: np.ndarray,
    lats: np.ndarray,
    max_time: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean latency in exact 1s bins [0,1), [1,2), ..., returning (bin_centers, mean_lat).
    Bins with no requests are omitted (NaN-masked so line breaks naturally)."""
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Completed req/s in 1s bins — uses completion time (send + latency)."""
    if not recs:
        return np.array([]), np.array([])
    times = np.array([r[0] + r[1] / 1000.0 for r in recs])
    end   = max_time if max_time is not None else float(times.max())
    return _bin_rate(times, end)


def _compute_offered_load(
    recs: List[Record],
    max_time: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Offered load from send times in 1s bins."""
    if not recs:
        return np.array([]), np.array([])
    times = np.array([r[0] for r in recs])
    end   = max_time if max_time is not None else float(times.max())
    return _bin_rate(times, end)


def plot_throughput(
    policy_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_task: str,
    out_path: Path,
    max_time: Optional[float] = None,
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
            victim_data[policy]    = _compute_throughput(v_recs, max_time)
            meta_ref = meta
        if a_recs:
            aggressor_data[policy] = _compute_throughput(a_recs, max_time)

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
            vic_offered = _compute_offered_load([(t, 0.0) for t in sends], end)
        if aggressor_task in trace:
            sends = [t for t in trace[aggressor_task] if max_time is None or t <= max_time]
            agg_offered = _compute_offered_load([(t, 0.0) for t in sends], end)
    else:
        # Fallback: use send times from first available policy's CSV
        for _, d in policy_dirs.items():
            if not d.exists():
                continue
            v_recs_raw, _ = load_task(d, victim_task, max_time)
            a_recs_raw, _ = load_task(d, aggressor_task, max_time)
            if v_recs_raw:
                vic_offered = _compute_offered_load(v_recs_raw, max_time)
            if a_recs_raw:
                agg_offered = _compute_offered_load(a_recs_raw, max_time)
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

        ax.set_xlim(0, xlim_max)
        _set_clean_ticks(ax, xlim_max, ylim_max)
        ax.set_ylabel("Req/s")
        ax.text(0.02, 0.96, task_labels[task_name], transform=ax.transAxes,
                fontsize=6.5, va="top", ha="left", color=PALETTE["charcoal"])

    axes[1].set_xlabel("Time (s)")
    # # Single shared legend above the figure with 4 columns
    # handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(pad=0.4)
    # leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
    #                  ncol=4, frameon=False, handlelength=1.2, columnspacing=0.8)
    # fig.subplots_adjust(top=1.0 - (leg.get_window_extent(fig.canvas.get_renderer()).height
    #                                / fig.get_window_extent().height) - 0.04)
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    default_base = "experiments/noisy_neighbor/tsfm_withadapters/results"
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-base",   default=default_base,
                        help="Directory containing per-scheduler subdirectories")
    parser.add_argument("--plot-dir",       default=None,
                        help="Output plot directory (default: <results-base>/plots)")
    parser.add_argument("--victim-task",    default="ecgclass")
    parser.add_argument("--aggressor-task", default="gestureclass")
    parser.add_argument("--schedulers",     default=None,
                        help="Comma-separated list of schedulers to plot "
                             "(default: auto-discover from results-base). "
                             "Example: fifo,wfq,stfq")
    parser.add_argument("--num-phases",     type=int, default=None,
                        help="Limit plot to the first N phases")
    args = parser.parse_args()

    apply_paper_style()

    base     = (SERVING_DIR / args.results_base).resolve()
    plot_dir = (
        (SERVING_DIR / args.plot_dir).resolve()
        if args.plot_dir
        else base / "plots"
    )

    # Resolve scheduler list
    if args.schedulers:
        scheduler_list = [s.strip() for s in args.schedulers.split(",") if s.strip()]
    else:
        scheduler_list = _discover_schedulers(base)
        if not scheduler_list:
            print(f"[Error] No scheduler result directories found under {base}")
            return 1
        print(f"[Plot] Auto-discovered schedulers: {scheduler_list}")

    # Build policy_dirs — sort by POLICY_ORDER so plots layer FCFS→STFQ→BFQ
    scheduler_list = sorted(scheduler_list,
                            key=lambda s: POLICY_ORDER.index(s) if s in POLICY_ORDER else 999)
    policy_dirs: Dict[str, Path] = {s: base / s for s in scheduler_list}

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

    # 3. Throughput plot
    plot_throughput(
        policy_dirs, args.victim_task, args.aggressor_task,
        plot_dir / "throughput.png",
        max_time=max_time,
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
