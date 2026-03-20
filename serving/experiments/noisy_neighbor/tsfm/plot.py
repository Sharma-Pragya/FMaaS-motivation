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
import numpy as np

SERVING_DIR = Path(__file__).resolve().parents[3]

PALETTE = {
    "charcoal":    "#2F3640",
    "slate":       "#5C6773",
    "grid":        "#D9DEE5",
    "background":  "#FAFBFC",
}

# Phase shading colours — cycles if there are more phases than colours
PHASE_COLORS = ["#EAF4FB", "#FFF3CD", "#FFE5E5", "#E8F8E8",
                "#F3E8FF", "#FFF0E8", "#E8FFF0", "#FFFBE8"]

# Style registry for known schedulers; unknown ones get auto-assigned colours/styles
POLICIES: Dict[str, Dict] = {
    "fifo":           {"color": "#D65F5F", "label": "FIFO",           "ls": "-"},
    "round_robin":    {"color": "#4878CF", "label": "Round-Robin",    "ls": "--"},
    "wfq":            {"color": "#4DAF4A", "label": "WFQ",            "ls": "-."},
    "stfq":           {"color": "#8B5CF6", "label": "STFQ",          "ls": (0, (4, 1, 1, 1))},
    "token_bucket":   {"color": "#FF7F00", "label": "Token Bucket",  "ls": (0, (3, 1, 1, 1))},
    "saba":           {"color": "#9B59B6", "label": "SABA",           "ls": (0, (5, 1))},
    "deadline_split": {"color": "#E67E22", "label": "DeadlineSplit",  "ls": (0, (1, 1))},
}

# Fallback colour/style pools for unknown scheduler names
_FALLBACK_COLORS = ["#1ABC9C", "#F39C12", "#2ECC71", "#E74C3C", "#3498DB"]
_FALLBACK_LS     = [(0, (2, 1)), (0, (6, 2)), (0, (3, 2, 1, 2)), "-.", "--"]

VICTIM_COLOR    = "#D65F5F"
AGGRESSOR_COLOR = "#4878CF"

Record = Tuple[float, float]  # (send_time_s, latency_ms)


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
        "figure.facecolor":  "white",
        "axes.facecolor":    PALETTE["background"],
        "axes.edgecolor":    PALETTE["slate"],
        "axes.labelcolor":   PALETTE["charcoal"],
        "axes.linewidth":    0.9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "grid.color":        PALETTE["grid"],
        "grid.linestyle":    "--",
        "grid.linewidth":    0.7,
        "grid.alpha":        0.8,
        "xtick.color":       PALETTE["charcoal"],
        "ytick.color":       PALETTE["charcoal"],
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "savefig.facecolor": "white",
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
    """Shade each phase and add vertical lines at transitions."""
    boundaries = [0.0] + phase_boundaries
    for i in range(len(phase_boundaries)):
        x0 = boundaries[i]
        x1 = min(boundaries[i + 1], xlim_max)
        ax.axvspan(x0, x1, color=PHASE_COLORS[i % len(PHASE_COLORS)],
                   alpha=0.35, zorder=0)

    for i, bnd in enumerate(phase_boundaries[:-1]):  # skip last (end of experiment)
        rps = aggressor_rps_phases[i + 1]
        ax.axvline(bnd, color=PALETTE["charcoal"], linewidth=1.2,
                   linestyle=":", zorder=4)
        ax.text(bnd + 0.5, ylim_max * 0.97,
                f"agg→{rps:.0f}", fontsize=8, color=PALETTE["charcoal"],
                va="top", ha="left")


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

    xlim_max = max_time if max_time is not None else max(
        max((r[0] for r in victim_recs),    default=0),
        max((r[0] for r in aggressor_recs), default=0),
    ) + 1

    all_smoothed_vals = []
    v_smooth = a_smooth = None
    if victim_recs:
        vt = [r[0] for r in victim_recs]
        vl = [r[1] * scale for r in victim_recs]
        v_smooth = _smooth(vt, vl)
        all_smoothed_vals.extend(v_smooth[1].tolist())
    if aggressor_recs:
        at = [r[0] for r in aggressor_recs]
        al = [r[1] * scale for r in aggressor_recs]
        a_smooth = _smooth(at, al)
        all_smoothed_vals.extend(a_smooth[1].tolist())
    ylim_max = max(all_smoothed_vals) * 1.1 if all_smoothed_vals else 1.0

    fig, ax = plt.subplots(figsize=(9.0, 4.5))

    if victim_recs and v_smooth is not None:
        ax.scatter(vt, vl, s=3, color=VICTIM_COLOR, alpha=0.12, zorder=2)
        ax.plot(v_smooth[0], v_smooth[1], color=VICTIM_COLOR, linewidth=2.2,
                linestyle="-", zorder=3, label=f"Victim ({victim_task})")

    if aggressor_recs and a_smooth is not None:
        ax.scatter(at, al, s=3, color=AGGRESSOR_COLOR, alpha=0.12, zorder=2)
        ax.plot(a_smooth[0], a_smooth[1], color=AGGRESSOR_COLOR, linewidth=2.2,
                linestyle="--", zorder=3, label=f"Aggressor ({aggressor_task})")

    if phase_boundaries:
        _add_phase_annotations(ax, phase_boundaries, aggressor_rps_phases, xlim_max, ylim_max)

    ax.set_xlim(0, xlim_max)
    ax.set_ylim(0, ylim_max)
    ax.set_xlabel("Time (s)", fontsize=12, fontweight="semibold")
    ax.set_ylabel(f"Latency ({unit})", fontsize=12, fontweight="semibold")
    ax.set_title(
        f"[{scheduler_label}]  Victim vs Aggressor Latency — {n_phases}-Phase Ramp",
        fontsize=11, fontweight="semibold", pad=8,
    )
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    fig.tight_layout()
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
    data: Dict[str, List[Record]] = {}
    meta_ref: Dict[str, dict] = {}

    for policy, d in policy_dirs.items():
        if not d.exists():
            print(f"[Info] {policy} results not found at {d} — skipping")
            continue
        recs, meta = load_task(d, victim_task, max_time)
        if recs:
            data[policy]     = recs
            meta_ref[policy] = meta

    if not data:
        print("[Error] No data found for any policy.")
        return

    all_lats = [r[1] for recs in data.values() for r in recs]
    scale    = 1 / 1000 if max(all_lats) > 2000 else 1.0
    unit     = "s" if scale < 1 else "ms"

    meta                 = next(iter(meta_ref.values()))
    phase_boundaries     = meta.get("phase_boundaries_s", [])
    aggressor_rps_phases = meta.get("aggressor_rps_phases", [])
    if max_time is not None:
        phase_boundaries     = [b for b in phase_boundaries if b <= max_time]
        aggressor_rps_phases = aggressor_rps_phases[:len(phase_boundaries)]
    xlim_max = max_time if max_time is not None else max(r[0] for recs in data.values() for r in recs) + 1

    smoothed = {}
    all_smoothed_vals = []
    for policy, recs in data.items():
        times = [r[0] for r in recs]
        lats  = [r[1] * scale for r in recs]
        sm    = _smooth(times, lats)
        smoothed[policy] = (times, lats, sm)
        all_smoothed_vals.extend(sm[1].tolist())
    ylim_max = max(all_smoothed_vals) * 1.1 if all_smoothed_vals else 1.0

    fig, ax = plt.subplots(figsize=(9.0, 4.5))

    for idx, (policy, (times, lats, sm)) in enumerate(smoothed.items()):
        cfg = _policy_cfg(policy, idx)
        ax.scatter(times, lats, s=3, color=cfg["color"], alpha=0.10, zorder=2)
        ax.plot(sm[0], sm[1], color=cfg["color"], linewidth=2.2,
                linestyle=cfg["ls"], zorder=3, label=cfg["label"])

    if phase_boundaries:
        _add_phase_annotations(ax, phase_boundaries, aggressor_rps_phases, xlim_max, ylim_max)

    ax.set_xlim(0, xlim_max)
    ax.set_ylim(0, ylim_max)
    ax.set_xlabel("Time (s)", fontsize=12, fontweight="semibold")
    ax.set_ylabel(f"Victim Latency ({unit})", fontsize=12, fontweight="semibold")
    ax.set_title(
        f"Noisy Neighbor — Victim ({victim_task}) Latency  |  All Schedulers",
        fontsize=11, fontweight="semibold", pad=8,
    )
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    fig.tight_layout()
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

    fig, ax = plt.subplots(figsize=(9.0, 4.5))

    for i, (policy, fracs) in enumerate(composition.items()):
        cfg    = _policy_cfg(policy, i)
        offset = (i - n_policies / 2 + 0.5) * w
        bars   = ax.bar(x + offset, fracs, width=w,
                        color=cfg["color"], alpha=0.85,
                        edgecolor=PALETTE["charcoal"], linewidth=0.7,
                        label=cfg["label"])
        for bar, v in zip(bars, fracs):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=7,
                        color=PALETTE["charcoal"])

    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, fontsize=10)
    ax.set_xlabel("Aggressor Load Phase", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Mean Victim Fraction per Batch", fontsize=12, fontweight="semibold")
    ax.set_title(
        f"Batch Composition: Victim ({victim_task}) Slot Share by Policy",
        fontsize=11, fontweight="semibold", pad=8,
    )
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
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

    fig, ax = plt.subplots(figsize=(9.0, 4.5))

    for i, (policy, vals) in enumerate(p50s.items()):
        cfg    = _policy_cfg(policy, i)
        offset = (i - n_policies / 2 + 0.5) * w
        scaled = [v * scale for v in vals]
        ax.bar(x + offset, scaled, width=w,
               color=cfg["color"], alpha=0.85,
               edgecolor=PALETTE["charcoal"], linewidth=0.7,
               label=cfg["label"])

    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, fontsize=10)
    ax.set_xlabel("Aggressor Load Phase", fontsize=12, fontweight="semibold")
    ax.set_ylabel(f"Victim P50 Latency ({unit})", fontsize=12, fontweight="semibold")
    ax.set_title(
        f"Victim ({victim_task}) P50 Latency per Phase — All Policies",
        fontsize=11, fontweight="semibold", pad=8,
    )
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Throughput plot: completed requests/s over time, all policies overlaid
# ---------------------------------------------------------------------------

def _sliding_window_rate(
    times_sorted: np.ndarray,
    window_s: float,
    max_time: float,
    n_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate req/s on a dense time grid using a centred sliding window."""
    half   = window_s / 2.0
    t_grid = np.linspace(0.0, max_time, n_points)
    rps    = np.empty(len(t_grid), dtype=float)
    for i, t in enumerate(t_grid):
        lo = max(0.0, t - half)
        hi = min(max_time, t + half)
        effective_window = hi - lo
        count = (np.searchsorted(times_sorted, hi, side="right")
                 - np.searchsorted(times_sorted, lo, side="left"))
        rps[i] = count / effective_window if effective_window > 0 else 0.0
    return t_grid, rps


def _compute_throughput(
    recs: List[Record],
    window_s: float = 5.0,
    max_time: Optional[float] = None,
    n_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Completed req/s — uses completion time (send + latency)."""
    if not recs:
        return np.array([]), np.array([])
    times = np.array(sorted(r[0] + r[1] / 1000.0 for r in recs))
    end   = max_time if max_time is not None else times[-1]
    return _sliding_window_rate(times, window_s, end, n_points)


def _compute_offered_load(
    recs: List[Record],
    window_s: float = 5.0,
    max_time: Optional[float] = None,
    n_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Actual offered load from send times using the same window as throughput."""
    if not recs:
        return np.array([]), np.array([])
    times = np.array(sorted(r[0] for r in recs))
    end   = max_time if max_time is not None else times[-1]
    return _sliding_window_rate(times, window_s=window_s, max_time=end, n_points=n_points)


def plot_throughput(
    policy_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_task: str,
    out_path: Path,
    window_s: float = 5.0,
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
            victim_data[policy]    = _compute_throughput(v_recs, window_s, max_time)
            meta_ref = meta
        if a_recs:
            aggressor_data[policy] = _compute_throughput(a_recs, window_s, max_time)

    if not victim_data and not aggressor_data:
        print("[Plot] No throughput data found — skipping")
        return

    phase_boundaries     = meta_ref.get("phase_boundaries_s", [])
    aggressor_rps_phases = meta_ref.get("aggressor_rps_phases", [])
    xlim_max = max_time if max_time is not None else max(
        max((c.max() for c, _ in victim_data.values()),    default=0),
        max((c.max() for c, _ in aggressor_data.values()), default=0),
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
            vic_offered = _compute_offered_load(
                [(t, 0.0) for t in sends], window_s, end)
        if aggressor_task in trace:
            sends = [t for t in trace[aggressor_task] if max_time is None or t <= max_time]
            agg_offered = _compute_offered_load(
                [(t, 0.0) for t in sends], window_s, end)
    else:
        # Fallback: use send times from first available policy's CSV
        for _, d in policy_dirs.items():
            if not d.exists():
                continue
            v_recs_raw, _ = load_task(d, victim_task, max_time)
            a_recs_raw, _ = load_task(d, aggressor_task, max_time)
            if v_recs_raw:
                vic_offered = _compute_offered_load(v_recs_raw, window_s, max_time)
            if a_recs_raw:
                agg_offered = _compute_offered_load(a_recs_raw, window_s, max_time)
            break

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)

    for ax, task_data, title, offered in [
        (axes[0], victim_data,    f"Victim ({victim_task}) Throughput",       vic_offered),
        (axes[1], aggressor_data, f"Aggressor ({aggressor_task}) Throughput", agg_offered),
    ]:
        all_rps = [r for _, rps in task_data.values() for r in rps]
        offered_max = float(offered[1].max()) if offered is not None and len(offered[1]) else 0.0
        ylim_max = max(all_rps + [offered_max]) * 1.15 if (all_rps or offered_max) else 1.0

        for idx, (policy, (centers, rps)) in enumerate(task_data.items()):
            cfg = _policy_cfg(policy, idx)
            ax.plot(centers, rps, color=cfg["color"], linewidth=2.0,
                    linestyle=cfg["ls"], label=cfg["label"], zorder=3)

        if offered is not None:
            ax.plot(offered[0], offered[1], color=PALETTE["charcoal"],
                    linewidth=1.5, linestyle="--", label="Offered load", zorder=5)

        boundaries = [0.0] + phase_boundaries
        for i in range(len(phase_boundaries)):
            x0 = boundaries[i]
            x1 = min(boundaries[i + 1] if i + 1 < len(boundaries) else xlim_max, xlim_max)
            ax.axvspan(x0, x1, color=PHASE_COLORS[i % len(PHASE_COLORS)], alpha=0.30, zorder=0)
        for bnd in phase_boundaries[:-1]:
            ax.axvline(bnd, color=PALETTE["charcoal"], linewidth=1.0, linestyle=":", zorder=4)

        ax.set_xlim(0, xlim_max)
        ax.set_ylim(0, ylim_max)
        ax.set_ylabel("Req/s", fontsize=11, fontweight="semibold")
        ax.set_title(title, fontsize=11, fontweight="semibold", pad=6)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, frameon=False, loc="upper left")

    axes[1].set_xlabel("Time (s)", fontsize=12, fontweight="semibold")
    fig.suptitle("Throughput Over Time — All Schedulers", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
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
    default_base = "experiments/noisy_neighbor/tsfm_inaction/results"
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
    parser.add_argument("--throughput-window-s", type=float, default=5.0,
                        help="Sliding-window width in seconds for throughput smoothing (default: 5)")
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

    # Build policy_dirs — each scheduler maps to <base>/<scheduler>
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
        plot_dir / "all_policies_victim.png",
        max_time=max_time,
    )

    # 3. Throughput plot
    plot_throughput(
        policy_dirs, args.victim_task, args.aggressor_task,
        plot_dir / "throughput.png",
        window_s=args.throughput_window_s,
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
