#!/usr/bin/env python3
"""noisy_neighbor/llm — Time-series plots for vLLM noisy-neighbor runs.

Produces:
  1. One plot per policy/run directory — victim + aggressor latency over time.
  2. One combined latency plot — victim (top) and aggressor (bottom), all runs overlaid.
  3. Throughput over time — victim (top) and aggressor (bottom), all runs overlaid.
  4. Per-phase p50 victim latency bar chart.

Run from serving/:
    python experiments/noisy_neighbor/llm/plot.py \
        --results-base experiments/noisy_neighbor/llm/results \
        --plot-dir     experiments/noisy_neighbor/llm/results/plots
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SERVING_DIR = Path(__file__).resolve().parents[3]

PALETTE = {
    "charcoal": "#2F3640",
    "grid": "#D9DEE5",
}

POLICIES: Dict[str, Dict] = {
    "continuous_batching": {"color": "#6B9AC4", "label": "Continuous Batching", "ls": "-"},
    "fair_admission": {"color": "#E06C75", "label": "Fair Admission", "ls": "--"},
    "token_bucket": {"color": "#5BA890", "label": "Token Bucket", "ls": (0, (4, 1, 1, 1))},
}
POLICY_ORDER = ["continuous_batching", "fair_admission", "token_bucket"]
FALLBACK_COLORS = ["#E8B298", "#A1C181", "#9BB1FF", "#D4A5A5"]
FALLBACK_LS = ["-.", ":", (0, (3, 2)), (0, (5, 2))]

Record = Tuple[float, float]


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "text.color": "black",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[Plot] Saved: {out_path}")


def _policy_cfg(policy: str, idx: int = 0) -> Dict:
    if policy in POLICIES:
        return POLICIES[policy]
    return {
        "color": FALLBACK_COLORS[idx % len(FALLBACK_COLORS)],
        "label": policy,
        "ls": FALLBACK_LS[idx % len(FALLBACK_LS)],
    }


def _set_clean_ticks(ax: plt.Axes, xdata_max: float, ydata_max: float, n_y: int = 4) -> Tuple[float, float]:
    def _ticks_and_limit(data_max: float, n: int = 5) -> Tuple[np.ndarray, float]:
        step_raw = max(data_max / n, 1e-9)
        magnitude = 10 ** np.floor(np.log10(step_raw))
        nice = [1, 2, 2.5, 5, 10]
        step = magnitude * min(nice, key=lambda s: abs(s - step_raw / magnitude))
        nice_limit = np.ceil(max(data_max, step) / step) * step
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


def _bin_latency(times: np.ndarray, lats: np.ndarray, max_time: float) -> Tuple[np.ndarray, np.ndarray]:
    n_bins = int(np.ceil(max_time))
    sums = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=float)
    for t, l in zip(times, lats):
        idx = int(t)
        if 0 <= idx < n_bins:
            sums[idx] += l
            counts[idx] += 1.0
    means = np.where(counts > 0, sums / counts, np.nan)
    centers = np.arange(n_bins) + 0.5
    return centers, means


def _bin_rate(times: np.ndarray, max_time: float) -> Tuple[np.ndarray, np.ndarray]:
    n_bins = int(np.ceil(max_time))
    counts = np.zeros(n_bins, dtype=float)
    for t in times:
        idx = int(t)
        if 0 <= idx < n_bins:
            counts[idx] += 1.0
    centers = np.arange(n_bins) + 0.5
    return centers, counts


def load_task(results_dir: Path, task: str, max_time: Optional[float] = None) -> Tuple[List[Record], dict]:
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


def _compute_throughput(recs: List[Record], max_time: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    if not recs:
        return np.array([]), np.array([])
    times = np.array([t + lat / 1000.0 for t, lat in recs])
    end = max_time if max_time is not None else float(times.max())
    return _bin_rate(times, end)


def _compute_offered_load(recs: List[Record], max_time: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    if not recs:
        return np.array([]), np.array([])
    times = np.array([t for t, _ in recs])
    end = max_time if max_time is not None else float(times.max())
    return _bin_rate(times, end)


def _add_phase_annotations(ax: plt.Axes, phase_boundaries: List[float]) -> None:
    for bnd in phase_boundaries[:-1]:
        ax.axvline(bnd, color=PALETTE["charcoal"], linewidth=0.9, linestyle=":", zorder=4)


def plot_scheduler(
    results_dir: Path,
    victim_task: str,
    aggressor_task: str,
    out_path: Path,
    max_time: Optional[float] = None,
) -> None:
    victim_recs, meta = load_task(results_dir, victim_task, max_time)
    aggressor_recs, _ = load_task(results_dir, aggressor_task, max_time)
    if not victim_recs and not aggressor_recs:
        print(f"[Info] No data in {results_dir} - skipping")
        return

    phase_boundaries = meta.get("phase_boundaries_s", [])
    if max_time is not None:
        phase_boundaries = [b for b in phase_boundaries if b <= max_time]

    xlim_max = max_time if max_time is not None else (
        phase_boundaries[-1] if phase_boundaries else max(
            max((r[0] for r in victim_recs), default=0.0),
            max((r[0] for r in aggressor_recs), default=0.0),
        )
    )

    fig, axes = plt.subplots(2, 1, figsize=(2.8, 2.4), sharex=True)
    for ax, recs, title, color in [
        (axes[0], victim_recs, f"Victim ({victim_task})", "#6B9AC4"),
        (axes[1], aggressor_recs, f"Aggressor ({aggressor_task})", "#E06C75"),
    ]:
        if recs:
            times = np.array([r[0] for r in recs])
            lats = np.array([r[1] for r in recs])
            centers, means = _bin_latency(times, lats, xlim_max)
            ax.plot(centers, means, color=color, linewidth=1.2, zorder=3)
            ylim_max = max(float(np.nanmax(means)), 1.0)
        else:
            ylim_max = 1.0
        _set_clean_ticks(ax, xlim_max, ylim_max, n_y=4)
        _add_phase_annotations(ax, phase_boundaries)
        ax.set_ylabel("Latency (ms)")
        ax.set_title(title, pad=2)

    axes[1].set_xlabel("Time (s)")
    fig.tight_layout(pad=0.4)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_all_policies(
    policy_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_task: str,
    out_path: Path,
    max_time: Optional[float] = None,
) -> None:
    victim_data: Dict[str, List[Record]] = {}
    aggressor_data: Dict[str, List[Record]] = {}
    meta_ref: dict = {}

    for policy, d in policy_dirs.items():
        if not d.exists():
            continue
        v_recs, meta = load_task(d, victim_task, max_time)
        a_recs, _ = load_task(d, aggressor_task, max_time)
        if v_recs:
            victim_data[policy] = v_recs
            meta_ref = meta
        if a_recs:
            aggressor_data[policy] = a_recs

    if not victim_data and not aggressor_data:
        print("[Error] No data found for any policy.")
        return

    phase_boundaries = meta_ref.get("phase_boundaries_s", [])
    if max_time is not None:
        phase_boundaries = [b for b in phase_boundaries if b <= max_time]
    xlim_max = max_time if max_time is not None else (
        phase_boundaries[-1] if phase_boundaries else max(
            max((r[0] for recs in victim_data.values() for r in recs), default=0.0),
            max((r[0] for recs in aggressor_data.values() for r in recs), default=0.0),
        )
    )

    fig, axes = plt.subplots(2, 1, figsize=(2.8, 2.4), sharex=True)
    for ax, data, title in [
        (axes[0], victim_data, "Victim"),
        (axes[1], aggressor_data, "Aggressor"),
    ]:
        ylim_max = 1.0
        for idx, (policy, recs) in enumerate(data.items()):
            times = np.array([r[0] for r in recs])
            lats = np.array([r[1] for r in recs])
            centers, means = _bin_latency(times, lats, xlim_max)
            cfg = _policy_cfg(policy, idx)
            ax.plot(centers, means, color=cfg["color"], linestyle=cfg["ls"], label=cfg["label"], zorder=3)
            if np.any(~np.isnan(means)):
                ylim_max = max(ylim_max, float(np.nanmax(means)))
        _set_clean_ticks(ax, xlim_max, ylim_max, n_y=4)
        _add_phase_annotations(ax, phase_boundaries)
        ax.set_ylabel("Latency (ms)")
        ax.text(0.02, 0.96, title, transform=ax.transAxes, fontsize=6.5, va="top", ha="left")

    axes[1].set_xlabel("Time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(pad=0.4)
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=max(1, len(handles)), frameon=False)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_throughput(
    policy_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_task: str,
    out_path: Path,
    max_time: Optional[float] = None,
) -> None:
    victim_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    aggressor_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    meta_ref: dict = {}

    for policy, d in policy_dirs.items():
        if not d.exists():
            continue
        v_recs, meta = load_task(d, victim_task, max_time)
        a_recs, _ = load_task(d, aggressor_task, max_time)
        if v_recs:
            victim_data[policy] = _compute_throughput(v_recs, max_time)
            meta_ref = meta
        if a_recs:
            aggressor_data[policy] = _compute_throughput(a_recs, max_time)

    if not victim_data and not aggressor_data:
        print("[Plot] No throughput data found - skipping")
        return

    phase_boundaries = meta_ref.get("phase_boundaries_s", [])
    xlim_max = max_time if max_time is not None else (
        phase_boundaries[-1] if phase_boundaries else max(
            max((c.max() for c, _ in victim_data.values()), default=0.0),
            max((c.max() for c, _ in aggressor_data.values()), default=0.0),
        )
    )

    base = next((d.parent for d in policy_dirs.values() if d.exists()), None)
    trace_path = base / "trace.json" if base is not None else None
    vic_offered = agg_offered = None
    if trace_path is not None and trace_path.exists():
        trace = json.loads(trace_path.read_text())
        end = max_time if max_time is not None else xlim_max
        if victim_task in trace:
            vic_offered = _compute_offered_load([(t, 0.0) for t in trace[victim_task] if max_time is None or t <= max_time], end)
        if aggressor_task in trace:
            agg_offered = _compute_offered_load([(t, 0.0) for t in trace[aggressor_task] if max_time is None or t <= max_time], end)

    fig, axes = plt.subplots(2, 1, figsize=(2.8, 2.4), sharex=True)
    for ax, task_data, title, offered in [
        (axes[0], victim_data, "Victim", vic_offered),
        (axes[1], aggressor_data, "Aggressor", agg_offered),
    ]:
        ylim_max = max(
            [float(np.max(rps)) for _, rps in task_data.values() if len(rps)] +
            ([float(np.max(offered[1]))] if offered is not None and len(offered[1]) else [1.0])
        )
        for idx, (policy, (centers, rps)) in enumerate(task_data.items()):
            cfg = _policy_cfg(policy, idx)
            ax.plot(centers, rps, color=cfg["color"], linestyle=cfg["ls"], label=cfg["label"], zorder=3)
        if offered is not None:
            ax.plot(offered[0], offered[1], color=PALETTE["charcoal"], linewidth=0.8, linestyle=":", label="Offered load", zorder=4)
        _set_clean_ticks(ax, xlim_max, ylim_max, n_y=4)
        _add_phase_annotations(ax, phase_boundaries)
        ax.set_ylabel("Req/s")
        ax.text(0.02, 0.96, title, transform=ax.transAxes, fontsize=6.5, va="top", ha="left")

    axes[1].set_xlabel("Time (s)")
    fig.tight_layout(pad=0.4)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_phase_summary(
    policy_dirs: Dict[str, Path],
    victim_task: str,
    aggressor_rps_phases: List[float],
    out_path: Path,
) -> None:
    phase_labels = [f"agg={int(r)}rps" for r in aggressor_rps_phases]
    n_phases = len(phase_labels)
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

    n_policies = len(p50s)
    x = np.arange(n_phases)
    total_width = 0.7
    width = total_width / max(1, n_policies)

    fig, ax = plt.subplots(figsize=(2.8, 1.5))
    for i, (policy, vals) in enumerate(p50s.items()):
        cfg = _policy_cfg(policy, i)
        offset = (i - n_policies / 2 + 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            color=cfg["color"],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
            label=cfg["label"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels)
    ax.set_xlabel("Aggressor Load Phase")
    ax.set_ylabel("P50 Latency (ms)")
    ax.legend(frameon=False, handlelength=1.2)
    fig.tight_layout(pad=0.4)
    save_figure(fig, out_path)
    plt.close(fig)


def _discover_policies(base: Path) -> List[str]:
    found = []
    if not base.exists():
        return found
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
    return None


def main() -> int:
    default_base = "experiments/noisy_neighbor/llm/results"
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-base", default=default_base)
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--victim-task", default="llm_sst2")
    parser.add_argument("--aggressor-task", default="llm_ag_news")
    parser.add_argument("--policies", default=None, help="Comma-separated list of run directories to plot")
    parser.add_argument("--num-phases", type=int, default=None)
    args = parser.parse_args()

    apply_paper_style()

    base = (SERVING_DIR / args.results_base).resolve()
    plot_dir = (SERVING_DIR / args.plot_dir).resolve() if args.plot_dir else base / "plots"

    if args.policies:
        policy_list = [p.strip() for p in args.policies.split(",") if p.strip()]
    else:
        policy_list = _discover_policies(base)
        if not policy_list:
            print(f"[Error] No policy result directories found under {base}")
            return 1
        print(f"[Plot] Auto-discovered policies: {policy_list}")

    policy_list = sorted(policy_list, key=lambda p: POLICY_ORDER.index(p) if p in POLICY_ORDER else 999)
    policy_dirs: Dict[str, Path] = {p: base / p for p in policy_list}
    meta = _read_meta(policy_dirs)
    max_time = _resolve_max_time(meta, args.num_phases)

    for policy, d in policy_dirs.items():
        if not d.exists():
            print(f"[Info] Skipping {policy} - {d} not found")
            continue
        plot_scheduler(
            d,
            args.victim_task,
            args.aggressor_task,
            plot_dir / f"{policy}_victim_aggressor.png",
            max_time=max_time,
        )

    plot_all_policies(
        policy_dirs,
        args.victim_task,
        args.aggressor_task,
        plot_dir / "latency.png",
        max_time=max_time,
    )
    plot_throughput(
        policy_dirs,
        args.victim_task,
        args.aggressor_task,
        plot_dir / "throughput.png",
        max_time=max_time,
    )

    agg_phases = meta.get("aggressor_rps_phases", [])
    if args.num_phases is not None:
        agg_phases = agg_phases[:args.num_phases]
    if agg_phases:
        plot_phase_summary(
            {p: d for p, d in policy_dirs.items() if d.exists()},
            args.victim_task,
            agg_phases,
            plot_dir / "phase_summary.png",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
