#!/usr/bin/env python3
"""fair_share/llm — fairness, throughput, and latency plots.

Two methods in scope:
  vllm  — Punica-class baseline (vLLM continuous batching, no fair share)
  bfq   — FMVisor: STFQ admission scheduler in front of vLLM (reuses
          device/scheduler.STFQPolicy; see device/vllm_admission.py)

For each operator weight ratio (1:1, 2:1, 3:1) produces:
  fairness_<tag>.{pdf,png}    — twin-axis bars: fairness + throughput in phase 2
  throughput_<tag>.{pdf,png}  — two-panel throughput timeseries
  latency_<tag>.{pdf,png}     — two-panel latency timeseries

Run from serving/:
    python experiments/fair_share/llm/plot.py
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
# Style (mirrors fair_share/tsfm/plot.py palette so the LLM figure sits
# alongside the TSFM ones in the paper without colour clashes)
# ---------------------------------------------------------------------------

COLORS = {
    "vllm": "#F0A500",   # amber/orange — baseline
    "bfq":  "#E06C75",   # pink-red     — FMVisor
}

LABELS = {
    "vllm": "vLLM (Punica-class)",
    "bfq":  "FMVisor",
}

LINESTYLES = {
    "vllm": (0, (3, 1, 1, 1)),  # dash-dot-dot
    "bfq":  "-",
}

MARKERS = {
    "vllm": "P",  # plus-filled
    "bfq":  "o",  # circle
}

METHOD_ORDER = ["vllm", "bfq"]


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
# Data loading + math (kept self-contained; same semantics as tsfm/plot.py)
# ---------------------------------------------------------------------------

def _read_meta(results_dir: Path) -> dict:
    p = results_dir / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _load_records(results_dir: Path, task: str) -> List[Tuple[float, float]]:
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


def _completions_in_window(recs, t_start: float, t_end: float) -> int:
    return sum(1 for s, l in recs if t_start <= (s + l / 1000.0) < t_end)


def _phase2_window(meta: dict) -> Tuple[float, float]:
    bounds = meta.get("phase_boundaries_s", [])
    if len(bounds) >= 2:
        return float(bounds[0]), float(bounds[1])
    return 0.0, 0.0


def _add_phase_lines(ax: plt.Axes, meta: dict, t_max: float) -> None:
    for b in meta.get("phase_boundaries_s", [])[:-1]:
        if b < t_max:
            ax.axvline(b, color="black", linewidth=0.4, linestyle=":", zorder=2)


SATISFIED_TOL = 0.95


def _weighted_maxmin_ideal(d_a: float, d_b: float,
                           w_a: float, w_b: float,
                           capacity: float) -> Tuple[float, float]:
    if capacity <= 0:
        return 0.0, 0.0
    if d_a / max(w_a, 1e-12) <= d_b / max(w_b, 1e-12):
        base_a = w_a * capacity / (w_a + w_b)
        if d_a <= base_a:
            return d_a, min(d_b, capacity - d_a)
        return base_a, capacity - base_a
    base_b = w_b * capacity / (w_a + w_b)
    if d_b <= base_b:
        return min(d_a, capacity - d_b), d_b
    return capacity - base_b, base_b


def minmax_fairness(a_recs, b_recs,
                    offered_a: float, offered_b: float,
                    w_a: float, w_b: float,
                    t_start: float, t_end: float) -> float:
    """Hybrid weighted max-min fairness (same shape as tsfm/plot.py)."""
    dur = t_end - t_start
    if dur <= 0 or offered_a <= 0 or offered_b <= 0:
        return float("nan")
    T_a = _completions_in_window(a_recs, t_start, t_end) / dur
    T_b = _completions_in_window(b_recs, t_start, t_end) / dur
    if T_a >= SATISFIED_TOL * offered_a and T_b >= SATISFIED_TOL * offered_b:
        return 1.0
    cap = T_a + T_b
    if cap <= 0 or w_a <= 0 or w_b <= 0:
        return float("nan")
    ideal_a, ideal_b = _weighted_maxmin_ideal(offered_a, offered_b, w_a, w_b, cap)
    if ideal_a <= 0 or ideal_b <= 0:
        return float("nan")
    r_a = min(T_a / ideal_a, 1.0)
    r_b = min(T_b / ideal_b, 1.0)
    if r_a <= 0 and r_b <= 0:
        return float("nan")
    return min(r_a, r_b) / max(r_a, r_b)


def _offered_rate_from_trace(trace_path: Path, task: str,
                             t_start: float, t_end: float) -> float:
    if not trace_path.exists() or t_end <= t_start:
        return 0.0
    raw = json.loads(trace_path.read_text())
    times = raw.get(task, [])
    n = sum(1 for t in times if t_start <= float(t) < t_end)
    return n / (t_end - t_start)


def _bin_rate(times: np.ndarray, t_max: float, bin_s: float = 1.0):
    n_bins = int(np.ceil(t_max / bin_s))
    counts = np.zeros(n_bins, dtype=float)
    for t in times:
        idx = int(t / bin_s)
        if 0 <= idx < n_bins:
            counts[idx] += 1.0
    centers = (np.arange(n_bins) + 0.5) * bin_s
    return centers, counts / bin_s


def _bin_mean(times: np.ndarray, vals: np.ndarray, t_max: float, bin_s: float = 1.0):
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


def _nice_ceil(v: float) -> float:
    if v <= 0 or not np.isfinite(v):
        return 1.0
    mag = 10.0 ** np.floor(np.log10(v))
    frac = v / mag
    for cap in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if frac <= cap + 1e-9:
            return cap * mag
    return 10.0 * mag


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_fairness_summary(method_dirs: Dict[str, Path],
                          victim_task: str, aggressor_task: str,
                          w_a: float, w_b: float,
                          meta: dict, out_path: Path) -> None:
    methods = [m for m in METHOD_ORDER if m in method_dirs]
    if not methods:
        return
    p_start, p_end = _phase2_window(meta)
    p_dur = max(p_end - p_start, 1e-6)

    base = next(iter(method_dirs.values())).parent
    trace_path = base / "trace.json"
    offered_a = _offered_rate_from_trace(trace_path, victim_task,    p_start, p_end)
    offered_b = _offered_rate_from_trace(trace_path, aggressor_task, p_start, p_end)

    fairness, sys_rps = [], []
    for m in methods:
        a_recs = _load_records(method_dirs[m], victim_task)
        b_recs = _load_records(method_dirs[m], aggressor_task)
        f = minmax_fairness(a_recs, b_recs, offered_a, offered_b,
                            w_a, w_b, p_start, p_end)
        fairness.append(f if np.isfinite(f) else 0.0)
        T_a = _completions_in_window(a_recs, p_start, p_end) / p_dur
        T_b = _completions_in_window(b_recs, p_start, p_end) / p_dur
        sys_rps.append(T_a + T_b)

    labels = [LABELS[m] for m in methods]
    bar_w = 0.38
    x = np.arange(len(methods))

    fig, ax_left = plt.subplots(figsize=(4.0, 3.4))
    ax_right = ax_left.twinx()
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(True)

    FAIR_COLOR = "#34495E"
    TPUT_COLOR = "#E67E22"

    ax_left.bar(x - bar_w / 2, fairness, width=bar_w, color=FAIR_COLOR,
                edgecolor="black", linewidth=0.6, zorder=3)
    ax_right.bar(x + bar_w / 2, sys_rps, width=bar_w, color=TPUT_COLOR,
                 edgecolor="black", linewidth=0.6, zorder=3)

    ax_left.set_ylabel("Fairness", color=FAIR_COLOR, fontweight="bold")
    ax_left.tick_params(axis="y", colors=FAIR_COLOR)
    ax_right.set_ylabel("Throughput (rps)", color=TPUT_COLOR, fontweight="bold")
    ax_right.tick_params(axis="y", colors=TPUT_COLOR)

    sys_top = _nice_ceil(max(sys_rps, default=1.0))
    ax_left.set_ylim(0, 1.12)
    ax_right.set_ylim(0, sys_top * 1.06)
    ax_left.set_yticks(np.linspace(0, 1.0, 5))
    ax_right.set_yticks(np.linspace(0, sys_top, 5))
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, rotation=20, ha="right")
    ax_left.grid(axis="y", zorder=0)
    ax_left.set_title(rf"$w_A:w_B = {w_a:g}:{w_b:g}$  (phase 2)", pad=10)

    for xi, v in zip(x, fairness):
        if v > 0:
            ax_left.annotate(f"{v:.2f}", xy=(xi - bar_w / 2, v),
                             xytext=(0, 5), textcoords="offset points",
                             ha="center", va="bottom", rotation=90,
                             color=FAIR_COLOR)
    for xi, v in zip(x, sys_rps):
        if v > 0:
            ax_right.annotate(f"{v:.1f}", xy=(xi + bar_w / 2, v),
                              xytext=(0, 5), textcoords="offset points",
                              ha="center", va="bottom", rotation=90,
                              color=TPUT_COLOR)

    fig.tight_layout(pad=0.2)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_throughput_timeseries(method_dirs: Dict[str, Path],
                               victim_task: str, aggressor_task: str,
                               meta: dict, out_path: Path,
                               w_a: float = 1.0, w_b: float = 1.0,
                               bin_s: float = 2.0) -> None:
    methods = [m for m in METHOD_ORDER if m in method_dirs]
    if not methods:
        return
    bounds = meta.get("phase_boundaries_s", [])
    t_max = float(bounds[-1]) if bounds else 30.0

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(5.0, 4.0), sharex=True)
    panels = [(ax_a, victim_task,    f"Client A (w={w_a:g})"),
              (ax_b, aggressor_task, f"Client B (w={w_b:g})")]

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
                    marker=MARKERS[m], markevery=max(1, len(centers) // 8),
                    markersize=plt.rcParams["lines.markersize"],
                    linewidth=plt.rcParams["lines.linewidth"],
                    label=LABELS[m], zorder=3)
            if rps.size:
                panel_max = max(panel_max, float(rps.max()))
        _add_phase_lines(ax, meta, t_max)
        ax.set_ylabel("Throughput (RPS)")
        ax.text(0.98, 0.92, panel_label, transform=ax.transAxes,
                fontweight="bold", va="top", ha="right",
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="none", pad=1.5))
        ax.grid(axis="y")

    y_nice = _nice_ceil(panel_max * 1.05) if panel_max > 0 else 1.0
    x_nice = _nice_ceil(t_max)
    for ax, _, _ in panels:
        ax.set_xlim(0, x_nice)
        ax.set_ylim(0, y_nice)
    ax_b.set_xlabel("Time (s)")

    h, l = ax_a.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.3, h_pad=1.2)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=len(h), frameon=False, handlelength=2.0,
               columnspacing=1.0)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_latency_timeseries(method_dirs: Dict[str, Path],
                            victim_task: str, aggressor_task: str,
                            meta: dict, out_path: Path,
                            w_a: float = 1.0, w_b: float = 1.0,
                            bin_s: float = 2.0) -> None:
    methods = [m for m in METHOD_ORDER if m in method_dirs]
    if not methods:
        return
    bounds = meta.get("phase_boundaries_s", [])
    t_max = float(bounds[-1]) if bounds else 30.0

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

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(5.0, 4.0), sharex=True)
    panels = [(ax_a, victim_task,    f"Client A (w={w_a:g})"),
              (ax_b, aggressor_task, f"Client B (w={w_b:g})")]

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
                    marker=MARKERS[m], markevery=max(1, len(centers) // 8),
                    markersize=plt.rcParams["lines.markersize"],
                    linewidth=plt.rcParams["lines.linewidth"],
                    label=LABELS[m], zorder=3)
            valid = mean_lat[~np.isnan(mean_lat)]
            if valid.size:
                panel_max = max(panel_max, float(valid.max()))
        _add_phase_lines(ax, meta, t_max)
        ax.set_ylabel(f"Latency ({unit})")
        ax.text(0.02, 0.95, panel_label, transform=ax.transAxes,
                fontweight="bold", va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="none", pad=1.5))
        ax.grid(axis="y")

    y_nice = _nice_ceil(panel_max * 1.1) if panel_max > 0 else 1.0
    x_nice = _nice_ceil(t_max)
    for ax, _, _ in panels:
        ax.set_xlim(0, x_nice)
        ax.set_ylim(0, y_nice)
    ax_b.set_xlabel("Time (s)")

    h, l = ax_a.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.3, h_pad=1.2)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=len(h), frameon=False, handlelength=2.0,
               columnspacing=1.0)
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# (BFQ dir, w_A, w_B, filename tag)
WEIGHT_SCENARIOS = [
    ("bfq_1_1", 1.0, 1.0, "1to1"),
    ("bfq_2_1", 2.0, 1.0, "2to1"),
    ("bfq_3_1", 3.0, 1.0, "3to1"),
]
BASELINE_DIR = "vllm_baseline"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-base", default="experiments/fair_share/llm/results_t4")
    ap.add_argument("--plot-dir",     default=None)
    ap.add_argument("--victim-task",    default="qwenA")
    ap.add_argument("--aggressor-task", default="qwenB")
    ap.add_argument("--bin-size-s",     type=float, default=2.0)
    args = ap.parse_args()

    apply_paper_style()

    base = (SERVING_DIR / args.results_base).resolve()
    if not base.exists():
        print(f"[Error] results dir not found: {base}")
        return 1
    plot_dir = (SERVING_DIR / args.plot_dir).resolve() if args.plot_dir \
               else base / "plots"

    # Use any available meta.json for phase boundaries (all runs share trace).
    meta: dict = {}
    for d in base.iterdir():
        if d.is_dir() and (d / "meta.json").exists():
            meta = _read_meta(d)
            if meta:
                break
    if not meta:
        print(f"[Error] no meta.json found under {base}")
        return 1

    vllm_dir = base / BASELINE_DIR

    for bfq_name, w_a, w_b, tag in WEIGHT_SCENARIOS:
        bfq_dir = base / bfq_name
        if not bfq_dir.exists():
            print(f"[Skip] {bfq_name}: dir not found")
            continue

        method_dirs: Dict[str, Path] = {"bfq": bfq_dir}
        if vllm_dir.exists():
            method_dirs["vllm"] = vllm_dir
        else:
            print(f"[Skip] {BASELINE_DIR}: dir not found — baseline omitted for {tag}")

        plot_fairness_summary(method_dirs, args.victim_task, args.aggressor_task,
                              w_a, w_b, meta, plot_dir / f"fairness_{tag}.png")
        plot_throughput_timeseries(method_dirs, args.victim_task, args.aggressor_task,
                                   meta, plot_dir / f"throughput_{tag}.png",
                                   w_a=w_a, w_b=w_b, bin_s=args.bin_size_s)
        plot_latency_timeseries(method_dirs, args.victim_task, args.aggressor_task,
                                meta, plot_dir / f"latency_{tag}.png",
                                w_a=w_a, w_b=w_b, bin_s=args.bin_size_s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
