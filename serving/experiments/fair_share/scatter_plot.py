#!/usr/bin/env python3
"""fair_share — common scatter plot across models.

For the 3:1 weight scenario (w_A=3, w_B=1) we plot a single scatter chart
combining results from three model backbones:

    - tsfm/results_t4         (momentlarge)
    - vision/results_t4_dinolarge   (dinolarge)
    - vision/results_t4_swintiny    (swintiny)

Each point is one (method, model) pair measured in phase 2:
    x = system throughput  (T_A + T_B,  req/s)
    y = fairness           (hybrid satisfaction + weighted max-min)

Methods shown: S-BE (fcfs), S-STFQ (stfq_1_3), FMVISOR (bfq_3_1).
Color encodes the method; marker shape encodes the model.

Run from serving/:
    python experiments/fair_share/scatter_plot.py
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

SERVING_DIR = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "fcfs": "#FF8C00",   # DarkOrange — S-BE
    "stfq": "#2CA02C",   # ForestGreen — S-STFQ
    "bfq":  "#D62728",   # Crimson — FMVisor
}
METHOD_LABELS = {
    "fcfs": "S-BE",
    "stfq": "S-STFQ",
    "bfq":  "FMVisor",
}
METHOD_ORDER = ["fcfs", "stfq", "bfq"]

MODEL_MARKERS = {
    "momentlarge": "o",   # circle
    "dinolarge":   "s",   # square
    "swintiny":    "^",   # triangle up
}
MODEL_LABELS = {
    "momentlarge": "MOMENT-large",
    "dinolarge":   "DINO-large",
    "swintiny":    "Swin-tiny",
}
MODEL_ORDER = ["momentlarge", "dinolarge", "swintiny"]


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     "black",
        "axes.labelcolor":    "black",
        "axes.linewidth":     0.5,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.color":         "#e5e5e5",
        "grid.linestyle":     "-",
        "grid.linewidth":     0.3,
        "grid.alpha":         1.0,
        "xtick.color":        "black",
        "ytick.color":        "black",
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.width":  0.5,
        "ytick.major.width":  0.5,
        "xtick.major.size":   3.0,
        "ytick.major.size":   3.0,
        "text.color":         "black",
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          8,
        "axes.titlesize":     8,
        "axes.labelsize":     8,
        "xtick.labelsize":    7.5,
        "ytick.labelsize":    7.5,
        "legend.fontsize":    7,
        "legend.frameon":     False,
        "lines.linewidth":    1.0,
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
    print(f"[Plot] saved {out_path}  and  {out_path.with_suffix('.pdf')}")


# ---------------------------------------------------------------------------
# Data loading (mirrors tsfm/plot.py and vision/plot.py)
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


def _completions_in_window(recs: List[Tuple[float, float]],
                           t_start: float, t_end: float) -> int:
    n = 0
    for send_t, lat_ms in recs:
        done = send_t + lat_ms / 1000.0
        if t_start <= done < t_end:
            n += 1
    return n


def _phase2_window(meta: dict) -> Tuple[float, float]:
    bounds = meta.get("phase_boundaries_s", [])
    if len(bounds) >= 2:
        return float(bounds[0]), float(bounds[1])
    return 0.0, 0.0


def _offered_rate_from_trace(trace_path: Path, task: str,
                             t_start: float, t_end: float) -> float:
    if not trace_path.exists() or t_end <= t_start:
        return 0.0
    raw = json.loads(trace_path.read_text())
    times = raw.get(task, [])
    n = sum(1 for t in times if t_start <= float(t) < t_end)
    return n / (t_end - t_start)


SATISFIED_TOL = 0.95


def _weighted_maxmin_ideal(d_a: float, d_b: float,
                           w_a: float, w_b: float,
                           capacity: float) -> Tuple[float, float]:
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
) -> float:
    """Hybrid fairness over [t_start, t_end). See tsfm/plot.py for details."""
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


# ---------------------------------------------------------------------------
# Per-model configuration
# ---------------------------------------------------------------------------

# For each model: results-base, method-subdir map, victim/aggressor task names.
# All entries are for the 3:1 weight scenario (w_A = 3, w_B = 1).
MODEL_CONFIGS: Dict[str, Dict] = {
    "momentlarge": {
        "results_base": "experiments/fair_share/tsfm/results_t4",
        "victim":       "ecgclass",
        "aggressor":    "gestureclass",
        "method_subdir": {
            "fcfs": "fcfs",
            "stfq": "stfq_1_3",
            "bfq":  "bfq_3_1",
        },
    },
    "dinolarge": {
        "results_base": "experiments/fair_share/vision/results_t4_dinolarge",
        "victim":       "nyudepth",
        "aggressor":    "vocseg",
        "method_subdir": {
            "fcfs": "fcfs",
            "stfq": "stfq_1_3",
            "bfq":  "bfq_3_1",
        },
    },
    "swintiny": {
        "results_base": "experiments/fair_share/vision/results_t4_swintiny",
        "victim":       "nyudepth",
        "aggressor":    "vocseg",
        "method_subdir": {
            "fcfs": "fcfs",
            "stfq": "stfq_1_3",
            "bfq":  "bfq_3_1",
        },
    },
}

W_A, W_B = 3.0, 1.0


# ---------------------------------------------------------------------------
# Compute & plot
# ---------------------------------------------------------------------------

def _nice_ceil(value: float) -> float:
    if value <= 0 or not np.isfinite(value):
        return 1.0
    magnitude = 10.0 ** np.floor(np.log10(value))
    fraction = value / magnitude
    for cap in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if fraction <= cap + 1e-9:
            return cap * magnitude
    return 10.0 * magnitude


def _collect_point(base: Path, method_dir: Path,
                   victim: str, aggressor: str,
                   w_a: float, w_b: float) -> Tuple[float, float]:
    """Return (throughput, fairness) for a single (method, model) cell."""
    meta = _read_meta(method_dir)
    p_start, p_end = _phase2_window(meta)
    if p_end <= p_start:
        return float("nan"), float("nan")
    p_dur = p_end - p_start

    trace_path = base / "trace.json"
    offered_a = _offered_rate_from_trace(trace_path, victim,    p_start, p_end)
    offered_b = _offered_rate_from_trace(trace_path, aggressor, p_start, p_end)

    a_recs = _load_records(method_dir, victim)
    b_recs = _load_records(method_dir, aggressor)

    fair = minmax_fairness(a_recs, b_recs,
                           offered_a, offered_b,
                           w_a, w_b,
                           p_start, p_end)
    T_A = _completions_in_window(a_recs, p_start, p_end) / p_dur
    T_B = _completions_in_window(b_recs, p_start, p_end) / p_dur
    return T_A + T_B, fair


def gather_points() -> List[Dict]:
    """Return list of {model, method, throughput, fairness} entries."""
    rows: List[Dict] = []
    for model in MODEL_ORDER:
        cfg = MODEL_CONFIGS[model]
        base = (SERVING_DIR / cfg["results_base"]).resolve()
        if not base.exists():
            print(f"[Skip] {model}: results dir not found ({base})")
            continue
        for method in METHOD_ORDER:
            sub = cfg["method_subdir"].get(method)
            if not sub:
                continue
            mdir = base / sub
            if not mdir.exists():
                print(f"[Skip] {model}/{method}: {mdir} not found")
                continue
            tput, fair = _collect_point(
                base, mdir,
                cfg["victim"], cfg["aggressor"],
                W_A, W_B,
            )
            if not (np.isfinite(tput) and np.isfinite(fair)):
                print(f"[Skip] {model}/{method}: missing metrics")
                continue
            rows.append({
                "model":      model,
                "method":     method,
                "throughput": tput,
                "fairness":   fair,
            })
            print(f"[Point] {model:11s} {METHOD_LABELS[method]:7s} "
                  f"tput={tput:7.2f} req/s  fairness={fair:.3f}")
    return rows


def plot_scatter(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        print("[Error] no points to plot")
        return

    fig, ax = plt.subplots(figsize=(3.3, 2.6))

    tputs = [r["throughput"] for r in rows]
    x_max = _nice_ceil(max(tputs) * 1.05) if tputs else 1.0

    for r in rows:
        ax.scatter(
            r["throughput"], r["fairness"],
            s=72,
            color=METHOD_COLORS[r["method"]],
            marker=MODEL_MARKERS[r["model"]],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

    ax.set_xlabel("System throughput (req/s)")
    ax.set_ylabel("Fairness")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, linewidth=0.3, zorder=0)

    # Two legends: one for method (color), one for model (marker).
    method_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=METHOD_COLORS[m],
                   markeredgecolor="black", markeredgewidth=0.6,
                   markersize=7, label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ]
    model_handles = [
        plt.Line2D([0], [0], marker=MODEL_MARKERS[mdl], linestyle="",
                   markerfacecolor="white",
                   markeredgecolor="black", markeredgewidth=0.8,
                   markersize=7, label=MODEL_LABELS[mdl])
        for mdl in MODEL_ORDER
    ]

    leg1 = ax.legend(
        handles=method_handles,
        title="Method",
        loc="lower left",
        bbox_to_anchor=(0.005, 0.005),
        handlelength=1.0, handletextpad=0.5,
        labelspacing=0.25, borderpad=0.3,
        title_fontsize=7,
    )
    leg1.get_title().set_fontweight("bold")
    ax.add_artist(leg1)
    leg2 = ax.legend(
        handles=model_handles,
        title="Model",
        loc="lower right",
        bbox_to_anchor=(0.995, 0.005),
        handlelength=1.0, handletextpad=0.5,
        labelspacing=0.25, borderpad=0.3,
        title_fontsize=7,
    )
    leg2.get_title().set_fontweight("bold")

    fig.tight_layout(pad=0.2)
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="experiments/fair_share/plots/fairness_vs_throughput_3to1.png",
        help="Output PNG path (PDF written alongside).",
    )
    args = ap.parse_args()

    apply_paper_style()
    rows = gather_points()
    out_path = (SERVING_DIR / args.out).resolve()
    plot_scatter(rows, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
