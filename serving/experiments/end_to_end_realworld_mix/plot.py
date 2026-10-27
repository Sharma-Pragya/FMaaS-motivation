#!/usr/bin/env python3
"""Plots for the end_to_end_realworld_mix experiment.

For each mix_label under results/, compares conditions (e.g. fmaas vs no_sharing)
on:
  - summary_response_time.pdf : mean / p95 / p99 end-to-end latency bars
  - timeseries_response_time.pdf : mean & p99 latency over time (binned)
  - timeseries_offered_load.pdf  : offered requests/sec from the trace
  - timeseries_throughput.pdf    : completed requests/sec
  - summary_batch_size.pdf       : mean / p95 / max observed batch size bars
  - timeseries_batch_size.pdf    : mean batch size over time (binned)
  - placement_tasks_per_device.pdf  : tasks placed on each device, per condition
  - placement_backbones_per_host.pdf: backbone composition per physical host (GPU)
  - per_task_mean_latency.pdf       : mean latency per task, both conditions

Usage (from serving/):
    python -m experiments.end_to_end_realworld_mix.plot \
        [--results-dir experiments/end_to_end_realworld_mix/results] \
        [--mix-label mix_L40_M40_H20] \
        [--bin-sec 5]
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
import matplotlib.ticker as ticker
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = EXP_DIR / "results"

# Color palette mirrors serving/experiments/sharing_benefit/*/plot.py so the
# end-to-end and sharing-benefit figures form a visually consistent set.
CONDITION_COLORS = {
    "fmaas":      "#E06C75",   # pink-red — FMVisor (proposed)
    "no_sharing": "#888888",   # mid gray — BE (baseline)
}
CONDITION_LABELS = {
    "fmaas":      "FMVisor",
    "no_sharing": "BE",
}

# Baseline first, proposed second — standard paper convention.
PAPER_METHODS = ["no_sharing", "fmaas"]

# Markers and linestyles match sharing_benefit/tpc/plot.py (BE ≈ no_sharing,
# FMVisor ≈ sharing).
CONDITION_MARKER = {
    "no_sharing": "D",
    "fmaas":      "o",
}
CONDITION_LINESTYLE = {
    "no_sharing": "-.",
    "fmaas":      "-",
}
PAPER_TS_MARKER_EVERY_SEC = 25.0  # markevery ≈ one marker per this many seconds
PAPER_TS_MARKERSIZE = 4.0
PAPER_TS_MARKEREDGE = 0.35


def _color(cond: str) -> str:
    return CONDITION_COLORS.get(cond, None) or "#555555"


def _label(cond: str) -> str:
    return CONDITION_LABELS.get(cond, cond)


def _load_condition(cdir: Path) -> Dict:
    csv_path = cdir / "request_latency_results.csv"
    if not csv_path.is_file():
        return {}
    req_time: List[float] = []
    e2e_ms: List[float] = []
    batch_keys: List[Tuple[str, float]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rt = float(row["req_time"])
                lat = float(row["end_to_end_latency(ms)"])
                dev = row.get("device", "")
                ds = float(row.get("device_start_time", "nan"))
            except (KeyError, ValueError):
                continue
            req_time.append(rt)
            e2e_ms.append(lat)
            batch_keys.append((dev, ds))
    return {
        "req_time":   np.asarray(req_time, dtype=float),
        "e2e_ms":     np.asarray(e2e_ms,   dtype=float),
        "batch_keys": batch_keys,
    }


def _batch_sizes(d: Dict, warmup_sec: float = 0.0):
    """Returns (batch_sizes, batch_start_times) per observed batch.
    Batches are identified by (device, device_start_time) — all requests in the
    same batch share the same device_start_time."""
    grouped: Dict[Tuple[str, float], List[float]] = {}
    for (dev, ds), rt in zip(d["batch_keys"], d["req_time"]):
        if rt < warmup_sec:
            continue
        if not dev or np.isnan(ds):
            continue
        grouped.setdefault((dev, ds), []).append(rt)
    sizes = np.array([len(v) for v in grouped.values()], dtype=float)
    # Use the earliest req_time in each batch as its time anchor.
    times = np.array([min(v) for v in grouped.values()], dtype=float)
    return sizes, times


def _summary_bars(cond_data: Dict[str, Dict], out_path: Path, title: str) -> None:
    conds = list(cond_data.keys())
    metrics = ["mean", "p95", "p99"]
    values = {m: [] for m in metrics}
    for c in conds:
        lat = cond_data[c]["e2e_ms"]
        if len(lat) == 0:
            for m in metrics:
                values[m].append(0.0)
            continue
        values["mean"].append(float(np.mean(lat)))
        values["p95"].append(float(np.percentile(lat, 95)))
        values["p99"].append(float(np.percentile(lat, 99)))

    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(conds))
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, c in enumerate(conds):
        bar_vals = [values[m][i] for m in metrics]
        offset = (i - (len(conds) - 1) / 2) * width
        bars = ax.bar(x + offset, bar_vals, width, label=_label(c), color=_color(c))
        for b, v in zip(bars, bar_vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("End-to-end latency (ms)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _binned(times: np.ndarray, values: np.ndarray, t_max: float, bin_sec: float,
            reducer) -> Tuple[np.ndarray, np.ndarray]:
    if t_max <= 0:
        return np.array([]), np.array([])
    n_bins = int(np.ceil(t_max / bin_sec))
    edges = np.arange(n_bins + 1) * bin_sec
    centers = edges[:-1] + bin_sec / 2
    out = np.full(n_bins, np.nan)
    if len(times) == 0:
        return centers, out
    idx = np.clip((times // bin_sec).astype(int), 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            out[b] = reducer(values[mask])
    return centers, out


def _binned_counts(times: np.ndarray, t_max: float, bin_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    if t_max <= 0:
        return np.array([]), np.array([])
    n_bins = int(np.ceil(t_max / bin_sec))
    edges = np.arange(n_bins + 1) * bin_sec
    centers = edges[:-1] + bin_sec / 2
    counts, _ = np.histogram(times, bins=edges)
    return centers, counts.astype(float) / bin_sec


def _timeseries_response_time(cond_data: Dict[str, Dict], out_path: Path,
                              bin_sec: float, t_max: float, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for c, d in cond_data.items():
        col = _color(c)
        x, mean_v = _binned(d["req_time"], d["e2e_ms"], t_max, bin_sec, np.mean)
        _,  p99_v = _binned(d["req_time"], d["e2e_ms"], t_max, bin_sec,
                            lambda a: np.percentile(a, 99))
        axes[0].plot(x, mean_v, label=_label(c), color=col, lw=1.0)
        axes[1].plot(x, p99_v,  label=_label(c), color=col, lw=1.0)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[1].set_ylabel("P99 Latency (ms)")
    axes[1].set_xlabel("Time since start (s)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _timeseries_offered_load(cond_data: Dict[str, Dict], out_path: Path,
                             bin_sec: float, t_max: float, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for c, d in cond_data.items():
        x, rate = _binned_counts(d["req_time"], t_max, bin_sec)
        ax.plot(x, rate, label=_label(c), color=_color(c), lw=1.0)
    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel("Offered load (req/s)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _timeseries_throughput(cond_data: Dict[str, Dict], out_path: Path,
                           bin_sec: float, t_max: float, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for c, d in cond_data.items():
        completion_time = d["req_time"] + d["e2e_ms"] / 1000.0
        x, rate = _binned_counts(completion_time, t_max, bin_sec)
        ax.plot(x, rate, label=_label(c), color=_color(c), lw=1.0)
    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel("Throughput (completions/s)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _summary_batch_size(cond_data: Dict[str, Dict], out_path: Path, title: str) -> None:
    conds = list(cond_data.keys())
    metrics = ["mean", "p95", "max"]
    values = {m: [] for m in metrics}
    for c in conds:
        sizes, _ = _batch_sizes(cond_data[c])
        if len(sizes) == 0:
            for m in metrics:
                values[m].append(0.0)
            continue
        values["mean"].append(float(np.mean(sizes)))
        values["p95"].append(float(np.percentile(sizes, 95)))
        values["max"].append(float(np.max(sizes)))

    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(conds))
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, c in enumerate(conds):
        bar_vals = [values[m][i] for m in metrics]
        offset = (i - (len(conds) - 1) / 2) * width
        bars = ax.bar(x + offset, bar_vals, width, label=_label(c), color=_color(c))
        for b, v in zip(bars, bar_vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("Batch size (requests/batch)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _timeseries_batch_size(cond_data: Dict[str, Dict], out_path: Path,
                           bin_sec: float, t_max: float, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for c, d in cond_data.items():
        sizes, times = _batch_sizes(d)
        if len(sizes) == 0:
            continue
        x, mean_v = _binned(times, sizes, t_max, bin_sec, np.mean)
        ax.plot(x, mean_v, label=_label(c), color=_color(c), lw=1.0)
    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel("Mean batch size")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _load_slots(deployments_root: Path, mix_label: str, cond: str):
    """Return list of (device_url, backbone, [tasks]) for the given condition.

    Handles both slot schemas:
      - fmaas: each entry is a deployment with a `tasks` list.
      - no_sharing: each entry is one (task, device) pairing.
    """
    p = deployments_root / mix_label / f"{cond}_slots.json"
    if not p.is_file():
        return []
    raw = json.loads(p.read_text())
    out = []
    if not raw:
        return out
    if isinstance(raw[0].get("tasks"), list):
        for s in raw:
            tasks = [t.get("task") for t in s.get("tasks", []) if t.get("task")]
            out.append((s["device_url"], s.get("backbone", "?"), tasks))
    else:
        bucket: Dict[Tuple[str, str], List[str]] = {}
        for s in raw:
            key = (s["device_url"], s.get("backbone", "?"))
            bucket.setdefault(key, []).append(s["task"])
        for (dev, bb), ts in bucket.items():
            out.append((dev, bb, ts))
    return out


def _plot_tasks_per_device(slots_per_cond: Dict[str, List], out_path: Path,
                           title: str) -> None:
    """Grouped bar over physical hosts (one GPU per host).

    x-axis: physical hosts (IPs), shared across conditions.
    y-axis: total number of tasks placed on the host across all its
    deployments on that GPU.
    """
    conds = list(slots_per_cond.keys())
    hosts = sorted({dev.split(":")[0]
                    for slots in slots_per_cond.values()
                    for dev, _, _ in slots})

    counts_per_cond: Dict[str, List[int]] = {}
    for c in conds:
        by_host: Dict[str, int] = {h: 0 for h in hosts}
        for dev, _, ts in slots_per_cond[c]:
            by_host[dev.split(":")[0]] += len(ts)
        counts_per_cond[c] = [by_host[h] for h in hosts]

    x = np.arange(len(hosts))
    width = 0.8 / max(1, len(conds))
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(hosts) + 2), 4.5))
    for i, c in enumerate(conds):
        offset = (i - (len(conds) - 1) / 2) * width
        bars = ax.bar(x + offset, counts_per_cond[c], width,
                      label=_label(c), color=_color(c))
        for b, v in zip(bars, counts_per_cond[c]):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v, str(v),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(hosts, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel(f"Physical host / GPU ({len(hosts)} devices)")
    ax.set_ylabel("# tasks placed on host")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_backbones_per_host(slots_per_cond: Dict[str, List], out_path: Path,
                             title: str) -> None:
    """Stacked bar: per physical host (IP), count of deployments by backbone."""
    conds = list(slots_per_cond.keys())
    all_backbones = sorted({bb for slots in slots_per_cond.values()
                            for _, bb, _ in slots})
    cmap = plt.get_cmap("tab20")
    bb_color = {bb: cmap(i % cmap.N) for i, bb in enumerate(all_backbones)}

    fig, axes = plt.subplots(len(conds), 1,
                             figsize=(11, 3.5 * len(conds)),
                             sharex=False)
    if len(conds) == 1:
        axes = [axes]
    for ax, c in zip(axes, conds):
        slots = slots_per_cond[c]
        by_host: Dict[str, Dict[str, int]] = {}
        for dev, bb, _ in slots:
            host = dev.split(":")[0]
            by_host.setdefault(host, {}).setdefault(bb, 0)
            by_host[host][bb] += 1
        hosts = sorted(by_host.keys(),
                       key=lambda h: -sum(by_host[h].values()))
        x = np.arange(len(hosts))
        bottoms = np.zeros(len(hosts))
        for bb in all_backbones:
            heights = np.array([by_host[h].get(bb, 0) for h in hosts],
                               dtype=float)
            if heights.sum() == 0:
                continue
            ax.bar(x, heights, bottom=bottoms, color=bb_color[bb],
                   edgecolor="white", linewidth=0.4, label=bb)
            bottoms += heights
        ax.set_xticks(x)
        ax.set_xticklabels(hosts, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("# deployments on host (GPU)")
        ax.set_title(f"{_label(c)} — {len(hosts)} hosts, "
                     f"{len(slots)} deployments")
        ax.grid(axis="y", alpha=0.3)
    # Single shared legend on the last axis
    axes[-1].legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _per_task_mean_latency(cond_data: Dict[str, Dict], mix_dir: Path) -> Dict[str, Dict[str, float]]:
    """Returns {cond: {task: mean_latency_ms}}. Re-reads CSV for task column."""
    out: Dict[str, Dict[str, float]] = {}
    for c in cond_data:
        agg: Dict[str, List[float]] = {}
        with (mix_dir / c / "request_latency_results.csv").open() as f:
            for row in csv.DictReader(f):
                try:
                    lat = float(row["end_to_end_latency(ms)"])
                except (KeyError, ValueError):
                    continue
                agg.setdefault(row["task"], []).append(lat)
        out[c] = {t: float(np.mean(v)) for t, v in agg.items() if v}
    return out


def _plot_per_task_mean_latency(per_task: Dict[str, Dict[str, float]],
                                out_path: Path, title: str) -> None:
    conds = list(per_task.keys())
    tasks = sorted(set().union(*(d.keys() for d in per_task.values())))
    # sort by max-across-conds latency descending so worst tasks are visible
    tasks.sort(key=lambda t: -max(per_task[c].get(t, 0.0) for c in conds))

    x = np.arange(len(tasks))
    width = 0.8 / max(1, len(conds))
    height = max(6, 0.12 * len(tasks))
    fig, ax = plt.subplots(figsize=(11, height))
    for i, c in enumerate(conds):
        vals = [per_task[c].get(t, 0.0) for t in tasks]
        offset = (i - (len(conds) - 1) / 2) * width
        ax.barh(x + offset, vals, width, label=_label(c), color=_color(c))
    ax.set_yticks(x)
    ax.set_yticklabels(tasks, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Mean end-to-end latency (ms)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Paper-ready plots
# ---------------------------------------------------------------------------

# Paper time-series window: trace time in [PAPER_TS_START, PAPER_TS_END) (s).
# X-axis is re-zeroed to 0 … (end − start) so e.g. 100–400 s reads as 0–300 s.
PAPER_TS_START = 100.0
PAPER_TS_END   = 400.0
PAPER_TS_BIN_S = 1.0


# Printed figure geometry. ACM sigconf, measured from the compiled paper --
# the same constants as experiments/paper_style.py. A figure is generated at
# exactly the width it occupies on the page and included with no width= key,
# so the point sizes below are the point sizes that appear in the PDF.
COLUMN_WIDTH_IN = 241.14 / 72.27   # 3.336 in, \columnwidth
TEXT_WIDTH_IN   = 506.61 / 72.27   # 7.010 in, \textwidth (figure*)

# Height of one panel's axes box. The \columnwidth panels in Figures 7/8 are
# 0.515in tall (3.33x1.30in canvas, tight_layout rect (0.055,0.28,0.985,0.885));
# matching that is what makes a \textwidth figure sit at the same visual weight
# on the page. Before this was set, the 2x2 drew 1.17in panels -- 2.3x taller
# than every other figure in the paper.
PANEL_AXES_H_IN   = 0.55   # axes box per panel
_ROW_GAP_IN       = 0.11   # between the two rows (they share the x scale)
_BOTTOM_CHROME_IN = 0.34   # x tick labels + x-label, bottom row only
_LEGEND_STRIP_IN  = 0.17   # shared legend above the top row
COMBINED_FIG_H_IN = (2 * PANEL_AXES_H_IN + _ROW_GAP_IN
                     + _BOTTOM_CHROME_IN + _LEGEND_STRIP_IN + 0.02)

# The mean bar and the mean time series are typeset side by side as two
# subfigures filling one \columnwidth. They are generated as a matched pair:
# same height, widths summing to just under the column so \subfloat has slack.
# Height matches the house half-column single panel (1.63x1.10in, 0.702in of
# axes) rather than the 3-panel rows, which is the closer analogue.
# The bar takes the smaller share but cannot go much below 1.40in: its x tick
# labels are "BE" and "FMVisor", and "FMVisor" is ~0.40in wide at 7.5pt.
PAIR_H_IN     = 1.10
PAIR_BAR_W_IN = 1.40
PAIR_TS_W_IN  = 1.90   # 1.40 + 1.90 = 3.30in, 0.036in inside \columnwidth

# Equal canvas height is not enough to make the pair line up: tight_layout
# fits each panel's chrome independently, and the time series spends ~0.32in
# below its axes on tick labels plus "Time (s)" while the bar spends ~0.14in on
# tick labels alone. Fitted separately, the bar's plot box is the taller of the
# two and the pair reads as mismatched. Both therefore reserve the SAME bottom
# band -- the time series' requirement, the larger of the two -- so the two
# axes boxes are identical. The bar simply leaves the extra space empty.
# Measured from tight_layout at the sizes and fonts above: the time series
# needs 0.374in below its axes (tick labels + "Time (s)"), the bar only 0.209in
# (tick labels alone); both want 0.062in of overhang above for the top tick
# label. Taking the larger bottom gives both panels a 0.664in axes box. Re-run
# the measurement if PAIR_H_IN or the point sizes change.
_PAIR_BOTTOM_IN = 0.375
_PAIR_TOP_IN    = 0.062


def _pair_vertical(fig) -> None:
    """Pin top and bottom to the shared band, keeping tight_layout's left/right.

    Call after tight_layout. Only ever grows the reserved chrome, never shrinks
    it: if a panel genuinely needs more room than the shared band, it keeps its
    own and the pair is merely misaligned rather than clipped.
    """
    pos = fig.axes[0].get_position()
    fig.subplots_adjust(
        bottom=max(pos.y0, _PAIR_BOTTOM_IN / PAIR_H_IN),
        top=min(pos.y1, 1.0 - _PAIR_TOP_IN / PAIR_H_IN),
    )

# Printed point sizes, matched to the \columnwidth Evaluation panels
# (sharing_benefit/tpc/plot.py). Do not override these per-plot: a local bump
# only makes sense when the canvas is not the printed size, which is exactly
# the bug this block fixes.
BASE_FONT_PT  = 8.0
TICK_FONT_PT  = 7.5
VALUE_FONT_PT = 7.0   # in-axes bar value labels
ANNOT_FONT_PT = 7.5   # improvement arrow label


def _paper_style() -> None:
    """Publication-ready rcParams mirroring sharing_benefit/*/plot.py."""
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
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.width":  0.5,
        "ytick.major.width":  0.5,
        "xtick.major.size":   2.5,
        "ytick.major.size":   2.5,
        "text.color":         "black",
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          BASE_FONT_PT,
        "axes.titlesize":     BASE_FONT_PT,
        "axes.labelsize":     BASE_FONT_PT,
        "xtick.labelsize":    TICK_FONT_PT,
        "ytick.labelsize":    TICK_FONT_PT,
        "legend.fontsize":    TICK_FONT_PT,
        "legend.frameon":     False,
        "lines.linewidth":    1.2,
        "lines.markersize":   3,
        "hatch.linewidth":    0.85,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.facecolor":  "white",
    })


def _paper_save(fig, out_path: Path) -> None:
    """Save at exactly `fig`'s figsize.

    Deliberately no bbox_inches="tight": a tight bbox crops to the drawn
    content, so the saved width depends on how long the tick labels happen to
    be, and \includegraphics[width=...] then rescales the fonts by a factor
    that differs per figure. The layout call has already fitted the labels
    inside the canvas, so saving the canvas as-is keeps printed pt == rcParams pt.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"[Plot] Saved: {out_path} ({fig.get_size_inches()[0]:.2f}in exact)")
    plt.close(fig)


# Rotated at 8pt, "P99 Latency (ms)" is ~1.0in long and does not fit beside a
# 0.70in-tall panel. The subcaption already says which statistic the panel is.
SHORT_YLABEL = {"mean": "Mean (ms)", "p99": "P99 (ms)"}


def _bar_vals(cond_data: Dict[str, Dict], conds: List[str], metric: str
              ) -> Tuple[List[float], str]:
    if metric == "mean":
        vals = [float(np.mean(cond_data[c]["e2e_ms"])) for c in conds]
        ylabel = "Latency (ms)"
    elif metric == "p99":
        vals = [float(np.percentile(cond_data[c]["e2e_ms"], 99)) for c in conds]
        ylabel = "P99 Latency (ms)"
    else:
        raise ValueError(metric)
    return vals, ylabel


_NICE_STEPS = (1.0, 2.0, 2.5, 5.0, 10.0)


def _nice_top_and_step(value: float, target_ticks: int = 5,
                        min_ticks: int = 4, max_ticks: int = 9
                        ) -> Tuple[float, float]:
    """Return (top, step) such that:
      • `step` is an integer-friendly value in {1, 2, 2.5, 5, 10} × 10^k
      • `top = ceil(value / step) * step` (so the top tick lands exactly on
        the axis limit)
      • the tick count `top/step + 1` sits in [min_ticks, max_ticks]
      • among the valid candidates, the one with the *smallest* overshoot
        (top − value) is chosen, so we don't waste headroom.
    """
    if value <= 0 or not np.isfinite(value):
        return 1.0, 1.0

    magnitude = 10.0 ** np.floor(np.log10(value))
    # Search across the magnitude one decade smaller and one larger so we
    # always have a fine-grained step available even when `value` sits just
    # above a step boundary.
    candidates: List[Tuple[float, float, float, int]] = []  # (overshoot, step, top, n_intervals)
    for mag in (magnitude * 0.1, magnitude, magnitude * 10):
        for s in _NICE_STEPS:
            step = s * mag
            if step <= 0:
                continue
            n = int(np.ceil(value / step - 1e-9))
            if n < 1:
                continue
            top = n * step
            overshoot = (top - value) / value
            candidates.append((overshoot, step, top, n))

    if not candidates:
        return value, value / 4.0

    valid = [c for c in candidates if min_ticks - 1 <= c[3] <= max_ticks - 1]
    if not valid:
        # Relax tick-count bounds rather than emit a degenerate axis.
        valid = candidates
    valid.sort(key=lambda c: (c[0], abs(c[3] - (target_ticks - 1))))
    _, step, top, _ = valid[0]
    return top, step


def _set_y_endpoint(ax, value: float, target_ticks: int = 5,
                    max_ticks: int = 9) -> float:
    """Set y-axis to [0, top] with integer-friendly ticks so the last visible
    tick lands exactly on `top`. Returns the chosen `top`."""
    top, step = _nice_top_and_step(value, target_ticks=target_ticks,
                                    max_ticks=max_ticks)
    ax.set_ylim(0, top)
    n = int(round(top / step))
    ax.set_yticks([i * step for i in range(n + 1)])
    return top


def _draw_paper_bars(ax, cond_data: Dict[str, Dict], metric: str,
                     ylabel: str = None, target_ticks: int = 5,
                     max_ticks: int = 8, headroom: float = 1.35,
                     arrow: bool = True) -> None:
    """Render a 2-bar comparison (BE vs FMVisor) on `ax` with per-bar value
    labels and a single downward-arrow improvement annotation above the
    FMVisor bar."""
    conds = [c for c in PAPER_METHODS if c in cond_data]
    vals, default_ylabel = _bar_vals(cond_data, conds, metric)
    ylabel = default_ylabel if ylabel is None else ylabel

    x = np.arange(len(conds))
    bars = ax.bar(
        x, vals, width=0.6,
        color=[_color(c) for c in conds],
        edgecolor="black", linewidth=0.7, zorder=2,
    )
    for bar, c in zip(bars, conds):
        if c == "fmaas":
            bar.set_hatch("//")
    ax.set_xticks(x)
    ax.set_xticklabels([_label(c) for c in conds])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.18)

    if not vals:
        return
    vmax = max(vals)
    # Axis top is a clean round number. The headroom has to clear the value
    # label AND the improvement arrow stacked above it; at PANEL_AXES_H_IN the
    # old 1.08 put the arrow head inside the "28" and the % label on the tail.
    y_top = _set_y_endpoint(ax, vmax * headroom, target_ticks=target_ticks,
                            max_ticks=max_ticks)

    # Numerical value just above each bar. Offset in POINTS, not in data
    # units: the axis top comes from _nice_top_and_step, so a data-unit offset
    # is a different physical gap on every panel.
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:.0f}", xy=(xi, v), xytext=(0, 1.5),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=VALUE_FONT_PT)

    # Improvement (or regression) of FMVisor relative to BE. A short vertical
    # arrow points DOWN at the top of the FMVisor bar with the % label
    # rendered just above the arrow tail.
    if "no_sharing" in conds and "fmaas" in conds:
        i_fm = conds.index("fmaas")
        i_be = conds.index("no_sharing")
        be_v, fm_v = vals[i_be], vals[i_fm]
        if be_v > 0:
            delta_pct = (be_v - fm_v) / be_v * 100.0
            arrow_char = "↓" if delta_pct >= 0 else "↑"
            ann_color = _color("fmaas")

            # The arrow only survives where there is real vertical room between
            # the bar's value label and the % label. On a 1.10in panel the axis
            # top is whatever _nice_top_and_step lands on, so that gap is not
            # under our control -- at top=40 the arrow came out inverted. There
            # the down-glyph in the label carries the direction on its own.
            if arrow:
                y_tail = y_top * 0.85
                y_head = fm_v + y_top * 0.16
                ax.annotate(
                    "",
                    xy=(x[i_fm], y_head),
                    xytext=(x[i_fm], y_tail),
                    arrowprops=dict(arrowstyle="-|>", color=ann_color,
                                    lw=1.0, mutation_scale=8),
                )
            if arrow:
                # Pinned to the axis top, with the arrow spanning the gap.
                ax.text(
                    x[i_fm], y_top * 0.99,
                    f"{arrow_char} {abs(delta_pct):.0f}%",
                    ha="center", va="top",
                    fontsize=ANNOT_FONT_PT, fontweight="bold", color=ann_color,
                )
            else:
                # No arrow to connect it, so the label has to sit ON the bar it
                # describes: directly above the bar's own value label, a fixed
                # 10pt up. Parked at the axis top it floated in white space with
                # nothing tying it to the FMVisor bar.
                ax.annotate(
                    f"{arrow_char} {abs(delta_pct):.0f}%",
                    xy=(x[i_fm], fm_v), xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=ANNOT_FONT_PT, fontweight="bold", color=ann_color,
                )


def _paper_binned(times: np.ndarray, values: np.ndarray, start: float,
                  end: float, bin_sec: float, reducer):
    n_bins = int(round((end - start) / bin_sec))
    edges = start + np.arange(n_bins + 1) * bin_sec
    # Re-zero to 0 … (end − start) on the plot x-axis (data still from [start,end)).
    centers = edges[:-1] + bin_sec / 2.0 - start
    out = np.full(n_bins, np.nan)
    mask = (times >= start) & (times < end)
    if not mask.any():
        return centers, out
    idx = np.searchsorted(edges, times[mask], side="right") - 1
    idx = np.clip(idx, 0, n_bins - 1)
    vals = values[mask]
    for b in range(n_bins):
        cur = vals[idx == b]
        if cur.size:
            out[b] = float(reducer(cur))
    return centers, out


def _draw_paper_timeseries(ax, cond_data: Dict[str, Dict], metric: str,
                           start: float, end: float, bin_sec: float,
                           ylabel: str = None, headroom: float = 1.02,
                           target_ticks: int = 5, max_ticks: int = 8) -> None:
    """Render the binned response-time series on `ax`. Data are taken from
    trace time [start, end); the x-axis is re-zeroed to 0 … (end − start).
    Markers and linestyles follow sharing_benefit/tpc/plot.py (BE: diamond
    + dash-dot; FMVisor: circle + solid)."""
    if metric == "mean":
        reducer = np.mean
        default_ylabel = "Latency (ms)"
    elif metric == "p99":
        reducer = lambda a: np.percentile(a, 99)
        default_ylabel = "P99 Latency (ms)"
    else:
        raise ValueError(metric)

    markevery = max(1, int(round(PAPER_TS_MARKER_EVERY_SEC / max(bin_sec, 1e-9))))

    series_max = 0.0
    for c in PAPER_METHODS:
        if c not in cond_data:
            continue
        x, y = _paper_binned(
            cond_data[c]["req_time"], cond_data[c]["e2e_ms"],
            start, end, bin_sec, reducer,
        )
        col = _color(c)
        valid = np.isfinite(y)
        ax.plot(
            x, y,
            label=_label(c),
            color=col,
            linestyle=CONDITION_LINESTYLE.get(c, "-"),
            linewidth=1.2,
            zorder=3,
            clip_on=False,
        )
        if valid.any():
            series_max = max(series_max, float(np.nanmax(y[valid])))

    x_span = end - start
    ax.set_xlim(0, x_span)
    ax.set_xticks(np.linspace(0, x_span, 7 if x_span >= 200 else 6))

    # y-axis: clean upper bound with the top tick landing on the limit.
    # Use a tight 2% pad above the peak so the data fills the panel — the
    # nice-step rounding will round up to the next clean tick from there.
    if series_max > 0:
        _set_y_endpoint(ax, series_max * headroom,
                        target_ticks=target_ticks, max_ticks=max_ticks)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel if ylabel is not None else default_ylabel)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)


def _paper_bar(cond_data: Dict[str, Dict], metric: str, out_path: Path) -> None:
    """The bar half of the side-by-side \columnwidth pair."""
    fig, ax = plt.subplots(figsize=(PAIR_BAR_W_IN, PAIR_H_IN))
    # 1.22 headroom still clears the value label and the arrow stacked above
    # it, but lands on a top of 50 rather than 60 for the mean panel -- at 1.35
    # with at most 5 ticks the axis wasted 40% of its height on empty space.
    _draw_paper_bars(ax, cond_data, metric, ylabel=SHORT_YLABEL[metric],
                     target_ticks=5, max_ticks=6, headroom=1.30, arrow=False)
    fig.tight_layout(pad=0.2)
    _pair_vertical(fig)
    _paper_save(fig, out_path)


def _paper_timeseries(cond_data: Dict[str, Dict], metric: str, out_path: Path,
                      start: float = PAPER_TS_START,
                      end:   float = PAPER_TS_END,
                      bin_sec: float = PAPER_TS_BIN_S) -> None:
    """The time-series half of the side-by-side \columnwidth pair.

    Carries the only BE/FMVisor legend: the bar panel's x-axis already names
    both, and the two panels share the palette. No font overrides -- the canvas
    is the printed size, so the rcParams point sizes are the printed ones.
    """
    fig, ax = plt.subplots(figsize=(PAIR_TS_W_IN, PAIR_H_IN))
    # The default 1.02 headroom leaves the legend sitting on the peaks. 1.28
    # opens a clear band above the data for it without a separate legend strip,
    # which a 1.10in-tall panel cannot spare.
    _draw_paper_timeseries(ax, cond_data, metric, start, end, bin_sec,
                           ylabel=SHORT_YLABEL[metric], headroom=1.28,
                           target_ticks=5, max_ticks=6)
    ax.legend(loc="upper right", ncol=2, handlelength=1.2,
              labelspacing=0.2, columnspacing=0.8, handletextpad=0.4,
              borderaxespad=0.2)
    fig.tight_layout(pad=0.2)
    _pair_vertical(fig)
    _paper_save(fig, out_path)


def _paper_combined(cond_data: Dict[str, Dict], out_path: Path,
                    start: float = PAPER_TS_START,
                    end:   float = PAPER_TS_END,
                    bin_sec: float = PAPER_TS_BIN_S) -> None:
    """One paper-ready 2×2 figure: top row = mean (bar | ts),
    bottom row = p99 (bar | ts). Bars are narrower than the time series so
    the four panels read at the same visual weight.
    """
    if not any(c in cond_data for c in PAPER_METHODS):
        return

    # Generated at exactly \textwidth and included with no width= key, so the
    # rcParams point sizes are what appears on the page -- the same 8pt/7.5pt
    # as the \columnwidth panels in Figures 7/8. The previous version drew an
    # 8.0in canvas with 13pt fonts and saved it with a tight bbox (7.96in on
    # disk); scaled to \textwidth that printed ~11.4pt text, half again larger
    # than every neighbouring figure.
    fig, axes = plt.subplots(
        2, 2, figsize=(TEXT_WIDTH_IN, COMBINED_FIG_H_IN),
        gridspec_kw={"width_ratios": [1.0, 1.8]},
    )
    (ax_mean_bar, ax_mean_ts), (ax_p99_bar, ax_p99_ts) = axes

    # Short y-labels: rotated 90 deg, "P99 Latency (ms)" at 8pt is ~1.0in long
    # and does not fit beside a PANEL_AXES_H_IN-tall panel. The row already says
    # which statistic it is. Four y-ticks, as in the columnwidth panels -- the
    # default five to eight crowd into 0.55in.
    _draw_paper_bars(ax_mean_bar, cond_data, "mean", ylabel="Mean (ms)",
                     target_ticks=4, max_ticks=5)
    _draw_paper_bars(ax_p99_bar,  cond_data, "p99",  ylabel="P99 (ms)",
                     target_ticks=4, max_ticks=5)
    _draw_paper_timeseries(ax_mean_ts, cond_data, "mean", start, end, bin_sec,
                           ylabel="Mean (ms)")
    _draw_paper_timeseries(ax_p99_ts,  cond_data, "p99",  start, end, bin_sec,
                           ylabel="P99 (ms)")
    for ax in (ax_mean_ts, ax_p99_ts):
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

    # The two rows share the x scale, so only the bottom row carries tick
    # labels and an x-label -- one chrome strip for the figure instead of two.
    for ax in (ax_mean_bar, ax_mean_ts):
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)

    # One shared legend above the row, as in the \columnwidth panels. Two
    # in-axes legends would cover a third of a PANEL_AXES_H_IN-tall panel.
    handles, labels = ax_mean_ts.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, borderaxespad=0.0,
               bbox_to_anchor=(0.5, 1.005))

    # left leaves room for the 3-digit p99 ticks plus the y-label; right for the
    # last x tick label ("300"), which a tight bbox used to absorb for free.
    # bottom/top/hspace are derived from the target panel height above.
    bottom = _BOTTOM_CHROME_IN / COMBINED_FIG_H_IN
    top    = 1.0 - (_LEGEND_STRIP_IN + 0.02) / COMBINED_FIG_H_IN
    fig.subplots_adjust(left=0.075, right=0.985, top=top, bottom=bottom,
                        hspace=_ROW_GAP_IN / PANEL_AXES_H_IN, wspace=0.19)
    _paper_save(fig, out_path)


def _paper_plots(cond_data: Dict[str, Dict], out_dir: Path) -> None:
    _paper_style()
    # Filename reflects the (start, end) time window on the trace timeline.
    s, e = int(PAPER_TS_START), int(PAPER_TS_END)
    _paper_bar(cond_data, "mean", out_dir / "paper_mean_bar.pdf")
    _paper_bar(cond_data, "p99",  out_dir / "paper_p99_bar.pdf")
    _paper_timeseries(cond_data, "mean", out_dir / f"paper_mean_ts_{s}_{e}.pdf")
    _paper_timeseries(cond_data, "p99",  out_dir / f"paper_p99_ts_{s}_{e}.pdf")
    _paper_combined(cond_data, out_dir / "paper_combined.pdf")


def _plot_mix(mix_dir: Path, bin_sec: float) -> None:
    conds = sorted(p.name for p in mix_dir.iterdir()
                   if p.is_dir() and (p / "request_latency_results.csv").is_file())
    if not conds:
        print(f"[plot] {mix_dir.name}: no condition CSVs, skipping")
        return

    cond_data: Dict[str, Dict] = {}
    duration_hint = 0.0
    for c in conds:
        d = _load_condition(mix_dir / c)
        if not d or len(d["req_time"]) == 0:
            continue
        cond_data[c] = d
        cfg_path = mix_dir / c / "run_config.json"
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text())
                duration_hint = max(duration_hint, float(cfg.get("duration", 0)))
            except Exception:
                pass

    if not cond_data:
        print(f"[plot] {mix_dir.name}: no usable data")
        return

    t_max = max(float(d["req_time"].max()) for d in cond_data.values())
    t_max = max(t_max, duration_hint) + bin_sec

    out_dir = mix_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    label = mix_dir.name

    _summary_bars(cond_data, out_dir / "summary_response_time.pdf",
                  f"{label}: response-time summary")
    _timeseries_response_time(cond_data, out_dir / "timeseries_response_time.pdf",
                              bin_sec, t_max,
                              f"{label}: response time over time (bin={bin_sec:g}s)")
    _timeseries_offered_load(cond_data, out_dir / "timeseries_offered_load.pdf",
                             bin_sec, t_max,
                             f"{label}: offered load (bin={bin_sec:g}s)")
    _timeseries_throughput(cond_data, out_dir / "timeseries_throughput.pdf",
                           bin_sec, t_max,
                           f"{label}: achieved throughput (bin={bin_sec:g}s)")
    _summary_batch_size(cond_data, out_dir / "summary_batch_size.pdf",
                        f"{label}: observed batch size")
    _timeseries_batch_size(cond_data, out_dir / "timeseries_batch_size.pdf",
                           bin_sec, t_max,
                           f"{label}: mean batch size over time (bin={bin_sec:g}s)")

    deployments_root = mix_dir.parents[1] / "deployments"
    slots_per_cond = {c: _load_slots(deployments_root, label, c) for c in cond_data}
    slots_per_cond = {c: s for c, s in slots_per_cond.items() if s}
    if slots_per_cond:
        _plot_tasks_per_device(slots_per_cond,
                               out_dir / "placement_tasks_per_device.pdf",
                               f"{label}: tasks placed per device")
        _plot_backbones_per_host(slots_per_cond,
                                 out_dir / "placement_backbones_per_host.pdf",
                                 f"{label}: backbones placed per physical host")

    per_task = _per_task_mean_latency(cond_data, mix_dir)
    _plot_per_task_mean_latency(per_task,
                                out_dir / "per_task_mean_latency.pdf",
                                f"{label}: mean latency per task")
    _paper_plots(cond_data, out_dir)

    print(f"[plot] {label}: wrote plots to {out_dir}")
    for c, d in cond_data.items():
        lat = d["e2e_ms"]
        sizes, _ = _batch_sizes(d)
        bs_mean = float(np.mean(sizes)) if len(sizes) else 0.0
        bs_max = float(np.max(sizes)) if len(sizes) else 0.0
        print(f"  {c:>12s}: n={len(lat):6d}  "
              f"mean={np.mean(lat):7.1f} ms  "
              f"p95={np.percentile(lat, 95):7.1f} ms  "
              f"p99={np.percentile(lat, 99):7.1f} ms  "
              f"bs_mean={bs_mean:5.2f}  bs_max={bs_max:3.0f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--mix-label", default=None,
                        help="Only plot this mix label (default: all).")
    parser.add_argument("--bin-sec", type=float, default=5.0,
                        help="Bin width (s) for timeseries plots.")
    args = parser.parse_args()

    root = Path(args.results_dir)
    if not root.is_dir():
        print(f"[plot] results dir not found: {root}")
        return 1

    if args.mix_label:
        mix_dirs = [root / args.mix_label]
    else:
        mix_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    for mix_dir in mix_dirs:
        if not mix_dir.is_dir():
            print(f"[plot] missing: {mix_dir}")
            continue
        _plot_mix(mix_dir, args.bin_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
