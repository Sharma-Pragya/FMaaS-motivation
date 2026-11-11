#!/usr/bin/env python3
"""RTVSntask/vlm/plot.py — response time vs. number of tasks (vLLM VLM multi-LoRA).

The VLM sweep writes the same latencies.csv / task_results.csv / failure.json
schema as the LLM sweep, so this is RTVSntask/llm/plot.py with VLM defaults —
kept byte-for-byte identical in styling and layout so LLM, VLM, vision and TPC
figures sit side by side in the paper.

Paper-style figures, kept in lockstep with RTVSntask/vision/plot.py: same
rcParams, same SERIES_COLORS / SERIES_LABELS / SERIES_LINESTYLE / SERIES_MARKER,
same line geometry, same standalone-legend convention, same file naming
(<stem>_rps<R>.<ext> plus <stem>_legend.<ext>). Output is PDF by default, as in
vision/plot.py and tpc/plot.py — vector figures for the paper; pass
--format png for a quick look.

Reads the sweep laid down by run.sh:

    <results>/ntasks_<N>/rps_<R>/<condition>/task_results.csv     (ran)
    <results>/ntasks_<N>/rps_<R>/<condition>/failure.json         (could not be placed)

Produces, per RPS:

    ntasks_mean_latency_rps<R>.pdf    mean latency vs #Tasks
    ntasks_p99_latency_rps<R>.pdf     P99 latency vs #Tasks
    ntasks_throughput_rps<R>.pdf      achieved throughput vs #Tasks
    summary_rps<R>.csv                every point, ok / failed + reason

A condition that could not be placed is annotated "OOM" on the frontier in that
series' colour (the LLM story: N backbone copies stop fitting), exactly as in
RTVSntask/tpc/plot.py, and recorded in the summary CSV with its classified
reason.

Usage:
    python experiments/RTVSntask/vlm/plot.py \
        --results experiments/RTVSntask/vlm/results_vlm_l40s_qwen2vl7b_natural
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Styles — identical to RTVSntask/vision/plot.py
# ---------------------------------------------------------------------------

SERIES_ORDER = ["single", "no_sharing_tpc", "no_sharing", "sharing"]
DEFAULT_INCLUDED_SERIES = ["no_sharing", "no_sharing_tpc", "sharing"]
SERIES_COLORS = {
    "single":         "#A9C7B5",   # sage green
    "no_sharing_tpc": "#6B9AC4",   # muted blue
    "no_sharing":     "#888888",   # mid gray
    "sharing":        "#E06C75",   # pink-red
}
SERIES_LABELS = {
    "single":         "ST",
    "no_sharing":     "BE",
    "no_sharing_tpc": "SP",
    "sharing":        "FMVisor",
}
SERIES_LINESTYLE = {
    "single":         ":",
    "no_sharing_tpc": "--",
    "no_sharing":     "-.",
    "sharing":        "-",
}
SERIES_MARKER = {
    "single":         "s",
    "no_sharing_tpc": "^",
    "no_sharing":     "D",
    "sharing":        "o",
}

COND_LINE_MARKERSIZE = 2.6
COND_LINE_MARKEREDGEWIDTH = 0.25
COND_LINE_MARKER_EDGECOLOR = "black"
COND_LINE_LINEWIDTH = 0.9

# Canvas size for the #Tasks sweep panels. These panels sit side by side in
# one figure row in the paper, so every script must emit them at the same
# size or the row comes out ragged.
NTASKS_SWEEP_FIG_W = 1.33
NTASKS_SWEEP_FIG_H = 1.10

REASON_LABEL = {
    "kv_cache_insufficient":   "no KV cache room",
    "cuda_oom":                "CUDA OOM",
    "insufficient_gpu_memory": "not enough GPU mem",
    "adapter_shape_mismatch":  "adapter/backbone mismatch",
    "zero_requests":           "0 requests completed",
    "port_conflict":           "port conflict",
    "adapter_missing":         "adapter missing",
    "tpc_partition":           "too few TPCs",
    "no_log":                  "no device log",
    "unknown":                 "unknown",
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


def save_figure_exact(fig, out_path) -> None:
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


def _endpoint_formatter(decimals: int = 0) -> ticker.FuncFormatter:
    def _fmt(v: float, _: object) -> str:
        if np.isclose(v, round(v)):
            return f"{int(round(v))}"
        return f"{v:.{decimals}f}".rstrip("0").rstrip(".")
    return ticker.FuncFormatter(_fmt)


def _set_linear_axis_with_endpoint(ax: plt.Axes, axis: str, lower: float,
                                   upper: float, target_ticks: int = 5,
                                   decimals: int = 0) -> None:
    """Set linear axis limits and ticks so the final tick lands exactly on upper."""
    if upper <= lower:
        upper = lower + 1.0
    ticks = np.linspace(lower, upper, num=target_ticks)
    if axis == "x":
        ax.set_xlim(lower, upper)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(_endpoint_formatter(decimals))
    else:
        ax.set_ylim(lower, upper)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(_endpoint_formatter(decimals))


def _nice_upper(value: float) -> float:
    """Round up to a 1/2/2.5/5 x 10^k value so the 5 ticks land on round numbers."""
    if value <= 0:
        return 1.0
    exp = np.floor(np.log10(value))
    frac = value / (10 ** exp)
    for nice in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if frac <= nice + 1e-9:
            return nice * (10 ** exp)
    return 10.0 ** (exp + 1)


def _legend_handles(series_keys: List[str]) -> List:
    return [
        plt.Line2D([0], [0],
                   color=SERIES_COLORS[s],
                   linestyle=SERIES_LINESTYLE[s],
                   marker=SERIES_MARKER[s],
                   markersize=COND_LINE_MARKERSIZE,
                   markerfacecolor=SERIES_COLORS[s],
                   markeredgecolor=COND_LINE_MARKER_EDGECOLOR,
                   markeredgewidth=COND_LINE_MARKEREDGEWIDTH,
                   linewidth=COND_LINE_LINEWIDTH,
                   label=SERIES_LABELS[s])
        for s in series_keys
    ]


def plot_standalone_legend(handles: List, out_path: Path, ncol: int = 4) -> None:
    """Save a legend as a standalone figure."""
    if not handles:
        return
    fig = plt.figure(figsize=(1, 1))
    legend = fig.legend(handles=handles, loc="center", ncol=ncol, frameon=False,
                        handlelength=1.2, columnspacing=0.7, handletextpad=0.25,
                        labelspacing=0.2, fontsize=6)
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.set_size_inches(bbox.width, bbox.height)
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def canon(condition_dir: str) -> str:
    """Map a results dir name to a series key ('single_qwenA' -> 'single')."""
    return "single" if condition_dir.startswith("single_") else condition_dir


def read_task_results(path: Path) -> Optional[Dict[str, float]]:
    """Aggregate one task_results.csv into request-weighted latency stats."""
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "n":    float(row["n_requests"]),
                    "avg":  float(row["avg_latency_ms"]),
                    "p95":  float(row["p95_latency_ms"]),
                    "p99":  float(row["p99_latency_ms"]),
                    "tput": float(row["throughput_rps"]),
                })
            except (KeyError, ValueError):
                continue
    if not rows:
        return None
    total_n = sum(r["n"] for r in rows) or 1.0
    return {
        "avg":  sum(r["avg"] * r["n"] for r in rows) / total_n,
        # Tails across tasks: the worst task is what a user actually feels.
        "p95":  max(r["p95"] for r in rows),
        "p99":  max(r["p99"] for r in rows),
        "tput": sum(r["tput"] for r in rows),
        "n_requests": total_n,
        "n_tasks_measured": len(rows),
    }


def load_sweep(results: Path) -> Dict[float, Dict[str, Dict[int, dict]]]:
    """-> {rps: {series_key: {ntasks: record}}}; record has 'stats' or 'failure'."""
    out: Dict[float, Dict[str, Dict[int, dict]]] = {}
    for ntasks_dir in sorted(results.glob("ntasks_*")):
        m = re.match(r"ntasks_(\d+)$", ntasks_dir.name)
        if not m:
            continue
        ntasks = int(m.group(1))
        for rps_dir in sorted(ntasks_dir.glob("rps_*")):
            m2 = re.match(r"rps_([0-9.]+)$", rps_dir.name)
            if not m2:
                continue
            rps = float(m2.group(1))
            for cond_dir in sorted(p for p in rps_dir.iterdir() if p.is_dir()):
                key = canon(cond_dir.name)
                if key not in SERIES_COLORS:
                    continue
                rec: dict = {"dir": cond_dir.name}
                tr = cond_dir / "task_results.csv"
                fail = cond_dir / "failure.json"
                stats = read_task_results(tr) if tr.exists() else None
                if stats is not None:
                    rec["stats"] = stats
                elif fail.exists():
                    try:
                        rec["failure"] = json.loads(fail.read_text())
                    except json.JSONDecodeError:
                        rec["failure"] = {"reason": "unknown"}
                elif tr.exists():
                    # Ran, produced a header and nothing else — the engine died
                    # after the server came up. Older sweeps have no failure.json.
                    rec["failure"] = {"reason": "zero_requests"}
                else:
                    continue
                out.setdefault(rps, {}).setdefault(key, {})[ntasks] = rec
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

OOM_LABEL_FONTSIZE = 5.0
OOM_MARKER_CLEARANCE = 0.035   # gap above the anchoring marker, in span fractions
OOM_ROW_GAP = 0.02             # gap between two stacked labels at the same x
OOM_TOP_PAD = 0.01             # keep the top label inside the axes


def _text_height_frac(ax: plt.Axes) -> float:
    """Height of one OOM label as a fraction of the y-axis span."""
    ax_h_px = ax.get_window_extent().height
    if ax_h_px <= 0:
        return 0.09
    return (OOM_LABEL_FONTSIZE * ax.figure.dpi / 72.0) / ax_h_px


def _annotate_oom(ax: plt.Axes, fails_by_series: Dict[str, List[int]],
                  curve_points: Dict[int, Dict[str, float]]) -> None:
    """Write "OOM" at every #Tasks each series could not be placed at.

    Same convention as RTVSntask/tpc/plot.py — the word "OOM" in the series
    colour marks where that condition stopped fitting on the GPU — but written
    at every failing point rather than only the first, and anchored just above
    the FMVisor point at that #Tasks. FMVisor is the series that survives the
    whole sweep, so its curve is the natural baseline to read the labels
    against: at each x the reader sees what FMVisor did and, stacked directly
    above it, which conditions could not run there at all.

    curve_points: {ntasks: {series: y}} — the measurements actually plotted,
    used to find the anchor (FMVisor where present, otherwise the topmost
    surviving point at that #Tasks).
    """
    if not fails_by_series:
        return
    rows = [s for s in SERIES_ORDER if fails_by_series.get(s)]
    y_lo, y_hi = ax.get_ylim()
    span = y_hi - y_lo
    if span <= 0:
        return

    th = _text_height_frac(ax)
    step = th + OOM_ROW_GAP

    labelled_x = sorted({x for xs in fails_by_series.values() for x in xs})
    for x in labelled_x:
        at_x = curve_points.get(x, {})
        if "sharing" in at_x:
            anchor = at_x["sharing"]
        elif at_x:
            anchor = max(at_x.values())
        else:
            anchor = y_lo
        base = (anchor - y_lo) / span + OOM_MARKER_CLEARANCE

        stack = [s for s in rows if x in fails_by_series[s]]
        # Slide the stack down if it would run off the top of the axes.
        top = base + step * (len(stack) - 1) + th
        if top > 1.0 - OOM_TOP_PAD:
            base -= top - (1.0 - OOM_TOP_PAD)

        for row, series in enumerate(stack):
            ax.text(x, y_lo + (base + row * step) * span, "OOM",
                    ha="center", va="bottom",
                    fontsize=OOM_LABEL_FONTSIZE, fontweight="bold",
                    color=SERIES_COLORS[series],
                    zorder=6)


def plot_ntasks_line_sweep(by_cond: Dict[str, Dict[int, dict]], rps: float,
                           metric: str, ylabel: str, out_path: Path,
                           y_max: Optional[float] = None,
                           fig_w: float = NTASKS_SWEEP_FIG_W,
                           fig_h: float = NTASKS_SWEEP_FIG_H,
                           mark_failures: bool = True) -> None:
    present = [s for s in SERIES_ORDER if s in by_cond]
    if not present:
        print(f"[Warn] No conditions present, skipping {out_path.name}")
        return

    ntasks_list = sorted({n for s in by_cond.values() for n in s})
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ymax_data = 0.0

    oom_fails: Dict[str, List[int]] = {}
    curve_points: Dict[int, Dict[str, float]] = {}
    for s in present:
        series = by_cond[s]
        xs = [n for n in ntasks_list if "stats" in series.get(n, {})]
        ys = [series[n]["stats"][metric] for n in xs]
        if xs:
            ymax_data = max(ymax_data, max(ys))
            for n, y in zip(xs, ys):
                curve_points.setdefault(n, {})[s] = y
            ax.plot(xs, ys,
                    color=SERIES_COLORS[s],
                    linestyle=SERIES_LINESTYLE[s],
                    marker=SERIES_MARKER[s],
                    markersize=COND_LINE_MARKERSIZE,
                    markerfacecolor=SERIES_COLORS[s],
                    markeredgecolor=COND_LINE_MARKER_EDGECOLOR,
                    markeredgewidth=COND_LINE_MARKEREDGEWIDTH,
                    linewidth=COND_LINE_LINEWIDTH,
                    zorder=3)

        # Placement ceiling: every #Tasks this condition could not be placed
        # at, annotated "OOM" in the series colour — same convention as
        # tpc/plot.py.
        if mark_failures:
            fails = sorted(n for n in ntasks_list if "failure" in series.get(n, {}))
            if fails:
                oom_fails[s] = fails

    ax.set_xlabel("#Tasks")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ntasks_list)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.grid(axis="both", zorder=0)
    ax.set_axisbelow(True)

    y_top = y_max if y_max else _nice_upper(ymax_data * 1.07)
    _set_linear_axis_with_endpoint(ax, "y", 0.0, y_top, target_ticks=5)
    n_lo, n_hi = min(ntasks_list), max(ntasks_list)
    ax.set_xlim(n_lo - 0.55, n_hi + 0.55)

    _annotate_oom(ax, oom_fails, curve_points)

    fig.tight_layout(pad=0.2)
    save_figure_exact(fig, out_path)
    plt.close(fig)


def write_summary(rps: float, by_cond: Dict[str, Dict[int, dict]],
                  out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rps_per_task", "series", "condition", "num_tasks", "status",
                    "avg_latency_ms", "p95_latency_ms", "p99_latency_ms",
                    "throughput_rps", "n_requests", "failure_reason"])
        for key in SERIES_ORDER:
            for n in sorted(by_cond.get(key, {})):
                rec = by_cond[key][n]
                if "stats" in rec:
                    s = rec["stats"]
                    w.writerow([f"{rps:g}", SERIES_LABELS[key], rec["dir"], n, "ok",
                                round(s["avg"], 3), round(s["p95"], 3),
                                round(s["p99"], 3), round(s["tput"], 3),
                                int(s["n_requests"]), ""])
                else:
                    reason = rec["failure"].get("reason", "unknown")
                    w.writerow([f"{rps:g}", SERIES_LABELS[key], rec["dir"], n, "failed",
                                "", "", "", "", "",
                                REASON_LABEL.get(reason, reason)])
    print(f"[Plot] Saved: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/RTVSntask/vlm/results",
                    help="Sweep results dir (contains ntasks_*/).")
    ap.add_argument("--out", default=None,
                    help="Where to write figures (default: <results>/plots).")
    ap.add_argument("--series", default=",".join(DEFAULT_INCLUDED_SERIES),
                    help="Comma-separated series to include "
                         f"(any of {','.join(SERIES_ORDER)}).")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"],
                    help="Figure format (default: pdf, matching vision/plot.py "
                         "and tpc/plot.py).")
    ap.add_argument("--y-max", type=float, default=None,
                    help="Hard ceiling for the latency y-axis (ms). Set the same "
                         "value across GPUs to make figures directly comparable.")
    ap.add_argument("--no-mark-failures", dest="mark_failures",
                    action="store_false",
                    help="Do not write \"OOM\" at the first #Tasks each condition "
                         "could not be placed. On by default, matching "
                         "tpc/plot.py. Failures are always listed in "
                         "summary_rps<R>.csv either way.")
    ap.set_defaults(mark_failures=True)
    args = ap.parse_args()

    results = Path(args.results).resolve()
    if not results.is_dir():
        print(f"ERROR: results dir not found: {results}")
        return 1
    out_dir = Path(args.out).resolve() if args.out else results / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = [s.strip() for s in args.series.split(",") if s.strip()]
    bad = [s for s in wanted if s not in SERIES_COLORS]
    if bad:
        print(f"ERROR: unknown series {bad}; choose from {SERIES_ORDER}")
        return 1

    apply_paper_style()

    sweep = load_sweep(results)
    if not sweep:
        print(f"ERROR: no ntasks_*/rps_*/ results under {results}")
        return 1

    for rps, by_cond in sorted(sweep.items()):
        by_cond = {k: v for k, v in by_cond.items() if k in wanted}
        if not by_cond:
            continue
        tag = f"{rps:g}"
        ext = args.format
        present = [s for s in SERIES_ORDER if s in by_cond]
        plot_standalone_legend(_legend_handles(present),
                               out_dir / f"ntasks_latency_legend.{ext}",
                               ncol=min(len(present), 4))
        for metric, ylabel, stem in (
            ("avg",  "Latency (ms)",     "ntasks_mean_latency"),
            ("p99",  "P99 Latency (ms)", "ntasks_p99_latency"),
            ("tput", "Throughput (rps)", "ntasks_throughput"),
        ):
            plot_ntasks_line_sweep(
                by_cond, rps, metric, ylabel,
                out_dir / f"{stem}_rps{tag}.{ext}",
                y_max=args.y_max if metric != "tput" else None,
                mark_failures=args.mark_failures)
        write_summary(rps, by_cond, out_dir / f"summary_rps{tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
