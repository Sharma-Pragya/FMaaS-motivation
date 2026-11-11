#!/usr/bin/env python3
"""Combined #Tasks-sweep panels for Figures 9 and 10.

Same layout as sharing_benefit/tpc/plot_combined.py: one \\columnwidth row of
panels, a single shared y-axis label, one shared legend above, and the backbone
names as (a)/(b)/(c) subcaptions below the shared x-axis label.

    ntasks_mean_vision_tsfm.pdf   Figure 9  — MOMENT-Large, DINOv2-Base, Swin-Large
    ntasks_mean_llm_vlm.pdf       Figure 10 — Qwen-3B, Qwen2-VL-7B

The per-backbone plots are untouched; this writes to its own output directory.
Data comes from the four per-modality modules, which use two different shapes:
tpc/vision expose load_mean_latency_for_dir() per rps dir, llm/vlm expose
load_sweep() returning {rps: {series: {ntasks: {"stats"|"failure"}}}}.

Usage:
    python experiments/RTVSntask/plot_combined_ntasks.py
    python experiments/RTVSntask/plot_combined_ntasks.py --height 1.45
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple

EXP_DIR = Path(__file__).resolve().parent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FIG_W = 3.33
FIG_H = 1.30

# (short title, module kind, results dir, rps, y cap or None for data-driven)
FIG9 = [
    ("Moment-L", "tpc",    "tpc/results_tsfm_t4",     5, 300.0),
    ("DINOv2-B", "vision", "vision/results_dinobase", 5, 300.0),
    ("Swin-L",   "vision", "vision/results_swinlarge", 2, 300.0),
]
FIG10 = [
    ("Qwen-3B",    "llm", "llm/results_llm_l40s_qwen3b",             2, None),
    ("Qwen2-VL-7B", "vlm", "vlm/results_vlm_l40s_qwen2vl7b_eos32",   2, None),
]

_MODULES: Dict[str, object] = {}


def load_module(kind: str):
    """Import <kind>/plot.py under a unique name (all four are called 'plot')."""
    if kind in _MODULES:
        return _MODULES[kind]
    path = EXP_DIR / kind / "plot.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"rtv_{kind}_plot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MODULES[kind] = mod
    return mod


def load_panel(kind: str, root: Path, rps: int
               ) -> Tuple[Dict[str, Dict[int, float]], List[int]]:
    """Return ({series: {ntasks: mean latency ms}}, sorted ntasks list)."""
    mod = load_module(kind)
    out: Dict[str, Dict[int, float]] = {}

    if kind in ("tpc", "vision"):
        ns = sorted(int(p.name.split("_")[1]) for p in root.glob("ntasks_*"))
        for n in ns:
            for series, value in mod.load_mean_latency_for_dir(
                    root / f"ntasks_{n}" / f"rps_{rps}").items():
                out.setdefault(series, {})[n] = value
    else:
        sweep = mod.load_sweep(root)
        by_cond = sweep[float(rps)]
        ns = sorted({n for byn in by_cond.values() for n in byn})
        for series, byn in by_cond.items():
            for n, entry in byn.items():
                if "stats" in entry:
                    out.setdefault(series, {})[n] = entry["stats"]["avg"]
    return out, ns


def oom_points(panel: Dict[str, Dict[int, float]], ns: List[int]
               ) -> Dict[str, List[int]]:
    """{series: [ntasks it could not be placed at]} — a gap after real data."""
    fails: Dict[str, List[int]] = {}
    for series, pts in panel.items():
        seen = False
        missing: List[int] = []
        for n in ns:
            if n in pts:
                seen = True
            elif seen:
                missing.append(n)
        if missing:
            fails[series] = missing
    return fails


def plot_row(panels, out_path: Path, fig_h: float) -> None:
    style = load_module("vision")
    loaded = [(title, *load_panel(kind, EXP_DIR / d, rps), cap)
              for title, kind, d, rps, cap in panels]

    style.apply_paper_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(FIG_W, fig_h))
    axes = list(axes) if len(panels) > 1 else [axes]

    order = [s for s in style.ACTIVE_SERIES_ORDER
             if any(s in p for _, p, _, _ in loaded)]

    pending_oom = []
    for ax, (title, panel, ns, cap) in zip(axes, loaded):
        # The cap has to be known before drawing: SP runs to ~35,000ms against a
        # 300ms axis, and an unclipped artist that far off canvas makes
        # tight_layout give up entirely (leaving the rect unapplied).
        vals = [v for pts in panel.values() for v in pts.values()]
        upper = min(cap, max(vals) * 1.10) if cap else style._nice_upper(max(vals) * 1.07)

        curve_points: Dict[int, Dict[str, float]] = {}
        for series in order:
            pts = panel.get(series)
            if not pts:
                continue
            xs = [n for n in ns if n in pts]
            for n in xs:
                curve_points.setdefault(n, {})[series] = pts[n]
            # Line clipped at the axis (so an off-scale series runs off the top,
            # as in the per-backbone figures); markers drawn unclipped but only
            # for in-range points, so the ones on the x limits stay whole.
            ax.plot(xs, [pts[n] for n in xs],
                    color=style.SERIES_COLORS[series],
                    linestyle=style.SERIES_LINESTYLE[series],
                    marker="None", linewidth=0.9,
                    label=style.SERIES_LABELS[series])
            xin = [n for n in xs if pts[n] <= upper]
            ax.plot(xin, [pts[n] for n in xin],
                    color=style.SERIES_COLORS[series],
                    linestyle="None", marker=style.SERIES_MARKER[series],
                    markersize=2.2, markerfacecolor=style.SERIES_COLORS[series],
                    markeredgecolor="black", markeredgewidth=0.25,
                    clip_on=False, zorder=3)

        ax.set_xticks(ns)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xlim(ns[0], ns[-1])
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        style._set_linear_axis_with_endpoint(ax, axis="y", lower=0.0,
                                             upper=upper, target_ticks=4, decimals=0)
        pending_oom.append((ax, oom_points(panel, ns), curve_points))

    lab = plt.rcParams["axes.labelsize"]
    fig.text(0.005, 0.615, "Latency (ms)", rotation=90,
             ha="left", va="center", fontsize=lab)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(order),
               frameon=False, fontsize=8.0, markerscale=1.35,
               handlelength=1.5, handletextpad=0.4, columnspacing=1.0,
               borderaxespad=0.0, bbox_to_anchor=(0.5, 1.005))

    fig.tight_layout(pad=0.2, rect=(0.055, 0.30, 0.985, 0.865))
    fig.subplots_adjust(wspace=0.72)

    # After the layout, not before: _annotate_oom sizes its row step from the
    # axes height in pixels, so annotating first and letting tight_layout shrink
    # the axes underneath makes stacked labels overlap each other.
    for ax, fails, pts in pending_oom:
        style._annotate_oom(ax, fails, pts)

    box0 = axes[0].get_position()
    tick_row = (plt.rcParams["xtick.labelsize"] + 7.0) / (fig.get_size_inches()[1] * 72)
    fig.text(0.55, box0.y0 - tick_row - 0.02, "#Tasks",
             ha="center", va="top", fontsize=lab)

    line_h = lab / (fig.get_size_inches()[1] * 72)
    names_y = box0.y0 - tick_row - 0.02 - line_h - 0.045
    for idx, (ax, (title, _, _, _)) in enumerate(zip(axes, loaded)):
        box = ax.get_position()
        fig.text(box.x0 + box.width / 2, names_y,
                 f"({chr(ord('a') + idx)}) {title}",
                 ha="center", va="top", fontsize=lab)

    style.save_figure_exact(fig, out_path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=EXP_DIR / "plots_combined")
    ap.add_argument("--height", type=float, default=FIG_H)
    args = ap.parse_args()

    plot_row(FIG9,  args.out_dir / "ntasks_mean_vision_tsfm.pdf", args.height)
    plot_row(FIG10, args.out_dir / "ntasks_mean_llm_vlm.pdf",     args.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
