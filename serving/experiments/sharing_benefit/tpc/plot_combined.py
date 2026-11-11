#!/usr/bin/env python3
"""Combined three-backbone latency panels (a merge of Figures 7 and 8).

Figures 7 and 8 present the same experiment — two tasks sharing a backbone,
swept over per-task request rate — split across two figures and four panels.
This draws all three backbones as one row per statistic:

    latency_mean_3backbones.pdf   mean latency vs RPS/task
    latency_p99_3backbones.pdf    p99  latency vs RPS/task

Both are one \\columnwidth wide with a single shared y-axis label and one shared
legend, so the three backbones read as one comparison. The original per-backbone
plots are untouched; this writes to its own output directory.

Usage:
    python experiments/sharing_benefit/tpc/plot_combined.py
    python experiments/sharing_benefit/tpc/plot_combined.py --share-y
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import plot as P  # the per-backbone module: loaders, palette, style, exact save

# One \columnwidth. Three panels share it, so each is ~1.11in; the shared y-label
# and (with --share-y) the single tick column buy that width back.
FIG_W = 3.33
FIG_H = 1.30

# Backbone panels, in paper order: (title, task_set, results dir, y caps).
# (short panel title, task_set, results dir). The title is what fits above a
# 0.66in panel: "(a) Moment" is 0.64in at 8pt, "(a) MOMENT-Large" is 1.03in and
# would spill into the neighbouring panel. Full names go in the LaTeX caption.
#
# The last element is {stat: y cap}, overriding LINE_RPS_PLOT_OVERRIDES for that
# panel alone. Both vision backbones need one, because the table is keyed by task
# set and they share "vision", and because their SP curves dwarf everything else:
# 1720ms p99 for DINOv2-B, 31,600ms mean / 59,600ms p99 for Swin-L, against
# non-SP points that stay under 250ms and 3400ms respectively. A data-driven axis
# that tall squashes the other three series into the bottom of the panel.
#
# The p99 caps are 350, which is also the bound DINOv2-B's mean panel picks on
# its own, so the two vision panels read on one scale there. Swin-L's mean panel
# is capped at 700 instead: at 350 its FMVisor curve left the top at RPS 20
# (691ms), and the paper quotes that point, so the reader could not see the
# number being claimed. At 700 the whole FMVisor curve is on scale and only BE
# (1285ms at RPS 20) and SP (31,600ms) run off the top, which is the intended
# reading — both are saturating there.
PANELS = [
    ("Moment-L", "tsfm",   "t4/results_tsfm",            {}),
    ("DINOv2-B", "vision", "t4/results_vision",          {"p99": 350.0}),
    ("Swin-L",   "vision", "t4/results_vision_swinlarge",
     {"mean": 700.0, "p99": 350.0}),
]

NTASKS = 2  # both figures report the two-task co-location


def load_panel(root: Path, task_set: str, stat: str) -> Dict[str, Dict[int, float]]:
    """{series: {rps: value}} for one backbone.

    The loaders in plot.py read two module globals to resolve the single-task
    (ST) baseline, and they differ per task set, so set them before each call.
    """
    P.CURRENT_TASK_SET = task_set
    P.SINGLE_CONDITIONS = P.SINGLE_CONDITIONS_BY_TASK_SET[task_set]

    ntasks_dir = root / f"ntasks_{NTASKS}"
    rps_list = sorted(int(p.name.split("_")[1]) for p in ntasks_dir.glob("rps_*"))
    loader = P.load_mean_latency_for_dir if stat == "mean" else P.load_p99_for_dir

    out: Dict[str, Dict[int, float]] = {}
    for rps in rps_list:
        for series, value in loader(ntasks_dir / f"rps_{rps}").items():
            out.setdefault(series, {})[rps] = value
    return out


def plot_row(base: Path, stat: str, out_path: Path, share_y: bool,
             fig_h: float = FIG_H) -> None:
    data = [(title, ts, load_panel(base / d, ts, stat), caps.get(stat))
            for title, ts, d, caps in PANELS]

    P.apply_paper_style()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(FIG_W, fig_h), sharey=share_y)

    present: List[str] = [s for s in P.SERIES_ORDER
                          if any(s in panel for _, _, panel, _ in data)]

    for idx, (ax, (title, task_set, panel, panel_cap)) in enumerate(zip(axes, data)):
        rps_list = sorted({r for s in panel.values() for r in s})

        # The cap has to be known before drawing, not just before set_ylim: the
        # markers below are unclipped, and an artist tens of thousands of ms off
        # canvas makes tight_layout give up and leave its rect unapplied.
        # Same y scale as the per-backbone Figures 7/8: the panel's own cap where
        # it has one, else the fixed cap from LINE_RPS_PLOT_OVERRIDES, else the
        # same tight data-driven bound. _set_linear_axis_with_endpoint lands the
        # final tick exactly on the top of the axis, so every axis ends on a number.
        cap = panel_cap if panel_cap is not None else P.LINE_RPS_PLOT_OVERRIDES.get(
            task_set, {}).get(NTASKS, {}).get(f"{stat}_ymax")
        vals = [v for pts in panel.values() for v in pts.values()]
        upper = cap if cap is not None else P._tight_linear_upper(max(vals))
        for series in present:
            pts = panel.get(series)
            if not pts:
                continue
            xs = [r for r in rps_list if r in pts]
            # Line clipped at the axis, so an off-scale series runs off the top
            # as it does in the per-backbone figures; markers drawn unclipped but
            # only for in-range points. The first and last x values sit exactly
            # on the axis limits, so a clipped marker there would be drawn as a
            # half circle. plot.py does the same for its edge markers.
            ax.plot(
                xs, [pts[r] for r in xs],
                color=P.SERIES_COLORS[series],
                linestyle=P.SERIES_LINESTYLE[series],
                marker="None",
                linewidth=0.9,
                label=P.SERIES_LABELS[series],
            )
            xin = [r for r in xs if pts[r] <= upper]
            ax.plot(
                xin, [pts[r] for r in xin],
                color=P.SERIES_COLORS[series],
                linestyle="None",
                marker=P.SERIES_MARKER[series],
                markersize=2.2,
                markerfacecolor=P.SERIES_COLORS[series],
                markeredgecolor="black",
                markeredgewidth=0.25,
                clip_on=False,
                zorder=3,
            )
        ax.set_xticks(rps_list)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xlim(rps_list[0], rps_list[-1])
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)

        P._set_linear_axis_with_endpoint(ax, axis="y", lower=0.0, upper=upper,
                                         target_ticks=4, decimals=0)

    # One x-label centred under the row, and one y-label for all three panels,
    # instead of repeating "RPS/task" and "Latency (ms)" three times each.
    # matplotlib 3.3 has no Figure.supxlabel/supylabel (added in 3.4), so place
    # the shared labels directly.
    lab = plt.rcParams["axes.labelsize"]
    fig.text(0.005, 0.615, "Latency (ms)", rotation=90,
             ha="left", va="center", fontsize=lab)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(present),
               frameon=False, fontsize=8.0, markerscale=1.35,
               handlelength=1.5, handletextpad=0.4,
               columnspacing=1.0, borderaxespad=0.0,
               bbox_to_anchor=(0.5, 1.005))

    # rect leaves room for the shared y-label (left), x-label (bottom) and legend (top).
    fig.tight_layout(pad=0.2, rect=(0.055, 0.28, 0.985, 0.885))
    fig.subplots_adjust(wspace=0.55)

    # Panel names go below the shared x-axis label, in subcaption position. They
    # are placed after tight_layout so each sits centred on its final axes; at
    # the bottom row the neighbouring tick labels are no longer alongside, so
    # there is more room here than there was above the panels.
    # "RPS/task" hangs directly under the x tick labels rather than floating
    # midway to the panel names: one tick-label line plus a small pad below the
    # axes, measured in figure fractions so it tracks --height.
    box0 = axes[0].get_position()
    # A tick label occupies its point size plus matplotlib's tick pad and the
    # glyph descent; +7 measured true, +4 left the label overlapping the ticks.
    tick_row = (plt.rcParams["xtick.labelsize"] + 7.0) / (fig.get_size_inches()[1] * 72)
    fig.text(0.55, box0.y0 - tick_row - 0.02, "RPS/task",
             ha="center", va="top", fontsize=lab)

    line_h = lab / (fig.get_size_inches()[1] * 72)
    names_y = box0.y0 - tick_row - 0.02 - line_h - 0.045
    for idx, (ax, (title, _, _, _)) in enumerate(zip(axes, data)):
        box = ax.get_position()
        fig.text(box.x0 + box.width / 2, names_y,
                 f"({chr(ord('a') + idx)}) {title}",
                 ha="center", va="top", fontsize=lab)
    P.save_figure_exact(fig, out_path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=EXP_DIR,
                    help="Directory holding t4/results_* (default: this experiment dir)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: <base>/t4/plots_combined)")
    ap.add_argument("--height", type=float, default=FIG_H,
                    help=f"Figure height in inches (default {FIG_H}).")
    ap.add_argument("--share-y", action="store_true",
                    help="Share one y scale across the three panels. Fine for mean "
                         "(21-277ms); for p99 the DINOv2 SP point at 1720ms flattens "
                         "the other two backbones.")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.base / "t4" / "plots_combined")
    for stat, name in (("mean", "latency_mean_3backbones.pdf"),
                       ("p99",  "latency_p99_3backbones.pdf")):
        plot_row(args.base, stat, out_dir / name, args.share_y, args.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
