"""
Microbenchmark plots — publication quality, fits in 1/4 of a single column.

3×3 grid (rows = Memory | Load Time | Latency, cols = Time Series | Vision | VLM).

Usage (from serving/):
  python experiments/modalities/plot.py
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         5.5,
    "axes.titlesize":    6,
    "axes.labelsize":    5.5,
    "xtick.labelsize":   5,
    "ytick.labelsize":   5,
    "legend.fontsize":   4.8,
    "lines.linewidth":   0.6,
    "axes.linewidth":    0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size":  2,
    "ytick.major.size":  2,
    "figure.dpi":        300,
    "savefig.dpi":       300,
})

BENCH_DIR = Path(__file__).parent
SOURCES = {
    "Time Series": BENCH_DIR / "tsfm"   / "results" / "summary.csv",
    "Vision":      BENCH_DIR / "vision" / "results" / "summary.csv",
    "VLM":         BENCH_DIR / "vlm"    / "results" / "summary.csv",
}
OUT_DIR = BENCH_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)

C_BB   = "#4878CF"
C_TASK = "#E06C75"

BB_DISPLAY = {
    "phi":         "Phi-3.5",
    "qwen-2B":     "Qwen-2B",
    "dinobase":    "DINOv2",
    "swinsmall":   "Swin-S",
    "swinbase":    "Swin-B",
    "momentbase":  "Moment",
    "chronosbase": "Chronos",
}


def load_by_backbone(path: Path) -> dict:
    df = pd.read_csv(path)
    return {bb: grp.mean(numeric_only=True) for bb, grp in df.groupby("backbone")}


def get(row: pd.Series, col: str) -> float:
    if col in row.index and not pd.isna(row[col]):
        return float(row[col])
    return 0.0


def _clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _xtick_labels(backbones):
    return [BB_DISPLAY.get(b, b) for b in backbones]


def _fmt(v):
    if v == 0:
        return "0"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _nice_ceil(ymax):
    magnitude = 10 ** np.floor(np.log10(max(ymax, 1e-9)))
    for frac in [0.05, 0.1, 0.2, 0.25, 0.5, 1.0]:
        step = frac * magnitude
        ceil_val = np.ceil(ymax / step) * step
        if ceil_val <= ymax * 1.25:
            return ceil_val, step
    return np.ceil(ymax / magnitude) * magnitude, magnitude


def _bar_labels(ax, xi, bb_vals, task_vals, total_vals, fs=4):
    for x, bot, task, tot in zip(xi, bb_vals, task_vals, total_vals):
        ax.text(x, bot / 2, _fmt(bot),
                ha="center", va="center", fontsize=fs,
                color="white", fontweight="bold", zorder=5)
        if task > 0:
            ax.text(x, tot * 1.03, _fmt(task),
                    ha="center", va="bottom", fontsize=fs,
                    color="black", zorder=5)


def _set_yaxis(ax, ymax_with_headroom):
    ceil_val, _ = _nice_ceil(ymax_with_headroom)
    ax.set_ylim(0, ceil_val)
    ax.set_yticks([0, ceil_val / 2, ceil_val])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{int(round(v))}"
    ))


def fill_col(axes, bb_data: dict, col_title: str):
    backbones = list(bb_data.keys())
    xi        = np.arange(len(backbones))
    w         = 0.5
    xlabels   = _xtick_labels(backbones)

    # ── Row 0 : Memory ────────────────────────────────────────────────────────
    ax = axes[0]
    bb_mem   = np.array([get(bb_data[b], "backbone memory(MB)") for b in backbones])
    task_mem = np.array([get(bb_data[b], "decoder memory(MB)") or
                         get(bb_data[b], "adapter memory(MB)") for b in backbones])
    ax.bar(xi, bb_mem,   w, color=C_BB,   label="Backbone", zorder=3)
    ax.bar(xi, task_mem, w, color=C_TASK, label="Task",     bottom=bb_mem, zorder=3)
    _bar_labels(ax, xi, bb_mem, task_mem, bb_mem + task_mem)
    _set_yaxis(ax, (bb_mem + task_mem).max() * 1.2)
    ax.set_title(col_title, pad=2)
    ax.set_xticks(xi)
    ax.set_xticklabels([])
    ax.grid(axis="y", linewidth=0.3, linestyle="--", alpha=0.5, zorder=0)
    _clean_spines(ax)

    # ── Row 1 : Load time ─────────────────────────────────────────────────────
    ax = axes[1]
    bb_load   = np.array([get(bb_data[b], "backbone load time(ms)") for b in backbones])
    task_load = np.array([get(bb_data[b], "decoder load time(ms)") or
                          get(bb_data[b], "adapter load time(ms)") for b in backbones])
    ax.bar(xi, bb_load,   w, color=C_BB,   label="Backbone", zorder=3)
    ax.bar(xi, task_load, w, color=C_TASK, label="Task",     bottom=bb_load, zorder=3)
    _bar_labels(ax, xi, bb_load, task_load, bb_load + task_load)
    _set_yaxis(ax, (bb_load + task_load).max() * 1.2)
    ax.set_xticks(xi)
    ax.set_xticklabels([])
    ax.grid(axis="y", linewidth=0.3, linestyle="--", alpha=0.5, zorder=0)
    _clean_spines(ax)

    # ── Row 2 : Latency ───────────────────────────────────────────────────────
    ax = axes[2]
    bb_lat  = np.array([get(bb_data[b], "backbone_mean_ms") for b in backbones])
    dec_lat = np.array([get(bb_data[b], "decoder_mean_ms")  for b in backbones])
    tot_lat = bb_lat + dec_lat
    ax.bar(xi, bb_lat,  w, color=C_BB,   label="Backbone", zorder=3)
    ax.bar(xi, dec_lat, w, color=C_TASK, label="Task",     bottom=bb_lat, zorder=3)
    _bar_labels(ax, xi, bb_lat, dec_lat, tot_lat)
    _set_yaxis(ax, tot_lat.max() * 1.2)
    ax.set_xticks(xi)
    ax.set_xticklabels(xlabels, rotation=20, ha="right")
    ax.grid(axis="y", linewidth=0.3, linestyle="--", alpha=0.5, zorder=0)
    _clean_spines(ax)


def make_plots():
    type_data = {}
    for label, path in SOURCES.items():
        if path.exists():
            type_data[label] = load_by_backbone(path)
        else:
            print(f"[plot] Missing: {path}")

    if not type_data:
        print("[plot] No data found.")
        return

    types = list(type_data.keys())
    ncols = len(types)

    fig, axes = plt.subplots(
        3, ncols,
        figsize=(3.3, 2.8),
        gridspec_kw={"hspace": 0.32, "wspace": 0.48},
    )
    if ncols == 1:
        axes = axes.reshape(3, 1)

    for col_idx, label in enumerate(types):
        fill_col(axes[:, col_idx], type_data[label], label)

    axes[0, 0].set_ylabel("Memory (MB)")
    axes[1, 0].set_ylabel("Load time (ms)")
    axes[2, 0].set_ylabel("Latency (ms)")

    handles = [
        matplotlib.patches.Patch(color=C_BB,   label="Backbone"),
        matplotlib.patches.Patch(color=C_TASK, label="Task"),
    ]
    fig.subplots_adjust(top=0.86)
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        fontsize=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )

    fig.savefig(OUT_DIR / "modalities.pdf")
    fig.savefig(OUT_DIR / "modalities.png", dpi=300)
    plt.close(fig)
    print(f"[plot] Saved {OUT_DIR / 'modalities.pdf'} and modalities.png")


if __name__ == "__main__":
    make_plots()
