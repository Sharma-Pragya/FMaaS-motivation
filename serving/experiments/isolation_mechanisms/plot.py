#!/usr/bin/env python3
"""Plot isolation_mechanisms results (summary.csv written by run.py).

One overview figure per backbone (overview_<backbone>), six panels, x = number
of tasks:
  (a) p50 latency            (b) latency overhead vs model_only (p50, p99)
  (c) throughput             (d) total GPU memory (NVML, all processes)
  (e) GPU memory split: PyTorch allocator vs everything else (CUDA context,
      cuBLAS/cuDNN workspaces) = NVML total - torch reserved
  (f) startup time (all processes ready)
With several backbones in the CSV, summary_backbones compares them at each
backbone's largest task count: p50 / p99 overhead vs model only, GPU memory.

If a (backbone, mode, ntasks) point was run more than once, the last row wins.

Usage (from serving/):
    python experiments/isolation_mechanisms/plot.py
    python experiments/isolation_mechanisms/plot.py --csv <summary.csv> --backbone resnet50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent

MODES = ["model_only", "thread_stream", "process_context", "process_context_ipc",
         "process_context_ipc_mps", "instance_per_task", "instance_per_task_mps"]
LABELS = {
    "model_only": "Model only",
    "thread_stream": "Thread + stream",
    "process_context": "Process + context",
    "process_context_ipc": "Process + context + CUDA IPC",
    "process_context_ipc_mps": "Process + context + CUDA IPC + MPS",
    "instance_per_task": "No sharing (model per task)",
    "instance_per_task_mps": "No sharing + MPS",
}
# Reference categorical palette; each mode keeps the slot it was first given
# (color follows the entity), markers are the secondary encoding.
COLORS = {"model_only": "#2a78d6", "thread_stream": "#eb6834",
          "process_context_ipc": "#1baf7a", "process_context_ipc_mps": "#eda100",
          "process_context": "#e87ba4", "instance_per_task": "#008300",
          "instance_per_task_mps": "#6250d6"}
MARKERS = {"model_only": "o", "thread_stream": "s", "process_context": "v",
           "process_context_ipc": "^", "process_context_ipc_mps": "D",
           "instance_per_task": "P", "instance_per_task_mps": "X"}

PRETTY = {"momentlarge": "MOMENT-Large", "papageip": "PaPaGei", "dinobase": "DINOv2-Base",
          "swinlarge": "Swin-Large", "resnet50": "ResNet50"}
BACKBONE_ORDER = ["momentlarge", "papageip", "dinobase", "swinlarge", "resnet50"]

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"


def load(raw: pd.DataFrame, backbone: str) -> pd.DataFrame:
    df = raw[raw["backbone"] == backbone].copy()
    df = df.drop_duplicates(subset=["mode", "ntasks"], keep="last")
    # NVML may not report per-client memory under MPS (or across PID
    # namespaces); fall back to the device-level delta for those rows.
    missing = df["gpu_mem_procs_mb"] <= 0
    if missing.any():
        print(f"[plot] per-process GPU memory missing for "
              f"{sorted(set(zip(df.loc[missing, 'mode'], df.loc[missing, 'ntasks'])))}; "
              f"using device delta")
        df.loc[missing, "gpu_mem_procs_mb"] = df.loc[missing, "gpu_mem_device_delta_mb"]
    df["other_mb"] = (df["gpu_mem_procs_mb"] - df["torch_reserved_mb"]).clip(lower=0)
    return df.sort_values(["mode", "ntasks"])


def style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=TEXT, pad=8)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8.5, length=0)


def line_panel(ax, df, col: str, ntasks: list[int], label_ends: bool = True) -> None:
    present = [m for m in MODES if m in set(df["mode"])]
    for i, mode in enumerate(present):
        d = df[df["mode"] == mode]
        # Dodge each mode a few percent along the log2 x-axis so series with
        # identical values (e.g. memory of the two process modes) stay visible.
        dodge = 2 ** (0.045 * (i - (len(present) - 1) / 2))
        ax.plot(d["ntasks"] * dodge, d[col], color=COLORS[mode], marker=MARKERS[mode],
                markersize=6, linewidth=2, label=LABELS[mode],
                markeredgecolor=SURFACE, markeredgewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ntasks)
    ax.set_xticklabels([str(n) for n in ntasks])
    ax.set_xlabel("Tasks sharing the backbone", fontsize=9, color=TEXT_2)
    ax.set_ylim(bottom=0)


def bar_positions(ntasks, n_bars, width):
    x = np.arange(len(ntasks))
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * width
    return x, offsets


def plot_overview(df: pd.DataFrame, backbone: str, out_dir: Path) -> None:
    ntasks = sorted(df["ntasks"].unique())
    base = df[df["mode"] == "model_only"].set_index("ntasks")

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), facecolor=SURFACE)
    (ax_lat, ax_ovh, ax_tput), (ax_mem, ax_split, ax_ready) = axes

    # (a) p50 latency
    style_axes(ax_lat, "(a) p50 latency", "ms")
    line_panel(ax_lat, df, "lat_p50_ms", ntasks)

    # (b) overhead vs model_only: p50 filled, p99 hatched outline
    style_axes(ax_ovh, "(b) Latency overhead vs model only", "% over model only")
    present = [m for m in MODES if m in set(df["mode"])]
    cmp_modes = [m for m in present if m != "model_only"]
    width = 0.8 / (2 * len(cmp_modes))
    # Narrow bars: stand value labels on end so neighbours don't collide.
    rot = 90 if len(cmp_modes) > 2 else 0
    x, offsets = bar_positions(ntasks, 2 * len(cmp_modes), width)
    k = 0
    for mode in cmp_modes:
        d = df[df["mode"] == mode].set_index("ntasks").reindex(ntasks)
        for pct, filled in (("lat_p50_ms", True), ("lat_p99_ms", False)):
            ovh = (d[pct] / base[pct].reindex(ntasks) - 1) * 100
            bars = ax_ovh.bar(x + offsets[k], ovh, width * 0.9,
                              color=COLORS[mode] if filled else SURFACE,
                              edgecolor=COLORS[mode], linewidth=1.2,
                              hatch=None if filled else "////")
            # Too many bars to label legibly: the printed table carries the values.
            for b, v in zip(bars, ovh if len(cmp_modes) <= 4 else []):
                if np.isfinite(v):
                    ax_ovh.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.3,
                                f"{v:.1f}", ha="center", va="bottom", fontsize=7,
                                color=TEXT_2, rotation=rot)
            k += 1
    ax_ovh.axhline(0, color=TEXT_2, linewidth=0.8)
    ax_ovh.set_ylim(top=ax_ovh.get_ylim()[1] * (1.15 if rot else 1.0))
    ax_ovh.set_xticks(x)
    ax_ovh.set_xticklabels([str(n) for n in ntasks])
    ax_ovh.set_xlabel("Tasks sharing the backbone", fontsize=9, color=TEXT_2)
    from matplotlib.patches import Patch
    ax_ovh.legend(handles=[Patch(facecolor=TEXT_2, edgecolor=TEXT_2, label="p50"),
                           Patch(facecolor=SURFACE, edgecolor=TEXT_2, hatch="////", label="p99")],
                  frameon=False, fontsize=8, ncol=2, loc="upper right")
    # Headroom so the tallest bars stay clear of the p50/p99 legend.
    ax_ovh.set_ylim(top=ax_ovh.get_ylim()[1] * 1.12)

    # (c) throughput
    style_axes(ax_tput, "(c) Throughput", "requests / s")
    line_panel(ax_tput, df, "throughput_rps", ntasks)

    # (d) total GPU memory
    style_axes(ax_mem, "(d) GPU memory, all processes (NVML)", "MB")
    line_panel(ax_mem, df, "gpu_mem_procs_mb", ntasks)

    # (e) memory split: PyTorch allocator (solid) + everything else (hatched)
    style_axes(ax_split, "(e) GPU memory: PyTorch vs CUDA context & runtime", "MB")
    width = 0.8 / len(present)
    rot = 90 if len(present) > 3 else 0
    x, offsets = bar_positions(ntasks, len(present), width)
    for i, mode in enumerate(present):
        d = df[df["mode"] == mode].set_index("ntasks").reindex(ntasks)
        ax_split.bar(x + offsets[i], d["torch_reserved_mb"], width * 0.9,
                     color=COLORS[mode], edgecolor=SURFACE, linewidth=1)
        ax_split.bar(x + offsets[i], d["other_mb"], width * 0.9,
                     bottom=d["torch_reserved_mb"], color=SURFACE,
                     edgecolor=COLORS[mode], linewidth=1.2, hatch="////")
        for xi, tot in zip(x + offsets[i], d["gpu_mem_procs_mb"]):
            if np.isfinite(tot):
                ax_split.text(xi, tot + 15, f"{tot:.0f}", ha="center", va="bottom",
                              fontsize=7, color=TEXT_2, rotation=rot)
    ax_split.set_ylim(top=ax_split.get_ylim()[1] * (1.12 if rot else 1.0))
    ax_split.set_xticks(x)
    ax_split.set_xticklabels([str(n) for n in ntasks])
    ax_split.set_xlabel("Tasks sharing the backbone", fontsize=9, color=TEXT_2)
    ax_split.legend(handles=[Patch(facecolor=TEXT_2, edgecolor=SURFACE, label="PyTorch reserved"),
                             Patch(facecolor=SURFACE, edgecolor=TEXT_2, hatch="////",
                                   label="Context & runtime (NVML − PyTorch)")],
                    frameon=False, fontsize=8, loc="upper left")

    # (f) startup time; clip outliers so the normal range stays readable
    style_axes(ax_ready, "(f) Startup: all processes ready", "s")
    finite = df["ready_s"][np.isfinite(df["ready_s"])]
    cap = float(np.percentile(finite, 75) * 2.0) if len(finite) else 1.0
    for i, mode in enumerate(present):
        d = df[df["mode"] == mode].set_index("ntasks").reindex(ntasks)
        vals = d["ready_s"]
        bars = ax_ready.bar(x + offsets[i], vals.clip(upper=cap), width * 0.9,
                            color=COLORS[mode], edgecolor=SURFACE, linewidth=1)
        for b, v in zip(bars, vals):
            if np.isfinite(v) and v > cap:
                # Inside the clipped bar, on end, so neighbouring clipped bars
                # don't collide.
                ax_ready.text(b.get_x() + b.get_width() / 2, cap * 0.97, f"{v:.0f} s ↑",
                              ha="center", va="top", fontsize=7, color=SURFACE,
                              rotation=90)
    ax_ready.set_ylim(0, cap * 1.15)
    ax_ready.set_xticks(x)
    ax_ready.set_xticklabels([str(n) for n in ntasks])
    ax_ready.set_xlabel("Tasks sharing the backbone", fontsize=9, color=TEXT_2)

    handles, labels = ax_lat.get_legend_handles_labels()
    fig.suptitle(f"Isolation mechanism overhead — {PRETTY.get(backbone, backbone)}, "
                 f"closed loop, 1 outstanding request per task",
                 x=0.01, y=0.99, ha="left", fontsize=11, color=TEXT)
    ncol = len(handles) if len(handles) <= 4 else 3
    fig.legend(handles, labels, loc="upper left", ncol=ncol, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.005, 0.96))
    fig.tight_layout(rect=(0, 0, 1, 0.92 if ncol == len(handles) else 0.89))

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"overview_{backbone}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] wrote {out_dir}/overview_{backbone}.png/.pdf")

    # Text table: overhead vs model_only and per-task context cost.
    print(f"\n{backbone}\n{'mode':<21}{'N':>3}{'p50 ms':>9}{'p50 +%':>8}{'p99 +%':>8}"
          f"{'rps':>8}{'rps %':>8}{'GPU MB':>8}{'+MB/task':>10}")
    for mode in MODES:
        d = df[df["mode"] == mode].set_index("ntasks")
        for n in ntasks:
            if n not in d.index:
                continue
            r, b = d.loc[n], base.loc[n] if n in base.index else None
            p50 = (r.lat_p50_ms / b.lat_p50_ms - 1) * 100 if b is not None else float("nan")
            p99 = (r.lat_p99_ms / b.lat_p99_ms - 1) * 100 if b is not None else float("nan")
            rps = (r.throughput_rps / b.throughput_rps - 1) * 100 if b is not None else float("nan")
            per_task = ((r.gpu_mem_procs_mb - d.loc[ntasks[0]].gpu_mem_procs_mb) / (n - ntasks[0])
                        if n != ntasks[0] else float("nan"))
            print(f"{mode:<21}{n:>3}{r.lat_p50_ms:>9.2f}{p50:>8.1f}{p99:>8.1f}"
                  f"{r.throughput_rps:>8.1f}{rps:>8.1f}{r.gpu_mem_procs_mb:>8.0f}{per_task:>10.1f}")


def plot_summary(frames: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Cross-model view at each backbone's largest task count: p50 and p99
    overhead vs model only, and total GPU memory."""
    backbones = list(frames)
    present = [m for m in MODES if any(m in set(df["mode"]) for df in frames.values())]
    cmp_modes = [m for m in present if m != "model_only"]
    nmax = {b: int(df.loc[df["mode"] == "model_only", "ntasks"].max()) for b, df in frames.items()}

    def at_nmax(b: str, mode: str, col: str) -> float:
        df = frames[b]
        row = df[(df["mode"] == mode) & (df["ntasks"] == nmax[b])]
        return float(row[col].iloc[0]) if len(row) else float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor=SURFACE)
    x = np.arange(len(backbones))
    xt = [f"{PRETTY.get(b, b)}\n(N={nmax[b]})" for b in backbones]

    for ax, col, title in ((axes[0], "lat_p50_ms", "(a) p50 latency overhead vs model only"),
                           (axes[1], "lat_p99_ms", "(b) p99 latency overhead vs model only")):
        style_axes(ax, title, "% over model only")
        width = 0.8 / len(cmp_modes)
        _, offsets = bar_positions(backbones, len(cmp_modes), width)
        for i, mode in enumerate(cmp_modes):
            vals = [(at_nmax(b, mode, col) / at_nmax(b, "model_only", col) - 1) * 100
                    for b in backbones]
            bars = ax.bar(x + offsets[i], vals, width * 0.9, color=COLORS[mode],
                          edgecolor=SURFACE, linewidth=1, label=LABELS[mode])
            for bar, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 0.3, f"{v:.1f}",
                            ha="center", va="bottom", fontsize=6.5, color=TEXT_2, rotation=90)
        ax.axhline(0, color=TEXT_2, linewidth=0.8)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels(xt)

    ax = axes[2]
    style_axes(ax, "(c) GPU memory, all processes", "MB")
    width = 0.8 / len(present)
    _, offsets = bar_positions(backbones, len(present), width)
    for i, mode in enumerate(present):
        vals = [at_nmax(b, mode, "gpu_mem_procs_mb") for b in backbones]
        bars = ax.bar(x + offsets[i], vals, width * 0.9, color=COLORS[mode],
                      edgecolor=SURFACE, linewidth=1)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 20, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=6.5, color=TEXT_2, rotation=90)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(xt)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in present]
    fig.suptitle("Isolation mechanism overhead across backbones (largest task count per backbone)",
                 x=0.01, y=0.99, ha="left", fontsize=11, color=TEXT)
    ncol = len(present) if len(present) <= 4 else 3
    fig.legend(handles, [LABELS[m] for m in present], loc="upper left", ncol=ncol,
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.005, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.87 if ncol == len(present) else 0.82))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"summary_backbones.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] wrote {out_dir}/summary_backbones.png/.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(EXP_DIR / "results" / "summary.csv"))
    parser.add_argument("--backbone", default=None,
                        help="Plot one backbone. Default: every backbone in the CSV, "
                             "plus a cross-backbone summary when there are several.")
    parser.add_argument("--out-dir", default=str(EXP_DIR / "results" / "plots"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID})

    raw = pd.read_csv(args.csv)
    order = [b for b in BACKBONE_ORDER if b in set(raw["backbone"])]
    order += [b for b in dict.fromkeys(raw["backbone"]) if b not in order]
    backbones = [args.backbone] if args.backbone else order
    frames = {b: load(raw, b) for b in backbones}
    for b, df in frames.items():
        plot_overview(df, b, out_dir)
    if len(frames) > 1:
        plot_summary(frames, out_dir)


if __name__ == "__main__":
    main()
