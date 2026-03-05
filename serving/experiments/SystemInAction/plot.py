"""
FMaaS System-in-Action — Plots

Three figures:
  1. Mean latency over time, one row per GPU (y=GPU label, x=time)
  2. System throughput over time, one row per GPU
  3. Deployment diagram — backbone boxes with decoder + task children per GPU

Usage:
    cd serving
    python experiments/SystemInAction/plot.py
    python experiments/SystemInAction/plot.py --data experiments/SystemInAction/results/fmaas_share/100
"""

import ast
import csv
import json
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data", default=None,
                    help="Path to result directory containing CSV and JSONs")
args = parser.parse_args()

BASE     = os.path.dirname(__file__)
DATA_DIR = args.data or os.path.join(BASE, "results", "fmaas_share", "150")
CSV_PATH    = os.path.join(DATA_DIR, "request_latency_results.csv")
PLAN_PATH   = os.path.join(DATA_DIR, "deployment_plan.json")
DEPLOY_PATH = os.path.join(DATA_DIR, "model_deployment_results.json")

OUT_LATENCY    = os.path.join(BASE, "plot_latency.png")
OUT_THROUGHPUT = os.path.join(BASE, "plot_throughput.png")
OUT_DEPLOY     = os.path.join(BASE, "plot_deployment.png")

# ── Rolling window config ──────────────────────────────────────────────────────
LATENCY_WINDOW = 20   # requests per task
THROUGHPUT_BIN = 2.0  # seconds

# ── Color palettes ─────────────────────────────────────────────────────────────
TASK_PALETTE = [
    "#4C72B0", "#C44E52", "#55A868", "#DD8452",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]
BACKBONE_COLORS = {
    "momentsmall":  "#2166ac",
    "momentbase":   "#4393c3",
    "momentlarge":  "#92c5de",
    "chronostiny":  "#d6604d",
    "chronossmall": "#f4a582",
    "chronosbase":  "#b2182b",
    "chronoslarge": "#67001f",
}
DECODER_BG   = "#f7f7f7"
TASK_TYPE_COLOR = {
    "classification": "#1b7837",
    "regression":     "#762a83",
    "forecasting":    "#b35806",
}

# ── Load CSV ───────────────────────────────────────────────────────────────────
rows = []
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        r["req_time"]              = float(r["req_time"])
        r["device_start_time"]     = float(r["device_start_time"])
        r["device_end_time"]       = float(r["device_end_time"])
        r["end_to_end_latency(ms)"]= float(r["end_to_end_latency(ms)"])
        rows.append(r)

# Normalise to experiment time (t=0 = first request starts)
t0 = min(r["device_start_time"] for r in rows)
for r in rows:
    r["exec_time"] = r["device_start_time"] - t0

# ── Load deployment plan ───────────────────────────────────────────────────────
with open(PLAN_PATH) as f:
    plan = json.load(f)

# Build device → GPU label mapping from plan
device_info = {}   # device_url -> {"label": "GPU 0 (cuda:0)", "backbone": ..., "decoders": [...], "tasks": {...}}
for site in plan["sites"]:
    for d in site["deployments"]:
        cuda = d.get("cuda", "")
        gpu_idx = cuda.replace("cuda:", "") if cuda else str(len(device_info))
        label = f"GPU {gpu_idx} ({cuda})" if cuda else f"GPU {len(device_info)}"
        device_info[d["device"]] = {
            "label":    label,
            "backbone": d["backbone"],
            "decoders": d["decoders"],
            "tasks":    d.get("tasks", {}),
            "device_name": d.get("device_name", ""),
            "device_type": d.get("device_type", ""),
            "util":     d.get("util", 0),
        }

# Sort devices by GPU index
def _gpu_idx(dev_url):
    info = device_info.get(dev_url, {})
    label = info.get("label", dev_url)
    try:
        return int(label.split("GPU ")[1].split(" ")[0])
    except Exception:
        return 99

devices_ordered = sorted(device_info.keys(), key=_gpu_idx)
n_devices = len(devices_ordered)

# All unique tasks sorted
all_tasks = sorted(set(r["task"] for r in rows))
task_color = {t: TASK_PALETTE[i % len(TASK_PALETTE)] for i, t in enumerate(all_tasks)}

# ── Helpers ────────────────────────────────────────────────────────────────────
def rolling_mean_latency(device_rows, task, window=LATENCY_WINDOW):
    subset = sorted(
        [r for r in device_rows if r["task"] == task],
        key=lambda r: r["exec_time"]
    )
    if not subset:
        return [], []
    xs = [r["exec_time"] for r in subset]
    ys = [r["end_to_end_latency(ms)"] for r in subset]
    smoothed = np.convolve(ys, np.ones(window) / window, mode="same").tolist()
    return xs, smoothed


def sliding_throughput(device_rows, task=None, bin_s=THROUGHPUT_BIN):
    subset = sorted(
        ([r for r in device_rows if r["task"] == task] if task else device_rows),
        key=lambda r: r["exec_time"]
    )
    if not subset:
        return [], []
    half = bin_s / 2
    times = np.array([r["exec_time"] for r in subset])
    xs, ys = [], []
    for t in times:
        count = np.sum((times >= t - half) & (times <= t + half))
        xs.append(t)
        ys.append(count / bin_s)
    return xs, ys


# Group rows by device
rows_by_device = defaultdict(list)
for r in rows:
    rows_by_device[r["device"]].append(r)

# Experiment duration
t_max = max(r["exec_time"] for r in rows)

# ── FIGURE 1: Mean Latency per GPU ────────────────────────────────────────────
fig1, axes1 = plt.subplots(n_devices, 1, figsize=(13, 4 * n_devices),
                            sharex=True, squeeze=False)
fig1.suptitle("End-to-End Latency per GPU  [rolling mean]",
              fontsize=13, fontweight="bold")
fig1.subplots_adjust(hspace=0.35, left=0.12, right=0.97, top=0.94, bottom=0.07)

for row_idx, dev in enumerate(devices_ordered):
    ax = axes1[row_idx][0]
    info = device_info[dev]
    dev_rows = rows_by_device[dev]
    tasks_on_dev = sorted(set(r["task"] for r in dev_rows))

    for task in tasks_on_dev:
        xs, ys = rolling_mean_latency(dev_rows, task)
        if not xs:
            continue
        ax.plot(xs, ys, color=task_color[task], linewidth=2.0,
                label=task, alpha=0.9)

    ax.set_ylabel("Latency (ms)", fontsize=9)
    ax.set_title(f"{info['label']}  —  {info['backbone']}  ({info['device_type']})",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, t_max * 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))

axes1[-1][0].set_xlabel("Experiment time (s)", fontsize=10)

plt.savefig(OUT_LATENCY, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_LATENCY}")
plt.close(fig1)

# ── FIGURE 2: Throughput per GPU ──────────────────────────────────────────────
fig2, axes2 = plt.subplots(n_devices, 1, figsize=(13, 4 * n_devices),
                            sharex=True, squeeze=False)
fig2.suptitle(f"Request Throughput per GPU  [sliding {THROUGHPUT_BIN}s window]",
              fontsize=13, fontweight="bold")
fig2.subplots_adjust(hspace=0.35, left=0.12, right=0.97, top=0.94, bottom=0.07)

for row_idx, dev in enumerate(devices_ordered):
    ax = axes2[row_idx][0]
    info = device_info[dev]
    dev_rows = rows_by_device[dev]
    tasks_on_dev = sorted(set(r["task"] for r in dev_rows))

    # Per-task throughput
    for task in tasks_on_dev:
        xs, ys = sliding_throughput(dev_rows, task)
        if not xs:
            continue
        ax.plot(xs, ys, color=task_color[task], linewidth=1.8,
                label=task, alpha=0.85)

    # Total throughput for this GPU
    xs_tot, ys_tot = sliding_throughput(dev_rows)
    if xs_tot:
        ax.plot(xs_tot, ys_tot, color="black", linewidth=2.2,
                linestyle="--", label="Total", alpha=0.9)

    ax.set_ylabel("Throughput (req/s)", fontsize=9)
    ax.set_title(f"{info['label']}  —  {info['backbone']}  ({info['device_type']})",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, t_max * 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9)

axes2[-1][0].set_xlabel("Experiment time (s)", fontsize=10)

plt.savefig(OUT_THROUGHPUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_THROUGHPUT}")
plt.close(fig2)

# ── FIGURE 3: Deployment Diagram ──────────────────────────────────────────────
#
# Layout (per GPU row):
#
#   ┌─────────────────────────────────────────────────────────────┐
#   │  BACKBONE (large box, colored)                              │
#   │  ┌──────────────────┐  ┌──────────────────┐  ...           │
#   │  │  decoder+task    │  │  decoder+task    │                │
#   │  │  (small box)     │  │  (small box)     │                │
#   │  └──────────────────┘  └──────────────────┘                │
#   └─────────────────────────────────────────────────────────────┘

# Load deployment timing from model_deployment_results.json
deploy_timing = {}  # backbone -> {load_ms, decoders: {task: ms}}
if os.path.exists(DEPLOY_PATH):
    with open(DEPLOY_PATH) as f:
        deploy_results = json.load(f)
    for entry in deploy_results:
        status = entry.get("status", "")
        summary_str = entry.get("logger_summary", "{}")
        try:
            summary = ast.literal_eval(summary_str)
        except Exception:
            summary = {}
        # Extract backbone name from status string "loaded_<backbone>"
        bb = status.replace("loaded_", "") if status.startswith("loaded_") else status
        load_ms = summary.get("load_backbone", {}).get("wall time", 0)
        dec_times = {}
        for k, v in summary.items():
            if k.startswith("add_decoder_") and isinstance(v, dict):
                # key format: add_decoder_{task}_{backbone}_{arch}
                parts = k.split("_")
                task_name = parts[2] if len(parts) > 2 else k
                dec_times[task_name] = v.get("wall time", 0)
        deploy_timing[bb] = {"load_ms": load_ms, "decoders": dec_times}

# Figure dimensions — one tall row per GPU
ROW_H      = 3.2    # figure height per GPU row (inches)
FIG_W      = 14.0
fig3_h     = max(4.0, ROW_H * n_devices + 1.5)
fig3, axes3 = plt.subplots(n_devices, 1, figsize=(FIG_W, fig3_h),
                            squeeze=False)
fig3.suptitle("FMaaS Deployment Layout — Backbones, Decoders & Tasks",
              fontsize=13, fontweight="bold")
fig3.subplots_adjust(hspace=0.6, left=0.02, right=0.98, top=0.93, bottom=0.04)

for row_idx, dev in enumerate(devices_ordered):
    ax = axes3[row_idx][0]
    info = device_info[dev]
    bb   = info["backbone"]
    decoders = info["decoders"]   # list of {task, type, path}

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n_dec = len(decoders)

    # ── Backbone outer box ───────────────────────────────────────────────
    BB_PAD_X  = 0.03
    BB_PAD_Y  = 0.08
    bb_x      = BB_PAD_X
    bb_y      = BB_PAD_Y
    bb_w      = 1 - 2 * BB_PAD_X
    bb_h      = 1 - 2 * BB_PAD_Y
    bb_color  = BACKBONE_COLORS.get(bb, "#4393c3")

    backbone_box = FancyBboxPatch(
        (bb_x, bb_y), bb_w, bb_h,
        boxstyle="round,pad=0.01",
        facecolor=bb_color, alpha=0.18,
        edgecolor=bb_color, linewidth=2.5,
        transform=ax.transAxes, clip_on=False
    )
    ax.add_patch(backbone_box)

    # Backbone label (top left of box)
    load_s = deploy_timing.get(bb, {}).get("load_ms", 0) / 1000
    bb_label = f"Backbone: {bb}"
    if load_s > 0:
        bb_label += f"\nload: {load_s:.1f}s"
    ax.text(bb_x + 0.01, bb_y + bb_h - 0.01, bb_label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11, fontweight="bold", color=bb_color)

    # GPU label (top right)
    gpu_info_str = f"{info['label']}\n{info['device_type']}\nutil: {info['util']:.1%}"
    ax.text(bb_x + bb_w - 0.01, bb_y + bb_h - 0.01, gpu_info_str,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8, color="#333333")

    # ── Decoder/task boxes inside backbone ───────────────────────────────
    if n_dec == 0:
        ax.text(0.5, 0.5, "(no decoders)", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#888")
    else:
        # Layout decoders in a single row inside the backbone box
        DEC_MARGIN_X = 0.025   # gap from backbone edge
        DEC_MARGIN_Y = 0.08    # space for backbone label at top
        DEC_GAP      = 0.012
        DEC_TOP_Y    = bb_y + bb_h - DEC_MARGIN_Y - 0.05
        DEC_H        = bb_h - DEC_MARGIN_Y - 0.06  # height of each decoder box
        DEC_BOTTOM_Y = bb_y + BB_PAD_Y * 0.3

        total_gap  = DEC_GAP * (n_dec - 1)
        avail_w    = bb_w - 2 * DEC_MARGIN_X
        dec_w      = (avail_w - total_gap) / n_dec
        dec_h      = DEC_TOP_Y - DEC_BOTTOM_Y

        for i, dec in enumerate(decoders):
            task_name = dec["task"]
            task_type = dec["type"]
            dec_path  = dec.get("path", "")

            dec_x = bb_x + DEC_MARGIN_X + i * (dec_w + DEC_GAP)
            dec_y = DEC_BOTTOM_Y
            dec_color = TASK_TYPE_COLOR.get(task_type, "#555555")

            # Decoder outer box
            dec_box = FancyBboxPatch(
                (dec_x, dec_y), dec_w, dec_h,
                boxstyle="round,pad=0.005",
                facecolor=DECODER_BG, alpha=0.95,
                edgecolor=dec_color, linewidth=1.8,
                transform=ax.transAxes, clip_on=False
            )
            ax.add_patch(dec_box)

            # Connector line from backbone top to decoder box
            mid_x = dec_x + dec_w / 2
            ax.annotate("",
                xy=(mid_x, dec_y + dec_h),
                xytext=(mid_x, bb_y + bb_h - DEC_MARGIN_Y + 0.01),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color=bb_color,
                                lw=1.5, alpha=0.6))

            # Decoder header (colored bar at top of box)
            hdr_h = 0.12
            hdr_box = FancyBboxPatch(
                (dec_x, dec_y + dec_h - hdr_h), dec_w, hdr_h,
                boxstyle="round,pad=0.003",
                facecolor=dec_color, alpha=0.85,
                edgecolor=dec_color, linewidth=0,
                transform=ax.transAxes, clip_on=False
            )
            ax.add_patch(hdr_box)

            # Decoder label in header
            ax.text(dec_x + dec_w / 2, dec_y + dec_h - hdr_h / 2,
                    "decoder",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color="white")

            # Task name (large, center)
            ax.text(dec_x + dec_w / 2, dec_y + dec_h * 0.48,
                    task_name,
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#222222")

            # Task type badge
            type_badge_y = dec_y + dec_h * 0.22
            ax.text(dec_x + dec_w / 2, type_badge_y,
                    f"[{task_type}]",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=7, color=dec_color,
                    style="italic")

            # Decoder load time (if available)
            dec_ms = deploy_timing.get(bb, {}).get("decoders", {}).get(task_name, 0)
            if dec_ms > 0:
                ax.text(dec_x + dec_w / 2, dec_y + 0.04,
                        f"{dec_ms:.0f}ms",
                        transform=ax.transAxes,
                        ha="center", va="bottom",
                        fontsize=6.5, color="#666666")

# ── Legend for task types ──────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=c, label=t.capitalize(), alpha=0.8)
    for t, c in TASK_TYPE_COLOR.items()
]
bb_patches = [
    mpatches.Patch(facecolor=BACKBONE_COLORS.get(b, "#999"), label=b, alpha=0.5)
    for b in sorted(set(device_info[d]["backbone"] for d in devices_ordered))
]
fig3.legend(
    handles=legend_patches + bb_patches,
    title="Task type / Backbone",
    loc="lower center", ncol=6,
    fontsize=8, framealpha=0.9,
    bbox_to_anchor=(0.5, 0.0)
)

plt.savefig(OUT_DEPLOY, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_DEPLOY}")
plt.close(fig3)

print("Done.")
