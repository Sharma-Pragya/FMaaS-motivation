"""
Runtime dynamics plots comparing clipper-ha vs fmaas_share.

Plots (in a single figure with subplots):
  1. Latency timeline  — e2e latency vs req_time, per task, per scheduler
  2. CDF               — latency distribution for gestureclass (the new task)
  3. Bar chart         — mean + p99 latency per task per scheduler
  4. Deployment cost   — backbone load time + decoder add time (stacked bar)
"""

import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Paths ────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(__file__)

SCHEDULERS = {
    "FMaaS (share)": os.path.join(BASE, "results/fmaas_share/10"),
    "Clipper-HA":    os.path.join(BASE, "results/clipper-ha/10"),
}

TASK_COLORS = {
    "ecgclass":    "#4C72B0",
    "gestureclass": "#DD8452",
}

SCHED_COLORS = {
    "FMaaS (share)": "#2ca02c",
    "Clipper-HA":    "#d62728",
}

TASK_LABELS = {
    "ecgclass":    "ECG Classification",
    "gestureclass": "Gesture Classification (new task)",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path):
    with open(os.path.join(path, "request_latency_results.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["req_time"]               = float(r["req_time"])
        r["end_to_end_latency(ms)"] = float(r["end_to_end_latency(ms)"])
        r["device_start_time"]      = float(r["device_start_time"])

    # Normalize device_start_time relative to experiment start (first device_start_time)
    t0 = min(r["device_start_time"] for r in rows)
    for r in rows:
        r["exec_time"] = r["device_start_time"] - t0

    return rows


def load_deployment_times(path):
    """
    Return (backbone_ms, decoder_ms) for the runtime-added task.
    backbone_ms: wall time to load backbone (or 0 if hot-add, no new backbone)
    decoder_ms:  wall time to add decoder
    """
    with open(os.path.join(path, "model_deployment_results.json")) as f:
        data = json.load(f)

    backbone_ms = 0.0
    decoder_ms  = 0.0

    for entry in data:
        if isinstance(entry, dict) and "event" in entry:
            summary_str = entry["result"]["logger_summary"]
            # Parse the summary string (it's a Python repr dict)
            summary = eval(summary_str)  # safe: controlled internal data
            if "load_backbone" in summary:
                backbone_ms = summary["load_backbone"]["wall time"]
            # decoder key varies; find the add_decoder key
            for k, v in summary.items():
                if k.startswith("add_decoder"):
                    decoder_ms = v["wall time"]

    return backbone_ms, decoder_ms


def percentile(vals, p):
    return float(np.percentile(vals, p))


# ── Load data ─────────────────────────────────────────────────────────────────

data = {}
deployment_times = {}

for label, path in SCHEDULERS.items():
    rows = load_csv(path)
    data[label] = {
        "ecgclass":     [r for r in rows if r["task"] == "ecgclass"],
        "gestureclass": [r for r in rows if r["task"] == "gestureclass"],
    }
    deployment_times[label] = load_deployment_times(path)

# ── Figure layout ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 12))
fig.suptitle("FMaaS Runtime Dynamics: New Task Addition", fontsize=15, fontweight="bold", y=0.98)

# 2x2 grid; plot 1 spans full top row
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)
ax_timeline = fig.add_subplot(gs[0, :])   # top full-width
ax_cdf      = fig.add_subplot(gs[1, 0])
ax_bar      = fig.add_subplot(gs[1, 1])

# ── Plot 1: Latency timeline ──────────────────────────────────────────────────
# x-axis: exec_time = device_start_time - t0 (when the request actually ran)
# vertical line: req_time of first gestureclass request (when it was scheduled)
# gap between the two = cold-start / deployment delay

WINDOW = 20  # rolling window for smoothing (in requests, within each task)

for sched_label, tasks in data.items():
    ls = "-" if "FMaaS" in sched_label else "--"
    for task, rows in tasks.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: r["exec_time"])
        xs = [r["exec_time"] for r in rows_sorted]
        ys = [r["end_to_end_latency(ms)"] for r in rows_sorted]

        # Rolling mean
        ys_smooth = np.convolve(ys, np.ones(WINDOW) / WINDOW, mode="same")

        color = TASK_COLORS[task]
        alpha = 0.85 if "FMaaS" in sched_label else 0.7
        ax_timeline.plot(
            xs, ys_smooth,
            linestyle=ls,
            color=color,
            alpha=alpha,
            linewidth=1.8,
            label=f"{sched_label} — {TASK_LABELS[task]}",
        )

# Single vertical line for new-task arrival (same req_time=10s for both schedulers)
gest_arrival = min(
    r["req_time"]
    for tasks in data.values()
    for r in tasks["gestureclass"]
)
ax_timeline.axvline(
    gest_arrival,
    color="black",
    linestyle=":",
    linewidth=1.6,
    alpha=0.7,
    label=f"New task requested @ t={gest_arrival:.0f}s",
)
ax_timeline.text(
    gest_arrival -20, 700,
    f"New task\nrequested\n@ t={gest_arrival:.0f}s",
    fontsize=8, va="top", color="black",
)

# Annotate cold-start gap for clipper-HA: arrow from req_time to first exec_time
for sched_label, tasks in data.items():
    gest = tasks["gestureclass"]
    if not gest:
        continue
    first_exec = min(r["exec_time"] for r in gest)
    gap = first_exec - gest_arrival
    if gap > 1.0:  # only annotate if meaningful gap (clipper-HA)
        ax_timeline.annotate(
            "",
            xy=(first_exec, 800),
            xytext=(gest_arrival, 800),
            arrowprops=dict(arrowstyle="<->", color=SCHED_COLORS[sched_label], lw=1.8),
        )
        ax_timeline.text(
            (gest_arrival + first_exec) / 2, 1100,
            f"{sched_label}\ncold-start\n{gap:.2f}s",
            ha="center", fontsize=7.5, color=SCHED_COLORS[sched_label],
        )

ax_timeline.set_xlabel("Actual execution time (s from experiment start)", fontsize=11)
ax_timeline.set_ylabel("End-to-end latency (ms)", fontsize=11)
ax_timeline.set_title(
    "Latency Over Time — x-axis: when request actually executed (rolling mean, window=20)",
    fontsize=11,
)
ax_timeline.set_yscale("log")
ax_timeline.set_ylim(bottom=5)
ax_timeline.grid(True, which="both", alpha=0.3)

# Legend: task (color) + scheduler (linestyle) + arrival line
legend_elements = [
    Line2D([0], [0], color=TASK_COLORS["ecgclass"],     lw=2, label="ECG Classification"),
    Line2D([0], [0], color=TASK_COLORS["gestureclass"], lw=2, label="Gesture Classification (new)"),
    Line2D([0], [0], color="gray", lw=2, ls="-",  label="FMaaS (share)"),
    Line2D([0], [0], color="gray", lw=2, ls="--", label="Clipper-HA"),
]
ax_timeline.legend(handles=legend_elements, fontsize=8.5, loc="upper right", ncol=2)

# ── Plot 2: CDF of gestureclass latency ───────────────────────────────────────

for sched_label, tasks in data.items():
    gest = tasks["gestureclass"]
    if not gest:
        continue
    lats = sorted(r["end_to_end_latency(ms)"] for r in gest)
    cdf  = np.arange(1, len(lats) + 1) / len(lats)
    ax_cdf.plot(
        lats, cdf,
        color=SCHED_COLORS[sched_label],
        linewidth=2,
        label=sched_label,
    )
    # Mark p50 and p99
    for p, marker in [(50, "o"), (99, "^")]:
        val = percentile(lats, p)
        cdf_val = p / 100
        ax_cdf.plot(val, cdf_val, marker=marker, color=SCHED_COLORS[sched_label], markersize=7)
        ax_cdf.annotate(
            f"p{p}={val:.0f}ms",
            xy=(val, cdf_val),
            xytext=(8, -12 if p == 99 else 8),
            textcoords="offset points",
            fontsize=7.5,
            color=SCHED_COLORS[sched_label],
        )

ax_cdf.set_xscale("log")
ax_cdf.set_xlabel("End-to-end latency (ms, log scale)", fontsize=10)
ax_cdf.set_ylabel("CDF", fontsize=10)
ax_cdf.set_title("Latency CDF — Gesture Classification (new task)", fontsize=10.5)
ax_cdf.legend(fontsize=9)
ax_cdf.grid(True, which="both", alpha=0.3)
ax_cdf.set_ylim(0, 1.05)

# ── Plot 3: Bar chart mean + p99 ──────────────────────────────────────────────

tasks_order   = ["ecgclass", "gestureclass"]
sched_labels  = list(SCHEDULERS.keys())
n_tasks       = len(tasks_order)
n_scheds      = len(sched_labels)
x             = np.arange(n_tasks)
width         = 0.35

for i, sched_label in enumerate(sched_labels):
    means = []
    p99s  = []
    for task in tasks_order:
        rows = data[sched_label][task]
        lats = [r["end_to_end_latency(ms)"] for r in rows] if rows else [0]
        means.append(np.mean(lats))
        p99s.append(percentile(lats, 99))

    offset = (i - (n_scheds - 1) / 2) * width
    bars = ax_bar.bar(
        x + offset, means,
        width=width * 0.9,
        color=SCHED_COLORS[sched_label],
        alpha=0.85,
        label=sched_label,
        zorder=3,
    )
    # p99 as error bar above mean
    ax_bar.errorbar(
        x + offset, means,
        yerr=[[0] * n_tasks, [p99 - mean for p99, mean in zip(p99s, means)]],
        fmt="none",
        color="black",
        capsize=4,
        linewidth=1.5,
        zorder=4,
    )
    # Annotate mean
    for bar, mean in zip(bars, means):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 5,
            f"{mean:.0f}",
            ha="center", va="bottom",
            fontsize=7.5, fontweight="bold",
        )

ax_bar.set_yscale("log")
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(["ECG\nClassification", "Gesture\nClassification\n(new task)"], fontsize=9)
ax_bar.set_ylabel("Latency (ms, log scale)", fontsize=10)
ax_bar.set_title("Mean Latency (bar) + p99 (whisker)", fontsize=10.5)
ax_bar.legend(fontsize=9)
ax_bar.grid(True, axis="y", which="both", alpha=0.3, zorder=0)

# ── Save ─────────────────────────────────────────────────────────────────────

out_path = os.path.join(BASE, "runtime_dynamics.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
