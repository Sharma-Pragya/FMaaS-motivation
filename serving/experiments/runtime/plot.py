"""
Runtime "System in Action" — timeline plots (fmaas_share only).

Five timeline panels stacked vertically:
  1. Rolling mean e2e latency per task (log scale)
  2. Rolling mean e2e latency per task (linear scale)
  3. Active backbone per device over time
  4. Deployment event durations (Gantt)
  5. Per-task + total system throughput

Seven-event timeline:
  t=0s    Initial deploy: ecgclass @ 10 req/s
  t=60s   EVENT 1: gestureclass added         → add_decoder (shares device1 momentbase)
  t=120s  EVENT 2: ecgclass +5 req/s (→15)   → ramp step 1
  t=180s  EVENT 3: ecgclass +5 req/s (→20)   → ramp step 2
  t=240s  EVENT 4: ecgclass +5 req/s (→25)   → ramp step 3
  t=300s  EVENT 5: sysbp added               → runtime_add (new backbone, device2)
  t=360s  EVENT 6: diasbp added              → add_decoder (shares device2 backbone)
  t=420s  EVENT 7: ecgclass +15 req/s (→40)  → fit (device2 backbone downsize)

Setup:
  - 2 GPUs, device1 (cuda:0, momentbase) and device2 (cuda:1)
  - Initial task: ecgclass on device1 @ momentbase
  - Device IPs/ports are inferred from data (not hardcoded)
"""

import csv
import json
import os
import ast
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(__file__)
DATA_DIR  = os.path.join(BASE, "results/fmaas_share/10")
CSV_PATH  = os.path.join(DATA_DIR, "request_latency_results.csv")
JSON_PATH = os.path.join(DATA_DIR, "model_deployment_results.json")
OUT_PATH  = os.path.join(BASE, "runtime_dynamics.png")

# ── Experiment timeline ────────────────────────────────────────────────────────
EVENTS = [
    {"t":  60, "label": "EVENT 1\ngestureclass\n(add decoder)",          "color": "#4dac26"},
    {"t": 120, "label": "EVENT 2\necgclass +5\n(ramp step 1)",           "color": "#d01c8b"},
    {"t": 180, "label": "EVENT 3\necgclass +5\n(ramp step 2)",           "color": "#7b2d8b"},
    {"t": 240, "label": "EVENT 4\necgclass +5\n(ramp step 3)",           "color": "#0571b0"},
    {"t": 300, "label": "EVENT 5\nsysbp\n(new backbone)",                "color": "#f1a340"},
    {"t": 360, "label": "EVENT 6\ndiabp\n(add decoder)",                 "color": "#ca0020"},
    {"t": 420, "label": "EVENT 7\necgclass +15\n(fit/dev2 downsize)",    "color": "#5e3c99"},
]
DURATION = 480  # seconds

# ── Visual style ───────────────────────────────────────────────────────────────
TASK_COLORS = {
    "ecgclass":     "#4C72B0",
    "gestureclass": "#C44E52",
    "sysbp":        "#DD8452",
    "diasbp":       "#8172B2",
}
TASK_LABELS = {
    "ecgclass":     "ECG Classification",
    "gestureclass": "Gesture Classification",
    "sysbp":        "Sys. Blood Pressure",
    "diasbp":       "Dias. Blood Pressure",
}
# Device colors/labels are assigned dynamically from data (not hardcoded by IP).
# device1 = first device seen in CSV (earliest exec_time), device2 = second.
DEVICE_COLOR_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
BACKBONE_HATCH = {
    "momentbase":   "",
    "momentlarge":  "---",
    "momentsmall":  "+++",
    "chronosbase":  "///",
    "chronossmall": "xxx",
    "chronostiny":  "...",
}

BIN    = 15   # seconds (rolling throughput window)
WINDOW = 30   # requests (rolling latency window)

# ── Load CSV ───────────────────────────────────────────────────────────────────
rows = []
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        r["req_time"]               = float(r["req_time"])
        r["end_to_end_latency(ms)"] = float(r["end_to_end_latency(ms)"])
        r["device_start_time"]      = float(r["device_start_time"])
        rows.append(r)

# Normalise device_start_time → exec_time (seconds from first request start)
t0 = min(r["device_start_time"] for r in rows)
for r in rows:
    r["exec_time"] = r["device_start_time"] - t0

# ── Load deployment JSON ───────────────────────────────────────────────────────
with open(JSON_PATH) as f:
    deploy_data = json.load(f)

# ── Helpers ────────────────────────────────────────────────────────────────────

def rolling_latency(rows, task, window=WINDOW):
    """Return (exec_times, rolling_mean_latency) sorted by exec_time."""
    task_rows = sorted(
        [r for r in rows if r["task"] == task],
        key=lambda r: r["exec_time"]
    )
    if not task_rows:
        return [], []
    xs = [r["exec_time"] for r in task_rows]
    ys = [r["end_to_end_latency(ms)"] for r in task_rows]
    ys_smooth = np.convolve(ys, np.ones(window) / window, mode="same").tolist()
    return xs, ys_smooth


def rolling_throughput(rows, task, window_s=BIN):
    """Return (exec_times, req/s) using a sliding time-window over exec_time."""
    task_rows = sorted(
        [r for r in rows if r["task"] == task] if task else rows,
        key=lambda r: r["exec_time"]
    )
    if not task_rows:
        return [], []
    half  = window_s / 2
    times = np.array([r["exec_time"] for r in task_rows])
    xs, ys = [], []
    for t in times:
        count = np.sum((times >= t - half) & (times <= t + half))
        xs.append(t)
        ys.append(count / window_s)
    return xs, ys

# ── Infer device1 / device2 from data (no hardcoded IPs) ──────────────────────
# device1 = device serving the initial task (ecgclass); device2 = the other one.
# We determine order by first appearance in sorted rows (by exec_time).
_devices_seen = []
for r in sorted(rows, key=lambda x: x["exec_time"]):
    d = r.get("device", "")
    if d and d not in _devices_seen:
        _devices_seen.append(d)
DEVICE1 = _devices_seen[0] if len(_devices_seen) > 0 else ""
DEVICE2 = _devices_seen[1] if len(_devices_seen) > 1 else ""

DEVICE_COLORS = {d: DEVICE_COLOR_PALETTE[i] for i, d in enumerate(_devices_seen)}
DEVICE_LABELS = {d: f"Device {i+1} — {d}" for i, d in enumerate(_devices_seen)}

# ── Parse deployment JSON into gantt segments ──────────────────────────────────
# runtime_update on device1 → EVENT 1 (gestureclass add_decoder)
# runtime_add              → EVENT 5 (sysbp new backbone on device2)
# runtime_update on device2 → EVENT 6 (diasbp add_decoder)
# runtime_migrate on device2 → EVENT 7 (fit/backbone downsize)

EV_T = [ev["t"] for ev in EVENTS]  # [60, 120, 180, 240, 300, 360, 420]

gantt = []

update_entries_dev1 = [e for e in deploy_data
                       if e.get("event") == "runtime_update"
                       and e.get("device", "") == DEVICE1]
update_entries_dev2 = [e for e in deploy_data
                       if e.get("event") == "runtime_update"
                       and e.get("device", "") == DEVICE2]
add_entries = [e for e in deploy_data if e.get("event") == "runtime_add"]
migrate_entries_dev2 = [e for e in deploy_data
                        if e.get("event") == "runtime_migrate"
                        and e.get("device", "") == DEVICE2]
# Fallback: unmatched migrates go to dev2 bucket (EVENT 7)
migrate_entries_unmatched = [e for e in deploy_data
                              if e.get("event") == "runtime_migrate"
                              and e.get("device", "") not in (DEVICE1, DEVICE2)]
migrate_entries_dev2 = migrate_entries_dev2 + migrate_entries_unmatched

def parse_summary(entry):
    summary_str = entry.get("result", {}).get("logger_summary", "{}")
    try:
        return ast.literal_eval(summary_str)
    except Exception:
        return {}

# EVENT 1 (t=60): gestureclass add_decoder on device1
cur_t = EV_T[0]  # 60
for entry in update_entries_dev1:
    summary = parse_summary(entry)
    dur = sum(v.get("wall time", 0) for v in summary.values()
              if isinstance(v, dict)) / 1000
    if dur > 0:
        gantt.append({
            "t_start":  cur_t,
            "duration": dur,
            "label":    "add_decoder\n(gestureclass)",
            "device":   entry.get("device", ""),
            "color":    EVENTS[0]["color"],
        })
        cur_t += dur

# EVENT 5 (t=300): sysbp runtime_add — load backbone + decoder on device2
cur_t = EV_T[4]  # 300
for entry in add_entries:
    summary = parse_summary(entry)
    backbone_ms = summary.get("load_backbone", {}).get("wall time", 0)
    decoder_ms  = sum(v.get("wall time", 0) for k, v in summary.items()
                      if k.startswith("add_decoder") and isinstance(v, dict))
    total_s = (backbone_ms + decoder_ms) / 1000
    devs   = entry.get("deployments", [])
    device = devs[0]["device"] if devs else entry.get("device", "")
    task   = devs[0]["decoders"][0]["task"] if devs and devs[0].get("decoders") else "new task"
    if total_s > 0:
        gantt.append({
            "t_start":  cur_t,
            "duration": total_s,
            "label":    f"load_backbone\n+ add_decoder\n({task})",
            "device":   device,
            "color":    EVENTS[4]["color"],
        })
        cur_t += total_s

# EVENT 6 (t=360): diasbp add_decoder on device2
cur_t = EV_T[5]  # 360
for entry in update_entries_dev2:
    summary = parse_summary(entry)
    dur = sum(v.get("wall time", 0) for v in summary.values()
              if isinstance(v, dict)) / 1000
    if dur > 0:
        gantt.append({
            "t_start":  cur_t,
            "duration": dur,
            "label":    "add_decoder\n(diasbp)",
            "device":   entry.get("device", ""),
            "color":    EVENTS[5]["color"],
        })
        cur_t += dur

# EVENT 7 (t=420): backbone migrate/fit on device2
cur_t = EV_T[6]  # 420
for entry in migrate_entries_dev2:
    summary = parse_summary(entry)
    backbone_ms = summary.get("load_backbone", {}).get("wall time", 0)
    decoder_ms  = sum(v.get("wall time", 0) for k, v in summary.items()
                      if k.startswith("add_decoder") and isinstance(v, dict))
    total_s = (backbone_ms + decoder_ms) / 1000
    if total_s > 0:
        old_bb = entry.get("old_backbone", "?")
        new_bb = entry.get("new_backbone", "?")
        gantt.append({
            "t_start":  cur_t,
            "duration": total_s,
            "label":    f"swap_backbone\n({old_bb}→{new_bb})",
            "device":   entry.get("device", ""),
            "color":    EVENTS[6]["color"],
        })
        cur_t += total_s

# ── Backbone timeline per device ───────────────────────────────────────────────
device_backbone_timeline = defaultdict(list)
for r in sorted(rows, key=lambda x: x["exec_time"]):
    device_backbone_timeline[r["device"]].append((r["exec_time"], r["backbone"]))

# ── Figure setup ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(16, 22),
                         gridspec_kw={"height_ratios": [3, 3, 1.5, 2, 2]})
fig.suptitle(
    "FMaaS System in Action — Runtime Dynamics (fmaas_share, 2 GPUs)",
    fontsize=13, fontweight="bold", y=0.995
)
fig.subplots_adjust(hspace=0.55, left=0.10, right=0.97, top=0.97, bottom=0.04)

def draw_event_lines(ax):
    for ev in EVENTS:
        ax.axvline(ev["t"], color=ev["color"], linestyle="--", linewidth=1.2, alpha=0.75)
    ax.set_xlim(0, DURATION)

def annotate_events(ax):
    for ev in EVENTS:
        ax.text(ev["t"] + 1.5, 0.97, ev["label"],
                transform=ax.get_xaxis_transform(),
                fontsize=6.5, va="top", color=ev["color"],
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec=ev["color"]))

all_tasks = sorted(TASK_COLORS.keys())

# ── Panel 1: Rolling mean latency (log scale) ──────────────────────────────────
ax1 = axes[0]
draw_event_lines(ax1)
for task in all_tasks:
    xs, ys = rolling_latency(rows, task)
    if not xs:
        continue
    ax1.plot(xs, ys, color=TASK_COLORS[task], linewidth=2.0,
             label=TASK_LABELS[task], alpha=0.9)
annotate_events(ax1)
ax1.set_ylabel("Latency (ms)", fontsize=10)
ax1.set_title(f"End-to-End Latency per Task  [rolling mean, window={WINDOW} reqs, log scale]", fontsize=10)
ax1.set_yscale("log")
ax1.grid(True, which="both", alpha=0.2)
ax1.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)

# ── Panel 2: Rolling mean latency (linear scale) ───────────────────────────────
ax2 = axes[1]
draw_event_lines(ax2)
for task in all_tasks:
    xs, ys = rolling_latency(rows, task)
    if not xs:
        continue
    ax2.plot(xs, ys, color=TASK_COLORS[task], linewidth=2.0,
             label=TASK_LABELS[task], alpha=0.9)
annotate_events(ax2)
ax2.set_ylabel("Latency (ms)", fontsize=10)
ax2.set_title(f"End-to-End Latency per Task  [rolling mean, window={WINDOW} reqs, linear scale]", fontsize=10)
ax2.grid(True, alpha=0.2)
ax2.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
ax2.set_ylim(bottom=0)

# ── Panel 3: Backbone timeline per device ──────────────────────────────────────
ax3 = axes[2]
draw_event_lines(ax3)

devices_ordered = list(DEVICE_COLORS.keys())
for dev in device_backbone_timeline:
    if dev not in devices_ordered:
        devices_ordered.append(dev)
y_positions = {dev: i for i, dev in enumerate(devices_ordered)}
backbone_legend_patches = {}

for dev, timeline in device_backbone_timeline.items():
    y = y_positions[dev]
    segments = []
    cur_bb, cur_t = timeline[0][1], timeline[0][0]
    for t, bb in timeline[1:]:
        if bb != cur_bb:
            segments.append((cur_t, t, cur_bb))
            cur_bb, cur_t = bb, t
    segments.append((cur_t, DURATION, cur_bb))

    for t_start, t_end, bb in segments:
        color = DEVICE_COLORS.get(dev, "#999")
        hatch = BACKBONE_HATCH.get(bb, "")
        ax3.barh(y, t_end - t_start, left=t_start, height=0.5,
                 color=color, alpha=0.7, hatch=hatch, edgecolor="white")
        width = t_end - t_start
        if width > 20:
            ax3.text(t_start + width / 2, y, bb,
                     ha="center", va="center", fontsize=7, fontweight="bold", color="white")
        if bb not in backbone_legend_patches:
            backbone_legend_patches[bb] = mpatches.Patch(
                facecolor="gray", hatch=hatch, edgecolor="white", label=bb, alpha=0.7)

ax3.set_yticks(list(y_positions.values()))
ax3.set_yticklabels([DEVICE_LABELS.get(d, d) for d in y_positions], fontsize=8)
ax3.set_title("Active Backbone per Device over Time", fontsize=10)
ax3.grid(True, axis="x", alpha=0.2)

dev_patches = [mpatches.Patch(color=DEVICE_COLORS.get(d, "#999"),
                               label=DEVICE_LABELS.get(d, d), alpha=0.7)
               for d in y_positions]
ax3.legend(handles=dev_patches + list(backbone_legend_patches.values()),
           fontsize=7, loc="upper right", ncol=2, framealpha=0.9)

# ── Panel 4: Deployment event gantt ───────────────────────────────────────────
ax4 = axes[3]
draw_event_lines(ax4)

devices_in_gantt = []
for d in list(DEVICE_COLORS.keys()) + [g["device"] for g in gantt]:
    if d and d not in devices_in_gantt:
        devices_in_gantt.append(d)
devices_in_gantt = [d for d in devices_in_gantt if any(g["device"] == d for g in gantt)]
gantt_y = {dev: i for i, dev in enumerate(devices_in_gantt)}

for g in gantt:
    dev = g["device"]
    if not dev or dev not in gantt_y:
        continue
    y = gantt_y[dev]
    ax4.barh(y, g["duration"], left=g["t_start"], height=0.45,
             color=g["color"], alpha=0.85, edgecolor="white")
    if g["duration"] > 0.5:
        ax4.text(g["t_start"] + g["duration"] / 2, y,
                 f"{g['duration']:.1f}s",
                 ha="center", va="center", fontsize=7, color="white", fontweight="bold")

ax4.set_yticks(list(gantt_y.values()))
ax4.set_yticklabels([DEVICE_LABELS.get(d, d) for d in devices_in_gantt], fontsize=8)
ax4.set_title("Deployment Event Duration (Gantt)", fontsize=10)
ax4.grid(True, axis="x", alpha=0.2)

ev_patches = [mpatches.Patch(color=ev["color"],
                              label=ev["label"].replace("\n", " "), alpha=0.85)
              for ev in EVENTS if ev["t"] not in (120, 180, 240, 420)]  # skip ramp/fit events (no gantt bar)
ax4.legend(handles=ev_patches, fontsize=7, loc="upper right", framealpha=0.9)

# ── Panel 5: Per-task + total throughput ───────────────────────────────────────
ax5 = axes[4]
draw_event_lines(ax5)

for task in all_tasks:
    xs, ys = rolling_throughput(rows, task)
    if not xs:
        continue
    ax5.plot(xs, ys, color=TASK_COLORS[task], linewidth=2.0,
             label=TASK_LABELS[task], alpha=0.9)

xs_total, ys_total = rolling_throughput(rows, None)
ax5.plot(xs_total, ys_total, color="black", linewidth=2.5,
         linestyle="--", label="Total (all tasks)")

ax5.set_xlabel("Experiment time (s)", fontsize=10)
ax5.set_ylabel("Throughput (req/s)", fontsize=10)
ax5.set_title(f"Served Request Throughput — Per Task & Total System  [rolling {BIN}s window]", fontsize=10)
ax5.grid(True, alpha=0.2)
ax5.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
ax5.set_ylim(bottom=0)

# ── Shared x-axis (hide tick labels on upper panels) ──────────────────────────
for ax in axes[:4]:
    ax.set_xticklabels([])

# ── Save ───────────────────────────────────────────────────────────────────────
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
