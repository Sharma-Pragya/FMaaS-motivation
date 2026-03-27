"""
FMaaS System-in-Action — Plots

Figures (single-method mode):
  1. Mean latency per second, one row per GPU (y=GPU label, x=time)
  2. Throughput per second, one row per GPU
  3. Deployment diagram — backbone boxes with decoder + task children per GPU
  4. Trace workload over req_time (per-task + total req/s)

Comparison mode (--compare):
  5. One subplot per GPU: mean latency vs time, one line per method

Usage:
    cd serving
    python experiments/SystemInAction/plot.py
    python experiments/SystemInAction/plot.py --data experiments/SystemInAction/results/fmaas_share/150
    python experiments/SystemInAction/plot.py --compare experiments/SystemInAction/results
    python experiments/SystemInAction/plot.py --compare experiments/SystemInAction/results --run 150
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
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data", default=None,
                    help="Path to result directory containing CSV and JSONs")
parser.add_argument("--compare", default="experiments/SystemInAction/results",
                    help="Path to results root dir; generates latency comparison across methods")
parser.add_argument("--run", default="150",
                    help="Run subdirectory name used with --compare (default: 150)")
args = parser.parse_args()

BASE     = os.path.dirname(__file__)
DATA_DIR = args.data or os.path.join(BASE, "results", "fmaas_share", "150")
CSV_PATH    = os.path.join(DATA_DIR, "request_latency_results.csv")
PLAN_PATH   = os.path.join(DATA_DIR, "deployment_plan.json")
DEPLOY_PATH = os.path.join(DATA_DIR, "model_deployment_results.json")

# Plots go into <DATA_DIR>/plots/
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

OUT_LATENCY    = os.path.join(PLOTS_DIR, "plot_latency.png")
OUT_THROUGHPUT = os.path.join(PLOTS_DIR, "plot_throughput.png")
OUT_DEPLOY     = os.path.join(PLOTS_DIR, "plot_deployment.png")
OUT_WORKLOAD   = os.path.join(PLOTS_DIR, "plot_workload_trace.png")

# ── Aggregation config ─────────────────────────────────────────────────────────
TIME_BIN = 1.0        # seconds, used for latency/throughput plots
TRACE_BIN      = 1.0  # seconds

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
        "font.size":          7,
        "axes.titlesize":     7.5,
        "axes.labelsize":     7,
        "xtick.labelsize":    6.5,
        "ytick.labelsize":    6.5,
        "legend.fontsize":    6.5,
        "lines.linewidth":    1.2,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.facecolor":  "white",
        "savefig.bbox":       "tight",
    })


def _set_clean_ticks(ax: plt.Axes, xdata_max: float, ydata_max: float, n_y: int = 4):
    def _ticks_and_limit(data_max: float, n: int = 5):
        step_raw = data_max / n if data_max > 0 else 1.0
        magnitude = 10 ** np.floor(np.log10(max(step_raw, 1e-9)))
        nice = [1, 2, 2.5, 5, 10]
        step = magnitude * min(nice, key=lambda s: abs(s - step_raw / magnitude))
        nice_limit = np.ceil(max(data_max, step) / step) * step
        ticks = np.round(np.arange(0, nice_limit + step * 0.01, step), 10)
        return ticks, float(nice_limit)

    xt, _      = _ticks_and_limit(xdata_max, n=5)
    yt, ylim_nice = _ticks_and_limit(ydata_max, n=n_y)
    # Use the actual data max for xlim (no over-extension from nice rounding)
    ax.set_xlim(0, xdata_max)
    ax.set_ylim(0, ylim_nice)
    ax.set_xticks(xt[xt <= xdata_max])
    ax.set_yticks(yt)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    return xdata_max, ylim_nice


def save_figure(fig: plt.Figure, out_path: str) -> None:
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    base, _ = os.path.splitext(out_path)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    print(f"Saved → {out_path}")


def _bin_rate(times: np.ndarray, max_time: float):
    n_bins = int(np.ceil(max_time))
    counts = np.zeros(n_bins, dtype=float)
    for t in times:
        idx = int(t)
        if 0 <= idx < n_bins:
            counts[idx] += 1.0
    centers = np.arange(n_bins) + 0.5
    return centers, counts


def _bin_latency(times: np.ndarray, lats: np.ndarray, max_time: float):
    n_bins = int(np.ceil(max_time))
    sums = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=float)
    for t, l in zip(times, lats):
        idx = int(t)
        if 0 <= idx < n_bins:
            sums[idx] += l
            counts[idx] += 1.0
    means = np.full(n_bins, np.nan, dtype=float)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero]
    centers = np.arange(n_bins) + 0.5
    return centers, means


apply_paper_style()

# ── Comparison plots ───────────────────────────────────────────────────────────
if args.compare:
    COMPARE_DIR = args.compare
    RUN = args.run

    # Discover methods: subdirs of COMPARE_DIR that contain <RUN>/request_latency_results.csv
    methods = sorted([
        m for m in os.listdir(COMPARE_DIR)
        if os.path.isfile(os.path.join(COMPARE_DIR, m, RUN, "request_latency_results.csv"))
    ])
    if not methods:
        print(f"No method dirs with run '{RUN}' found under {COMPARE_DIR}")
    else:
        METHOD_COLORS = [
            "#4C72B0", "#C44E52", "#55A868", "#DD8452",
            "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
        ]

        def _load_method(results_root, method, run):
            """Load rows and device→GPU-label map for one method/run."""
            run_dir   = os.path.join(results_root, method, run)
            csv_path  = os.path.join(run_dir, "request_latency_results.csv")
            plan_path = os.path.join(run_dir, "deployment_plan.json")
            rows = []
            with open(csv_path) as f:
                for r in csv.DictReader(f):
                    r["site_manager_send_time"] = float(r["site_manager_send_time"])
                    r["client_receive_time"]    = float(r["client_receive_time"])
                    r["end_to_end_latency(ms)"] = float(r["end_to_end_latency(ms)"])
                    rows.append(r)
            t0 = min(r["site_manager_send_time"] for r in rows)
            for r in rows:
                r["send_time"]       = r["site_manager_send_time"] - t0
                r["completion_time"] = r["client_receive_time"] - t0
            dev_labels = {}
            if os.path.exists(plan_path):
                with open(plan_path) as pf:
                    plan = json.load(pf)
                for site in plan["sites"]:
                    for d in site["deployments"]:
                        cuda = d.get("cuda", "")
                        gpu_idx = cuda.replace("cuda:", "") if cuda else str(len(dev_labels))
                        dev_labels[d["device"]] = (
                            f"GPU {gpu_idx} ({cuda})" if cuda else f"GPU {len(dev_labels)}"
                        )
            return rows, dev_labels

        # Load all methods
        method_data = {}   # method -> (rows, dev_labels)
        for m in methods:
            method_data[m] = _load_method(COMPARE_DIR, m, RUN)

        cmp_plots_dir = os.path.join(COMPARE_DIR, "plots")
        os.makedirs(cmp_plots_dir, exist_ok=True)

        # ── Comparison Figure A: System Mean Latency ──────────────────────────
        fig_lat, ax_lat = plt.subplots(1, 1, figsize=(3.3, 1.6))

        all_ys_lat, t_max_lat = [], 0.0
        for m_idx, m in enumerate(methods):
            rows_m, _ = method_data[m]
            times = np.array([r["send_time"] for r in rows_m])
            lats  = np.array([r["end_to_end_latency(ms)"] for r in rows_m])
            t_max_m = float(times.max())
            t_max_lat = max(t_max_lat, t_max_m)
            xs, ys = _bin_latency(times, lats, t_max_m)
            color = METHOD_COLORS[m_idx % len(METHOD_COLORS)]
            ax_lat.plot(xs, ys, color=color, linewidth=1.0, label=m)
            all_ys_lat.extend([v for v in ys if np.isfinite(v)])

        ax_lat.set_xlabel("Time (s)")
        ax_lat.set_ylabel("Mean Latency (ms)")
        ymax = max(all_ys_lat) if all_ys_lat else 1.0
        _set_clean_ticks(ax_lat, t_max_lat, ymax, n_y=4)
        ax_lat.grid(axis="both", zorder=0)
        ax_lat.set_axisbelow(True)
        handles_lat, labels_lat = ax_lat.get_legend_handles_labels()
        fig_lat.legend(handles_lat, labels_lat, loc="upper center",
                       bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
                       handlelength=1.2, handletextpad=0.3, columnspacing=0.8)
        fig_lat.tight_layout(rect=(0, 0, 1, 0.97))

        out_lat = os.path.join(cmp_plots_dir, f"plot_latency_comparison_{RUN}.png")
        save_figure(fig_lat, out_lat)
        plt.close(fig_lat)

        # ── Comparison Figure B: System Throughput ────────────────────────────
        fig_thr, ax_thr = plt.subplots(1, 1, figsize=(3.3, 1.6))

        all_ys_thr, t_max_thr = [], 0.0
        for m_idx, m in enumerate(methods):
            rows_m, _ = method_data[m]
            completion_times = np.array([r["completion_time"] for r in rows_m])
            t_max_m = float(completion_times.max())
            t_max_thr = max(t_max_thr, t_max_m)
            xs, ys = _bin_rate(completion_times, t_max_m)
            color = METHOD_COLORS[m_idx % len(METHOD_COLORS)]
            ax_thr.plot(xs, ys, color=color, linewidth=1.0, label=m)
            all_ys_thr.extend(list(ys))

        ax_thr.set_xlabel("Time (s)")
        ax_thr.set_ylabel("Throughput (req/s)")
        ymax = max(all_ys_thr) if all_ys_thr else 1.0
        _set_clean_ticks(ax_thr, t_max_thr, ymax, n_y=4)
        ax_thr.grid(axis="both", zorder=0)
        ax_thr.set_axisbelow(True)
        handles_thr, labels_thr = ax_thr.get_legend_handles_labels()
        fig_thr.legend(handles_thr, labels_thr, loc="upper center",
                       bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
                       handlelength=1.2, handletextpad=0.3, columnspacing=0.8)
        fig_thr.tight_layout(rect=(0, 0, 1, 0.97))

        out_thr = os.path.join(cmp_plots_dir, f"plot_throughput_comparison_{RUN}.png")
        save_figure(fig_thr, out_thr)
        plt.close(fig_thr)


def _plot_single_method(data_dir):
    csv_path    = os.path.join(data_dir, "request_latency_results.csv")
    plan_path   = os.path.join(data_dir, "deployment_plan.json")
    deploy_path = os.path.join(data_dir, "model_deployment_results.json")
    plots_dir   = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    out_latency    = os.path.join(plots_dir, "plot_latency.png")
    out_throughput = os.path.join(plots_dir, "plot_throughput.png")
    out_deploy     = os.path.join(plots_dir, "plot_deployment.png")
    out_workload   = os.path.join(plots_dir, "plot_workload_trace.png")

    # ── Load CSV ──────────────────────────────────────────────────────────
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            r["req_time"]               = float(r["req_time"])
            r["site_manager_send_time"] = float(r["site_manager_send_time"])
            r["device_start_time"]      = float(r["device_start_time"])
            r["device_end_time"]        = float(r["device_end_time"])
            r["client_receive_time"]    = float(r["client_receive_time"])
            r["end_to_end_latency(ms)"] = float(r["end_to_end_latency(ms)"])
            rows.append(r)

    t0 = min(r["site_manager_send_time"] for r in rows)
    for r in rows:
        r["send_time"]       = r["site_manager_send_time"] - t0
        r["completion_time"] = r["client_receive_time"] - t0

    # ── Load deployment plan ──────────────────────────────────────────────
    with open(plan_path) as f:
        plan = json.load(f)

    device_info = {}
    for site in plan["sites"]:
        for d in site["deployments"]:
            cuda = d.get("cuda", "")
            device_name = d.get("device_name", "")
            gpu_idx = cuda.replace("cuda:", "") if cuda else str(len(device_info))
            label = f"GPU {gpu_idx} ({cuda})" if cuda else f"GPU {len(device_info)}"
            # Key by (device_name, backbone) so multiple deployments on the same
            # physical device (e.g. Clipper) don't overwrite each other.
            key = (device_name, d["backbone"])
            device_info[key] = {
                "label":       label,
                "backbone":    d["backbone"],
                "decoders":    d["decoders"],
                "tasks":       d.get("tasks", {}),
                "device_name": device_name,
                "device_type": d.get("device_type", ""),
                "util":        d.get("util", 0),
            }

    def _sort_key(dev_key):
        # Sort by device_name first (device1 < device2 < ... < device4),
        # then by backbone to keep deployments on the same physical device together.
        dname = device_info.get(dev_key, {}).get("device_name", str(dev_key))
        bb    = device_info.get(dev_key, {}).get("backbone", "")
        import re
        m = re.search(r'(\d+)$', dname)
        return (int(m.group(1)) if m else 999, dname, bb)

    devices_ordered = sorted(device_info.keys(), key=_sort_key)
    n_devices = len(devices_ordered)

    all_tasks  = sorted(set(r["task"] for r in rows))
    task_color = {t: TASK_PALETTE[i % len(TASK_PALETTE)] for i, t in enumerate(all_tasks)}

    def per_second_avg_latency(device_rows, task=None, max_time=None):
        subset = [r for r in device_rows if (r["task"] == task if task else True)]
        if not subset:
            return [], []
        times = np.array([r["send_time"] for r in subset])
        lats  = np.array([r["end_to_end_latency(ms)"] for r in subset])
        end   = max_time if max_time is not None else float(times.max())
        return _bin_latency(times, lats, end)

    def per_second_throughput(device_rows, task=None, max_time=None):
        subset = [r for r in device_rows if (r["task"] == task if task else True)]
        if not subset:
            return [], []
        ct  = np.array([r["completion_time"] for r in subset])
        end = max_time if max_time is not None else float(ct.max())
        return _bin_rate(ct, end)

    def binned_workload(trace_rows, task=None, bin_s=TRACE_BIN):
        subset = [r for r in trace_rows if (r["task"] == task if task else True)]
        if not subset:
            return [], []
        times    = np.array([r["req_time"] for r in subset], dtype=float)
        max_time = float(times.max()) if len(times) else 0.0
        if max_time <= 0:
            return [], []
        centers, counts = _bin_rate(times, max_time)
        return centers, counts / max(bin_s, 1e-9)

    # Build a lookup: (device_url, task) -> device_key=(device_name, backbone)
    # so CSV rows (which have device URL + task) can be grouped by deployment key.
    url_task_to_key = {}
    for site in plan["sites"]:
        for d in site["deployments"]:
            dname = d.get("device_name", "")
            bb = d["backbone"]
            key = (dname, bb)
            for dec in d.get("decoders", []):
                url_task_to_key[(d["device"], dec["task"])] = key

    rows_by_device = defaultdict(list)
    for r in rows:
        dev_key = url_task_to_key.get((r["device"], r["task"]), (r["device"], ""))
        rows_by_device[dev_key].append(r)

    t_max = max(r["completion_time"] for r in rows)

    # ── FIGURE 1: Mean Latency per GPU (one panel per device, in a row) ──
    panel_w = 1.65   # inches per panel
    fig1, axes1 = plt.subplots(1, n_devices, figsize=(panel_w * n_devices, 1.6),
                                sharey=False, sharex=True, squeeze=False)
    fig1.subplots_adjust(wspace=0.12)

    for col_idx, dev in enumerate(devices_ordered):
        ax   = axes1[0][col_idx]
        dev_rows     = rows_by_device[dev]
        tasks_on_dev = sorted(set(r["task"] for r in dev_rows))
        all_ys = []
        for task in tasks_on_dev:
            xs, ys = per_second_avg_latency(dev_rows, task, max_time=t_max)
            if not len(xs):
                continue
            ax.plot(xs, ys, color=task_color[task], linewidth=1.0, label=task)
            all_ys.extend([v for v in ys if np.isfinite(v)])
        ymax = max(all_ys) if all_ys else 1.0
        _set_clean_ticks(ax, t_max, ymax, n_y=4)
        ax.grid(axis="both", zorder=0)
        ax.set_axisbelow(True)
        dname, bb = dev
        ax.set_title(f"{dname} ({bb})", pad=2)
        ax.set_xlabel("Time (s)")
        if col_idx == 0:
            ax.set_ylabel("Mean Latency (ms)")
        else:
            ax.tick_params(axis="y", left=False)

    handles1, labels1 = axes1[0][0].get_legend_handles_labels()
    fig1.legend(handles1, labels1, loc="upper center",
                bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
                handlelength=1.2, handletextpad=0.3, columnspacing=0.8)
    fig1.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig1, out_latency)
    plt.close(fig1)

    # ── FIGURE 2: Throughput per GPU (one panel per device, in a row) ────
    fig2, axes2 = plt.subplots(1, n_devices, figsize=(panel_w * n_devices, 1.6),
                                sharey=False, sharex=True, squeeze=False)
    fig2.subplots_adjust(wspace=0.12)

    for col_idx, dev in enumerate(devices_ordered):
        ax   = axes2[0][col_idx]
        dev_rows     = rows_by_device[dev]
        tasks_on_dev = sorted(set(r["task"] for r in dev_rows))
        all_ys = []
        for task in tasks_on_dev:
            xs, ys = per_second_throughput(dev_rows, task, max_time=t_max)
            if not len(xs):
                continue
            ax.plot(xs, ys, color=task_color[task], linewidth=1.0, label=task)
            all_ys.extend(list(ys))
        xs_tot, ys_tot = per_second_throughput(dev_rows, max_time=t_max)
        if len(xs_tot):
            ax.plot(xs_tot, ys_tot, color="black", linewidth=1.2,
                    linestyle="--", label="Total")
            all_ys.extend(list(ys_tot))
        ymax = max(all_ys) if all_ys else 1.0
        _set_clean_ticks(ax, t_max, ymax, n_y=4)
        ax.grid(axis="both", zorder=0)
        ax.set_axisbelow(True)
        dname, bb = dev
        ax.set_title(f"{dname} ({bb})", pad=2)
        ax.set_xlabel("Time (s)")
        if col_idx == 0:
            ax.set_ylabel("Throughput (req/s)")
        else:
            ax.tick_params(axis="y", left=False)

    handles2, labels2 = axes2[0][0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc="upper center",
                bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
                handlelength=1.2, handletextpad=0.3, columnspacing=0.8)
    fig2.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig2, out_throughput)
    plt.close(fig2)

    # ── FIGURE 3: Deployment Diagram (paper style) ───────────────────────
    # Group deployments by physical GPU (cuda label).
    # One column per GPU; within each GPU column, deployments are sub-columns
    # side by side. Each sub-column: task boxes on top, backbone bar below.
    # Bottom of figure: one GPU bar spanning the full GPU column width.
    #
    #   GPU 0 (cuda:0)              GPU 1 (cuda:1)
    #   ┌────────┐ ┌────────┐       ┌────────┐ ┌────────┐
    #   │ Task A │ │ Task B │       │ Task C │ │ Task D │
    #   ├────────┤ ├────────┤       ├────────┤ ├────────┤
    #   │  BB-1  │ │  BB-2  │       │  BB-1  │ │  BB-3  │
    #   └──────────────────────┐    └────────────────────┐
    #          GPU 0            │           GPU 1          │
    #   └──────────────────────┘    └────────────────────┘

    TASK_PASTELS = [
        "#ffffcc", "#d4f1c0", "#ffd6e0", "#d6eaff",
        "#ffe8cc", "#e8d5f5", "#ccf5f1", "#fce4d6",
        "#dce8d0", "#f5e6cc",
    ]
    all_tasks_deploy = sorted(set(
        dec["task"] for dev in devices_ordered for dec in device_info[dev]["decoders"]
    ))
    task_pastel = {t: TASK_PASTELS[i % len(TASK_PASTELS)] for i, t in enumerate(all_tasks_deploy)}

    GPU_COLOR      = "#56c4d8"
    BACKBONE_COLOR = "#f0a830"

    def _wrap_task_name(name: str) -> str:
        """Split a task name into two lines at a natural break (digit/letter boundary or midpoint)."""
        import re
        # Try split at first transition from letters to digits or digit-boundary word
        m = re.search(r'(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])', name)
        if m:
            i = m.start()
            return name[:i] + "\n" + name[i:]
        # Try common suffixes
        for suffix in ("fore", "class", "reg", "detect", "seg"):
            if name.endswith(suffix) and len(name) > len(suffix):
                return name[:-len(suffix)] + "\n" + suffix
        # Fallback: split near midpoint at a natural char
        mid = len(name) // 2
        return name[:mid] + "\n" + name[mid:]

    # Group by device_name, preserving order of first appearance
    gpu_groups = {}   # device_name -> [dev_key, ...]
    for dev in devices_ordered:
        dname = device_info[dev]["device_name"]
        gpu_groups.setdefault(dname, []).append(dev)

    gpu_labels  = list(gpu_groups.keys())
    n_gpus      = len(gpu_labels)
    # total number of sub-columns (deployments) across all GPUs
    total_deps  = sum(len(devs) for devs in gpu_groups.values())

    # Heights in inches — keep very compact for paper
    # Tasks are side by side (not stacked), so height = 1 task row regardless of task count
    TASK_H_IN = 0.28    # one task row height (fits 2-line label)
    BB_H_IN   = 0.18    # backbone bar
    GPU_H_IN  = 0.15    # GPU bar at bottom
    PAD_IN    = 0.05    # between rows
    fig_h = GPU_H_IN + PAD_IN + BB_H_IN + PAD_IN + TASK_H_IN + 0.10
    fig_w = max(2.0, 0.45 * total_deps + 0.10 * n_gpus)

    fig3, ax3 = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    fig3.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis("off")

    # Convert inch heights to axes fractions
    def _frac(inches): return inches / fig_h

    GPU_H  = _frac(GPU_H_IN)
    BB_H   = _frac(BB_H_IN)
    TASK_H = _frac(TASK_H_IN)
    PAD    = _frac(PAD_IN)
    BOTTOM = _frac(0.03)

    # Sub-column widths: equal share, with small gaps between GPUs
    GPU_GAP = 0.020   # gap between GPU groups
    SUB_GAP = 0.006   # gap between sub-columns within a GPU
    # Account for all gaps in total width
    total_gap_w = GPU_GAP * (n_gpus - 1) + SUB_GAP * (total_deps - n_gpus)
    sub_w = (1.0 - total_gap_w) / total_deps   # each deployment gets equal width

    x_cursor = 0.0
    for gi, gpu_label in enumerate(gpu_labels):
        devs    = gpu_groups[gpu_label]
        n_deps  = len(devs)
        gpu_col_w = n_deps * sub_w + SUB_GAP * (n_deps - 1)
        # Use device_name directly (e.g. "device1", "device4")
        simple_label = gpu_label

        # GPU bar spanning entire GPU group (bottom)
        ax3.add_patch(FancyBboxPatch(
            (x_cursor, BOTTOM), gpu_col_w, GPU_H, boxstyle="round,pad=0.006",
            facecolor=GPU_COLOR, edgecolor="#2a9ab5", linewidth=0.6,
            transform=ax3.transAxes, clip_on=False))
        ax3.text(x_cursor + gpu_col_w / 2, BOTTOM + GPU_H / 2, simple_label,
                 transform=ax3.transAxes, ha="center", va="center",
                 fontsize=5.5, fontweight="bold", color="white")

        # Sub-columns for each deployment on this GPU
        sub_x = x_cursor
        for dev in devs:
            info     = device_info[dev]
            bb       = info["backbone"]
            decoders = info["decoders"]

            # Backbone bar
            bb_y = BOTTOM + GPU_H + PAD
            ax3.add_patch(FancyBboxPatch(
                (sub_x, bb_y), sub_w, BB_H, boxstyle="round,pad=0.006",
                facecolor=BACKBONE_COLOR, edgecolor="#c07800", linewidth=0.6,
                transform=ax3.transAxes, clip_on=False))
            ax3.text(sub_x + sub_w / 2, bb_y + BB_H / 2, bb,
                     transform=ax3.transAxes, ha="center", va="center",
                     fontsize=4.5, fontweight="bold", color="white")

            # Task boxes side by side above backbone
            task_y   = bb_y + BB_H + PAD
            n_dec    = len(decoders)
            t_gap    = 0.003
            t_w      = (sub_w - t_gap * (n_dec - 1)) / n_dec if n_dec > 0 else sub_w
            for ti, dec in enumerate(decoders):
                task_name = dec["task"]
                tx   = sub_x + ti * (t_w + t_gap)
                fill = task_pastel.get(task_name, "#eeeeee")
                ax3.add_patch(FancyBboxPatch(
                    (tx, task_y), t_w, TASK_H, boxstyle="round,pad=0.004",
                    facecolor=fill, edgecolor="#555555", linewidth=0.5, linestyle="--",
                    transform=ax3.transAxes, clip_on=False))
                ax3.text(tx + t_w / 2, task_y + TASK_H / 2, _wrap_task_name(task_name),
                         transform=ax3.transAxes, ha="center", va="center",
                         fontsize=4.5, fontweight="bold", color="#222222",
                         linespacing=1.1)

            sub_x += sub_w + SUB_GAP

        x_cursor += gpu_col_w + GPU_GAP

    save_figure(fig3, out_deploy)
    plt.close(fig3)

    # ── FIGURE 4: Trace Workload ──────────────────────────────────────────
    fig4, ax4 = plt.subplots(1, 1, figsize=(3.3, 1.6))

    all_workload_ys = []
    for task in all_tasks:
        xs, ys = binned_workload(rows, task=task, bin_s=TRACE_BIN)
        if not len(xs):
            continue
        ax4.plot(xs, ys, color=task_color[task], linewidth=1.0, label=task)
        all_workload_ys.extend(list(ys))
    xs_tot, ys_tot = binned_workload(rows, task=None, bin_s=TRACE_BIN)
    if len(xs_tot):
        ax4.plot(xs_tot, ys_tot, color="black", linewidth=1.2,
                 linestyle="--", label="Total")
        all_workload_ys.extend(list(ys_tot))

    req_max = max(r["req_time"] for r in rows)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Workload (req/s)")
    ymax = max(all_workload_ys) if all_workload_ys else 1.0
    _set_clean_ticks(ax4, req_max, ymax, n_y=4)
    ax4.grid(axis="both", zorder=0)
    ax4.set_axisbelow(True)
    handles4, labels4 = ax4.get_legend_handles_labels()
    fig4.legend(handles4, labels4, loc="upper center",
                bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
                handlelength=1.2, handletextpad=0.3, columnspacing=0.8)
    fig4.tight_layout(rect=(0, 0, 1, 0.97))

    save_figure(fig4, out_workload)
    plt.close(fig4)


# ── Generate per-method plots ──────────────────────────────────────────────────
if args.compare:
    # Already discovered methods in the comparison block above
    for m in methods:
        run_dir = os.path.join(args.compare, m, args.run)
        print(f"Plotting {m}...")
        _plot_single_method(run_dir)
else:
    _plot_single_method(DATA_DIR)

print("Done.")
