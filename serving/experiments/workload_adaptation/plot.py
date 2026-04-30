#!/usr/bin/env python3
"""Plotting for the workload-adaptation experiment.

Reads <results-dir>/request_latency_results.csv + scenario_events.json and
produces:

  1. response_time_timeseries.pdf
       Per-task p50 and p99 end-to-end latency vs. trace time.
       All tasks on one axis (lines), bump/attach/spinup events as vlines.

  2. throughput_timeseries.pdf
       Per-task throughput (requests completed per second) vs. trace time.
       All tasks on one axis, same event vlines.

  3. per_device_timeseries.pdf  (bonus)
       Throughput broken out by (task, device) so you can see traffic
       physically split across devices after the attach event.

Usage:
    python -m experiments.workload_adaptation.plot \\
        --results-dir experiments/workload_adaptation/results/case1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────
# Loading
# ────────────────────────────────────────────────────────────────────

def _load_results(results_dir: Path) -> tuple[pd.DataFrame, list[dict], float]:
    csv_path = results_dir / "request_latency_results.csv"
    events_path = results_dir / "scenario_events.json"

    if not csv_path.is_file():
        raise FileNotFoundError(f"missing {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["task"] != "unknown"].copy()
    # Drop warmup requests (req_id < 0).
    df = df[df["req_id"] >= 0].copy()

    events = []
    trace_start_epoch = None
    if events_path.is_file():
        events = json.loads(events_path.read_text())
        for e in events:
            if e["label"] == "trace_start":
                trace_start_epoch = float(e["params"]["start_epoch"])
                break

    # Two time axes:
    #   t_sec       — when the request COMPLETED (from client clock)
    #   arrival_sec — when the request was SCHEDULED to fire (trace req_time)
    df["recv_epoch"] = df["client_receive_time"].astype(float)
    if trace_start_epoch is None:
        trace_start_epoch = float(df["recv_epoch"].min())
        print(f"[plot] no trace_start event; using min recv_time = {trace_start_epoch}")
    df["t_sec"] = df["recv_epoch"] - trace_start_epoch
    # `req_time` is already in trace-relative seconds (set by trace generator).
    df["arrival_sec"] = df["req_time"].astype(float)
    return df, events, trace_start_epoch


def _event_offsets(events: list[dict], trace_start_epoch: float) -> list[tuple[float, str]]:
    """Return [(t_sec, label)] for events that should appear on the plot."""
    keep = {"bump_rps", "attach_decoder", "attach_decoder_done", "split_traffic",
            "start_backbone_begin", "start_backbone_done"}
    out = []
    for e in events:
        if e["label"] not in keep:
            continue
        t = float(e["epoch"]) - trace_start_epoch
        if t < 0:
            continue
        # Annotate attach_decoder_done with the measured latency so the
        # legend reads e.g. "attach done (+812 ms)".
        params = e.get("params", {}) or {}
        annot = ""
        if e["label"] == "attach_decoder_done" and "attach_e2e_ms" in params:
            annot = f" (+{float(params['attach_e2e_ms']):.0f} ms)"
        out.append((t, e["label"], annot))
    return out


# ────────────────────────────────────────────────────────────────────
# Aggregation
# ────────────────────────────────────────────────────────────────────

def _per_task_timeseries(df: pd.DataFrame, bin_s: float = 2.0) -> pd.DataFrame:
    """Bin requests by completion time into fixed windows, per task.

    Returns a DataFrame with columns:
      task, t_center, count, throughput_rps, p50_ms, p99_ms, mean_ms
    """
    if df.empty:
        return df

    edges = np.arange(0.0, df["t_sec"].max() + bin_s, bin_s)
    bins = pd.cut(df["t_sec"], edges, right=False, include_lowest=True)
    centers = edges[:-1] + bin_s / 2.0
    bin_label_to_center = dict(zip(bins.cat.categories, centers))

    rows = []
    for (task, b), grp in df.groupby(["task", bins]):
        if len(grp) == 0:
            continue
        center = bin_label_to_center.get(b)
        if center is None:
            continue
        lat = grp["end_to_end_latency(ms)"].astype(float).to_numpy()
        rows.append({
            "task": task,
            "t_center": float(center),
            "count": int(len(grp)),
            "throughput_rps": float(len(grp)) / bin_s,
            "p50_ms": float(np.percentile(lat, 50)),
            "p99_ms": float(np.percentile(lat, 99)),
            "mean_ms": float(np.mean(lat)),
        })
    return pd.DataFrame(rows).sort_values(["task", "t_center"])


def _arrival_vs_served(df: pd.DataFrame, bin_s: float) -> pd.DataFrame:
    """Per-task arrival rate (from trace) and served rate (from completions)
    in matching bins. Gap = unmet demand at that moment.
    """
    if df.empty:
        return df
    t_max = float(max(df["t_sec"].max(), df["arrival_sec"].max()))
    edges = np.arange(0.0, t_max + bin_s, bin_s)
    centers = edges[:-1] + bin_s / 2.0

    arr_bins = pd.cut(df["arrival_sec"], edges, right=False, include_lowest=True)
    srv_bins = pd.cut(df["t_sec"],       edges, right=False, include_lowest=True)
    bin_to_center = dict(zip(arr_bins.cat.categories, centers))

    arr_counts = (df.groupby(["task", arr_bins], observed=True)
                    .size().rename("arr_count").reset_index())
    srv_counts = (df.groupby(["task", srv_bins], observed=True)
                    .size().rename("srv_count").reset_index())
    arr_counts.columns = ["task", "bin", "arr_count"]
    srv_counts.columns = ["task", "bin", "srv_count"]
    # Map Categorical bin → numeric center BEFORE merging so fillna works.
    arr_counts["t_center"] = arr_counts["bin"].map(bin_to_center)
    srv_counts["t_center"] = srv_counts["bin"].map(bin_to_center)
    arr_counts = arr_counts.drop(columns=["bin"]).dropna(subset=["t_center"])
    srv_counts = srv_counts.drop(columns=["bin"]).dropna(subset=["t_center"])
    merged = pd.merge(arr_counts, srv_counts, on=["task", "t_center"], how="outer")
    merged[["arr_count", "srv_count"]] = merged[["arr_count", "srv_count"]].fillna(0)
    merged["arrival_rps"] = merged["arr_count"] / bin_s
    merged["served_rps"] = merged["srv_count"] / bin_s
    return merged.sort_values(["task", "t_center"])[
        ["task", "t_center", "arrival_rps", "served_rps", "arr_count", "srv_count"]
    ]


def _per_task_device_timeseries(df: pd.DataFrame, bin_s: float = 2.0) -> pd.DataFrame:
    """Same as _per_task_timeseries but also broken out by device."""
    if df.empty:
        return df
    edges = np.arange(0.0, df["t_sec"].max() + bin_s, bin_s)
    bins = pd.cut(df["t_sec"], edges, right=False, include_lowest=True)
    centers = edges[:-1] + bin_s / 2.0
    bin_label_to_center = dict(zip(bins.cat.categories, centers))

    rows = []
    for (task, dev, b), grp in df.groupby(["task", "device", bins]):
        if len(grp) == 0:
            continue
        center = bin_label_to_center.get(b)
        if center is None:
            continue
        lat = grp["end_to_end_latency(ms)"].astype(float).to_numpy()
        rows.append({
            "task": task,
            "device": dev,
            "t_center": float(center),
            "count": int(len(grp)),
            "throughput_rps": float(len(grp)) / bin_s,
            "p50_ms": float(np.percentile(lat, 50)),
            "p99_ms": float(np.percentile(lat, 99)),
        })
    return pd.DataFrame(rows).sort_values(["task", "device", "t_center"])


# ────────────────────────────────────────────────────────────────────
# Plotting helpers
# ────────────────────────────────────────────────────────────────────

_EVENT_STYLE = {
    "bump_rps":             ("tab:red",    "-",  "bump RPS"),
    "attach_decoder":       ("tab:green",  "-",  "attach start"),
    "attach_decoder_done":  ("tab:green",  "--", "attach done"),
    "split_traffic":        ("tab:olive",  ":",  "split traffic"),
    "start_backbone_begin": ("tab:purple", "-",  "spinup begin"),
    "start_backbone_done":  ("tab:purple", "--", "spinup done"),
}


def _draw_event_lines(ax, event_offsets):
    seen_labels = set()
    for t, label, annot in event_offsets:
        color, ls, pretty = _EVENT_STYLE.get(label, ("gray", ":", label))
        pretty_with_annot = f"{pretty}{annot}"
        legend_label = pretty_with_annot if pretty_with_annot not in seen_labels else None
        ax.axvline(t, color=color, linestyle=ls, alpha=0.7,
                   linewidth=1.2, label=legend_label, zorder=0)
        seen_labels.add(pretty_with_annot)


def _color_cycle(n: int):
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(n)]


# ────────────────────────────────────────────────────────────────────
# Plots
# ────────────────────────────────────────────────────────────────────

def plot_response_time(ts: pd.DataFrame, event_offsets, out_path: Path,
                       bin_s: float, title_suffix: str = ""):
    tasks = sorted(ts["task"].unique())
    colors = _color_cycle(len(tasks))
    color_map = dict(zip(tasks, colors))

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax_mean, ax_p50, ax_p99 = axes

    for task in tasks:
        sub = ts[ts["task"] == task]
        ax_mean.plot(sub["t_center"], sub["mean_ms"], label=task,
                     color=color_map[task], linewidth=1.4)
        ax_p50.plot(sub["t_center"], sub["p50_ms"], label=task,
                    color=color_map[task], linewidth=1.4)
        ax_p99.plot(sub["t_center"], sub["p99_ms"], label=task,
                    color=color_map[task], linewidth=1.4)

    for ax, ylabel in [(ax_mean, "mean e2e latency (ms)"),
                       (ax_p50, "p50 e2e latency (ms)"),
                       (ax_p99, "p99 e2e latency (ms)")]:
        _draw_event_lines(ax, event_offsets)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    ax_p99.set_xlabel(f"trace time (s)  [bin = {bin_s:g}s]")
    handles, labels = ax_mean.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.13, 0.5),
               fontsize=8, frameon=False)
    fig.suptitle(f"Per-task response time vs. time{title_suffix}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.88, 0.96))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_throughput(ts: pd.DataFrame, event_offsets, out_path: Path,
                    bin_s: float, title_suffix: str = ""):
    tasks = sorted(ts["task"].unique())
    colors = _color_cycle(len(tasks))
    color_map = dict(zip(tasks, colors))

    fig, ax = plt.subplots(figsize=(11, 5))
    for task in tasks:
        sub = ts[ts["task"] == task]
        ax.plot(sub["t_center"], sub["throughput_rps"], label=task,
                color=color_map[task], linewidth=1.4)

    _draw_event_lines(ax, event_offsets)
    ax.set_xlabel(f"trace time (s)  [bin = {bin_s:g}s]")
    ax.set_ylabel("throughput (req/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    fig.suptitle(f"Per-task throughput vs. time{title_suffix}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1.0, 0.96))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_cumulative_arrival_vs_served(df: pd.DataFrame, event_offsets,
                                      out_path: Path,
                                      xlim: tuple | None = None,
                                      title_suffix: str = ""):
    """One step-line per request — no binning.

    For each task, plots:
      - cumulative arrivals (one step per request, at req_time) — dashed
      - cumulative completions (one step per request, at recv time) — solid

    The vertical gap = number of requests offered but not yet completed at
    that instant. Pile-up during decoder attach shows up as the dashed line
    racing ahead of the solid line.
    """
    tasks = sorted(df["task"].unique())
    n = len(tasks)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.8 * n + 0.5),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    colors = _color_cycle(n)

    for ax, task, c in zip(axes, tasks, colors):
        sub = df[df["task"] == task]
        arr_t = np.sort(sub["arrival_sec"].to_numpy())
        srv_t = np.sort(sub["t_sec"].to_numpy())
        if len(arr_t) == 0:
            continue
        arr_y = np.arange(1, len(arr_t) + 1)
        srv_y = np.arange(1, len(srv_t) + 1)
        ax.step(arr_t, arr_y, where="post", linestyle="--",
                linewidth=1.0, color=c, label="offered")
        ax.step(srv_t, srv_y, where="post", linestyle="-",
                linewidth=1.4, color=c, label="served")
        _draw_event_lines(ax, event_offsets)
        ax.set_ylabel("requests")
        ax.set_title(f"task = {task}", fontsize=10, loc="left")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7, frameon=False)

    if xlim is not None:
        axes[-1].set_xlim(*xlim)
    axes[-1].set_xlabel("trace time (s)")
    fig.suptitle(f"Cumulative offered vs served per task (gap = in-flight){title_suffix}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1.0, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_arrival_vs_served(av: pd.DataFrame, event_offsets, out_path: Path,
                           bin_s: float, xlim: tuple | None = None,
                           title_suffix: str = ""):
    """One panel per task: arrival rate (dashed) vs served rate (solid).
    The visible gap = requests piling up because the system can't keep up.
    """
    tasks = sorted(av["task"].unique())
    n = len(tasks)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.8 * n + 0.5),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    colors = _color_cycle(n)
    for ax, task, c in zip(axes, tasks, colors):
        sub = av[av["task"] == task]
        ax.plot(sub["t_center"], sub["arrival_rps"], color=c, linestyle="--",
                linewidth=1.2, label="arrival (offered)")
        ax.plot(sub["t_center"], sub["served_rps"], color=c, linestyle="-",
                linewidth=1.6, label="served (completed)")
        # Shade unmet-demand region (arrival > served).
        ax.fill_between(sub["t_center"], sub["served_rps"], sub["arrival_rps"],
                        where=(sub["arrival_rps"] > sub["served_rps"]),
                        alpha=0.18, color=c, interpolate=True)
        _draw_event_lines(ax, event_offsets)
        ax.set_ylabel("rps")
        ax.set_title(f"task = {task}", fontsize=10, loc="left")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, frameon=False)

    if xlim is not None:
        axes[-1].set_xlim(*xlim)
    axes[-1].set_xlabel(f"trace time (s)  [bin = {bin_s:g}s]")
    fig.suptitle(f"Arrival vs. served (gap = unmet demand){title_suffix}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1.0, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_zoom_spike(df: pd.DataFrame, event_offsets, out_path: Path,
                    zoom_window: tuple, title_suffix: str = ""):
    """Tightly-zoomed plot of the bump-to-recovery window using individual
    request points (not bins) to expose sub-second behavior.

    Top: scatter of e2e latency (ms) per completed request.
    Bottom: arrival vs served rate at 100ms bins.
    """
    a, b = zoom_window
    sub = df[(df["t_sec"] >= a - 1) & (df["t_sec"] <= b + 1)].copy()
    if sub.empty:
        print(f"[plot] zoom window {zoom_window} has no requests; skip.")
        return

    av = _arrival_vs_served(sub, bin_s=0.1)

    tasks = sorted(sub["task"].unique())
    colors = _color_cycle(len(tasks))
    color_map = dict(zip(tasks, colors))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1]})
    ax_lat, ax_rate = axes

    for task in tasks:
        s = sub[sub["task"] == task]
        ax_lat.scatter(s["t_sec"], s["end_to_end_latency(ms)"],
                       s=14, alpha=0.7, color=color_map[task], label=task,
                       edgecolors="none")
    ax_lat.set_yscale("log")
    ax_lat.set_ylabel("e2e latency (ms)")
    ax_lat.grid(True, alpha=0.3)
    ax_lat.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    _draw_event_lines(ax_lat, event_offsets)

    for task in tasks:
        s = av[av["task"] == task]
        ax_rate.plot(s["t_center"], s["arrival_rps"], color=color_map[task],
                     linestyle="--", linewidth=1.0, alpha=0.6)
        ax_rate.plot(s["t_center"], s["served_rps"], color=color_map[task],
                     linestyle="-", linewidth=1.4, label=task)
        ax_rate.fill_between(s["t_center"], s["served_rps"], s["arrival_rps"],
                             where=(s["arrival_rps"] > s["served_rps"]),
                             alpha=0.18, color=color_map[task], interpolate=True)
    _draw_event_lines(ax_rate, event_offsets)
    ax_rate.set_ylabel("rps  (dashed=arrival, solid=served)")
    ax_rate.set_xlabel("trace time (s)  [bin = 0.1s]")
    ax_rate.grid(True, alpha=0.3)

    ax_lat.set_xlim(a, b)
    fig.suptitle(f"Zoomed view of bump → recovery{title_suffix}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1.0, 0.96))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_per_device(ts_dev: pd.DataFrame, event_offsets, out_path: Path,
                    bin_s: float, title_suffix: str = ""):
    """Stacked per-device throughput for the task whose load was bumped.

    We don't know a priori which task was bumped, so plot one panel per task
    that ever appeared on more than one device — those are the interesting
    ones for the offload story.
    """
    multi_dev_tasks = []
    for task, sub in ts_dev.groupby("task"):
        if sub["device"].nunique() > 1:
            multi_dev_tasks.append(task)

    if not multi_dev_tasks:
        print("[plot] no multi-device tasks; skipping per-device plot.")
        return

    n = len(multi_dev_tasks)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, task in zip(axes, multi_dev_tasks):
        sub = ts_dev[ts_dev["task"] == task]
        devs = sorted(sub["device"].unique())
        dev_colors = _color_cycle(len(devs))
        for dev, c in zip(devs, dev_colors):
            s = sub[sub["device"] == dev]
            ax.plot(s["t_center"], s["throughput_rps"],
                    label=dev, color=c, linewidth=1.4)
        _draw_event_lines(ax, event_offsets)
        ax.set_title(f"task = {task}", fontsize=10)
        ax.set_ylabel("req/s")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, frameon=False)

    axes[-1].set_xlabel(f"trace time (s)  [bin = {bin_s:g}s]")
    fig.suptitle(f"Per-device throughput for offloaded task(s){title_suffix}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1.0, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ────────────────────────────────────────────────────────────────────
# Deployment timeline (one diagram per scenario phase)
# ────────────────────────────────────────────────────────────────────

from copy import deepcopy
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BACKBONE_ABBREV = {
    "momentsmall": "MS", "momentbase": "MB", "momentlarge": "ML",
    "chronostiny": "CT", "chronossmall": "CS", "chronosbase": "CB",
    "chronoslarge": "CL", "dinosmall": "DS", "dinobase": "DB",
    "swinsmall": "SS", "swinbase": "SB",
}
TASK_PASTELS = [
    "#ffffcc", "#d4f1c0", "#ffd6e0", "#d6eaff",
    "#ffe8cc", "#e8d5f5", "#ccf5f1", "#fce4d6",
]
GPU_COLOR      = "#a8dde8"
BACKBONE_COLOR = "#fdd58a"
EMPTY_COLOR    = "#ececec"


def _all_devices_from_events(plan: dict, events: list[dict]) -> list[str]:
    """Union of devices that appear in the initial plan + any event."""
    seen: list[str] = []
    def _add(name: str):
        if name and name not in seen:
            seen.append(name)
    for site in plan.get("sites", []):
        for d in site.get("deployments", []):
            _add(d.get("device_name", d.get("device", "")))
    # device names referenced in events ("device1", "device2", ...)
    for e in events:
        p = e.get("params") or {}
        if "device" in p:
            _add(p["device"])
    return seen


def _dep_port(d: dict) -> int | None:
    """Extract port from a deployment dict (parses 'host:port' if needed)."""
    if "port" in d and d["port"] is not None:
        try:
            return int(d["port"])
        except (TypeError, ValueError):
            pass
    addr = d.get("device", "")
    if isinstance(addr, str) and ":" in addr:
        try:
            return int(addr.rsplit(":", 1)[-1])
        except ValueError:
            return None
    return None


def _phase_after_event(prev_plan: dict, event: dict) -> dict:
    """Return a new plan reflecting `event` applied to `prev_plan`.

    Only structural events (start_backbone_done, attach_decoder_done,
    attach_decoder when no _done is emitted) actually change the plan.
    Deployments are disambiguated by (device_name, port) since one GPU
    can host multiple backbones via TPC partitioning.
    """
    plan = deepcopy(prev_plan)
    label = event["label"]
    p = event.get("params") or {}
    site = plan["sites"][0] if plan.get("sites") else {"id": "site0", "deployments": []}
    if "sites" not in plan:
        plan["sites"] = [site]
    deps = site["deployments"]
    ev_port = p.get("port")
    try:
        ev_port = int(ev_port) if ev_port is not None else None
    except (TypeError, ValueError):
        ev_port = None

    if label == "start_backbone_done":
        dev_name = p.get("device", "")
        # Don't duplicate if a deployment with the same (device, port) exists.
        match = next(
            (d for d in deps
             if d.get("device_name") == dev_name and _dep_port(d) == ev_port),
            None,
        )
        if match is None:
            deps.append({
                "device_name": dev_name,
                "device": f"{dev_name}:{ev_port}" if ev_port is not None else dev_name,
                "port": ev_port,
                "backbone": p.get("backbone", "?"),
                "decoders": [],
                "tpc_partition": p.get("tpc_partition"),
            })
    elif label in ("attach_decoder_done", "attach_decoder"):
        dev_name = p.get("device", "")
        task = p.get("task", "")
        # If the event names a port, attach to that exact deployment;
        # otherwise attach to the most-recently-added deployment on that GPU
        # (decoder-attach typically follows a backbone-start on the same port).
        candidates = [d for d in deps if d.get("device_name") == dev_name]
        if ev_port is not None:
            candidates = [d for d in candidates if _dep_port(d) == ev_port] or candidates
        target = candidates[-1] if candidates else None
        if target is not None and task:
            tasks = [dec.get("task") for dec in target.get("decoders", [])]
            if task not in tasks:
                target.setdefault("decoders", []).append({"task": task})
    return plan


def _build_phases(plan0: dict, events: list[dict]) -> list[tuple[str, dict, float]]:
    """Return [(label, plan, t_sec)] phases.

    The first phase is the initial plan ("initial"). Each subsequent phase
    corresponds to a structural event applied cumulatively.
    """
    trace_start = next((e["epoch"] for e in events if e["label"] == "trace_start"), None)
    phases: list[tuple[str, dict, float]] = [("initial", plan0, 0.0)]
    structural = {"start_backbone_done", "attach_decoder_done"}
    # Fallback: if a case never emits attach_decoder_done, take attach_decoder.
    has_done = any(e["label"] == "attach_decoder_done" for e in events)
    if not has_done:
        structural.add("attach_decoder")

    # Remember context from start_backbone_begin keyed by device, since
    # start_backbone_done only carries the device.
    pending_backbone: dict[str, dict] = {}
    cur_plan = plan0
    for e in events:
        if e["label"] == "start_backbone_begin":
            pp = e.get("params") or {}
            pending_backbone[pp.get("device", "")] = {
                "backbone": pp.get("backbone", ""),
                "port": pp.get("port"),
                "tpc_partition": pp.get("tpc_partition"),
            }
            continue
        if e["label"] not in structural:
            continue
        # Inject backbone/port/tpc for start_backbone_done from the begin event.
        if e["label"] == "start_backbone_done":
            dev = (e.get("params") or {}).get("device", "")
            if dev in pending_backbone:
                e = deepcopy(e)
                e["params"].update(pending_backbone[dev])
        cur_plan = _phase_after_event(cur_plan, e)
        t = (float(e["epoch"]) - float(trace_start)) if trace_start else 0.0
        p = e.get("params") or {}
        if e["label"] == "start_backbone_done":
            lab = f"+ backbone\non {p.get('device','?')}"
        else:
            lab = f"+ decoder {p.get('task','?')}\non {p.get('device','?')}"
        phases.append((lab, cur_plan, t))
    return phases


def _draw_panel(ax, plan: dict, gpu_order: list[str], task_color: dict,
                task_abbrev: dict) -> None:
    """Draw one deployment diagram inside `ax` (axes set to 0..1, no spines).

    Each GPU column may host multiple deployments side-by-side (TPC sharing).
    Sub-column widths are proportional to TPC count when available, otherwise
    split evenly. Unused TPC capacity on a GPU is shown as a dashed empty box.
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Group deployments by GPU (device_name).
    deps_by_dev: dict[str, list[dict]] = {}
    for site in plan.get("sites", []):
        for d in site.get("deployments", []):
            deps_by_dev.setdefault(
                d.get("device_name", d.get("device", "")), []
            ).append(d)
    # Stable ordering within a GPU: by port (ascending) so the diagram is
    # deterministic across phases.
    for k in deps_by_dev:
        deps_by_dev[k].sort(key=lambda d: (_dep_port(d) is None, _dep_port(d) or 0))

    n = len(gpu_order)
    if n == 0:
        return
    GPU_GAP = 0.04
    col_w = (1.0 - GPU_GAP * (n - 1)) / n

    GPU_H, BB_H, TASK_H = 0.16, 0.18, 0.22
    BOTTOM = 0.05
    PAD = 0.03
    SUB_GAP = 0.008
    bb_y = BOTTOM + GPU_H + PAD
    task_y = bb_y + BB_H + PAD

    # Total TPCs per A2 (used to render unused capacity as a free sub-column).
    GPU_TPC_TOTAL = 5

    x = 0.0
    for gpu_label in gpu_order:
        ax.add_patch(FancyBboxPatch(
            (x, BOTTOM), col_w, GPU_H, boxstyle="round,pad=0.005",
            facecolor=GPU_COLOR, edgecolor="#5aafc4", linewidth=0.7,
            transform=ax.transAxes, clip_on=False))
        ax.text(x + col_w / 2, BOTTOM + GPU_H / 2, gpu_label,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7, fontweight="bold")

        devs = deps_by_dev.get(gpu_label, [])

        # Compute sub-column widths.
        # If every deployment has a tpc_partition, split proportionally to TPC
        # count and reserve a "free" sub-column for unused TPCs (so the visual
        # widths reflect actual GPU partitioning). Otherwise split evenly.
        tpc_counts = [
            (len(d["tpc_partition"]) if isinstance(d.get("tpc_partition"), list) else None)
            for d in devs
        ]
        all_have_tpc = devs and all(c is not None and c > 0 for c in tpc_counts)
        if all_have_tpc:
            used = sum(tpc_counts)
            free = max(0, GPU_TPC_TOTAL - used)
            slots = list(zip(devs, tpc_counts))  # (dev, count)
            if free > 0:
                slots.append((None, free))
            n_slots = len(slots)
            avail = col_w - SUB_GAP * (n_slots - 1)
            total_tpc = float(used + free)
            sub_ws = [avail * (c / total_tpc) for _, c in slots]
        elif devs:
            n_slots = len(devs)
            avail = col_w - SUB_GAP * (n_slots - 1)
            sub_ws = [avail / n_slots] * n_slots
            slots = [(d, None) for d in devs]
        else:
            # No deployments yet on this GPU — render one big empty slot.
            slots = [(None, None)]
            sub_ws = [col_w]

        sub_x = x
        for (d, _tc), sw in zip(slots, sub_ws):
            if d is None:
                ax.add_patch(FancyBboxPatch(
                    (sub_x, bb_y), sw, BB_H, boxstyle="round,pad=0.004",
                    facecolor=EMPTY_COLOR, edgecolor="#999999",
                    linewidth=0.5, linestyle="--",
                    transform=ax.transAxes, clip_on=False))
                ax.text(sub_x + sw / 2, bb_y + BB_H / 2, "—",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=7, color="#666666")
                sub_x += sw + SUB_GAP
                continue

            bb = d.get("backbone", "?")
            ax.add_patch(FancyBboxPatch(
                (sub_x, bb_y), sw, BB_H, boxstyle="round,pad=0.004",
                facecolor=BACKBONE_COLOR, edgecolor="#c8950a", linewidth=0.6,
                transform=ax.transAxes, clip_on=False))
            ax.text(sub_x + sw / 2, bb_y + BB_H / 2,
                    BACKBONE_ABBREV.get(bb, bb),
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7.5, fontweight="bold")

            task_names = [dec.get("task") for dec in d.get("decoders", [])]
            n_t = max(len(task_names), 1)
            t_gap = 0.003
            t_w = (sw - t_gap * (n_t - 1)) / n_t
            for ti, tname in enumerate(task_names):
                tx = sub_x + ti * (t_w + t_gap)
                ax.add_patch(FancyBboxPatch(
                    (tx, task_y), t_w, TASK_H, boxstyle="round,pad=0.004",
                    facecolor=task_color.get(tname, "#eeeeee"),
                    edgecolor="#555555", linewidth=0.5, linestyle="--",
                    transform=ax.transAxes, clip_on=False))
                ax.text(tx + t_w / 2, task_y + TASK_H / 2,
                        task_abbrev.get(tname, tname),
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=7, fontweight="bold")
            sub_x += sw + SUB_GAP
        x += col_w + GPU_GAP


def plot_deployment_timeline(plan: dict, events: list[dict],
                             out_path: Path, title_suffix: str = "") -> None:
    """Render a horizontal strip of deployment diagrams, one per phase.

    A phase is the initial plan or the plan after a structural event
    (backbone start, decoder attach). Phases are labeled with the event
    that produced them and the trace-relative time (s)."""
    phases = _build_phases(plan, events)
    if not phases:
        return

    gpu_order = _all_devices_from_events(plan, events)
    if not gpu_order:
        # fall back to whatever the last phase has
        gpu_order = []
        for site in phases[-1][1].get("sites", []):
            for d in site.get("deployments", []):
                gpu_order.append(d.get("device_name", d.get("device", "")))

    all_tasks: list[str] = []
    for _, pl, _ in phases:
        for site in pl.get("sites", []):
            for d in site.get("deployments", []):
                for dec in d.get("decoders", []):
                    t = dec.get("task")
                    if t and t not in all_tasks:
                        all_tasks.append(t)
    task_color = {t: TASK_PASTELS[i % len(TASK_PASTELS)]
                  for i, t in enumerate(all_tasks)}
    task_abbrev = {t: f"T{i+1}" for i, t in enumerate(all_tasks)}

    n_phases = len(phases)
    panel_w = max(1.6, 0.7 * len(gpu_order) + 0.6)
    fig_w = panel_w * n_phases + 0.4 * (n_phases - 1) + 0.5
    fig_h = 2.4

    fig = plt.figure(figsize=(fig_w, fig_h))
    # Reserve a small strip on top for arrows/event labels.
    gs = fig.add_gridspec(2, n_phases, height_ratios=[1.0, 4.0],
                          hspace=0.05, wspace=0.25,
                          left=0.02, right=0.98, top=0.94, bottom=0.06)

    # Title row: phase labels with timestamps
    for i, (label, _, t) in enumerate(phases):
        ax_top = fig.add_subplot(gs[0, i])
        ax_top.set_xlim(0, 1); ax_top.set_ylim(0, 1); ax_top.axis("off")
        prefix = "t=0s" if i == 0 else f"t={t:.1f}s"
        ax_top.text(0.5, 0.55, label, ha="center", va="center",
                    fontsize=8, fontweight="bold")
        ax_top.text(0.5, 0.10, prefix, ha="center", va="center",
                    fontsize=7, color="#444444")

    # Panel row: deployment diagrams
    panel_axes = []
    for i, (_, pl, _) in enumerate(phases):
        ax = fig.add_subplot(gs[1, i])
        _draw_panel(ax, pl, gpu_order, task_color, task_abbrev)
        panel_axes.append(ax)

    # Arrows between adjacent panels (figure-level coords)
    fig.canvas.draw()
    for i in range(n_phases - 1):
        a, b = panel_axes[i], panel_axes[i + 1]
        bb_a = a.get_position(); bb_b = b.get_position()
        y = (bb_a.y0 + bb_a.y1) / 2.0
        arrow = FancyArrowPatch(
            (bb_a.x1 + 0.002, y), (bb_b.x0 - 0.002, y),
            transform=fig.transFigure, arrowstyle="->",
            mutation_scale=10, linewidth=0.8, color="#444444")
        fig.patches.append(arrow)

    # Legend (tasks + backbones)
    legend_bits = []
    for t in all_tasks:
        legend_bits.append(f"{task_abbrev[t]}={t}")
    backbones_seen = sorted({
        d.get("backbone", "")
        for _, pl, _ in phases
        for site in pl.get("sites", [])
        for d in site.get("deployments", [])
        if d.get("backbone")
    })
    for bb in backbones_seen:
        legend_bits.append(f"{BACKBONE_ABBREV.get(bb, bb)}={bb}")
    if legend_bits:
        fig.text(0.5, 0.005, "  |  ".join(legend_bits),
                 ha="center", va="bottom", fontsize=7, color="#333333")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",default="experiments/workload_adaptation/results/case1",
                    help="Directory containing request_latency_results.csv")
    ap.add_argument("--bin-s", type=float, default=1.0,
                    help="Aggregation window for full-run timeseries (seconds)")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write plots (default: --results-dir)")
    ap.add_argument("--zoom", default=None,
                    help="Zoom window for spike plot, format 'A:B' in seconds "
                         "(default: auto, centered on bump_rps event)")
    args = ap.parse_args()

    results_dir = Path(os.path.abspath(args.results_dir))
    out_dir = Path(os.path.abspath(args.out_dir)) if args.out_dir else Path(f"{results_dir}/plots") 
    out_dir.mkdir(parents=True, exist_ok=True)

    df, events, trace_start_epoch = _load_results(results_dir)
    print(f"[plot] {len(df)} requests, "
          f"{df['task'].nunique()} tasks, "
          f"trace_start_epoch={trace_start_epoch:.3f}")

    event_offsets = _event_offsets(events, trace_start_epoch)
    print(f"[plot] events: {event_offsets}")

    title_suffix = f"  ({results_dir.name})"

    ts = _per_task_timeseries(df, bin_s=args.bin_s)
    ts.to_csv(out_dir / "per_task_timeseries.csv", index=False)
    print(f"[plot] wrote {out_dir / 'per_task_timeseries.csv'}")

    ts_dev = _per_task_device_timeseries(df, bin_s=args.bin_s)
    ts_dev.to_csv(out_dir / "per_task_device_timeseries.csv", index=False)

    plot_response_time(
        ts, event_offsets,
        out_dir / "response_time_timeseries.pdf",
        bin_s=args.bin_s, title_suffix=title_suffix,
    )
    plot_throughput(
        ts, event_offsets,
        out_dir / "throughput_timeseries.pdf",
        bin_s=args.bin_s, title_suffix=title_suffix,
    )
    plot_per_device(
        ts_dev, event_offsets,
        out_dir / "per_device_timeseries.pdf",
        bin_s=args.bin_s, title_suffix=title_suffix,
    )

    # Arrival vs served (full run, default bin).
    av = _arrival_vs_served(df, bin_s=args.bin_s)
    av.to_csv(out_dir / "arrival_vs_served.csv", index=False)
    plot_arrival_vs_served(
        av, event_offsets, out_dir / "arrival_vs_served.pdf",
        bin_s=args.bin_s, title_suffix=title_suffix,
    )

    # Cumulative arrival/served — no binning, one step per request.
    plot_cumulative_arrival_vs_served(
        df, event_offsets, out_dir / "cumulative_arrival_vs_served.pdf",
        title_suffix=title_suffix,
    )

    # Zoomed spike plot — defaults to ±5s around the bump event.
    if args.zoom:
        a_str, b_str = args.zoom.split(":")
        zoom_window = (float(a_str), float(b_str))
    else:
        bump_t = next((t for t, label, _ in event_offsets if label == "bump_rps"), None)
        if bump_t is not None:
            zoom_window = (max(0.0, bump_t - 3.0), bump_t + 8.0)
        else:
            zoom_window = None
    if zoom_window is not None:
        plot_zoom_spike(
            df, event_offsets, out_dir / "zoom_spike.pdf",
            zoom_window=zoom_window, title_suffix=title_suffix,
        )

    plan_path = results_dir / "deployment_plan.json"
    events_path = results_dir / "scenario_events.json"
    if plan_path.is_file() and events_path.is_file():
        plan0 = json.loads(plan_path.read_text())
        all_events = json.loads(events_path.read_text())
        plot_deployment_timeline(
            plan0, all_events,
            out_dir / "deployment_timeline.pdf",
            title_suffix=title_suffix,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
