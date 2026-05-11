#!/usr/bin/env python3
"""Plotting for the long-horizon experiment.

Reads results/<condition>/request_latency_results.csv for each condition
and produces:

  1. response_time_comparison.pdf
       Mean end-to-end latency vs. time, all conditions on one axis.

  2. throughput_timeseries.pdf
       Per-condition subplots: actual completed req/s (solid) vs. offered load
       (dashed step) with the unserved gap shaded.  The gap between the two
       curves is the cold-start penalty made explicit.

  3. goodput_ratio.pdf
       Fraction of offered load actually served (actual / offered), all
       conditions on one axis.  FMaaS ≈ 1.0; no_sharing dips to 0 during
       each cold-start.

  4. activation_latency.pdf
       Bar chart: time from task arrive event to first completed request,
       per condition.  Key metric: FMaaS (hot-attach, ~1 s) vs. no_sharing
       (cold-start, ~60–120 s).

  5. latency_timeseries.pdf
       P50 / P99 latency vs. time, one subplot per condition.

  6. per_task_latency.pdf
       Per-task P50 latency timeseries, one subplot per condition.

Usage (from serving/):
    python -m experiments.long_horizon.plot \
        --results-dir experiments/long_horizon/results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── Style ─────────────────────────────────────────────────────────────────────

CONDITION_STYLE = {
    "fmaas":          {"color": "#2166ac", "label": "FMaaS",            "zorder": 3},
    "no_sharing":     {"color": "#d6604d", "label": "No-Sharing",       "zorder": 2},
    "no_sharing_tpc": {"color": "#4dac26", "label": "No-Sharing (TPC)", "zorder": 1},
}

EVENT_STYLE = {
    "arrive": {"color": "#2ca02c", "ls": "--", "lw": 0.8},
    "depart": {"color": "#d62728", "ls": ":",  "lw": 0.8},
}

OFFERED_STYLE = {"color": "black", "ls": "--", "lw": 1.0, "alpha": 0.55}


def _paper_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "black",
        "axes.labelcolor":   "black",
        "axes.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "grid.color":        "#cccccc",
        "grid.linestyle":    ":",
        "grid.linewidth":    0.4,
        "xtick.color":       "black",
        "ytick.color":       "black",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size":  2.5,
        "ytick.major.size":  2.5,
        "text.color":        "black",
        "font.family":       "sans-serif",
        "font.size":         7,
        "axes.titlesize":    7.5,
        "axes.labelsize":    7,
        "xtick.labelsize":   6.5,
        "ytick.labelsize":   6.5,
        "legend.fontsize":   6.5,
        "lines.linewidth":   1.2,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"[plot] wrote {path}")


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_condition(results_dir: Path, condition: str) -> tuple[pd.DataFrame, dict]:
    cdir = results_dir / condition
    csv  = cdir / "request_latency_results.csv"
    cfg  = cdir / "run_config.json"
    if not csv.is_file():
        return pd.DataFrame(), {}
    df = pd.read_csv(csv)
    df = df[df["req_id"] >= 0].copy()
    run_cfg = json.loads(cfg.read_text()) if cfg.is_file() else {}
    return df, run_cfg


def _anchor_time(df: pd.DataFrame, run_cfg: dict) -> pd.DataFrame:
    """Add t_sec (completion relative to experiment start) from client_receive_time.

    Uses start_epoch from run_config.json (written by run.py after warmup) so
    t_sec matches the experiment clock used for arrive/depart times.  Falls back
    to min(client_receive_time) for older result files that lack start_epoch.
    """
    if df.empty:
        return df
    if run_cfg.get("start_epoch"):
        epoch0 = float(run_cfg["start_epoch"])
    else:
        # Estimate start_epoch: dispatch happens at start_epoch + req_time,
        # so start_epoch ≈ client_receive_time - req_time - e2e_latency_s.
        e2e_s  = df["end_to_end_latency(ms)"].astype(float) / 1000.0
        req_t  = pd.to_numeric(df["req_time"], errors="coerce")
        epoch0 = float((df["client_receive_time"].astype(float) - req_t - e2e_s).median())
    df = df.copy()
    df["t_sec"] = df["client_receive_time"].astype(float) - epoch0
    df["end_to_end_latency(ms)"] = df["end_to_end_latency(ms)"].astype(float)
    return df


def _timeseries(df: pd.DataFrame, bin_s: float = 5.0) -> pd.DataFrame:
    """Bin completions by t_sec; compute throughput + latency stats per bin."""
    if df.empty:
        return pd.DataFrame()
    t_max   = float(df["t_sec"].max())
    edges   = np.arange(0.0, t_max + bin_s, bin_s)
    centers = edges[:-1] + bin_s / 2.0
    bins    = pd.cut(df["t_sec"], edges, right=False, include_lowest=True)
    bin_to_center = dict(zip(bins.cat.categories, centers))

    rows = []
    for b, grp in df.groupby(bins, observed=True):
        if grp.empty:
            continue
        lat = grp["end_to_end_latency(ms)"].to_numpy()
        rows.append({
            "t_center":       float(bin_to_center[b]),
            "count":          len(grp),
            "throughput_rps": len(grp) / bin_s,
            "p50_ms":         float(np.percentile(lat, 50)),
            "p99_ms":         float(np.percentile(lat, 99)),
            "mean_ms":        float(np.mean(lat)),
        })
    return pd.DataFrame(rows).sort_values("t_center")


# ── Offered load ──────────────────────────────────────────────────────────────

def _offered_from_trace(trace: list, bin_s: float, duration: float) -> pd.DataFrame:
    """Bin the pre-generated trace req_times into offered req/s.

    The trace is condition-independent (generated from arrive times in
    generate.py), so the offered load curve is identical across all conditions.
    """
    if not trace:
        return pd.DataFrame(columns=["t_center", "offered_rps"])
    req_times = np.array([float(r["req_time"]) for r in trace])
    edges   = np.arange(0.0, duration + bin_s, bin_s)
    centers = (edges[:-1] + edges[1:]) / 2.0
    counts, _ = np.histogram(req_times, bins=edges)
    return pd.DataFrame({"t_center": centers, "offered_rps": counts / bin_s})


# ── Event helpers ─────────────────────────────────────────────────────────────

def _events_from_run_cfg(run_cfg: dict) -> list:
    """Extract flat list of {t, action} events from run_config.json.

    Handles both the old wave-based 'timeline' key and the new per-task
    'task_timeline' dict written by the updated run.py.
    """
    if "timeline" in run_cfg:
        return run_cfg["timeline"]

    task_timeline = run_cfg.get("task_timeline", {})
    duration      = float(run_cfg.get("experiment", {}).get("duration", 400))
    init_secs     = 5.0
    events: list  = []
    for task, info in task_timeline.items():
        arrive = float(info.get("arrive", 0))
        depart = float(info.get("depart", duration))
        if arrive >= init_secs:
            events.append({"t": arrive, "action": "arrive", "task": task})
        if depart < duration:
            events.append({"t": depart, "action": "depart", "task": task})
    return events


def _draw_events(ax: plt.Axes, events: list, duration: float,
                 alpha: float = 0.5) -> None:
    """Draw arrive (green dashed) and depart (red dotted) vlines."""
    for ev in events:
        t   = float(ev["t"])
        act = ev["action"]
        if t > duration:
            continue
        st = EVENT_STYLE.get(act, {"color": "gray", "ls": "--", "lw": 0.8})
        ax.axvline(t, color=st["color"], ls=st["ls"], lw=st["lw"], alpha=alpha)


# ── Activation latency ────────────────────────────────────────────────────────

def _activation_latency_per_task(
    df: pd.DataFrame,
    task_timeline: dict,
    initial_task_set: set,
) -> dict[str, float | None]:
    """For each dynamically-arrived task, compute seconds from arrive to first
    completed request.  Returns {task_name: latency_s | None}."""
    if df.empty:
        return {}
    result = {}
    for task, info in task_timeline.items():
        if task in initial_task_set:
            continue
        arrive = float(info["arrive"])
        sub = df[df["task"] == task]
        if sub.empty:
            result[task] = None
            continue
        first_t_sec = float(sub["t_sec"].min())
        result[task] = max(0.0, first_t_sec - arrive)
    return result


# ── Plot 0: deployment activity ──────────────────────────────────────────────

def _step_over_time(
    intervals: list[tuple[float, float]],
    duration: float,
    t_step: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Count how many intervals [arrive, depart) are active at each t."""
    t = np.arange(0.0, duration + t_step, t_step)
    counts = np.zeros(len(t))
    for arrive, depart in intervals:
        counts += ((t >= arrive) & (t < depart)).astype(float)
    return t, counts


def plot_deployment_activity(
    task_timeline: dict,
    slots_by_cond: dict[str, list],
    out_dir: Path,
    duration: float = 400.0,
) -> None:
    """Active model-process count over time, one line per condition.

    fmaas       — one backbone process per backbone type, all live from t=0
                  (flat line = minimum possible footprint).
    no_sharing  — one process per task, cold-started at arrive, removed at depart
                  (grows/shrinks with workload).

    A dashed black line shows the number of concurrently active tasks (the
    same across all conditions) so the reader can see how many tasks are being
    served at any moment vs. how many processes are running.
    """
    if not slots_by_cond and not task_timeline:
        return

    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    for cond, slots in slots_by_cond.items():
        sty = CONDITION_STYLE.get(cond)
        if sty is None:
            continue

        if cond == "fmaas":
            # Each backbone process goes live when its first task arrives
            # (cold-start then hot-attach) and stays up for the duration.
            intervals = []
            for group in slots:
                if not group.get("tasks"):
                    continue
                first_arrive = min(float(t.get("arrive", 0.0))
                                   for t in group["tasks"])
                intervals.append((first_arrive, duration))
            t, counts = _step_over_time(intervals, duration)
            ax.step(t, counts, where="post",
                    color=sty["color"], lw=1.6, label=sty["label"],
                    zorder=sty["zorder"])
        else:
            # Each slot entry = one task process; active during [arrive, depart)
            intervals = []
            for sl in slots:
                info = task_timeline.get(sl["task"], {})
                intervals.append((
                    float(info.get("arrive", 0.0)),
                    float(info.get("depart", duration)),
                ))
            t, counts = _step_over_time(intervals, duration)
            ax.step(t, counts, where="post",
                    color=sty["color"], lw=1.4, label=sty["label"],
                    zorder=sty["zorder"])

    # Active-task count (condition-independent)
    if task_timeline:
        intervals = [
            (float(v.get("arrive", 0.0)), float(v.get("depart", duration)))
            for v in task_timeline.values()
        ]
        t, active = _step_over_time(intervals, duration)
        ax.step(t, active, where="post",
                color="black", lw=1.0, ls="--", alpha=0.55,
                label="Active tasks", zorder=4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Active model processes")
    ax.set_xlim(0, duration)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, out_dir / "deployment_activity.pdf")
    plt.close(fig)


# ── Plot 0b: GPU memory footprint ────────────────────────────────────────────

def plot_gpu_memory_footprint(
    data: dict[str, tuple[pd.DataFrame, dict]],
    out_dir: Path,
    bin_s: float = 5.0,
) -> None:
    """Total GPU memory footprint over time, all conditions on one axis.

    For each time bin: sum of mean gpu_alloc_peak_mb across all active devices.

    FMaaS: footprint grows gradually as decoders hot-attach to each backbone
           process (decoder weights accumulate on the shared backbone device).
    No-Sharing: each process has a fixed footprint (1 backbone + 1 decoder) but
                the process count grows with active tasks, so total rises steeply.
    """
    conditions = [c for c in CONDITION_STYLE if c in data and not data[c][0].empty]
    if not conditions:
        return

    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    for cond in conditions:
        df, run_cfg = data[cond]
        sty         = CONDITION_STYLE[cond]
        duration    = float(run_cfg.get("experiment", {}).get("duration", 400))

        if "gpu_alloc_peak_mb" not in df.columns:
            continue

        edges   = np.arange(0.0, duration + bin_s, bin_s)
        centers = edges[:-1] + bin_s / 2.0

        # Per-device mean peak MB in each time bin, then sum across devices.
        df = df.copy()
        df["bin_idx"] = np.searchsorted(edges, df["t_sec"].to_numpy(), side="right") - 1
        df["bin_idx"] = df["bin_idx"].clip(0, len(centers) - 1)

        total = np.zeros(len(centers))
        for _, dev_df in df.groupby("device"):
            per_bin = dev_df.groupby("bin_idx")["gpu_alloc_peak_mb"].mean()
            # Forward-fill: once a process is active it stays loaded
            series = pd.Series(index=range(len(centers)), dtype=float)
            series.update(per_bin)
            series = series.ffill().fillna(0.0)
            total += series.to_numpy()

        ax.plot(centers, total,
                color=sty["color"], lw=1.4, label=sty["label"],
                zorder=sty["zorder"])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total GPU alloc peak (MB)")
    ax.set_xlim(0)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, out_dir / "gpu_memory_footprint.pdf")
    plt.close(fig)


# ── Plot 0c: RPS per task ─────────────────────────────────────────────────────

def plot_rps_per_task(
    full_trace: list,
    data: dict,
    task_meta: dict,
    out_dir: Path,
    duration: float,
    bin_s: float = 5.0,
) -> None:
    """Offered and completed RPS per task over time.

    Top panel: offered load per task (from pre-generated trace, shared across
               all conditions) — shows the Alibaba burst shape for each task.
    Lower panels: one per condition, completed RPS per task (actual throughput
                  broken down by task).
    """
    if not full_trace:
        return

    tasks_in_trace = sorted(set(r["task"] for r in full_trace))
    if not tasks_in_trace:
        return

    conditions = [c for c in CONDITION_STYLE if c in data and not data[c][0].empty]

    # Color each task instance uniquely; tasks that share a base use adjacent shades
    # from the same hue block so they're visually grouped but distinguishable.
    cmap = plt.get_cmap("tab20")
    task_color = {t: cmap(i % 20) for i, t in enumerate(tasks_in_trace)}

    edges   = np.arange(0.0, duration + bin_s, bin_s)
    centers = (edges[:-1] + edges[1:]) / 2.0

    n_panels = 1 + len(conditions)
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(6, 2.2 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    # ── Top panel: offered RPS per task ──────────────────────────────────────
    ax_off = axes[0]
    for task in tasks_in_trace:
        req_times = np.array([float(r["req_time"])
                              for r in full_trace if r["task"] == task])
        if req_times.size == 0:
            continue
        counts, _ = np.histogram(req_times, bins=edges)
        rps = counts / bin_s
        if rps.max() == 0:
            continue
        ax_off.step(centers, rps, where="mid",
                    color=task_color[task], lw=0.9, alpha=0.85, label=task)

    ax_off.set_ylabel("Offered req/s")
    ax_off.set_title("Offered load per task (Alibaba compressed trace)", pad=3)
    ax_off.set_ylim(bottom=0)
    ax_off.legend(frameon=False, ncol=4, fontsize=5.0, loc="upper right")
    ax_off.grid(axis="y")

    # ── Per-condition panels: completed RPS per task ──────────────────────────
    for ax, cond in zip(axes[1:], conditions):
        df, run_cfg = data[cond]
        sty = CONDITION_STYLE[cond]

        for task, grp in df.groupby("task"):
            req_times = grp["t_sec"].to_numpy().astype(float)
            counts, _ = np.histogram(req_times, bins=edges)
            rps = counts / bin_s
            if rps.max() == 0:
                continue
            ax.step(centers, rps, where="mid",
                    color=task_color.get(task, "gray"), lw=0.9, alpha=0.85,
                    label=task)

        _draw_events(ax, _events_from_run_cfg(run_cfg), duration, alpha=0.3)
        ax.set_ylabel("Completed req/s")
        ax.set_title(f"{sty['label']} — completed per task", pad=3)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, ncol=4, fontsize=5.0, loc="upper right")
        ax.grid(axis="y")

    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(0, duration)
    fig.tight_layout(h_pad=1.0)
    _save(fig, out_dir / "rps_per_task.pdf")
    plt.close(fig)


# ── Plot 1: response time comparison ─────────────────────────────────────────

def plot_response_time_comparison(
    data: dict[str, tuple[pd.DataFrame, dict]],
    out_dir: Path,
    bin_s: float = 1.0,
) -> None:
    """All conditions on one axis: mean latency vs time."""
    conditions = [c for c in CONDITION_STYLE if c in data and not data[c][0].empty]
    if not conditions:
        return

    fig, ax = plt.subplots(figsize=(6, 3))

    for cond in conditions:
        df, run_cfg = data[cond]
        ts  = _timeseries(df, bin_s)
        sty = CONDITION_STYLE[cond]
        if ts.empty:
            continue
        ax.plot(ts["t_center"], ts["mean_ms"],
                color=sty["color"], lw=1.4, label=sty["label"],
                zorder=sty["zorder"])

    # Event vlines from first available condition (faint — many events per-task)
    _, run_cfg0 = data[conditions[0]]
    duration = float(run_cfg0.get("experiment", {}).get("duration", 400))
    events   = _events_from_run_cfg(run_cfg0)
    added: set = set()
    for ev in events:
        t = float(ev["t"])
        act = ev["action"]
        if t > duration:
            continue
        st    = EVENT_STYLE[act]
        label = act if act not in added else None
        added.add(act)
        ax.axvline(t, color=st["color"], ls=st["ls"], lw=st["lw"],
                   alpha=0.35, label=label, zorder=0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean response time (ms)")
    ax.set_xlim(left=0, right=duration)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=2, fontsize=6)
    fig.tight_layout()
    _save(fig, out_dir / "response_time_comparison.pdf")
    plt.close(fig)


# ── Plot 2: throughput with offered load ──────────────────────────────────────

def plot_throughput_timeseries(
    data: dict[str, tuple[pd.DataFrame, dict]],
    full_trace: list,
    out_dir: Path,
    bin_s: float = 5.0,
) -> None:
    """Per-condition subplots showing actual throughput vs offered load.

    The shaded gap between offered (dashed) and actual (solid) is the
    cold-start penalty made visually explicit.
    """
    conditions = [c for c in CONDITION_STYLE if c in data and not data[c][0].empty]
    if not conditions:
        return

    n = len(conditions)
    fig, axes = plt.subplots(n, 1, figsize=(5.5, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        df, run_cfg  = data[cond]
        ts           = _timeseries(df, bin_s)
        sty          = CONDITION_STYLE[cond]
        duration     = float(run_cfg.get("experiment", {}).get("duration", 400))

        # Offered load from pre-generated trace (same for all conditions)
        offered = _offered_from_trace(full_trace, bin_s, duration)

        # Shaded unserved gap
        if not offered.empty and not ts.empty:
            thr = dict(zip(ts["t_center"].round(2), ts["throughput_rps"]))
            actual = np.array([thr.get(round(t, 2), 0.0)
                               for t in offered["t_center"]])
            ax.fill_between(
                offered["t_center"], actual, offered["offered_rps"],
                where=offered["offered_rps"] > actual,
                alpha=0.18, color=sty["color"], label="Unserved",
            )

        # Offered load (noisy — binned from trace req_times)
        if not offered.empty:
            ax.plot(
                offered["t_center"], offered["offered_rps"],
                color="black", lw=1.0, alpha=0.65, label="Offered",
                zorder=2,
            )

        # Actual throughput (completions)
        if not ts.empty:
            ax.plot(
                ts["t_center"], ts["throughput_rps"],
                color=sty["color"], lw=1.4, label="Completed",
                zorder=3,
            )

        _draw_events(ax, _events_from_run_cfg(run_cfg), duration, alpha=0.4)

        ax.set_ylabel("Req / s")
        ax.set_title(sty["label"], pad=3)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, ncol=3, fontsize=6)
        ax.grid(axis="y")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout(h_pad=1.0)
    _save(fig, out_dir / "throughput_timeseries.pdf")
    plt.close(fig)


# ── Plot 3: goodput ratio ─────────────────────────────────────────────────────

def plot_goodput_ratio(
    data: dict[str, tuple[pd.DataFrame, dict]],
    full_trace: list,
    out_dir: Path,
    bin_s: float = 5.0,
) -> None:
    """Fraction of offered load actually completed per time bin.

    1.0 = perfect serving.  Drops toward 0 during cold-start windows.
    This is the single clearest signal separating FMaaS from no_sharing.
    """
    conditions = [c for c in CONDITION_STYLE if c in data and not data[c][0].empty]
    if not conditions:
        return

    fig, ax = plt.subplots(figsize=(5.5, 2.5))

    for cond in conditions:
        df, run_cfg = data[cond]
        ts          = _timeseries(df, bin_s)
        sty         = CONDITION_STYLE[cond]
        duration    = float(run_cfg.get("experiment", {}).get("duration", 400))

        offered = _offered_from_trace(full_trace, bin_s, duration)
        if offered.empty:
            continue

        if not ts.empty:
            thr = dict(zip(ts["t_center"].round(2), ts["throughput_rps"]))
        else:
            thr = {}
        actual  = np.array([thr.get(round(t, 2), 0.0)
                             for t in offered["t_center"]])
        offered_arr = offered["offered_rps"].to_numpy()

        # Avoid division by zero in bins with no offered load
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(offered_arr > 0, actual / offered_arr, np.nan)

        mask = ~np.isnan(ratio)
        ax.plot(
            offered["t_center"].to_numpy()[mask],
            np.clip(ratio[mask], 0, 1.5),
            color=sty["color"], lw=1.4, label=sty["label"],
            zorder=sty["zorder"],
        )

    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5, label="Ideal (1.0)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Goodput ratio  (actual / offered)")
    ax.set_ylim(0, 1.25)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y")
    fig.tight_layout()
    _save(fig, out_dir / "goodput_ratio.pdf")
    plt.close(fig)


# ── Plot 4: activation latency ────────────────────────────────────────────────

def plot_activation_latency(
    data: dict[str, tuple[pd.DataFrame, dict]],
    task_timeline: dict,
    out_dir: Path,
    results_dir: Path = None,
) -> None:
    """Bar chart: time from task arrive event to first completed request.

    One cluster of bars per dynamically-arrived task, coloured by condition.
    """
    conditions = [c for c in CONDITION_STYLE if c in data and not data[c][0].empty]
    if not conditions:
        return

    # Derive initially-active tasks from the first available run_config
    # (the initial_tasks list written by run.py — avoids hardcoded 5-second threshold).
    _, run_cfg0 = data[conditions[0]]
    initial_task_set = set(run_cfg0.get("initial_tasks", []))

    dynamic_tasks = sorted(
        (t for t in task_timeline if t not in initial_task_set),
        key=lambda t: float(task_timeline[t]["arrive"]),
    )
    if not dynamic_tasks:
        print("[plot] no dynamically-arrived tasks — skipping activation_latency plot")
        return

    # Collect {task: {condition: latency_s}}
    # Prefer activation_ready.json (t_ready - t_arrive = true attach/deploy time).
    # Fall back to first-completion minus arrive (confounded by trace sparsity).
    per_task: dict[str, dict] = {t: {} for t in dynamic_tasks}
    for cond in conditions:
        df, run_cfg = data[cond]
        cdir = results_dir / cond
        ar_path = (results_dir / cond / "activation_ready.json") if results_dir else cdir / "activation_ready.json"
        if ar_path.is_file():
            ar = json.loads(ar_path.read_text())
            for task in dynamic_tasks:
                if task in ar:
                    per_task[task][cond] = ar[task]["latency_s"]
                else:
                    per_task[task][cond] = None
        else:
            init_set = set(run_cfg.get("initial_tasks", []))
            act      = _activation_latency_per_task(df, task_timeline, init_set)
            for task, lat in act.items():
                if task in per_task:
                    per_task[task][cond] = lat

    x      = np.arange(len(dynamic_tasks))
    n_cond = len(conditions)
    width  = 0.7 / n_cond

    fig, ax = plt.subplots(figsize=(max(4, 0.9 * len(dynamic_tasks)), 3))

    for ci, cond in enumerate(conditions):
        sty  = CONDITION_STYLE[cond]
        vals = [per_task[t].get(cond) for t in dynamic_tasks]
        bars = ax.bar(
            x + (ci - n_cond / 2 + 0.5) * width,
            [v if v is not None else 0 for v in vals],
            width=width * 0.9,
            color=sty["color"],
            label=sty["label"],
            zorder=2,
        )
        for bar, v in zip(bars, vals):
            if v is None:
                continue
            txt = f"{v:.1f}s" if v >= 1 else f"{v * 1000:.0f}ms"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    txt, ha="center", va="bottom", fontsize=5.0)

    # X-tick labels: task name + arrive time
    xlabels = [
        f"{t}\n(t={task_timeline[t]['arrive']:.0f}s)"
        for t in dynamic_tasks
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=5.5)
    ax.set_ylabel("Activation latency (s)")
    ax.set_title("Time from Arrive Event to First Completed Request")
    ax.legend(frameon=False, ncol=n_cond)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out_dir / "activation_latency.pdf")
    plt.close(fig)


# ── Plot 5: latency timeseries ────────────────────────────────────────────────

def plot_latency_timeseries(
    data: dict[str, tuple[pd.DataFrame, dict]],
    out_dir: Path,
    bin_s: float = 5.0,
) -> None:
    """P50 / P99 latency vs time, one subplot per condition."""
    conditions = [c for c in CONDITION_STYLE if c in data and not data[c][0].empty]
    if not conditions:
        return

    n = len(conditions)
    fig, axes = plt.subplots(n, 1, figsize=(5, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        df, run_cfg = data[cond]
        ts          = _timeseries(df, bin_s)
        sty         = CONDITION_STYLE[cond]
        duration    = float(run_cfg.get("experiment", {}).get("duration", 400))
        events      = _events_from_run_cfg(run_cfg)

        if not ts.empty:
            ax.plot(ts["t_center"], ts["p50_ms"],
                    color=sty["color"], lw=1.2, label="P50")
            ax.plot(ts["t_center"], ts["p99_ms"],
                    color=sty["color"], lw=0.8, ls="--", alpha=0.7, label="P99")

        _draw_events(ax, events, duration, alpha=0.3)

        arrive_patch = mpatches.Patch(
            facecolor="none", edgecolor=EVENT_STYLE["arrive"]["color"],
            linestyle="--", linewidth=0.8, label="arrive")
        depart_patch = mpatches.Patch(
            facecolor="none", edgecolor=EVENT_STYLE["depart"]["color"],
            linestyle=":", linewidth=0.8, label="depart")

        ax.set_ylabel("Latency (ms)")
        ax.set_title(sty["label"], pad=3)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.grid(axis="y")
        leg_lines = ax.legend(loc="upper right", frameon=False, ncol=2)
        ax.add_artist(leg_lines)
        ax.legend(handles=[arrive_patch, depart_patch],
                  loc="upper left", frameon=False, ncol=2)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout(h_pad=1.0)
    _save(fig, out_dir / "latency_timeseries.pdf")
    plt.close(fig)


# ── Plot 6: per-task latency breakdown ────────────────────────────────────────

def plot_per_task_latency(
    data: dict[str, tuple[pd.DataFrame, dict]],
    task_meta: dict,
    out_dir: Path,
    bin_s: float = 10.0,
) -> None:
    """Per-task P50 latency timeseries, one subplot per condition."""
    conditions = [c for c in CONDITION_STYLE if c in data and not data[c][0].empty]
    if not conditions:
        return

    base_tasks  = sorted(set(task_meta.values()))
    cmap        = plt.get_cmap("tab10")
    task_color  = {bt: cmap(i % 10) for i, bt in enumerate(base_tasks)}

    n = len(conditions)
    fig, axes = plt.subplots(n, 1, figsize=(6, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        df, run_cfg = data[cond]
        sty         = CONDITION_STYLE[cond]
        duration    = float(run_cfg.get("experiment", {}).get("duration", 400))
        events      = _events_from_run_cfg(run_cfg)

        t_max  = float(df["t_sec"].max())
        edges  = np.arange(0.0, t_max + bin_s, bin_s)
        centers = edges[:-1] + bin_s / 2.0
        bins   = pd.cut(df["t_sec"], edges, right=False, include_lowest=True)
        bin_to_center = dict(zip(bins.cat.categories, centers))
        df = df.copy()
        df["bin"] = bins

        plotted_bases: set = set()
        for task, grp in df.groupby("task"):
            base  = task_meta.get(task, task)
            color = task_color.get(base, "gray")
            label = base if base not in plotted_bases else None
            plotted_bases.add(base)

            ts_rows = []
            for b, sg in grp.groupby("bin", observed=True):
                if sg.empty:
                    continue
                lat = sg["end_to_end_latency(ms)"].to_numpy()
                ts_rows.append({
                    "t_center": float(bin_to_center[b]),
                    "p50_ms":   float(np.percentile(lat, 50)),
                })
            if not ts_rows:
                continue
            ts_df = pd.DataFrame(ts_rows).sort_values("t_center")
            ax.plot(ts_df["t_center"], ts_df["p50_ms"],
                    color=color, lw=0.9, alpha=0.75, label=label)

        _draw_events(ax, events, duration, alpha=0.25)
        ax.set_ylabel("P50 latency (ms)")
        ax.set_title(sty["label"], pad=3)
        ax.legend(frameon=False, ncol=3, loc="upper right")
        ax.grid(axis="y")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout(h_pad=1.0)
    _save(fig, out_dir / "per_task_latency.pdf")
    plt.close(fig)


def _batch_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-batch view: rows are unique (device, device_start_time) — i.e. one
    forward pass — with batch_size and backend_exec_time(ms)."""
    g = df.groupby(["device", "device_start_time"])
    return pd.DataFrame({
        "backbone":   g["backbone"].first(),
        "batch_size": g.size(),
        "exec_ms":    g["backend_exec_time(ms)"].first(),
    }).reset_index()


def plot_batch_sizes(
    data: dict[str, tuple[pd.DataFrame, dict]],
    out_dir: Path,
) -> None:
    """Four-panel batch-size diagnostic:
        a. Batch-size distribution per backbone × condition (boxplot).
        b. Mean / p95 batch-size bars per backbone × condition.
        c. backend_exec_time vs batch_size scatter, per backbone × condition.
        d. Queue wait (client_submit_to_backend_start) p95 per backbone × condition.
    """
    if not data:
        return

    # Collect per-batch tables
    batch_by_cond = {c: _batch_table(df) for c, (df, _) in data.items()}
    # Backbones present in any condition, ordered by total request volume
    bb_order = (
        pd.concat([df.assign(c=c) for c, (df, _) in data.items()], ignore_index=True)
          .groupby("backbone").size().sort_values(ascending=False).index.tolist()
    )
    conds = list(data.keys())

    # ── (a) Boxplot of batch sizes per backbone × condition ───────────────────
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(bb_order) * len(conds)), 4.0))
    width = 0.8 / max(1, len(conds))
    x_base = np.arange(len(bb_order))
    for i, c in enumerate(conds):
        bt = batch_by_cond[c]
        groups = [bt.loc[bt["backbone"] == bb, "batch_size"].values for bb in bb_order]
        positions = x_base + (i - (len(conds) - 1) / 2) * width
        bp = ax.boxplot(
            groups, positions=positions, widths=width * 0.9,
            patch_artist=True, showfliers=False, whis=(5, 95),
        )
        color = CONDITION_STYLE[c]["color"]
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.55, edgecolor=color, linewidth=0.6)
        for med in bp["medians"]:
            med.set(color="black", linewidth=1.0)
        for whisk in bp["whiskers"] + bp["caps"]:
            whisk.set(color=color, linewidth=0.6)
    ax.set_xticks(x_base)
    ax.set_xticklabels(bb_order, rotation=30, ha="right")
    ax.set_ylabel("Batch size (per forward pass)")
    ax.set_title("Effective batch-size distribution (5–95 pct, median = black)", pad=4)
    ax.grid(axis="y")
    handles = [mpatches.Patch(color=CONDITION_STYLE[c]["color"], alpha=0.55,
                              label=CONDITION_STYLE[c]["label"]) for c in conds]
    ax.legend(handles=handles, frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, out_dir / "batch_size_distribution.pdf")
    plt.close(fig)

    # ── (b) Mean & p95 batch-size bars ────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.45 * len(bb_order)), 5.5),
                             sharex=True)
    x_base = np.arange(len(bb_order))
    bw = 0.8 / max(1, len(conds))
    for stat_idx, (stat_name, stat_fn) in enumerate(
        [("Mean batch size", np.mean), ("p95 batch size", lambda v: np.percentile(v, 95) if len(v) else np.nan)]
    ):
        ax = axes[stat_idx]
        for i, c in enumerate(conds):
            bt = batch_by_cond[c]
            vals = [stat_fn(bt.loc[bt["backbone"] == bb, "batch_size"].values)
                    if (bt["backbone"] == bb).any() else np.nan for bb in bb_order]
            ax.bar(x_base + (i - (len(conds) - 1) / 2) * bw, vals, bw * 0.95,
                   color=CONDITION_STYLE[c]["color"], alpha=0.85,
                   label=CONDITION_STYLE[c]["label"])
        ax.set_ylabel(stat_name)
        ax.grid(axis="y")
        if stat_idx == 0:
            ax.legend(frameon=False, loc="upper right")
    axes[-1].set_xticks(x_base)
    axes[-1].set_xticklabels(bb_order, rotation=30, ha="right")
    fig.tight_layout(h_pad=0.8)
    _save(fig, out_dir / "batch_size_summary.pdf")
    plt.close(fig)

    # ── (c) exec_ms vs batch_size scatter, per backbone ───────────────────────
    n = len(bb_order)
    ncol = min(4, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow),
                             squeeze=False)
    for k, bb in enumerate(bb_order):
        ax = axes[k // ncol][k % ncol]
        for c in conds:
            bt = batch_by_cond[c]
            sub = bt[bt["backbone"] == bb]
            if sub.empty:
                continue
            ax.scatter(
                sub["batch_size"], sub["exec_ms"],
                s=6, alpha=0.25, color=CONDITION_STYLE[c]["color"],
                label=CONDITION_STYLE[c]["label"], edgecolors="none",
            )
        ax.set_title(bb, pad=2, fontsize=9)
        ax.set_yscale("log")
        ax.grid(axis="both", which="both", alpha=0.4)
        if k % ncol == 0:
            ax.set_ylabel("exec (ms, log)")
        if k // ncol == nrow - 1:
            ax.set_xlabel("batch size")
    # Hide unused
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].set_visible(False)
    # Single shared legend
    handles = [mpatches.Patch(color=CONDITION_STYLE[c]["color"], alpha=0.7,
                              label=CONDITION_STYLE[c]["label"]) for c in conds]
    fig.legend(handles=handles, frameon=False, loc="upper center",
               ncol=len(conds), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    _save(fig, out_dir / "batch_size_vs_exec.pdf")
    plt.close(fig)

    # ── (d) Queue wait p50/p95 per backbone × condition ───────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.45 * len(bb_order)), 5.5),
                             sharex=True)
    x_base = np.arange(len(bb_order))
    for stat_idx, (stat_name, q) in enumerate([("Queue wait p50 (ms)", 0.50),
                                                ("Queue wait p95 (ms)", 0.95)]):
        ax = axes[stat_idx]
        for i, c in enumerate(conds):
            df, _ = data[c]
            vals = []
            for bb in bb_order:
                sub = df.loc[df["backbone"] == bb, "client_submit_to_backend_start(ms)"]
                vals.append(sub.quantile(q) if len(sub) else np.nan)
            ax.bar(x_base + (i - (len(conds) - 1) / 2) * bw, vals, bw * 0.95,
                   color=CONDITION_STYLE[c]["color"], alpha=0.85,
                   label=CONDITION_STYLE[c]["label"])
        ax.set_ylabel(stat_name)
        ax.grid(axis="y")
        if stat_idx == 0:
            ax.legend(frameon=False, loc="upper right")
    axes[-1].set_xticks(x_base)
    axes[-1].set_xticklabels(bb_order, rotation=30, ha="right")
    fig.tight_layout(h_pad=0.8)
    _save(fig, out_dir / "queue_wait_per_backbone.pdf")
    plt.close(fig)


# ── Plot: placement capacity ──────────────────────────────────────────────────

def plot_placement_capacity(
    task_timeline: dict,
    slots_by_cond: dict[str, list],
    out_dir: Path,
    duration: float = 1800.0,
) -> None:
    """Supported-task count over time, per placement method.

    Three lines:
      - Active tasks (demand): dashed black — how many tasks exist at each t.
      - FMaaS: how many placed tasks are active at each t (should match demand).
      - No-Sharing: same, but saturates when memory runs out.
    """
    if not task_timeline:
        return

    # Demand: every task in the timeline
    demand_ivs = [
        (float(v["arrive"]), float(v["depart"]))
        for v in task_timeline.values()
    ]

    # Per-condition: extract placed task names from slots
    cond_data = {}  # cond -> (intervals, n_placed)
    for cond, slots in slots_by_cond.items():
        sty = CONDITION_STYLE.get(cond)
        if sty is None:
            continue
        placed = set()
        if cond == "fmaas":
            for s in slots:
                for t in s.get("tasks", []):
                    placed.add(t["task"] if isinstance(t, dict) else t)
        else:
            for s in slots:
                if "task" in s:
                    placed.add(s["task"])
        ivs = [
            (float(task_timeline[t]["arrive"]), float(task_timeline[t]["depart"]))
            for t in placed if t in task_timeline
        ]
        cond_data[cond] = (ivs, len(placed))

    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    # Demand
    t, dem = _step_over_time(demand_ivs, duration)
    ax.step(t, dem, where="post", color="black", lw=1.0, ls="--",
            alpha=0.6, label="Active tasks (demand)", zorder=4)

    # Per-condition
    for cond in ["fmaas", "no_sharing", "no_sharing_tpc"]:
        if cond not in cond_data:
            continue
        sty = CONDITION_STYLE[cond]
        ivs, n_placed = cond_data[cond]
        t_c, cnt = _step_over_time(ivs, duration)
        ax.step(t_c, cnt, where="post",
                color=sty["color"], lw=1.6,
                label=f"{sty['label']} ({n_placed} placed)",
                zorder=sty["zorder"])

        # Shade gap for no_sharing
        if "no_sharing" in cond:
            ax.fill_between(t, cnt[:len(t)], dem,
                            where=dem > cnt[:len(t)], step="post",
                            alpha=0.10, color=sty["color"])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Number of tasks")
    ax.set_xlim(0, duration)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, out_dir / "placement_capacity.pdf")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="experiments/long_horizon/results",
                        help="Directory containing fmaas/, no_sharing/, etc.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for PDFs (default: results-dir/plots)")
    parser.add_argument("--bin-s", type=float, default=5.0,
                        help="Bin width in seconds for timeseries plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir) if args.out_dir else results_dir / "plots"

    deploy_dir    = Path(__file__).resolve().parent / "deployments"
    task_timeline: dict = {}
    task_meta:     dict = {}
    full_trace:    list = []
    slots_by_cond: dict = {}

    tt_path  = deploy_dir / "task_timeline.json"
    tm_path  = deploy_dir / "task_meta.json"
    tr_path  = deploy_dir / "trace.json"
    if tt_path.is_file():
        task_timeline = json.loads(tt_path.read_text())
        print(f"[plot] loaded task_timeline: {len(task_timeline)} tasks")
    else:
        print("[plot] WARNING: task_timeline.json not found")
    if tm_path.is_file():
        task_meta = json.loads(tm_path.read_text())
    if tr_path.is_file():
        full_trace = json.loads(tr_path.read_text())
        print(f"[plot] loaded trace: {len(full_trace)} requests (offered load)")
    else:
        print("[plot] WARNING: trace.json not found — run generate.py first")
    for cond in CONDITION_STYLE:
        sp = deploy_dir / f"{cond}_slots.json"
        if sp.is_file():
            slots_by_cond[cond] = json.loads(sp.read_text())

    _paper_style()

    data: dict[str, tuple[pd.DataFrame, dict]] = {}
    for cond in CONDITION_STYLE:
        df, run_cfg = _load_condition(results_dir, cond)
        if not df.empty:
            df = _anchor_time(df, run_cfg)
            data[cond] = (df, run_cfg)
            print(f"[plot] loaded {cond}: {len(df)} requests")
        else:
            print(f"[plot] {cond}: no data, skipping")

    if not data:
        print("[plot] no data found — nothing to plot")
        return

    # Fallback: derive task_timeline from run_config if generate.py hasn't been run
    if not task_timeline and data:
        _, run_cfg0 = next(iter(data.values()))
        task_timeline = run_cfg0.get("task_timeline", {})

    duration = 400.0
    if data:
        _, run_cfg0 = next(iter(data.values()))
        duration = float(run_cfg0.get("experiment", {}).get("duration", 400))
    elif task_timeline:
        duration = max(float(v.get("depart", 400)) for v in task_timeline.values())

    plot_deployment_activity(task_timeline, slots_by_cond, out_dir, duration=duration)
    plot_placement_capacity(task_timeline, slots_by_cond, out_dir, duration=duration)
    plot_gpu_memory_footprint(data, out_dir, bin_s=args.bin_s)
    plot_rps_per_task(full_trace, data, task_meta, out_dir, duration=duration, bin_s=args.bin_s)
    plot_response_time_comparison(data, out_dir, bin_s=1.0)
    plot_throughput_timeseries(data, full_trace, out_dir, bin_s=args.bin_s)
    plot_goodput_ratio(data, full_trace, out_dir, bin_s=args.bin_s)
    plot_activation_latency(data, task_timeline, out_dir, results_dir=results_dir)
    plot_latency_timeseries(data, out_dir, bin_s=args.bin_s)
    plot_per_task_latency(data, task_meta, out_dir, bin_s=args.bin_s * 2)
    plot_batch_sizes(data, out_dir)

    print(f"[plot] done — figures in {out_dir}")


if __name__ == "__main__":
    main()
