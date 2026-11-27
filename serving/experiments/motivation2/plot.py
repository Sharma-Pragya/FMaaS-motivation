#!/usr/bin/env python3
"""motivation2/plot.py — the motivation figure from the device-server runs
(run.sh / run.py), laid out as memory_isolation_vision/motivation_plot.py:

    (a) Efficiency   efficiency/summary.csv: GPU memory (bars) and throughput
                     (lines) vs #tasks, partitioned vs shared
    (b) Perf. iso.   victims' completed req/s over time
    (c) Mem. iso.    noisy task B's GPU memory over time
    (d) Fault iso.   victims' failed requests (cumulative)

(c) B's GPU memory in use, from the device server's Control("stats"),
polled every 0.5 s (isolation/mem_<condition>.jsonl, device/memstats.py).
A CUDA context is charged to B only where it exists for B alone:
    mig         B's server: the slice's used memory (cudaMemGetInfo), its
                context, weights and queue included.
    no_sharing  B's server: torch reserved (the GPU is shared by both
                servers, so the device's used memory is not B's alone).
    shared(_bfq) one server for every task, so its torch reserved memory is
                split: its mean before the ramp evenly, 1/N per task; what it
                grows above that by each task's queued bytes (inputs on the
                GPU, staging=gpu). Its context is the system's.
    fmvisor     B's slots in use x slot size (its part of its pool, in the
                backbone process) + B's task process (torch reserved) + that
                process's context, estimated as the device's used memory not
                in any process's torch reserved, split evenly over the
                processes. Its quota (whole pool + the same) is drawn dashed.
Runs without a mem_<condition>.jsonl fall back to the gpu_alloc_peak_mb of
completed requests (torch allocated, after each batch; B's split as above
with every byte above the base charged to B), which stops when nothing
completes.
(a) GPU memory is the device's used memory (CUDA contexts included) averaged
over the run (avg_device_used_mb), or, for older rows, the sum of each
server's torch allocated (contexts left out).

Writes into --exp-dir:
    motivation2       (a)-(d), Partitioned vs Shared: the motivation figure
    motivation2_all   (b)-(d), every isolation condition that was run
                      (Partitioned, Shared, Shared + BFQ, FMVisor)
    motivation2_tput, motivation2_tput_all
                      the same, laid out as (a) Efficiency, (b) noisy task B's
                      throughput, (c) the
                      victims' throughput, (d) noisy task B's memory; no fault
                      panel
    rough_memory      rough, not for the paper: per condition, noisy B, the
                      victim, everything else, and the device's used memory
(a)'s Partitioned is mig_tpc if it was run, else mig, else partitioned.
--oom condition:n,... draws (a) points as OOM that left no row (e.g. the
memory cap killed the run's client rather than a server).

Usage (from serving/):
    python experiments/motivation2/plot.py --exp-dir experiments/motivation2/results \\
        [--oom mig_tpc:24]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

EXP_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "mem_plot", EXP_DIR.parent / "memory_isolation_vision" / "plot.py")
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)            # rate_per_bin, styles

# Conditions -> memory_isolation_vision styles (mig, s_be, be), plus Shared + BFQ.
mp.COLORS["s_bfq"], mp.STYLES["s_bfq"], mp.MARKERS["s_bfq"] = "#7B4F9E", "--", "s"
STYLE_OF = {"mig": "mig", "no_sharing": "be", "shared": "s_be", "shared_bfq": "s_bfq",
            "fmvisor": "fmvisor",
            "partitioned": None}
LABELS = {"mig": "Partitioned", "no_sharing": "Partitioned", "shared": "Shared",
          "shared_bfq": "Shared + BFQ", "fmvisor": "FMVisor"}
FIG_W, FIG_H = 7.0, 1.6
LW, MS, N_MARKERS = 1.3, 3.5, 6
GRID_S = 0.5
HEADROOM = 1.5      # time panels' y-limit over their peak: room for the event notes
RED = mp.SURGE_COLOR


def style(key: str, n_points: int) -> dict:
    return dict(color=mp.COLORS[key], ls=mp.STYLES[key], lw=LW, marker=mp.MARKERS[key],
                markersize=MS, markevery=max(1, n_points // N_MARKERS))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_events(path: Path) -> list:
    rows = []
    for r in csv.DictReader(open(path)):
        r["t_send"], r["t_done"] = float(r["t_send"]), float(r["t_done"])
        r["ok"] = r["ok"] == "1"
        r["mem_mb"] = float(r["mem_mb"]) if r["mem_mb"] else None
        r["mem_t"] = float(r["mem_t"]) if r["mem_t"] else None
        rows.append(r)
    return rows


def memory_series(ev: list, url: str, dur: float) -> tuple:
    """(grid, MB) of one server: max sample per GRID_S bin, gaps interpolated,
    NaN before the first and after the last sample."""
    pts = sorted((r["mem_t"], r["mem_mb"]) for r in ev
                 if r["url"] == url and r["ok"] and r["mem_mb"] is not None)
    grid = np.arange(0.0, dur, GRID_S)
    if not pts:
        return grid, np.full_like(grid, np.nan)
    t = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    b = np.floor(t / GRID_S).astype(int)
    keep = (b >= 0) & (b < len(grid))
    peak = np.full(len(grid), -np.inf)
    np.maximum.at(peak, b[keep], y[keep])
    have = np.isfinite(peak)
    if not have.any():
        return grid, np.full_like(grid, np.nan)
    out = np.interp(grid, grid[have], peak[have], left=np.nan, right=np.nan)
    return grid, out


def noisy_memory_gb(ev: list, s: dict) -> tuple:
    """(grid, B's GB) — see the module docstring."""
    urls = s["task_urls"]
    b_url = urls[s["surging"]]
    grid, y = memory_series(ev, b_url, s["duration"])
    sharers = [t for t, u in urls.items() if u == b_url]
    if len(sharers) > 1:
        pre = y[(grid >= 5.0) & (grid < s["ramp_at"]) & np.isfinite(y)]
        base = float(pre.mean()) if len(pre) else 0.0
        y = base / len(sharers) + np.maximum(0.0, y - base)
    return grid, y / 1024


# A request lost to a failure: a CUDA OOM error, or its server (process) gone
# after one (the server exits on OOM): the stream broke, or the front end
# reports the task's / backbone's process down.
FAILURE_CODES = ("out of memory", "UNAVAILABLE", "INTERNAL", "STREAM_ENDED", "CANCELLED",
                 "task_down", "backbone_down")


CRASH_CODES = FAILURE_CODES[1:]     # the server (process) is gone, not just a failed request


def first_crash(ev: list, tasks: set) -> float | None:
    """When the process serving tasks went down (it exits on OOM)."""
    ts = [r["t_done"] for r in ev
          if not r["ok"] and r["task"] in tasks
          and any(c in (r.get("code") or "") for c in CRASH_CODES)]
    return min(ts) if ts else None


def drop_at_crash(t, y, t_crash: float | None, dur: float) -> tuple:
    """A crashed process holds no memory: the line drops straight to 0 at
    the crash and stays there (as memory_isolation_vision's task_memory_gb)."""
    if t_crash is None:
        return t, y
    t, y = np.asarray(t, float), np.asarray(y, float)
    keep = t < t_crash
    last = y[keep][-1] if keep.any() else 0.0
    return (np.concatenate([t[keep], [t_crash, t_crash, dur]]),
            np.concatenate([y[keep], [last, 0.0, 0.0]]))


def first_oom(ev: list, tasks: set) -> float | None:
    """When any of tasks first lost a request to an OOM failure."""
    ts = [r["t_done"] for r in ev
          if not r["ok"] and r["task"] in tasks
          and any(c in (r.get("code") or "") for c in FAILURE_CODES)]
    return min(ts) if ts else None


def mark_ooms(ax, ooms: dict, dur: float, merge_s: float = 6.0) -> None:
    """Each run's OOM failure as an event, like the noisy ramp: a vertical
    line at its time, in the run's colour, and a red "failure" note, as
    " noisy: a→b rps" is written. ooms: {condition: t}. Runs failing within
    merge_s of each other share one note."""
    groups: list = []                  # [[t, [conditions]]]
    for c, t in sorted(ooms.items(), key=lambda kv: kv[1]):
        ax.axvline(t, color=mp.COLORS[STYLE_OF[c]], lw=0.7, ls="--", zorder=1)
        if groups and t - groups[-1][0] < merge_s:
            groups[-1][1].append(c)
        else:
            groups.append([t, [c]])
    tr = ax.get_xaxis_transform()      # x in data, y in axes fraction
    renderer = ax.figure.canvas.get_renderer()
    placed: list = []                  # (x0, x1, row) in data units
    for t, cs in groups:
        # In the headroom above the curves (HEADROOM), below the noisy note:
        # right of the line unless that leaves the panel, and one row lower
        # than any note it would overlap.
        txt = ax.text(t + 1, 0.87, "failure", transform=tr, color=RED,
                      fontsize=5.5, va="top", ha="left", zorder=4)
        bb = txt.get_window_extent(renderer)
        (x0, _), (x1, _) = ax.transData.inverted().transform([(bb.x0, 0), (bb.x1, 0)])
        w = x1 - x0
        if t + 1 + w > ax.get_xlim()[1]:
            txt.set_x(t - 1)
            txt.set_horizontalalignment("right")
            x0, x1 = t - 1 - w, t - 1
        else:
            x0, x1 = t + 1, t + 1 + w
        row = 0
        while any(r == row and a < x1 and x0 < b for a, b, r in placed):
            row += 1
        txt.set_y(0.87 - 0.1 * row)
        placed.append((x0, x1, row))


def load_stats(path: Path) -> list:
    return [json.loads(l) for l in open(path)] if path.exists() else []


def noisy_memory_from_stats(rows: list, s: dict, task: str | None = None) -> tuple:
    """(t, GB, quota GB or None) of one task (default the surging task B)
    from Control("stats") samples (see the module docstring)."""
    b, urls = task or s["surging"], s["task_urls"]
    b_url = urls[b]
    sharers = [t for t, u in urls.items() if u == b_url]
    rows = sorted((r for r in rows if r["url"] == b_url), key=lambda r: r["t"])
    t, y, quota = [], [], []
    if rows and rows[0].get("mode") == "shared" and len(sharers) > 1:
        res = np.array([r["reserved_mb"] for r in rows])
        tt = np.array([r["t"] for r in rows])
        pre = res[(tt >= 5.0) & (tt < s["ramp_at"])]
        base = float(pre.mean()) if len(pre) else 0.0
        for r, v in zip(rows, res):
            q = {k: d.get("queued_mb", 0.0) for k, d in r.get("tasks", {}).items()}
            qb, qall = q.get(b, 0.0), sum(q.values())
            frac = qb / qall if qall > 0 else 1.0 / len(sharers)
            t.append(r["t"])
            y.append(base / len(sharers) + max(0.0, v - base) * frac)
        return np.array(t), np.array(y) / 1024, None
    for r in rows:
        if r.get("mode") == "process":
            tb = r.get("tasks", {}).get(b)
            if tb is None or (tb.get("age_s") or 0) > 5:      # B's process is gone
                continue
            procs = 1 + len(r["tasks"])
            in_torch = r["backbone"]["reserved_mb"] + sum(d["reserved_mb"] for d in r["tasks"].values())
            ctx = max(0.0, r["device_used_mb"] - in_torch) / procs
            t.append(r["t"])
            y.append(tb["pool_used_mb"] + tb["reserved_mb"] + ctx)
            quota.append(tb["pool_mb"] + tb["reserved_mb"] + ctx)
        elif s["condition"] == "mig":
            t.append(r["t"])
            y.append(r["device_used_mb"])
        else:
            t.append(r["t"])
            y.append(r["reserved_mb"])
    q = float(np.median(quota)) / 1024 if quota else None
    return np.array(t), np.array(y) / 1024, q


def latest_summaries(path: Path) -> dict:
    out = {}
    for line in open(path):
        s = json.loads(line)
        out[s["condition"]] = s             # last line of each condition wins
    return out


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def efficiency_panel(ax, path: Path, part_key: str, oom: dict | None = None) -> None:
    """oom: {condition: {n, ...}} drawn as OOM where the run left no row (e.g.
    the memory cap killed the client, not a server; see --oom)."""
    data: dict = {}
    for r in csv.DictReader(open(path)):
        data.setdefault(r["condition"], {})[int(r["n_tasks"])] = r   # latest row wins
    for cond, ns_oom in (oom or {}).items():
        for n in ns_oom:
            data.setdefault(cond, {}).setdefault(n, {"status": "oom"})
    ns = sorted({n for d in data.values() for n in d})
    x = np.arange(len(ns))
    w = 0.38
    ax_t = ax.twinx()
    top_mem, top_thr = 1.0, 1.0
    # Partitioned: one server per task on MIG slices with each slice's TPCs
    # split among its servers (mig_tpc) if that was run, else on MIG slices
    # (mig), else on the whole GPU (partitioned).
    part = next((c for c in ("mig_tpc", "mig", "partitioned") if c in data), "partitioned")
    for i, (cond, key) in enumerate(((part, part_key), ("shared", "s_be"))):
        rows = data.get(cond, {})
        off = (i - 0.5) * w
        ys = []
        for j, n in enumerate(ns):
            r = rows.get(n)
            mem = (r.get("avg_device_used_mb") or r.get("avg_gpu_mem_mb")) if r else None
            mem = mem if r and r.get("status") in ("ok", "partial") else None
            if r and r["status"] in ("ok", "partial") and mem:
                gb = float(mem) / 1024
                top_mem = max(top_mem, gb)
                ax.bar(x[j] + off, gb, w, color=mp.COLORS[key], alpha=0.35,
                       edgecolor=mp.COLORS[key], lw=0.6, zorder=2)
                ys.append(float(r["throughput_rps"]))
            else:
                if r is not None:
                    ax.text(x[j] + off, 0.02, "OOM" if r["status"] in ("oom", "start_failed")
                            else "fail", rotation=90, ha="center", va="bottom",
                            transform=ax.get_xaxis_transform(),
                            color=RED, fontsize=6, fontweight="bold")
                ys.append(np.nan)
        top_thr = max([top_thr] + [v for v in ys if np.isfinite(v)])
        ax_t.plot(x, ys, **{**style(key, len(x)), "markevery": 1}, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("#Tasks", labelpad=1)
    ax.set_ylabel("GPU memory (GB)", labelpad=1)
    ax.set_ylim(0, top_mem * 1.15)
    ax_t.set_ylim(0, top_thr * 1.15)
    ax_t.set_ylabel("Throughput (req/s)", labelpad=1)
    ax_t.spines["right"].set_visible(True)
    ax_t.spines["right"].set_linewidth(0.6)
    ax.yaxis.grid(True, color="#dddddd", lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor="#bbbbbb", edgecolor="#777777", lw=0.6, label="memory"),
                       Line2D([], [], color="#555555", lw=LW, label="throughput")],
              loc="upper left", frameon=False, fontsize=6, handlelength=1.4,
              borderaxespad=0.1, labelspacing=0.2)
    ax.set_title("(a) Efficiency")


RC = {
    "font.family": "sans-serif", "font.size": 7, "axes.labelsize": 7,
    "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "standard",
}


def draw_figure(summ: dict, conds: list, iso_dir: Path, out: Path, thr_bin: float,
                eff_csv: Path | None = None, part_key: str = "mig", oom: dict | None = None,
                fig_w: float = FIG_W, layout: str = "isolation") -> None:
    """(a) if eff_csv, then, for the isolation conditions conds:
        layout "isolation"  (b) victim throughput, (c) noisy task memory,
                            (d) victim failures
        layout "tput"       (b) noisy task throughput, (c) victim throughput,
                            (d) noisy task memory"""
    s0 = summ[conds[0]]
    dur, ramp = s0["duration"], s0["ramp_at"]
    V = set(s0["victims"])
    B = {s0["surging"]}
    edges = np.arange(0, dur + 1e-9, thr_bin)
    mids = edges[:-1] + thr_bin / 2
    tput = layout == "tput"
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(fig_w, FIG_H))
        if eff_csv is not None:
            gs = fig.add_gridspec(1, 5, width_ratios=[1.3, 0.22, 1, 1, 1], wspace=0.42,
                                  left=0.055, right=0.99, top=0.76, bottom=0.21)
            efficiency_panel(fig.add_subplot(gs[0]), eff_csv, part_key, oom)
            cols = (gs[2], gs[3], gs[4])
        else:
            gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.07, right=0.99, top=0.76, bottom=0.21)
            cols = (gs[0], gs[1], gs[2])
        ax1 = fig.add_subplot(cols[0])
        ax2, ax3 = fig.add_subplot(cols[1], sharex=ax1), fig.add_subplot(cols[2], sharex=ax1)
        if tput:
            ax_n, ax_p, ax_m, ax_f = ax1, ax2, ax3, None
        else:
            ax_n, ax_p, ax_m, ax_f = None, ax1, ax2, ax3

        max_fail, max_mem, max_thr, max_nthr = 0, 0.0, 0.0, 0.0
        fails = {}
        # condition -> first CUDA OOM of the noisy task / of a victim. Each
        # panel marks only failures of the task it shows.
        ooms_b: dict = {}
        ooms_v: dict = {}
        mem_fail: list = []                   # (t, GB) where B's server crashed
        for c in conds:
            s, key = summ[c], STYLE_OF[c]
            ev = load_events(iso_dir / f"events_{c}.csv")
            ok_ev = [{"kind": "ok", "tenant": r["task"], "t": r["t_done"]} for r in ev if r["ok"]]
            thr = mp.rate_per_bin(ok_ev, V, edges)
            max_thr = max(max_thr, float(np.nanmax(thr)) if len(thr) else 0.0)
            ax_p.plot(mids, thr, **style(key, len(mids)))
            if ax_n is not None:
                nthr = mp.rate_per_bin(ok_ev, B, edges)
                max_nthr = max(max_nthr, float(np.nanmax(nthr)) if len(nthr) else 0.0)
                ax_n.plot(mids, nthr, **style(key, len(mids)))
            stats = load_stats(iso_dir / f"mem_{c}.jsonl")
            quota = None
            if stats:
                t, yb, quota = noisy_memory_from_stats(stats, s)
            else:
                t, yb = noisy_memory_gb(ev, s)
            peak = float(np.nanmax(yb)) if len(yb) and np.isfinite(yb).any() else 0.0
            max_mem = max(max_mem, peak, quota or 0.0)
            # A run with no memory numbers (process mode before stats): no line,
            # rather than one at 0.
            if peak > 0:
                t_crash = first_crash(ev, B)
                if t_crash is not None:
                    # Memory panel: "failure" written at the top of the peak.
                    ok = np.isfinite(yb) & (np.asarray(t) < t_crash)
                    if ok.any():
                        mem_fail.append((t_crash, float(np.asarray(yb)[ok][-1])))
                t, yb = drop_at_crash(t, yb, t_crash, dur)
                ax_m.plot(t, yb, **style(key, len(t)))
            for tasks, into in ((B, ooms_b), (V, ooms_v)):
                t_oom = first_oom(ev, tasks)
                if t_oom is not None:
                    into[c] = t_oom
            if quota:
                ax_m.axhline(quota, color=mp.COLORS[key], lw=0.6, ls="--")
                ax_m.text(dur - 2, quota, f"{LABELS[c]} quota", color=mp.COLORS[key],
                          fontsize=5.5, va="top", ha="right")
            ft = np.sort([r["t_done"] for r in ev if not r["ok"] and r["task"] in V])
            fails[c] = np.searchsorted(ft, mids, side="right")
            max_fail = max(max_fail, len(ft))

        k = max_fail >= 2000
        if ax_f is not None:
            for c in conds:
                ax_f.plot(mids, fails[c] / (1000 if k else 1), **style(STYLE_OF[c], len(mids)))

        # (c) capacity lines (from the summary; absent if nvidia-smi / NVML had none).
        cap = (s0.get("gpu_total_mb") or summ.get("shared", {}).get("gpu_total_mb") or 0) / 1024
        sl = (summ.get("mig", {}).get("mig_slice_mb") or 0) / 1024
        top = max(max_mem, cap, sl) * 1.15 or 1.0
        if cap:
            ax_m.axhline(cap, color=mp.CAPACITY_COLOR, lw=0.6, ls="--")
            ax_m.text(4, cap, "GPU capacity", color=mp.CAPACITY_COLOR, fontsize=5.5, va="bottom")
        if sl and "mig" in conds:
            ax_m.axhline(sl, color=mp.COLORS["mig"], lw=0.6, ls="--")
            ax_m.text(4, sl, "MIG slice", color=mp.COLORS["mig"], fontsize=5.5, va="bottom")
        ax_m.set_ylim(0, top)
        ax_p.set_ylim(0, max_thr * HEADROOM or 1.0)
        if tput:
            # B's offered load is in the "noisy: a→b rps" note; drawn as a line
            # it would flatten what B is actually served.
            ax_n.set_ylim(0, max_nthr * HEADROOM or 1.0)
            panels = ((ax_n, "(b) Noisy task", "Noisy task tput (req/s)"),
                      (ax_p, "(c) Victim task", "Victim tput (req/s)"),
                      (ax_m, "(d) Memory isolation", "Noisy task mem. (GB)"))
        else:
            ax_f.set_ylim(0, (max_fail / (1000 if k else 1)) * HEADROOM or 1.0)
            panels = ((ax_p, "(b) Performance isolation", "Victim tput (req/s)"),
                      (ax_m, "(c) Memory isolation", "Noisy task mem. (GB)"),
                      (ax_f, "(d) Fault isolation",
                       "Victim failed req." + (" (K)" if k else "")))
        # Memory panel: no failure lines, the word at the peak instead (below).
        oom_of = {ax_n: ooms_b, ax_m: {}, ax_p: ooms_v, ax_f: ooms_v}
        for ax, title, yl in panels:
            if eff_csv is None:             # no (a): the time panels start at (a)
                title = f"({chr(ord(title[1]) - 1)}){title[3:]}"
            ax.set_title(title)
            ax.set_ylabel(yl, labelpad=1)
            ax.set_xlabel("Time (s)", labelpad=1)
            ax.set_xlim(0, dur)
            ax.axvline(ramp, **mp.PHASE_LINE)
            ax.yaxis.grid(True, color="#dddddd", lw=0.4)
            ax.set_axisbelow(True)
            mark_ooms(ax, oom_of[ax], dur)
        for t_c, y_c in mem_fail:
            ax_m.text(t_c, y_c + 0.03 * top, "failure", color=RED, fontsize=5.5,
                      ha="center", va="bottom")
        # The noisy ramp note goes on the first time panel only: the victim
        # panel in the isolation layout, the noisy task's panel in "tput".
        if ax_n is None:
            ax_p.text(ramp, ax_p.get_ylim()[1], f" {s0['rate_b']:g}→{s0['rate_b_max']:g} rps",
                      color=RED, fontsize=5.5, va="top")
        else:
            ax_n.text(ramp, ax_n.get_ylim()[1], f" {s0['rate_b']:g}→{s0['rate_b_max']:g} rps",
                      color=RED, fontsize=5.5, va="top")

        handles = [Line2D([], [], **{**style(STYLE_OF[c], 1), "markevery": None}, label=LABELS[c])
                   for c in conds]
        fig.legend(handles=handles, frameon=False, ncol=len(handles), loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=1.5)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".pdf"))
        fig.savefig(out.with_suffix(".png"), dpi=300)
        plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')}")


# ---------------------------------------------------------------------------
# Rough figure (not for the paper): where the GPU's memory goes, per condition
# ---------------------------------------------------------------------------

def _on_grid(t, y, grid):
    if not len(t):
        return np.full_like(grid, np.nan)
    return np.interp(grid, t, y, left=np.nan, right=np.nan)


def _device_used_gb(rows: list, s: dict, grid) -> np.ndarray:
    """GB the GPU has in use: one server's view, or the MIG slices' summed."""
    by_url: dict = {}
    for r in rows:
        if r.get("device_used_mb") is not None:
            by_url.setdefault(r["url"], []).append((r["t"], r["device_used_mb"]))
    if not by_url:
        return np.full_like(grid, np.nan)
    curves = []
    for pts in by_url.values():
        pts.sort()
        curves.append(_on_grid(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), grid))
    if s["condition"] == "mig":
        return np.nansum(curves, axis=0) / 1024
    return np.nanmax(curves, axis=0) / 1024       # one GPU: every server sees the same


def draw_rough_memory(summ: dict, iso_dir: Path, out: Path) -> None:
    """One panel per condition: noisy B and the victim(s) (as panel (c) charges
    them), everything else (device used - B - victims: weights, decoders,
    activations, CUDA contexts, cache), and the device's used memory."""
    conds = [c for c in ("shared", "shared_bfq", "fmvisor", "mig", "no_sharing")
             if c in summ and (iso_dir / f"mem_{c}.jsonl").exists()]
    if not conds:
        return
    fig, axes = plt.subplots(1, len(conds), figsize=(3.0 * len(conds), 2.4),
                             sharey=True, squeeze=False)
    for ax, c in zip(axes[0], conds):
        s = summ[c]
        rows = load_stats(iso_dir / f"mem_{c}.jsonl")
        dur = s["duration"]
        grid = np.arange(0.0, dur, GRID_S)
        tb, yb, quota = noisy_memory_from_stats(rows, s)
        b = _on_grid(tb, yb, grid)
        v = np.zeros_like(grid)
        for task in s["victims"]:
            tv, yv, _ = noisy_memory_from_stats(rows, s, task)
            v = v + np.nan_to_num(_on_grid(tv, yv, grid))
        dev = _device_used_gb(rows, s, grid)
        other = np.clip(dev - np.nan_to_num(b) - v, 0, None)
        ax.plot(grid, b, color="#c0392b", lw=1.4, label="noisy B")
        ax.plot(grid, v, color="#2471a3", lw=1.4, label="victim")
        ax.plot(grid, other, color="#7f8c8d", lw=1.0, ls="--", label="other (weights, ctx, cache)")
        ax.plot(grid, dev, color="black", lw=0.9, label="device used")
        cap = (s.get("gpu_total_mb") or 0) / 1024
        if cap:
            ax.axhline(cap, color="#999999", lw=0.6, ls=":")
            ax.text(2, cap, "GPU capacity", fontsize=6, color="#777777", va="bottom")
        if c == "mig" and s.get("mig_slice_mb"):
            sl = s["mig_slice_mb"] / 1024
            ax.axhline(sl, color="#2E8B57", lw=0.6, ls=":")
            ax.text(2, sl, "one slice", fontsize=6, color="#2E8B57", va="bottom")
        if quota:
            ax.axhline(quota, color="#c0392b", lw=0.6, ls=":")
            ax.text(2, quota, "B's quota", fontsize=6, color="#c0392b", va="bottom")
        ax.axvline(s["ramp_at"], color="black", lw=0.4, ls=":")
        ax.set_title("Partitioned (MIG)" if c == "mig" else LABELS.get(c, c), fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_xlim(0, dur)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, color="#eeeeee", lw=0.4)
    axes[0][0].set_ylabel("GPU memory (GB)", fontsize=7)
    axes[0][0].legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.92))
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')}")


def parse_oom(spec: str | None) -> dict:
    """'mig_tpc:24,partitioned:16' -> {'mig_tpc': {24}, 'partitioned': {16}}"""
    out: dict = {}
    for item in (spec or "").split(","):
        if item.strip():
            c, n = item.split(":")
            out.setdefault(c.strip(), set()).add(int(n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, default=EXP_DIR / "results")
    ap.add_argument("--partitioned", default=None,
                    help="isolation condition drawn as partitioned: mig or no_sharing "
                         "(default: mig if it was run)")
    ap.add_argument("--out", type=Path, default=None,
                    help="main figure (default <exp-dir>/motivation2); the others are "
                         "written next to it as <out>_all and rough_memory")
    ap.add_argument("--thr-bin", type=float, default=5.0,
                    help="(b) and (d) bin width in s (default 5; larger: smoother lines)")
    ap.add_argument("--oom", default=None,
                    help="(a): points to draw as OOM that have no row, condition:n,... "
                         "(e.g. mig_tpc:24 when the memory cap killed the run's client)")
    args = ap.parse_args()
    iso_dir, eff_csv = args.exp_dir / "isolation", args.exp_dir / "efficiency" / "summary.csv"
    out = args.out or args.exp_dir / "motivation2"

    summ = latest_summaries(iso_dir / "summary.jsonl")
    ded = args.partitioned or ("mig" if "mig" in summ else "no_sharing")
    main_conds = [c for c in (ded, "shared") if c in summ]
    all_conds = [c for c in dict.fromkeys((ded, "shared", "shared_bfq", "fmvisor")) if c in summ]
    if not main_conds:
        raise SystemExit(f"no isolation runs of {ded} / shared in {iso_dir}")

    # 1) The motivation figure: Partitioned vs Shared, (a)-(d).
    draw_figure(summ, main_conds, iso_dir, out, args.thr_bin,
                eff_csv if eff_csv.exists() else None, STYLE_OF[ded], parse_oom(args.oom))
    # 2) Every isolation condition that was run, (b)-(d).
    if len(all_conds) > len(main_conds):
        draw_figure(summ, all_conds, iso_dir, out.with_name(out.name + "_all"), args.thr_bin)
    # 2b) Variant: (a) efficiency, (b) noisy tput, (c) victim tput, (d) memory.
    tp = out.with_name(out.name + "_tput")
    draw_figure(summ, main_conds, iso_dir, tp, args.thr_bin,
                eff_csv if eff_csv.exists() else None, STYLE_OF[ded], parse_oom(args.oom),
                layout="tput")
    if len(all_conds) > len(main_conds):
        draw_figure(summ, all_conds, iso_dir, tp.with_name(tp.name + "_all"), args.thr_bin,
                    eff_csv if eff_csv.exists() else None, STYLE_OF[ded], parse_oom(args.oom),
                    layout="tput")
    # 3) Rough: where the GPU's memory goes, per condition.
    draw_rough_memory(summ, iso_dir, out.with_name("rough_memory"))

    for c in all_conds:
        s = summ[c]
        print(f"  {c}: victims {s['victim_rps_before']} -> {s['victim_rps_after']} req/s, "
              f"B {s['surging_rps_before']} -> {s['surging_rps_after']} req/s, "
              + ", ".join(f"{t}: ok={v['ok']} fail={v['fail']} peak={v['peak_mem_mb']} MB"
                          for t, v in s["tasks"].items()))


if __name__ == "__main__":
    main()
