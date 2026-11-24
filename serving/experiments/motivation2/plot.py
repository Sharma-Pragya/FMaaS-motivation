#!/usr/bin/env python3
"""motivation2/plot.py — the motivation figure from the device-server runs
(run.sh / run.py), laid out as memory_isolation_vision/motivation_plot.py:

    (a) Efficiency   efficiency/summary.csv: GPU memory (bars) and throughput
                     (lines) vs #tasks, partitioned vs shared
    (b) Perf. iso.   victims' completed req/s over time
    (c) Mem. iso.    noisy task B's GPU memory over time
    (d) Fault iso.   victims' failed requests (cumulative)

Memory is the device server's gpu_alloc_peak_mb (torch allocated in the
server process, after each batch), one sample per completed request at the
server's end time; resampled on a GRID_S grid (max per bin, gaps
interpolated; nothing drawn after a server's last sample).
    partitioned (mig / no_sharing): B's own server.
    shared: one server for every task, so its memory is split: its mean
        before the ramp (backbone, decoders, steady queues) evenly, 1/N per
        task; everything above that goes to B (the victims' load is
        constant, so the growth is B's backlog).
(a) sums each server's mean memory, so partitioned counts one backbone copy
per task (the CUDA contexts are not in torch's count).

Usage (from serving/):
    python experiments/motivation2/plot.py --exp-dir experiments/motivation2/results
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

# Conditions -> memory_isolation_vision styles (mig, s_be, be).
STYLE_OF = {"mig": "mig", "no_sharing": "be", "shared": "s_be", "partitioned": None}
LABELS = {"mig": "Partitioned (MIG)", "no_sharing": "Partitioned", "shared": "Shared"}
FIG_W, FIG_H = 7.0, 1.6
LW, MS, N_MARKERS = 1.3, 3.5, 6
GRID_S = 0.5
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


def latest_summaries(path: Path) -> dict:
    out = {}
    for line in open(path):
        s = json.loads(line)
        out[s["condition"]] = s             # last line of each condition wins
    return out


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def efficiency_panel(ax, path: Path, part_key: str) -> None:
    data: dict = {}
    for r in csv.DictReader(open(path)):
        data.setdefault(r["condition"], {})[int(r["n_tasks"])] = r   # latest row wins
    ns = sorted({n for d in data.values() for n in d})
    x = np.arange(len(ns))
    w = 0.38
    ax_t = ax.twinx()
    top_mem, top_thr = 1.0, 1.0
    for i, (cond, key) in enumerate((("partitioned", part_key), ("shared", "s_be"))):
        rows = data.get(cond, {})
        off = (i - 0.5) * w
        ys = []
        for j, n in enumerate(ns):
            r = rows.get(n)
            if r and r["status"] in ("ok", "partial") and r["avg_gpu_mem_mb"]:
                gb = float(r["avg_gpu_mem_mb"]) / 1024
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, default=EXP_DIR / "results")
    ap.add_argument("--partitioned", default=None,
                    help="isolation condition drawn as partitioned: mig or no_sharing "
                         "(default: mig if it was run)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    iso_dir, eff_csv = args.exp_dir / "isolation", args.exp_dir / "efficiency" / "summary.csv"
    out = args.out or args.exp_dir / "motivation2"

    summ = latest_summaries(iso_dir / "summary.jsonl")
    ded = args.partitioned or ("mig" if "mig" in summ else "no_sharing")
    conds = [c for c in (ded, "shared") if c in summ]
    if not conds:
        raise SystemExit(f"no isolation runs of {ded} / shared in {iso_dir}")
    s0 = summ[conds[0]]
    dur, ramp = s0["duration"], s0["ramp_at"]
    V = set(s0["victims"])
    edges = np.arange(0, dur + 1e-9, mp.THR_BIN_S)
    mids = edges[:-1] + mp.THR_BIN_S / 2
    has_eff = eff_csv.exists()

    with plt.rc_context({
        "font.family": "sans-serif", "font.size": 7, "axes.labelsize": 7,
        "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "standard",
    }):
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        if has_eff:
            gs = fig.add_gridspec(1, 5, width_ratios=[1.3, 0.22, 1, 1, 1], wspace=0.42,
                                  left=0.055, right=0.99, top=0.76, bottom=0.21)
            efficiency_panel(fig.add_subplot(gs[0]), eff_csv, STYLE_OF[ded])
            ax_p = fig.add_subplot(gs[2])
            ax_m, ax_f = fig.add_subplot(gs[3], sharex=ax_p), fig.add_subplot(gs[4], sharex=ax_p)
        else:
            gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.07, right=0.99, top=0.76, bottom=0.21)
            ax_p = fig.add_subplot(gs[0])
            ax_m, ax_f = fig.add_subplot(gs[1], sharex=ax_p), fig.add_subplot(gs[2], sharex=ax_p)

        max_fail, max_mem, max_thr = 0, 0.0, 0.0
        fails = {}
        for c in conds:
            s, key = summ[c], STYLE_OF[c]
            ev = load_events(iso_dir / f"events_{c}.csv")
            ok_ev = [{"kind": "ok", "tenant": r["task"], "t": r["t_done"]} for r in ev if r["ok"]]
            thr = mp.rate_per_bin(ok_ev, V, edges)
            max_thr = max(max_thr, float(np.nanmax(thr)) if len(thr) else 0.0)
            ax_p.plot(mids, thr, **style(key, len(mids)))
            t, yb = noisy_memory_gb(ev, s)
            max_mem = max(max_mem, float(np.nanmax(yb)) if np.isfinite(yb).any() else 0.0)
            ax_m.plot(t, yb, **style(key, len(t)))
            ft = np.sort([r["t_done"] for r in ev if not r["ok"] and r["task"] in V])
            fails[c] = np.searchsorted(ft, mids, side="right")
            max_fail = max(max_fail, len(ft))

        k = max_fail >= 2000
        for c in conds:
            ax_f.plot(mids, fails[c] / (1000 if k else 1), **style(STYLE_OF[c], len(mids)))

        # (c) capacity lines (from the summary; absent if nvidia-smi / NVML had none).
        cap = (s0.get("gpu_total_mb") or summ.get("shared", {}).get("gpu_total_mb") or 0) / 1024
        sl = (summ.get("mig", {}).get("mig_slice_mb") or 0) / 1024
        top = max(max_mem, cap, sl) * 1.1 or 1.0
        if cap:
            ax_m.axhline(cap, color=mp.CAPACITY_COLOR, lw=0.6, ls="--")
            ax_m.text(4, cap, "GPU capacity", color=mp.CAPACITY_COLOR, fontsize=5.5, va="bottom")
        if sl and "mig" in conds:
            ax_m.axhline(sl, color=mp.COLORS["mig"], lw=0.6, ls="--")
            ax_m.text(4, sl, "MIG slice", color=mp.COLORS["mig"], fontsize=5.5, va="bottom")
        ax_m.set_ylim(0, top)
        ax_p.set_ylim(0, max_thr * 1.25 or 1.0)
        ax_f.set_ylim(0, (max_fail / (1000 if k else 1)) * 1.15 or 1.0)

        for ax, title, yl in ((ax_p, "(b) Performance isolation", "Victim tput (req/s)"),
                              (ax_m, "(c) Memory isolation", "Noisy task mem. (GB)"),
                              (ax_f, "(d) Fault isolation",
                               "Victim failed req." + (" (K)" if k else ""))):
            ax.set_title(title)
            ax.set_ylabel(yl, labelpad=1)
            ax.set_xlabel("Time (s)", labelpad=1)
            ax.set_xlim(0, dur)
            ax.axvline(ramp, **mp.PHASE_LINE)
            ax.yaxis.grid(True, color="#dddddd", lw=0.4)
            ax.set_axisbelow(True)
        ax_p.text(ramp, ax_p.get_ylim()[1], f" noisy: {s0['rate_b']:g}→{s0['rate_b_max']:g} rps",
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
    for c in conds:
        s = summ[c]
        print(f"  {c}: victims {s['victim_rps_before']} -> {s['victim_rps_after']} req/s, "
              f"B {s['surging_rps_before']} -> {s['surging_rps_after']} req/s, "
              + ", ".join(f"{t}: ok={v['ok']} fail={v['fail']} peak={v['peak_mem_mb']} MB"
                          for t, v in s["tasks"].items()))


if __name__ == "__main__":
    main()
