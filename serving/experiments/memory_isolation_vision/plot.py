#!/usr/bin/env python3
"""memory_isolation_vision/plot.py — GPU memory and throughput of the surging
task and the victims (run.py).

Reads <exp-dir>/<staging>/events_<mode>.csv and summary.jsonl (latest run per
mode). Six panels on a shared time axis:
    (a) surging task B: its GPU memory, with the GPU's capacity (and one MIG
        slice's, if mig is drawn)
    (b) victims: their GPU memory (all victims together), same scale
GPU memory per task, from NVML's per-process samples (CUDA context, weights,
activations, allocator cache, queued requests):
    be / sp  a task's own instance (the victims' instances summed).
    s_be     one engine serves every task, so it is split. Its memory before
             the surge (mean over t < ramp_at: backbone, heads, context) is
             shared evenly, 1/N per task. What it grows above that is split
             in proportion to the GPU memory of each side's queued requests
             (staged inputs + features waiting for a head, as the engine
             counts them per task): B's queue and the allocator overhead it
             brings go to B.
Falls back to torch's reserved memory per process if NVML has no per-process
numbers (a container with its own PID namespace).
    (c) B: completed req/s per THR_BIN_S bin (with --spread: each batch spread
        over the time since the task's previous one, rate_per_bin)
    (d) victims: the same, summed over all victims
    Offered load is written above (c) and (d), one label per phase (before /
    from the surge), as fair_share/tsfm/plot.py does; it is not drawn.
    (e) B: median latency of the requests completed in each LAT_BIN_S bin (log scale;
        a gap where none completed)
    (f) victims: the same, pooled over all victims
A process that has stopped is drawn at 0 from then on.
Colours, line styles and legend order follow lora_fair_share/fault_plot.py.

Sized at \\textwidth, saved with savefig.bbox "standard", so \\includegraphics
needs no width=.

Usage (from serving/):
    python experiments/memory_isolation_vision/plot.py --exp-dir experiments/memory_isolation_vision/results_t4_n8
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

EXP_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "fault_plot", EXP_DIR.parent / "lora_fair_share" / "fault_plot.py")
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)

# fault_plot.py's modes and styles, plus MIG (not in fault_plot.py).
MODES = ["mig", "sp_mps"] + [m for m in fp.MODES if m in ("sp", "be", "s_be", "fmvisor")]   # legend order
LABELS = {**fp.LABELS, "mig": "MIG", "sp_mps": "SP (MPS)"}
COLORS = {**fp.COLORS, "mig": "#2E8B57", "sp_mps": "#1F4E79"}
STYLES = {**fp.STYLES, "mig": (0, (1, 1)), "sp_mps": (0, (5, 1))}
# Markers as fair_share/tsfm/plot.py MARKERS (SP ^, S-BE P, BE D, FMVisor o),
# about 8 per line (its markevery = len // 8).
MARKERS = {"sp": "^", "s_be": "P", "be": "D", "fmvisor": "o", "mig": "v", "sp_mps": "<"}
N_MARKERS = 8
MARKER_SIZE = 5.0           # fair_share/tsfm/plot.py lines.markersize
LINE_WIDTH = 1.6            # fair_share/tsfm/plot.py lines.linewidth
PHASE_LINE = dict(color="black", lw=0.4, ls=":", zorder=2)   # fair_share _add_phase_lines
FIG_W, FIG_H = 7.010, 4.6   # \textwidth
THR_BIN_S = 2.0             # throughput bins (s), fair_share's --bin-size-s default
THR_SPREAD_MAX_S = 2.0      # (c), (d): see rate_per_bin
LAT_BIN_S = 1.0             # latency bins (s)
CAPACITY_COLOR = "#999999"
OFFERED_TEXT_COLOR = "#333333"   # offered-load labels above (c), (d), as fair_share
OFFERED_TITLE_PAD = 11           # (c), (d) titles sit above those labels (pt)
SURGE_COLOR = "#c0392b"


def cut_at_down(t: np.ndarray, y: np.ndarray, t_down, dur: float) -> tuple:
    """y up to t_down, then 0 to the end (a dead process sends no samples)."""
    if t_down is None or t_down >= dur:
        return t, y
    keep = t < t_down
    return np.concatenate([t[keep], [t_down, dur]]), np.concatenate([y[keep], [0.0, 0.0]])


def queued_mb(ev: list, tasks: set) -> tuple:
    """(times, MB) of tasks' staged inputs + features waiting for their heads
    (one engine reports every task it serves at the same t)."""
    rows: dict = {}
    for e in ev:
        if e["kind"] in ("staged", "feat") and e["tenant"] in tasks:
            rows[e["t"]] = rows.get(e["t"], 0.0) + float(e["mb"] or 0.0)
    t = np.array(sorted(rows), dtype=float)
    return t, np.array([rows[x] for x in t], dtype=float)


def process_mb(ev: list, comp: str) -> tuple:
    """(times, MB) of one serving process: NVML if sampled, else torch reserved."""
    rows = [(e["t"], float(e["mb"])) for e in ev
            if e["kind"] == "mem" and e["comp"] == comp and e["mb"] not in ("", "None")]
    if not rows:
        rows = [(e["t"], float(e["note"])) for e in ev
                if e["kind"] == "torch" and e["comp"] == comp and e["note"] not in ("", "None")]
    rows.sort()
    return (np.array([r[0] for r in rows], dtype=float),
            np.array([r[1] for r in rows], dtype=float))


def task_memory_gb(ev: list, s: dict, dur: float) -> tuple:
    """(times, B's GB, victims' GB) on a 0.5 s grid (see module docstring)."""
    down = {x["comp"]: x["t_down"] for x in s["downs"]}
    surge, victims = s["surging"], set(s["victims"])
    grid = np.arange(0.0, dur, 0.5)
    b_mb, v_mb = np.zeros_like(grid), np.zeros_like(grid)
    for c, ts in s["layout"].items():
        t, y = process_mb(ev, c)
        if not len(t):
            continue
        t, y = cut_at_down(t, y, down.get(c), dur)
        # A process contributes nothing before its first sample or after its last.
        y = np.interp(grid, t, y, left=0.0, right=0.0)
        if surge not in ts:
            v_mb += y
        elif not victims & set(ts):
            b_mb += y
        else:                                   # s_be: one engine for every task
            n = len(ts)
            pre = y[(grid < s["ramp_at"]) & (y > 0)]
            base = float(pre.mean()) if len(pre) else 0.0
            tb, qb = queued_mb(ev, {surge})
            tv, qv = queued_mb(ev, victims & set(ts))
            qb = np.interp(grid, tb, qb) if len(tb) else np.zeros_like(grid)
            qv = np.interp(grid, tv, qv) if len(tv) else np.zeros_like(grid)
            frac_b = np.divide(qb, qb + qv, out=np.full_like(grid, 1.0 / n),
                               where=(qb + qv) > 0)
            fixed = np.minimum(y, base)         # 0 once the engine is down
            grown = np.maximum(0.0, y - base)
            # fmvisor: once B's own process is down, none of the shared
            # engine is B's. What it still holds above its base then is B's
            # memory mapped through IPC, not yet released: counted for nobody.
            b_only = [c2 for c2, ts2 in s["layout"].items() if list(ts2) == [surge]]
            b_down = down.get(b_only[0]) if b_only else None
            if b_down is not None:
                gone = grid >= b_down
                frac_b = np.where(gone, 0.0, frac_b)
                grown = np.where(gone, 0.0, grown)
                b_mb += np.where(gone, 0.0, fixed / n)
            else:
                b_mb += fixed / n
            b_mb += grown * frac_b
            v_mb += fixed * (n - 1) / n + grown * (1.0 - frac_b)
    return grid, b_mb / 1024, v_mb / 1024


def rate_per_bin(ev: list, tasks: set, edges: np.ndarray,
                 max_gap: float = THR_SPREAD_MAX_S) -> np.ndarray:
    """Completed req/s per bin, each batch's completions spread evenly over
    the time since the same task's previous completion (at most max_gap s).

    A backlogged process finishes whole batches, e.g. 32 every 0.92 s; counted
    by completion time, 1 s bins show that as 32, 0 or 64 req/s, and a moving
    average only as multiples of 32 / window. Spreading gives the rate it
    actually serves at (34.8 req/s there). Tasks are spread separately, then
    summed. A process that stops completes nothing after its last batch, so
    a crash still drops to 0 at once."""
    out = np.zeros(len(edges) - 1)
    for task in tasks:
        counts: dict = {}
        for e in ev:
            if e["kind"] == "ok" and e["tenant"] == task:
                counts[e["t"]] = counts.get(e["t"], 0) + 1
        ts = sorted(counts)
        for i, t in enumerate(ts):
            if i > 0:
                gap = t - ts[i - 1]
            else:                           # first batch: as long as the next gap
                gap = ts[1] - ts[0] if len(ts) > 1 else 0.0
            gap = min(gap, max_gap)
            n = counts[t]
            if gap <= 0:
                b = np.searchsorted(edges, t, side="right") - 1
                if 0 <= b < len(out):
                    out[b] += n
                continue
            lo = t - gap
            b0 = max(0, np.searchsorted(edges, lo, side="right") - 1)
            b1 = min(len(out) - 1, np.searchsorted(edges, t, side="right") - 1)
            for b in range(b0, b1 + 1):
                ov = min(t, edges[b + 1]) - max(lo, edges[b])
                if ov > 0:
                    out[b] += n * ov / gap
    return out / np.diff(edges)


def p50_per_bin(ev: list, tasks: set, edges: np.ndarray) -> np.ndarray:
    """Median latency (ms) of the requests of tasks completed in each bin;
    NaN (a gap in the line) where none completed."""
    ok = [(e["t"], float(e["lat_ms"])) for e in ev if e["kind"] == "ok" and e["tenant"] in tasks]
    idx = np.digitize([t for t, _ in ok], edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(out)):
        lat = [l for (_, l), i in zip(ok, idx) if i == b]
        if lat:
            out[b] = float(np.median(lat))
    return out


def end_on_tick(ax, peak: float) -> None:
    ticks = MaxNLocator(nbins=4).tick_values(0, max(peak, 1e-9))
    ticks = ticks[ticks >= 0]
    top = ticks[ticks >= peak - 1e-9][0]
    ax.set_yticks(ticks[ticks <= top])
    ax.set_ylim(0, top)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, default=EXP_DIR / "results_t4_n8")
    ap.add_argument("--staging", default="gpu")
    ap.add_argument("--spread", action="store_true",
                    help="(c), (d): spread each batch's completions over the time since "
                         "the task's previous batch (rate_per_bin) instead of counting "
                         "them in the bin they complete in")
    ap.add_argument("--modes", default="s_be,be,fmvisor",
                    help="modes to draw (comma-separated; sp is run but not drawn by default)")
    args = ap.parse_args()
    want = set(args.modes.split(","))
    d = args.exp_dir / args.staging
    summ = {}
    for line in open(d / "summary.jsonl"):
        s = json.loads(line)
        if "victims" not in s:        # a two-task run from before --n-tasks
            continue
        summ[s["mode"]] = s           # latest run per mode
    modes = [m for m in MODES
             if m in want and m in summ and (d / f"events_{m}.csv").exists()]
    if not modes:
        raise SystemExit(f"no N-task runs in {d}")
    any_s = summ[modes[0]]
    dur, ramp_at = any_s["duration"], any_s["ramp_at"]
    surge, victims = any_s["surging"], any_s["victims"]
    edges = np.arange(0, dur + 1e-9, THR_BIN_S)
    mids = edges[:-1] + THR_BIN_S / 2
    lat_edges = np.arange(0, dur + 1e-9, LAT_BIN_S)
    lat_mids = lat_edges[:-1] + LAT_BIN_S / 2
    events = {m: [e for e in fp.load(d / f"events_{m}.csv") if 0 <= e["t"] < dur] for m in modes}
    capacity = max((summ[m].get("gpu_total_mb") or 0) for m in modes) / 1024

    with plt.rc_context({
        "font.family": "sans-serif", "font.size": 7, "axes.labelsize": 7,
        "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "standard",
    }):
        fig, axs = plt.subplots(3, 2, figsize=(FIG_W, FIG_H), sharex=True)
        (ax_bm, ax_vm), (ax_bt, ax_vt), (ax_bl, ax_vl) = axs
        lat_axes = (ax_bl, ax_vl)
        peak = {ax: 0.0 for ax in axs.flat}
        lat_lo, lat_hi = np.inf, 0.0
        for m in modes:
            s, ev = summ[m], events[m]
            kw = dict(color=COLORS[m], ls=STYLES[m], lw=LINE_WIDTH, label=LABELS[m],
                      marker=MARKERS[m], markersize=MARKER_SIZE)

            t, yb, yv = task_memory_gb(ev, s, dur)
            for ax, y in ((ax_bm, yb), (ax_vm, yv)):
                ax.plot(t, y, markevery=max(1, len(t) // N_MARKERS), **kw)
                peak[ax] = max(peak[ax], float(y.max(initial=0.0)))

            for ax, ts in ((ax_bt, {surge}), (ax_vt, set(victims))):
                if args.spread:             # each batch spread over its interval
                    y = rate_per_bin(ev, ts, edges)
                else:                       # completions per 1 s bin, as counted
                    y = fp.per_bin(ev, ts, edges, 1, "req") / THR_BIN_S

                ax.plot(mids, y, markevery=max(1, len(mids) // N_MARKERS), **kw)
                peak[ax] = max(peak[ax], float(y.max(initial=0.0)))

            for ax, ts in ((ax_bl, {surge}), (ax_vl, set(victims))):
                y = p50_per_bin(ev, ts, lat_edges)
                ax.plot(lat_mids, y, markevery=max(1, len(lat_mids) // N_MARKERS), **kw)
                if np.isfinite(y).any():
                    lat_lo = min(lat_lo, float(np.nanmin(y)))
                    lat_hi = max(lat_hi, float(np.nanmax(y)))

        # Both memory panels on one scale, up to the GPU's capacity.
        peak[ax_bm] = peak[ax_vm] = max(peak[ax_bm], peak[ax_vm], capacity)
        if capacity:
            for ax in (ax_bm, ax_vm):
                ax.axhline(capacity, color=CAPACITY_COLOR, lw=0.8, ls="--")
            slice_gb = summ["mig"].get("mig_slice_mb") if "mig" in modes else None
            if slice_gb:                # what B can use under MIG: one slice
                slice_gb /= 1024
                ax_bm.axhline(slice_gb, color=COLORS["mig"], lw=0.6, ls="--")
                ax_bm.text(dur, slice_gb, "MIG slice ", color=COLORS["mig"],
                           fontsize=6.5, ha="right", va="bottom")
            ax_bm.text(dur, capacity, "GPU capacity ", color=CAPACITY_COLOR,
                       fontsize=6.5, ha="right", va="bottom")
        for ax in axs.flat:
            ax.axvline(ramp_at, **PHASE_LINE)
            ax.yaxis.grid(True, color="#dddddd", lw=0.4)
            ax.set_axisbelow(True)
            if ax not in lat_axes:
                end_on_tick(ax, peak[ax])
        if lat_hi > 0:   # both latency panels on one log scale, whole decades
            for ax in lat_axes:
                ax.set_yscale("log")
                ax.set_ylim(10 ** np.floor(np.log10(lat_lo)), 10 ** np.ceil(np.log10(lat_hi)))
        ax_vm.text(ramp_at, 0.97 * capacity if capacity else ax_vm.get_ylim()[1], " B's surge starts", color=SURGE_COLOR,
                   fontsize=6.5, va="top")

        nv = len(victims)
        # Offered load per phase, written above (c) and (d) as in
        # fair_share/tsfm/plot.py _annotate_offered_loads (no offered-load line).
        ra, rb, rb_max = any_s["rate_a"], any_s["rate_b"], any_s["rate_b_max"]
        a_rps = f"{nv} × {ra:g} rps" if nv > 1 else f"{ra:g} rps"
        for ax, labels in ((ax_bt, (f"{rb:g} rps", f"{rb:g} → {rb_max:g} rps")),
                           (ax_vt, (a_rps, a_rps))):
            for (t0, t1), text in zip(((0.0, ramp_at), (ramp_at, dur)), labels):
                ax.text((t0 + t1) / 2 / dur, 1.03, text, transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=6.5, fontweight="bold",
                        color=OFFERED_TEXT_COLOR, clip_on=False)
        ax_bm.set_title("(a) Surging task B: GPU memory")
        vname = f"{nv} victim tasks" if nv > 1 else f"Victim task {victims[0]}"
        ax_vm.set_title(f"(b) {vname}: GPU memory")
        ax_bt.set_title("(c) Surging task B: throughput", pad=OFFERED_TITLE_PAD)
        ax_vt.set_title(f"(d) {vname}: {'total ' if nv > 1 else ''}throughput",
                        pad=OFFERED_TITLE_PAD)
        ax_bl.set_title("(e) Surging task B: median latency")
        ax_vl.set_title(f"(f) {vname}: median latency")
        ax_bm.set_ylabel("Memory (GB)")
        ax_bt.set_ylabel("Throughput (req/s)")
        ax_bl.set_ylabel("Latency (ms)")
        for ax in (ax_bl, ax_vl):
            ax.set_xlabel("Time (s)")
        for ax in axs.flat:
            ax.set_xlim(0, dur)
        handles, labels = ax_vt.get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncol=len(labels), loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.2)
        fig.subplots_adjust(left=0.07, right=0.99, top=0.885, bottom=0.085,
                            hspace=0.5, wspace=0.18)
        out = d / "mem_timeline"
        fig.savefig(f"{out}.pdf")
        fig.savefig(f"{out}.png", dpi=300)
        plt.close(fig)
    print(f"wrote {out}.pdf")
    for m in modes:
        s, st = summ[m], summ[m]["stats"]
        v_fail = sum(s["fail"][t] for t in s["victims"])
        print(f"  {LABELS[m]:5s} victims {st['victims_before']['rps']} -> "
              f"{st['victims_surge']['rps']} req/s (p50 {st['victims_before']['lat_p50_ms']} -> "
              f"{st['victims_surge']['lat_p50_ms']} ms, {v_fail} failed)  "
              f"B {st['surging_before']['rps']} -> {st['surging_surge']['rps']} req/s "
              f"({s['fail'][s['surging']]} failed)  peak GPU used={s.get('gpu_peak_used_mb')} MB  "
              f"down={[(x['comp'], x['t_down'], x['why'][:40]) for x in s['downs']]}")


if __name__ == "__main__":
    main()
