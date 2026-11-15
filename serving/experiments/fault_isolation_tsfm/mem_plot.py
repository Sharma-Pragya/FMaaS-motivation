#!/usr/bin/env python3
"""fault_isolation_tsfm/mem_plot.py — memory leak in the faulty task's decoder.

Reads <exp-dir>/fault_leak/events_<mode>.csv and summary.jsonl (run.py
--fault-kind leak). Four panels on a shared time axis:
    (a) faulty task's GPU memory, with the GPU's capacity
    (b) victim tasks' GPU memory (all victims together)
    (c) faulty task's completed req/s
    (d) victim tasks' completed req/s
Memory per task, from NVML's per-process samples:
    BE / SP  faulty = P's own instance (its backbone copy, decoder, leak);
             victims = the other instances.
    S-BE     one process serves every task, so NVML cannot split it:
             faulty = what the shared engine has grown since the leak started
             (its mean over the 5 s before --fault-at is the baseline; under
             steady load nothing else grows); victims = the rest (backbone,
             all decoders, allocator cache). The decoder's own "leak" events
             are not used: the last ones are lost when the engine dies.
A process that has stopped is drawn at 0 GB from then on. Styling, legend
order and binning follow lora_fair_share/fault_plot.py.

Sized at \\textwidth and saved with savefig.bbox "standard", so
\\includegraphics needs no width=.

Usage (from serving/):
    python experiments/fault_isolation_tsfm/mem_plot.py --exp-dir experiments/fault_isolation_tsfm/results_t4
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

FIG_W, FIG_H = 7.010, 3.2   # \textwidth
CAPACITY_COLOR = "#999999"


def sample_times(ev: list) -> np.ndarray:
    return np.array([e["t"] for e in ev if e["kind"] == "mem" and e["comp"] == "gpu_used"])


def mem_series(ev: list, comp: str, down_t: float | None) -> np.ndarray:
    """GB for comp at each sample time; 0 once it has stopped (including the
    sample just before the exit is noticed, which NVML may already miss),
    NaN if not sampled."""
    got = {e["t"]: float(e["mb"]) / 1024 for e in ev
           if e["kind"] == "mem" and e["comp"] == comp}
    ys = []
    for t in sample_times(ev):
        if t in got:
            ys.append(got[t])
        elif down_t is not None and t >= down_t - 1.0:
            ys.append(0.0)
        else:
            ys.append(np.nan)
    return np.array(ys)


def task_memory(s: dict, ev: list, faulty: str, victims: set) -> tuple:
    """(faulty GB, victims GB) at each sample time; see the module docstring."""
    downs = {d["comp"]: d["t_down"] for d in s["restarts"]}
    chain = s["chain"]
    comps = sorted({c for cs in chain.values() for c in cs})
    series = {c: mem_series(ev, c, downs.get(c)) for c in comps}
    stack = np.vstack([series[c] for c in comps])
    total = np.where(np.all(np.isnan(stack), axis=0), np.nan, np.nansum(stack, axis=0))
    own = [c for c in chain[faulty] if all(c not in chain[v] for v in victims)]
    if own:                       # a process that serves only the faulty task
        f = series[own[-1]]
    else:                         # shared process: its growth since the leak began
        t = sample_times(ev)
        before = (t >= s["fault_at"] - 5) & (t < s["fault_at"])
        base = np.nanmean(total[before]) if before.any() else 0.0
        f = np.clip(total - base, 0, None)
    return f, np.clip(total - f, 0, None)


def end_on_tick(ax, peak: float) -> None:
    ticks = MaxNLocator(nbins=4).tick_values(0, max(peak, 1e-9))
    ticks = ticks[ticks >= 0]
    top = ticks[ticks >= peak - 1e-9][0]
    ax.set_yticks(ticks[ticks <= top])
    ax.set_ylim(0, top)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, default=EXP_DIR / "results")
    ap.add_argument("--fault-kind", default="leak")
    args = ap.parse_args()
    fdir = args.exp_dir / f"fault_{args.fault_kind}"
    summ = {}
    for line in open(fdir / "summary.jsonl"):
        s = json.loads(line)
        summ[s["mode"]] = s          # latest run per mode
    modes = [m for m in fp.MODES if (fdir / f"events_{m}.csv").exists() and m in summ]
    if not modes:
        raise SystemExit(f"no events in {fdir}")
    any_s = summ[modes[0]]
    dur, fault_at = any_s["duration"], any_s["fault_at"]
    faulty, victims = any_s["fault_tenant"], set(any_s["victims"])
    edges = np.arange(0, dur + 1e-9, 1.0)
    mids = edges[:-1] + 0.5
    events = {m: [e for e in fp.load(fdir / f"events_{m}.csv") if e["t"] < dur] for m in modes}
    capacity = max((summ[m].get("gpu_total_mb") or 0) for m in modes) / 1024

    with plt.rc_context({
        "font.family": "sans-serif", "font.size": 7, "axes.labelsize": 7,
        "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "standard",
    }):
        fig, axs = plt.subplots(2, 2, figsize=(FIG_W, FIG_H), sharex=True)
        (ax_fmem, ax_vmem), (ax_p, ax_v) = axs
        peak = {ax: 0.0 for ax in axs.flat}
        for m in modes:
            s, ev = summ[m], events[m]
            kw = dict(color=fp.COLORS[m], ls=fp.STYLES[m], lw=1.1, label=fp.LABELS[m])
            t = sample_times(ev)
            if len(t):
                f, v = task_memory(s, ev, faulty, victims)
                for ax, y in ((ax_fmem, f), (ax_vmem, v)):
                    if not np.all(np.isnan(y)):
                        ax.plot(t, y, **kw)
                        peak[ax] = max(peak[ax], np.nanmax(y))
            for ax, tasks in ((ax_p, {faulty}), (ax_v, victims)):
                y = fp.per_bin(ev, tasks, edges, 1, "req")
                ax.plot(mids, y, **kw)
                peak[ax] = max(peak[ax], float(y.max(initial=0.0)))

        if capacity:
            ax_fmem.axhline(capacity, color=CAPACITY_COLOR, lw=0.8, ls="--")
            peak[ax_fmem] = max(peak[ax_fmem], capacity)
            ax_fmem.text(dur, capacity, "GPU capacity ", color=CAPACITY_COLOR,
                         fontsize=6.5, ha="right", va="bottom")
        for ax in axs.flat:
            ax.axvline(fault_at, color="#c0392b", lw=0.8, ls=":")
            ax.yaxis.grid(True, color="#dddddd", lw=0.4)
            ax.set_axisbelow(True)
            end_on_tick(ax, peak[ax])
        # The victims' memory panel is flat, so its top is clear at the fault.
        ax_vmem.text(fault_at, ax_vmem.get_ylim()[1], " leak starts", color="#c0392b",
                     fontsize=6.5, va="top")

        ax_fmem.set_title(f"(a) {fp.FAULTY_LABEL}: memory")
        ax_vmem.set_title(f"(b) {fp.VICTIM_LABEL}: memory")
        ax_p.set_title(f"(c) {fp.FAULTY_LABEL}: throughput")
        ax_v.set_title(f"(d) {fp.VICTIM_LABEL}: throughput")
        ax_fmem.set_ylabel("Memory (GB)")
        ax_p.set_ylabel("Throughput (req/s)")
        for ax in (ax_p, ax_v):
            ax.set_xlabel("Time (s)")
            ax.set_xlim(0, dur)
        handles, labels = ax_v.get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncol=len(modes), loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.2)
        fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.12,
                            hspace=0.45, wspace=0.18)
        out = fdir / "mem_timeline"
        fig.savefig(f"{out}.pdf")
        fig.savefig(f"{out}.png", dpi=300)
        plt.close(fig)
    print(f"wrote {out}.pdf")
    for m in modes:
        s = summ[m]
        downs = [(d["comp"], d["t_down"] and round(d["t_down"], 1)) for d in s["restarts"]]
        print(f"  {fp.LABELS[m]:8s} victims ok={s['victims_ok']} fail={s['victims_fail']}  "
              f"{faulty} ok={s['ok'][faulty]} fail={s['fail'][faulty]}  "
              f"peak GPU used={s.get('gpu_peak_used_mb')} MB  down={downs}")


if __name__ == "__main__":
    main()
