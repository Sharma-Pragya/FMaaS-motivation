#!/usr/bin/env python3
"""lora_fair_share/fault_plot.py — service around a fault in one task.

Reads <exp-dir>/fault/events_<mode>.csv (fault.py; --fault-kind segv reads
fault_segv/ and writes its plots there). Two panels on a shared
time axis, tokens of requests completed per 1 s bin (--smooth k: centered
k-bin moving average):
    top     the victims' total throughput (A1..A{N-1}, "Victim tasks"): does
            the fault reach the tasks that did nothing wrong?
    bottom  the faulty task P's throughput ("Faulty task").
Legend order SP, BE, S-BE, FMVisor; each y axis ends on a labelled tick.
Written twice: fault_timeline (tok/s) and fault_timeline_rps (requests
completed per 1 s bin, req/s); fault_isolation_tsfm counts one token per
request, so use the req/s plot there.

Sized at the paper's column width and saved with savefig.bbox "standard",
so \\includegraphics needs no width= (see RTVSntask plot.py).

Usage (from serving/):
    python experiments/lora_fair_share/fault_plot.py [--exp-dir ...] [--smooth 3] [--fault-kind segv]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

EXP_DIR = Path(__file__).resolve().parent
MODES = ["sp", "be", "s_be", "fmvisor"]   # legend order; FMVisor drawn on top
VICTIM_LABEL, FAULTY_LABEL = "Victim tasks", "Faulty task"
# fair_share/llm/plot.py palette and line styles
LABELS = {"s_be": "S-BE", "be": "BE", "sp": "SP", "fmvisor": "FMVisor"}
COLORS = {"s_be": "#F0A500", "be": "#888888", "sp": "#6B9AC4", "fmvisor": "#E06C75"}
STYLES = {"s_be": (0, (3, 1, 1, 1)), "be": "-.", "sp": "--", "fmvisor": "-"}
FIG_W, FIG_H = 3.336, 2.6   # \columnwidth


def load(path: Path) -> list:
    rows = []
    for r in csv.DictReader(open(path)):
        if r["t"] in ("", "None"):
            continue
        r["t"] = float(r["t"])
        rows.append(r)
    return rows


def per_bin(ev: list, tasks: set, edges: np.ndarray, smooth: int,
            metric: str = "tok") -> np.ndarray:
    """Completed tokens (metric "tok") or requests ("req") per bin."""
    ok = [e for e in ev if e["kind"] == "ok" and e["tenant"] in tasks]
    w = [float(e["n_tok"]) for e in ok] if metric == "tok" else None
    tok = np.histogram([e["t"] for e in ok], edges, weights=w)[0].astype(float)
    if smooth <= 1:
        return tok
    k = np.ones(smooth)
    # Edge bins average over the bins that exist.
    return np.convolve(tok, k, "same") / np.convolve(np.ones_like(tok), k, "same")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, default=EXP_DIR / "results_l40s")
    ap.add_argument("--smooth", type=int, default=1,
                    help="moving-average window in 1 s bins (1 = raw)")
    ap.add_argument("--fault-kind", default="gpu",
                    help="which fault.py run to plot: gpu reads fault/, others fault_<kind>/")
    args = ap.parse_args()
    # Mirrors fault.py:fault_dir (not imported: fault.py pulls in torch).
    fdir = args.exp_dir / ("fault" if args.fault_kind == "gpu" else f"fault_{args.fault_kind}")
    summ = {}
    for line in open(fdir / "summary.jsonl"):
        s = json.loads(line)
        if "victims" in s:           # skip runs from the two-tenant fault.py
            summ[s["mode"]] = s      # latest run per mode
    modes = [m for m in MODES if (fdir / f"events_{m}.csv").exists() and m in summ]
    if not modes:
        raise SystemExit(f"no events in {fdir}")
    any_s = summ[modes[0]]
    dur, fault_at = any_s["duration"], any_s["fault_at"]
    faulty, victims = any_s["fault_tenant"], set(any_s["victims"])
    edges = np.arange(0, dur + 1e-9, 1.0)
    mids = edges[:-1] + 0.5
    vname = (f"{min(victims)}–{max(victims)}" if len(victims) > 1 else next(iter(victims)))

    events = {m: [e for e in load(fdir / f"events_{m}.csv") if e["t"] < dur] for m in modes}
    smooths = [(args.smooth, "")]
    if args.smooth == 1:
        smooths.append((5, "_smooth"))
    outs = [plot_timeline(metric, modes, summ, events, edges, mids, s,
                          dur, fault_at, faulty, victims, vname, fdir, suffix)
            for metric in ("tok", "req") for s, suffix in smooths]
    for out in outs:
        print(f"wrote {out}.pdf")
    for m in modes:
        s = summ[m]
        rec = [(r["comp"], r["t_down"], r["t_up"]) for r in s["restarts"]]
        print(f"  {LABELS[m]:8s} victims ok={s['victims_ok']} fail={s['victims_fail']}  "
              f"{faulty} ok={s['ok'][faulty]} fail={s['fail'][faulty]}  outages={rec}")


def plot_timeline(metric, modes, summ, events, edges, mids, smooth,
                  dur, fault_at, faulty, victims, vname, fdir, suffix="") -> Path:
    """metric "tok": tok/s (fault_timeline); "req": req/s (fault_timeline_rps)."""
    unit = "tok/s" if metric == "tok" else "req/s"
    with plt.rc_context({
        "font.family": "sans-serif", "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "standard",
    }):
        fig, (ax_v, ax_p) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), sharex=True)
        peak = {ax_v: 0.0, ax_p: 0.0}
        for m in modes:
            ev = events[m]
            kw = dict(color=COLORS[m], ls=STYLES[m], lw=1.1, label=LABELS[m])
            for ax, tasks in ((ax_v, victims), (ax_p, {faulty})):
                y = per_bin(ev, tasks, edges, smooth, metric)
                ax.plot(mids, y, **kw)
                peak[ax] = max(peak[ax], float(y.max(initial=0.0)))

        for ax in (ax_v, ax_p):
            ax.axvline(fault_at, color="#c0392b", lw=0.8, ls=":")
            ax.yaxis.grid(True, color="#dddddd", lw=0.4)
            ax.set_axisbelow(True)
            # End the axis on a labelled tick at or above the data.
            ticks = MaxNLocator(nbins=4, integer=True).tick_values(0, max(peak[ax], 1.0))
            ticks = ticks[ticks >= 0]
            top = ticks[ticks >= peak[ax]][0]
            ax.set_yticks(ticks[ticks <= top])
            ax.set_ylim(0, top)
        # The faulty task's panel is empty above its line after the fault.
        ax_p.text(fault_at, ax_p.get_ylim()[1], " fault", color="#c0392b",
                  fontsize=6.5, va="top")
        ax_v.set_ylabel(f"{VICTIM_LABEL}\n({unit})")
        ax_p.set_ylabel(f"{FAULTY_LABEL}\n({unit})")
        ax_p.set_xlim(0, dur)
        ax_p.set_xlabel("Time (s)")
        ax_v.legend(frameon=False, ncol=len(modes), loc="lower center",
                    bbox_to_anchor=(0.5, 1.02), handlelength=2.2, columnspacing=1.0)
        fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.15, hspace=0.18)
        base = "fault_timeline" if metric == "tok" else "fault_timeline_rps"
        out = fdir / f"{base}{suffix}"
        fig.savefig(f"{out}.pdf")
        fig.savefig(f"{out}.png", dpi=300)
        plt.close(fig)
    return out


if __name__ == "__main__":
    main()
