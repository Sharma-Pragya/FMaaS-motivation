#!/usr/bin/env python3
"""memory_isolation_llm/plot.py — KV cache and throughput per tenant (run.py).

Four panels on a shared time axis:
    (a) surging task B: KV cache held (GB), with each mode's pool for B dashed
    (b) victim task A:  KV cache held (GB), with each mode's pool for A dashed
    (c) B: completed req/s per 1 s bin, offered load in grey
    (d) A: completed req/s per 1 s bin, offered load in grey
KV held = tokens of the tenant's running requests x KV bytes per token (both
from the engine). In s_be one pool serves both tenants; in be / sp each
tenant has its own. Colours, line styles and legend order follow
lora_fair_share/fault_plot.py.

Sized at \\textwidth, saved with savefig.bbox "standard".

Usage (from serving/):
    python experiments/memory_isolation_llm/plot.py --exp-dir experiments/memory_isolation_llm/results_l40s
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
from matplotlib.ticker import MaxNLocator

EXP_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "fault_plot", EXP_DIR.parent / "lora_fair_share" / "fault_plot.py")
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)

MODES = [m for m in fp.MODES if m in ("sp", "be", "s_be")]   # legend order
FIG_W, FIG_H = 7.010, 3.2   # \textwidth
OFFERED = "#aaaaaa"
GB = 2 ** 30


def load(path: Path) -> list:
    rows = []
    for r in csv.DictReader(open(path)):
        if r["t"] in ("", "None"):
            continue
        r["t"] = float(r["t"])
        rows.append(r)
    return rows


def rate_at(s: dict, tenant: str, t: float) -> float:
    if tenant == "A":
        return s["rate_a"]
    if t < s["ramp_at"]:
        return s["rate_b"]
    frac = min(1.0, (t - s["ramp_at"]) / s["ramp_s"]) if s["ramp_s"] > 0 else 1.0
    return s["rate_b"] + frac * (s["rate_b_max"] - s["rate_b"])


def end_on_tick(ax, peak: float) -> None:
    ticks = MaxNLocator(nbins=4).tick_values(0, max(peak, 1e-9))
    ticks = ticks[ticks >= 0]
    top = ticks[ticks >= peak - 1e-9][0]
    ax.set_yticks(ticks[ticks <= top])
    ax.set_ylim(0, top)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, default=EXP_DIR / "results_l40s")
    args = ap.parse_args()
    summ = {}
    for line in open(args.exp_dir / "summary.jsonl"):
        s = json.loads(line)
        summ[s["mode"]] = s           # latest run per mode
    modes = [m for m in MODES if m in summ and (args.exp_dir / f"events_{m}.csv").exists()]
    if not modes:
        raise SystemExit(f"no events in {args.exp_dir}")
    any_s = summ[modes[0]]
    dur, ramp_at = any_s["duration"], any_s["ramp_at"]
    victim, surging = any_s["victim"], any_s["surging"]
    edges = np.arange(0, dur + 1e-9, 1.0)
    mids = edges[:-1] + 0.5

    with plt.rc_context({
        "font.family": "sans-serif", "font.size": 7, "axes.labelsize": 7,
        "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "standard",
    }):
        fig, axs = plt.subplots(2, 2, figsize=(FIG_W, FIG_H), sharex=True)
        (ax_bkv, ax_akv), (ax_b, ax_a) = axs
        kv_ax = {surging: ax_bkv, victim: ax_akv}
        tp_ax = {surging: ax_b, victim: ax_a}
        peak = {ax: 0.0 for ax in axs.flat}
        for m in modes:
            s = summ[m]
            ev = load(args.exp_dir / f"events_{m}.csv")
            kw = dict(color=fp.COLORS[m], ls=fp.STYLES[m], lw=1.1, label=fp.LABELS[m])
            for t in (surging, victim):
                eng = s["engines"][s["engine_of"][t]]
                bpt = eng.get("kv_bytes_per_token") or 0
                pts = sorted((e["t"], float(e["tokens"])) for e in ev
                             if e["kind"] == "kv" and e["tenant"] == t and 0 <= e["t"] < dur)
                if pts and bpt:
                    x, y = np.array(pts).T
                    kv_ax[t].plot(x, y * bpt / GB, **kw)
                    peak[kv_ax[t]] = max(peak[kv_ax[t]], (y * bpt / GB).max())
                if eng.get("kv_capacity_tokens") and bpt:
                    cap = eng["kv_capacity_tokens"] * bpt / GB
                    kv_ax[t].axhline(cap, color=fp.COLORS[m], lw=0.7, ls="--", alpha=0.7)
                    peak[kv_ax[t]] = max(peak[kv_ax[t]], cap)
                done = [e["t"] for e in ev if e["kind"] == "done" and e["tenant"] == t]
                y = np.histogram(done, edges)[0].astype(float)
                tp_ax[t].plot(mids, y, **kw)
                peak[tp_ax[t]] = max(peak[tp_ax[t]], y.max(initial=0.0))
        for t, ax in tp_ax.items():
            off = [rate_at(any_s, t, x) for x in mids]
            ax.plot(mids, off, color=OFFERED, lw=0.8, ls="-", zorder=0)
            peak[ax] = max(peak[ax], max(off))
        ax_b.text(dur, rate_at(any_s, surging, dur), "offered ", color=OFFERED,
                  fontsize=6.5, ha="right", va="bottom")

        for ax in axs.flat:
            ax.axvline(ramp_at, color="#c0392b", lw=0.8, ls=":")
            ax.yaxis.grid(True, color="#dddddd", lw=0.4)
            ax.set_axisbelow(True)
            end_on_tick(ax, peak[ax])
        ax_akv.text(ramp_at, ax_akv.get_ylim()[1], " B ramps up", color="#c0392b",
                    fontsize=6.5, va="top")
        ax_bkv.set_title("(a) Surging task: KV cache held")
        ax_akv.set_title("(b) Victim task: KV cache held")
        ax_b.set_title("(c) Surging task: throughput")
        ax_a.set_title("(d) Victim task: throughput")
        ax_bkv.set_ylabel("KV cache (GB)")
        ax_b.set_ylabel("Throughput (req/s)")
        for ax in (ax_b, ax_a):
            ax.set_xlabel("Time (s)")
            ax.set_xlim(0, dur)
        handles, labels = ax_a.get_legend_handles_labels()
        handles, labels = handles[:len(modes)], labels[:len(modes)]
        fig.legend(handles, labels, frameon=False, ncol=len(modes), loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.2)
        fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.12,
                            hspace=0.45, wspace=0.18)
        out = args.exp_dir / "kv_timeline"
        fig.savefig(f"{out}.pdf")
        fig.savefig(f"{out}.png", dpi=300)
        plt.close(fig)
    print(f"wrote {out}.pdf")
    for m in modes:
        st = summ[m]["stats"]
        caps = {n: e.get("kv_capacity_tokens") for n, e in summ[m]["engines"].items()}
        print(f"  {fp.LABELS[m]:5s} KV pool tokens {caps}")
        for t in (victim, surging):
            b, a = st[f"{t}_before"], st[f"{t}_surge"]
            print(f"        {t}: {b['rps']} -> {a['rps']} req/s, p50 {b['lat_p50_ms']} -> "
                  f"{a['lat_p50_ms']} ms, p99 {b['lat_p99_ms']} -> {a['lat_p99_ms']} ms")


if __name__ == "__main__":
    main()
