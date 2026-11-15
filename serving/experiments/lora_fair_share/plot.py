#!/usr/bin/env python3
"""lora_fair_share/plot.py — Normalized share per scheduler + fairness table.

Reads <exp-dir>/summary.jsonl. For each (batch cap, weights) it divides each
tenant's contention tokens/s by its standalone tokens/s at the same cap
(S_i), normalizes by the weight share (S_i / (w_i / sum w)), and computes
Jain's index over the normalized shares. Predictions from the profiled cost
model (decode only) are printed and drawn as ticks.

Usage (from serving/):
    python experiments/lora_fair_share/plot.py [--exp-dir ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
from cost_model import CASES, DEFAULT_PROFILE, Profile  # noqa: E402

SCHEDULERS = ["be", "sp", "fcfs", "vtc", "tuned_vtc", "bfq", "bfq_stfq"]
LABELS = {"be": "BE", "sp": "SP", "fcfs": "FCFS", "vtc": "VTC",
          "tuned_vtc": "Tuned VTC", "bfq": "BFQ", "bfq_stfq": "BFQ-STFQ"}
TENANT_COLORS = {"P": "#2a78d6", "A": "#eb6834"}
CASE_LABELS = {"mixed": "plain + rank-64", "both_adapter": "rank-64 + rank-64",
               "both_plain": "plain + plain"}


def tenant_label(t: str, case: str) -> str:
    return f"{'Rank-64' if CASES[case][t] else 'Plain'} tenant ({t})"


def case_suffix(case: str, demand: str = "1:1") -> str:
    """File-name suffix; the original mixed case at 1:1 demand keeps its old names."""
    return (("" if case == "mixed" else f"_{case}")
            + ("" if demand == "1:1" else f"_d{demand.replace(':', '-')}"))


def scen_label(case: str, demand: str) -> str:
    return CASE_LABELS[case] + ("" if demand == "1:1" else f", demand {demand}")


def ratio(s: str) -> dict:
    a, p = (float(x) for x in s.split(":"))
    return {"A": a, "P": p}


def predict(mode, cap, w, profile, m_A, demand=None):
    """Normalized shares under the profiled per-token costs (prefill ignored).

    Tokens served: n_t = s_t * x / mult_t. Fair schedulers equalize weighted
    charge (s = w, mult_t = per-token charge: 1 for VTC, c_t for BFQ); FCFS
    serves in arrival order (s = demand, mult = 1). x is set by
    n_P c_P + n_A c_A = 1. Normalized share = n_t c_t W / w_t.
    None for be/sp: the shared-engine profile does not model two engines.
    """
    if mode in ("be", "sp"):
        return None
    c = {t: profile.decode_tok_ms(t, cap) for t in ("P", "A")}
    s = w
    if mode == "fcfs":
        s, mult = demand or {"P": 1, "A": 1}, {"P": 1, "A": 1}
    elif mode == "vtc":
        mult = {"P": 1, "A": 1}
    elif mode == "tuned_vtc":
        mult = {"P": 1, "A": m_A}
    else:
        mult = c
    x = 1 / sum(s[t] * c[t] / mult[t] for t in c)
    return {t: s[t] * x * c[t] * sum(w.values()) / (mult[t] * w[t]) for t in c}


def gpu_time_fairness(r: dict) -> float:
    """fair_share's hybrid weighted max-min fairness (fair_share/llm/plot.py:
    minmax_fairness), with work measured in GPU time instead of requests.

    S_t = tenant t's tok/s as a fraction of its standalone tok/s, i.e. the
    fraction of the GPU it received. Capacity = S_P + S_A (achieved, as in
    fair_share), ideal_t = w_t / W x capacity, r_t = min(S_t / ideal_t, 1),
    fairness = min(r) / max(r). Both tenants are backlogged, so fair_share's
    "both satisfied -> 1" case never applies.
    """
    w_a, w_p = (float(x) for x in r["weights_A_P"].split(":"))
    w = {"A": w_a, "P": w_p}
    W = w_a + w_p
    s = {t: r[f"{t}_norm_share"] * w[t] / W for t in w}
    cap = sum(s.values())
    ratio = {t: min(s[t] / (w[t] / W * cap), 1.0) for t in w}
    return min(ratio.values()) / max(ratio.values())


def token_fairness(r: dict) -> float:
    """Same hybrid weighted max-min fairness as gpu_time_fairness, but work is
    raw tokens/s (VTC's notion of service), not the GPU-time fraction. A
    scheduler that equalizes GPU time gives the costlier rank-64 tenant fewer
    tokens, so it scores low here."""
    w_a, w_p = (float(x) for x in r["weights_A_P"].split(":"))
    w = {"A": w_a, "P": w_p}
    W = w_a + w_p
    s = {t: r[f"{t}_tok_s"] for t in w}
    cap = sum(s.values())
    ratio = {t: min(s[t] / (w[t] / W * cap), 1.0) for t in w}
    return min(ratio.values()) / max(ratio.values())


# Paper figure: same look as fair_share/llm/plot.py:plot_fairness_summary.
PAPER_METHODS = ["be", "sp", "fcfs", "vtc", "bfq_stfq"]
PAPER_LABELS = {"be": "BE", "sp": "SP", "fcfs": "S-BE", "vtc": "VTC",
                "bfq_stfq": "FMVisor"}
PAPER_STYLE = {
    "axes.edgecolor": "black", "axes.labelcolor": "black", "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#cccccc", "grid.linestyle": ":", "grid.linewidth": 0.5,
    "xtick.color": "black", "ytick.color": "black",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
FAIR_COLOR = "#34495E"
TPUT_COLOR = "#E67E22"


def nice_ceil(v: float) -> float:
    mag = 10.0 ** np.floor(np.log10(v))
    for cap in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if v / mag <= cap + 1e-9:
            return cap * mag
    return 10.0 * mag


def plot_fairness_twin(sub: list, cap: int, wts: str, out: Path,
                       fair_fn=gpu_time_fairness,
                       fair_label: str = "Fairness (GPU time)",
                       case: str = "mixed", demand: str = "1:1") -> None:
    """Twin-axis bars: fairness per fair_fn (left) + aggregate tok/s (right).
    Methods without results keep their slot, marked n/a."""
    by = {r["scheduler"]: r for r in sub}
    fair = [fair_fn(by[m]) if m in by else None for m in PAPER_METHODS]
    tput = [by[m]["agg_tok_s"] if m in by else None for m in PAPER_METHODS]
    x = np.arange(len(PAPER_METHODS))
    bar_w = 0.38
    w_a, w_p = wts.split(":")

    with plt.rc_context(PAPER_STYLE):
        fig, ax_l = plt.subplots(figsize=(4.0, 3.4))
        ax_r = ax_l.twinx()
        ax_r.spines["right"].set_visible(True)
        for ax, off, vals, color, fmt in ((ax_l, -1, fair, FAIR_COLOR, "{:.2f}"),
                                          (ax_r, 1, tput, TPUT_COLOR, "{:.0f}")):
            xs = [xi + off * bar_w / 2 for xi, v in zip(x, vals) if v is not None]
            vs = [v for v in vals if v is not None]
            ax.bar(xs, vs, width=bar_w, color=color, edgecolor="black",
                   linewidth=0.6, zorder=3)
            for xi, v in zip(xs, vs):
                ax.annotate(fmt.format(v), xy=(xi, v), xytext=(0, 5),
                            textcoords="offset points", ha="center", va="bottom",
                            rotation=90, color=color)
        for xi, v in zip(x, fair):
            if v is None:
                ax_l.text(xi, 0.03, "n/a", ha="center", va="bottom",
                          color="#888888", fontsize=9)

        top = nice_ceil(max(v for v in tput if v is not None))
        ax_l.set_ylim(0, 1.30)
        ax_r.set_ylim(0, top * 1.24)
        ax_l.set_yticks(np.linspace(0, 1.0, 5))
        ax_r.set_yticks(np.linspace(0, top, 5))
        ax_l.set_ylabel(fair_label, color=FAIR_COLOR, fontweight="bold")
        ax_l.tick_params(axis="y", colors=FAIR_COLOR)
        ax_r.set_ylabel("Throughput (tok/s)", color=TPUT_COLOR, fontweight="bold")
        ax_r.tick_params(axis="y", colors=TPUT_COLOR)
        # Explicit on both: with n/a slots, matplotlib 3.3 widens ax_l's
        # xlim for the ticks but not the twin's, shifting the orange bars.
        for ax in (ax_l, ax_r):
            ax.set_xlim(-0.6, len(x) - 0.4)
        ax_l.set_xticks(x)
        ax_l.set_xticklabels([PAPER_LABELS[m] for m in PAPER_METHODS],
                             rotation=20, ha="right")
        ax_l.grid(axis="y", zorder=0)
        title = rf"$w_A:w_P = {w_a}:{w_p}$  (B = {cap})"
        if (case, demand) != ("mixed", "1:1"):
            title += f"\n{scen_label(case, demand)}"
        ax_l.set_title(title, pad=16)
        fig.tight_layout(pad=0.2)
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')}")
    for m, f, t in zip(PAPER_METHODS, fair, tput):
        print(f"    {PAPER_LABELS[m]:8s} fairness="
              + ("n/a" if f is None else f"{f:.3f}  tok/s={t:.0f}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, default=EXP_DIR / "results_t4")
    ap.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    ap.add_argument("--tune-b", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.exp_dir / "summary.jsonl")]
    groups = defaultdict(list)
    for r in rows:
        # Rows from before --case / --demand existed are mixed, 1:1.
        groups[(r["batch_cap"], r["weights_A_P"], r.get("case", "mixed"),
                r.get("demand_A_P", "1:1"), r["mode"])].append(r)
    def median(vals):
        vals = [x for x in vals if x is not None]   # be/sp have no gpu_check
        return statistics.median(vals) if vals else None

    med = {k: {f: median(r[f] for r in v)
               for f in ("P_tok_s", "A_tok_s", "agg_tok_s", "gpu_check_decode",
                         "gpu_check_total", "P_tbt_ms_p99", "prefill_time_frac")}
           for k, v in groups.items()}
    nan = float("nan")

    table = []
    for cap, wts, case, dem in sorted({k[:4] for k in med}):
        profile = Profile(args.profile, 128, CASES[case])
        m_A = profile.m_A(args.tune_b)

        def standalone(t):
            # A tenant alone does not depend on demand: fall back to the
            # same case's standalone run at any demand.
            m = med.get((cap, wts, case, dem, f"standalone_{t}"))
            if m is None:
                m = next((v for k, v in sorted(med.items())
                          if k[:3] == (cap, wts, case) and k[4] == f"standalone_{t}"), {})
            return m.get(f"{t}_tok_s")
        alone = {t: standalone(t) for t in ("P", "A")}
        if None in alone.values():
            print(f"[skip] B={cap} {wts} {case} demand {dem}: standalone runs missing")
            continue
        w_a, w_p = (float(x) for x in wts.split(":"))
        w = {"A": w_a, "P": w_p}
        for mode in SCHEDULERS:
            m = med.get((cap, wts, case, dem, mode))
            if m is None:
                continue
            norm = {t: m[f"{t}_tok_s"] / alone[t] / (w[t] / (w_a + w_p))
                    for t in ("P", "A")}
            jain = sum(norm.values()) ** 2 / (2 * sum(v * v for v in norm.values()))
            pred = (predict(mode, cap, w, profile, m_A, ratio(dem))
                    or {"P": None, "A": None})
            table.append({
                "scheduler": mode, "batch_cap": cap, "weights_A_P": wts, "case": case,
                "demand_A_P": dem,
                "P_tok_s": m["P_tok_s"], "A_tok_s": m["A_tok_s"],
                "P_norm_share": norm["P"], "A_norm_share": norm["A"],
                "P_norm_pred": pred["P"], "A_norm_pred": pred["A"],
                "jain": jain, "agg_tok_s": m["agg_tok_s"],
                "gpu_check_decode": m["gpu_check_decode"],
                "gpu_check_total": m["gpu_check_total"],
                "prefill_time_frac": m["prefill_time_frac"],
                "P_tbt_ms_p99": m["P_tbt_ms_p99"],
            })
        print(f"\nB={cap}  A:P={wts}  case={case} ({scen_label(case, dem)})  "
              f"standalone P={alone['P']:.0f}  A={alone['A']:.0f} tok/s")
        print(f"{'sched':10s}{'P tok/s':>9s}{'A tok/s':>9s}{'P norm':>8s}{'(pred)':>7s}"
              f"{'A norm':>8s}{'(pred)':>7s}{'Jain':>6s}{'agg':>7s}{'gpu':>6s}{'P p99':>8s}")
        for r in table:
            if (r["batch_cap"], r["weights_A_P"], r["case"],
                    r["demand_A_P"]) != (cap, wts, case, dem):
                continue
            def nn(v):
                return nan if v is None else v
            print(f"{LABELS[r['scheduler']]:10s}{r['P_tok_s']:9.0f}{r['A_tok_s']:9.0f}"
                  f"{r['P_norm_share']:8.2f}{nn(r['P_norm_pred']):7.2f}"
                  f"{r['A_norm_share']:8.2f}{nn(r['A_norm_pred']):7.2f}{r['jain']:6.3f}"
                  f"{r['agg_tok_s']:7.0f}{nn(r['gpu_check_decode']):6.2f}"
                  f"{nn(r['P_tbt_ms_p99']):8.0f}")

    if not table:
        return
    with open(args.exp_dir / "fairness.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(table[0]))
        wr.writeheader()
        wr.writerows(table)

    # Cross-case comparison: the same scheduler under each adapter assignment
    # and demand ratio.
    for cap, wts in sorted({(r["batch_cap"], r["weights_A_P"]) for r in table}):
        cases = sorted({(r["case"], r["demand_A_P"]) for r in table
                        if (r["batch_cap"], r["weights_A_P"]) == (cap, wts)},
                       key=lambda cd: (cd[1], list(CASES).index(cd[0])))
        if len(cases) < 2:
            continue
        by = {((r["case"], r["demand_A_P"]), r["scheduler"]): r for r in table
              if (r["batch_cap"], r["weights_A_P"]) == (cap, wts)}
        print(f"\nAcross cases, B={cap}  A:P={wts}  "
              f"(fairness in GPU time / in tokens; agg tok/s)")
        print(f"{'sched':10s}" + "".join(f"{scen_label(*c):>30s}" for c in cases))
        for mode in SCHEDULERS:
            if not any((c, mode) in by for c in cases):
                continue
            cells = []
            for c in cases:
                r = by.get((c, mode))
                cells.append(f"{'-':>30s}" if r is None else
                             f"{gpu_time_fairness(r):>14.3f} /{token_fairness(r):6.3f};"
                             f"{r['agg_tok_s']:7.0f}")
            print(f"{LABELS[mode]:10s}" + "".join(cells))

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#555555", "xtick.color": "#333333", "ytick.color": "#333333",
    })
    plot_dir = args.exp_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    for cap, wts, case, dem in sorted({(r["batch_cap"], r["weights_A_P"], r["case"],
                                        r["demand_A_P"]) for r in table}):
        sub = [r for r in table if (r["batch_cap"], r["weights_A_P"], r["case"],
                                    r["demand_A_P"]) == (cap, wts, case, dem)]
        x = np.arange(len(sub))
        bw = 0.36
        fig, ax = plt.subplots(figsize=(3.4, 2.1))
        ax.axhline(1.0, color="#888888", lw=0.6, ls="--", zorder=1)
        for i, t in enumerate(("P", "A")):
            off = (i - 0.5) * (bw + 0.02)
            vals = [r[f"{t}_norm_share"] for r in sub]
            ax.bar(x + off, vals, bw, color=TENANT_COLORS[t],
                   label=tenant_label(t, case), zorder=2)
            px = [(xi, r[f"{t}_norm_pred"]) for xi, r in zip(x + off, sub)
                  if r[f"{t}_norm_pred"] is not None]
            ax.scatter([p[0] for p in px], [p[1] for p in px], marker="_",
                       s=60, color="#222222", lw=1.0, zorder=3,
                       label="Profile prediction" if i == 0 else None)
            if t == "P":
                for xi, v in zip(x + off, vals):
                    ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                            fontsize=6, color="#333333")
        ax.set_xticks(x)
        # Seven groups at 3.4 in: wrap the two long labels so they don't collide.
        ax.set_xticklabels([{"tuned_vtc": "Tuned\nVTC", "bfq_stfq": "BFQ-\nSTFQ"}
                            .get(r["scheduler"], LABELS[r["scheduler"]]) for r in sub])
        ax.set_ylabel("Normalized share\n(1 = fair share of standalone)")
        top = max(max(r["P_norm_share"], r["A_norm_share"], r["A_norm_pred"] or 0)
                  for r in sub)
        ax.set_ylim(0, max(1.3, top + 0.15))
        ax.yaxis.grid(True, color="#dddddd", lw=0.4)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  ncol=3, handlelength=1.2, columnspacing=1.0)
        ax.set_xlabel(f"Scheduler (B = {cap}, weights A:P = {wts}"
                      + ("" if dem == "1:1" else f", demand A:P = {dem}") + ")")
        fig.tight_layout()
        tag = f"B{cap}_w{wts.replace(':', '-')}{case_suffix(case, dem)}"
        stem = plot_dir / f"norm_share_{tag}"
        fig.savefig(f"{stem}.pdf")
        fig.savefig(f"{stem}.png", dpi=200)
        plt.close(fig)
        print(f"wrote {stem}.pdf")
        plot_fairness_twin(sub, cap, wts, plot_dir / f"fairness_{tag}",
                           case=case, demand=dem)
        plot_fairness_twin(sub, cap, wts, plot_dir / f"fairness_tokens_{tag}",
                           fair_fn=token_fairness, fair_label="Fairness (tokens)",
                           case=case, demand=dem)


if __name__ == "__main__":
    main()
