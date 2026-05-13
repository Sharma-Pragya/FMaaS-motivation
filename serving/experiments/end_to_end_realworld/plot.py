#!/usr/bin/env python3
"""All plots for the end-to-end real-world experiment.

Reads deployments/{regime}_N{N}/placement_summary.json for every scenario.

Usage (from serving/):
    python -m experiments.end_to_end_realworld.plot
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

from experiments.end_to_end_realworld import user_config as uc

DEP_DIR = Path(__file__).resolve().parent / "deployments"
OUT_DIR = Path(__file__).resolve().parent


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_summary(regime: str, n: int) -> dict | None:
    p = DEP_DIR / f"{regime}_N{n}" / "placement_summary.json"
    return json.loads(p.read_text()) if p.is_file() else None


def _load_latencies(regime: str, n: int, method: str) -> np.ndarray | None:
    p = OUT_DIR / "results" / f"{regime}_N{n}" / method / "request_latency_results.csv"
    if not p.is_file():
        return None
    
    lats = []
    try:
        with open(p, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lats.append(float(row['end_to_end_latency(ms)']))
                except (ValueError, KeyError):
                    pass
    except Exception:
        pass
    
    if not lats:
        return None
    return np.array(lats)


def _get_batch_size_rw(regime: str, n: int, method: str) -> float | None:
    p = OUT_DIR / "results" / f"{regime}_N{n}" / method / "request_latency_results.csv"
    if not p.is_file():
        return None
    try:
        df = pd.read_csv(p, usecols=["device", "device_start_time"])
        if df.empty:
            return None
        g = df.groupby(["device", "device_start_time"]).size().rename("bs")
        d = df.merge(g, on=["device", "device_start_time"])
        bs_rw = float((d["bs"] * d["bs"]).sum() / d["bs"].sum())
        return bs_rw
    except Exception:
        return None


def _count_deployments(regime: str, n: int, method: str) -> int | None:
    """Number of backbone-process deployments in the method's plan."""
    p = DEP_DIR / f"{regime}_N{n}" / f"{method}.json"
    if not p.is_file():
        return None
    plan = json.loads(p.read_text())
    return sum(len(site.get("deployments", [])) for site in plan.get("sites", []))


# ── plot 1: placed-count per (regime, N, method) ─────────────────────────────

def plot_placed_counts() -> None:
    regimes = uc.experiment['load_regimes']
    ns      = uc.experiment['n_tasks_sweep']
    methods = uc.conditions

    fig, axes = plt.subplots(1, len(regimes), figsize=(4 * len(regimes), 4.5),
                             sharey=True)
    if len(regimes) == 1:
        axes = [axes]
    width = 0.8 / len(methods)
    x = np.arange(len(ns))

    for ax, regime in zip(axes, regimes):
        for i, m in enumerate(methods):
            vals = []
            for n in ns:
                data = _load_summary(regime, n)
                placed = data.get(m, {}).get('placed_count', 0) if data else 0
                # Store both placed and total requested tasks for later annotation
                vals.append(placed)
            bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=m)
            for j, v in enumerate(vals):
                ax.text(
                    bars[j].get_x() + bars[j].get_width() / 2,
                    v + 0.2, str(v), ha='center', va='bottom',
                    fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(ns)
        ax.set_xlabel("N (requested tasks)")
        ax.set_title(f"regime = {regime}")
        # Show the RPS range for this regime from user config
        rps_low, rps_high = uc.experiment['rate_bands_req_per_s'][regime]
        ax.text(0.5, 1.08, f"RPS: {rps_low:.1f}–{rps_high:.1f}",
                transform=ax.transAxes, ha='center', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(ns) * 1.15)

    axes[0].set_ylabel("tasks placed")
    axes[-1].legend(loc='upper left')
    fig.suptitle("FMaaS vs Clipper — tasks placed per scenario")
    fig.tight_layout()
    out = OUT_DIR / "placement_counts.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] saved → {out}")


# ── plot: number of deployments per method ───────────────────────────────────

def plot_deployment_counts() -> None:
    """How many backbone processes each method ends up running.

    FMaaS shares backbones across tasks, so this should be ≤ Clipper's count
    for the same scenario.  Helps explain memory pressure: more deployments
    means more backbone copies in GPU memory.
    """
    regimes = uc.experiment['load_regimes']
    ns      = uc.experiment['n_tasks_sweep']
    methods = uc.conditions

    fig, axes = plt.subplots(1, len(regimes), figsize=(4 * len(regimes), 4.5),
                             sharey=True)
    if len(regimes) == 1:
        axes = [axes]
    width = 0.8 / len(methods)
    x = np.arange(len(ns))

    for ax, regime in zip(axes, regimes):
        for i, m in enumerate(methods):
            vals = [(_count_deployments(regime, n, m) or 0) for n in ns]
            bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=m)
            for j, v in enumerate(vals):
                ax.text(bars[j].get_x() + bars[j].get_width() / 2,
                        v + 0.2, str(v), ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(ns)
        ax.set_xlabel("N (requested tasks)")
        ax.set_title(f"regime = {regime}")
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel("# backbone deployments")
    axes[-1].legend(loc='upper left')
    fig.suptitle("Backbone deployments per method (more deployments ⇒ more GPU memory used)")
    fig.tight_layout()
    out = OUT_DIR / "deployment_counts.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] saved → {out}")


# ── plot: bottleneck breakdown of rejections ─────────────────────────────────

_BN_ORDER  = ['memory', 'memory (restricted candidate pool)', 'compute', 'other']
_BN_COLORS = {
    'memory':                              '#d62728',
    'memory (restricted candidate pool)':  '#ff9896',
    'compute':                             '#1f77b4',
    'other':                               '#7f7f7f',
}

def plot_bottleneck_breakdown() -> None:
    regimes = uc.experiment['load_regimes']
    ns      = uc.experiment['n_tasks_sweep']
    methods = uc.conditions

    fig, axes = plt.subplots(len(regimes), len(methods),
                             figsize=(4 * len(methods), 3.2 * len(regimes)),
                             sharex=True, sharey=True)
    if len(regimes) == 1:
        axes = np.array([axes])
    if len(methods) == 1:
        axes = axes.reshape(-1, 1)

    x = np.arange(len(ns))
    for ri, regime in enumerate(regimes):
        for ci, m in enumerate(methods):
            ax = axes[ri, ci]
            stacks = {bn: np.zeros(len(ns)) for bn in _BN_ORDER}
            for j, n in enumerate(ns):
                data = _load_summary(regime, n) or {}
                counts = (data.get(m, {}) or {}).get('bottleneck_counts', {}) or {}
                for bn, cnt in counts.items():
                    key = bn if bn in stacks else 'other'
                    stacks[key][j] += cnt

            bottom = np.zeros(len(ns))
            for bn in _BN_ORDER:
                vals = stacks[bn]
                if vals.sum() == 0:
                    continue
                ax.bar(x, vals, bottom=bottom, label=bn, color=_BN_COLORS[bn])
                bottom += vals

            ax.set_title(f"{regime} / {m}", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(ns)
            ax.grid(axis='y', alpha=0.3)

    axes[-1, 0].set_xlabel("N")
    axes[0, 0].set_ylabel("rejected tasks")
    # Legend: collect once
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    if handles:
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 1.02), ncol=len(labels), fontsize=9)
    fig.suptitle("Rejection bottlenecks per method × regime × N", y=1.05)
    fig.tight_layout()
    out = OUT_DIR / "bottleneck_breakdown.pdf"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] saved → {out}")


# ── plot: latency ────────────────────────────────────────────────────────────

SERIES_COLORS = {
    "no_sharing": "#888888",
    "fmaas":      "#E06C75",
}
SERIES_LABELS = {
    "no_sharing": "Clipper",
    "fmaas":      "FMaaS",
}
SERIES_LINESTYLE = {
    "no_sharing": "-.",
    "fmaas":      "-",
}
SERIES_MARKER = {
    "no_sharing": "D",
    "fmaas":      "o",
}

def apply_paper_style() -> None:
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
        "font.family":       "sans-serif",
        "font.size":         8,
        "axes.titlesize":    9,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   7,
        "lines.linewidth":   1.3,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "savefig.facecolor": "white",
        "savefig.bbox":      "tight",
    })

def _plain_number_axis(ax: plt.Axes, axis: str = "y") -> None:
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    elif axis == "x":
        ax.xaxis.set_major_formatter(formatter)

def plot_latency() -> None:
    apply_paper_style()
    regimes = uc.experiment['load_regimes']
    ns      = uc.experiment['n_tasks_sweep']
    methods = uc.conditions

    fig_mean, axes_mean = plt.subplots(1, len(regimes), figsize=(max(4.4, 2.5 * len(regimes)), 2.5), sharey=False)
    if len(regimes) == 1:
        axes_mean = [axes_mean]
    else:
        axes_mean = axes_mean.flatten()

    fig_p99, axes_p99 = plt.subplots(1, len(regimes), figsize=(max(4.4, 2.5 * len(regimes)), 2.5), sharey=False)
    if len(regimes) == 1:
        axes_p99 = [axes_p99]
    else:
        axes_p99 = axes_p99.flatten()

    for ri, regime in enumerate(regimes):
        ax_m = axes_mean[ri]
        ax_p = axes_p99[ri]
        
        for m in methods:
            means = []
            p99s = []
            for n in ns:
                lats = _load_latencies(regime, n, m)
                if lats is not None and len(lats) > 0:
                    means.append(np.mean(lats))
                    p99s.append(np.percentile(lats, 99))
                else:
                    means.append(np.nan)
                    p99s.append(np.nan)
            
            # plot mean
            ax_m.plot(ns, means, color=SERIES_COLORS.get(m, "black"), linestyle=SERIES_LINESTYLE.get(m, "-"),
                      marker=SERIES_MARKER.get(m, "o"), markersize=4.0, markeredgewidth=0.0,
                      label=SERIES_LABELS.get(m, m))
            # plot p99
            ax_p.plot(ns, p99s, color=SERIES_COLORS.get(m, "black"), linestyle=SERIES_LINESTYLE.get(m, "-"),
                      marker=SERIES_MARKER.get(m, "o"), markersize=4.0, markeredgewidth=0.0,
                      label=SERIES_LABELS.get(m, m))
        
        for ax in (ax_m, ax_p):
            ax.set_xticks(ns)
            ax.set_xticklabels([str(n) for n in ns])
            ax.set_xlabel("Number of applications")
            ax.set_title(f"regime = {regime}")
            ax.grid(True, axis="y")
            ax.set_ylim(bottom=0.0)
            _plain_number_axis(ax, "y")

        if ri == 0:
            ax_m.set_ylabel("Mean RT (ms)")
            ax_p.set_ylabel("p99 RT (ms)")
            
    axes_mean[-1].legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    axes_p99[-1].legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    
    fig_mean.tight_layout()
    out_m = OUT_DIR / "latency_vs_napps_mean.pdf"
    fig_mean.savefig(out_m)
    plt.close(fig_mean)
    print(f"[plot] saved → {out_m}")

    fig_p99.tight_layout()
    out_p = OUT_DIR / "latency_vs_napps_p99.pdf"
    fig_p99.savefig(out_p)
    plt.close(fig_p99)
    print(f"[plot] saved → {out_p}")


# ── plot: batch size ─────────────────────────────────────────────────────────

def plot_batch_size() -> None:
    apply_paper_style()
    regimes = uc.experiment['load_regimes']
    ns      = uc.experiment['n_tasks_sweep']
    methods = uc.conditions

    fig_bs, axes_bs = plt.subplots(1, len(regimes), figsize=(max(4.4, 2.5 * len(regimes)), 2.5), sharey=False)
    if len(regimes) == 1:
        axes_bs = [axes_bs]
    else:
        axes_bs = axes_bs.flatten()

    for ri, regime in enumerate(regimes):
        ax = axes_bs[ri]
        for m in methods:
            bss = []
            for n in ns:
                bs = _get_batch_size_rw(regime, n, m)
                bss.append(bs if bs is not None else np.nan)
            
            ax.plot(ns, bss, color=SERIES_COLORS.get(m, "black"), linestyle=SERIES_LINESTYLE.get(m, "-"),
                      marker=SERIES_MARKER.get(m, "o"), markersize=4.0, markeredgewidth=0.0,
                      label=SERIES_LABELS.get(m, m))
            
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("Number of applications")
        ax.set_title(f"regime = {regime}")
        ax.grid(True, axis="y")
        ax.set_ylim(bottom=1.0)
        _plain_number_axis(ax, "y")

        if ri == 0:
            ax.set_ylabel("Req-weighted Batch Size")
            
    axes_bs[-1].legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    
    fig_bs.tight_layout()
    out = OUT_DIR / "batch_size_vs_napps.pdf"
    fig_bs.savefig(out)
    plt.close(fig_bs)
    print(f"[plot] saved → {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    plot_placed_counts()
    plot_deployment_counts()
    plot_bottleneck_breakdown()
    plot_latency()
    plot_batch_size()


if __name__ == "__main__":
    main()
