#!/usr/bin/env python3
"""noisy_neighbor/tsfm — Plotting script.

Single-policy mode (--exp-dir points to one results dir):
    python experiments/noisy_neighbor/tsfm/plot.py \
        --exp-dir experiments/noisy_neighbor/tsfm/results/fifo_bwait_1ms

Comparison mode (--fifo-dir and --rr-dir both provided):
    python experiments/noisy_neighbor/tsfm/plot.py \
        --fifo-dir experiments/noisy_neighbor/tsfm/results/fifo_bwait_1ms \
        --rr-dir   experiments/noisy_neighbor/tsfm/results/round_robin_bwait_1ms

Plots produced:
  tsfm_victim_latency_vs_aggressor_rps.png  — single policy
  tsfm_victim_vs_aggressor.png              — single policy
  tsfm_fifo_vs_rr.png                       — comparison (requires both dirs)
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SERVING_DIR = Path(__file__).resolve().parents[3]

PALETTE = {
    "charcoal":   "#2F3640",
    "slate":      "#5C6773",
    "grid":       "#D9DEE5",
    "background": "#FAFBFC",
}
FIFO_COLOR   = "#D65F5F"   # red  — bad policy
RR_COLOR     = "#4878CF"   # blue — better policy
VICTIM_COLOR = "#4878CF"
AGG_COLOR    = "#D65F5F"


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    PALETTE["background"],
        "axes.edgecolor":    PALETTE["slate"],
        "axes.labelcolor":   PALETTE["charcoal"],
        "axes.linewidth":    0.9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "grid.color":        PALETTE["grid"],
        "grid.linestyle":    "--",
        "grid.linewidth":    0.7,
        "grid.alpha":        0.8,
        "xtick.color":       PALETTE["charcoal"],
        "ytick.color":       PALETTE["charcoal"],
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "savefig.facecolor": "white",
        "savefig.bbox":      "tight",
    })


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[Plot] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summary(path: Path):
    """Returns ({aggressor_rps: {role: row}}, meta)."""
    data: dict[float, dict[str, dict]] = defaultdict(dict)
    meta: dict = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            agg_rps = float(row["aggressor_rps"])
            role    = row["role"]
            data[agg_rps][role] = {
                k: float(v) if k not in
                   ("backbone", "victim_task", "aggressor_task", "role", "task")
                   else v
                for k, v in row.items()
            }
            meta.setdefault("victim_task",    row["victim_task"])
            meta.setdefault("aggressor_task", row["aggressor_task"])
            meta.setdefault("victim_rps",     float(row["victim_rps"]))
            meta.setdefault("backbone",       row["backbone"])
    return data, meta


def _victim_series(data, key: str) -> tuple[list[float], list[float]]:
    rps_vals = sorted(data.keys())
    vals     = [data[r]["victim"][key] for r in rps_vals]
    return rps_vals, vals


def _autoscale(vals_ms: list[float]) -> tuple[list[float], str]:
    if max(vals_ms) > 2000:
        return [v / 1000 for v in vals_ms], "s"
    return vals_ms, "ms"


# ---------------------------------------------------------------------------
# Plot 1: victim latency vs aggressor RPS (single policy)
# ---------------------------------------------------------------------------

def plot_victim_latency(data, meta, out_path: Path) -> None:
    rps_vals, p50 = _victim_series(data, "p50_lat_ms")
    _,         p99 = _victim_series(data, "p99_lat_ms")
    p50_sc, unit = _autoscale(p50)
    p99_sc, _    = _autoscale(p99)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(rps_vals, p50_sc, color=VICTIM_COLOR, linewidth=2.2,
            marker="o", markersize=6,
            markerfacecolor="white", markeredgewidth=1.5,
            markeredgecolor=VICTIM_COLOR, label="P50", zorder=3)
    ax.plot(rps_vals, p99_sc, color=VICTIM_COLOR, linewidth=2.2,
            marker="o", markersize=6,
            markerfacecolor="white", markeredgewidth=1.5,
            markeredgecolor=VICTIM_COLOR, linestyle="--", alpha=0.7,
            label="P99", zorder=3)
    ax.fill_between(rps_vals, p50_sc, p99_sc, color=VICTIM_COLOR, alpha=0.10)

    if 0.0 in data:
        base = data[0.0]["victim"]["p50_lat_ms"]
        base_sc = base / 1000 if unit == "s" else base
        ax.axhline(base_sc, color=PALETTE["slate"], linewidth=1.0, linestyle=":",
                   label=f"Baseline P50 ({base_sc:.1f}{unit})")

    ax.set_xlabel("Aggressor RPS (req/s)", fontsize=12, fontweight="semibold")
    ax.set_ylabel(f"Victim Latency ({unit})", fontsize=12, fontweight="semibold")
    ax.set_title(
        f"TSFM Noisy Neighbor — victim: {meta['victim_task']} @ "
        f"{meta['victim_rps']:.0f} rps  |  aggressor: {meta['aggressor_task']}",
        fontsize=11, fontweight="semibold", pad=8,
    )
    ax.set_xticks(rps_vals)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: victim vs aggressor side-by-side (single policy)
# ---------------------------------------------------------------------------

def plot_victim_vs_aggressor(data, meta, out_path: Path) -> None:
    rps_vals = sorted(r for r in data.keys() if r > 0)
    x, w = np.arange(len(rps_vals)), 0.3

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, role, color, label in [
        (axes[0], "victim",    VICTIM_COLOR, f"Victim ({meta['victim_task']})"),
        (axes[1], "aggressor", AGG_COLOR,    f"Aggressor ({meta['aggressor_task']})"),
    ]:
        p50 = [data[r][role]["p50_lat_ms"] for r in rps_vals]
        p99 = [data[r][role]["p99_lat_ms"] for r in rps_vals]
        p50_sc, unit = _autoscale(p50)
        p99_sc, _    = _autoscale(p99)

        ax.bar(x - w/2, p50_sc, width=w, color=color, alpha=0.88,
               edgecolor=PALETTE["charcoal"], linewidth=0.7, label="P50")
        ax.bar(x + w/2, p99_sc, width=w, color=color, alpha=0.45,
               edgecolor=PALETTE["charcoal"], linewidth=0.7, label="P99", hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(r)) for r in rps_vals], fontsize=10)
        ax.set_xlabel("Aggressor RPS (req/s)", fontsize=11, fontweight="semibold")
        ax.set_ylabel(f"Latency ({unit})", fontsize=11, fontweight="semibold")
        ax.set_title(label, fontsize=11, fontweight="semibold")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, frameon=False)

    fig.suptitle(
        f"TSFM Noisy Neighbor — backbone: {meta['backbone']}  "
        f"victim fixed @ {meta['victim_rps']:.0f} rps",
        fontsize=11, fontweight="semibold",
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: FIFO vs Round-Robin comparison (main paper figure)
# ---------------------------------------------------------------------------

def plot_fifo_vs_rr(fifo_data, rr_data, meta, out_path: Path) -> None:
    """Two subplots: victim p50 and victim p99, each with FIFO vs RR lines."""
    # Use only RPS points present in both datasets
    common_rps = sorted(set(fifo_data.keys()) & set(rr_data.keys()))

    fifo_p50 = [fifo_data[r]["victim"]["p50_lat_ms"] for r in common_rps]
    fifo_p99 = [fifo_data[r]["victim"]["p99_lat_ms"] for r in common_rps]
    rr_p50   = [rr_data[r]["victim"]["p50_lat_ms"]   for r in common_rps]
    rr_p99   = [rr_data[r]["victim"]["p99_lat_ms"]   for r in common_rps]

    # Auto-scale: if any value > 2000ms use seconds
    all_vals = fifo_p50 + fifo_p99 + rr_p50 + rr_p99
    use_s    = max(all_vals) > 2000
    scale    = 1000.0 if use_s else 1.0
    unit     = "s" if use_s else "ms"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    for ax, (fifo_vals, rr_vals, metric) in zip(axes, [
        (fifo_p50, rr_p50, "P50"),
        (fifo_p99, rr_p99, "P99"),
    ]):
        fifo_sc = [v / scale for v in fifo_vals]
        rr_sc   = [v / scale for v in rr_vals]

        ax.plot(common_rps, fifo_sc,
                color=FIFO_COLOR, linewidth=2.2, marker="o", markersize=7,
                markerfacecolor="white", markeredgewidth=1.8,
                markeredgecolor=FIFO_COLOR, label="FIFO", zorder=3)
        ax.plot(common_rps, rr_sc,
                color=RR_COLOR, linewidth=2.2, marker="s", markersize=7,
                markerfacecolor="white", markeredgewidth=1.8,
                markeredgecolor=RR_COLOR, label="Round-Robin", zorder=3)

        # Shade the gap (FIFO worse region)
        ax.fill_between(common_rps, rr_sc, fifo_sc,
                        where=[f >= r for f, r in zip(fifo_sc, rr_sc)],
                        color=FIFO_COLOR, alpha=0.08, label="FIFO overhead")

        # Baseline reference
        if 0.0 in fifo_data:
            base = fifo_data[0.0]["victim"][
                "p50_lat_ms" if metric == "P50" else "p99_lat_ms"
            ]
            ax.axhline(base / scale, color=PALETTE["slate"], linewidth=1.0,
                       linestyle=":", label=f"Baseline ({base/scale:.1f}{unit})")

        ax.set_xlabel("Aggressor RPS (req/s)", fontsize=12, fontweight="semibold")
        ax.set_ylabel(f"Victim {metric} Latency ({unit})", fontsize=12, fontweight="semibold")
        ax.set_title(f"Victim {metric}", fontsize=12, fontweight="semibold")
        ax.set_xticks(common_rps)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, frameon=False)

    fig.suptitle(
        f"FIFO vs Round-Robin — victim: {meta['victim_task']} @ "
        f"{meta['victim_rps']:.0f} rps  |  aggressor: {meta['aggressor_task']}  "
        f"|  backbone: {meta['backbone']}",
        fontsize=11, fontweight="semibold",
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Slowdown factor FIFO vs RR
# ---------------------------------------------------------------------------

def plot_slowdown_comparison(fifo_data, rr_data, meta, out_path: Path) -> None:
    """Bar chart: victim P50 slowdown relative to no-aggressor baseline."""
    if 0.0 not in fifo_data:
        print("[Plot] No baseline — skipping slowdown plot")
        return

    baseline    = fifo_data[0.0]["victim"]["p50_lat_ms"]
    common_rps  = sorted((set(fifo_data.keys()) & set(rr_data.keys())) - {0.0})
    fifo_slow   = [fifo_data[r]["victim"]["p50_lat_ms"] / baseline for r in common_rps]
    rr_slow     = [rr_data[r]["victim"]["p50_lat_ms"]   / baseline for r in common_rps]

    x, w = np.arange(len(common_rps)), 0.3
    fig, ax = plt.subplots(figsize=(8.0, 4.2))

    ax.bar(x - w/2, fifo_slow, width=w, color=FIFO_COLOR, alpha=0.88,
           edgecolor=PALETTE["charcoal"], linewidth=0.7, label="FIFO")
    ax.bar(x + w/2, rr_slow,   width=w, color=RR_COLOR,   alpha=0.88,
           edgecolor=PALETTE["charcoal"], linewidth=0.7, label="Round-Robin")
    ax.axhline(1.0, color=PALETTE["slate"], linewidth=1.2, linestyle="--",
               label="No interference (1×)")

    # Annotate FIFO bars with slowdown value
    for i, v in enumerate(fifo_slow):
        if v > 1.5:
            ax.text(i - w/2, v + 0.05, f"{v:.0f}×", ha="center", va="bottom",
                    fontsize=8, color=FIFO_COLOR, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(r)) for r in common_rps], fontsize=10)
    ax.set_xlabel("Aggressor RPS (req/s)", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Victim P50 Slowdown (×)", fontsize=12, fontweight="semibold")
    ax.set_title(
        f"Victim Slowdown: FIFO vs Round-Robin\n"
        f"victim: {meta['victim_task']} @ {meta['victim_rps']:.0f} rps  |  "
        f"aggressor: {meta['aggressor_task']}",
        fontsize=11, fontweight="semibold", pad=8,
    )
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    results_root = "experiments/noisy_neighbor/tsfm/results"
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir",  default=None,
                        help="Single-policy results dir.")
    parser.add_argument("--fifo-dir", default=os.path.join(results_root, "fifo_bwait_1ms"),
                        help="FIFO results dir (for comparison plot).")
    parser.add_argument("--rr-dir",   default=os.path.join(results_root, "round_robin_bwait_1ms"),
                        help="Round-robin results dir (for comparison plot).")
    parser.add_argument("--plot-dir", default=None,
                        help="Output directory for plots (default: inside --exp-dir or results/).")
    args = parser.parse_args()

    # --- Single-policy plots ---
    if args.exp_dir:
        result_root  = (SERVING_DIR / args.exp_dir).resolve()
        summary_path = result_root / "summary.csv"
        plot_dir     = Path(args.plot_dir).resolve() if args.plot_dir else result_root / "plots"
        if not summary_path.exists():
            print(f"[Error] Not found: {summary_path}")
            return 1
        data, meta = load_summary(summary_path)
        print(f"[Plot] Single policy: {summary_path}")
        plot_victim_latency(data, meta,
                            plot_dir / "tsfm_victim_latency_vs_aggressor_rps.png")
        plot_victim_vs_aggressor(data, meta,
                                 plot_dir / "tsfm_victim_vs_aggressor.png")

    # --- FIFO vs RR comparison plots ---
    fifo_path = (SERVING_DIR / args.fifo_dir).resolve() / "summary.csv"
    rr_path   = (SERVING_DIR / args.rr_dir).resolve()   / "summary.csv"

    if fifo_path.exists() and rr_path.exists():
        fifo_data, meta = load_summary(fifo_path)
        rr_data,   _    = load_summary(rr_path)
        plot_dir = (
            Path(args.plot_dir).resolve() if args.plot_dir
            else fifo_path.parent.parent / "plots"
        )
        print(f"[Plot] FIFO vs RR comparison")
        plot_fifo_vs_rr(fifo_data, rr_data, meta,
                        plot_dir / "tsfm_fifo_vs_rr.png")
        plot_slowdown_comparison(fifo_data, rr_data, meta,
                                 plot_dir / "tsfm_slowdown_fifo_vs_rr.png")
    else:
        if not fifo_path.exists():
            print(f"[Info] FIFO results not found: {fifo_path} — skipping comparison")
        if not rr_path.exists():
            print(f"[Info] RR results not found: {rr_path} — skipping comparison")

    return 0


if __name__ == "__main__":
    apply_paper_style()
    raise SystemExit(main())
