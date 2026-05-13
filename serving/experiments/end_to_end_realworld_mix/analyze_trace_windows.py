#!/usr/bin/env python3
"""Per-owner variability of 600-second window rates.

For every (owner, day) row in the preprocessed MAF data, slide a 600 s (10 min)
window minute by minute across that day. Each owner produces a time series
of window-rates (req/s). We summarise each owner by two numbers:

    mean_rate  = mean of all per-window rates over the day            (req/s)
    cv         = std(per-window rates) / mean_rate
                 — how much an owner's 600 s rate swings depending on
                   which window you pick. cv ≈ 0 means flat all day.
                   cv >> 1 means most of the day is idle with brief bursts.

Outputs (in analysis/):
  per_owner_cv_hist.pdf       histogram of owner cv (one bar = many owners).
  mean_rate_vs_cv.pdf         scatter of mean_rate (log x) vs cv,
                              colored by regime band — shows which owners
                              are "bursty" and which are "flat".
  example_owner_traces.pdf    a few example owners: their per-window
                              rate vs. window-start minute, picked across
                              the (mean_rate, cv) plane.
  per_owner_summary.csv       one row per (owner_id, day): mean_rate,
                              cv, max_rate, regime.

Usage (from serving/):
  python -m experiments.end_to_end_realworld_mix.analyze_trace_windows \
      [--n-days 14] [--window-min 10]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from traces.maf_preprocess import load_hashowner_minutes
from experiments.end_to_end_realworld_mix import user_config as cfg

MINUTES_PER_DAY = 1440
SECONDS_PER_MINUTE = 60
OUT_DIR = Path(__file__).resolve().parent / "analysis"

BAND_COLOR = {"low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728",
              "below": "#9467bd", "above": "#7f7f7f"}


def _classify(mean_rate: np.ndarray,
              bands: Dict[str, Tuple[float, float]]) -> np.ndarray:
    out = np.full(mean_rate.shape, "above", dtype=object)
    out[mean_rate <= 0.1] = "below"
    for band, (lo, hi) in bands.items():
        out[(mean_rate > lo) & (mean_rate <= hi)] = band
    return out


def _per_owner_stats(M: np.ndarray, w: int):
    """Per-owner mean window rate and CV across windows.

    M:    (n_owners, 1440) int32 for one day.
    Returns (mean_rate, cv, max_rate, window_rates) — last is full
    (n_owners, n_windows) array for plotting examples.
    """
    csum = np.cumsum(M.astype(np.int64), axis=1)
    a = csum[:, w - 1:]
    b = np.concatenate([np.zeros((csum.shape[0], 1), dtype=csum.dtype),
                        csum[:, :-w]], axis=1)
    win_count = (a - b).astype(np.float64)
    window_rates = win_count / (w * SECONDS_PER_MINUTE)  # req/s
    mean_rate = window_rates.mean(axis=1)
    std_rate  = window_rates.std(axis=1)
    cv = np.divide(std_rate, mean_rate,
                   out=np.zeros_like(mean_rate),
                   where=mean_rate > 0)
    max_rate = window_rates.max(axis=1)
    return mean_rate, cv, max_rate, window_rates


def _plot_cv_hist(cv: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    # Clip very large CVs to a readable range; show count above as overflow bar.
    clip = 10.0
    cv_clip = np.minimum(cv, clip)
    ax.hist(cv_clip, bins=60, color="#1f77b4", edgecolor="white")
    n_over = int((cv > clip).sum())
    ax.set_xlabel("CV of 600 s window rates within one day "
                  "(std / mean)")
    ax.set_ylabel("# owners")
    ax.set_title(f"Per-owner variability of 600 s rate "
                 f"({len(cv):,} owners; {n_over:,} clipped at {clip:g})")
    ax.axvline(1.0, ls="--", color="gray", lw=1,
               label="CV=1 (std equals mean)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_mean_vs_cv(mean_rate: np.ndarray, cv: np.ndarray,
                     regime: np.ndarray, out_path: Path,
                     bands: Dict[str, Tuple[float, float]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    # plot order so low/medium/high land on top
    order = ["below", "above", "low", "medium", "high"]
    for r in order:
        mask = regime == r
        if not mask.any():
            continue
        ax.scatter(mean_rate[mask], cv[mask], s=4, alpha=0.35,
                   color=BAND_COLOR.get(r, "#444444"),
                   label=f"{r} (n={int(mask.sum())})")
    for band, (lo, hi) in bands.items():
        ax.axvline(lo, ls=":", color=BAND_COLOR.get(band), lw=0.7, alpha=0.6)
        ax.axvline(hi, ls=":", color=BAND_COLOR.get(band), lw=0.7, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Owner mean 600 s rate (req/s, log)")
    ax.set_ylabel("CV across 600 s windows (within one day)")
    ax.set_title("Owner-level variability vs. mean rate "
                 "— top-right = high-rate AND bursty")
    ax.set_ylim(0, min(10, np.nanpercentile(cv, 99) + 1))
    ax.grid(alpha=0.3, which="both")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_example_traces(rates_all: np.ndarray, mean_rate: np.ndarray,
                         cv: np.ndarray, regime: np.ndarray,
                         out_path: Path, w: int) -> None:
    """Pick 6 owners spanning (mean_rate, cv) and plot their window-rate
    timeseries within their day."""
    n_per_band = 2
    picks = []
    for band in ("low", "medium", "high"):
        idx_band = np.where(regime == band)[0]
        if idx_band.size == 0:
            continue
        # one steady (low CV) and one bursty (high CV) example
        cvs = cv[idx_band]
        order = idx_band[np.argsort(cvs)]
        picks.append((band, "steady",  order[len(order) // 20] if len(order) > 20 else order[0]))
        picks.append((band, "bursty",  order[-1]))
    if not picks:
        return

    n_win = rates_all.shape[1]
    x = np.arange(n_win)  # minute-of-day for the window start
    fig, axes = plt.subplots(len(picks), 1,
                             figsize=(10, 1.8 * len(picks)),
                             sharex=True)
    if len(picks) == 1:
        axes = [axes]
    for ax, (band, kind, i) in zip(axes, picks):
        ax.plot(x, rates_all[i], color=BAND_COLOR[band], lw=1.0)
        ax.set_ylabel("req/s")
        ax.set_title(f"{band} band, {kind} owner — "
                     f"mean={mean_rate[i]:.2f} req/s, "
                     f"max={rates_all[i].max():.2f}, cv={cv[i]:.2f}",
                     fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel(
        f"Window start (minute of day; window width = {w} min = "
        f"{w*SECONDS_PER_MINUTE} s)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_summary_csv(records: list, out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["day", "owner_idx_in_day", "mean_rate_rps",
                     "cv", "max_rate_rps", "regime"])
        wr.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-days", type=int,
                        default=int(cfg.experiment.get("maf_n_days", 14)))
    parser.add_argument("--window-min", type=int, default=10,
                        help="Window width in minutes (10 = 600 s).")
    parser.add_argument("--min-inv", type=int,
                        default=int(cfg.experiment.get("maf_min_invocations", 10)))
    args = parser.parse_args()

    bands = {b: (float(lo), float(hi))
             for b, (lo, hi) in cfg.experiment["rate_bands_req_per_s"].items()}

    OUT_DIR.mkdir(exist_ok=True)

    all_mean, all_cv, all_max, all_regime = [], [], [], []
    csv_records = []
    # We only keep example-trace data from day 1 to bound memory.
    keep_rates_for_examples = None
    keep_regime_for_examples = None
    keep_mean_for_examples = None
    keep_cv_for_examples = None

    for day in range(1, args.n_days + 1):
        z = load_hashowner_minutes(day=day)
        M = z["minutes"].astype(np.int32)
        totals = M.sum(axis=1, dtype=np.int64)
        keep = totals >= args.min_inv
        M = M[keep]
        if M.shape[0] == 0:
            continue
        mean_rate, cv, max_rate, rates_all = _per_owner_stats(M, args.window_min)
        regime = _classify(mean_rate, bands)
        all_mean.append(mean_rate)
        all_cv.append(cv)
        all_max.append(max_rate)
        all_regime.append(regime)
        for i in range(M.shape[0]):
            csv_records.append((day, i, float(mean_rate[i]), float(cv[i]),
                                float(max_rate[i]), str(regime[i])))
        if day == 1:
            keep_rates_for_examples = rates_all
            keep_regime_for_examples = regime
            keep_mean_for_examples = mean_rate
            keep_cv_for_examples = cv
        print(f"[analyze] day {day}: {M.shape[0]} owners  "
              f"median cv={np.median(cv):.2f}  "
              f"max cv={cv.max():.2f}")

    mean_rate = np.concatenate(all_mean)
    cv        = np.concatenate(all_cv)
    regime    = np.concatenate(all_regime)

    _plot_cv_hist(cv, OUT_DIR / "per_owner_cv_hist.pdf")
    _plot_mean_vs_cv(mean_rate, cv, regime,
                     OUT_DIR / "mean_rate_vs_cv.pdf", bands)
    if keep_rates_for_examples is not None:
        _plot_example_traces(keep_rates_for_examples,
                             keep_mean_for_examples,
                             keep_cv_for_examples,
                             keep_regime_for_examples,
                             OUT_DIR / "example_owner_traces.pdf",
                             args.window_min)
    _save_summary_csv(csv_records, OUT_DIR / "per_owner_summary.csv")

    print(f"\n[analyze] wrote plots + CSV to {OUT_DIR}")
    print(f"[analyze] total owners across {args.n_days} day(s): {len(cv):,}")
    print(f"[analyze] cv distribution: "
          f"p50={np.median(cv):.2f}  "
          f"p90={np.percentile(cv, 90):.2f}  "
          f"p99={np.percentile(cv, 99):.2f}  "
          f"max={cv.max():.2f}")
    # Per regime cv summary
    for r in ["low", "medium", "high"]:
        mask = regime == r
        if mask.any():
            print(f"  regime={r:6s} n={int(mask.sum()):6d}  "
                  f"cv: p50={np.median(cv[mask]):.2f}  "
                  f"p90={np.percentile(cv[mask], 90):.2f}  "
                  f"max={cv[mask].max():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
