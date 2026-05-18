"""Quick plot of per-second invocation rate per HashOwner (day 1).

Reads the preprocessed npz produced by `maf_preprocess.py` and renders the
per-second rate (= per-minute count / 60) for the top-N and bottom-N owners
above a minimum-invocations cutoff.

Usage (from serving/):
    python -m traces.maf_plot_preprocess              # defaults: top-8 + bottom-8, min_inv=200
    python -m traces.maf_plot_preprocess --top 16 --min-inv 500 --out plot.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from traces.maf_preprocess import load_hashowner_minutes

MINUTES_PER_DAY = 1440
SECONDS_PER_MINUTE = 60


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--day",     type=int,   default=1)
    p.add_argument("--top",     type=int,   default=8)
    p.add_argument("--min-inv", type=int,   default=200)
    p.add_argument("--out",     type=str,   default="traces/azurefunctions/preprocessed/rate_top_bottom.png")
    args = p.parse_args()

    data = load_hashowner_minutes(day=args.day)
    owner_ids = data["owner_ids"]
    minutes   = data["minutes"]
    n_req     = data["n_req"]

    mask = n_req >= args.min_inv
    owner_ids = owner_ids[mask]
    minutes   = minutes[mask]
    n_req     = n_req[mask]

    # Rate = n_req / active-window (sec).
    nonzero = minutes > 0
    first   = np.argmax(nonzero, axis=1)
    last    = (MINUTES_PER_DAY - 1) - np.argmax(nonzero[:, ::-1], axis=1)
    window_s = np.maximum((last - first + 1).astype(float) * SECONDS_PER_MINUTE,
                          float(SECONDS_PER_MINUTE))
    rate    = n_req.astype(float) / window_s

    order = np.argsort(rate)[::-1]
    top_idx = order[:args.top]
    bot_idx = order[-args.top:][::-1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    t_min = np.arange(MINUTES_PER_DAY)

    for ax, idx, label in [(axes[0], top_idx, "high"), (axes[1], bot_idx, "low")]:
        for i in idx:
            per_sec = minutes[i] / SECONDS_PER_MINUTE  # avg req/sec within each minute
            ax.plot(t_min, per_sec, lw=0.7,
                    label=f"{str(owner_ids[i])[:8]}… (n={int(n_req[i])}, "
                          f"r={rate[i]:.3f}/s)")
        ax.set_ylabel("req / sec")
        ax.set_title(f"{label}-load HashOwners (top {args.top} by rate)"
                     if label == "high"
                     else f"{label}-load HashOwners (bottom {args.top} by rate)")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("minute of day 1")
    fig.suptitle(f"Azure Functions day {args.day:02d} — "
                 f"HashOwner per-second invocation rate (min_inv={args.min_inv})")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"[plot] saved → {out}")


if __name__ == "__main__":
    main()
