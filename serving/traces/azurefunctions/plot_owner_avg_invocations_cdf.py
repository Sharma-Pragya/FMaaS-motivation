"""CDF of average invocations per minute, per HashOwner — one subplot per day.

For each day:
  - aggregate per HashOwner: total invocations summed across all functions
    and minutes
  - divide by 1440 to get avg invocations per minute for that owner
  - CDF over unique owners
"""

import argparse
import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MINUTE_COLS = [str(i) for i in range(1, 1441)]


def owner_avg_invocations_per_minute(path):
    df = pd.read_csv(path, usecols=["HashOwner"] + MINUTE_COLS)
    for c in MINUTE_COLS:
        if c not in df.columns:
            df[c] = 0
    totals = df[MINUTE_COLS].fillna(0).astype(np.int64).sum(axis=1)
    per_owner = totals.groupby(df["HashOwner"]).sum()
    return (per_owner / 1440.0).values  # one value per unique owner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--out", default=os.path.join(
        HERE, "plots", "owner_avg_invocations_per_minute_cdf.png"))
    parser.add_argument("--logx", action="store_true", default=True,
                        help="log-scale x-axis (default on; heavy tail)")
    parser.add_argument("--no-logx", dest="logx", action="store_false")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(
        HERE, "invocations_per_function_md.anon.d*.csv")))[: args.days]
    if not files:
        raise SystemExit("No invocation CSVs found.")

    n = len(files)
    ncols = 4 if n > 6 else min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    for idx, f in enumerate(files):
        print(f"Processing {os.path.basename(f)}")
        vals = owner_avg_invocations_per_minute(f)
        vals = vals[vals > 0]
        vals_sorted = np.sort(vals)
        cdf = np.arange(1, len(vals_sorted) + 1) / len(vals_sorted)
        ax = axes[idx // ncols, idx % ncols]
        ax.plot(vals_sorted, cdf)
        ax.set_title(f"day {idx + 1:02d}  "
                     f"(owners={len(vals)}, median={np.median(vals):.2f}/min)")
        ax.grid(True, which="both", alpha=0.3)
        if args.logx:
            ax.set_xscale("log")

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    for r in range(nrows):
        axes[r, 0].set_ylabel("CDF over owners")
    for c in range(ncols):
        axes[-1, c].set_xlabel("Avg invocations per minute (per HashOwner)")

    fig.suptitle("Per-HashOwner average invocations per minute — CDF per day")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
