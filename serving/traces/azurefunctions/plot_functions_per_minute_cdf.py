"""For each day, plot the CDF of 'active entities per minute'.

For minute m, count how many distinct entities (HashFunction, HashApp,
HashOwner) had >=1 invocation in that minute. That gives 1440 values per
day. Plot the CDF of those values as one subplot per day, and produce one
figure per granularity.
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
KEYS = ["HashOwner", "HashApp", "HashFunction"]


def active_per_minute(path):
    """Return dict {key: array(1440)} of distinct active entities per minute."""
    df = pd.read_csv(path, usecols=KEYS + MINUTE_COLS)
    for c in MINUTE_COLS:
        if c not in df.columns:
            df[c] = 0
    mat = df[MINUTE_COLS].fillna(0).to_numpy(dtype=np.int32) > 0  # (F, 1440)
    out = {}
    for key in KEYS:
        # For each minute, count distinct keys with any active function.
        # Group rows by key and OR across rows, then sum across keys.
        # Implementation: map key -> integer code, then for each minute build
        # a mask of active rows and count unique codes among them.
        codes, _ = pd.factorize(df[key], sort=False)
        counts = np.zeros(1440, dtype=np.int64)
        # Per-minute loop is acceptable (1440 iters); each uses np.unique on
        # codes[mask] which is fast.
        for m in range(1440):
            mask = mat[:, m]
            if mask.any():
                counts[m] = np.unique(codes[mask]).size
        out[key] = counts
    return out


def plot_cdf_grid(per_day_counts, key, out_path, logx):
    n = len(per_day_counts)
    ncols = 4 if n > 6 else min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    for idx, counts in enumerate(per_day_counts):
        counts_sorted = np.sort(counts)
        cdf = np.arange(1, len(counts_sorted) + 1) / len(counts_sorted)
        ax = axes[idx // ncols, idx % ncols]
        ax.plot(counts_sorted, cdf)
        ax.set_title(f"day {idx + 1:02d}  (median={np.median(counts):.0f})")
        ax.grid(True, alpha=0.3)
        if logx:
            ax.set_xscale("log")

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    for r in range(nrows):
        axes[r, 0].set_ylabel("CDF over minutes")
    for c in range(ncols):
        axes[-1, c].set_xlabel(f"Active {key} per minute")

    fig.suptitle(f"CDF of active {key} per minute, per day")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--outdir", default=os.path.join(HERE, "plots"))
    parser.add_argument("--logx", action="store_true",
                        help="log-scale x-axis")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(
        HERE, "invocations_per_function_md.anon.d*.csv")))[: args.days]
    if not files:
        raise SystemExit("No invocation CSVs found.")

    by_key = {k: [] for k in KEYS}
    for f in files:
        print(f"Processing {os.path.basename(f)}")
        day = active_per_minute(f)
        for k in KEYS:
            by_key[k].append(day[k])

    for k in KEYS:
        out_path = os.path.join(args.outdir,
                                f"active_{k.lower()}_per_minute_cdf.png")
        plot_cdf_grid(by_key[k], k, out_path, args.logx)


if __name__ == "__main__":
    main()
