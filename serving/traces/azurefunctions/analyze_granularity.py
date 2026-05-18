"""Compare HashOwner vs HashApp vs HashFunction as the unit of work.

For each granularity, computes the distribution of total invocations across
the 14-day trace and reports:
  - count of unique entities
  - summary stats (min/median/p90/p99/max/mean) of invocations per entity
  - skew (top-1%, top-10% share of all invocations)
  - CDF and Lorenz plots overlayed across the three granularities

Use this to decide whether your experiment should map tasks to owners,
apps, or individual functions.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MINUTE_COLS = [str(i) for i in range(1, 1441)]
KEYS = ["HashOwner", "HashApp", "HashFunction"]


def load_totals(path):
    """Return df with HashOwner, HashApp, HashFunction, total_invocations."""
    df = pd.read_csv(path)
    for c in MINUTE_COLS:
        if c not in df.columns:
            df[c] = 0
    totals = df[MINUTE_COLS].fillna(0).astype(np.int64).sum(axis=1)
    return pd.DataFrame({
        "HashOwner": df["HashOwner"],
        "HashApp": df["HashApp"],
        "HashFunction": df["HashFunction"],
        "total": totals,
    })


def summarize(series, label):
    s = series.values.astype(np.int64)
    s_sorted = np.sort(s)[::-1]
    total = s_sorted.sum()
    n = len(s_sorted)
    top1 = s_sorted[: max(1, n // 100)].sum() / total
    top10 = s_sorted[: max(1, n // 10)].sum() / total
    print(f"\n== {label} ==")
    print(f"  unique entities : {n:,}")
    print(f"  total invocs    : {total:,}")
    print(f"  mean / entity   : {s.mean():,.1f}")
    print(f"  median          : {np.median(s):,.0f}")
    print(f"  p90 / p99 / max : {np.percentile(s, 90):,.0f} / "
          f"{np.percentile(s, 99):,.0f} / {s.max():,}")
    print(f"  top-1%  share   : {top1:.1%}")
    print(f"  top-10% share   : {top10:.1%}")
    return s_sorted, total


def plot_cdfs(per_entity, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, arr in per_entity.items():
        arr = np.sort(arr)
        arr = arr[arr > 0]
        cdf = np.arange(1, len(arr) + 1) / len(arr)
        ax.plot(arr, cdf, label=f"{label} (n={len(arr):,})")
    ax.set_xscale("log")
    ax.set_xlabel("Total invocations per entity (14 days)")
    ax.set_ylabel("CDF")
    ax.set_title("Invocation distribution by granularity")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lorenz(per_entity, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, arr in per_entity.items():
        arr = np.sort(arr)
        arr = arr[arr > 0]
        cum = np.cumsum(arr) / arr.sum()
        x = np.arange(1, len(arr) + 1) / len(arr)
        ax.plot(x, cum, label=label)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="equality")
    ax.set_xlabel("Cumulative share of entities (sorted asc)")
    ax.set_ylabel("Cumulative share of invocations")
    ax.set_title("Lorenz curves: workload skew by granularity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_topk_share(per_entity, out_path, k_max=1000):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, arr in per_entity.items():
        arr = np.sort(arr)[::-1]
        cum = np.cumsum(arr) / arr.sum()
        k = min(k_max, len(cum))
        ax.plot(np.arange(1, k + 1), cum[:k], label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Top-K entities (sorted desc by invocations)")
    ax.set_ylabel("Cumulative share of all invocations")
    ax.set_title("How few top entities capture most of the load")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--out", default=os.path.join(HERE, "plots"))
    parser.add_argument("--top-owners", type=int, default=20,
                        help="print top-N owners with their invocation counts")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(
        HERE, "invocations_per_function_md.anon.d*.csv")))[: args.days]
    if not files:
        raise SystemExit("No invocation CSVs found.")

    print(f"Loading {len(files)} day(s)...")
    parts = []
    for f in files:
        print(f"  {os.path.basename(f)}")
        parts.append(load_totals(f))
    df = pd.concat(parts, ignore_index=True)

    # Aggregate over the full window at each granularity.
    per_function = df.groupby("HashFunction")["total"].sum()
    per_app = df.groupby("HashApp")["total"].sum()
    per_owner = df.groupby("HashOwner")["total"].sum()

    # Companion ratios.
    apps_per_owner = df.groupby("HashOwner")["HashApp"].nunique()
    fns_per_app = df.groupby("HashApp")["HashFunction"].nunique()
    print(f"\nApps per owner    : median={apps_per_owner.median():.0f}, "
          f"p99={apps_per_owner.quantile(0.99):.0f}, "
          f"max={apps_per_owner.max()}")
    print(f"Functions per app : median={fns_per_app.median():.0f}, "
          f"p99={fns_per_app.quantile(0.99):.0f}, "
          f"max={fns_per_app.max()}")

    summarize(per_owner, "HashOwner")
    summarize(per_app, "HashApp")
    summarize(per_function, "HashFunction")

    print(f"\nTop {args.top_owners} owners by invocation count:")
    top_owners = per_owner.sort_values(ascending=False).head(args.top_owners)
    total_all = per_owner.sum()
    for i, (owner, v) in enumerate(top_owners.items(), 1):
        print(f"  {i:3d}. {owner[:16]}... invocations={v:>12,} "
              f"({v / total_all:.2%})")

    per_entity = {
        "HashOwner": per_owner.values,
        "HashApp": per_app.values,
        "HashFunction": per_function.values,
    }

    plot_cdfs(per_entity, os.path.join(args.out, "granularity_cdf.png"))
    plot_lorenz(per_entity, os.path.join(args.out, "granularity_lorenz.png"))
    plot_topk_share(per_entity, os.path.join(args.out, "granularity_topk_share.png"))

    # Save per-owner CSV for downstream use.
    out_csv = os.path.join(args.out, "invocations_per_owner.csv")
    per_owner.sort_values(ascending=False).to_csv(
        out_csv, header=["total_invocations"])
    print(f"\nWrote per-owner totals to {out_csv}")
    print(f"Plots written to {args.out}")


if __name__ == "__main__":
    main()
