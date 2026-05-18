"""Analyze the Azure Functions 2019 invocation traces and produce plots.

Reads `invocations_per_function_md.anon.d{01..14}.csv` from this directory
and emits PNGs into ./plots/.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MINUTE_COLS = [str(i) for i in range(1, 1441)]
TRIGGERS = ["http", "timer", "event", "queue", "storage", "orchestration", "others"]


def load_day(path):
    df = pd.read_csv(path)
    # Ensure all 1440 minute cols present and numeric.
    for c in MINUTE_COLS:
        if c not in df.columns:
            df[c] = 0
    df[MINUTE_COLS] = df[MINUTE_COLS].fillna(0).astype(np.int32)
    return df


def discover_days(pattern):
    files = sorted(glob.glob(os.path.join(HERE, pattern)))
    return files


def plot_invocations_per_minute(days, out_path):
    """Total invocations per minute of day, one line per day + mean."""
    fig, ax = plt.subplots(figsize=(11, 5))
    stacked = []
    for i, df in enumerate(days, start=1):
        per_min = df[MINUTE_COLS].sum(axis=0).values
        stacked.append(per_min)
        ax.plot(np.arange(1440), per_min, alpha=0.35, linewidth=0.8,
                label=f"day {i:02d}")
    mean_curve = np.mean(stacked, axis=0)
    ax.plot(np.arange(1440), mean_curve, color="black", linewidth=2.0,
            label="mean")
    ax.set_xlabel("Minute of day (UTC)")
    ax.set_ylabel("Total invocations")
    ax.set_title("Invocations per minute across days")
    ax.set_xlim(0, 1440)
    ax.legend(ncol=4, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trigger_breakdown(days, out_path):
    """Share of functions vs share of invocations, per trigger group."""
    func_counts = pd.Series(0, index=TRIGGERS, dtype=np.int64)
    inv_counts = pd.Series(0, index=TRIGGERS, dtype=np.int64)
    for df in days:
        trig = df["Trigger"].where(df["Trigger"].isin(TRIGGERS), "others")
        func_counts = func_counts.add(trig.value_counts(), fill_value=0)
        inv_per_fn = df[MINUTE_COLS].sum(axis=1)
        inv_counts = inv_counts.add(
            inv_per_fn.groupby(trig).sum(), fill_value=0
        )
    func_share = func_counts / func_counts.sum()
    inv_share = inv_counts / inv_counts.sum()

    x = np.arange(len(TRIGGERS))
    w = 0.4
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, [func_share[t] for t in TRIGGERS], w, label="% of functions")
    ax.bar(x + w / 2, [inv_share[t] for t in TRIGGERS], w, label="% of invocations")
    ax.set_xticks(x)
    ax.set_xticklabels(TRIGGERS)
    ax.set_ylabel("Share")
    ax.set_title("Trigger breakdown: function count vs invocation count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_invocations_cdf(days, out_path):
    """CDF of total daily invocations per function (log x)."""
    totals = []
    for df in days:
        totals.append(df[MINUTE_COLS].sum(axis=1).values)
    totals = np.concatenate(totals)
    totals = totals[totals > 0]
    totals.sort()
    cdf = np.arange(1, len(totals) + 1) / len(totals)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(totals, cdf)
    ax.set_xscale("log")
    ax.set_xlabel("Invocations per function per day")
    ax.set_ylabel("CDF over (function, day) pairs")
    ax.set_title("Distribution of daily per-function invocation counts")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_busy_minutes_cdf(days, out_path):
    """CDF of how many minutes per day a function is invoked at least once."""
    busy = []
    for df in days:
        busy.append((df[MINUTE_COLS] > 0).sum(axis=1).values)
    busy = np.concatenate(busy)
    busy = busy[busy > 0]
    busy.sort()
    cdf = np.arange(1, len(busy) + 1) / len(busy)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(busy, cdf)
    ax.set_xlabel("Active minutes per day (out of 1440)")
    ax.set_ylabel("CDF over (function, day) pairs")
    ax.set_title("Function activity coverage within a day")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_app_skew(days, out_path):
    """Lorenz-style curve: cumulative invocation share vs cumulative app share."""
    app_inv = {}
    for df in days:
        per_fn = df[MINUTE_COLS].sum(axis=1)
        agg = per_fn.groupby(df["HashApp"]).sum()
        for app, v in agg.items():
            app_inv[app] = app_inv.get(app, 0) + int(v)
    vals = np.array(sorted(app_inv.values()))
    vals = vals[vals > 0]
    cum = np.cumsum(vals) / vals.sum()
    x = np.arange(1, len(vals) + 1) / len(vals)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x, cum, label="apps (sorted asc by invocations)")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="equality")
    ax.set_xlabel("Cumulative share of apps")
    ax.set_ylabel("Cumulative share of invocations")
    ax.set_title("Workload skew across applications")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stacked_trigger_over_time(days, out_path):
    """Mean invocations per minute, stacked by trigger group."""
    accum = {t: np.zeros(1440, dtype=np.float64) for t in TRIGGERS}
    for df in days:
        trig = df["Trigger"].where(df["Trigger"].isin(TRIGGERS), "others")
        for t in TRIGGERS:
            mask = (trig == t).values
            if mask.any():
                accum[t] += df.loc[mask, MINUTE_COLS].sum(axis=0).values
    n = len(days)
    for t in TRIGGERS:
        accum[t] /= n

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(np.arange(1440),
                 [accum[t] for t in TRIGGERS],
                 labels=TRIGGERS)
    ax.set_xlabel("Minute of day (UTC)")
    ax.set_ylabel("Mean invocations / minute")
    ax.set_title("Mean per-minute invocations by trigger group")
    ax.set_xlim(0, 1440)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14,
                        help="number of day-files to use (1..14)")
    parser.add_argument("--out", default=os.path.join(HERE, "plots"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = discover_days("invocations_per_function_md.anon.d*.csv")[: args.days]
    if not files:
        raise SystemExit("No invocation CSVs found.")
    print(f"Loading {len(files)} day(s)...")
    days = []
    for f in files:
        print(f"  {os.path.basename(f)}")
        days.append(load_day(f))

    print("Plotting invocations per minute...")
    plot_invocations_per_minute(days, os.path.join(args.out, "invocations_per_minute.png"))
    print("Plotting trigger breakdown...")
    plot_trigger_breakdown(days, os.path.join(args.out, "trigger_breakdown.png"))
    print("Plotting per-function invocation CDF...")
    plot_invocations_cdf(days, os.path.join(args.out, "invocations_cdf.png"))
    print("Plotting busy-minutes CDF...")
    plot_busy_minutes_cdf(days, os.path.join(args.out, "busy_minutes_cdf.png"))
    print("Plotting app skew...")
    plot_app_skew(days, os.path.join(args.out, "app_invocation_skew.png"))
    print("Plotting stacked trigger over time...")
    plot_stacked_trigger_over_time(days, os.path.join(args.out, "trigger_stacked_over_time.png"))
    print(f"Done. Plots written to {args.out}")


if __name__ == "__main__":
    main()
