#!/usr/bin/env python3
"""Plot throughput vs batch size from summary.csv files."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Plot throughput vs batch size.")
    p.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        required=True,
        default="a2/summary.csv",
        help="One or more summary.csv files to plot.",
    )
    p.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1,16],
        help="Batch sizes to include (default: all in CSV).",
    )
    p.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Filter by backbone (e.g. momentlarge).",
    )
    p.add_argument(
        "--task",
        type=str,
        default=None,
        help="Filter by task (e.g. ecgclass).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("throughput_vs_batch_size.pdf"),
        help="Output image path.",
    )
    return p.parse_args()


def load(csv_paths):
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df["source"] = path.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    args = parse_args()
    df = load(args.csv)

    if args.backbone:
        df = df[df["backbone"] == args.backbone]
    if args.task:
        df = df[df["task"] == args.task]
    if args.batch_sizes:
        df = df[df["batch_size"].isin(args.batch_sizes)]

    if df.empty:
        raise SystemExit("No rows match the given filters.")

    df = df.sort_values("batch_size")
    agg = df.groupby("batch_size", as_index=False)["throughput_rps"].mean()

    x_labels = [str(b) for b in agg["batch_size"].tolist()]
    x_pos = list(range(len(x_labels)))

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(x_pos, agg["throughput_rps"], color="#4C72B0", edgecolor="black", width=0.65)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Sequential','Batched'])
    # ax.set_xlabel("Batch Size",fontsize=14)
    ax.set_ylabel("Throughput (req/s)",fontsize=14)
    ax.tick_params(axis='both', labelsize=14) 
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    import math
    ymax = agg["throughput_rps"].max()
    step = 10 ** max(0, int(math.floor(math.log10(ymax))) - 1) * 5
    top = math.ceil(ymax / step) * step
    ax.set_ylim(0, top)
    ticks = list(range(0, int(top) + 1, 25))
    ax.set_yticks(ticks)

    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
