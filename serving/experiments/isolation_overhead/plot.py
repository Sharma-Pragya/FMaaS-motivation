#!/usr/bin/env python3
"""Plot isolation overhead results from summary.csv."""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR   = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

CSV = RESULTS_DIR / "summary.csv"

MODE_LABELS = {"none": "None\n(direct)", "shared": "Thread\n(queue)", "process": "Process\n(IPC)"}
MODE_ORDER  = ["none", "shared", "process"]
COLORS      = {"none": "#4CAF50", "shared": "#2196F3", "process": "#FF9800"}


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["mode_label"] = df["isolation_mode"].map(MODE_LABELS)
    df["order"] = df["isolation_mode"].map({m: i for i, m in enumerate(MODE_ORDER)})
    return df.sort_values("order")


def plot_latency(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))

    modes  = [m for m in MODE_ORDER if m in df["isolation_mode"].values]
    labels = [MODE_LABELS[m] for m in modes]
    colors = [COLORS[m] for m in modes]

    avg = [df[df["isolation_mode"] == m]["avg_lat_ms"].values[0] for m in modes]
    p50 = [df[df["isolation_mode"] == m]["p50_lat_ms"].values[0] for m in modes]
    p95 = [df[df["isolation_mode"] == m]["p95_lat_ms"].values[0] for m in modes]
    p99 = [df[df["isolation_mode"] == m]["p99_lat_ms"].values[0] for m in modes]

    x = range(len(modes))
    w = 0.2

    ax.bar([i - 1.5*w for i in x], avg, w, label="avg",  color=[c+"cc" for c in colors], edgecolor="black", linewidth=0.6)
    ax.bar([i - 0.5*w for i in x], p50, w, label="p50",  color=colors, edgecolor="black", linewidth=0.6)
    ax.bar([i + 0.5*w for i in x], p95, w, label="p95",  color=colors, edgecolor="black", linewidth=0.6, alpha=0.7)
    ax.bar([i + 1.5*w for i in x], p99, w, label="p99",  color=colors, edgecolor="black", linewidth=0.6, alpha=0.4)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency by Isolation Mode")
    ax.legend(loc="upper left", fontsize=8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = PLOTS_DIR / "latency.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_throughput(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 4))

    modes  = [m for m in MODE_ORDER if m in df["isolation_mode"].values]
    labels = [MODE_LABELS[m] for m in modes]
    colors = [COLORS[m] for m in modes]
    rps    = [df[df["isolation_mode"] == m]["throughput_rps"].values[0] for m in modes]

    bars = ax.bar(labels, rps, color=colors, edgecolor="black", linewidth=0.7, width=0.5)
    for bar, val in zip(bars, rps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Throughput by Isolation Mode")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = PLOTS_DIR / "throughput.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_memory(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 4))

    modes  = [m for m in MODE_ORDER if m in df["isolation_mode"].values]
    labels = [MODE_LABELS[m] for m in modes]
    colors = [COLORS[m] for m in modes]
    mem    = [df[df["isolation_mode"] == m]["gpu_reserved_mb"].values[0] for m in modes]

    bars = ax.bar(labels, mem, color=colors, edgecolor="black", linewidth=0.7, width=0.5)
    for bar, val in zip(bars, mem):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("GPU Reserved Memory (MB)")
    ax.set_title("GPU Memory by Isolation Mode")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = PLOTS_DIR / "memory.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    df = load()
    print(df[["isolation_mode", "avg_lat_ms", "p99_lat_ms", "throughput_rps", "gpu_reserved_mb"]].to_string(index=False))
    plot_latency(df)
    plot_throughput(df)
    plot_memory(df)
