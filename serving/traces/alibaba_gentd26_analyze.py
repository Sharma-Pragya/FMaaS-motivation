"""Analysis + plotting for the Alibaba GenTD26 trace.

Run:
    python -m traces.alibaba_gentd26_analyze
    python -m traces.alibaba_gentd26_analyze --group-by groupId --top-k 20
    python -m traces.alibaba_gentd26_analyze --outdir traces/alibaba_gentd26/plots

Produces:
  - Printed summary: row counts, time span, distinct task/group counts,
    per-task request counts / RPS / mean interarrival / CoV.
  - PNG plots (when --outdir given, or default alongside trace):
      * top_k_request_count.png      Request count per top-K task.
      * top_k_rps_timeseries.png     Hourly RPS over time, top-K tasks.
      * interarrival_cov.png         Histogram of CoV over all tasks (>=N reqs).
      * exec_time_distribution.png   Exec time CDF + per predict_type.
      * group_vs_model_heatmap.png   (top-20 groups) x (top-20 models) heatmap.
      * requests_per_task_cdf.png    CDF of request count per task.
"""
import argparse
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from traces.alibaba_gentd26 import (
    DEFAULT_GROUP_BY,
    DEFAULT_TRACE_PATH,
    _interarrivals_for_task,
    load_trace,
    top_k_tasks,
)


def _savefig(fig, outdir: str, name: str) -> str:
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def print_summary(df: pd.DataFrame, group_by: str, top_k: int) -> None:
    dur_s = (df["gmt_create"].max() - df["gmt_create"].min()).total_seconds()
    print("=" * 72)
    print(f"trace rows (SUCCEED only): {len(df):,}")
    print(f"time span: {df['gmt_create'].min()} -> {df['gmt_create'].max()}  "
          f"({dur_s/86400:.2f} days, {dur_s:.0f}s)")
    print(f"overall avg rps: {len(df)/dur_s:.4f}")
    print()
    print(f"distinct groupId (tenants): {df['groupId'].nunique():,}")
    print(f"distinct checkpoint_model_version_id (models): "
          f"{df['checkpoint_model_version_id'].nunique()}")
    print(f"distinct (groupId, model) pairs: "
          f"{df.groupby(['groupId','checkpoint_model_version_id']).ngroups:,}")
    print()
    print(f"predict_type distribution:")
    for pt, n in df["predict_type"].value_counts().items():
        print(f"  {pt:<12} {n:>6}  ({100*n/len(df):.1f}%)")
    print()
    print("num_lora distribution:")
    for v, n in df["num_lora"].value_counts().sort_index().items():
        print(f"  num_lora={v:<3} {n:>6}  ({100*n/len(df):.1f}%)")
    print()
    vc = df[group_by].value_counts()
    usable = (vc >= 2).sum()
    print(f"tasks (group_by={group_by}) with >=2 requests: {usable}  "
          f"(of {len(vc)} total)")
    print(f"tasks with >=10 requests: {(vc >= 10).sum()}")
    print(f"tasks with >=100 requests: {(vc >= 100).sum()}")
    print()
    print(f"Top-{top_k} tasks by {group_by}:")
    print(f"  {'rank':>4} {'task':<40} {'n_req':>8} {'rps':>10} "
          f"{'mean_ia(s)':>12} {'cov_ia':>8}")
    for i, name in enumerate(top_k_tasks(df, top_k, group_by)):
        ia = _interarrivals_for_task(df, name, group_by)
        n = len(ia) + 1
        rps = n / dur_s
        mean_ia = ia.mean() if ia.size else float("nan")
        cov = ia.std() / mean_ia if ia.size and mean_ia > 0 else float("nan")
        print(f"  {i+1:>4} {str(name):<40} {n:>8d} {rps:>10.4f} "
              f"{mean_ia:>12.2f} {cov:>8.2f}")
    print("=" * 72)


def plot_top_k_request_count(df: pd.DataFrame, group_by: str, top_k: int,
                             outdir: str) -> str:
    counts = df[group_by].value_counts().head(top_k)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(counts)), counts.values)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Request count")
    ax.set_xlabel(group_by)
    ax.set_title(f"Top-{top_k} {group_by} by request count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return _savefig(fig, outdir, "top_k_request_count.png")


def plot_top_k_rps_timeseries(df: pd.DataFrame, group_by: str, top_k: int,
                              outdir: str) -> str:
    tasks = top_k_tasks(df, top_k, group_by)
    fig, ax = plt.subplots(figsize=(11, 4))
    for name in tasks:
        ts = (df[df[group_by] == name]
              .set_index("gmt_create")
              .resample("1H").size())
        ax.plot(ts.index, ts.values / 3600.0, label=str(name), alpha=0.75)
    ax.set_ylabel("RPS (hourly avg)")
    ax.set_xlabel("Time")
    ax.set_title(f"Top-{top_k} {group_by} — hourly RPS over trace")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    return _savefig(fig, outdir, "top_k_rps_timeseries.png")


def plot_cov_histogram(df: pd.DataFrame, group_by: str, min_reqs: int,
                       outdir: str) -> str:
    vc = df[group_by].value_counts()
    eligible = vc[vc >= min_reqs].index.tolist()
    covs: List[float] = []
    for name in eligible:
        ia = _interarrivals_for_task(df, name, group_by)
        if ia.size and ia.mean() > 0:
            covs.append(ia.std() / ia.mean())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(covs, bins=40, edgecolor="black", alpha=0.7)
    ax.axvline(1.0, color="red", linestyle="--", label="Poisson (CoV=1)")
    ax.set_xlabel("Interarrival CoV (std/mean)")
    ax.set_ylabel(f"# tasks ({group_by}, >= {min_reqs} reqs)")
    ax.set_title(f"Burstiness across tasks  "
                 f"(n={len(covs)}, median={np.median(covs):.2f})")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return _savefig(fig, outdir, "interarrival_cov.png")


def plot_exec_time_distribution(df: pd.DataFrame, outdir: str) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    vals = df["exec_time_seconds"].dropna().values
    sv = np.sort(vals)
    ax1.plot(sv, np.arange(1, len(sv) + 1) / len(sv))
    ax1.set_xlabel("Exec time (s)")
    ax1.set_ylabel("CDF")
    ax1.set_title(f"Exec time CDF (median={np.median(sv):.1f}s, "
                  f"p95={np.percentile(sv,95):.1f}s)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    for pt in df["predict_type"].unique():
        v = df.loc[df["predict_type"] == pt, "exec_time_seconds"].dropna().values
        if v.size == 0:
            continue
        sv = np.sort(v)
        ax2.plot(sv, np.arange(1, len(sv) + 1) / len(sv), label=f"{pt} (n={len(v)})")
    ax2.set_xlabel("Exec time (s)")
    ax2.set_ylabel("CDF")
    ax2.set_title("Exec time by predict_type")
    ax2.set_xlim(0, np.percentile(df["exec_time_seconds"].dropna(), 99))
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)
    return _savefig(fig, outdir, "exec_time_distribution.png")


def plot_group_vs_model_heatmap(df: pd.DataFrame, top_n: int,
                                outdir: str) -> str:
    top_groups = df["groupId"].value_counts().head(top_n).index.tolist()
    top_models = (df["checkpoint_model_version_id"].value_counts()
                  .head(top_n).index.tolist())
    sub = df[df["groupId"].isin(top_groups) &
             df["checkpoint_model_version_id"].isin(top_models)]
    pivot = (sub.groupby(["groupId", "checkpoint_model_version_id"]).size()
             .unstack(fill_value=0)
             .reindex(index=top_groups, columns=top_models, fill_value=0))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.log1p(pivot.values), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(top_models)))
    ax.set_xticklabels(top_models, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(top_groups)))
    ax.set_yticklabels(top_groups, fontsize=7)
    ax.set_xlabel("checkpoint_model_version_id")
    ax.set_ylabel("groupId")
    ax.set_title(f"Request count (log1p): top-{top_n} groups x top-{top_n} models")
    fig.colorbar(im, ax=ax, label="log1p(count)")
    return _savefig(fig, outdir, "group_vs_model_heatmap.png")


def plot_requests_per_task_cdf(df: pd.DataFrame, outdir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    for gb, label in [("checkpoint_model_version_id", "models"),
                      ("groupId", "groups/tenants")]:
        counts = df[gb].value_counts().values
        sv = np.sort(counts)
        ax.plot(sv, np.arange(1, len(sv) + 1) / len(sv),
                label=f"{label} (n={len(sv)})")
    ax.set_xscale("log")
    ax.set_xlabel("Requests per task (log scale)")
    ax.set_ylabel("CDF")
    ax.set_title("How skewed is per-task load?")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4, which="both")
    return _savefig(fig, outdir, "requests_per_task_cdf.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trace-path", default=DEFAULT_TRACE_PATH)
    p.add_argument("--group-by", default=DEFAULT_GROUP_BY,
                   choices=["checkpoint_model_version_id", "groupId"])
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--min-reqs", type=int, default=10,
                   help="CoV histogram includes only tasks with >= this many reqs")
    p.add_argument("--outdir", default=None,
                   help="Directory for PNG plots (default: next to trace)")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    df = load_trace(args.trace_path)
    print_summary(df, args.group_by, args.top_k)

    if args.no_plots:
        return

    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.abspath(args.trace_path)), "plots")
    os.makedirs(outdir, exist_ok=True)

    made = [
        plot_top_k_request_count(df, args.group_by, args.top_k, outdir),
        plot_top_k_rps_timeseries(df, args.group_by, args.top_k, outdir),
        plot_cov_histogram(df, args.group_by, args.min_reqs, outdir),
        plot_exec_time_distribution(df, outdir),
        plot_requests_per_task_cdf(df, outdir),
    ]
    try:
        made.append(plot_group_vs_model_heatmap(df, 20, outdir))
    except Exception as e:
        print(f"[warn] heatmap skipped: {e}")

    print(f"\nSaved {len(made)} plots to {outdir}:")
    for m in made:
        print(f"  {m}")


if __name__ == "__main__":
    main()
