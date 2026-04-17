#!/usr/bin/env python3
"""sharing_benefit/tpc/compare.py — Compare OLD vs NEW runs side by side.

Pairs two result directories (same layout) and reports the delta across
common (ntasks, rps, condition) tuples. Intended for measuring the impact
of the per-task decoder-thread change vs the inline-decoder baseline.

Directory layout expected (identical in both dirs):
    <dir>/ntasks_{n}/rps_{rps}/{condition}/latencies.csv

Usage:
    python compare.py                                      # defaults: tsfm vs tsfm0
    python compare.py --old results_vision0 --new results_vision
    python compare.py --old results_tsfm0 --new results_tsfm --metric latency_ms
    python compare.py --old results_tsfm0 --new results_tsfm --per-task
    python compare.py --old results_tsfm0 --new results_tsfm --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict


METRICS = [
    "latency_ms",
    "server_exec_ms",
    "server_proc_ms",
    "server_decoder_ms",
    "queue_wait_plus_rpc_ms",
]


def load_rows(path: str) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    # drop the first row (warmup / cold start) to match our analysis convention
    return rows[1:] if len(rows) > 1 else rows


def stats(rows: list[dict], col: str) -> dict:
    vals = sorted(float(r[col]) for r in rows)
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": 0.0, "med": 0.0, "p25": 0.0, "p75": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "n":    n,
        "mean": statistics.mean(vals),
        "med":  statistics.median(vals),
        "p25":  vals[n // 4],
        "p75":  vals[3 * n // 4],
        "p99":  vals[int(n * 0.99)] if n > 100 else vals[-1],
        "max":  vals[-1],
    }


def pct(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0


def discover_runs(root: str) -> list[tuple[str, str, str, str]]:
    """Walk <root>/ntasks_*/rps_*/*/latencies.csv — return (ntasks, rps, condition, path)."""
    found = []
    if not os.path.isdir(root):
        return found
    for nt in sorted(os.listdir(root)):
        nt_dir = os.path.join(root, nt)
        if not nt.startswith("ntasks_") or not os.path.isdir(nt_dir):
            continue
        for rps in sorted(os.listdir(nt_dir)):
            rps_dir = os.path.join(nt_dir, rps)
            if not rps.startswith("rps_") or not os.path.isdir(rps_dir):
                continue
            for cond in sorted(os.listdir(rps_dir)):
                cond_dir = os.path.join(rps_dir, cond)
                csv_path = os.path.join(cond_dir, "latencies.csv")
                if os.path.isfile(csv_path):
                    found.append((nt, rps, cond, csv_path))
    return found


def intersect_runs(old_root: str, new_root: str) -> list[tuple[str, str, str, str, str]]:
    """Return (ntasks, rps, condition, old_path, new_path) for common tuples."""
    old_index = {(nt, rps, cond): path for nt, rps, cond, path in discover_runs(old_root)}
    new_index = {(nt, rps, cond): path for nt, rps, cond, path in discover_runs(new_root)}
    common = sorted(set(old_index) & set(new_index))
    return [(nt, rps, cond, old_index[k], new_index[k]) for k in common for nt, rps, cond in [k]]


def format_metric_row(metric: str, old_s: dict, new_s: dict) -> str:
    d_mean = new_s["mean"] - old_s["mean"]
    d_med  = new_s["med"]  - old_s["med"]
    d_p99  = new_s["p99"]  - old_s["p99"]
    return (
        f"  {metric:>24s}  "
        f"OLD mean={old_s['mean']:8.2f} med={old_s['med']:8.2f} p99={old_s['p99']:8.2f}  "
        f"NEW mean={new_s['mean']:8.2f} med={new_s['med']:8.2f} p99={new_s['p99']:8.2f}  "
        f"Δmean={d_mean:+7.2f} ({pct(new_s['mean'], old_s['mean']):+6.1f}%) "
        f"Δp99={d_p99:+7.2f} ({pct(new_s['p99'], old_s['p99']):+6.1f}%)"
    )


def compare_pair(nt: str, rps: str, cond: str, old_path: str, new_path: str,
                 metrics: list[str], per_task: bool) -> list[list]:
    """Print one header + metric rows for this tuple. Return CSV rows."""
    old_rows = load_rows(old_path)
    new_rows = load_rows(new_path)

    header = f"  {nt} / {rps} / {cond}   OLD n={len(old_rows)}  NEW n={len(new_rows)}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    csv_out = []
    for m in metrics:
        o = stats(old_rows, m)
        n = stats(new_rows, m)
        print(format_metric_row(m, o, n))
        csv_out.append([
            nt, rps, cond, m, "overall",
            o["n"], o["mean"], o["med"], o["p99"],
            n["n"], n["mean"], n["med"], n["p99"],
            n["mean"] - o["mean"], pct(n["mean"], o["mean"]),
            n["p99"] - o["p99"], pct(n["p99"], o["p99"]),
        ])

    if per_task:
        tasks = sorted(set(r["task"] for r in old_rows) | set(r["task"] for r in new_rows))
        for task in tasks:
            o_task = [r for r in old_rows if r["task"] == task]
            n_task = [r for r in new_rows if r["task"] == task]
            if not o_task or not n_task:
                continue
            print(f"\n  -- task={task} (OLD n={len(o_task)}, NEW n={len(n_task)}) --")
            for m in metrics:
                o = stats(o_task, m)
                n = stats(n_task, m)
                print(format_metric_row(m, o, n))
                csv_out.append([
                    nt, rps, cond, m, task,
                    o["n"], o["mean"], o["med"], o["p99"],
                    n["n"], n["mean"], n["med"], n["p99"],
                    n["mean"] - o["mean"], pct(n["mean"], o["mean"]),
                    n["p99"] - o["p99"], pct(n["p99"], o["p99"]),
                ])

    print()
    return csv_out


def print_summary(all_rows: list[list]):
    """Roll up deltas by (metric, condition) to highlight the pattern."""
    by_key = defaultdict(list)
    for row in all_rows:
        if row[4] != "overall":
            continue
        _, _, cond, metric, _, *_ = row
        d_mean_pct = row[13]
        d_p99_pct  = row[15]
        by_key[(metric, cond)].append((d_mean_pct, d_p99_pct))

    if not by_key:
        return

    print("=" * 100)
    print("  SUMMARY: average Δ% across all (ntasks, rps) for each (metric, condition)")
    print("=" * 100)
    print(f"  {'metric':>24s}  {'condition':>18s}  {'runs':>5s}  {'avg Δmean%':>12s}  {'avg Δp99%':>12s}")
    print("  " + "-" * 80)
    for (metric, cond), vals in sorted(by_key.items()):
        mean_avg = statistics.mean(v[0] for v in vals)
        p99_avg  = statistics.mean(v[1] for v in vals)
        print(f"  {metric:>24s}  {cond:>18s}  {len(vals):>5d}  {mean_avg:+11.1f}%  {p99_avg:+11.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", default="results_tsfm0",
                        help="OLD results directory (default: results_tsfm0)")
    parser.add_argument("--new", default="results_tsfm",
                        help="NEW results directory (default: results_tsfm)")
    parser.add_argument("--metric", action="append", default=None,
                        help="Metric(s) to compare. Repeat for multiple. Default: all core metrics.")
    parser.add_argument("--per-task", action="store_true",
                        help="Also show per-task breakdown under each condition.")
    parser.add_argument("--csv", default=None,
                        help="Write the full comparison to this CSV file.")
    parser.add_argument("--filter", default=None,
                        help="Substring filter on ntasks/rps/condition (e.g. 'sharing', 'rps_5').")
    args = parser.parse_args()

    # Resolve dirs relative to this script if they're not absolute
    here = os.path.dirname(os.path.abspath(__file__))
    old_root = args.old if os.path.isabs(args.old) else os.path.join(here, args.old)
    new_root = args.new if os.path.isabs(args.new) else os.path.join(here, args.new)

    if not os.path.isdir(old_root):
        print(f"OLD dir not found: {old_root}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(new_root):
        print(f"NEW dir not found: {new_root}", file=sys.stderr)
        sys.exit(1)

    metrics = args.metric if args.metric else METRICS

    common = intersect_runs(old_root, new_root)
    if args.filter:
        flt = args.filter
        common = [t for t in common if flt in t[0] or flt in t[1] or flt in t[2]]

    if not common:
        print(f"No overlapping (ntasks, rps, condition) runs between {old_root} and {new_root}")
        # Show what exists in each for debugging
        old_keys = {(nt, rps, cond) for nt, rps, cond, _ in discover_runs(old_root)}
        new_keys = {(nt, rps, cond) for nt, rps, cond, _ in discover_runs(new_root)}
        print(f"  OLD only: {sorted(old_keys - new_keys)[:10]}")
        print(f"  NEW only: {sorted(new_keys - old_keys)[:10]}")
        sys.exit(0)

    print(f"OLD: {old_root}")
    print(f"NEW: {new_root}")
    print(f"Comparing {len(common)} common runs\n")

    all_csv_rows = []
    for nt, rps, cond, old_path, new_path in common:
        all_csv_rows.extend(
            compare_pair(nt, rps, cond, old_path, new_path, metrics, args.per_task)
        )

    print_summary(all_csv_rows)

    if args.csv:
        csv_header = [
            "ntasks", "rps", "condition", "metric", "scope",
            "old_n", "old_mean", "old_med", "old_p99",
            "new_n", "new_mean", "new_med", "new_p99",
            "delta_mean", "delta_mean_pct", "delta_p99", "delta_p99_pct",
        ]
        out_path = args.csv if os.path.isabs(args.csv) else os.path.join(here, args.csv)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(csv_header)
            w.writerows(all_csv_rows)
        print(f"\nWrote {len(all_csv_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
