#!/usr/bin/env python3
"""Plot summaries for runtime_collective phase results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("experiments/runtime_collective/results/fmaas_share/10")


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(len(values) * q))
    return values[idx]


def load_phase_rows(phase_dir: Path):
    rows = []
    csv_path = phase_dir / "request_latency_results.csv"
    if not csv_path.exists():
        return rows

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["req_time"] = float(row["req_time"])
            row["lat_ms"] = float(row["end_to_end_latency(ms)"])
            row["queue_ms"] = (
                float(row["device_start_time"]) - float(row["site_manager_send_time"])
            ) * 1000.0
            rows.append(row)
    return rows


def load_phase_plan(phase_dir: Path):
    plan_path = phase_dir / "deployment_plan.json"
    if not plan_path.exists():
        return []
    with plan_path.open() as f:
        return json.load(f)["sites"]


def summarize():
    phases = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("phase"))
    summaries = []

    for phase_dir in phases:
        rows = load_phase_rows(phase_dir)
        if not rows:
            continue
        plan_sites = load_phase_plan(phase_dir)

        devices = {}
        for site in plan_sites:
            for dep in site["deployments"]:
                devices[dep["device"]] = {
                    "util": dep.get("util", 0.0),
                    "tasks": dep["tasks"],
                    "backbone": dep["backbone"],
                }

        summaries.append(
            {
                "phase": phase_dir.name,
                "rows": len(rows),
                "avg_lat_ms": sum(r["lat_ms"] for r in rows) / len(rows),
                "p95_lat_ms": percentile([r["lat_ms"] for r in rows], 0.95),
                "max_lat_ms": max(r["lat_ms"] for r in rows),
                "avg_queue_ms": sum(r["queue_ms"] for r in rows) / len(rows),
                "p95_queue_ms": percentile([r["queue_ms"] for r in rows], 0.95),
                "devices": devices,
                "rows_by_device": {
                    d: sum(1 for r in rows if r["device"] == d)
                    for d in sorted({r["device"] for r in rows})
                },
            }
        )

    return summaries


def write_summary_csv(summaries):
    out_path = ROOT / "phase_summary.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "phase",
                "rows",
                "avg_lat_ms",
                "p95_lat_ms",
                "max_lat_ms",
                "avg_queue_ms",
                "p95_queue_ms",
            ]
        )
        for s in summaries:
            writer.writerow(
                [
                    s["phase"],
                    s["rows"],
                    f"{s['avg_lat_ms']:.3f}",
                    f"{s['p95_lat_ms']:.3f}",
                    f"{s['max_lat_ms']:.3f}",
                    f"{s['avg_queue_ms']:.3f}",
                    f"{s['p95_queue_ms']:.3f}",
                ]
            )
    return out_path


def plot_latency(summaries):
    labels = [s["phase"].replace("phase", "P") for s in summaries]
    x = list(range(len(summaries)))
    avg = [s["avg_lat_ms"] for s in summaries]
    p95 = [s["p95_lat_ms"] for s in summaries]
    mx = [s["max_lat_ms"] for s in summaries]

    plt.figure(figsize=(12, 5))
    plt.plot(x, avg, marker="o", label="avg latency")
    plt.plot(x, p95, marker="s", label="p95 latency")
    plt.plot(x, mx, marker="^", label="max latency")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Latency (ms)")
    plt.title("Runtime Collective: Latency by Phase")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = ROOT / "latency_by_phase.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_queue(summaries):
    labels = [s["phase"].replace("phase", "P") for s in summaries]
    x = list(range(len(summaries)))
    avg = [s["avg_queue_ms"] for s in summaries]
    p95 = [s["p95_queue_ms"] for s in summaries]

    plt.figure(figsize=(12, 5))
    plt.plot(x, avg, marker="o", label="avg queue")
    plt.plot(x, p95, marker="s", label="p95 queue")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Queue Before device_start_time (ms)")
    plt.title("Runtime Collective: Queueing by Phase")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = ROOT / "queue_by_phase.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_device_load(summaries):
    labels = [s["phase"].replace("phase", "P") for s in summaries]
    x = list(range(len(summaries)))

    all_devices = sorted({d for s in summaries for d in s["rows_by_device"]})
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    plt.figure(figsize=(12, 5))
    bottom = [0] * len(summaries)
    for idx, device in enumerate(all_devices):
        vals = [s["rows_by_device"].get(device, 0) for s in summaries]
        plt.bar(x, vals, bottom=bottom, label=device, color=colors[idx % len(colors)])
        bottom = [bottom[i] + vals[i] for i in range(len(vals))]

    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Completed requests")
    plt.title("Runtime Collective: Completed Requests by Device and Phase")
    plt.legend()
    plt.tight_layout()
    out = ROOT / "requests_by_device_phase.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def main():
    summaries = summarize()
    if not summaries:
        raise SystemExit(f"No phase results found under {ROOT}")

    summary_csv = write_summary_csv(summaries)
    lat_plot = plot_latency(summaries)
    queue_plot = plot_queue(summaries)
    device_plot = plot_device_load(summaries)

    print(f"Wrote {summary_csv}")
    print(f"Wrote {lat_plot}")
    print(f"Wrote {queue_plot}")
    print(f"Wrote {device_plot}")


if __name__ == "__main__":
    main()
