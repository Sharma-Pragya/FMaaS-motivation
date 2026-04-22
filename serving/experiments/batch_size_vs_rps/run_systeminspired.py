#!/usr/bin/env python3
"""batch_size_vs_rps/run_systeminspired.py — multi-task Poisson like SystemInAction.

Runs the 10 SystemInAction tasks against a single device, sending independent
Poisson traces per task (same generator as SystemInAction). No deployment plan.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import json
import os
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import numpy as np

from client.trace import generate_trace
from site_manager.grpc_client import EdgeRuntimeClient


# SystemInAction task specs (type + seed)
TASK_SPECS = {
    "heartrate":    {"type": "regression",    "seed": 100},
    "sysbp":        {"type": "regression",    "seed": 200},
    "diasbp":       {"type": "regression",    "seed": 300},
    "ecgclass":     {"type": "classification","seed": 400},
    "gestureclass": {"type": "classification","seed": 500},
    "etth1fore":    {"type": "forecasting",   "seed": 600},
    "weatherfore":  {"type": "forecasting",   "seed": 700},
    "exchangefore": {"type": "forecasting",   "seed": 800},
    "eclfore":      {"type": "forecasting",   "seed": 900},
    "trafficfore":  {"type": "forecasting",   "seed": 1000},
}


def _parse_req_rate(raw):
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return None
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return [float(p) for p in parts]
    return float(s)


def _load_task_data():
    # Reuse TraceRunner's dataset loading without the plan machinery.
    from client import runner as trace_runner
    trace_runner._initialize_data()
    return trace_runner._DATA


async def _deploy(device_url: str, backbone: str, tasks: dict) -> None:
    client = EdgeRuntimeClient(device_url)
    try:
        await client.wait_ready()
        payload = json.dumps({
            "backbone": backbone,
            "decoders": [
                {"task": name, "type": spec["type"], "path": f"{name}_{backbone}_mlp"}
                for name, spec in tasks.items()
            ],
        })
        print(f"[Deploy] {device_url} backbone={backbone} tasks={len(tasks)}")
        resp = await client.control("load", payload)
        print(f"[Deploy] status={resp['status']}")
    finally:
        await client.close()


def _build_input_cache(data_by_task: dict, tasks: list[str]) -> dict:
    cache = {}
    for task in tasks:
        batch = data_by_task.get(task)
        if batch is None:
            continue
        inp = {"task": task}
        if "x" in batch:
            inp["x"] = batch["x"] if isinstance(batch["x"], (list, str)) else batch["x"].numpy().astype(np.float32)
        if "mask" in batch:
            inp["mask"] = batch["mask"].numpy().astype(np.float32)
        if "question" in batch:
            inp["question"] = batch["question"]
        cache[task] = inp
    return cache


def _normalize_trace(trace: list) -> list[dict]:
    """Normalize Request objects / dicts to plain dicts."""
    normalized = []
    for req in trace:
        if hasattr(req, "req_id"):
            normalized.append({
                "req_id": req.req_id,
                "task": req.task,
                "req_time": req.req_time,
            })
        else:
            normalized.append(dict(req))
    normalized.sort(key=lambda r: (r["req_time"], r["req_id"]))
    return normalized


async def _run_open_loop(device_url: str, events: list[dict], input_cache: dict, req_timeout: float = 60.0):
    client = EdgeRuntimeClient(device_url)
    await client.wait_ready()
    records = []

    async def _fire(req_id: int, task: str, t_send_abs: float, t_start: float) -> None:
        try:
            payload = input_cache.get(task)
            if payload is None:
                return
            resp = await asyncio.wait_for(client.infer(payload | {"req_id": req_id}), timeout=req_timeout)
            lat_ms = (time.time() - t_send_abs) * 1000
            records.append({
                "req_id": req_id,
                "task": task,
                "send_time_rel": round(t_send_abs - t_start, 6),
                "latency_ms": round(lat_ms, 4),
                "start_time_ns": resp["start_time_ns"],
                "end_time_ns": resp["end_time_ns"],
            })
        except Exception:
            pass

    t_start = time.time()
    in_flight = []
    for event in events:
        target = t_start + event["req_time"]
        wait = target - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        t_send = time.time()
        in_flight.append(
            asyncio.create_task(_fire(event["req_id"], event["task"], t_send, t_start))
        )

    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=True)
    await client.close()
    return records


def _compute_batch_sizes(records: list[dict], warmup_secs: float) -> list[dict]:
    groups = defaultdict(list)
    for r in records:
        if r["send_time_rel"] > warmup_secs:
            groups[r["start_time_ns"]].append(r)
    rows = []
    for start_ns, reqs in sorted(groups.items()):
        rows.append({
            "batch_start_ns": start_ns,
            "observed_batch_size": len(reqs),
            "send_time_rel_first": round(min(r["send_time_rel"] for r in reqs), 6),
        })
    return rows


def _write_summary(batch_rows, out_dir: Path):
    sizes = [r["observed_batch_size"] for r in batch_rows]
    if not sizes:
        return {
            "n_batches": 0,
            "mean_batch_size": 0,
            "p50_batch_size": 0,
            "p95_batch_size": 0,
            "max_batch_size_obs": 0,
        }

    sizes_sorted = sorted(sizes)

    def _pct(p):
        idx = int(round((p / 100.0) * (len(sizes_sorted) - 1)))
        return float(sizes_sorted[idx])

    return {
        "n_batches": len(sizes_sorted),
        "mean_batch_size": float(sum(sizes_sorted) / len(sizes_sorted)),
        "p50_batch_size": _pct(50),
        "p95_batch_size": _pct(95),
        "max_batch_size_obs": int(max(sizes_sorted)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-url", default="localhost:8000")
    parser.add_argument("--backbone", default=os.environ.get("BACKBONE", "momentbase"))
    parser.add_argument("--req-rate", default="10",
                        help="Total req/s or comma list per task (sorted task order).")
    parser.add_argument("--duration", type=float, default=float(os.environ.get("PHASE_DURATION", "60")))
    parser.add_argument("--trace", default="poisson_per_task")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-secs", type=float, default=10.0)
    parser.add_argument("--batch-wait-ms", type=float, default=float(os.environ.get("BATCH_WAIT_MS", "0")))
    parser.add_argument("--exp-dir", default=os.environ.get(
        "EXP_DIR", "experiments/batch_size_vs_rps/systeminspired_results"))
    args = parser.parse_args()

    out_dir = (SERVING_DIR / args.exp_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = dict(TASK_SPECS)
    task_names = sorted(tasks.keys())
    req_rate = _parse_req_rate(args.req_rate)

    print("=" * 65)
    print("  Experiment: batch_size_vs_rps (SystemInAction tasks)")
    print(f"  Backbone    : {args.backbone}")
    print(f"  Device URL  : {args.device_url}")
    print(f"  Tasks       : {len(task_names)}  {task_names}")
    print(f"  Req rate    : {args.req_rate}")
    print(f"  Duration    : {args.duration}s  (warmup={args.warmup_secs}s)")
    print(f"  Results     : {out_dir}")
    print("=" * 65)

    data_by_task = _load_task_data()
    input_cache = _build_input_cache(data_by_task, task_names)

    asyncio.run(_deploy(args.device_url, args.backbone, tasks))
    asyncio.run(asyncio.sleep(1))

    trace, avg_workload, _ = generate_trace(
        args.trace, req_rate, args.duration, tasks, args.seed
    )
    events = _normalize_trace(trace)

    # Save scheduled workload
    with (out_dir / "send_times.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["req_id", "task", "send_time_rel"])
        for event in events:
            w.writerow([event["req_id"], event["task"], event["req_time"]])

    t_start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    records = asyncio.run(_run_open_loop(args.device_url, events, input_cache))
    t_end_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    batch_rows = _compute_batch_sizes(records, args.warmup_secs)

    with (out_dir / "latencies.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["req_id", "task", "send_time_rel", "latency_ms", "start_time_ns", "end_time_ns"],
        )
        w.writeheader()
        w.writerows(records)

    with (out_dir / "batch_sizes.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["batch_start_ns", "observed_batch_size", "send_time_rel_first"],
        )
        w.writeheader()
        w.writerows(batch_rows)

    batch_summary = _write_summary(batch_rows, out_dir)

    measured_lats = [r["latency_ms"] for r in records if r["send_time_rel"] > args.warmup_secs]
    if measured_lats:
        avg_latency_ms = float(np.mean(measured_lats))
        p99_latency_ms = float(np.percentile(measured_lats, 99))
    else:
        avg_latency_ms = 0.0
        p99_latency_ms = 0.0

    config = {
        "experiment": "batch_size_vs_rps_systeminspired",
        "device_url": args.device_url,
        "backbone": args.backbone,
        "tasks": task_names,
        "req_rate": req_rate,
        "duration_s": args.duration,
        "warmup_secs": args.warmup_secs,
        "batch_wait_ms": args.batch_wait_ms,
        "trace": args.trace,
        "seed": args.seed,
        "t_start": t_start_iso,
        "t_end": t_end_iso,
        "hostname": socket.gethostname(),
    }
    with (out_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    summary = {
        **batch_summary,
        "task_count": len(task_names),
        "rps": req_rate,
        "duration_s": args.duration,
        "warmup_secs": args.warmup_secs,
        "batch_wait_ms": args.batch_wait_ms,
        "n_requests_total": len(records),
        "n_requests_measured": len(measured_lats),
        "avg_latency_ms": avg_latency_ms,
        "p99_latency_ms": p99_latency_ms,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Save] {len(records)} requests, {len(batch_rows)} batches → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
