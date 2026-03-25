#!/usr/bin/env python3
"""motivation2/run.py — Sharing benefit experiment (ecgclass + gestureclass).

Four conditions:
  single_ecgclass     — 1 device server, ecgclass only, FIFO
  single_gestureclass — 1 device server, gestureclass only, FIFO
  no_sharing          — 2 device servers (port A + B), one backbone each, FIFO
  sharing             — 1 device server, both tasks, STFQ

Each condition: deploy → run open-loop Poisson at fixed RPS → save latencies.csv.
run.sh handles starting/stopping device servers; this script just runs the experiment.

Usage (called by run.sh):
    python experiments/motivation2/run.py \
        --condition sharing \
        --device-url localhost:8000 \
        --device-url-2 localhost:8001 \
        --backbone momentbase \
        --rps 20 \
        --duration 180 \
        --exp-dir experiments/motivation2/results/sharing
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import numpy as np
from torch.utils.data import DataLoader

from site_manager.grpc_client import EdgeRuntimeClient
from site_manager.config import DATASET_DIR as _DATASET_DIR

TASK_TYPES: Dict[str, str] = {
    "ecgclass":     "classification",
    "gestureclass": "classification",
}

BOTH_TASKS = ["ecgclass", "gestureclass"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def build_data(tasks: List[str]) -> Dict[str, Dict]:
    from fmtk.datasetloaders.ecg5000 import ECG5000Dataset
    from fmtk.datasetloaders.uwavegesture import UWaveGestureLibraryALLDataset

    d = _DATASET_DIR
    cfg = {"batch_size": 1, "shuffle": False}
    loaders = {
        "ecgclass": lambda: DataLoader(
            ECG5000Dataset({"dataset_path": f"{d}/ECG5000"}, {"task_type": "classification"}, "test"),
            **cfg,
        ),
        "gestureclass": lambda: DataLoader(
            UWaveGestureLibraryALLDataset(
                {"dataset_path": f"{d}/UWaveGestureLibraryAll", "seq_len": 512},
                {"task_type": "classification"}, "test",
            ),
            **cfg,
        ),
    }
    data = {}
    for task in tasks:
        loader = loaders[task]()
        batch = next(iter(loader))
        data[task] = {
            "x":    batch["x"].numpy().astype(np.float32),
            "mask": batch["mask"].numpy().astype(np.float32) if "mask" in batch else None,
        }
        print(f"[Data] Loaded {task}: x.shape={data[task]['x'].shape}")
    return data


# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

async def deploy(device_url: str, backbone: str, tasks: List[str]) -> None:
    decoders = [{"task": t, "type": TASK_TYPES[t], "path": f"{t}_{backbone}_mlp"} for t in tasks]
    client = EdgeRuntimeClient(device_url)
    try:
        await client.wait_ready()
        payload = json.dumps({"backbone": backbone, "decoders": decoders})
        print(f"[Deploy] {device_url}  backbone={backbone}  tasks={tasks}")
        resp = await client.control("load", payload)
        print(f"[Deploy] {device_url}  status={resp['status']}")
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Open-loop Poisson sender
# ---------------------------------------------------------------------------

Record = Tuple[float, float]  # (send_time_relative_s, latency_ms)

TASK_SEEDS = {"ecgclass": 42, "gestureclass": 43}


def generate_traces(tasks: List[str], rps: float, duration: float) -> Dict[str, List[float]]:
    """Generate per-task Poisson send times. Each task uses its own fixed seed
    so the trace is identical regardless of which condition/other tasks are present."""
    send_times: Dict[str, List[float]] = {}
    for task in tasks:
        rng = np.random.default_rng(TASK_SEEDS.get(task, 42))
        times, t = [], 0.0
        while t < duration:
            times.append(t)
            t += rng.exponential(1.0 / rps)
        send_times[task] = times
    return send_times


async def run_open_loop(
    task_urls: Dict[str, str],          # {task: device_url}
    data: Dict[str, Dict],
    send_times: Dict[str, List[float]], # pre-generated traces from generate_traces()
    req_timeout: float = 60.0,
) -> Dict[str, List[Record]]:
    """Send each task using pre-generated send times.

    Returns {task: [(rel_send_time_s, latency_ms), ...]}
    """
    # One persistent client per unique URL
    unique_urls = list(set(task_urls.values()))
    clients: Dict[str, EdgeRuntimeClient] = {}
    for url in unique_urls:
        c = EdgeRuntimeClient(url)
        await c.wait_ready()
        clients[url] = c

    records: Dict[str, List[Record]] = {t: [] for t in task_urls}

    async def _fire(task: str, req_id: int, t_send_abs: float, t_start: float) -> None:
        d = data[task]
        try:
            await asyncio.wait_for(clients[task_urls[task]].infer({
                "req_id": req_id,
                "task":   task,
                "x":      d["x"],
                "mask":   d.get("mask"),
            }), timeout=req_timeout)
            lat_ms = (time.time() - t_send_abs) * 1000
            records[task].append((t_send_abs - t_start, lat_ms))
        except Exception:
            pass  # drop errors — don't pollute time-series

    # Shared t_start so all tasks begin from the same wall clock
    t_start = time.time()

    async def _sender(task: str, req_id_offset: int) -> None:
        in_flight = []
        for req_id, rel_t in enumerate(send_times[task]):
            target = t_start + rel_t
            wait = target - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            t_send = time.time()
            in_flight.append(asyncio.create_task(
                _fire(task, req_id_offset + req_id, t_send, t_start)
            ))
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

    senders = [
        _sender(task, i * 1_000_000)
        for i, task in enumerate(task_urls)
    ]
    await asyncio.gather(*senders, return_exceptions=True)

    for c in clients.values():
        await c.close()
    return records


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(records: Dict[str, List[Record]], out_dir: Path, condition: str,
                 duration: float, warmup_secs: float = 10.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # latencies.csv — all requests with elapsed_sec for warmup trimming
    with (out_dir / "latencies.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "condition", "elapsed_sec", "latency_ms"])
        for task, recs in records.items():
            for rel_t, lat in recs:
                w.writerow([task, condition, round(rel_t, 4), round(lat, 4)])

    # task_results.csv — per-task summary (excluding warmup)
    with (out_dir / "task_results.csv").open("w", newline="") as f:
        fields = ["task", "condition", "n_requests", "throughput_rps",
                  "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for task, recs in records.items():
            lats = [lat for rel_t, lat in recs if rel_t > warmup_secs]
            n = len(lats)
            if n == 0:
                continue
            w.writerow({
                "task":           task,
                "condition":      condition,
                "n_requests":     n,
                "throughput_rps": round(n / (duration - warmup_secs), 4),
                "avg_latency_ms": round(float(np.mean(lats)), 3),
                "p50_latency_ms": round(float(np.percentile(lats, 50)), 3),
                "p95_latency_ms": round(float(np.percentile(lats, 95)), 3),
                "p99_latency_ms": round(float(np.percentile(lats, 99)), 3),
            })

    for task, recs in records.items():
        print(f"[Save] {task}: {len(recs)} total requests → {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition",    required=True,
                        choices=["single_ecgclass", "single_gestureclass", "no_sharing", "sharing"])
    parser.add_argument("--device-url",   default="localhost:8000",
                        help="Primary device server URL")
    parser.add_argument("--device-url-2", default="localhost:8001",
                        help="Second device server URL (used only for no_sharing)")
    parser.add_argument("--backbone",     default=os.environ.get("BACKBONE", "momentbase"))
    parser.add_argument("--rps",          type=float, default=float(os.environ.get("RPS", "20")))
    parser.add_argument("--duration",     type=float, default=float(os.environ.get("PHASE_DURATION", "180")))
    parser.add_argument("--warmup-secs",  type=float, default=10.0)
    parser.add_argument("--exp-dir",      default=os.environ.get("EXP_DIR", "experiments/motivation2/results"))
    args = parser.parse_args()

    out_dir = (SERVING_DIR / args.exp_dir).resolve()

    print("=" * 65)
    print(f"  Motivation Experiment #2 — condition={args.condition}")
    print(f"  Backbone  : {args.backbone}")
    print(f"  RPS/task  : {args.rps}")
    print(f"  Duration  : {args.duration}s  (warmup={args.warmup_secs}s)")
    print(f"  Results   : {out_dir}")
    print("=" * 65)

    # Determine tasks and URL mapping per condition
    if args.condition == "single_ecgclass":
        tasks = ["ecgclass"]
        task_urls = {"ecgclass": args.device_url}
    elif args.condition == "single_gestureclass":
        tasks = ["gestureclass"]
        task_urls = {"gestureclass": args.device_url}
    elif args.condition == "no_sharing":
        tasks = BOTH_TASKS
        # each task has its own server
        task_urls = {"ecgclass": args.device_url, "gestureclass": args.device_url_2}
    else:  # sharing
        tasks = BOTH_TASKS
        task_urls = {"ecgclass": args.device_url, "gestureclass": args.device_url}

    print(f"[INFO] Loading data for: {tasks}")
    data = build_data(tasks)

    # Deploy: for no_sharing deploy each task to its own server
    if args.condition == "no_sharing":
        asyncio.run(deploy(args.device_url,   args.backbone, ["ecgclass"]))
        asyncio.run(deploy(args.device_url_2, args.backbone, ["gestureclass"]))
    else:
        asyncio.run(deploy(args.device_url, args.backbone, tasks))

    asyncio.run(asyncio.sleep(1))

    # Generate traces once — each task uses a fixed seed so the trace is
    # identical across all conditions (single, no_sharing, sharing)
    send_times = generate_traces(BOTH_TASKS, args.rps, args.duration)

    print(f"\n[Run] Starting open-loop send ({args.duration}s) ...")
    records = asyncio.run(run_open_loop(
        task_urls=task_urls,
        data=data,
        send_times={t: send_times[t] for t in tasks},
    ))

    save_results(records, out_dir, args.condition, args.duration, args.warmup_secs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
