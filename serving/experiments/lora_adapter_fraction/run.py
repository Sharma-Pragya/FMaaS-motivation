#!/usr/bin/env python3
"""lora_adapter_fraction — Throughput vs # LoRA-adapted tasks (fixed total N).

Total number of deployed tasks is fixed (default 10). What varies across
runs is how many of those tasks carry a LoRA adapter:

   K=0   → 10 tasks, all plain MLP decoder (no adapter swapping at all)
   K=1   → 1 task with LoRA, 9 plain
   ...
   K=10  → all 10 tasks have a LoRA adapter

All tasks share a single backbone on one device server. Closed-loop
traffic: K_workers per task, send next request after previous returns.
Aggregate throughput is the dependent variable; the goal is to isolate
how peft `set_adapter()` swap cost grows with the number of distinct
adapters touched per second.

Usage (from serving/):
    python experiments/lora_adapter_fraction/run.py \
        --device-url localhost:8000 \
        --backbone momentlarge \
        --base-task ecgclass \
        --num-tasks 10 --num-adapted 4 \
        --concurrency-per-task 1 --duration 60 \
        --exp-dir experiments/lora_adapter_fraction/results/K4
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
from typing import Dict, List, Tuple

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import numpy as np
from torch.utils.data import DataLoader

from site_manager.grpc_client import EdgeRuntimeClient, encode_infer_request
from site_manager.config import DATASET_DIR as _DATASET_DIR


BASE_TASK_TYPES: Dict[str, str] = {
    "ecgclass":     "classification",
    "gestureclass": "classification",
}

BASE_TASK_ADAPTERS: Dict[str, str] = {
    "ecgclass":     "lora",
    "gestureclass": "lora",
}


def build_base_data(base_task: str) -> Dict[str, np.ndarray | None]:
    from fmtk.datasetloaders.ecg5000 import ECG5000Dataset
    from fmtk.datasetloaders.uwavegesture import UWaveGestureLibraryALLDataset

    d = _DATASET_DIR
    cfg = {"batch_size": 1, "shuffle": False}
    loaders = {
        "ecgclass":     lambda: DataLoader(ECG5000Dataset({"dataset_path": f"{d}/ECG5000"}, {"task_type": "classification"}, "test"), **cfg),
        "gestureclass": lambda: DataLoader(UWaveGestureLibraryALLDataset({"dataset_path": f"{d}/UWaveGestureLibraryAll", "seq_len": 512}, {"task_type": "classification"}, "test"), **cfg),
    }
    if base_task not in loaders:
        raise ValueError(f"Unknown base task: {base_task}")
    batch = next(iter(loaders[base_task]()))
    data = {
        "x":    batch["x"].numpy().astype(np.float32),
        "mask": batch["mask"].numpy().astype(np.float32) if "mask" in batch and batch["mask"] is not None else None,
    }
    print(f"[Data] Loaded {base_task}: x.shape={data['x'].shape}")
    return data


def _replica_task_name(base_task: str, i: int) -> str:
    return base_task if i == 0 else f"{base_task}__app{i}"


def _task_spec(task_name: str, base_task: str, *, with_adapter: bool) -> dict:
    spec = {
        "task":      task_name,
        "base_task": base_task,
        "type":      BASE_TASK_TYPES[base_task],
        "path":      None,   # random-init decoder (and adapter, when present)
    }
    if with_adapter:
        spec["adapter"] = BASE_TASK_ADAPTERS[base_task]
    return spec


async def deploy_mixed(device_url: str, backbone: str, base_task: str,
                       num_tasks: int, num_adapted: int) -> dict:
    """Deploy num_tasks replicas: first num_adapted carry a LoRA adapter,
    the rest use only the MLP decoder."""
    if not 0 <= num_adapted <= num_tasks:
        raise ValueError(f"num_adapted={num_adapted} must be in [0, {num_tasks}]")
    task_specs = [
        _task_spec(
            _replica_task_name(base_task, i),
            base_task,
            with_adapter=(i < num_adapted),
        )
        for i in range(num_tasks)
    ]
    n_adapted_actual = sum("adapter" in s for s in task_specs)
    print(f"[Deploy] {device_url} backbone={backbone} "
          f"tasks={num_tasks} adapted={n_adapted_actual} plain={num_tasks - n_adapted_actual}")
    client = EdgeRuntimeClient(device_url)
    try:
        await client.wait_ready()
        payload = json.dumps({"backbone": backbone, "decoders": task_specs})
        resp = await client.control("load", payload)
        print(f"[Deploy] status={resp['status']}")
        return resp
    finally:
        await client.close()


# (task, send_elapsed_s, latency_ms, server_exec_ms, worker_id, has_adapter)
Record = Tuple[str, float, float, float, int, int]


async def run_closed_loop(
    device_url: str,
    task_has_adapter: Dict[str, bool],
    data: Dict[str, np.ndarray | None],
    concurrency_per_task: int,
    duration: float,
    req_timeout: float = 60.0,
) -> List[Record]:
    client = EdgeRuntimeClient(device_url)
    await client.wait_ready()
    stub = client._stub

    records: List[Record] = []
    rec_lock = asyncio.Lock()
    start_wall = time.time()
    req_counter = 0
    req_lock = asyncio.Lock()

    async def _next_req_id() -> int:
        nonlocal req_counter
        async with req_lock:
            rid = req_counter
            req_counter += 1
            return rid

    async def _worker(task: str, worker_id: int) -> None:
        proto = encode_infer_request(task=task, x=data["x"], mask=data.get("mask"))
        ad = 1 if task_has_adapter[task] else 0
        while True:
            if time.time() - start_wall >= duration:
                return
            proto.req_id = await _next_req_id()
            send_abs = time.time()
            try:
                resp = await asyncio.wait_for(stub.Infer(proto), timeout=req_timeout)
            except Exception:
                continue
            done_abs = time.time()
            lat_ms = (done_abs - send_abs) * 1000.0
            exec_ms = (resp.end_time_ns - resp.start_time_ns) / 1e6 if resp.end_time_ns else 0.0
            async with rec_lock:
                records.append((task, send_abs - start_wall, lat_ms, exec_ms, worker_id, ad))

    workers = [
        _worker(task, w)
        for task in task_has_adapter
        for w in range(concurrency_per_task)
    ]
    await asyncio.gather(*workers, return_exceptions=True)
    await client.close()
    return records


def save_results(records: List[Record], out_dir: Path, *, duration: float,
                 warmup_secs: float, num_tasks: int, num_adapted: int,
                 concurrency_per_task: int, backbone: str, base_task: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "latencies.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "send_elapsed_s", "latency_ms", "server_exec_ms",
                    "worker_id", "has_adapter"])
        for task, t, lat, sx, wid, ad in records:
            w.writerow([task, f"{t:.4f}", f"{lat:.3f}", f"{sx:.3f}", wid, ad])

    trimmed = [r for r in records if r[1] > warmup_secs]
    measured_window = max(duration - warmup_secs, 1e-9)

    # Split by adapter / plain so we can see per-group latency too.
    adapted_lats = np.array([r[2] for r in trimmed if r[5] == 1])
    plain_lats   = np.array([r[2] for r in trimmed if r[5] == 0])
    all_lats     = np.array([r[2] for r in trimmed]) if trimmed else np.array([])
    total_n      = len(trimmed)

    by_task: Dict[str, List[Record]] = {}
    for r in trimmed:
        by_task.setdefault(r[0], []).append(r)

    per_task_rows = []
    for task in sorted(by_task):
        lat = np.array([r[2] for r in by_task[task]])
        has_ad = by_task[task][0][5]
        per_task_rows.append({
            "task":             task,
            "has_adapter":      has_ad,
            "n_requests":       int(len(lat)),
            "throughput_rps":   round(len(lat) / measured_window, 4),
            "avg_latency_ms":   round(float(lat.mean()), 3) if len(lat) else None,
            "p50_latency_ms":   round(float(np.percentile(lat, 50)), 3) if len(lat) else None,
            "p95_latency_ms":   round(float(np.percentile(lat, 95)), 3) if len(lat) else None,
            "p99_latency_ms":   round(float(np.percentile(lat, 99)), 3) if len(lat) else None,
        })

    with (out_dir / "per_task.csv").open("w", newline="") as f:
        fields = list(per_task_rows[0].keys()) if per_task_rows else \
                 ["task", "has_adapter", "n_requests", "throughput_rps",
                  "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in per_task_rows:
            w.writerow(row)

    def _pct(arr, p):
        return round(float(np.percentile(arr, p)), 3) if arr.size else None

    summary = {
        "backbone":               backbone,
        "base_task":              base_task,
        "num_tasks":              num_tasks,
        "num_adapted":            num_adapted,
        "num_plain":              num_tasks - num_adapted,
        "concurrency_per_task":   concurrency_per_task,
        "duration_s":             duration,
        "warmup_s":               warmup_secs,
        "n_requests_total":       total_n,
        "n_requests_pre_warmup":  len(records) - total_n,
        "aggregate_throughput_rps": round(total_n / measured_window, 4),
        "avg_latency_ms_all":     round(float(all_lats.mean()), 3) if total_n else None,
        "p50_latency_ms_all":     _pct(all_lats, 50),
        "p95_latency_ms_all":     _pct(all_lats, 95),
        "p99_latency_ms_all":     _pct(all_lats, 99),
        "avg_latency_ms_adapted": round(float(adapted_lats.mean()), 3) if adapted_lats.size else None,
        "avg_latency_ms_plain":   round(float(plain_lats.mean()), 3)   if plain_lats.size   else None,
        "p99_latency_ms_adapted": _pct(adapted_lats, 99),
        "p99_latency_ms_plain":   _pct(plain_lats, 99),
        "throughput_rps_adapted": round(adapted_lats.size / measured_window, 4),
        "throughput_rps_plain":   round(plain_lats.size   / measured_window, 4),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Save] {len(records)} requests → {out_dir}")
    print(f"[Summary] K={num_adapted}/{num_tasks} adapted | "
          f"tput={summary['aggregate_throughput_rps']} rps | "
          f"avg_lat={summary['avg_latency_ms_all']} ms")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device-url",           default="localhost:8000")
    p.add_argument("--backbone",             default="momentlarge")
    p.add_argument("--base-task",            default="ecgclass",
                   choices=list(BASE_TASK_TYPES.keys()))
    p.add_argument("--num-tasks",            type=int, default=10,
                   help="Total number of deployed task replicas (fixed across runs).")
    p.add_argument("--num-adapted",          type=int, required=True,
                   help="How many of the deployed tasks carry a LoRA adapter (0..num-tasks).")
    p.add_argument("--concurrency-per-task", type=int, default=1)
    p.add_argument("--duration",             type=float, default=60.0)
    p.add_argument("--warmup-secs",          type=float, default=5.0)
    p.add_argument("--exp-dir",              default=os.environ.get(
                   "EXP_DIR", "experiments/lora_adapter_fraction/results/run"))
    args = p.parse_args()

    if args.num_tasks < 1:
        raise ValueError("--num-tasks must be >= 1")
    if not 0 <= args.num_adapted <= args.num_tasks:
        raise ValueError(f"--num-adapted must be in [0, {args.num_tasks}]")

    out_dir = (SERVING_DIR / args.exp_dir).resolve()
    print("=" * 70)
    print("  lora_adapter_fraction — closed-loop sweep")
    print(f"  Backbone             : {args.backbone}")
    print(f"  Base task            : {args.base_task}")
    print(f"  Total tasks (N)      : {args.num_tasks}")
    print(f"  Adapted tasks (K)    : {args.num_adapted}")
    print(f"  Concurrency / task   : {args.concurrency_per_task}")
    print(f"  Duration             : {args.duration}s (warmup={args.warmup_secs}s)")
    print(f"  Device URL           : {args.device_url}")
    print(f"  Results              : {out_dir}")
    print("=" * 70)

    data = build_base_data(args.base_task)

    resp = asyncio.run(deploy_mixed(
        args.device_url, args.backbone, args.base_task,
        args.num_tasks, args.num_adapted,
    ))
    if "error" in resp.get("status", "").lower():
        print(f"[Error] Deploy failed: {resp}")
        return 1

    asyncio.run(asyncio.sleep(1))

    task_has_adapter = {
        _replica_task_name(args.base_task, i): (i < args.num_adapted)
        for i in range(args.num_tasks)
    }

    req_timeout = max(60.0, args.duration * 2)
    records = asyncio.run(run_closed_loop(
        device_url=args.device_url,
        task_has_adapter=task_has_adapter,
        data=data,
        concurrency_per_task=args.concurrency_per_task,
        duration=args.duration,
        req_timeout=req_timeout,
    ))

    save_results(
        records, out_dir,
        duration=args.duration,
        warmup_secs=args.warmup_secs,
        num_tasks=args.num_tasks,
        num_adapted=args.num_adapted,
        concurrency_per_task=args.concurrency_per_task,
        backbone=args.backbone,
        base_task=args.base_task,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
