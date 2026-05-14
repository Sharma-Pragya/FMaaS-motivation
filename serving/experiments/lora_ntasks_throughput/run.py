#!/usr/bin/env python3
"""lora_ntasks_throughput — Throughput sweep over N LoRA-adapted tasks.

Loads N replicas of a base task (default: ecgclass) on a single backbone
(default: momentlarge) using MLP decoder + LoRA adapter. Each replica is a
synthetic app id (e.g. ecgclass__app0, ecgclass__app1, ...) that resolves
back to the base task for decoder/adapter lookup.

Drives a closed-loop workload: K workers per task, each sending the next
request only after the previous response returns. Aggregate throughput is
reported as a function of N (chosen via --num-tasks).

Usage (from serving/):
    python experiments/lora_ntasks_throughput/run.py \
        --device-url localhost:8000 \
        --backbone momentlarge \
        --base-task ecgclass \
        --num-tasks 4 \
        --concurrency-per-task 1 \
        --duration 60 \
        --exp-dir experiments/lora_ntasks_throughput/results/N4
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


# ---------------------------------------------------------------------------
# Base-task library (extend here if you want other base tasks)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

def _replica_task_name(base_task: str, i: int) -> str:
    # First replica keeps the base name so it works even when num_tasks == 1.
    return base_task if i == 0 else f"{base_task}__app{i}"


def _task_spec(task_name: str, base_task: str, use_adapter: bool = True) -> dict:
    # path=None triggers random-init for both decoder and adapter
    # (see fmtk.pipeline.add_decoder and add_adapter). Throughput / latency
    # are weight-invariant, so untrained weights are fine and avoid needing
    # finetuned checkpoints on disk.
    return {
        "task":      task_name,
        "base_task": base_task,
        "type":      BASE_TASK_TYPES[base_task],
        "path":      None,
        "adapter":   BASE_TASK_ADAPTERS[base_task] if use_adapter else None,
    }


async def deploy_replicas(device_url: str, backbone: str, base_task: str,
                          num_tasks: int, use_adapter: bool = True) -> dict:
    """sharing mode: deploy N replicas (decoder ± LoRA adapter) on one backbone."""
    task_specs = [
        _task_spec(_replica_task_name(base_task, i), base_task, use_adapter=use_adapter)
        for i in range(num_tasks)
    ]
    adapter_str = "lora" if use_adapter else "none"
    print(f"[Deploy] {device_url} backbone={backbone} replicas={num_tasks} "
          f"base={base_task} adapter={adapter_str}")
    client = EdgeRuntimeClient(device_url)
    try:
        await client.wait_ready()
        payload = json.dumps({"backbone": backbone, "decoders": task_specs})
        resp = await client.control("load", payload)
        print(f"[Deploy] status={resp['status']}")
        return resp
    finally:
        await client.close()


async def deploy_no_sharing(device_urls: List[str], backbone: str,
                            base_task: str) -> List[dict]:
    """no_sharing mode: deploy 1 LoRA-adapted task per device server.

    One backbone instance per server, one adapter on top — the per-server
    setup is identical to the sharing case at N=1, but there are N servers.
    """
    async def _one(idx: int, url: str) -> dict:
        task_name = _replica_task_name(base_task, idx)
        spec = _task_spec(task_name, base_task)
        print(f"[Deploy] {url} backbone={backbone} task={task_name} (no_sharing)")
        client = EdgeRuntimeClient(url)
        try:
            await client.wait_ready()
            payload = json.dumps({"backbone": backbone, "decoders": [spec]})
            resp = await client.control("load", payload)
            print(f"[Deploy] {url} status={resp['status']}")
            return resp
        finally:
            await client.close()

    # Sequential deploy so backbone loads don't all hammer the GPU at once.
    return [await _one(i, url) for i, url in enumerate(device_urls)]


# ---------------------------------------------------------------------------
# Closed-loop sender
# ---------------------------------------------------------------------------

# (task, send_elapsed_s, latency_ms, server_exec_ms, worker_id)
Record = Tuple[str, float, float, float, int]


async def run_closed_loop(
    task_to_url: Dict[str, str],
    data: Dict[str, np.ndarray | None],
    concurrency_per_task: int,
    duration: float,
    req_timeout: float = 60.0,
) -> List[Record]:
    """One gRPC client per *distinct* device URL; workers reuse them.

    For sharing mode all tasks map to the same URL → 1 client total.
    For no_sharing mode each task gets its own URL → N clients.
    """
    unique_urls = sorted(set(task_to_url.values()))
    clients = {url: EdgeRuntimeClient(url) for url in unique_urls}
    await asyncio.gather(*(c.wait_ready() for c in clients.values()))

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
        stub = clients[task_to_url[task]]._stub
        proto = encode_infer_request(task=task, x=data["x"], mask=data.get("mask"))
        while True:
            now = time.time()
            if now - start_wall >= duration:
                return
            proto.req_id = await _next_req_id()
            send_abs = time.time()
            try:
                resp = await asyncio.wait_for(stub.Infer(proto), timeout=req_timeout)
            except Exception:
                continue
            done_abs = time.time()
            latency_ms = (done_abs - send_abs) * 1000.0
            server_exec_ms = (resp.end_time_ns - resp.start_time_ns) / 1e6 if resp.end_time_ns else 0.0
            async with rec_lock:
                records.append((task, send_abs - start_wall, latency_ms, server_exec_ms, worker_id))

    workers = [
        _worker(task, w)
        for task in task_to_url
        for w in range(concurrency_per_task)
    ]
    await asyncio.gather(*workers, return_exceptions=True)
    await asyncio.gather(*(c.close() for c in clients.values()))
    return records


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(records: List[Record], out_dir: Path, *, duration: float,
                 warmup_secs: float, num_tasks: int, concurrency_per_task: int,
                 backbone: str, base_task: str, mode: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-request CSV
    with (out_dir / "latencies.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "send_elapsed_s", "latency_ms", "server_exec_ms", "worker_id"])
        for task, t, lat, sx, wid in records:
            w.writerow([task, f"{t:.4f}", f"{lat:.3f}", f"{sx:.3f}", wid])

    # Per-task summary (after warmup)
    trimmed = [r for r in records if r[1] > warmup_secs]
    measured_window = max(duration - warmup_secs, 1e-9)

    by_task: Dict[str, List[Record]] = {}
    for r in trimmed:
        by_task.setdefault(r[0], []).append(r)

    per_task_rows = []
    for task in sorted(by_task):
        lat = np.array([r[2] for r in by_task[task]])
        sx  = np.array([r[3] for r in by_task[task]])
        per_task_rows.append({
            "task":             task,
            "n_requests":       int(len(lat)),
            "throughput_rps":   round(len(lat) / measured_window, 4),
            "avg_latency_ms":   round(float(lat.mean()), 3) if len(lat) else None,
            "p50_latency_ms":   round(float(np.percentile(lat, 50)), 3) if len(lat) else None,
            "p95_latency_ms":   round(float(np.percentile(lat, 95)), 3) if len(lat) else None,
            "p99_latency_ms":   round(float(np.percentile(lat, 99)), 3) if len(lat) else None,
            "avg_server_exec_ms": round(float(sx.mean()), 3) if len(sx) else None,
        })

    with (out_dir / "per_task.csv").open("w", newline="") as f:
        fields = list(per_task_rows[0].keys()) if per_task_rows else \
                 ["task", "n_requests", "throughput_rps", "avg_latency_ms",
                  "p50_latency_ms", "p95_latency_ms", "p99_latency_ms", "avg_server_exec_ms"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in per_task_rows:
            w.writerow(row)

    total_n = len(trimmed)
    all_lat = np.array([r[2] for r in trimmed]) if trimmed else np.array([])
    summary = {
        "mode":                   mode,
        "backbone":               backbone,
        "base_task":              base_task,
        "num_tasks":              num_tasks,
        "concurrency_per_task":   concurrency_per_task,
        "duration_s":             duration,
        "warmup_s":               warmup_secs,
        "n_requests_total":       total_n,
        "n_requests_pre_warmup":  len(records) - total_n,
        "aggregate_throughput_rps": round(total_n / measured_window, 4),
        "avg_latency_ms_all":     round(float(all_lat.mean()), 3) if total_n else None,
        "p50_latency_ms_all":     round(float(np.percentile(all_lat, 50)), 3) if total_n else None,
        "p95_latency_ms_all":     round(float(np.percentile(all_lat, 95)), 3) if total_n else None,
        "p99_latency_ms_all":     round(float(np.percentile(all_lat, 99)), 3) if total_n else None,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Save] {len(records)} requests → {out_dir}")
    print(f"[Summary] N={num_tasks} K={concurrency_per_task} "
          f"aggregate_throughput={summary['aggregate_throughput_rps']} rps "
          f"avg_latency={summary['avg_latency_ms_all']} ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",                 default="sharing",
                        choices=["sharing", "no_sharing", "sharing_no_adapter"],
                        help="sharing: N adapters on 1 shared backbone (single device). "
                             "no_sharing: 1 task per device server (N independent backbones). "
                             "sharing_no_adapter: N decoders on 1 shared backbone, no LoRA.")
    parser.add_argument("--device-url",           default="localhost:8000",
                        help="Used in sharing / sharing_no_adapter modes. "
                             "Ignored when --device-urls is provided.")
    parser.add_argument("--device-urls",          default=None,
                        help="Comma-separated list of N device URLs (no_sharing mode). "
                             "Required if --mode no_sharing; length must equal --num-tasks.")
    parser.add_argument("--backbone",             default="momentlarge")
    parser.add_argument("--base-task",            default="ecgclass",
                        choices=list(BASE_TASK_TYPES.keys()))
    parser.add_argument("--num-tasks",            type=int, required=True,
                        help="Number of LoRA-adapted task replicas to deploy.")
    parser.add_argument("--concurrency-per-task", type=int, default=1)
    parser.add_argument("--duration",             type=float, default=60.0)
    parser.add_argument("--warmup-secs",          type=float, default=5.0)
    parser.add_argument("--exp-dir",              default=os.environ.get(
                        "EXP_DIR", "experiments/lora_ntasks_throughput/results/run"))
    args = parser.parse_args()

    if args.num_tasks < 1:
        raise ValueError("--num-tasks must be >= 1")

    # Resolve task -> device URL mapping based on mode.
    if args.mode in ("sharing", "sharing_no_adapter"):
        device_urls = [args.device_url] * args.num_tasks
    else:
        if not args.device_urls:
            raise ValueError("--mode no_sharing requires --device-urls (comma-separated, one per task).")
        device_urls = [u.strip() for u in args.device_urls.split(",") if u.strip()]
        if len(device_urls) != args.num_tasks:
            raise ValueError(
                f"--device-urls has {len(device_urls)} entries but --num-tasks={args.num_tasks}."
            )

    tasks = [_replica_task_name(args.base_task, i) for i in range(args.num_tasks)]
    task_to_url: Dict[str, str] = dict(zip(tasks, device_urls))

    out_dir = (SERVING_DIR / args.exp_dir).resolve()

    use_adapter = args.mode != "sharing_no_adapter"
    backbone_str = f"{args.backbone} + LoRA" if use_adapter else f"{args.backbone} (no adapter)"

    print("=" * 70)
    print("  lora_ntasks_throughput — closed-loop throughput sweep")
    print(f"  Mode                 : {args.mode}")
    print(f"  Backbone             : {backbone_str}")
    print(f"  Base task            : {args.base_task}")
    print(f"  Num tasks (replicas) : {args.num_tasks}")
    print(f"  Concurrency / task   : {args.concurrency_per_task}")
    print(f"  Duration             : {args.duration}s (warmup={args.warmup_secs}s)")
    if args.mode in ("sharing", "sharing_no_adapter"):
        print(f"  Device URL           : {args.device_url}")
    else:
        print(f"  Device URLs          : {device_urls}")
    print(f"  Results              : {out_dir}")
    print("=" * 70)

    data = build_base_data(args.base_task)

    if args.mode in ("sharing", "sharing_no_adapter"):
        resp = asyncio.run(deploy_replicas(
            args.device_url, args.backbone, args.base_task, args.num_tasks,
            use_adapter=use_adapter,
        ))
        if "error" in resp.get("status", "").lower():
            print(f"[Error] Deploy failed: {resp}")
            return 1
    else:
        resps = asyncio.run(deploy_no_sharing(
            device_urls, args.backbone, args.base_task,
        ))
        bad = [r for r in resps if "error" in r.get("status", "").lower()]
        if bad:
            print(f"[Error] Deploy failed on {len(bad)} server(s): {bad}")
            return 1

    asyncio.run(asyncio.sleep(1))

    req_timeout = max(60.0, args.duration * 2)
    records = asyncio.run(run_closed_loop(
        task_to_url=task_to_url,
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
        concurrency_per_task=args.concurrency_per_task,
        backbone=args.backbone,
        base_task=args.base_task,
        mode=args.mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
