#!/usr/bin/env python3
"""Closed-loop TPC sweep for ecgclass on a single device server.

This experiment reuses the gRPC device server path and varies the number of
TPCs assigned to that server. For each TPC budget, it runs a closed-loop load:
each worker sends the next request only after the previous response returns.

Usage (typically called by run.sh):
    python experiments/tpc_closed_loop_ecg/run.py \
        --device-url localhost:8000 \
        --backbone momentbase \
        --concurrency 4 \
        --duration 180 \
        --tpc-count 54 \
        --exp-dir experiments/tpc_closed_loop_ecg/results/tpc_54/cc_4
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

import numpy as np
from torch.utils.data import DataLoader

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

from site_manager.config import DATASET_DIR as _DATASET_DIR
from site_manager.grpc_client import EdgeRuntimeClient

TASK = "ecgclass"
TASK_TYPE = "classification"

Record = Tuple[float, float, float, float, float, float, float, float, int, int]
# (
#   send_elapsed_sec,
#   complete_elapsed_sec,
#   latency_ms,
#   total_response_time_ms,
#   server_exec_ms,
#   server_proc_ms,
#   server_swap_ms,
#   queue_wait_plus_rpc_ms,
#   server_start_ns,
#   worker_id,
# )


def build_data() -> Dict[str, np.ndarray | None]:
    from fmtk.datasetloaders.ecg5000 import ECG5000Dataset

    loader = DataLoader(
        ECG5000Dataset(
            {"dataset_path": f"{_DATASET_DIR}/ECG5000"},
            {"task_type": "classification"},
            "test",
        ),
        batch_size=1,
        shuffle=False,
    )
    batch = next(iter(loader))
    data = {
        "x": batch["x"].numpy().astype(np.float32),
        "mask": batch["mask"].numpy().astype(np.float32) if "mask" in batch else None,
    }
    print(f"[Data] Loaded {TASK}: x.shape={data['x'].shape}")
    return data


async def deploy(device_url: str, backbone: str) -> None:
    decoders = [{"task": TASK, "type": TASK_TYPE, "path": f"{TASK}_{backbone}_mlp"}]
    client = EdgeRuntimeClient(device_url)
    try:
        await client.wait_ready()
        payload = json.dumps({"backbone": backbone, "decoders": decoders})
        print(f"[Deploy] {device_url} backbone={backbone} task={TASK}")
        resp = await client.control("load", payload)
        print(f"[Deploy] {device_url} status={resp['status']}")
    finally:
        await client.close()


async def run_closed_loop(
    device_url: str,
    data: Dict[str, np.ndarray | None],
    concurrency: int,
    duration: float,
    req_timeout: float = 60.0,
) -> List[Record]:
    client = EdgeRuntimeClient(device_url)
    await client.wait_ready()
    records: List[Record] = []
    start_wall = time.time()
    req_counter = 0
    req_lock = asyncio.Lock()
    record_lock = asyncio.Lock()

    async def _next_req_id() -> int:
        nonlocal req_counter
        async with req_lock:
            req_id = req_counter
            req_counter += 1
            return req_id

    async def _worker(worker_id: int) -> None:
        while True:
            send_abs = time.time()
            if send_abs - start_wall >= duration:
                return
            req_id = await _next_req_id()
            try:
                resp = await asyncio.wait_for(
                    client.infer({
                        "req_id": req_id,
                        "task": TASK,
                        "x": data["x"],
                        "mask": data.get("mask"),
                    }),
                    timeout=req_timeout,
                )
            except Exception:
                continue

            done_abs = time.time()
            total_response_time_ms = (done_abs - send_abs) * 1000.0
            server_start_s = resp["start_time_ns"] / 1e9
            server_exec_ms = (resp["end_time_ns"] - resp["start_time_ns"]) / 1e6
            server_proc_ms = resp["proc_time_ns"] / 1e6
            server_swap_ms = resp["swap_time_ns"] / 1e6
            queue_wait_plus_rpc_ms = max(0.0, (server_start_s - send_abs) * 1000.0)

            async with record_lock:
                records.append((
                    send_abs - start_wall,
                    done_abs - start_wall,
                    total_response_time_ms,
                    total_response_time_ms,
                    server_exec_ms,
                    server_proc_ms,
                    server_swap_ms,
                    queue_wait_plus_rpc_ms,
                    resp["start_time_ns"],
                    worker_id,
                ))

    await asyncio.gather(*[_worker(worker_id) for worker_id in range(concurrency)])
    await client.close()
    return records


def save_results(
    records: List[Record],
    out_dir: Path,
    duration: float,
    warmup_secs: float,
    concurrency: int,
    tpc_count: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "latencies.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "task",
            "tpc_count",
            "concurrency",
            "send_elapsed_sec",
            "complete_elapsed_sec",
            "latency_ms",
            "total_response_time_ms",
            "server_exec_ms",
            "server_proc_ms",
            "server_swap_ms",
            "queue_wait_plus_rpc_ms",
            "non_server_exec_overhead_ms",
            "server_start_ns",
            "worker_id",
        ])
        for rec in records:
            (
                send_elapsed_sec,
                complete_elapsed_sec,
                latency_ms,
                total_response_time_ms,
                server_exec_ms,
                server_proc_ms,
                server_swap_ms,
                queue_wait_plus_rpc_ms,
                server_start_ns,
                worker_id,
            ) = rec
            w.writerow([
                TASK,
                tpc_count,
                concurrency,
                round(send_elapsed_sec, 4),
                round(complete_elapsed_sec, 4),
                round(latency_ms, 4),
                round(total_response_time_ms, 4),
                round(server_exec_ms, 4),
                round(server_proc_ms, 4),
                round(server_swap_ms, 4),
                round(queue_wait_plus_rpc_ms, 4),
                round(max(0.0, total_response_time_ms - server_exec_ms), 4),
                server_start_ns,
                worker_id,
            ])

    trimmed = [rec for rec in records if rec[1] > warmup_secs]
    latencies = [rec[2] for rec in trimmed]
    mean_response_time_ms = float(np.mean([rec[3] for rec in trimmed])) if trimmed else None
    mean_service_time_ms = float(np.mean([rec[4] for rec in trimmed])) if trimmed else None
    mean_proc_time_ms = float(np.mean([rec[5] for rec in trimmed])) if trimmed else None

    with (out_dir / "task_results.csv").open("w", newline="") as f:
        fields = [
            "task",
            "tpc_count",
            "concurrency",
            "n_requests",
            "throughput_rps",
            "avg_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "avg_total_response_time_ms",
            "avg_server_exec_ms",
            "avg_server_proc_ms",
            "avg_server_swap_ms",
            "avg_queue_wait_plus_rpc_ms",
            "avg_non_server_exec_overhead_ms",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        if latencies:
            w.writerow({
                "task": TASK,
                "tpc_count": tpc_count,
                "concurrency": concurrency,
                "n_requests": len(trimmed),
                "throughput_rps": round(len(trimmed) / max(duration - warmup_secs, 1e-9), 4),
                "avg_latency_ms": round(float(np.mean(latencies)), 3),
                "p50_latency_ms": round(float(np.percentile(latencies, 50)), 3),
                "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
                "p99_latency_ms": round(float(np.percentile(latencies, 99)), 3),
                "avg_total_response_time_ms": round(float(np.mean([rec[3] for rec in trimmed])), 3),
                "avg_server_exec_ms": round(mean_service_time_ms, 3),
                "avg_server_proc_ms": round(mean_proc_time_ms, 3),
                "avg_server_swap_ms": round(float(np.mean([rec[6] for rec in trimmed])), 3),
                "avg_queue_wait_plus_rpc_ms": round(float(np.mean([rec[7] for rec in trimmed])), 3),
                "avg_non_server_exec_overhead_ms": round(
                    float(np.mean([max(0.0, rec[3] - rec[4]) for rec in trimmed])),
                    3,
                ),
            })

    with (out_dir / "summary.json").open("w") as f:
        json.dump({
            "task": TASK,
            "tpc_count": tpc_count,
            "concurrency": concurrency,
            "duration_s": duration,
            "warmup_s": warmup_secs,
            "total_completed_requests": len(records),
            "completed_after_warmup": len(trimmed),
            "mean_response_time_ms": round(mean_response_time_ms, 3) if mean_response_time_ms is not None else None,
            "mean_service_time_ms": round(mean_service_time_ms, 3) if mean_service_time_ms is not None else None,
            "mean_proc_time_ms": round(mean_proc_time_ms, 3) if mean_proc_time_ms is not None else None,
        }, f, indent=2)

    print(f"[Save] {len(records)} requests -> {out_dir}")
    if mean_service_time_ms is not None:
        print(
            f"[Summary] tpc_count={tpc_count} "
            f"avg_response_time_ms={mean_response_time_ms:.3f} "
            f"mean_service_time_ms={mean_service_time_ms:.3f} "
            f"mean_proc_time_ms={mean_proc_time_ms:.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-url", default="localhost:8000")
    parser.add_argument("--backbone", default=os.environ.get("BACKBONE", "momentbase"))
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", "1")))
    parser.add_argument("--duration", type=float, default=float(os.environ.get("PHASE_DURATION", "180")))
    parser.add_argument("--warmup-secs", type=float, default=float(os.environ.get("WARMUP_SECS", "10")))
    parser.add_argument("--tpc-count", type=int, required=True)
    parser.add_argument(
        "--exp-dir",
        default=os.environ.get("EXP_DIR", "experiments/tpc_closed_loop_ecg/results"),
    )
    args = parser.parse_args()

    out_dir = (SERVING_DIR / args.exp_dir).resolve()

    print("=" * 72)
    print("  Closed-Loop ECG TPC Sweep")
    print(f"  Backbone     : {args.backbone}")
    print(f"  Device URL   : {args.device_url}")
    print(f"  TPC count    : {args.tpc_count}")
    print(f"  Concurrency  : {args.concurrency}")
    print(f"  Duration     : {args.duration}s (warmup={args.warmup_secs}s)")
    print(f"  Results      : {out_dir}")
    print("=" * 72)

    data = build_data()
    asyncio.run(deploy(args.device_url, args.backbone))
    asyncio.run(asyncio.sleep(1))

    print(f"[Run] Starting closed-loop traffic for {args.duration}s ...")
    records = asyncio.run(run_closed_loop(
        device_url=args.device_url,
        data=data,
        concurrency=args.concurrency,
        duration=args.duration,
    ))

    save_results(
        records=records,
        out_dir=out_dir,
        duration=args.duration,
        warmup_secs=args.warmup_secs,
        concurrency=args.concurrency,
        tpc_count=args.tpc_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
