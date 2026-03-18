#!/usr/bin/env python3
"""isolation_overhead — Compare none / shared / process isolation modes.

Sends requests directly to a running device gRPC server (no orchestrator).
Uses a closed-loop sender: the next request is sent only after the previous
one completes. This isolates pure per-request latency overhead.

Metrics per mode:
  - avg / p50 / p95 / p99 end-to-end latency (ms)
  - throughput (req/s)
  - GPU memory (allocated MB, reserved MB)

Usage (from serving/):
    python experiments/isolation_overhead/run.py \\
        --backbone momentbase --task ecgclass --n-requests 100 \\
        --isolation-mode shared
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
from typing import Dict, List

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import numpy as np
from torch.utils.data import DataLoader

from site_manager.grpc_client import EdgeRuntimeClient
from site_manager.config import DATASET_DIR as _DATASET_DIR


# ---------------------------------------------------------------------------
# Task library
# ---------------------------------------------------------------------------

TASK_TYPES: Dict[str, str] = {
    "ecgclass":     "classification",
    "gestureclass": "classification",
    "sysbp":        "regression",
    "diasbp":       "regression",
    "heartrate":    "regression",
    "eclfore":      "forecasting",
    "etth1fore":    "forecasting",
    "exchangefore": "forecasting",
    "trafficfore":  "forecasting",
    "weatherfore":  "forecasting",
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_sample(task: str) -> Dict:
    """Load a single sample for the given task."""
    from fmtk.datasetloaders.ecg5000 import ECG5000Dataset
    from fmtk.datasetloaders.uwavegesture import UWaveGestureLibraryALLDataset
    from fmtk.datasetloaders.ppg import PPGDataset
    from fmtk.datasetloaders.ecl import ECLDataset
    from fmtk.datasetloaders.etth1 import ETTh1Dataset
    from fmtk.datasetloaders.exchange import ExchangeDataset
    from fmtk.datasetloaders.traffic import TrafficDataset
    from fmtk.datasetloaders.weather import WeatherDataset

    d = _DATASET_DIR
    cfg = {"batch_size": 1, "shuffle": False}
    loaders = {
        "ecgclass":     lambda: DataLoader(ECG5000Dataset({"dataset_path": f"{d}/ECG5000"}, {"task_type": "classification"}, "test"), **cfg),
        "gestureclass": lambda: DataLoader(UWaveGestureLibraryALLDataset({"dataset_path": f"{d}/UWaveGestureLibraryAll", "seq_len": 512}, {"task_type": "classification"}, "test"), **cfg),
        "sysbp":        lambda: DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data", "seq_len": 512, "num_channels": 1}, {"task_type": "regression", "label": "sysbp"}, "test"), **cfg),
        "diasbp":       lambda: DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data", "seq_len": 512, "num_channels": 1}, {"task_type": "regression", "label": "diasbp"}, "test"), **cfg),
        "heartrate":    lambda: DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data", "seq_len": 512, "num_channels": 1}, {"task_type": "regression", "label": "hr"}, "test"), **cfg),
        "eclfore":      lambda: DataLoader(ECLDataset({"dataset_path": f"{d}/ElectricityLoad-data"}, {"task_type": "forecasting", "seq_len": 512}, "test"), **cfg),
        "etth1fore":    lambda: DataLoader(ETTh1Dataset({"dataset_path": f"{d}/ETTh1"}, {"task_type": "forecasting", "seq_len": 512}, "test"), **cfg),
        "exchangefore": lambda: DataLoader(ExchangeDataset({"dataset_path": f"{d}/Exchange"}, {"task_type": "forecasting", "seq_len": 512}, "test"), **cfg),
        "trafficfore":  lambda: DataLoader(TrafficDataset({"dataset_path": f"{d}/Traffic"}, {"task_type": "forecasting", "seq_len": 512}, "test"), **cfg),
        "weatherfore":  lambda: DataLoader(WeatherDataset({"dataset_path": f"{d}/Weather"}, {"task_type": "forecasting"}, "test"), **cfg),
    }
    if task not in loaders:
        raise ValueError(f"Unknown task: {task}")
    loader = loaders[task]()
    batch = next(iter(loader))
    x    = batch["x"].numpy().astype(np.float32)
    mask = batch.get("mask")
    if mask is not None:
        mask = mask.numpy().astype(np.float32)
    print(f"[Data] Loaded {task}: x.shape={x.shape}")
    return {"x": x, "mask": mask}


# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

async def deploy(device_url: str, backbone: str, task: str) -> Dict:
    client = EdgeRuntimeClient(device_url)
    try:
        await client.wait_ready()
        decoders = [{"task": task, "type": TASK_TYPES[task], "path": f"{task}_{backbone}_mlp"}]
        payload  = json.dumps({"backbone": backbone, "decoders": decoders})
        print(f"[Deploy] backbone={backbone}  task={task} ...")
        resp = await client.control("load", payload)
        print(f"[Deploy] status={resp['status']}")
        if "error" in resp.get("status", "").lower():
            raise RuntimeError(f"Deploy failed: {resp}")
        mem = _parse_gpu_mb(resp.get("logger_summary", ""))
        print(f"[Deploy] gpu_allocated={mem['gpu_allocated_mb']:.1f} MB ")
        return mem
    finally:
        await client.close()


def _parse_gpu_mb(logger_summary_str: str) -> Dict:
    try:
        summary = eval(logger_summary_str)
        allocated = reserved = 0.0
        for section_stats in summary.values():
            if isinstance(section_stats, dict):
                allocated += float(section_stats.get("gpu allocated", section_stats.get("gpu peak", 0.0)))
                reserved  += float(section_stats.get("gpu reserved", 0.0))
        return {"gpu_allocated_mb": allocated}
    except Exception as e:
        print(f"[Memory] Parse failed: {e!r}  raw={logger_summary_str!r}")
        return {"gpu_allocated_mb": 0.0}


# ---------------------------------------------------------------------------
# Closed-loop sender
# ---------------------------------------------------------------------------

async def run_closed_loop(
    device_url: str,
    task: str,
    sample: Dict,
    duration: float,
) -> List[float]:
    """Send requests one at a time for `duration` seconds. Returns latencies (s)."""
    client = EdgeRuntimeClient(device_url)
    await client.wait_ready()

    latencies: List[float] = []
    req_id = 0
    deadline = time.time() + duration
    while time.time() < deadline:
        t_send = time.time()
        try:
            await client.infer({
                "req_id": req_id,
                "task":   task,
                "x":      sample["x"],
                "mask":   sample.get("mask"),
            })
            latencies.append(time.time() - t_send)
        except Exception as e:
            print(f"[Run] req_id={req_id} error: {e}")
        req_id += 1

    await client.close()
    n = len(latencies)
    print(f"[Run] completed={n}  "
          f"avg={sum(latencies)/n*1000:.1f}ms  "
          f"p99={np.percentile(latencies, 99)*1000:.1f}ms")
    return latencies


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

SUMMARY_FIELDS = [
    "isolation_mode", "backbone", "task",
    "duration_s", "n_completed",
    "avg_lat_ms", "p50_lat_ms", "p95_lat_ms", "p99_lat_ms",
    "throughput_rps",
    "gpu_allocated_mb",
]


def _compute_row(isolation_mode: str, backbone: str, task: str,
                 duration: float, latencies: List[float],
                 gpu_mem: Dict) -> Dict:
    n = len(latencies)
    return {
        "isolation_mode":  isolation_mode,
        "backbone":        backbone,
        "task":            task,
        "duration_s":      duration,
        "n_completed":     n,
        "avg_lat_ms":      round(sum(latencies) / n * 1000 if n else 0.0, 3),
        "p50_lat_ms":      round(float(np.percentile(latencies, 50)) * 1000 if n else 0.0, 3),
        "p95_lat_ms":      round(float(np.percentile(latencies, 95)) * 1000 if n else 0.0, 3),
        "p99_lat_ms":      round(float(np.percentile(latencies, 99)) * 1000 if n else 0.0, 3),
        "throughput_rps":  round(n / duration if duration > 0 else 0.0, 4),
        "gpu_allocated_mb": round(gpu_mem.get("gpu_allocated_mb", 0.0), 2),
    }


def _append_summary(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def _print_row(row: Dict) -> None:
    print(
        f"\n  {'mode':<10}  {'task':<14}  {'n':>6}  "
        f"{'avg_ms':>8}  {'p50_ms':>8}  {'p95_ms':>8}  {'p99_ms':>8}  "
        f"{'rps':>6}  {'gpu_alloc_mb':>12}  {'gpu_res_mb':>10}"
    )
    print(
        f"  {row['isolation_mode']:<10}  {row['task']:<14}  {row['n_completed']:>6}  "
        f"{float(row['avg_lat_ms']):>8.1f}  {float(row['p50_lat_ms']):>8.1f}  "
        f"{float(row['p95_lat_ms']):>8.1f}  {float(row['p99_lat_ms']):>8.1f}  "
        f"{float(row['throughput_rps']):>6.2f}  "
        f"{float(row['gpu_allocated_mb']):>12.1f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="isolation_overhead experiment")
    parser.add_argument("--device-url",     default=os.environ.get("DEVICE_URL", "localhost:8000"))
    parser.add_argument("--backbone",       default=os.environ.get("BACKBONE", "momentbase"))
    parser.add_argument("--task",           default=os.environ.get("TASK", "ecgclass"))
    parser.add_argument("--duration",       type=float, default=float(os.environ.get("DURATION", "30")))
    parser.add_argument("--isolation-mode", default=os.environ.get("ISOLATION_MODE", "shared"),
                        choices=["shared", "process", "none"])
    parser.add_argument("--exp-dir",        default=os.environ.get("EXP_DIR",
                                            "experiments/isolation_overhead/results"))
    parser.add_argument("--skip-load",      action="store_true",
                        default=os.environ.get("SKIP_LOAD", "").lower() in ("1", "true"))
    args = parser.parse_args()

    summary_path = (SERVING_DIR / args.exp_dir / "summary.csv").resolve()

    print("\n" + "═" * 60)
    print("  isolation_overhead experiment")
    print("═" * 60)
    print(f"  Device URL     : {args.device_url}")
    print(f"  Backbone       : {args.backbone}")
    print(f"  Task           : {args.task}")
    print(f"  Isolation mode : {args.isolation_mode}")
    print(f"  Duration       : {args.duration}s")
    print(f"  Summary        : {summary_path}")
    print("═" * 60)

    sample = load_sample(args.task)

    async def run() -> Dict:
        gpu_mem = {"gpu_allocated_mb": 0.0}
        if not args.skip_load:
            gpu_mem = await deploy(args.device_url, args.backbone, args.task)
            await asyncio.sleep(1)

        print(f"\n[Run] Starting closed-loop for {args.duration}s ...")
        latencies = await run_closed_loop(
            device_url=args.device_url,
            task=args.task,
            sample=sample,
            duration=args.duration,
        )
        return _compute_row(
            args.isolation_mode, args.backbone, args.task,
            args.duration, latencies, gpu_mem,
        )

    row = asyncio.run(run())
    _append_summary(summary_path, row)
    _print_row(row)
    print(f"\n[INFO] Summary appended to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
