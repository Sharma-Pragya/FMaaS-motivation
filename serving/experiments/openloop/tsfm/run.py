#!/usr/bin/env python3
"""motivation2/tsfm — Open-loop RPS benchmark against a running device gRPC server.

Assumes the device server is already running:
    python device/main.py --port 8000 --runtime-type pytorch --cuda cuda:0

This script:
  1. Sends a Control(load) to load backbone + decoder on the device.
  2. Fires Poisson-arrival open-loop requests at each target RPS in --rps-sweep.
  3. Records sent_count, completed_count, latency per request.
  4. Sweeps across --n-tasks (one task per run, multiple tasks share one backbone/device).
  5. Saves per-run CSVs and a summary.csv.

Usage:
    python run.py \
        --device-url gpu01:8000 \
        --backbone momentbase \
        --n-tasks 1,2,4 \
        --rps-sweep 2,5,10,20 \
        --duration 30
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

SERVING_DIR = Path(__file__).resolve().parents[3]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import numpy as np
from torch.utils.data import DataLoader

# Reuse site_manager gRPC client and dataset loaders
from site_manager.grpc_client import EdgeRuntimeClient
from site_manager.config import DATASET_DIR as _DATASET_DIR

# ---------------------------------------------------------------------------
# Task library (same as motivation1/tsfm)
# ---------------------------------------------------------------------------

TASK_ORDER: List[str] = [
    "ecgclass",
    "gestureclass",
    "sysbp",
    "diasbp",
    "heartrate",
    "eclfore",
    "etth1fore",
    "exchangefore",
    "trafficfore",
    "weatherfore",
]

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
# Dataset loading (reuse fmtk loaders, same as motivation1/tsfm)
# ---------------------------------------------------------------------------

def build_data(tasks: List[str]) -> Dict[str, Dict]:
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
    all_loaders = {
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
    data = {}
    for task in tasks:
        if task not in all_loaders:
            raise ValueError(f"Unknown task: {task}")
        loader = all_loaders[task]()
        batch = next(iter(loader))
        data[task] = {
            "x":    batch["x"].numpy().astype(np.float32),
            "mask": batch.get("mask", None),
        }
        if data[task]["mask"] is not None:
            data[task]["mask"] = data[task]["mask"].numpy().astype(np.float32)
        print(f"[Data] Loaded {task}: x.shape={data[task]['x'].shape}")
    return data


# ---------------------------------------------------------------------------
# Deploy: Control(load) on already-running device server
# ---------------------------------------------------------------------------

async def deploy_backbone_async(device_url: str, backbone: str, decoders: List[str]) -> dict:
    """Send load control command to pre-running device server."""
    print(f"[Deploy] Connecting to {device_url} ...")
    client = EdgeRuntimeClient(device_url)
    try:
        await client.wait_ready()
        payload = json.dumps({"backbone": backbone, "decoders": decoders})
        print(f"[Deploy] Sending Control(load) backbone={backbone} decoders={len(decoders)} ...")
        resp = await client.control("load", payload)
        print(f"[Deploy] Control(load) returned: {resp['status']}")
        return resp
    except Exception as e:
        print(f"[Deploy] ERROR: Control(load) failed: {e}")
        raise
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Open-loop Poisson sender
# ---------------------------------------------------------------------------

async def run_open_loop(
    device_url: str,
    tasks: List[str],
    data: Dict[str, Dict],
    target_rps_per_task: float,
    duration: float,
) -> Dict[str, List[float]]:
    """
    Fire Poisson-arrival requests for each task concurrently.
    Returns dict: task -> list of e2e latencies (seconds).
    All tasks share one gRPC channel to the same device.
    """
    client = EdgeRuntimeClient(device_url)
    await client.wait_ready()

    # Track per-task: sent timestamps, completion latencies
    sent: Dict[str, List[float]] = {t: [] for t in tasks}
    latencies: Dict[str, List[float]] = {t: [] for t in tasks}
    errors: Dict[str, int] = {t: 0 for t in tasks}

    req_timeout = max(30.0, duration * 2)

    async def _fire(task: str, req_id: int, t_send: float) -> None:
        d = data[task]
        try:
            resp = await asyncio.wait_for(client.infer({
                "req_id": req_id,
                "task":   task,
                "x":      d["x"],
                "mask":   d.get("mask"),
            }), timeout=req_timeout)
            lat = time.time() - t_send
            latencies[task].append(lat)
        except Exception:
            errors[task] += 1

    async def _task_sender(task: str, req_id_offset: int) -> None:
        deadline = time.time() + duration
        req_id = req_id_offset
        in_flight = []
        while time.time() < deadline:
            t_send = time.time()
            sent[task].append(t_send)
            in_flight.append(asyncio.create_task(_fire(task, req_id, t_send)))
            req_id += 1
            # Poisson inter-arrival
            gap = np.random.exponential(1.0 / target_rps_per_task)
            remaining = (t_send + gap) - time.time()
            if remaining > 0:
                await asyncio.sleep(remaining)
        # Wait for all in-flight to finish
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

    # Launch one sender coroutine per task, offset req_ids to avoid collisions
    await asyncio.gather(
        *[_task_sender(task, i * 1_000_000) for i, task in enumerate(tasks)],
        return_exceptions=True,
    )
    await client.close()

    for task in tasks:
        n_sent = len(sent[task])
        n_done = len(latencies[task])
        actual_rps = n_done / duration
        avg_lat = (sum(latencies[task]) / n_done * 1000) if n_done else 0.0
        print(f"  [{task}] target={target_rps_per_task:.1f} sent={n_sent} "
              f"completed={n_done} actual_rps={actual_rps:.2f} "
              f"avg_lat={avg_lat:.1f}ms errors={errors[task]}")

    return {"sent": sent, "latencies": latencies, "errors": errors}


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _save_run(result_dir: Path, task: str, target_rps: float,
              duration: float, sent: List[float], latencies: List[float],
              n_errors: int) -> Dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    n_sent = len(sent)
    n_completed = len(latencies)
    input_rps = n_sent / duration
    output_rps = n_completed / duration
    avg_lat_ms = (sum(latencies) / n_completed * 1000) if n_completed else 0.0
    p50 = float(np.percentile(latencies, 50)) * 1000 if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) * 1000 if latencies else 0.0
    p99 = float(np.percentile(latencies, 99)) * 1000 if latencies else 0.0

    # Per-request CSV
    with (result_dir / f"{task}_requests.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["req_idx", "send_time", "latency_ms"])
        for i, (t, lat) in enumerate(zip(sent, latencies)):
            w.writerow([i, f"{t:.6f}", f"{lat*1000:.3f}"])

    row = {
        "task":         task,
        "target_rps":   target_rps,
        "input_rps":    round(input_rps, 4),
        "output_rps":   round(output_rps, 4),
        "n_sent":       n_sent,
        "n_completed":  n_completed,
        "n_errors":     n_errors,
        "avg_lat_ms":   round(avg_lat_ms, 3),
        "p50_lat_ms":   round(p50, 3),
        "p95_lat_ms":   round(p95, 3),
        "p99_lat_ms":   round(p99, 3),
    }
    return row


def _write_summary(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fields = ["n_tasks", "backbone", "task", "target_rps", "input_rps", "output_rps",
              "n_sent", "n_completed", "n_errors",
              "avg_lat_ms", "p50_lat_ms", "p95_lat_ms", "p99_lat_ms"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _print_summary(rows: List[Dict]) -> None:
    print("\n" + "─" * 100)
    print(f"  {'n_tasks':>7}  {'task':<14}  {'target':>7}  {'in_rps':>7}  "
          f"{'out_rps':>7}  {'sent':>6}  {'done':>6}  {'err':>4}  "
          f"{'avg_ms':>7}  {'p95_ms':>7}  {'p99_ms':>7}")
    print("─" * 100)
    for r in rows:
        print(f"  {int(r['n_tasks']):>7}  {r['task']:<14}  "
              f"{float(r['target_rps']):>7.1f}  {float(r['input_rps']):>7.2f}  "
              f"{float(r['output_rps']):>7.2f}  {int(r['n_sent']):>6}  "
              f"{int(r['n_completed']):>6}  {int(r['n_errors']):>4}  "
              f"{float(r['avg_lat_ms']):>7.1f}  {float(r['p95_lat_ms']):>7.1f}  "
              f"{float(r['p99_lat_ms']):>7.1f}")
    print("─" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="motivation2/tsfm — Open-loop RPS overhead benchmark"
    )
    parser.add_argument("--device-url",  default=os.environ.get("DEVICE_URL",  "localhost:8000"),
                        help="gRPC device server address (host:port)")
    parser.add_argument("--backbone",    default=os.environ.get("BACKBONE",    "momentbase"))
    parser.add_argument("--n-tasks",     default=os.environ.get("N_TASKS",     "1"),
                        help="Comma-separated list of task counts to sweep, e.g. 1,2,4")
    parser.add_argument("--rps-sweep",   default=os.environ.get("RPS_SWEEP",   "2,5,10,20"),
                        help="Comma-separated target RPS per task to sweep")
    parser.add_argument("--duration",    type=float, default=float(os.environ.get("DURATION", "30")),
                        help="Duration of each RPS point (seconds)")
    parser.add_argument("--exp-dir",     default=os.environ.get("EXP_DIR",
                        "experiments/motivation2/tsfm/results"))
    parser.add_argument("--skip-load",   action="store_true",
                        default=os.environ.get("SKIP_LOAD", "").lower() in ("1", "true"),
                        help="Skip Control(load) — use if model already loaded on device")
    args = parser.parse_args()

    task_counts = [int(x.strip()) for x in args.n_tasks.split(",")]
    rps_values  = [float(x.strip()) for x in args.rps_sweep.split(",")]
    result_root = (SERVING_DIR / args.exp_dir).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    summary_path = result_root / "summary.csv"

    print("\n" + "═" * 70)
    print("  motivation2/tsfm — Open-loop device overhead experiment")
    print("═" * 70)
    print(f"  Device URL  : {args.device_url}")
    print(f"  Backbone    : {args.backbone}")
    print(f"  N-tasks     : {task_counts}")
    print(f"  RPS sweep   : {rps_values}")
    print(f"  Duration    : {args.duration}s per point")
    print(f"  Skip load   : {args.skip_load}")
    print(f"  Results     : {result_root}")
    print("═" * 70)

    # Load existing summary for resume
    rows: List[Dict] = []
    existing: set = set()
    if summary_path.exists():
        with summary_path.open() as f:
            for row in csv.DictReader(f):
                existing.add((int(row["n_tasks"]), row["task"], float(row["target_rps"])))
                rows.append(row)
        print(f"[INFO] Resuming — {len(existing)} points already done")

    async def run_experiment(n_tasks: int, active_tasks: List[str], data: Dict,
                             rps_values: List[float]) -> List[Dict]:
        """Deploy once then sweep RPS — all in one event loop to avoid races."""
        exp_rows: List[Dict] = []

        if not args.skip_load:
            decoders = [
                {"task": t, "type": TASK_TYPES[t], "path": f"{t}_{args.backbone}_mlp"}
                for t in active_tasks
            ]
            print(f"[Deploy] Loading {args.backbone} + {len(decoders)} decoder(s) on {args.device_url} ...")
            resp = await deploy_backbone_async(args.device_url, args.backbone, decoders)
            if "error" in resp.get("status", "").lower():
                raise RuntimeError(f"Model load failed: {resp}")
            print(f"[Deploy] Load complete: {resp['status']}")
            await asyncio.sleep(1)

        for target_rps in rps_values:
            all_done = all((n_tasks, t, target_rps) in existing for t in active_tasks)
            if all_done:
                print(f"[SKIP] n_tasks={n_tasks} rps={target_rps} already done.")
                continue

            run_dir = result_root / f"{n_tasks}_tasks" / f"rps_{target_rps:.1f}"
            print(f"\n{'─' * 70}")
            print(f"  n_tasks={n_tasks}  target_rps={target_rps}  tasks={active_tasks}")

            result = await run_open_loop(
                device_url=args.device_url,
                tasks=active_tasks,
                data=data,
                target_rps_per_task=target_rps,
                duration=args.duration,
            )

            for task in active_tasks:
                if (n_tasks, task, target_rps) in existing:
                    continue
                row = _save_run(
                    run_dir, task, target_rps, args.duration,
                    result["sent"][task],
                    result["latencies"][task],
                    result["errors"][task],
                )
                row["n_tasks"]  = n_tasks
                row["backbone"] = args.backbone
                exp_rows.append(row)
                existing.add((n_tasks, task, target_rps))

            _write_summary(summary_path, rows + exp_rows)
            await asyncio.sleep(1)

        return exp_rows

    for n_tasks in task_counts:
        active_tasks = TASK_ORDER[:n_tasks]
        print(f"\n[INFO] Loading data for {n_tasks} task(s): {active_tasks}")
        data = build_data(active_tasks)
        new_rows = asyncio.run(run_experiment(n_tasks, active_tasks, data, rps_values))
        rows.extend(new_rows)

    print(f"\n[INFO] Done. Summary: {summary_path}")
    _print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
