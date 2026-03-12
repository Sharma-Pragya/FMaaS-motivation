#!/usr/bin/env python3
"""motivation1/run.py — Client for the motivation1 experiment.

Benchmarks task_sharing vs deploy_sharing in-process.
Each task thread keeps `concurrency` requests in-flight (closed-loop).
The server (server.py) handles batching, queuing, and GPU execution.

Usage:
    python run.py [--n-tasks 10,8,6,4,2,1] [--duration 60]
                  [--strategies task_sharing,deploy_sharing]
                  [--backbone chronosbase] [--cuda cuda:0]
                  [--exp-dir experiments/motivation1/results]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

SERVING_DIR = Path(__file__).resolve().parents[3]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import numpy as np
from torch.utils.data import DataLoader

from experiments.motivation1.tsfm.server import InferenceServer

# ---------------------------------------------------------------------------
# Task library
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

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def build_data() -> Dict[str, Dict]:
    from site_manager.config import DATASET_DIR as d
    from fmtk.datasetloaders.ecg5000 import ECG5000Dataset
    from fmtk.datasetloaders.uwavegesture import UWaveGestureLibraryALLDataset
    from fmtk.datasetloaders.ppg import PPGDataset
    from fmtk.datasetloaders.ecl import ECLDataset
    from fmtk.datasetloaders.etth1 import ETTh1Dataset
    from fmtk.datasetloaders.exchange import ExchangeDataset
    from fmtk.datasetloaders.traffic import TrafficDataset
    from fmtk.datasetloaders.weather import WeatherDataset

    cfg = {"batch_size": 1, "shuffle": False}
    loaders = {
        "ecgclass":     DataLoader(ECG5000Dataset({"dataset_path": f"{d}/ECG5000"}, {"task_type": "classification"}, "test"), **cfg),
        "gestureclass": DataLoader(UWaveGestureLibraryALLDataset({"dataset_path": f"{d}/UWaveGestureLibraryAll",'seq_len':512}, {"task_type": "classification"}, "test"), **cfg),
        "sysbp":        DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data",'seq_len':512,'num_channels': 1}, {"task_type": "regression", "label": "sysbp"}, "test"), **cfg),
        "diasbp":       DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data",'seq_len':512,'num_channels': 1}, {"task_type": "regression", "label": "diasbp"}, "test"), **cfg),
        "heartrate":    DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data",'seq_len':512,'num_channels': 1}, {"task_type": "regression", "label": "hr"}, "test"), **cfg),
        "eclfore":      DataLoader(ECLDataset({"dataset_path": f"{d}/ElectricityLoad-data"}, {"task_type": "forecasting",'seq_len':512}, "test"), **cfg),
        "etth1fore":    DataLoader(ETTh1Dataset({"dataset_path": f"{d}/ETTh1"}, {"task_type": "forecasting",'seq_len':512}, "test"), **cfg),
        "exchangefore": DataLoader(ExchangeDataset({"dataset_path": f"{d}/Exchange"}, {"task_type": "forecasting",'seq_len':512}, "test"), **cfg),
        "trafficfore":  DataLoader(TrafficDataset({"dataset_path": f"{d}/Traffic"}, {"task_type": "forecasting",'seq_len':512}, "test"), **cfg),
        "weatherfore":  DataLoader(WeatherDataset({"dataset_path": f"{d}/Weather"}, {"task_type": "forecasting"}, "test"), **cfg),
    }
    return {task: next(iter(loader)) for task, loader in loaders.items()}


# ---------------------------------------------------------------------------
# Client: closed-loop task worker
# ---------------------------------------------------------------------------

def task_worker_thread(
    task: str,
    server: InferenceServer,
    x: np.ndarray,
    mask: Optional[np.ndarray],
    duration: float,
    results: List[Dict],
):
    """
    Closed-loop worker in its own OS thread. Calls server.infer() directly —
    no asyncio needed. For deploy_sharing all N threads hit the barrier inside
    _SharedBackbone.infer() simultaneously → one GPU forward pass per round.
    """
    latencies: List[float] = []
    deadline = time.time() + duration

    while time.time() < deadline:
        t0 = time.perf_counter()
        try:
            server.infer(task, x, mask)
            latencies.append((time.perf_counter() - t0) * 1000)
        except Exception as e:
            print(f"[Client:{task}] error: {e}")

    n_done = len(latencies)
    avg_lat = sum(latencies) / n_done if n_done else 0.0
    actual_rps = n_done / duration
    results.append({
        "task": task,
        "n_requests": n_done,
        "throughput_rps": actual_rps,
        "avg_latency_ms": avg_lat,
    })
    print(f"[Client:{task}] {n_done} reqs, {actual_rps:.2f} req/s, avg_lat={avg_lat:.1f} ms")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    strategy: str,
    active_tasks: List[str],
    data: Dict,
    duration: float,
    backbone: str,
    device: str,
    result_dir: Path,
    decoder_dir: Optional[str] = None,
) -> Dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    n_tasks = len(active_tasks)

    server = InferenceServer(
        strategy=strategy,
        tasks=active_tasks,
        backbone=backbone,
        device=device,
        decoder_dir=decoder_dir,
    )
    server.start()

    task_inputs: Dict[str, tuple] = {}
    for task in active_tasks:
        batch = data[task]
        x = batch["x"].numpy().astype(np.float32)
        mask = batch["mask"].numpy().astype(np.float32) if "mask" in batch else None
        task_inputs[task] = (x, mask)

    results: List[Dict] = []
    print(f"\n[Client] Starting {n_tasks} tasks (1 thread each, barrier-batched) for {duration}s "
          f"(strategy={strategy})")

    threads = [
        threading.Thread(
            target=task_worker_thread,
            args=(task, server, *task_inputs[task], duration, results),
            daemon=True,
        )
        for task in active_tasks
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    batch_stats = server.batch_stats()
    avg_gpu_mem_mb = server.avg_inference_gpu_mem_mb()
    server.stop()

    total_rps  = sum(r["throughput_rps"] for r in results)
    avg_lat    = sum(r["avg_latency_ms"] for r in results) / len(results) if results else 0.0
    gpu_mem_mb = server.peak_gpu_mem_mb()
    model_load_mem_mb = server.model_load_mem_mb()

    fields = ["task", "n_requests", "throughput_rps", "avg_latency_ms"]
    with (result_dir / "task_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    summary = {
        "backbone": backbone,
        "strategy": strategy,
        "n_tasks": n_tasks,
        "model_load_mem_mb": round(model_load_mem_mb, 3),
        "peak_gpu_mem_mb": round(gpu_mem_mb, 3),
        "avg_gpu_mem_mb": round(avg_gpu_mem_mb, 3),
        "throughput_rps": round(total_rps, 4),
        "avg_latency_ms": round(avg_lat, 3),
        "avg_batch_size": round(batch_stats["avg_batch_size"], 4),
        "mixed_batch_fraction": round(batch_stats["mixed_batch_fraction"], 4),
        "batch_count": int(batch_stats["batch_count"]),
    }
    with (result_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Client] {strategy} n={n_tasks}: "
          f"load_mem={model_load_mem_mb:.1f} MB  peak_mem={gpu_mem_mb:.1f} MB  avg_mem={avg_gpu_mem_mb:.1f} MB  "
          f"total_rps={total_rps:.2f}  avg_lat={avg_lat:.1f} ms  "
          f"avg_batch={batch_stats['avg_batch_size']:.2f}  mixed={batch_stats['mixed_batch_fraction']:.3f}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Motivation Experiment #1")
    parser.add_argument("--n-tasks",              default=os.environ.get("N_TASKS", "10,8,6,4,2,1"))
    parser.add_argument("--duration",             type=float, default=float(os.environ.get("PHASE_DURATION", "60")))
    parser.add_argument("--strategies",           default="task_sharing,deploy_sharing")
    parser.add_argument("--backbone",             default=os.environ.get("BACKBONE", "chronoslarge"))
    parser.add_argument("--cuda",                 default=os.environ.get("CUDA_DEVICE", "cuda:0"))
    parser.add_argument("--decoder-dir",          default=os.environ.get("DECODER_DIR", None),
                        help="Path to finetuned decoder checkpoints. "
                             "Expects {decoder_dir}/{task}_{backbone}_mlp/decoder.pth. "
                             "Omit for backbone-only inference.")
    parser.add_argument("--exp-dir",              default=os.environ.get("EXP_DIR", "experiments/motivation1/results"))
    args = parser.parse_args()

    task_counts = [int(x.strip()) for x in args.n_tasks.split(",")]
    strategies  = [s.strip() for s in args.strategies.split(",")]
    result_root = (SERVING_DIR / args.exp_dir).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    summary_path = result_root / "summary.csv"

    print("\n" + "═" * 65)
    print("  Motivation Experiment #1 — Deploy Sharing vs Task Sharing")
    print("═" * 65)
    print(f"  Backbone          : {args.backbone}")
    print(f"  CUDA device       : {args.cuda}")
    print(f"  Decoder dir       : {args.decoder_dir or '(backbone-only)'}")
    print(f"  Strategies        : {strategies}")
    print(f"  N sweep           : {task_counts}")
    print(f"  Duration/run      : {args.duration}s")
    print(f"  Workers/task      : 1 thread (barrier-batched, N tasks per GPU call)")
    print(f"  Results           : {result_root}")
    print("═" * 65)

    rows: List[Dict] = []
    existing: set = set()
    if summary_path.exists():
        with summary_path.open() as f:
            for row in csv.DictReader(f):
                existing.add((int(row["n_tasks"]), row["strategy"]))
                rows.append(row)
        print(f"[INFO] Resuming — {len(existing)} cells already done")

    print("[INFO] Loading datasets...")
    data = build_data()
    print(f"[INFO] Loaded {len(data)} task datasets")

    for n_tasks in task_counts:
        active_tasks = TASK_ORDER[:n_tasks]
        for strategy in strategies:
            if (n_tasks, strategy) in existing:
                print(f"[SKIP] n_tasks={n_tasks} strategy={strategy} already done.")
                continue

            print(f"\n{'─'*65}")
            print(f"  strategy={strategy}  n_tasks={n_tasks}")
            print(f"  tasks={active_tasks}")

            run_dir = result_root / strategy / f"{n_tasks}_tasks"
            row = run_benchmark(
                strategy=strategy,
                active_tasks=active_tasks,
                data=data,
                duration=args.duration,
                backbone=args.backbone,
                device=args.cuda,
                result_dir=run_dir,
                decoder_dir=args.decoder_dir,
            )
            rows.append(row)
            existing.add((n_tasks, strategy))
            _write_summary(summary_path, rows)
            time.sleep(3)

    print(f"\n[INFO] Done. Summary: {summary_path}")
    _print_summary(rows)
    return 0


def _write_summary(path: Path, rows: List) -> None:
    if not rows:
        return
    fields = ["backbone", "n_tasks", "strategy", "model_load_mem_mb", "peak_gpu_mem_mb",
              "avg_gpu_mem_mb", "throughput_rps", "avg_latency_ms", "avg_batch_size",
              "mixed_batch_fraction", "batch_count"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _print_summary(rows: List) -> None:
    print("\n" + "─" * 105)
    print(f"  {'backbone':<14}  {'n_tasks':>8}  {'strategy':<20}  "
          f"{'load_mem_mb':>11}  {'peak_mem_mb':>11}  {'avg_mem_mb':>10}  "
          f"{'throughput_rps':>14}  {'avg_lat_ms':>10}")
    print("─" * 105)
    for r in rows:
        print(f"  {r.get('backbone', 'unknown'):<14}  {int(r['n_tasks']):>8}  {r['strategy']:<20}  "
              f"{float(r.get('model_load_mem_mb', 0)):>11.1f}  "
              f"{float(r['peak_gpu_mem_mb']):>11.1f}  {float(r['avg_gpu_mem_mb']):>10.1f}  "
              f"{float(r['throughput_rps']):>14.3f}  {float(r['avg_latency_ms']):>10.1f}")
    print("─" * 105)


if __name__ == "__main__":
    raise SystemExit(main())
