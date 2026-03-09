#!/usr/bin/env python3
"""Motivation Experiment #2 — Memory vs. Throughput: Task Sharing vs. Deploy Sharing.

Simple closed-loop benchmark. No orchestrator or site_manager involved.

Strategies
----------
task_sharing   : 10 chronosbase instances across 3 GPUs (3 + 3 + 4).
                 Each task gets its own dedicated model instance on its own port.
                 Tasks never share a backbone — isolated per-task serving.

deploy_sharing : 1 chronosbase per GPU (3 total), each loaded with all 10 task decoders.
                 Tasks round-robin across 3 shared backbone instances.

Closed-loop inference
---------------------
Each task runs its own thread. The thread sends one request, waits for the response,
then immediately sends the next — tight closed loop for `duration` seconds.
Throughput per task = completed requests / duration.
Total throughput = sum across all tasks.

GPU memory
----------
Extracted from the gRPC load response (logger_summary gpu peak).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import threading
import ast
import re
from pathlib import Path
from typing import Dict, List

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import numpy as np
from torch.utils.data import DataLoader

from site_manager.deployment_handler import shutdown_devices
from site_manager.grpc_client import EdgeRuntimeClient

# ---------------------------------------------------------------------------
# Hardware layout — single GPU experiment
# ---------------------------------------------------------------------------

DEVICE_HOST = "10.100.31.64"
CUDA_DEVICE = "cuda:0"
BASE_PORT   = 8000   # task_sharing: ports 8000, 8001, ..., 800N-1
                     # deploy_sharing: port 8000 only

# ---------------------------------------------------------------------------
# Task library
# ---------------------------------------------------------------------------

TASK_ORDER: List[str] = [
    # (1, 1, 512) — 7 tasks
    "ecgclass",
    "gestureclass",
    "eclfore",
    "etth1fore",
    "exchangefore",
    "trafficfore",
    "weatherfore",
    # (1, 3, 512) — 3 tasks
    "sysbp",
    "diasbp",
    "heartrate",
]

TASK_META: Dict[str, Dict] = {
    "ecgclass":    {"type": "classification"},
    "gestureclass":{"type": "classification"},
    "sysbp":       {"type": "regression"},
    "diasbp":      {"type": "regression"},
    "heartrate":   {"type": "regression"},
    "eclfore":     {"type": "forecasting"},
    "etth1fore":   {"type": "forecasting"},
    "exchangefore":{"type": "forecasting"},
    "trafficfore": {"type": "forecasting"},
    "weatherfore": {"type": "forecasting"},
}

BACKBONE = "chronosbase"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decoder_spec(task: str) -> Dict:
    return {
        "task": task,
        "type": TASK_META[task]["type"],
        "path": f"{task}_{BACKBONE}_mlp",
    }


def extract_gpu_peak_mb(logger_summary: str) -> float:
    try:
        summary = ast.literal_eval(logger_summary)
        peaks = [v["gpu peak"] for v in summary.values() if "gpu peak" in v]
        return max(peaks) if peaks else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Deployment helpers — thin wrappers around deployment_handler
# ---------------------------------------------------------------------------

def make_spec(host: str, port: int, cuda: str, tasks: List[str]) -> dict:
    """Build a deployment spec for _deploy_one_with_retry()."""
    return {
        "device": f"{host}:{port}",
        "backbone": BACKBONE,
        "cuda": cuda,
        "decoders": [decoder_spec(t) for t in tasks],
    }


async def _deploy_one_with_retry(spec: dict, retries: int = 5, retry_delay: float = 5.0) -> dict:
    """Deploy one spec, retrying the load RPC if it fails (server may not be ready yet)."""
    from site_manager.deployment_handler import _parse_url, _ssh_start_server, _send_control
    from site_manager.config import timeseries_env, username

    ssh_host, grpc_url, grpc_port = _parse_url(spec["device"])
    cuda_device = spec.get("cuda", None)
    max_batch_size = int(spec.get("server_max_batch_size", 32))
    max_batch_wait_ms = float(spec.get("server_max_batch_wait_ms", 10.0))
    server_cmd = (
        f"python -u device/main.py --port {grpc_port} "
        f"--max-batch-size {max_batch_size} "
        f"--max-batch-wait-ms {max_batch_wait_ms} "
    )
    if cuda_device:
        server_cmd += f"--cuda {cuda_device} "
    cuda_suffix = spec.get("cuda", "").replace(":", "")
    log_suffix = spec.get("log_suffix", "")
    suffix = f"_{log_suffix}" if log_suffix else ""
    log_path = f"./device/logs/{ssh_host}_{cuda_suffix}_port{grpc_port}_{spec['backbone']}{suffix}.log"

    await _ssh_start_server(ssh_host, username, timeseries_env, server_cmd, log_path)

    config_payload = {"backbone": spec["backbone"], "decoders": spec["decoders"]}
    payload_json = json.dumps(config_payload)

    for attempt in range(retries):
        result = await _send_control(grpc_url, "load", payload_json)
        if isinstance(result, dict) and result.get("status", "").startswith("loaded_"):
            return {"result": result, "log_path": log_path}
        wait = retry_delay * (attempt + 1)
        print(f"[Deploy] {grpc_url} load attempt {attempt+1}/{retries} failed, retrying in {wait:.0f}s...")
        await asyncio.sleep(wait)

    raise RuntimeError(f"Failed to load model on {grpc_url} after {retries} attempts")


async def deploy_and_get_memory(specs: List[dict]) -> float:
    """Deploy all specs with retry and sum gpu peak MB across all results."""
    results = await asyncio.gather(*[_deploy_one_with_retry(s) for s in specs], return_exceptions=True)
    total = 0.0
    log_paths: List[str] = []
    for spec, r in zip(specs, results):
        if isinstance(r, Exception):
            raise RuntimeError(f"Deployment failed for {spec['device']}: {r}")
        if isinstance(r, dict):
            if r.get("log_path"):
                log_paths.append(r["log_path"])
            result = r.get("result", {})
            if "logger_summary" in result:
                total += extract_gpu_peak_mb(result["logger_summary"])
    return total, log_paths


async def cleanup(specs: List[dict]):
    """Shut down all device servers from the given specs."""
    await shutdown_devices(specs)
    await asyncio.sleep(15)  # wait for ports to be released


_BATCH_RE = re.compile(r"Prepared batch_size=(\d+)\s+req_ids=.*tasks=(\[.*\])")


def parse_batch_stats(log_path: str) -> Dict[str, float]:
    """Parse batch size / mixed-task stats from one device log file."""
    batch_count = 0
    batch_size_sum = 0
    mixed_batch_count = 0

    try:
        with open(log_path, "r") as f:
            for line in f:
                m = _BATCH_RE.search(line)
                if not m:
                    continue
                bs = int(m.group(1))
                task_list_text = m.group(2)
                try:
                    tasks = ast.literal_eval(task_list_text)
                except Exception:
                    tasks = []
                batch_count += 1
                batch_size_sum += bs
                if len(set(tasks)) > 1:
                    mixed_batch_count += 1
    except FileNotFoundError:
        pass

    avg_batch_size = (batch_size_sum / batch_count) if batch_count else 0.0
    mixed_frac = (mixed_batch_count / batch_count) if batch_count else 0.0
    return {
        "batch_count": batch_count,
        "avg_batch_size": avg_batch_size,
        "mixed_batch_fraction": mixed_frac,
    }


def aggregate_batch_stats(log_paths: List[str]) -> Dict[str, float]:
    """Aggregate batch stats across one or more servers."""
    total_batches = 0
    total_batch_items = 0.0
    total_mixed_batches = 0.0
    for p in log_paths:
        s = parse_batch_stats(p)
        n = int(s["batch_count"])
        total_batches += n
        total_batch_items += s["avg_batch_size"] * n
        total_mixed_batches += s["mixed_batch_fraction"] * n
    return {
        "batch_count": total_batches,
        "avg_batch_size": (total_batch_items / total_batches) if total_batches else 0.0,
        "mixed_batch_fraction": (total_mixed_batches / total_batches) if total_batches else 0.0,
    }


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def build_data() -> Dict[str, Dict]:
    """Return {task: one_batch_dict} using the same loaders as runtime_executor."""
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
        "gestureclass": DataLoader(UWaveGestureLibraryALLDataset({"dataset_path": f"{d}/UWaveGestureLibraryAll"}, {"task_type": "classification"}, "test"), **cfg),
        "sysbp":        DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data"}, {"task_type": "regression", "label": "sysbp"}, "test"), **cfg),
        "diasbp":       DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data"}, {"task_type": "regression", "label": "diasbp"}, "test"), **cfg),
        "heartrate":    DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data"}, {"task_type": "regression", "label": "hr"}, "test"), **cfg),
        "eclfore":      DataLoader(ECLDataset({"dataset_path": f"{d}/ElectricityLoad-data"}, {"task_type": "forecasting"}, "test"), **cfg),
        "etth1fore":    DataLoader(ETTh1Dataset({"dataset_path": f"{d}/ETTh1"}, {"task_type": "forecasting"}, "test"), **cfg),
        "exchangefore": DataLoader(ExchangeDataset({"dataset_path": f"{d}/Exchange"}, {"task_type": "forecasting"}, "test"), **cfg),
        "trafficfore":  DataLoader(TrafficDataset({"dataset_path": f"{d}/Traffic"}, {"task_type": "forecasting"}, "test"), **cfg),
        "weatherfore":  DataLoader(WeatherDataset({"dataset_path": f"{d}/Weather"}, {"task_type": "forecasting"}, "test"), **cfg),
    }
    return {task: next(iter(loader)) for task, loader in loaders.items()}


# ---------------------------------------------------------------------------
# Open-loop task worker — Poisson arrivals at fixed rate per task
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Benchmark mode: open_loop
# ---------------------------------------------------------------------------

def task_worker_open_loop(
    task: str,
    device_url: str,
    batch: Dict,
    duration: float,
    results: List,
    req_id_start: int,
    target_rps: float = 2.0,
):
    """
    Open-loop sender: fires requests at Poisson(target_rps) inter-arrivals.
    Sender does NOT wait for responses — all requests are in-flight concurrently.

    deploy_sharing advantage: batcher sees N*target_rps total req/s across tasks
    → cross-task batches form → backbone amortized over N tasks.
    task_sharing: each of N servers sees only target_rps → batch=1 (starved).
    """

    async def _run():
        client = EdgeRuntimeClient(device_url)
        await client.wait_ready()

        x = batch["x"].numpy().astype(np.float32)
        mask = batch["mask"].numpy().astype(np.float32) if "mask" in batch else None

        latencies: List[float] = []
        in_flight: List[asyncio.Task] = []
        deadline = time.time() + duration
        req_id = req_id_start
        inter_arrival = 1.0 / target_rps

        async def _fire(rid, send_time):
            try:
                await client.infer({"req_id": rid, "task": task, "x": x, "mask": mask})
                latencies.append((time.time() - send_time) * 1000)
            except Exception as e:
                print(f"[Worker:{task}] req {rid} error: {e}")

        while time.time() < deadline:
            t0 = time.time()
            in_flight.append(asyncio.create_task(_fire(req_id, t0)))
            req_id += 1
            sleep_s = np.random.exponential(inter_arrival)
            next_t = t0 + sleep_s
            remaining = next_t - time.time()
            if remaining > 0:
                await asyncio.sleep(remaining)

        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

        await client.close()
        return latencies

    latencies = asyncio.run(_run())
    n = len(latencies)
    avg_lat = sum(latencies) / n if n else 0.0
    actual_rps = n / duration
    results.append({
        "task": task,
        "device_url": device_url,
        "n_requests": n,
        "throughput_rps": actual_rps,
        "avg_latency_ms": avg_lat,
    })
    print(f"[Worker:{task}] done — {n} reqs @ target={target_rps:.1f} actual={actual_rps:.2f} req/s, "
          f"avg_lat={avg_lat:.1f} ms")


def run_open_loop(
    task_device_map: Dict[str, str],
    data: Dict[str, Dict],
    duration: float,
    target_rps_per_task: float,
) -> List[Dict]:
    """
    Launch one thread per task, each firing at Poisson(target_rps_per_task).
    """
    results = []
    threads = []
    for i, (task, url) in enumerate(task_device_map.items()):
        t = threading.Thread(
            target=task_worker_open_loop,
            args=(task, url, data[task], duration, results, i * 10000),
            kwargs={"target_rps": target_rps_per_task},
            daemon=True,
        )
        threads.append(t)

    print(f"\n[Benchmark:open_loop] {len(threads)} tasks @ {target_rps_per_task:.1f} req/s each "
          f"({target_rps_per_task * len(threads):.1f} req/s total) for {duration}s...")
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=duration + 120)

    return results


# ---------------------------------------------------------------------------
# Benchmark mode: closed_loop (fixed concurrency)
# ---------------------------------------------------------------------------

def task_worker_closed_loop(
    task: str,
    device_url: str,
    batch: Dict,
    duration: float,
    results: List,
    req_id_start: int,
    concurrency: int = 8,
):
    """
    Closed-loop sender with fixed concurrency per task.
    Keeps exactly `concurrency` requests in-flight at all times.

    deploy_sharing advantage: all N*concurrency workers share ONE server queue
    → batcher sees mixed-task requests simultaneously → cross-task batches form
    → backbone forward pass amortized over N tasks → higher total throughput.

    task_sharing: each server has only `concurrency` workers for ONE task
    → GPU time-sliced across N processes → lower throughput per process.
    """

    async def _run():
        client = EdgeRuntimeClient(device_url)
        await client.wait_ready()

        x = batch["x"].numpy().astype(np.float32)
        mask = batch["mask"].numpy().astype(np.float32) if "mask" in batch else None

        latencies: List[float] = []
        semaphore = asyncio.Semaphore(concurrency)
        stop_event = asyncio.Event()
        req_id = req_id_start
        deadline = time.time() + duration

        async def _one_worker():
            nonlocal req_id
            while not stop_event.is_set():
                async with semaphore:
                    if stop_event.is_set():
                        break
                    rid = req_id
                    req_id += 1
                    t0 = time.time()
                    try:
                        await client.infer({"req_id": rid, "task": task, "x": x, "mask": mask})
                        latencies.append((time.time() - t0) * 1000)
                    except Exception as e:
                        print(f"[Worker:{task}] req {rid} error: {e}")

        async def _timer():
            remaining = deadline - time.time()
            if remaining > 0:
                await asyncio.sleep(remaining)
            stop_event.set()

        workers = [asyncio.create_task(_one_worker()) for _ in range(concurrency)]
        timer = asyncio.create_task(_timer())
        await asyncio.gather(timer, *workers, return_exceptions=True)

        await client.close()
        return latencies

    latencies = asyncio.run(_run())
    n = len(latencies)
    avg_lat = sum(latencies) / n if n else 0.0
    actual_rps = n / duration
    results.append({
        "task": task,
        "device_url": device_url,
        "n_requests": n,
        "throughput_rps": actual_rps,
        "avg_latency_ms": avg_lat,
    })
    print(f"[Worker:{task}] done — {n} reqs, concurrency={concurrency}, "
          f"actual={actual_rps:.2f} req/s, avg_lat={avg_lat:.1f} ms")


def run_closed_loop(
    task_device_map: Dict[str, str],
    data: Dict[str, Dict],
    duration: float,
    concurrency_per_task: int,
) -> List[Dict]:
    """
    Launch one thread per task. Each thread keeps `concurrency_per_task`
    requests in-flight at all times (closed-loop, fixed concurrency).

    deploy_sharing: all N tasks share one server → N*concurrency concurrent
    requests in the queue → cross-task batches form reliably.
    task_sharing: each of N servers sees only `concurrency` requests → no
    cross-task batching possible.
    """
    results = []
    threads = []
    for i, (task, url) in enumerate(task_device_map.items()):
        t = threading.Thread(
            target=task_worker_closed_loop,
            args=(task, url, data[task], duration, results, i * 10000),
            kwargs={"concurrency": concurrency_per_task},
            daemon=True,
        )
        threads.append(t)

    total_concurrency = concurrency_per_task * len(threads)
    print(f"\n[Benchmark:closed_loop] {len(threads)} tasks × {concurrency_per_task} workers "
          f"= {total_concurrency} concurrent requests, for {duration}s...")
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=duration + 120)

    return results


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

async def run_task_sharing(
    active_tasks: List[str],
    duration: float,
    result_dir: Path,
    data: Dict,
    benchmark_mode: str = "open_loop",
    target_rps_per_task: float = 8.0,
    concurrency_per_task: int = 8,
    server_max_batch_size: int = 32,
    server_max_batch_wait_ms: float = 10.0,
) -> Dict:
    """
    task_sharing: N separate chronosbase processes on one GPU.
    Each task gets its own server (BASE_PORT+i) and its own backbone copy.
    All processes share the same CUDA_DEVICE → GPU time-sliced across N processes.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    n_tasks = len(active_tasks)

    task_assignments = [
        (task, DEVICE_HOST, BASE_PORT + i, CUDA_DEVICE)
        for i, task in enumerate(active_tasks)
    ]

    print(f"[task_sharing] {n_tasks} backbone instances on {CUDA_DEVICE}:")
    for task, host, port, cuda in task_assignments:
        print(f"  {task:15s} → {host}:{port}")

    specs = []
    for task, host, port, cuda in task_assignments:
        spec = make_spec(host, port, cuda, [task])
        spec["server_max_batch_size"] = server_max_batch_size
        spec["server_max_batch_wait_ms"] = server_max_batch_wait_ms
        spec["log_suffix"] = f"task_sharing_{n_tasks}tasks_{task}"
        specs.append(spec)
    gpu_mem_mb, log_paths = await deploy_and_get_memory(specs)
    task_device_map = {task: f"{host}:{port}" for task, host, port, _ in task_assignments}

    with (result_dir / "deployment_info.json").open("w") as f:
        json.dump({"strategy": "task_sharing", "n_tasks": n_tasks, "backbone": BACKBONE,
                   "benchmark_mode": benchmark_mode,
                   "assignments": [{"task": t, "url": f"{h}:{p}"} for t, h, p, _ in task_assignments],
                   "gpu_mem_mb": gpu_mem_mb,
                   "log_paths": log_paths}, f, indent=2)

    if benchmark_mode == "closed_loop":
        task_results = run_closed_loop(task_device_map, data, duration, concurrency_per_task)
    else:
        task_results = run_open_loop(task_device_map, data, duration, target_rps_per_task)

    fields = ["task", "device_url", "n_requests", "throughput_rps", "avg_latency_ms"]
    with (result_dir / "task_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(task_results)

    total_rps = sum(r["throughput_rps"] for r in task_results)
    avg_lat = sum(r["avg_latency_ms"] for r in task_results) / len(task_results) if task_results else 0.0
    batch_stats = aggregate_batch_stats(log_paths)
    await cleanup(specs)

    print(
        f"  => gpu_mem={gpu_mem_mb:.1f} MB  total_throughput={total_rps:.2f} req/s  "
        f"avg_lat={avg_lat:.1f} ms  avg_batch={batch_stats['avg_batch_size']:.2f}  "
        f"mixed_batch_frac={batch_stats['mixed_batch_fraction']:.3f}"
    )
    return {"strategy": "task_sharing", "n_tasks": n_tasks,
            "gpu_mem_mb": round(gpu_mem_mb, 3), "throughput_rps": round(total_rps, 4),
            "avg_latency_ms": round(avg_lat, 3),
            "avg_batch_size": round(batch_stats["avg_batch_size"], 4),
            "mixed_batch_fraction": round(batch_stats["mixed_batch_fraction"], 4),
            "batch_count": int(batch_stats["batch_count"])}


async def run_deploy_sharing(
    active_tasks: List[str],
    duration: float,
    result_dir: Path,
    data: Dict,
    benchmark_mode: str = "open_loop",
    target_rps_per_task: float = 8.0,
    concurrency_per_task: int = 8,
    server_max_batch_size: int = 32,
    server_max_batch_wait_ms: float = 10.0,
) -> Dict:
    """
    deploy_sharing: 1 chronosbase process on one GPU, N decoders loaded.
    All tasks share the single backbone → cross-task batching possible.
    GPU used exclusively by one process → no time-slicing overhead.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    n_tasks = len(active_tasks)

    specs = [make_spec(DEVICE_HOST, BASE_PORT, CUDA_DEVICE, active_tasks)]
    specs[0]["server_max_batch_size"] = server_max_batch_size
    specs[0]["server_max_batch_wait_ms"] = server_max_batch_wait_ms
    specs[0]["log_suffix"] = f"deploy_sharing_{n_tasks}tasks"

    print(f"[deploy_sharing] 1 backbone instance on {CUDA_DEVICE}, {n_tasks} decoders:")
    print(f"  tasks: {active_tasks}")

    gpu_mem_mb, log_paths = await deploy_and_get_memory(specs)
    task_device_map = {task: f"{DEVICE_HOST}:{BASE_PORT}" for task in active_tasks}

    with (result_dir / "deployment_info.json").open("w") as f:
        json.dump({"strategy": "deploy_sharing", "n_tasks": n_tasks, "backbone": BACKBONE,
                   "benchmark_mode": benchmark_mode,
                   "server": f"{DEVICE_HOST}:{BASE_PORT}", "tasks": active_tasks,
                   "log_paths": log_paths,
                   "gpu_mem_mb": gpu_mem_mb}, f, indent=2)

    if benchmark_mode == "closed_loop":
        task_results = run_closed_loop(task_device_map, data, duration, concurrency_per_task)
    else:
        task_results = run_open_loop(task_device_map, data, duration, target_rps_per_task)

    fields = ["task", "device_url", "n_requests", "throughput_rps", "avg_latency_ms"]
    with (result_dir / "task_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(task_results)

    total_rps = sum(r["throughput_rps"] for r in task_results)
    avg_lat = sum(r["avg_latency_ms"] for r in task_results) / len(task_results) if task_results else 0.0
    batch_stats = aggregate_batch_stats(log_paths)
    await cleanup(specs)

    print(
        f"  => gpu_mem={gpu_mem_mb:.1f} MB  total_throughput={total_rps:.2f} req/s  "
        f"avg_lat={avg_lat:.1f} ms  avg_batch={batch_stats['avg_batch_size']:.2f}  "
        f"mixed_batch_frac={batch_stats['mixed_batch_fraction']:.3f}"
    )
    return {"strategy": "deploy_sharing", "n_tasks": n_tasks,
            "gpu_mem_mb": round(gpu_mem_mb, 3), "throughput_rps": round(total_rps, 4),
            "avg_latency_ms": round(avg_lat, 3),
            "avg_batch_size": round(batch_stats["avg_batch_size"], 4),
            "mixed_batch_fraction": round(batch_stats["mixed_batch_fraction"], 4),
            "batch_count": int(batch_stats["batch_count"])}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STRATEGY_RUNNERS = {
    "task_sharing":   run_task_sharing,
    "deploy_sharing": run_deploy_sharing,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Motivation Experiment #2")
    parser.add_argument("--n-tasks",     default=os.environ.get("N_TASKS", "10,8,6,2,1"),
                        help="Comma-separated list of task counts to sweep, e.g. '10,8,6,2,1'")
    parser.add_argument("--duration",    type=float, default=float(os.environ.get("PHASE_DURATION", "60")))
    parser.add_argument("--exp-dir",     default=os.environ.get("EXP_DIR", "experiments/motivation2/results"))
    parser.add_argument("--strategies",  default="task_sharing,deploy_sharing")
    parser.add_argument("--benchmark-mode", default=os.environ.get("BENCHMARK_MODE", "closed_loop"),
                        choices=["open_loop", "closed_loop", "total_rps"],
                        help="open_loop: Poisson arrivals at --target-rps per task. "
                             "closed_loop: fixed --concurrency workers per task firing as fast as possible. "
                             "total_rps: Poisson arrivals at --total-rps / n_tasks per task (fixed total load).")
    parser.add_argument("--target-rps",  type=float, default=float(os.environ.get("TARGET_RPS", "8.0")),
                        help="[open_loop] Target arrival rate per task (req/s).")
    parser.add_argument("--total-rps",   type=float, default=float(os.environ.get("TOTAL_RPS", "48.0")),
                        help="[total_rps] Fixed total arrival rate across all tasks (req/s). "
                             "Per-task rate = total_rps / n_tasks.")
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", "8")),
                        help="[closed_loop] Number of concurrent workers per task.")
    parser.add_argument("--server-max-batch-size", type=int, default=int(os.environ.get("SERVER_MAX_BATCH_SIZE", "32")),
                        help="Device-side max batch size for the runtime server.")
    parser.add_argument("--server-max-batch-wait-ms", type=float, default=float(os.environ.get("SERVER_MAX_BATCH_WAIT_MS", "10")),
                        help="Device-side max batch formation wait in milliseconds.")
    args = parser.parse_args()

    task_counts = [int(x.strip()) for x in args.n_tasks.split(",")]

    result_root = (SERVING_DIR / args.exp_dir).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    summary_path = result_root / "summary.csv"

    strategies_to_run = [s.strip() for s in args.strategies.split(",")]

    print("\n" + "═" * 65)
    print(f"  Motivation Experiment #2 — {args.benchmark_mode} benchmark")
    print("═" * 65)
    print(f"  Backbone        : {BACKBONE}")
    print(f"  Device          : {DEVICE_HOST} {CUDA_DEVICE}")
    print(f"  task_sharing    : N chronosbase processes on {CUDA_DEVICE} (ports {BASE_PORT}..{BASE_PORT+9})")
    print(f"  deploy_sharing  : 1 chronosbase on {CUDA_DEVICE} port {BASE_PORT}, N decoders")
    print(f"  Strategies      : {strategies_to_run}")
    print(f"  N sweep         : {task_counts}")
    print(f"  Duration/run    : {args.duration}s")
    print(f"  Benchmark mode  : {args.benchmark_mode}")
    if args.benchmark_mode == "open_loop":
        print(f"  Target RPS/task : {args.target_rps} req/s (Poisson arrivals)")
    elif args.benchmark_mode == "total_rps":
        print(f"  Total RPS       : {args.total_rps} req/s fixed, split evenly across N tasks")
    else:
        print(f"  Concurrency/task: {args.concurrency} workers (fixed concurrency)")
    print(f"  Server batch cfg: max_batch_size={args.server_max_batch_size}, max_wait={args.server_max_batch_wait_ms} ms")
    print(f"  Results         : {result_root}")
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

    async def _sweep():
        for n_tasks in task_counts:
            active_tasks = TASK_ORDER[:n_tasks]
            for strategy in strategies_to_run:
                if strategy not in STRATEGY_RUNNERS:
                    print(f"[WARN] Unknown strategy '{strategy}', skipping.")
                    continue
                if (n_tasks, strategy) in existing:
                    print(f"[SKIP] n_tasks={n_tasks} strategy={strategy} already done.")
                    continue

                run_dir = result_root / strategy / f"{n_tasks}_tasks"
                print(f"\n{'─'*65}")
                print(f"  strategy={strategy}  n_tasks={n_tasks}  mode={args.benchmark_mode}")
                print(f"  tasks={active_tasks}")

                if args.benchmark_mode == "total_rps":
                    effective_rps_per_task = args.total_rps / n_tasks
                    effective_mode = "open_loop"
                    print(f"  total_rps={args.total_rps} / n_tasks={n_tasks} → {effective_rps_per_task:.2f} req/s per task")
                else:
                    effective_rps_per_task = args.target_rps
                    effective_mode = args.benchmark_mode

                row = await STRATEGY_RUNNERS[strategy](
                    active_tasks, args.duration, run_dir, data,
                    benchmark_mode=effective_mode,
                    target_rps_per_task=effective_rps_per_task,
                    concurrency_per_task=args.concurrency,
                    server_max_batch_size=args.server_max_batch_size,
                    server_max_batch_wait_ms=args.server_max_batch_wait_ms,
                )
                rows.append(row)
                existing.add((n_tasks, strategy))
                _write_summary(summary_path, rows)
                await asyncio.sleep(3)

    asyncio.run(_sweep())

    print(f"\n[INFO] Done. Summary: {summary_path}")
    _print_summary(rows)
    return 0


def _write_summary(path: Path, rows: List) -> None:
    if not rows:
        return
    fields = [
        "n_tasks",
        "strategy",
        "gpu_mem_mb",
        "throughput_rps",
        "avg_latency_ms",
        "avg_batch_size",
        "mixed_batch_fraction",
        "batch_count",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _print_summary(rows: List) -> None:
    print("\n" + "─" * 75)
    print(f"  {'n_tasks':>8}  {'strategy':<20}  {'gpu_mem_mb':>12}  {'throughput_rps':>14}  {'avg_lat_ms':>10}")
    print("─" * 75)
    for r in rows:
        print(f"  {int(r['n_tasks']):>8}  {r['strategy']:<20}  "
              f"{float(r['gpu_mem_mb']):>12.1f}  {float(r['throughput_rps']):>14.3f}  "
              f"{float(r['avg_latency_ms']):>10.1f}")
    print("─" * 75)


if __name__ == "__main__":
    raise SystemExit(main())
