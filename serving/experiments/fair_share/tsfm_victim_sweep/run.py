#!/usr/bin/env python3
"""fair_share/tsfm_victim_sweep — Single-phase victim-RPS sweep.

Both victim and aggressor send Poisson open-loop traffic at constant RPS for
DURATION seconds. Aggressor RPS is fixed; victim RPS is set per invocation
(the outer shell script sweeps it across multiple values).

Records per-request (send_time_s, latency_ms, server breakdown) for both
tasks and writes timeseries.csv, latencies.csv, task_results.csv, meta.json.
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

SERVING_DIR = Path(__file__).resolve().parents[3]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import numpy as np
from torch.utils.data import DataLoader

from site_manager.grpc_client import EdgeRuntimeClient, encode_infer_request
from site_manager.config import DATASET_DIR as _DATASET_DIR


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


async def deploy_backbone_async(device_url: str, backbone: str,
                                decoders: List[Dict]) -> dict:
    print(f"[Deploy] Connecting to {device_url} ...")
    client = EdgeRuntimeClient(device_url)
    try:
        await client.wait_ready()
        payload = json.dumps({"backbone": backbone, "decoders": decoders})
        print(f"[Deploy] Sending Control(load) backbone={backbone} decoders={len(decoders)} ...")
        resp = await client.control("load", payload)
        print(f"[Deploy] Control(load) returned: {resp['status']}")
        return resp
    finally:
        await client.close()


def generate_trace(rates: Dict[str, float], duration: float,
                   seed: int = 42) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    trace: Dict[str, List[float]] = {}
    for task, rps in rates.items():
        sends: List[float] = []
        if rps > 0:
            t = 0.0
            while t < duration:
                sends.append(t)
                t += rng.exponential(1.0 / rps)
        trace[task] = sends
    return trace


Record = Tuple[float, float, float, float, float, float, float, float, int]


async def run_constant(
    task_urls: Dict[str, str],
    trace: Dict[str, List[float]],
    data: Dict[str, Dict],
    req_timeout: float = 60.0,
) -> Dict[str, List[Record]]:
    clients: Dict[str, EdgeRuntimeClient] = {
        t: EdgeRuntimeClient(url) for t, url in task_urls.items()
    }
    for t, c in clients.items():
        print(f"[Run] Waiting for {t} server ({task_urls[t]}) ...")
        await c.wait_ready()

    task_proto = {
        task: encode_infer_request(task=task, x=data[task]["x"], mask=data[task].get("mask"))
        for task in trace
    }
    task_stub = {task: clients[task]._stub for task in trace}
    records: Dict[str, List[Record]] = {t: [] for t in trace}

    async def _fire(task: str, req_id: int, t_start: float) -> None:
        proto = task_proto[task]
        proto.req_id = req_id
        try:
            t_send_abs = time.time()
            response = await asyncio.wait_for(task_stub[task].Infer(proto), timeout=req_timeout)
            t_done_abs = time.time()
            if response.status and response.status != "ok":
                return
            client_lat_ms     = (t_done_abs - t_send_abs) * 1000
            server_exec_ms    = (response.end_time_ns - response.start_time_ns) / 1e6
            server_proc_ms    = response.proc_time_ns / 1e6
            server_swap_ms    = response.swap_time_ns / 1e6
            server_decoder_ms = response.decoder_time_ns / 1e6
            queue_wait_plus_rpc_ms = max(0.0, (response.start_time_ns / 1e9 - t_send_abs) * 1000)
            records[task].append((
                t_send_abs - t_start,
                client_lat_ms,
                server_exec_ms,
                server_proc_ms,
                server_swap_ms,
                server_decoder_ms,
                queue_wait_plus_rpc_ms,
                0.0,
                response.start_time_ns,
            ))
        except Exception:
            pass

    async def _sender(task: str, send_times: List[float],
                      req_id_offset: int, t_start: float) -> None:
        in_flight = []
        for req_id, rel_t in enumerate(send_times):
            target = t_start + rel_t
            wait   = target - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            in_flight.append(asyncio.create_task(
                _fire(task, req_id_offset + req_id, t_start)
            ))
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

    t_start = time.time()
    senders = [
        _sender(t, trace[t], i * 1_000_000, t_start)
        for i, t in enumerate(trace)
    ]
    await asyncio.gather(*senders, return_exceptions=True)
    for c in clients.values():
        await c.close()
    return records


def save_records(records: Dict[str, List[Record]], out_dir: Path,
                 duration: float, victim_rps: float, aggressor_rps: float,
                 scheduler_policy: str, warmup_secs: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for task, recs in records.items():
        path = out_dir / f"{task}_timeseries.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["task", "send_time_s", "latency_ms"])
            for rec in recs:
                w.writerow([task, f"{rec[0]:.4f}", f"{rec[1]:.3f}"])
        print(f"[Save] {path}  ({len(recs)} records)")

    with (out_dir / "latencies.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "task", "condition", "elapsed_sec", "latency_ms",
            "server_exec_ms", "server_proc_ms", "server_swap_ms",
            "server_decoder_ms", "queue_wait_plus_rpc_ms",
            "client_pre_rpc_ms", "non_server_exec_overhead_ms", "server_start_ns",
        ])
        for task, recs in records.items():
            for rec in recs:
                rel_t, lat, exec_ms, proc_ms, swap_ms, dec_ms, queue_ms, pre_rpc_ms, start_ns = rec
                w.writerow([
                    task, scheduler_policy,
                    round(rel_t, 4), round(lat, 4),
                    round(exec_ms, 4), round(proc_ms, 4), round(swap_ms, 4),
                    round(dec_ms, 4), round(queue_ms, 4), round(pre_rpc_ms, 4),
                    round(max(0.0, lat - exec_ms), 4),
                    start_ns,
                ])

    with (out_dir / "task_results.csv").open("w", newline="") as f:
        fields = [
            "task", "condition", "target_rps", "n_requests", "throughput_rps",
            "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
            "avg_server_exec_ms", "avg_server_proc_ms", "avg_server_swap_ms",
            "avg_server_decoder_ms", "avg_queue_wait_plus_rpc_ms",
            "avg_client_pre_rpc_ms", "avg_non_server_exec_overhead_ms",
        ]
        dw = csv.DictWriter(f, fieldnames=fields)
        dw.writeheader()
        target_rps_map = {t: (victim_rps if i == 0 else aggressor_rps)
                          for i, t in enumerate(records.keys())}
        effective_duration = max(duration - warmup_secs, 1.0)
        for task, recs in records.items():
            trimmed = [rec for rec in recs if rec[0] > warmup_secs]
            lats = [rec[1] for rec in trimmed]
            n = len(lats)
            if n == 0:
                continue
            dw.writerow({
                "task":           task,
                "condition":      scheduler_policy,
                "target_rps":     target_rps_map[task],
                "n_requests":     n,
                "throughput_rps": round(n / effective_duration, 4),
                "avg_latency_ms": round(float(np.mean(lats)), 3),
                "p50_latency_ms": round(float(np.percentile(lats, 50)), 3),
                "p95_latency_ms": round(float(np.percentile(lats, 95)), 3),
                "p99_latency_ms": round(float(np.percentile(lats, 99)), 3),
                "avg_server_exec_ms":              round(float(np.mean([rec[2] for rec in trimmed])), 3),
                "avg_server_proc_ms":              round(float(np.mean([rec[3] for rec in trimmed])), 3),
                "avg_server_swap_ms":              round(float(np.mean([rec[4] for rec in trimmed])), 3),
                "avg_server_decoder_ms":           round(float(np.mean([rec[5] for rec in trimmed])), 3),
                "avg_queue_wait_plus_rpc_ms":      round(float(np.mean([rec[6] for rec in trimmed])), 3),
                "avg_client_pre_rpc_ms":           round(float(np.mean([rec[7] for rec in trimmed])), 3),
                "avg_non_server_exec_overhead_ms": round(float(np.mean([max(0.0, rec[1] - rec[2]) for rec in trimmed])), 3),
            })

    with (out_dir / "meta.json").open("w") as f:
        json.dump({
            "duration_s":     duration,
            "victim_rps":     victim_rps,
            "aggressor_rps":  aggressor_rps,
            "scheduler":      scheduler_policy,
            "warmup_secs":    warmup_secs,
        }, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-url",       default="localhost:8000")
    parser.add_argument("--victim-url",       default=None)
    parser.add_argument("--aggressor-url",    default=None)
    parser.add_argument("--backbone",         default="momentlarge")
    parser.add_argument("--victim-task",      default="ecgclass")
    parser.add_argument("--aggressor-task",   default="gestureclass")
    parser.add_argument("--victim-rps",       type=float, required=True)
    parser.add_argument("--aggressor-rps",    type=float, required=True)
    parser.add_argument("--duration",         type=float, default=10.0)
    parser.add_argument("--warmup-secs",      type=float, default=2.0)
    parser.add_argument("--scheduler-policy", default="fifo")
    parser.add_argument("--exp-dir",          default=os.environ.get(
                        "EXP_DIR", "experiments/fair_share/tsfm_victim_sweep/results/run"))
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    victim_url    = args.victim_url    or args.device_url
    aggressor_url = args.aggressor_url or args.device_url
    task_urls = {
        args.victim_task:    victim_url,
        args.aggressor_task: aggressor_url,
    }

    print("=" * 65)
    print(f"  fair_share/tsfm_victim_sweep — single-phase run")
    print(f"  Backbone   : {args.backbone}")
    print(f"  Victim     : {args.victim_task} @ {args.victim_rps} rps")
    print(f"  Aggressor  : {args.aggressor_task} @ {args.aggressor_rps} rps")
    print(f"  Duration   : {args.duration}s   (warmup {args.warmup_secs}s)")
    print(f"  Scheduler  : {args.scheduler_policy}")
    print("=" * 65)

    tasks = [args.victim_task, args.aggressor_task]
    print(f"\n[INFO] Loading data for: {tasks}")
    data = build_data(tasks)

    if victim_url == aggressor_url:
        decoders = [
            {"task": t, "type": TASK_TYPES[t], "path": f"{t}_{args.backbone}_mlp"}
            for t in tasks
        ]
        resp = asyncio.run(deploy_backbone_async(victim_url, args.backbone, decoders))
        if "error" in resp.get("status", "").lower():
            print(f"[Error] Deploy failed: {resp}")
            return 1
    else:
        for task, url in task_urls.items():
            decoder = [{"task": task, "type": TASK_TYPES[task],
                        "path": f"{task}_{args.backbone}_mlp"}]
            resp = asyncio.run(deploy_backbone_async(url, args.backbone, decoder))
            if "error" in resp.get("status", "").lower():
                print(f"[Error] Deploy failed for {task} on {url}: {resp}")
                return 1

    asyncio.run(asyncio.sleep(1))

    rates = {
        args.victim_task:    args.victim_rps,
        args.aggressor_task: args.aggressor_rps,
    }
    trace = generate_trace(rates, args.duration, seed=args.seed)
    print(f"[Trace] {{ {', '.join(f'{t}: {len(s)}' for t, s in trace.items())} }} requests over {args.duration}s")

    print(f"\n[Run] Starting ({args.duration:.0f}s) ...")
    print(f"  {args.victim_task}    → {victim_url}")
    print(f"  {args.aggressor_task} → {aggressor_url}")
    req_timeout = max(60.0, args.duration * 2)
    records = asyncio.run(run_constant(
        task_urls, trace, data, req_timeout=req_timeout,
    ))

    out_dir = (SERVING_DIR / args.exp_dir).resolve()
    save_records(records, out_dir,
                 duration=args.duration,
                 victim_rps=args.victim_rps,
                 aggressor_rps=args.aggressor_rps,
                 scheduler_policy=args.scheduler_policy,
                 warmup_secs=args.warmup_secs)

    for task, recs in records.items():
        print(f"  [{task}] {len(recs)} requests recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
