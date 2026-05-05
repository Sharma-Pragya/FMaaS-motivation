#!/usr/bin/env python3
"""noisy_neighbor/tsfm — Time-series interference experiment.

Shows the noisy-neighbor effect "in action" across N phases:
  Phase 1 (0 → p1_end):            both tasks at their phase-1 RPS
  Phase k (p_{k-1}_end → pk_end):  aggressor at phase-k RPS

Records per-request (send_time_s, latency_ms) for each task so we can plot
latency vs wall-clock time and see the victim degrade as aggressor ramps.

Supports both sharing (single device URL) and no-sharing (separate URLs per
task) configurations via --device-url / --victim-url / --aggressor-url.
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
# Deploy
# ---------------------------------------------------------------------------

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
    except Exception as e:
        print(f"[Deploy] ERROR: {e}")
        raise
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Trace generation — pre-compute send times once, replay across all runs
# ---------------------------------------------------------------------------

def generate_trace(
    schedules: Dict[str, List[Tuple[float, float]]],
    seed: int = 42,
) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    trace: Dict[str, List[float]] = {}
    for task, schedule in schedules.items():
        sends: List[float] = []
        t = 0.0
        for end_t, rps in schedule:
            if rps <= 0:
                t = end_t
                continue
            while t < end_t:
                sends.append(t)
                t += rng.exponential(1.0 / rps)
        trace[task] = sends
    return trace


def save_trace(trace: Dict[str, List[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(trace, f)
    print(f"[Trace] Saved {sum(len(v) for v in trace.values())} send times → {path}")


def load_trace(path: Path) -> Dict[str, List[float]]:
    with path.open() as f:
        trace = json.load(f)
    print(f"[Trace] Loaded {sum(len(v) for v in trace.values())} send times ← {path}")
    return trace


# ---------------------------------------------------------------------------
# Time-series open-loop sender
# ---------------------------------------------------------------------------

Record = Tuple[float, float, float, float, float, float, float, float, int]
# (send_time_relative_s, client_latency_ms, server_exec_ms, server_proc_ms,
#  server_swap_ms, server_decoder_ms, queue_wait_plus_rpc_ms,
#  client_pre_rpc_ms, server_start_ns)


async def run_timeseries(
    task_urls: Dict[str, str],
    schedules: Dict[str, List[Tuple[float, float]]],
    data: Dict[str, Dict],
    req_timeout: float = 60.0,
    trace: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, List[Record]]:
    clients: Dict[str, EdgeRuntimeClient] = {
        t: EdgeRuntimeClient(url) for t, url in task_urls.items()
    }
    for t, c in clients.items():
        print(f"[Run] Waiting for {t} server ({task_urls[t]}) ...")
        await c.wait_ready()

    task_proto = {
        task: encode_infer_request(task=task, x=data[task]["x"], mask=data[task].get("mask"))
        for task in schedules
    }
    task_stub = {task: clients[task]._stub for task in schedules}

    records: Dict[str, List[Record]] = {t: [] for t in schedules}

    async def _fire(task: str, req_id: int, t_send_abs: float,
                    t_start: float) -> None:
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

    async def _task_sender_trace(task: str,
                                 send_times: List[float],
                                 req_id_offset: int,
                                 t_start: float) -> None:
        in_flight = []
        for req_id, rel_t in enumerate(send_times):
            target = t_start + rel_t
            wait   = target - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            t_send = time.time()
            in_flight.append(asyncio.create_task(
                _fire(task, req_id_offset + req_id, t_send, t_start)
            ))
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

    async def _task_sender_live(task: str,
                                schedule: List[Tuple[float, float]],
                                req_id_offset: int,
                                t_start: float) -> None:
        req_id    = req_id_offset
        in_flight = []
        phase_idx = 0

        while phase_idx < len(schedule):
            end_t, rps = schedule[phase_idx]
            phase_end  = t_start + end_t

            if time.time() >= phase_end:
                phase_idx += 1
                continue

            if rps <= 0:
                await asyncio.sleep(max(0, phase_end - time.time()))
                phase_idx += 1
                continue

            t_send = time.time()
            in_flight.append(asyncio.create_task(
                _fire(task, req_id, t_send, t_start)
            ))
            req_id += 1

            gap       = np.random.exponential(1.0 / rps)
            sleep_for = min(t_send + gap, phase_end) - time.time()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

            if time.time() >= phase_end:
                phase_idx += 1

        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

    t_start = time.time()
    if trace is not None:
        senders = [
            _task_sender_trace(t, trace[t], i * 1_000_000, t_start)
            for i, t in enumerate(schedules)
        ]
    else:
        senders = [
            _task_sender_live(t, sched, i * 1_000_000, t_start)
            for i, (t, sched) in enumerate(schedules.items())
        ]
    await asyncio.gather(*senders, return_exceptions=True)
    for c in clients.values():
        await c.close()
    return records


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_records(records: Dict[str, List[Record]], out_dir: Path,
                 phase_boundaries: List[float],
                 aggressor_rps_phases: List[float],
                 scheduler_policy: str = "",
                 warmup_secs: float = 10.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- timeseries CSV (existing format, kept for plot.py compatibility) ---
    for task, recs in records.items():
        path = out_dir / f"{task}_timeseries.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["task", "send_time_s", "latency_ms", "phase"])
            for rec in recs:
                rel_t, lat = rec[0], rec[1]
                phase = len(phase_boundaries)
                for i, boundary in enumerate(phase_boundaries):
                    if rel_t < boundary:
                        phase = i + 1
                        break
                w.writerow([task, f"{rel_t:.4f}", f"{lat:.3f}", phase])
        print(f"[Save] {path}  ({len(recs)} records)")

    # --- latencies.csv (rich per-request breakdown, same schema as sharing_benefit) ---
    with (out_dir / "latencies.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "task", "condition", "elapsed_sec", "latency_ms",
            "server_exec_ms", "server_proc_ms", "server_swap_ms",
            "server_decoder_ms", "queue_wait_plus_rpc_ms",
            "client_pre_rpc_ms", "non_server_exec_overhead_ms", "server_start_ns",
            "phase",
        ])
        for task, recs in records.items():
            for rec in recs:
                rel_t, lat, exec_ms, proc_ms, swap_ms, dec_ms, queue_ms, pre_rpc_ms, start_ns = rec
                phase = len(phase_boundaries)
                for i, boundary in enumerate(phase_boundaries):
                    if rel_t < boundary:
                        phase = i + 1
                        break
                w.writerow([
                    task, scheduler_policy,
                    round(rel_t, 4), round(lat, 4),
                    round(exec_ms, 4), round(proc_ms, 4), round(swap_ms, 4),
                    round(dec_ms, 4), round(queue_ms, 4), round(pre_rpc_ms, 4),
                    round(max(0.0, lat - exec_ms), 4),
                    start_ns, phase,
                ])

    # --- task_results.csv (per-task aggregates, same schema as sharing_benefit) ---
    total_duration = phase_boundaries[-1] if phase_boundaries else 0.0
    with (out_dir / "task_results.csv").open("w", newline="") as f:
        fields = [
            "task", "condition", "n_requests", "throughput_rps",
            "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
            "avg_server_exec_ms", "avg_server_proc_ms", "avg_server_swap_ms",
            "avg_server_decoder_ms", "avg_queue_wait_plus_rpc_ms",
            "avg_client_pre_rpc_ms", "avg_non_server_exec_overhead_ms",
        ]
        dw = csv.DictWriter(f, fieldnames=fields)
        dw.writeheader()
        for task, recs in records.items():
            trimmed = [rec for rec in recs if rec[0] > warmup_secs]
            lats = [rec[1] for rec in trimmed]
            n = len(lats)
            if n == 0:
                continue
            effective_duration = max(total_duration - warmup_secs, 1.0)
            dw.writerow({
                "task":           task,
                "condition":      scheduler_policy,
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

    meta = {
        "phase_boundaries_s": phase_boundaries,
        "aggressor_rps_phases": aggressor_rps_phases,
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def _parse_float_list(s: str) -> List[float]:
    return [float(v) for v in s.split(",")]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-url",           default="localhost:8000",
                        help="Default device URL for both tasks (sharing / single-server runs)")
    parser.add_argument("--victim-url",           default=None,
                        help="Device URL for the victim task. Overrides --device-url for that task.")
    parser.add_argument("--aggressor-url",        default=None,
                        help="Device URL for the aggressor task. Overrides --device-url for that task.")
    parser.add_argument("--backbone",             default="momentbase")
    parser.add_argument("--victim-task",          default="ecgclass")
    parser.add_argument("--aggressor-task",       default="gestureclass")
    parser.add_argument("--victim-rps",           type=float, default=20.0,
                        help="Constant victim RPS (used if --victim-rps-phases is not set).")
    parser.add_argument("--victim-rps-phases",    default=None,
                        help="Comma-separated victim RPS per phase. If set, overrides "
                             "--victim-rps and must have the same length as "
                             "--aggressor-rps-phases.")
    parser.add_argument("--aggressor-rps-phases", default="20,30,60,90")
    parser.add_argument("--phase-durations",      default="30")
    parser.add_argument("--scheduler-policy",     default="fifo",
                        choices=["fifo", "round_robin", "wfq", "token_bucket", "saba",
                                 "deadline_split", "stfq"])
    parser.add_argument("--exp-dir",              default=os.environ.get(
                        "EXP_DIR", "experiments/noisy_neighbor/tsfm/results/fcfs"))
    parser.add_argument("--trace-file",           default=None)
    args = parser.parse_args()

    victim_url    = args.victim_url    or args.device_url
    aggressor_url = args.aggressor_url or args.device_url
    task_urls = {
        args.victim_task:    victim_url,
        args.aggressor_task: aggressor_url,
    }

    aggressor_rps_list = _parse_float_list(args.aggressor_rps_phases)
    num_phases = len(aggressor_rps_list)

    raw_durations = _parse_float_list(args.phase_durations)
    if len(raw_durations) == 1:
        phase_durations = raw_durations * num_phases
    elif len(raw_durations) == num_phases:
        phase_durations = raw_durations
    else:
        raise ValueError(
            f"--phase-durations has {len(raw_durations)} entries but "
            f"--aggressor-rps-phases has {num_phases}."
        )

    phase_boundaries: List[float] = []
    t = 0.0
    for d in phase_durations:
        t += d
        phase_boundaries.append(t)
    total_duration = phase_boundaries[-1]

    print("=" * 65)
    print(f"  noisy_neighbor/tsfm — {num_phases}-phase experiment")
    print(f"  Backbone   : {args.backbone}")
    print(f"  Victim     : {args.victim_task} @ {args.victim_rps} rps (constant)")
    print(f"  Aggressor  : {args.aggressor_task}")
    for i, (dur, rps) in enumerate(zip(phase_durations, aggressor_rps_list)):
        print(f"  Phase {i+1} ({dur:.0f}s): aggressor @ {rps} rps")
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

    if args.scheduler_policy == "token_bucket":
        async def _set_rates():
            client = EdgeRuntimeClient(victim_url)
            rates = {
                args.victim_task:    args.victim_rps,
                args.aggressor_task: aggressor_rps_list[-1],
            }
            payload = json.dumps({"rates": rates})
            resp = await client.control("set_rates", payload)
            print(f"[Deploy] set_rates: {resp['status']}")
            await client.close()
        asyncio.run(_set_rates())

    asyncio.run(asyncio.sleep(1))

    if args.victim_rps_phases is not None:
        victim_rps_list = _parse_float_list(args.victim_rps_phases)
        if len(victim_rps_list) != num_phases:
            raise ValueError(
                f"--victim-rps-phases has {len(victim_rps_list)} entries but "
                f"--aggressor-rps-phases has {num_phases}."
            )
        victim_schedule = list(zip(phase_boundaries, victim_rps_list))
    else:
        victim_schedule = [(total_duration, args.victim_rps)]
    aggressor_schedule = list(zip(phase_boundaries, aggressor_rps_list))

    schedules = {
        args.victim_task:    victim_schedule,
        args.aggressor_task: aggressor_schedule,
    }

    trace: Optional[Dict[str, List[float]]] = None
    if args.trace_file:
        trace_path = (SERVING_DIR / args.trace_file).resolve()
        if trace_path.exists():
            trace = load_trace(trace_path)
        else:
            print(f"[Trace] {trace_path} not found — generating and saving ...")
            trace = generate_trace(schedules)
            save_trace(trace, trace_path)
    else:
        out_dir   = (SERVING_DIR / args.exp_dir).resolve()
        auto_path = out_dir.parent / "trace.json"
        if auto_path.exists():
            trace = load_trace(auto_path)
        else:
            print(f"[Trace] Generating trace (seed=42) → {auto_path}")
            trace = generate_trace(schedules)
            save_trace(trace, auto_path)

    print(f"\n[Run] Starting ({total_duration:.0f}s total) ...")
    print(f"  {args.victim_task}    → {victim_url}")
    print(f"  {args.aggressor_task} → {aggressor_url}")
    req_timeout = max(60.0, total_duration * 2)
    records = asyncio.run(run_timeseries(
        task_urls, schedules, data, req_timeout=req_timeout, trace=trace,
    ))

    out_dir = (SERVING_DIR / args.exp_dir).resolve()
    save_records(records, out_dir, phase_boundaries, aggressor_rps_list,
                 scheduler_policy=args.scheduler_policy)

    for task, recs in records.items():
        print(f"  [{task}] {len(recs)} requests recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
