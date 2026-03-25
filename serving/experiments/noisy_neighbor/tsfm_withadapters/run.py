#!/usr/bin/env python3
"""noisy_neighbor/tsfm_withadapters — Time-series interference experiment with LoRA adapters.

Same noisy-neighbor setup as tsfm, but each task uses a LoRA adapter on top of
momentbase + MLP decoder:
  victim:    ecgclass   (ecgclass_momentbase_mlp_lora)
  aggressor: gestureclass (gestureclass_momentbase_mlp_lora)

Usage (from serving/):
    python experiments/noisy_neighbor/tsfm_withadapters/run.py \
        --device-url localhost:8000 \
        --backbone momentbase \
        --victim-task ecgclass   --victim-rps 20 \
        --aggressor-task gestureclass \
        --aggressor-rps-phases 20,30,60,90 \
        --phase-durations 30 \
        --exp-dir experiments/noisy_neighbor/tsfm_withadapters/results/fifo
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

from site_manager.grpc_client import EdgeRuntimeClient
from site_manager.config import DATASET_DIR as _DATASET_DIR

# ---------------------------------------------------------------------------
# Task library
# ---------------------------------------------------------------------------

TASK_TYPES: Dict[str, str] = {
    "ecgclass":     "classification",
    "gestureclass": "classification",
}

# adapter type per task — both use lora on momentbase
TASK_ADAPTERS: Dict[str, str] = {
    "ecgclass":     "lora",
    "gestureclass": "lora",
}

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def build_data(tasks: List[str]) -> Dict[str, Dict]:
    from fmtk.datasetloaders.ecg5000 import ECG5000Dataset
    from fmtk.datasetloaders.uwavegesture import UWaveGestureLibraryALLDataset

    d = _DATASET_DIR
    cfg = {"batch_size": 1, "shuffle": False}
    all_loaders = {
        "ecgclass":     lambda: DataLoader(ECG5000Dataset({"dataset_path": f"{d}/ECG5000"}, {"task_type": "classification"}, "test"), **cfg),
        "gestureclass": lambda: DataLoader(UWaveGestureLibraryALLDataset({"dataset_path": f"{d}/UWaveGestureLibraryAll", "seq_len": 512}, {"task_type": "classification"}, "test"), **cfg),
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
                                task_specs: List[Dict]) -> dict:
    print(f"[Deploy] Connecting to {device_url} ...")
    client = EdgeRuntimeClient(device_url)
    try:
        await client.wait_ready()
        payload = json.dumps({"backbone": backbone, "decoders": task_specs})
        print(f"[Deploy] Sending Control(load) backbone={backbone} tasks={len(task_specs)} ...")
        resp = await client.control("load", payload)
        print(f"[Deploy] Control(load) returned: {resp['status']}")
        return resp
    except Exception as e:
        print(f"[Deploy] ERROR: {e}")
        raise
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Trace generation
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

Record = Tuple[float, float]  # (send_time_relative_to_start_s, latency_ms)


async def run_timeseries(
    device_url: str,
    schedules: Dict[str, List[Tuple[float, float]]],
    data: Dict[str, Dict],
    req_timeout: float = 60.0,
    trace: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, List[Record]]:
    client = EdgeRuntimeClient(device_url)
    await client.wait_ready()

    records: Dict[str, List[Record]] = {t: [] for t in schedules}

    async def _fire(task: str, req_id: int, t_send_abs: float,
                    t_start: float) -> None:
        d = data[task]
        try:
            await asyncio.wait_for(client.infer({
                "req_id": req_id,
                "task":   task,
                "x":      d["x"],
                "mask":   d.get("mask"),
            }), timeout=req_timeout)
            lat_ms = (time.time() - t_send_abs) * 1000
            records[task].append((t_send_abs - t_start, lat_ms))
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
    await client.close()
    return records


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_records(records: Dict[str, List[Record]], out_dir: Path,
                 phase_boundaries: List[float],
                 aggressor_rps_phases: List[float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for task, recs in records.items():
        path = out_dir / f"{task}_timeseries.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["task", "send_time_s", "latency_ms", "phase"])
            for rel_t, lat in recs:
                phase = len(phase_boundaries)
                for i, boundary in enumerate(phase_boundaries):
                    if rel_t < boundary:
                        phase = i + 1
                        break
                w.writerow([task, f"{rel_t:.4f}", f"{lat:.3f}", phase])
        print(f"[Save] {path}  ({len(recs)} records)")

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
    parser.add_argument("--device-url",           default="localhost:8000")
    parser.add_argument("--backbone",             default="momentbase")
    parser.add_argument("--victim-task",          default="ecgclass")
    parser.add_argument("--aggressor-task",       default="gestureclass")
    parser.add_argument("--victim-rps",           type=float, default=20.0)
    parser.add_argument("--aggressor-rps-phases", default="20,30,60,90",
                        help="Comma-separated aggressor RPS per phase, e.g. '20,30,50,150'")
    parser.add_argument("--phase-durations",      default="30",
                        help="Comma-separated phase durations (s). Single value = same for all phases.")
    parser.add_argument("--scheduler-policy",     default="fifo",
                        choices=["fifo", "round_robin", "wfq", "token_bucket", "saba",
                                 "deadline_split", "stfq"])
    parser.add_argument("--exp-dir",              default=os.environ.get(
                        "EXP_DIR", "experiments/noisy_neighbor/tsfm_withadapters/results/fifo"))
    parser.add_argument("--trace-file",           default=None,
                        help="Path to pre-generated trace JSON. If provided, replays "
                             "identical send times across runs. If not provided, generates "
                             "a fresh Poisson trace and saves it to <exp-dir>/../trace.json.")
    args = parser.parse_args()

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
    print(f"  tsfm_withadapters — {num_phases}-phase experiment")
    print(f"  Backbone   : {args.backbone} + LoRA adapters")
    print(f"  Victim     : {args.victim_task} @ {args.victim_rps} rps (constant)")
    print(f"  Aggressor  : {args.aggressor_task}")
    for i, (dur, rps) in enumerate(zip(phase_durations, aggressor_rps_list)):
        print(f"  Phase {i+1} ({dur:.0f}s): aggressor @ {rps} rps")
    print(f"  Scheduler  : {args.scheduler_policy}")
    print("=" * 65)

    tasks = [args.victim_task, args.aggressor_task]
    print(f"\n[INFO] Loading data for: {tasks}")
    data = build_data(tasks)

    # Each task spec includes decoder type, path, and adapter
    task_specs = [
        {
            "task":    t,
            "type":    TASK_TYPES[t],
            "path":    f"{t}_{args.backbone}_mlp_lora",
            "adapter": TASK_ADAPTERS[t],
        }
        for t in tasks
    ]
    resp = asyncio.run(deploy_backbone_async(args.device_url, args.backbone, task_specs))
    if "error" in resp.get("status", "").lower():
        print(f"[Error] Deploy failed: {resp}")
        return 1

    # For token_bucket: register per-task rates
    if args.scheduler_policy == "token_bucket":
        async def _set_rates():
            client = EdgeRuntimeClient(args.device_url)
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

    victim_schedule     = [(total_duration, args.victim_rps)]
    aggressor_schedule  = list(zip(phase_boundaries, aggressor_rps_list))

    schedules = {
        args.victim_task:    victim_schedule,
        args.aggressor_task: aggressor_schedule,
    }

    # Load or generate trace
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
    req_timeout = max(60.0, total_duration * 2)
    records = asyncio.run(run_timeseries(
        args.device_url, schedules, data, req_timeout=req_timeout, trace=trace,
    ))

    out_dir = (SERVING_DIR / args.exp_dir).resolve()
    save_records(records, out_dir, phase_boundaries, aggressor_rps_list)

    for task, recs in records.items():
        print(f"  [{task}] {len(recs)} requests recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
