#!/usr/bin/env python3
"""sharing_benefit/tpc/run.py — Sharing benefit with TPC isolation for no_sharing.

Supports two task sets:
  tsfm   — time-series foundation model tasks:
              ecgclass, heartrate, diasbp, sysbp, gestureclass,
              etth1fore, weatherfore, trafficfore, eclfore, exchangefore
  vision — nyudepth + vocseg (vision foundation models)

Condition types:
  single_{task}  — 1 device server, that task only, FIFO
  no_sharing_tpc — N device servers with --tpc-partition (TPC-pinned), FIFO
  no_sharing     — N device servers (one per task), FIFO
  sharing        — 1 device server, all tasks, STFQ

Usage (called by run.sh):
    python experiments/sharing_benefit/tpc/run.py \
        --task-set    tsfm \
        --tasks       ecgclass,gestureclass,heartrate,diasbp \
        --condition   no_sharing_tpc \
        --device-urls localhost:8000,localhost:8001,localhost:8002,localhost:8003 \
        --backbone    momentlarge \
        --rps 25 \
        --duration 300 \
        --exp-dir experiments/sharing_benefit/tpc/results/ntasks_4/rps_25/no_sharing_tpc
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

from serving.site_manager.grpc_client import EdgeRuntimeClient
from serving.site_manager.config import DATASET_DIR as _DATASET_DIR

# ---------------------------------------------------------------------------
# Task-set configuration
# ---------------------------------------------------------------------------

# Seeds for trace generation — one per task across all task sets
_TSFM_SEEDS = {
    "ecgclass":     42,
    "heartrate":    43,
    "diasbp":       44,
    "sysbp":        45,
    "gestureclass": 46,
    "etth1fore":    47,
    "weatherfore":  48,
    "trafficfore":  49,
    "eclfore":      50,
    "exchangefore": 51,
}

# All supported tsfm tasks in canonical order
ALL_TSFM_TASKS = list(_TSFM_SEEDS.keys())

TASK_SETS: Dict[str, Dict] = {
    "tsfm": {
        "tasks": ["ecgclass", "gestureclass"],  # default 2-task subset
        "types": {
            "ecgclass":     "classification",
            "heartrate":    "regression",
            "diasbp":       "regression",
            "sysbp":        "regression",
            "gestureclass": "classification",
            "etth1fore":    "forecasting",
            "weatherfore":  "forecasting",
            "trafficfore":  "forecasting",
            "eclfore":      "forecasting",
            "exchangefore": "forecasting",
        },
        "decoder_paths": {
            "ecgclass":     "{task}_{backbone}_mlp",
            "heartrate":    "{task}_{backbone}_mlp",
            "diasbp":       "{task}_{backbone}_mlp",
            "sysbp":        "{task}_{backbone}_mlp",
            "gestureclass": "{task}_{backbone}_mlp",
            "etth1fore":    "{task}_{backbone}_mlp",
            "weatherfore":  "{task}_{backbone}_mlp",
            "trafficfore":  "{task}_{backbone}_mlp",
            "eclfore":      "{task}_{backbone}_mlp",
            "exchangefore": "{task}_{backbone}_mlp",
        },
        "seeds": _TSFM_SEEDS,
    },
    "vision": {
        "tasks": ["nyudepth", "vocseg"],
        "types": {
            "nyudepth": "monocular",
            "vocseg":   "linear_seg",
        },
        "decoder_paths": {
            "nyudepth": "nyudepth_{backbone}_monocular",
            "vocseg":   None,
        },
        "seeds": {"nyudepth": 42, "vocseg": 43},
    },
}

# Dataset paths for vision (can be overridden via env vars)
NYUDEPTH_PATH = os.environ.get("NYUDEPTH_PATH", "../../FMTK/dataset/nyu-depth-v2")
PASCALVOC_PATH = os.environ.get("PASCALVOC_PATH", "../../FMTK/dataset/PASCAL-VOC")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def build_data(task_set: str, tasks: List[str]) -> Dict[str, Dict]:
    cfg = {"batch_size": 1, "shuffle": False}

    if task_set == "tsfm":
        from fmtk.datasetloaders.ecg5000 import ECG5000Dataset
        from fmtk.datasetloaders.uwavegesture import UWaveGestureLibraryALLDataset
        from fmtk.datasetloaders.ppg import PPGDataset
        from fmtk.datasetloaders.etth1 import ETTh1Dataset
        from fmtk.datasetloaders.weather import WeatherDataset
        from fmtk.datasetloaders.traffic import TrafficDataset
        from fmtk.datasetloaders.ecl import ECLDataset
        from fmtk.datasetloaders.exchange import ExchangeDataset
        d = _DATASET_DIR
        loaders = {
            "ecgclass": lambda: DataLoader(
                ECG5000Dataset({"dataset_path": f"{d}/ECG5000", "seq_len": 512},
                               {"task_type": "classification"}, "test"),
                **cfg,
            ),
            "heartrate": lambda: DataLoader(
                PPGDataset({"dataset_path": f"{d}/PPG-data", "seq_len": 512, "num_channels": 1},
                           {"task_type": "regression", "label": "hr"}, "test"),
                **cfg,
            ),
            "diasbp": lambda: DataLoader(
                PPGDataset({"dataset_path": f"{d}/PPG-data", "seq_len": 512, "num_channels": 1},
                           {"task_type": "regression", "label": "diasbp"}, "test"),
                **cfg,
            ),
            "sysbp": lambda: DataLoader(
                PPGDataset({"dataset_path": f"{d}/PPG-data", "seq_len": 512, "num_channels": 1},
                           {"task_type": "regression", "label": "sysbp"}, "test"),
                **cfg,
            ),
            "gestureclass": lambda: DataLoader(
                UWaveGestureLibraryALLDataset(
                    {"dataset_path": f"{d}/UWaveGestureLibraryAll", "seq_len": 512},
                    {"task_type": "classification"}, "test",
                ),
                **cfg,
            ),
            "etth1fore": lambda: DataLoader(
                ETTh1Dataset({"dataset_path": f"{d}/ETTh1", "seq_len": 512},
                             {"task_type": "forecasting"}, "test"),
                **cfg,
            ),
            "weatherfore": lambda: DataLoader(
                WeatherDataset({"dataset_path": f"{d}/Weather", "seq_len": 512},
                               {"task_type": "forecasting"}, "test"),
                **cfg,
            ),
            "trafficfore": lambda: DataLoader(
                TrafficDataset({"dataset_path": f"{d}/Traffic", "seq_len": 512},
                               {"task_type": "forecasting"}, "test"),
                **cfg,
            ),
            "eclfore": lambda: DataLoader(
                ECLDataset({"dataset_path": f"{d}/ElectricityLoad-data", "seq_len": 512},
                           {"task_type": "forecasting"}, "test"),
                **cfg,
            ),
            "exchangefore": lambda: DataLoader(
                ExchangeDataset({"dataset_path": f"{d}/Exchange", "seq_len": 512},
                                {"task_type": "forecasting"}, "test"),
                **cfg,
            ),
        }
    else:  # vision
        from fmtk.datasetloaders.nyudepthv2 import NYUDepthV2Dataset
        from fmtk.datasetloaders.voc12 import VOC12Dataset
        loaders = {
            "nyudepth": lambda: DataLoader(
                NYUDepthV2Dataset(
                    {"dataset_path": NYUDEPTH_PATH},
                    {"task_type": "regression"},
                    "test",
                ),
                **cfg,
            ),
            "vocseg": lambda: DataLoader(
                VOC12Dataset(
                    {"dataset_path": PASCALVOC_PATH, "target_size": 224},
                    {"task_type": "segmentation"},
                    "test",
                ),
                **cfg,
            ),
        }

    data = {}
    for task in tasks:
        loader = loaders[task]()
        batch = next(iter(loader))
        data[task] = {
            "x":    batch["x"].numpy().astype(np.float32),
            "mask": batch["mask"].numpy().astype(np.float32) if "mask" in batch else None,
        }
        print(f"[Data] Loaded {task}: x.shape={data[task]['x'].shape}")
    return data


# ---------------------------------------------------------------------------
# Trace generation
# ---------------------------------------------------------------------------

def generate_traces(tasks: List[str], seeds: Dict[str, int],
                    rps: float, duration: float,
                    slot_keys: Optional[List[str]] = None) -> Dict[str, List[float]]:
    """Generate one Poisson trace per entry in tasks (or slot_keys if provided).

    When slot_keys is given (e.g. ["slot_0:ecgclass", "slot_10:ecgclass"]) each
    slot gets a unique seed derived from its index so repeated tasks produce
    independent arrival streams.  The returned dict is keyed by slot_key.
    When slot_keys is None the dict is keyed by task name (legacy behaviour).
    """
    send_times: Dict[str, List[float]] = {}
    keys = slot_keys if slot_keys is not None else tasks
    for i, (key, task) in enumerate(zip(keys, tasks)):
        # Unique seed: base task seed offset by slot index so repeats differ
        base_seed = seeds.get(task, 42)
        seed = base_seed + i * 1000
        rng = np.random.default_rng(seed)
        times, t = [], 0.0
        while t < duration:
            times.append(t)
            t += rng.exponential(1.0 / rps)
        send_times[key] = times
    return send_times


def save_trace(trace: Dict[str, List[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(trace, f)
    print(f"[Trace] Saved {sum(len(v) for v in trace.values())} send times -> {path}")


def load_trace(path: Path) -> Dict[str, List[float]]:
    with path.open() as f:
        trace = json.load(f)
    print(f"[Trace] Loaded {sum(len(v) for v in trace.values())} send times <- {path}")
    return trace


# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

async def deploy(device_url: str, backbone: str, tasks: List[str],
                 task_types: Dict[str, str], decoder_paths: Dict[str, Optional[str]]) -> None:
    bb_short = backbone.replace("-patch", "")
    decoders = []
    for t in tasks:
        tmpl = decoder_paths[t]
        path = tmpl.format(task=t, backbone=bb_short) if tmpl else None
        decoders.append({"task": t, "type": task_types[t], "path": path})
    client = EdgeRuntimeClient(device_url)
    try:
        await client.wait_ready()
        payload = json.dumps({"backbone": backbone, "decoders": decoders})
        print(f"[Deploy] {device_url}  backbone={backbone}  tasks={tasks}")
        resp = await client.control("load", payload)
        print(f"[Deploy] {device_url}  status={resp['status']}")
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Open-loop Poisson sender (gRPC)
# ---------------------------------------------------------------------------

Record = Tuple[float, float, float, float, float, float, float, int]
# (
#   send_time_relative_s,
#   client_latency_ms,
#   server_exec_ms,
#   server_proc_ms,
#   server_swap_ms,
#   server_decoder_ms,
#   queue_wait_plus_rpc_ms,
#   server_start_ns,
# )


async def run_open_loop(
    task_urls: Dict[str, str],
    data: Dict[str, Dict],
    send_times: Dict[str, List[float]],
    req_timeout: float = 60.0,
) -> Dict[str, List[Record]]:
    # task_urls is keyed by slot key (e.g. "slot_0", "slot_1") when tasks repeat,
    # or by task name when all tasks are unique. The slot_task map carries the
    # actual task name for each slot.
    # For backwards compat: if keys are plain task names, wrap them transparently.
    slot_task: Dict[str, str] = {}   # slot_key -> task name
    slot_url:  Dict[str, str] = {}   # slot_key -> device url
    slot_times: Dict[str, List[float]] = {}  # slot_key -> send times

    for slot, url in task_urls.items():
        # slot is either "slot_N:task" (new) or plain task name (legacy)
        if slot.startswith("slot_") and ":" in slot:
            task = slot.split(":", 1)[1]
        else:
            task = slot
        # Use the full slot string as the key so records stay per-slot.
        slot_task[slot] = task
        slot_url[slot]  = url
        # Use per-slot trace keyed by full slot string; fall back to task-level.
        slot_times[slot] = send_times.get(slot, send_times.get(task, []))

    unique_urls = list(set(slot_url.values()))
    clients: Dict[str, EdgeRuntimeClient] = {}
    for url in unique_urls:
        c = EdgeRuntimeClient(url)
        await c.wait_ready()
        clients[url] = c

    records: Dict[str, List[Record]] = {sk: [] for sk in slot_task}

    async def _fire(slot_key: str, req_id: int, t_send_abs: float, t_start: float) -> None:
        task = slot_task[slot_key]
        d = data[task]
        try:
            resp = await asyncio.wait_for(clients[slot_url[slot_key]].infer({
                "req_id": req_id,
                "task":   task,
                "x":      d["x"],
                "mask":   d.get("mask"),
            }), timeout=req_timeout)
            t_done_abs = time.time()
            client_lat_ms = (t_done_abs - t_send_abs) * 1000
            server_start_s = resp["start_time_ns"] / 1e9
            server_exec_ms = (resp["end_time_ns"] - resp["start_time_ns"]) / 1e6
            server_proc_ms = resp.get("proc_time_ns", 0) / 1e6
            server_swap_ms = resp.get("swap_time_ns", 0) / 1e6
            server_decoder_ms = resp.get("decoder_time_ns", 0) / 1e6
            queue_wait_plus_rpc_ms = max(0.0, (server_start_s - t_send_abs) * 1000)
            records[slot_key].append((
                t_send_abs - t_start,
                client_lat_ms,
                server_exec_ms,
                server_proc_ms,
                server_swap_ms,
                server_decoder_ms,
                queue_wait_plus_rpc_ms,
                resp["start_time_ns"],
            ))
        except Exception:
            pass

    t_start = time.time()

    async def _sender(slot_key: str, req_id_offset: int) -> None:
        in_flight = []
        for req_id, rel_t in enumerate(slot_times[slot_key]):
            target = t_start + rel_t
            wait = target - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            t_send = time.time()
            in_flight.append(asyncio.create_task(
                _fire(slot_key, req_id_offset + req_id, t_send, t_start)
            ))
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

    senders = [
        _sender(sk, i * 1_000_000)
        for i, sk in enumerate(slot_task)
    ]
    await asyncio.gather(*senders, return_exceptions=True)

    for c in clients.values():
        await c.close()

    # Return records keyed by slot key (e.g. "slot_0" or plain task name).
    # save_results uses slot_task to write the task name into the CSV rows,
    # so each slot appears as a separate row but with its real task name.
    return records


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(records: Dict[str, List[Record]], out_dir: Path, condition: str,
                 duration: float, warmup_secs: float = 10.0,
                 slot_task: Optional[Dict[str, str]] = None) -> None:
    """Save per-slot records to CSV.

    slot_task maps slot key -> real task name.  When provided, the task name
    written to the CSV is the real task name (e.g. "ecgclass"), but each slot
    is a separate row so repeated tasks remain distinguishable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def _task_name(key: str) -> str:
        if slot_task and key in slot_task:
            return slot_task[key]
        # Legacy: key is already the task name
        return key

    with (out_dir / "latencies.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "task", "slot", "condition", "elapsed_sec", "latency_ms",
            "server_exec_ms", "server_proc_ms", "server_swap_ms",
            "server_decoder_ms", "queue_wait_plus_rpc_ms",
            "non_server_exec_overhead_ms", "server_start_ns",
        ])
        for slot_key, recs in records.items():
            task = _task_name(slot_key)
            for rel_t, lat, exec_ms, proc_ms, swap_ms, dec_ms, queue_ms, start_ns in recs:
                w.writerow([
                    task, slot_key, condition,
                    round(rel_t, 4), round(lat, 4),
                    round(exec_ms, 4), round(proc_ms, 4), round(swap_ms, 4),
                    round(dec_ms, 4), round(queue_ms, 4),
                    round(max(0.0, lat - exec_ms), 4),
                    start_ns,
                ])

    with (out_dir / "task_results.csv").open("w", newline="") as f:
        fields = [
            "task", "slot", "condition", "n_requests", "throughput_rps",
            "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
            "avg_server_exec_ms", "avg_server_proc_ms", "avg_server_swap_ms",
            "avg_server_decoder_ms", "avg_queue_wait_plus_rpc_ms",
            "avg_non_server_exec_overhead_ms",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for slot_key, recs in records.items():
            task = _task_name(slot_key)
            trimmed = [rec for rec in recs if rec[0] > warmup_secs]
            lats = [rec[1] for rec in trimmed]
            n = len(lats)
            if n == 0:
                continue
            w.writerow({
                "task":           task,
                "slot":           slot_key,
                "condition":      condition,
                "n_requests":     n,
                "throughput_rps": round(n / (duration - warmup_secs), 4),
                "avg_latency_ms": round(float(np.mean(lats)), 3),
                "p50_latency_ms": round(float(np.percentile(lats, 50)), 3),
                "p95_latency_ms": round(float(np.percentile(lats, 95)), 3),
                "p99_latency_ms": round(float(np.percentile(lats, 99)), 3),
                "avg_server_exec_ms":    round(float(np.mean([rec[2] for rec in trimmed])), 3),
                "avg_server_proc_ms":    round(float(np.mean([rec[3] for rec in trimmed])), 3),
                "avg_server_swap_ms":    round(float(np.mean([rec[4] for rec in trimmed])), 3),
                "avg_server_decoder_ms": round(float(np.mean([rec[5] for rec in trimmed])), 3),
                "avg_queue_wait_plus_rpc_ms": round(float(np.mean([rec[6] for rec in trimmed])), 3),
                "avg_non_server_exec_overhead_ms": round(
                    float(np.mean([max(0.0, rec[1] - rec[2]) for rec in trimmed])), 3),
            })

    for slot_key, recs in records.items():
        task = _task_name(slot_key)
        print(f"[Save] {task} (slot={slot_key}): {len(recs)} total requests -> {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-set", default=os.environ.get("TASK_SET", "tsfm"),
                        choices=["tsfm", "vision"],
                        help="Which task set to use")
    parser.add_argument("--tasks", default=None,
                        help="Comma-separated list of tasks to run (overrides task-set default). "
                             "tsfm options: " + ", ".join(ALL_TSFM_TASKS))
    parser.add_argument("--condition", required=True,
                        help="Condition name: single_{task}, no_sharing_tpc, no_sharing, or sharing")
    parser.add_argument("--device-urls", default=None,
                        help="Comma-separated device URLs, one per task for no_sharing* conditions "
                             "(e.g. localhost:8000,localhost:8001,localhost:8002)")
    parser.add_argument("--device-url",   default="localhost:8000",
                        help="Primary device URL (used when --device-urls is not set)")
    parser.add_argument("--device-url-2", default="localhost:8001",
                        help="Secondary device URL (legacy two-task fallback)")
    parser.add_argument("--backbone",     default=os.environ.get("BACKBONE", "momentbase"))
    parser.add_argument("--rps",          type=float, default=float(os.environ.get("RPS", "20")))
    parser.add_argument("--duration",     type=float, default=float(os.environ.get("PHASE_DURATION", "180")))
    parser.add_argument("--warmup-secs",  type=float, default=10.0)
    parser.add_argument("--exp-dir",      default=os.environ.get("EXP_DIR", "experiments/sharing_benefit/tpc/results"))
    parser.add_argument("--trace-file",   default=None)
    args = parser.parse_args()

    cfg = TASK_SETS[args.task_set]
    task_types    = cfg["types"]
    decoder_paths = cfg["decoder_paths"]
    task_seeds    = cfg["seeds"]

    # Resolve the active task list
    if args.tasks:
        all_tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        all_tasks = cfg["tasks"]

    # Resolve device URLs
    if args.device_urls:
        device_urls = [u.strip() for u in args.device_urls.split(",")]
    else:
        device_urls = [args.device_url, args.device_url_2]

    out_dir = (SERVING_DIR / args.exp_dir).resolve()

    print("=" * 65)
    print(f"  Sharing Benefit + TPC — condition={args.condition}")
    print(f"  Task set  : {args.task_set}  ({all_tasks})")
    print(f"  Backbone  : {args.backbone}")
    print(f"  RPS/task  : {args.rps}")
    print(f"  Duration  : {args.duration}s  (warmup={args.warmup_secs}s)")
    print(f"  Results   : {out_dir}")
    print("=" * 65)

    # Load or generate trace (covers all active tasks)
    if args.trace_file:
        trace_path = Path(args.trace_file)
        if not trace_path.is_absolute():
            trace_path = (SERVING_DIR / trace_path).resolve()
        if trace_path.exists():
            send_times = load_trace(trace_path)
            # Add missing tasks if trace was generated for fewer tasks
            missing = [t for t in all_tasks if t not in send_times]
            if missing:
                print(f"[Trace] Generating missing tasks: {missing}")
                extra = generate_traces(missing, task_seeds, args.rps, args.duration)
                send_times.update(extra)
                save_trace(send_times, trace_path)
        else:
            print(f"[Trace] {trace_path} not found — generating and saving ...")
            send_times = generate_traces(all_tasks, task_seeds, args.rps, args.duration)
            save_trace(send_times, trace_path)
    else:
        auto_path = (out_dir.parent / "trace.json").resolve()
        if auto_path.exists():
            send_times = load_trace(auto_path)
            missing = [t for t in all_tasks if t not in send_times]
            if missing:
                print(f"[Trace] Generating missing tasks: {missing}")
                extra = generate_traces(missing, task_seeds, args.rps, args.duration)
                send_times.update(extra)
                save_trace(send_times, auto_path)
        else:
            print(f"[Trace] Generating trace -> {auto_path}")
            send_times = generate_traces(all_tasks, task_seeds, args.rps, args.duration)
            save_trace(send_times, auto_path)

    # Determine active tasks and URL mapping per condition.
    # Always use "slot_N:task" keys so each slot gets its own independent trace
    # and records — even when the same task appears multiple times.
    def _slot_key(i: int, task: str) -> str:
        return f"slot_{i}:{task}"

    # _task_from_slot extracts the real task name from a slot key.
    def _task_from_slot(slot: str) -> str:
        return slot.split(":", 1)[1] if ":" in slot else slot

    if args.condition.startswith("single_"):
        single_task = args.condition[len("single_"):]
        if single_task not in all_tasks:
            print(f"ERROR: task '{single_task}' not in active task list {all_tasks}", file=sys.stderr)
            return 1
        tasks = [single_task]
        task_urls = {_slot_key(0, single_task): device_urls[0]}
    elif args.condition in ("no_sharing_tpc", "no_sharing", "no_sharing_mps"):
        tasks = all_tasks
        if len(device_urls) < len(all_tasks):
            print(f"ERROR: need {len(all_tasks)} device URLs for {args.condition}, got {len(device_urls)}",
                  file=sys.stderr)
            return 1
        task_urls = {_slot_key(i, t): device_urls[i] for i, t in enumerate(all_tasks)}
    elif args.condition == "sharing":
        tasks = all_tasks
        task_urls = {_slot_key(i, t): device_urls[0] for i, t in enumerate(all_tasks)}
    else:
        print(f"ERROR: unknown condition '{args.condition}'", file=sys.stderr)
        return 1

    # Build slot_task map (slot_key -> real task name) for use in saving results.
    slot_task_map: Dict[str, str] = {sk: _task_from_slot(sk) for sk in task_urls}

    # Build per-slot traces: each slot gets a unique arrival stream even when
    # the underlying task is the same (different seed = different inter-arrivals).
    slot_keys = list(task_urls.keys())
    slot_task_names = [_task_from_slot(sk) for sk in slot_keys]
    slot_send_times = generate_traces(
        slot_task_names, task_seeds, args.rps, args.duration, slot_keys=slot_keys
    )

    print(f"[INFO] Loading data for: {tasks}")
    data = build_data(args.task_set, tasks)

    # Deploy
    if args.condition in ("no_sharing_tpc", "no_sharing", "no_sharing_mps"):
        for slot, url in task_urls.items():
            task_name = _task_from_slot(slot)
            asyncio.run(deploy(url, args.backbone, [task_name], task_types, decoder_paths))
    else:
        asyncio.run(deploy(device_urls[0], args.backbone, list(dict.fromkeys(tasks)), task_types, decoder_paths))

    asyncio.run(asyncio.sleep(1))

    print(f"\n[Run] Starting open-loop send ({args.duration}s) ...")
    records = asyncio.run(run_open_loop(
        task_urls=task_urls,
        data=data,
        send_times=slot_send_times,
    ))

    save_results(records, out_dir, args.condition, args.duration, args.warmup_secs,
                 slot_task=slot_task_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
