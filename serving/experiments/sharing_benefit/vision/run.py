#!/usr/bin/env python3
"""sharing_benefit/vision/run.py — Sharing benefit experiment for vision models (nyudepth + vocseg).

Four conditions:
  single_nyudepth  — 1 device server, nyudepth only, FIFO
  single_vocseg    — 1 device server, vocseg only, FIFO
  no_sharing       — 2 device servers (port A + B), one backbone each, FIFO
  sharing          — 1 device server, both tasks, STFQ

Each condition: deploy -> run open-loop Poisson at fixed RPS -> save latencies.csv.
run.sh handles starting/stopping device servers; this script just runs the experiment.

Usage (called by run.sh):
    python experiments/sharing_benefit/vision/run.py \
        --condition sharing \
        --device-url localhost:8000 \
        --device-url-2 localhost:8001 \
        --backbone dinobase \
        --rps 20 \
        --duration 180 \
        --exp-dir experiments/sharing_benefit/vision/results/sharing
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

NYUDEPTH_PATH = "../../FMTK/dataset/nyu-depth-v2"
PASCALVOC_PATH = "../../FMTK/dataset/PASCAL-VOC"

TASK_TYPES: Dict[str, str] = {
    "nyudepth": "monocular_depth",
    "vocseg":   "linear_seg",
}

DECODER_PATHS: Dict[str, str] = {
    "nyudepth": "nyudepth_{backbone}_monocular",
    "vocseg":   None
}

BOTH_TASKS = ["nyudepth", "vocseg"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def build_data(tasks: List[str]) -> Dict[str, Dict]:
    from fmtk.datasetloaders.nyudepthv2 import NYUDepthV2Dataset
    from fmtk.datasetloaders.voc12 import VOC12Dataset

    cfg = {"batch_size": 1, "shuffle": False}
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
# Deploy
# ---------------------------------------------------------------------------

async def deploy(device_url: str, backbone: str, tasks: List[str]) -> None:
    bb_short = backbone.replace("-patch", "")
    # need to consider if path is None (e.g. for vocseg) since model loader needs to know which decoders to skip loading for
    decoders=[{"task": t, "type": TASK_TYPES[t], "path": DECODER_PATHS[t].format(backbone=bb_short) if DECODER_PATHS[t] else None} for t in tasks]
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
# Open-loop Poisson sender
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

TASK_SEEDS = {"nyudepth": 42, "vocseg": 43}


def generate_traces(tasks: List[str], rps: float, duration: float) -> Dict[str, List[float]]:
    """Generate per-task Poisson send times. Each task uses its own fixed seed
    so the trace is identical regardless of which condition/other tasks are present."""
    send_times: Dict[str, List[float]] = {}
    for task in tasks:
        rng = np.random.default_rng(TASK_SEEDS.get(task, 42))
        times, t = [], 0.0
        while t < duration:
            times.append(t)
            t += rng.exponential(1.0 / rps)
        send_times[task] = times
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


async def run_open_loop(
    task_urls: Dict[str, str],          # {task: device_url}
    data: Dict[str, Dict],
    send_times: Dict[str, List[float]], # pre-generated traces from generate_traces()
    req_timeout: float = 60.0,
) -> Dict[str, List[Record]]:
    """Send each task using pre-generated send times.

    Returns per-task records with client and server timing breakdowns.
    """
    # One persistent client per unique URL
    unique_urls = list(set(task_urls.values()))
    clients: Dict[str, EdgeRuntimeClient] = {}
    for url in unique_urls:
        c = EdgeRuntimeClient(url)
        await c.wait_ready()
        clients[url] = c

    records: Dict[str, List[Record]] = {t: [] for t in task_urls}

    async def _fire(task: str, req_id: int, t_send_abs: float, t_start: float) -> None:
        d = data[task]
        try:
            resp = await asyncio.wait_for(clients[task_urls[task]].infer({
                "req_id": req_id,
                "task":   task,
                "x":      d["x"],
                "mask":   d.get("mask"),
            }), timeout=req_timeout)
            t_done_abs = time.time()
            client_lat_ms = (t_done_abs - t_send_abs) * 1000
            server_start_s = resp["start_time_ns"] / 1e9
            server_exec_ms = (resp["end_time_ns"] - resp["start_time_ns"]) / 1e6
            server_proc_ms = resp["proc_time_ns"] / 1e6
            server_swap_ms = resp["swap_time_ns"] / 1e6
            server_decoder_ms = resp["decoder_time_ns"] / 1e6
            queue_wait_plus_rpc_ms = max(0.0, (server_start_s - t_send_abs) * 1000)
            records[task].append((
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
            pass  # drop errors -- don't pollute time-series

    # Shared t_start so all tasks begin from the same wall clock
    t_start = time.time()

    async def _sender(task: str, req_id_offset: int) -> None:
        in_flight = []
        for req_id, rel_t in enumerate(send_times[task]):
            target = t_start + rel_t
            wait = target - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            t_send = time.time()
            in_flight.append(asyncio.create_task(
                _fire(task, req_id_offset + req_id, t_send, t_start)
            ))
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

    senders = [
        _sender(task, i * 1_000_000)
        for i, task in enumerate(task_urls)
    ]
    await asyncio.gather(*senders, return_exceptions=True)

    for c in clients.values():
        await c.close()
    return records


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(records: Dict[str, List[Record]], out_dir: Path, condition: str,
                 duration: float, warmup_secs: float = 10.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # latencies.csv -- all requests with elapsed_sec for warmup trimming
    with (out_dir / "latencies.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "task",
            "condition",
            "elapsed_sec",
            "latency_ms",
            "server_exec_ms",
            "server_proc_ms",
            "server_swap_ms",
            "server_decoder_ms",
            "queue_wait_plus_rpc_ms",
            "non_server_exec_overhead_ms",
            "server_start_ns",
        ])
        for task, recs in records.items():
            for rel_t, lat, exec_ms, proc_ms, swap_ms, dec_ms, queue_ms, start_ns in recs:
                w.writerow([
                    task,
                    condition,
                    round(rel_t, 4),
                    round(lat, 4),
                    round(exec_ms, 4),
                    round(proc_ms, 4),
                    round(swap_ms, 4),
                    round(dec_ms, 4),
                    round(queue_ms, 4),
                    round(max(0.0, lat - exec_ms), 4),
                    start_ns,
                ])

    # task_results.csv -- per-task summary (excluding warmup)
    with (out_dir / "task_results.csv").open("w", newline="") as f:
        fields = [
            "task",
            "condition",
            "n_requests",
            "throughput_rps",
            "avg_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "avg_server_exec_ms",
            "avg_server_proc_ms",
            "avg_server_swap_ms",
            "avg_server_decoder_ms",
            "avg_queue_wait_plus_rpc_ms",
            "avg_non_server_exec_overhead_ms",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for task, recs in records.items():
            trimmed = [rec for rec in recs if rec[0] > warmup_secs]
            lats = [rec[1] for rec in trimmed]
            n = len(lats)
            if n == 0:
                continue
            w.writerow({
                "task":           task,
                "condition":      condition,
                "n_requests":     n,
                "throughput_rps": round(n / (duration - warmup_secs), 4),
                "avg_latency_ms": round(float(np.mean(lats)), 3),
                "p50_latency_ms": round(float(np.percentile(lats, 50)), 3),
                "p95_latency_ms": round(float(np.percentile(lats, 95)), 3),
                "p99_latency_ms": round(float(np.percentile(lats, 99)), 3),
                "avg_server_exec_ms": round(float(np.mean([rec[2] for rec in trimmed])), 3),
                "avg_server_proc_ms": round(float(np.mean([rec[3] for rec in trimmed])), 3),
                "avg_server_swap_ms": round(float(np.mean([rec[4] for rec in trimmed])), 3),
                "avg_server_decoder_ms": round(float(np.mean([rec[5] for rec in trimmed])), 3),
                "avg_queue_wait_plus_rpc_ms": round(float(np.mean([rec[6] for rec in trimmed])), 3),
                "avg_non_server_exec_overhead_ms": round(float(np.mean([max(0.0, rec[1] - rec[2]) for rec in trimmed])), 3),
            })

    for task, recs in records.items():
        print(f"[Save] {task}: {len(recs)} total requests -> {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition",    required=True,
                        choices=["single_nyudepth", "single_vocseg", "no_sharing", "sharing"])
    parser.add_argument("--device-url",   default="localhost:8000",
                        help="Primary device server URL")
    parser.add_argument("--device-url-2", default="localhost:8001",
                        help="Second device server URL (used only for no_sharing)")
    parser.add_argument("--backbone",     default=os.environ.get("BACKBONE", "dinobase-patch"))
    parser.add_argument("--rps",          type=float, default=float(os.environ.get("RPS", "20")))
    parser.add_argument("--duration",     type=float, default=float(os.environ.get("PHASE_DURATION", "180")))
    parser.add_argument("--warmup-secs",  type=float, default=10.0)
    parser.add_argument("--exp-dir",      default=os.environ.get("EXP_DIR", "experiments/sharing_benefit/vision/results"))
    parser.add_argument("--trace-file",   default=None,
                        help="Path to pre-generated trace JSON. If provided, replays "
                             "identical send times across runs. If not provided, generates "
                             "a fresh trace and saves it to <exp-dir>/../trace.json.")
    args = parser.parse_args()

    out_dir = (SERVING_DIR / args.exp_dir).resolve()

    print("=" * 65)
    print(f"  Vision Sharing Benefit Experiment — condition={args.condition}")
    print(f"  Backbone  : {args.backbone}")
    print(f"  RPS/task  : {args.rps}")
    print(f"  Duration  : {args.duration}s  (warmup={args.warmup_secs}s)")
    print(f"  Results   : {out_dir}")
    print("=" * 65)

    # Determine tasks and URL mapping per condition
    if args.condition == "single_nyudepth":
        tasks = ["nyudepth"]
        task_urls = {"nyudepth": args.device_url}
    elif args.condition == "single_vocseg":
        tasks = ["vocseg"]
        task_urls = {"vocseg": args.device_url}
    elif args.condition == "no_sharing":
        tasks = BOTH_TASKS
        # each task has its own server
        task_urls = {"nyudepth": args.device_url, "vocseg": args.device_url_2}
    else:  # sharing
        tasks = BOTH_TASKS
        task_urls = {"nyudepth": args.device_url, "vocseg": args.device_url}

    print(f"[INFO] Loading data for: {tasks}")
    data = build_data(tasks)

    # Deploy: for no_sharing deploy each task to its own server
    if args.condition == "no_sharing":
        asyncio.run(deploy(args.device_url,   args.backbone, ["nyudepth"]))
        asyncio.run(deploy(args.device_url_2, args.backbone, ["vocseg"]))
    else:
        asyncio.run(deploy(args.device_url, args.backbone, tasks))

    asyncio.run(asyncio.sleep(1))

    # Load or generate trace -- ensures identical send times across all conditions
    if args.trace_file:
        trace_path = Path(args.trace_file)
        if not trace_path.is_absolute():
            trace_path = (SERVING_DIR / trace_path).resolve()
        if trace_path.exists():
            send_times = load_trace(trace_path)
        else:
            print(f"[Trace] {trace_path} not found — generating and saving ...")
            send_times = generate_traces(BOTH_TASKS, args.rps, args.duration)
            save_trace(send_times, trace_path)
    else:
        # Auto-save trace alongside results so it can be reused
        auto_path = (out_dir.parent / "trace.json").resolve()
        if auto_path.exists():
            send_times = load_trace(auto_path)
        else:
            print(f"[Trace] Generating trace (seeds per task) -> {auto_path}")
            send_times = generate_traces(BOTH_TASKS, args.rps, args.duration)
            save_trace(send_times, auto_path)

    print(f"\n[Run] Starting open-loop send ({args.duration}s) ...")
    records = asyncio.run(run_open_loop(
        task_urls=task_urls,
        data=data,
        send_times={t: send_times[t] for t in tasks},
    ))

    save_results(records, out_dir, args.condition, args.duration, args.warmup_secs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
