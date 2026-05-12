#!/usr/bin/env python3
"""Single-task max-throughput calibration client.

Hammers one task on one device URL at a high open-loop offered rate for
DURATION seconds, then reports delivered throughput. Used to measure how
TPC count maps to throughput for a single workload.

Output: appends one CSV row to --out-csv:
    task,tpc_count,duration_s,offered_rps,delivered_rps
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path
from typing import List

SERVING_DIR = Path(__file__).resolve().parents[3]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import numpy as np
from torch.utils.data import DataLoader

from site_manager.grpc_client import EdgeRuntimeClient, encode_infer_request
from site_manager.config import DATASET_DIR as _DATASET_DIR

TASK_TYPES = {
    "ecgclass":     "classification",
    "gestureclass": "classification",
}


def build_data(task: str):
    from fmtk.datasetloaders.ecg5000 import ECG5000Dataset
    from fmtk.datasetloaders.uwavegesture import UWaveGestureLibraryALLDataset
    d = _DATASET_DIR
    cfg = {"batch_size": 1, "shuffle": False}
    loaders = {
        "ecgclass":     lambda: DataLoader(ECG5000Dataset({"dataset_path": f"{d}/ECG5000"}, {"task_type": "classification"}, "test"), **cfg),
        "gestureclass": lambda: DataLoader(UWaveGestureLibraryALLDataset({"dataset_path": f"{d}/UWaveGestureLibraryAll", "seq_len": 512}, {"task_type": "classification"}, "test"), **cfg),
    }
    batch = next(iter(loaders[task]()))
    return {
        "x":    batch["x"].numpy().astype(np.float32),
        "mask": batch.get("mask").numpy().astype(np.float32) if batch.get("mask") is not None else None,
    }


async def deploy(url: str, backbone: str, task: str):
    client = EdgeRuntimeClient(url)
    try:
        await client.wait_ready()
        decoders = [{"task": task, "type": TASK_TYPES[task],
                     "path": f"{task}_{backbone}_mlp"}]
        payload = json.dumps({"backbone": backbone, "decoders": decoders})
        resp = await client.control("load", payload)
        if "error" in resp.get("status", "").lower():
            raise RuntimeError(f"deploy failed: {resp}")
    finally:
        await client.close()


def gen_send_times(rps: float, duration: float, seed: int = 42) -> List[float]:
    rng = np.random.default_rng(seed)
    t, sends = 0.0, []
    while t < duration:
        sends.append(t)
        t += rng.exponential(1.0 / rps)
    return sends


async def run(url: str, task: str, data: dict,
              offered_rps: float, duration: float,
              req_timeout: float) -> int:
    client = EdgeRuntimeClient(url)
    await client.wait_ready()
    proto_template = encode_infer_request(task=task, x=data["x"], mask=data.get("mask"))
    stub = client._stub

    n_done = 0

    async def fire(req_id: int):
        nonlocal n_done
        proto = proto_template
        proto.req_id = req_id
        try:
            resp = await asyncio.wait_for(stub.Infer(proto), timeout=req_timeout)
            if not resp.status or resp.status == "ok":
                n_done += 1
        except Exception:
            pass

    sends = gen_send_times(offered_rps, duration)
    t_start = time.time()
    in_flight = []
    for i, rel_t in enumerate(sends):
        wait = (t_start + rel_t) - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        in_flight.append(asyncio.create_task(fire(i)))
    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=True)
    await client.close()
    return n_done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",         default="localhost:8000")
    ap.add_argument("--backbone",    default="momentlarge")
    ap.add_argument("--task",        required=True)
    ap.add_argument("--tpc-count",   type=int, required=True,
                    help="Recorded into the CSV; not enforced here (set when starting device).")
    ap.add_argument("--offered-rps", type=float, default=200.0)
    ap.add_argument("--duration",    type=float, default=10.0)
    ap.add_argument("--warmup-secs", type=float, default=2.0)
    ap.add_argument("--out-csv",     required=True)
    args = ap.parse_args()

    print(f"[Calib] task={args.task} tpcs={args.tpc_count} offered={args.offered_rps} dur={args.duration}s")
    data = build_data(args.task)
    asyncio.run(deploy(args.url, args.backbone, args.task))
    asyncio.run(asyncio.sleep(1))

    req_timeout = max(60.0, args.duration * 2)
    n = asyncio.run(run(args.url, args.task, data,
                        args.offered_rps, args.duration, req_timeout))
    delivered = n / max(args.duration - args.warmup_secs, 1.0)
    # Above is approximate (counts all completions, divides by post-warmup window).
    # Good enough for saturated calibration where rate is steady.
    print(f"[Calib] completions={n} → ~{delivered:.2f} RPS")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["task", "tpc_count", "duration_s", "offered_rps", "delivered_rps"])
        w.writerow([args.task, args.tpc_count, args.duration,
                    args.offered_rps, round(delivered, 4)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
