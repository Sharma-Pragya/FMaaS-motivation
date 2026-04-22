#!/usr/bin/env python3
"""batch_size_vs_rps/run.py — Measure observed batch size vs. request rate.

Sends ecgclass requests at a fixed Poisson arrival rate and records the
observed batch size for each processed batch.  All requests belonging to
the same batch share the same start_time_ns in the response, so we infer
batch membership from that field.

Usage (called by run.sh):
    python experiments/batch_size_vs_rps/run.py \
        --device-url localhost:8000 \
        --backbone   momentbase \
        --rps        10 \
        --duration   60 \
        --batch-wait-ms 50 \
        --exp-dir    experiments/batch_size_vs_rps/results/wait_50/rps_10
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import json
import os
import socket
import sys
import time
from collections import defaultdict
from itertools import count
from pathlib import Path

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import numpy as np
from torch.utils.data import DataLoader

from site_manager.grpc_client import EdgeRuntimeClient
from site_manager.config import DATASET_DIR as _DATASET_DIR

TASK = "ecgclass"
TASK_TYPE = "classification"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_data() -> dict:
    from fmtk.datasetloaders.ecg5000 import ECG5000Dataset

    loader = DataLoader(
        ECG5000Dataset(
            {"dataset_path": f"{_DATASET_DIR}/ECG5000"},
            {"task_type": "classification"},
            "test",
        ),
        batch_size=1,
        shuffle=False,
    )
    batch = next(iter(loader))
    data = {
        "x":    batch["x"].numpy().astype(np.float32),
        "mask": batch["mask"].numpy().astype(np.float32) if "mask" in batch else None,
    }
    print(f"[Data] Loaded {TASK}: x.shape={data['x'].shape}")
    return data


# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

async def deploy(device_url: str, backbone: str) -> None:
    client = EdgeRuntimeClient(device_url)
    try:
        await client.wait_ready()
        payload = json.dumps({
            "backbone": backbone,
            "decoders": [{"task": TASK, "type": TASK_TYPE, "path": f"{TASK}_{backbone}_mlp"}],
        })
        print(f"[Deploy] {device_url}  backbone={backbone}  task={TASK}")
        resp = await client.control("load", payload)
        print(f"[Deploy] status={resp['status']}")
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Open-loop Poisson sender
# ---------------------------------------------------------------------------

def generate_trace(rps: float, duration: float, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    times, t = [], 0.0
    while t < duration:
        times.append(t)
        t += rng.exponential(1.0 / rps)
    return times


def generate_client_traces(
    total_rps: float,
    duration: float,
    num_clients: int,
    seed: int = 42,
) -> list[dict]:
    """Generate one independent Poisson trace per client.

    The aggregate arrival rate is total_rps. Each client gets total_rps / num_clients.
    Returns globally time-sorted events with explicit client ids.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be >= 1")
    per_client_rps = total_rps / num_clients
    if per_client_rps <= 0:
        return []

    events: list[dict] = []
    req_ids = count()
    for client_id in range(num_clients):
        client_times = generate_trace(per_client_rps, duration, seed=seed + client_id)
        for rel_t in client_times:
            events.append({
                "req_id": next(req_ids),
                "client_id": client_id,
                "send_time_rel": rel_t,
            })
    events.sort(key=lambda event: (event["send_time_rel"], event["req_id"]))
    return events


async def run_open_loop(
    device_url: str,
    data: dict,
    events: list[dict],
    req_timeout: float = 60.0,
) -> list[dict]:
    """Send requests at pre-generated event times; return per-request records."""
    client = EdgeRuntimeClient(device_url)
    await client.wait_ready()

    records: list[dict] = []

    async def _fire(req_id: int, client_id: int, t_send_abs: float, t_start: float) -> None:
        try:
            resp = await asyncio.wait_for(
                client.infer({
                    "req_id": req_id,
                    "task":   TASK,
                    "x":      data["x"],
                    "mask":   data.get("mask"),
                }),
                timeout=req_timeout,
            )
            lat_ms = (time.time() - t_send_abs) * 1000
            records.append({
                "req_id":        req_id,
                "client_id":     client_id,
                "send_time_rel": round(t_send_abs - t_start, 6),
                "latency_ms":    round(lat_ms, 4),
                "start_time_ns": resp["start_time_ns"],
                "end_time_ns":   resp["end_time_ns"],
            })
        except Exception:
            pass  # drop timeouts / errors

    t_start = time.time()
    in_flight = []
    for event in events:
        req_id = event["req_id"]
        client_id = event["client_id"]
        rel_t = event["send_time_rel"]
        target = t_start + rel_t
        wait = target - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        t_send = time.time()
        in_flight.append(asyncio.create_task(_fire(req_id, client_id, t_send, t_start)))

    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=True)

    await client.close()
    return records


# ---------------------------------------------------------------------------
# Derive observed batch sizes
# ---------------------------------------------------------------------------

def compute_batch_sizes(records: list[dict], warmup_secs: float) -> list[dict]:
    """Group completed requests by start_time_ns; count = observed batch size.

    Returns one row per batch: {batch_start_ns, observed_batch_size, send_time_rel_first}.
    Only includes batches whose requests arrived after the warmup window.
    """
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        if r["send_time_rel"] > warmup_secs:
            groups[r["start_time_ns"]].append(r)

    rows = []
    for start_ns, reqs in sorted(groups.items()):
        rows.append({
            "batch_start_ns":       start_ns,
            "observed_batch_size":  len(reqs),
            "send_time_rel_first":  round(min(r["send_time_rel"] for r in reqs), 6),
        })
    return rows


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(
    records: list[dict],
    batch_rows: list[dict],
    events: list[dict],
    out_dir: Path,
    args,           # full argparse namespace — used for config.json
    t_start_iso: str,
    t_end_iso: str,
) -> None:
    rps           = args.rps
    batch_wait_ms = args.batch_wait_ms
    duration      = args.duration
    warmup_secs   = args.warmup_secs

    out_dir.mkdir(parents=True, exist_ok=True)

    # config.json — everything needed to reproduce this run
    config = {
        "experiment":            "batch_size_vs_rps",
        "task":                  TASK,
        "backbone":              args.backbone,
        "device_url":            args.device_url,
        "rps":                   rps,
        "num_clients":           args.num_clients,
        "per_client_rps":        (rps / args.num_clients) if args.num_clients else rps,
        "batch_wait_ms":         batch_wait_ms,
        "max_batch_size_server": int(os.environ.get("MAX_BATCH_SIZE", 64)),
        "scheduler_policy":      "fifo",
        "duration_s":            duration,
        "warmup_secs":           warmup_secs,
        "poisson_seed":          42,
        "t_start":               t_start_iso,
        "t_end":                 t_end_iso,
        "hostname":              socket.gethostname(),
    }
    with (out_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    # Scheduled send times — ground truth workload, includes dropped/timed-out requests
    with (out_dir / "send_times.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["req_id", "client_id", "send_time_rel"])
        for event in events:
            w.writerow([event["req_id"], event["client_id"], round(event["send_time_rel"], 6)])

    # Per-request latencies
    with (out_dir / "latencies.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["req_id", "client_id", "send_time_rel", "latency_ms",
                                           "start_time_ns", "end_time_ns"])
        w.writeheader()
        w.writerows(records)

    # Per-batch observed sizes
    with (out_dir / "batch_sizes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["batch_start_ns", "observed_batch_size",
                                           "send_time_rel_first"])
        w.writeheader()
        w.writerows(batch_rows)

    # Summary stats
    sizes = [r["observed_batch_size"] for r in batch_rows]
    lats  = [r["latency_ms"] for r in records if r["send_time_rel"] > warmup_secs]
    summary = {
        # config fields (duplicated here so summary.json is self-contained)
        "task":                TASK,
        "backbone":            args.backbone,
        "rps":                 rps,
        "batch_wait_ms":       batch_wait_ms,
        "duration_s":          duration,
        "warmup_secs":         warmup_secs,
        "t_start":             t_start_iso,
        "t_end":               t_end_iso,
        # measurements
        "n_requests_total":    len(records),
        "n_requests_measured": len(lats),
        "n_batches":           len(sizes),
        "mean_batch_size":     round(float(np.mean(sizes)),  4) if sizes else 0,
        "p50_batch_size":      round(float(np.percentile(sizes, 50)), 4) if sizes else 0,
        "p95_batch_size":      round(float(np.percentile(sizes, 95)), 4) if sizes else 0,
        "max_batch_size_obs":  int(max(sizes)) if sizes else 0,
        "avg_latency_ms":      round(float(np.mean(lats)),   3) if lats else 0,
        "p99_latency_ms":      round(float(np.percentile(lats, 99)), 3) if lats else 0,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Save] {len(records)} requests, {len(sizes)} batches → {out_dir}")
    print(f"[Save] mean_batch_size={summary['mean_batch_size']:.2f}  "
          f"p95={summary['p95_batch_size']:.2f}  max={summary['max_batch_size_obs']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-url",    default="localhost:8000")
    parser.add_argument("--backbone",      default=os.environ.get("BACKBONE", "momentbase"))
    parser.add_argument("--rps",           type=float, default=float(os.environ.get("RPS", "10")))
    parser.add_argument("--num-clients",   type=int, default=int(os.environ.get("NUM_CLIENTS", "1")))
    parser.add_argument("--duration",      type=float, default=float(os.environ.get("PHASE_DURATION", "60")))
    parser.add_argument("--warmup-secs",   type=float, default=10.0)
    parser.add_argument("--batch-wait-ms", type=float, default=float(os.environ.get("BATCH_WAIT_MS", "0")))
    parser.add_argument("--exp-dir",       default=os.environ.get("EXP_DIR",
                                           "experiments/batch_size_vs_rps/results"))
    args = parser.parse_args()

    out_dir = (SERVING_DIR / args.exp_dir).resolve()

    print("=" * 65)
    print(f"  Experiment: batch_size_vs_rps")
    print(f"  Task        : {TASK}")
    print(f"  Backbone    : {args.backbone}")
    print(f"  RPS         : {args.rps}")
    print(f"  Clients     : {args.num_clients}  ({args.rps / args.num_clients:.3f} req/s each)")
    print(f"  batch_wait  : {args.batch_wait_ms} ms")
    print(f"  Duration    : {args.duration}s  (warmup={args.warmup_secs}s)")
    print(f"  Results     : {out_dir}")
    print("=" * 65)

    data = build_data()
    asyncio.run(deploy(args.device_url, args.backbone))
    asyncio.run(asyncio.sleep(1))

    events = generate_client_traces(args.rps, args.duration, args.num_clients)
    print(f"[Run] {len(events)} requests scheduled over {args.duration}s ...")

    t_start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    records = asyncio.run(run_open_loop(args.device_url, data, events))
    t_end_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    batch_rows = compute_batch_sizes(records, args.warmup_secs)
    save_results(records, batch_rows, events, out_dir, args, t_start_iso, t_end_iso)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
