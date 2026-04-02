#!/usr/bin/env python3
"""noisy_neighbor/llm — LLM noisy-neighbor experiment on vLLM runtime.

This mirrors the time-series noisy-neighbor flow but sends prompt-only LLM
requests into a single shared vLLM engine. It is useful as a baseline for
studying how continuous batching behaves under mixed-task pressure before
adding any fairness-aware admission layer.

Usage (from serving/):
    python experiments/noisy_neighbor/llm/run.py \
        --device-url localhost:8000 \
        --backbone qwen2.5-0.5b \
        --victim-task llm_sst2 --victim-rps 2 \
        --aggressor-task llm_ag_news \
        --aggressor-rps-phases 2,6,12 \
        --phase-durations 30 \
        --exp-dir experiments/noisy_neighbor/llm/results/vllm_cb
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

from site_manager.config import DATASET_DIR as _DATASET_DIR
from site_manager.grpc_client import EdgeRuntimeClient


TASKS: Dict[str, Dict[str, str]] = {
    "llm_sst2": {"dataset": "sst2", "metric": "accuracy"},
    "llm_ag_news": {"dataset": "ag_news", "metric": "accuracy"},
    "llm_conll2003": {"dataset": "conll2003", "metric": "token_f1"},
}

Record = Tuple[float, float, float, float, str, str]
# (send_time_s, latency_ms, device_exec_ms, queue_delay_ms, pred, true)


def _first_scalar(value):
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.reshape(-1)[0]
    if isinstance(value, (list, tuple)):
        return value[0] if value else ""
    return value


def build_data(tasks: List[str], max_samples: int = 256) -> Dict[str, Dict]:
    from fmtk.datasetloaders.ag_news import AGNewsDataset
    from fmtk.datasetloaders.conll2003 import CoNLL2003Dataset
    from fmtk.datasetloaders.sst2 import SST2Dataset

    d = _DATASET_DIR
    cfg = {"batch_size": 1, "shuffle": False}
    cache_dir = os.path.join(d, "hf_cache")
    all_loaders = {
        "llm_sst2": lambda: DataLoader(
            SST2Dataset({"cache_dir": cache_dir, "max_samples": max_samples}, {"task_type": "llm"}, "test"),
            **cfg,
        ),
        "llm_ag_news": lambda: DataLoader(
            AGNewsDataset({"cache_dir": cache_dir, "max_samples": max_samples}, {"task_type": "llm"}, "test"),
            **cfg,
        ),
        "llm_conll2003": lambda: DataLoader(
            CoNLL2003Dataset({"cache_dir": cache_dir, "max_samples": max_samples}, {"task_type": "llm"}, "test"),
            **cfg,
        ),
    }

    data: Dict[str, Dict] = {}
    for task in tasks:
        if task not in all_loaders:
            raise ValueError(f"Unknown task: {task}")
        loader = all_loaders[task]()
        samples = list(loader)
        prompts: List[str] = []
        labels: List[str] = []
        for sample in samples:
            prompt = sample.get("question", sample.get("x"))
            if prompt is None:
                raise ValueError(f"Task {task} sample missing question/x field")
            prompts.append(str(_first_scalar(prompt)))
            labels.append(str(_first_scalar(sample["y"])))
        data[task] = {"prompts": prompts, "labels": labels}
        print(f"[Data] Loaded {task}: {len(prompts)} prompts")
    return data


async def deploy_backbone_async(device_url: str, backbone: str) -> dict:
    print(f"[Deploy] Connecting to {device_url} ...")
    client = EdgeRuntimeClient(device_url)
    try:
        await client.wait_ready()
        payload = json.dumps({"backbone": backbone, "decoders": []})
        print(f"[Deploy] Sending Control(load) backbone={backbone} decoders=0 ...")
        resp = await client.control("load", payload)
        print(f"[Deploy] Control(load) returned: {resp['status']}")
        return resp
    finally:
        await client.close()


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
    print(f"[Trace] Saved {sum(len(v) for v in trace.values())} send times -> {path}")


def load_trace(path: Path) -> Dict[str, List[float]]:
    with path.open() as f:
        trace = json.load(f)
    print(f"[Trace] Loaded {sum(len(v) for v in trace.values())} send times <- {path}")
    return trace


async def run_timeseries(
    device_url: str,
    schedules: Dict[str, List[Tuple[float, float]]],
    data: Dict[str, Dict],
    req_timeout: float = 180.0,
    trace: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, List[Record]]:
    client = EdgeRuntimeClient(device_url)
    await client.wait_ready()

    records: Dict[str, List[Record]] = {t: [] for t in schedules}

    async def _fire(task: str, req_id: int, t_send_abs: float, t_start: float, sample_idx: int) -> None:
        task_data = data[task]
        prompt = task_data["prompts"][sample_idx % len(task_data["prompts"])]
        true_label = task_data["labels"][sample_idx % len(task_data["labels"])]
        try:
            response = await asyncio.wait_for(
                client.infer(
                    {
                        "req_id": req_id,
                        "task": task,
                        "question": prompt,
                    }
                ),
                timeout=req_timeout,
            )
            t_done_abs = time.time()
            device_start_s = response["start_time_ns"] / 1e9
            device_end_s = response["end_time_ns"] / 1e9
            pred = (response.get("text_output") or "").strip()
            records[task].append(
                (
                    t_send_abs - t_start,
                    (t_done_abs - t_send_abs) * 1000.0,
                    max(0.0, (device_end_s - device_start_s) * 1000.0),
                    max(0.0, (device_start_s - t_send_abs) * 1000.0),
                    pred,
                    true_label,
                )
            )
        except Exception as exc:
            print(f"[Run] {task} req_id={req_id} failed: {exc}")

    async def _task_sender_trace(task: str, send_times: List[float], req_id_offset: int, t_start: float) -> None:
        in_flight = []
        for req_id, rel_t in enumerate(send_times):
            target = t_start + rel_t
            wait = target - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            t_send = time.time()
            in_flight.append(
                asyncio.create_task(_fire(task, req_id_offset + req_id, t_send, t_start, req_id))
            )
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

    async def _task_sender_live(task: str, schedule: List[Tuple[float, float]], req_id_offset: int, t_start: float) -> None:
        req_id = req_id_offset
        sample_idx = 0
        in_flight = []
        phase_idx = 0

        while phase_idx < len(schedule):
            end_t, rps = schedule[phase_idx]
            phase_end = t_start + end_t

            if time.time() >= phase_end:
                phase_idx += 1
                continue

            if rps <= 0:
                await asyncio.sleep(max(0, phase_end - time.time()))
                phase_idx += 1
                continue

            t_send = time.time()
            in_flight.append(
                asyncio.create_task(_fire(task, req_id, t_send, t_start, sample_idx))
            )
            req_id += 1
            sample_idx += 1

            gap = np.random.exponential(1.0 / rps)
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
            _task_sender_trace(task, trace[task], i * 1_000_000, t_start)
            for i, task in enumerate(schedules)
        ]
    else:
        senders = [
            _task_sender_live(task, sched, i * 1_000_000, t_start)
            for i, (task, sched) in enumerate(schedules.items())
        ]
    await asyncio.gather(*senders, return_exceptions=True)
    await client.close()
    return records


def save_records(
    records: Dict[str, List[Record]],
    out_dir: Path,
    phase_boundaries: List[float],
    aggressor_rps_phases: List[float],
    metadata: Dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for task, recs in records.items():
        path = out_dir / f"{task}_timeseries.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "task",
                    "send_time_s",
                    "latency_ms",
                    "device_exec_ms",
                    "queue_delay_ms",
                    "phase",
                    "pred",
                    "true",
                ]
            )
            for rel_t, lat_ms, exec_ms, queue_ms, pred, true in recs:
                phase = len(phase_boundaries)
                for i, boundary in enumerate(phase_boundaries):
                    if rel_t < boundary:
                        phase = i + 1
                        break
                w.writerow(
                    [
                        task,
                        f"{rel_t:.4f}",
                        f"{lat_ms:.3f}",
                        f"{exec_ms:.3f}",
                        f"{queue_ms:.3f}",
                        phase,
                        pred,
                        true,
                    ]
                )
        print(f"[Save] {path} ({len(recs)} records)")

    meta = dict(metadata)
    meta.update(
        {
            "phase_boundaries_s": phase_boundaries,
            "aggressor_rps_phases": aggressor_rps_phases,
        }
    )
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


def _parse_float_list(s: str) -> List[float]:
    return [float(v) for v in s.split(",")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-url", default="localhost:8000")
    parser.add_argument("--backbone", default="qwen2.5-0.5b")
    parser.add_argument("--victim-task", default="llm_sst2", choices=sorted(TASKS))
    parser.add_argument("--aggressor-task", default="llm_ag_news", choices=sorted(TASKS))
    parser.add_argument("--victim-rps", type=float, default=2.0)
    parser.add_argument("--aggressor-rps-phases", default="2,6,12")
    parser.add_argument("--phase-durations", default="30")
    parser.add_argument("--policy-label", default="continuous_batching")
    parser.add_argument("--exp-dir", default=os.environ.get("EXP_DIR", "experiments/noisy_neighbor/llm/results/continuous_batching"))
    parser.add_argument("--trace-file", default=None)
    parser.add_argument("--max-samples", type=int, default=256)
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

    print("=" * 68)
    print("  noisy_neighbor/llm — vLLM continuous-batching baseline")
    print(f"  Backbone   : {args.backbone}")
    print(f"  Policy     : {args.policy_label}")
    print(f"  Victim     : {args.victim_task} @ {args.victim_rps} rps (constant)")
    print(f"  Aggressor  : {args.aggressor_task}")
    for i, (dur, rps) in enumerate(zip(phase_durations, aggressor_rps_list)):
        print(f"  Phase {i+1} ({dur:.0f}s): aggressor @ {rps} rps")
    print("=" * 68)

    tasks = [args.victim_task, args.aggressor_task]
    data = build_data(tasks, max_samples=args.max_samples)

    resp = asyncio.run(deploy_backbone_async(args.device_url, args.backbone))
    if "error" in resp.get("status", "").lower():
        print(f"[Error] Deploy failed: {resp}")
        return 1

    asyncio.run(asyncio.sleep(1))

    victim_schedule = [(total_duration, args.victim_rps)]
    aggressor_schedule = list(zip(phase_boundaries, aggressor_rps_list))
    schedules = {
        args.victim_task: victim_schedule,
        args.aggressor_task: aggressor_schedule,
    }

    trace: Optional[Dict[str, List[float]]] = None
    if args.trace_file:
        trace_path = (SERVING_DIR / args.trace_file).resolve()
        if trace_path.exists():
            trace = load_trace(trace_path)
        else:
            print(f"[Trace] {trace_path} not found - generating and saving ...")
            trace = generate_trace(schedules)
            save_trace(trace, trace_path)
    else:
        out_dir = (SERVING_DIR / args.exp_dir).resolve()
        auto_path = out_dir.parent / "trace.json"
        if auto_path.exists():
            trace = load_trace(auto_path)
        else:
            print(f"[Trace] Generating trace (seed=42) -> {auto_path}")
            trace = generate_trace(schedules)
            save_trace(trace, auto_path)

    print(f"\n[Run] Starting ({total_duration:.0f}s total) ...")
    req_timeout = max(180.0, total_duration * 3)
    records = asyncio.run(
        run_timeseries(
            args.device_url,
            schedules,
            data,
            req_timeout=req_timeout,
            trace=trace,
        )
    )

    out_dir = (SERVING_DIR / args.exp_dir).resolve()
    save_records(
        records,
        out_dir,
        phase_boundaries,
        aggressor_rps_list,
        metadata={
            "backbone": args.backbone,
            "policy_label": args.policy_label,
            "victim_task": args.victim_task,
            "victim_rps": args.victim_rps,
            "aggressor_task": args.aggressor_task,
            "runtime_type": "vllm",
        },
    )

    for task, recs in records.items():
        print(f"  [{task}] {len(recs)} requests recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
