#!/usr/bin/env python3
"""fair_share/llm — Qwen2.5-3B + vLLM multi-LoRA noisy-neighbor experiment.

Two LLM tasks share a single vLLM-backed Qwen2.5-3B backbone, each bound to
its own LoRA adapter. Drives the same phase-based aggressor/victim pattern
as fair_share/tsfm/run.py and writes CSVs with the same schema so the
existing plot.py works unchanged.

Because runtime_type=vllm bypasses the device batcher (server.py:69),
scheduler_policy is ignored on the device — this serves as the *vLLM-baseline*
(vLLM's own continuous batching, no FMVisor scheduling).

Before running, generate two random LoRA adapters:
    python experiments/fair_share/llm/make_loras.py
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

from site_manager.grpc_client import EdgeRuntimeClient, encode_infer_request


# ---------------------------------------------------------------------------
# Prompt library — short prompts so generation cost dominates uniformly.
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS: List[str] = [
    "Summarize the impact of LoRA on serving cost in one sentence.",
    "Explain continuous batching to a junior engineer.",
    "Give three reasons to share a foundation model backbone across tasks.",
    "Why does GPU memory dominate the cost of LLM inference?",
    "Describe the difference between prefill and decode in vLLM.",
    "List two challenges of multi-tenant LLM serving.",
    "What is PagedAttention and why does it matter?",
    "Name a benefit of weighted fair queueing in a serving system.",
]


def build_prompts(tasks: List[str], pool: Optional[List[str]] = None) -> Dict[str, List[str]]:
    pool = pool or DEFAULT_PROMPTS
    return {t: list(pool) for t in tasks}


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
        print(f"[Deploy] Control(load) backbone={backbone} adapters={len(decoders)} ...")
        resp = await client.control("load", payload)
        print(f"[Deploy] Control(load) returned: {resp['status']}")
        return resp
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Trace generation (Poisson)
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
# Open-loop sender
# ---------------------------------------------------------------------------

Record = Tuple[float, float, float, float, float, float, float, float, int]


async def run_llm(
    task_urls: Dict[str, str],
    schedules: Dict[str, List[Tuple[float, float]]],
    prompts: Dict[str, List[str]],
    req_timeout: float = 120.0,
    trace: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, List[Record]]:
    clients: Dict[str, EdgeRuntimeClient] = {
        t: EdgeRuntimeClient(url) for t, url in task_urls.items()
    }
    for t, c in clients.items():
        print(f"[Run] Waiting for {t} server ({task_urls[t]}) ...")
        await c.wait_ready()

    task_stub = {task: clients[task]._stub for task in schedules}
    records: Dict[str, List[Record]] = {t: [] for t in schedules}

    async def _fire(task: str, req_id: int, t_start: float) -> None:
        # Build a fresh proto per request — proto objects are mutable & each
        # coroutine gets its own to avoid concurrent-mutation races.
        prompt = prompts[task][req_id % len(prompts[task])]
        proto = encode_infer_request(task=task, x=None, question=prompt)
        proto.req_id = req_id
        try:
            t_send_abs = time.time()
            response = await asyncio.wait_for(task_stub[task].Infer(proto), timeout=req_timeout)
            t_done_abs = time.time()
            if response.status and response.status != "ok":
                return
            client_lat_ms     = (t_done_abs - t_send_abs) * 1000
            server_exec_ms    = (response.end_time_ns - response.start_time_ns) / 1e6 \
                                if response.end_time_ns else 0.0
            server_proc_ms    = response.proc_time_ns / 1e6
            server_swap_ms    = response.swap_time_ns / 1e6
            server_decoder_ms = response.decoder_time_ns / 1e6
            queue_wait_plus_rpc_ms = max(0.0, (response.start_time_ns / 1e9 - t_send_abs) * 1000) \
                                     if response.start_time_ns else 0.0
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

    async def _task_sender_trace(task: str, send_times: List[float],
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
        _task_sender_trace(t, trace[t], i * 1_000_000, t_start)
        for i, t in enumerate(schedules)
    ]
    await asyncio.gather(*senders, return_exceptions=True)
    for c in clients.values():
        await c.close()
    return records


# ---------------------------------------------------------------------------
# Save (schema matches fair_share/tsfm)
# ---------------------------------------------------------------------------

def save_records(records: Dict[str, List[Record]], out_dir: Path,
                 phase_boundaries: List[float],
                 aggressor_rps_phases: List[float],
                 victim_rps_phases: List[float],
                 victim_task: str, aggressor_task: str,
                 scheduler_policy: str = "vllm-baseline",
                 warmup_secs: float = 10.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

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
                "avg_server_exec_ms":              round(float(np.mean([r[2] for r in trimmed])), 3),
                "avg_server_proc_ms":              round(float(np.mean([r[3] for r in trimmed])), 3),
                "avg_server_swap_ms":              round(float(np.mean([r[4] for r in trimmed])), 3),
                "avg_server_decoder_ms":           round(float(np.mean([r[5] for r in trimmed])), 3),
                "avg_queue_wait_plus_rpc_ms":      round(float(np.mean([r[6] for r in trimmed])), 3),
                "avg_client_pre_rpc_ms":           round(float(np.mean([r[7] for r in trimmed])), 3),
                "avg_non_server_exec_overhead_ms": round(float(np.mean([max(0.0, r[1] - r[2]) for r in trimmed])), 3),
            })

    meta = {
        "phase_boundaries_s": phase_boundaries,
        "aggressor_rps_phases": aggressor_rps_phases,
        "victim_rps_phases": victim_rps_phases,
        "victim_task": victim_task,
        "aggressor_task": aggressor_task,
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
    parser.add_argument("--victim-url",           default=None)
    parser.add_argument("--aggressor-url",        default=None)
    parser.add_argument("--backbone",             default="qwen2.5-1.5b")
    parser.add_argument("--victim-task",          default="qwenA")
    parser.add_argument("--aggressor-task",       default="qwenB")
    parser.add_argument("--victim-adapter-path",  default=None,
                        help="Adapter dir name under models/llm/finetuned/. Default = victim-task.")
    parser.add_argument("--aggressor-adapter-path", default=None,
                        help="Adapter dir name under models/llm/finetuned/. Default = aggressor-task.")
    parser.add_argument("--victim-rps",           type=float, default=2.0)
    parser.add_argument("--victim-rps-phases",    default=None)
    parser.add_argument("--aggressor-rps-phases", default="2,4,8,12")
    parser.add_argument("--phase-durations",      default="30")
    parser.add_argument("--scheduler-policy",     default="vllm-baseline",
                        help="Label written into latencies.csv (vLLM baseline = no scheduler).")
    parser.add_argument("--exp-dir",              default=os.environ.get(
                        "EXP_DIR", "experiments/fair_share/llm/results/vllm_baseline"))
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
        raise ValueError(f"--phase-durations has {len(raw_durations)} entries but "
                         f"--aggressor-rps-phases has {num_phases}.")

    phase_boundaries: List[float] = []
    t = 0.0
    for d in phase_durations:
        t += d
        phase_boundaries.append(t)
    total_duration = phase_boundaries[-1]

    victim_adapter    = args.victim_adapter_path    or args.victim_task
    aggressor_adapter = args.aggressor_adapter_path or args.aggressor_task

    print("=" * 65)
    print(f"  fair_share/llm — vLLM multi-LoRA, {num_phases}-phase experiment")
    print(f"  Backbone   : {args.backbone}")
    print(f"  Victim     : {args.victim_task} @ {args.victim_rps} rps (constant)  adapter={victim_adapter}")
    print(f"  Aggressor  : {args.aggressor_task}  adapter={aggressor_adapter}")
    for i, (dur, rps) in enumerate(zip(phase_durations, aggressor_rps_list)):
        print(f"  Phase {i+1} ({dur:.0f}s): aggressor @ {rps} rps")
    print(f"  Label      : {args.scheduler_policy}")
    print("=" * 65)

    tasks = [args.victim_task, args.aggressor_task]
    prompts = build_prompts(tasks)

    # In the shared (single-server) case we issue a single load that registers
    # both adapters under the same backbone. With separate URLs we deploy each
    # task to its own server (each with its own adapter).
    if victim_url == aggressor_url:
        decoders = [
            {"task": args.victim_task,    "adapter": "lora", "path": victim_adapter},
            {"task": args.aggressor_task, "adapter": "lora", "path": aggressor_adapter},
        ]
        resp = asyncio.run(deploy_backbone_async(victim_url, args.backbone, decoders))
        if "error" in resp.get("status", "").lower():
            print(f"[Error] Deploy failed: {resp}")
            return 1
    else:
        for task, url, adapter_path in (
            (args.victim_task,    victim_url,    victim_adapter),
            (args.aggressor_task, aggressor_url, aggressor_adapter),
        ):
            decoder = [{"task": task, "adapter": "lora", "path": adapter_path}]
            resp = asyncio.run(deploy_backbone_async(url, args.backbone, decoder))
            if "error" in resp.get("status", "").lower():
                print(f"[Error] Deploy failed for {task} on {url}: {resp}")
                return 1

    asyncio.run(asyncio.sleep(1))

    if args.victim_rps_phases is not None:
        victim_rps_list = _parse_float_list(args.victim_rps_phases)
        if len(victim_rps_list) != num_phases:
            raise ValueError(f"--victim-rps-phases has {len(victim_rps_list)} entries but "
                             f"--aggressor-rps-phases has {num_phases}.")
        victim_schedule = list(zip(phase_boundaries, victim_rps_list))
    else:
        victim_rps_list = [args.victim_rps] * num_phases
        victim_schedule = [(total_duration, args.victim_rps)]
    aggressor_schedule = list(zip(phase_boundaries, aggressor_rps_list))

    schedules = {
        args.victim_task:    victim_schedule,
        args.aggressor_task: aggressor_schedule,
    }

    trace: Optional[Dict[str, List[float]]] = None
    out_dir = (SERVING_DIR / args.exp_dir).resolve()
    if args.trace_file:
        trace_path = (SERVING_DIR / args.trace_file).resolve()
    else:
        trace_path = out_dir.parent / "trace.json"
    if trace_path.exists():
        trace = load_trace(trace_path)
    else:
        print(f"[Trace] Generating trace (seed=42) → {trace_path}")
        trace = generate_trace(schedules)
        save_trace(trace, trace_path)

    print(f"\n[Run] Starting ({total_duration:.0f}s total) ...")
    print(f"  {args.victim_task}    → {victim_url}")
    print(f"  {args.aggressor_task} → {aggressor_url}")
    req_timeout = max(120.0, total_duration * 2)
    records = asyncio.run(run_llm(
        task_urls, schedules, prompts, req_timeout=req_timeout, trace=trace,
    ))

    save_records(
        records, out_dir, phase_boundaries, aggressor_rps_list, victim_rps_list,
        victim_task=args.victim_task, aggressor_task=args.aggressor_task,
        scheduler_policy=args.scheduler_policy,
    )

    for task, recs in records.items():
        print(f"  [{task}] {len(recs)} requests recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
