#!/usr/bin/env python3
"""RTVSntask/vlm/run.py — response time vs. number of tasks, on vLLM multi-LoRA.

VLM analog of RTVSntask/llm/run.py: same conditions, same trace machinery, same
result schema, but each request carries an MNIST OCR image alongside the prompt
and the backbone is Qwen2-VL-2B served by vLLM. Each "task" is a LoRA adapter
over the shared backbone. Conditions:

  single_{task}    — 1 vLLM server, that LoRA only, driven at the *aggregate*
                     rate (rps * ntasks) so total arrivals match the other arms
  no_sharing       — N vLLM servers, each a full backbone copy + 1 LoRA,
                     gpu_memory_utilization split N ways
  no_sharing_tpc   — same, each server pinned to a TPC partition (needs libsmctrl)
  sharing          — 1 vLLM server, all N LoRAs (Punica multi-LoRA)

When ntasks exceeds the adapter pool, run.sh repeats base adapters with a
``__appN`` suffix so every slot is a distinct task ID served by its own LoRA
slot. ``base_task()`` strips that suffix to resolve the on-disk adapter dir.

Driven by run.sh, which sets up the device servers and passes URLs in.

Note on request size: the OCR digits are 28x28, so an image is ~9 KB on the
wire as float32 CHW and adds only ~4 vision tokens to the prompt (64 tokens
total, measured). Nothing here needs a raised gRPC message limit or a large
max_model_len. A dataset of full-size photographs would need both.
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


FMTK_DIR = Path(os.environ.get(
    "FMTK_DIR", str(SERVING_DIR.parent.parent / "FMTK"))).resolve()
OCR_DATASET = FMTK_DIR / "dataset" / "vlm" / "ocr"

# How many distinct images each task cycles through. Kept small: the point of
# the sweep is adapter/backbone scaling, not dataset coverage, and holding the
# working set in memory keeps the sender's per-request cost off the critical
# path.
N_SAMPLES = 16

Sample = Tuple[np.ndarray, str]


def base_task(t: str) -> str:
    """Strip the __appN suffix used when the same base adapter is repeated
    across multiple task slots (e.g. 'qwenvlA__app2' -> 'qwenvlA'). Task IDs
    stay unique for results keying; only the on-disk adapter path is stripped."""
    return t.split("__app", 1)[0]


def load_ocr_samples(n: int = N_SAMPLES) -> List[Sample]:
    """Load n (image, prompt) pairs from the FMTK OCR dataset.

    Uses fmtk's VLMDataset and TASK_REGISTRY so the prompt is byte-identical to
    the one the modalities/vlm microbenchmark and the profiler used. Images come
    back as float32 CHW in [0, 1] — ToPILImage on the device side inverts that.
    """
    from fmtk.datasetloaders.vlm_dataset import VLMDataset
    from fmtk.tasks.vlm_utils import TASK_REGISTRY

    prompt = TASK_REGISTRY["ocr"]["prompt"]
    ds = VLMDataset(
        {"dataset_path": str(OCR_DATASET), "json_file": "labels.json"},
        {"task_type": "vlm", "prompt": prompt},
        split="test",
    )
    if len(ds) == 0:
        raise RuntimeError(f"OCR dataset at {OCR_DATASET} is empty")
    n = min(n, len(ds))
    samples: List[Sample] = []
    for i in range(n):
        rec = ds[i]
        samples.append((np.ascontiguousarray(rec["x"].numpy(), dtype=np.float32),
                        rec["question"]))
    print(f"[Data] OCR: {len(samples)} sample(s) of {len(ds)}, "
          f"image shape={samples[0][0].shape}")
    return samples


def build_samples(tasks: List[str]) -> Dict[str, List[Sample]]:
    """Every task draws from the same image pool. Tasks differ by adapter, not
    by input distribution — the same choice the LLM version makes with prompts."""
    shared = load_ocr_samples()
    return {t: shared for t in tasks}


# ---------------------------------------------------------------------------
# Trace generation (open-loop Poisson, per task)
# ---------------------------------------------------------------------------

def generate_traces(tasks: List[str], rps: float, duration: float,
                    seed_base: int = 42) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for i, task in enumerate(tasks):
        rng = np.random.default_rng(seed_base + i)
        times, t = [], 0.0
        while t < duration:
            times.append(t)
            t += rng.exponential(1.0 / rps)
        out[task] = times
    return out


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
                 adapter_paths: Dict[str, str],
                 timeout_s: float = 1800.0) -> None:
    """Load the backbone + LoRAs on one device server.

    The default gRPC control deadline (120s) is far too short here: building
    the vLLM engine reads the backbone weights off disk, which on a slow or
    networked filesystem takes many minutes. Pass a deadline sized to the
    load, not to a normal control call.
    """
    # 'lora_vlm' is the r=16 / q_proj,v_proj entry in device/config.py ADAPTERS,
    # matching what make_loras.py wrote. On the vLLM path the peft config is
    # only used to pick the entry — Pipeline.add_adapter routes to
    # register_adapter and hands vLLM the directory, never wrapping the weights.
    decoders = [
        {"task": t, "adapter": "lora_vlm", "path": adapter_paths.get(t, base_task(t))}
        for t in tasks
    ]
    client = EdgeRuntimeClient(device_url)
    try:
        await client.wait_ready()
        payload = json.dumps({"backbone": backbone, "decoders": decoders})
        print(f"[Deploy] {device_url}  backbone={backbone}  tasks={tasks}")
        t0 = time.time()
        resp = await client.control("load", payload, timeout_s=timeout_s)
        print(f"[Deploy] {device_url}  status={resp['status']}  "
              f"({time.time() - t0:.1f}s)")
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Open-loop sender
# ---------------------------------------------------------------------------

Record = Tuple[float, float, float, float, float, float, float, float, int]


async def run_open_loop(
    task_urls: Dict[str, str],
    send_times: Dict[str, List[float]],
    samples: Dict[str, List[Sample]],
    req_timeout: float = 120.0,
) -> Dict[str, List[Record]]:
    unique_urls = list(set(task_urls.values()))
    clients: Dict[str, EdgeRuntimeClient] = {}
    for url in unique_urls:
        c = EdgeRuntimeClient(url)
        await c.wait_ready()
        clients[url] = c

    task_stub = {task: clients[url]._stub for task, url in task_urls.items()}
    records: Dict[str, List[Record]] = {t: [] for t in task_urls}
    # Requests can fail en masse when the engine dies on its background loop
    # (OOM, bad adapter). Count them so an all-failed run is loud, not silent.
    errors: Dict[str, int] = {}

    async def _fire(task: str, req_id: int, t_start: float) -> None:
        # Encoded per request, as in the LLM version. A shared pre-encoded proto
        # cannot be reused here: with N in-flight requests per task, two would
        # alias the same object and race on req_id. Serialising a 28x28 image is
        # a ~9 KB memcpy, immaterial against a VLM request.
        img, question = samples[task][req_id % len(samples[task])]
        proto = encode_infer_request(task=task, x=img, question=question)
        proto.req_id = req_id
        try:
            t_send_abs = time.time()
            response = await asyncio.wait_for(task_stub[task].Infer(proto), timeout=req_timeout)
            t_done_abs = time.time()
            if response.status and response.status != "ok":
                errors[f"status:{response.status}"] = \
                    errors.get(f"status:{response.status}", 0) + 1
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
        except Exception as exc:
            key = type(exc).__name__
            msg = str(exc).split("\n", 1)[0][:120]
            errors[f"{key}: {msg}"] = errors.get(f"{key}: {msg}", 0) + 1

    async def _sender(task: str, times: List[float], req_id_offset: int, t_start: float) -> None:
        in_flight = []
        for req_id, rel_t in enumerate(times):
            target = t_start + rel_t
            wait = target - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            in_flight.append(asyncio.create_task(
                _fire(task, req_id_offset + req_id, t_start)
            ))
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

    t_start = time.time()
    senders = [
        _sender(t, send_times[t], i * 1_000_000, t_start)
        for i, t in enumerate(task_urls)
    ]
    await asyncio.gather(*senders, return_exceptions=True)

    for c in clients.values():
        await c.close()
    if errors:
        total = sum(errors.values())
        print(f"\n[Run] {total} request(s) failed:")
        for k, v in sorted(errors.items(), key=lambda kv: -kv[1])[:5]:
            print(f"        {v:6d} x  {k}")
    return records


# ---------------------------------------------------------------------------
# Save results (schema matches sharing_benefit/tpc/run.py)
# ---------------------------------------------------------------------------

def save_results(records: Dict[str, List[Record]], out_dir: Path, condition: str,
                 duration: float, warmup_secs: float = 10.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "latencies.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "task", "condition", "elapsed_sec", "latency_ms",
            "server_exec_ms", "server_proc_ms", "server_swap_ms",
            "server_decoder_ms", "queue_wait_plus_rpc_ms",
            "client_pre_rpc_ms", "non_server_exec_overhead_ms", "server_start_ns",
        ])
        for task, recs in records.items():
            for rel_t, lat, exec_ms, proc_ms, swap_ms, dec_ms, queue_ms, pre_rpc_ms, start_ns in recs:
                w.writerow([
                    task, condition,
                    round(rel_t, 4), round(lat, 4),
                    round(exec_ms, 4), round(proc_ms, 4), round(swap_ms, 4),
                    round(dec_ms, 4), round(queue_ms, 4), round(pre_rpc_ms, 4),
                    round(max(0.0, lat - exec_ms), 4),
                    start_ns,
                ])

    with (out_dir / "task_results.csv").open("w", newline="") as f:
        fields = [
            "task", "condition", "n_requests", "throughput_rps",
            "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
            "avg_server_exec_ms", "avg_server_proc_ms", "avg_server_swap_ms",
            "avg_server_decoder_ms", "avg_queue_wait_plus_rpc_ms",
            "avg_client_pre_rpc_ms", "avg_non_server_exec_overhead_ms",
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
                "throughput_rps": round(n / max(duration - warmup_secs, 1.0), 4),
                "avg_latency_ms": round(float(np.mean(lats)), 3),
                "p50_latency_ms": round(float(np.percentile(lats, 50)), 3),
                "p95_latency_ms": round(float(np.percentile(lats, 95)), 3),
                "p99_latency_ms": round(float(np.percentile(lats, 99)), 3),
                "avg_server_exec_ms":    round(float(np.mean([r[2] for r in trimmed])), 3),
                "avg_server_proc_ms":    round(float(np.mean([r[3] for r in trimmed])), 3),
                "avg_server_swap_ms":    round(float(np.mean([r[4] for r in trimmed])), 3),
                "avg_server_decoder_ms": round(float(np.mean([r[5] for r in trimmed])), 3),
                "avg_queue_wait_plus_rpc_ms": round(float(np.mean([r[6] for r in trimmed])), 3),
                "avg_client_pre_rpc_ms": round(float(np.mean([r[7] for r in trimmed])), 3),
                "avg_non_server_exec_overhead_ms": round(
                    float(np.mean([max(0.0, r[1] - r[2]) for r in trimmed])), 3),
            })

    for task, recs in records.items():
        print(f"[Save] {task}: {len(recs)} total requests -> {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True,
                        help="Comma-separated task slots (e.g. qwenvlA,qwenvlB,qwenvlA__app2)")
    parser.add_argument("--condition", required=True,
                        help="single_{task} | no_sharing | no_sharing_mps | no_sharing_tpc | sharing")
    parser.add_argument("--device-urls", required=True,
                        help="Comma-separated device URLs (one per task for no_sharing*, "
                             "one for single/sharing)")
    parser.add_argument("--backbone",      default=os.environ.get("BACKBONE", "qwen2-vl-2b"))
    parser.add_argument("--rps",           type=float, default=2.0,
                        help="Per-task arrival rate (Poisson).")
    parser.add_argument("--duration",      type=float, default=120.0)
    parser.add_argument("--warmup-secs",   type=float, default=10.0)
    parser.add_argument("--exp-dir",       required=True)
    parser.add_argument("--trace-file",    default=None,
                        help="Shared trace path (so all conditions see identical arrivals).")
    parser.add_argument("--deploy-timeout", type=float,
                        default=float(os.environ.get("DEPLOY_TIMEOUT", 1800)),
                        help="gRPC deadline for the 'load' control call. vLLM engine "
                             "build + weight read can take many minutes on slow disks.")
    parser.add_argument("--adapter-paths", default=None,
                        help="Optional 'task1=dir1,task2=dir2' override; defaults to the task name with any __appN suffix stripped.")
    args = parser.parse_args()

    all_tasks   = [t.strip() for t in args.tasks.split(",") if t.strip()]
    device_urls = [u.strip() for u in args.device_urls.split(",") if u.strip()]

    adapter_paths: Dict[str, str] = {t: base_task(t) for t in all_tasks}
    if args.adapter_paths:
        for pair in args.adapter_paths.split(","):
            k, v = pair.split("=", 1)
            adapter_paths[k.strip()] = v.strip()

    # Resolve active tasks + url mapping
    if args.condition.startswith("single_"):
        task = args.condition[len("single_"):]
        if task not in all_tasks:
            print(f"ERROR: task '{task}' not in tasks list {all_tasks}", file=sys.stderr)
            return 1
        tasks = [task]
        task_urls = {task: device_urls[0]}
    elif args.condition in ("no_sharing", "no_sharing_mps", "no_sharing_tpc"):
        if len(device_urls) < len(all_tasks):
            print(f"ERROR: need {len(all_tasks)} URLs for {args.condition}, got {len(device_urls)}",
                  file=sys.stderr)
            return 1
        tasks = all_tasks
        task_urls = {t: device_urls[i] for i, t in enumerate(all_tasks)}
    elif args.condition == "sharing":
        tasks = all_tasks
        task_urls = {t: device_urls[0] for t in all_tasks}
    else:
        print(f"ERROR: unknown condition '{args.condition}'", file=sys.stderr)
        return 1

    out_dir = (SERVING_DIR / args.exp_dir).resolve()

    print("=" * 65)
    print(f"  RTVSntask/vlm — condition={args.condition}")
    print(f"  Backbone   : {args.backbone}")
    print(f"  Tasks      : {tasks}")
    rate_label = "Aggregate" if args.condition.startswith("single_") else "Per-task"
    print(f"  {rate_label:<10} : {args.rps} rps  ({args.duration}s, warmup={args.warmup_secs}s)")
    print(f"  Results    : {out_dir}")
    print("=" * 65)

    # Trace: per-task Poisson arrivals at args.rps; shared across conditions
    if args.trace_file:
        trace_path = Path(args.trace_file)
        if not trace_path.is_absolute():
            trace_path = (SERVING_DIR / trace_path).resolve()
    else:
        trace_path = out_dir.parent / "trace.json"

    if trace_path.exists():
        send_times = load_trace(trace_path)
        missing = [t for t in tasks if t not in send_times]
        if missing:
            print(f"[Trace] Generating missing tasks: {missing}")
            extra = generate_traces(missing, args.rps, args.duration)
            send_times.update(extra)
            save_trace(send_times, trace_path)
    else:
        print(f"[Trace] Generating -> {trace_path}")
        send_times = generate_traces(tasks, args.rps, args.duration)
        save_trace(send_times, trace_path)

    send_times = {t: send_times[t] for t in tasks}
    samples = build_samples(tasks)

    # Deploy: per-server for no_sharing*, single shared backbone for sharing/single
    try:
        if args.condition in ("no_sharing", "no_sharing_mps", "no_sharing_tpc"):
            for t, url in task_urls.items():
                asyncio.run(deploy(url, args.backbone, [t], adapter_paths,
                                   timeout_s=args.deploy_timeout))
        else:
            asyncio.run(deploy(device_urls[0], args.backbone, tasks, adapter_paths,
                               timeout_s=args.deploy_timeout))
    except Exception as exc:
        print(f"ERROR: deploy failed after {args.deploy_timeout}s: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("       If this is DEADLINE_EXCEEDED the server was probably still "
              "loading weights — raise DEPLOY_TIMEOUT and retry.", file=sys.stderr)
        return 1

    asyncio.run(asyncio.sleep(1))

    req_timeout = max(120.0, args.duration * 2)
    print(f"\n[Run] Open-loop send for {args.duration}s ...")
    records = asyncio.run(run_open_loop(task_urls, send_times, samples,
                                        req_timeout=req_timeout))

    save_results(records, out_dir, args.condition, args.duration, args.warmup_secs)
    for task, recs in records.items():
        print(f"  [{task}] {len(recs)} requests recorded")

    # Zero successful requests means the condition did not actually run —
    # typically the vLLM engine died on its background loop (OOM / bad adapter)
    # after the server had already started. Report it as a failure so run.sh
    # records a failure.json instead of leaving an empty, silently-dropped point.
    total_records = sum(len(r) for r in records.values())
    if total_records == 0:
        print(f"ERROR: {args.condition} recorded 0 requests across {len(tasks)} task(s) "
              f"in {args.duration}s — treating as a failed condition. Check the device "
              f"log for the engine error.", file=sys.stderr)
        return 2
    empty = [t for t, r in records.items() if not r]
    if empty:
        print(f"WARNING: {len(empty)} of {len(tasks)} task(s) recorded no requests: "
              f"{empty}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
