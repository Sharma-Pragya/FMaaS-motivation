#!/usr/bin/env python3
"""Motivation Experiment #1 — LLM in-process benchmark.

Mirrors TSFM flow in experiments/motivation1/tsfm:
- task_sharing: one process-isolated local vLLM runtime per task
- deploy_sharing: one shared local vLLM runtime across tasks

No gRPC/SSH deployment path is used in this experiment runner.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List

SERVING_DIR = Path(__file__).resolve().parents[3]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import numpy as np

from experiments.motivation1.llm.server import LLMInferenceServer

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BACKBONE = "qwen2.5-0.5b"

# ---------------------------------------------------------------------------
# Task library — 10 LLM tasks
# ---------------------------------------------------------------------------

TASK_ORDER: List[str] = [
    "ag_news",
    "sst2",
    "conll2003",
    "squad",
    "cnn_dailymail",
    "flores",
    "gsm8k",
    "humaneval",
    "fever",
    "hellaswag",
]

TASK_TYPE: Dict[str, str] = {
    "ag_news": "text_classification",
    "sst2": "sentiment",
    "conll2003": "ner",
    "squad": "qa",
    "cnn_dailymail": "summarization",
    "flores": "translation",
    "gsm8k": "math_reasoning",
    "humaneval": "code_generation",
    "fever": "fact_verification",
    "hellaswag": "reading_comprehension",
}

MAX_NEW_TOKENS: Dict[str, int] = {
    "text_classification": 8,
    "sentiment": 8,
    "ner": 64,
    "qa": 64,
    "summarization": 128,
    "translation": 128,
    "math_reasoning": 256,
    "code_generation": 256,
    "fact_verification": 8,
    "reading_comprehension": 8,
}


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def build_data(max_samples: int = 50) -> Dict[str, List[str]]:
    """Return {task: [prompt_string, ...]} for each LLM task."""
    from torch.utils.data import DataLoader
    from fmtk.datasetloaders.ag_news import AGNewsDataset
    from fmtk.datasetloaders.sst2 import SST2Dataset
    from fmtk.datasetloaders.conll2003 import CoNLL2003Dataset
    from fmtk.datasetloaders.squad import SQuADDataset
    from fmtk.datasetloaders.cnn_dailymail import CNNDailyMailDataset
    from fmtk.datasetloaders.flores import FLORESDataset
    from fmtk.datasetloaders.gsm8k import GSM8KDataset
    from fmtk.datasetloaders.humaneval import HumanEvalDataset
    from fmtk.datasetloaders.fever import FEVERDataset
    from fmtk.datasetloaders.hellaswag import HellaSwagDataset

    ds_cfg = {"max_samples": max_samples}
    collate = lambda batch: {"x": [i["x"] for i in batch], "y": [i["y"] for i in batch]}

    flores_task_cfg = {
        "task_type": "translation",
        "src_lang": "fra_Latn",
        "tgt_lang": "eng_Latn",
    }

    datasets = {
        "ag_news": AGNewsDataset(ds_cfg, {"task_type": "text_classification"}, "test"),
        "sst2": SST2Dataset(ds_cfg, {"task_type": "sentiment"}, "test"),
        "conll2003": CoNLL2003Dataset(ds_cfg, {"task_type": "ner"}, "test"),
        "squad": SQuADDataset(ds_cfg, {"task_type": "qa"}, "test"),
        "cnn_dailymail": CNNDailyMailDataset(ds_cfg, {"task_type": "summarization"}, "test"),
        "flores": FLORESDataset(ds_cfg, flores_task_cfg, "test"),
        "gsm8k": GSM8KDataset(ds_cfg, {"task_type": "math_reasoning"}, "test"),
        "humaneval": HumanEvalDataset(ds_cfg, {"task_type": "code_generation"}, "test"),
        "fever": FEVERDataset(ds_cfg, {"task_type": "fact_verification"}, "test"),
        "hellaswag": HellaSwagDataset(ds_cfg, {"task_type": "reading_comprehension"}, "test"),
    }

    result = {}
    for task, ds in datasets.items():
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
        result[task] = [batch["x"][0] for batch in loader]
    return result


def homogenize_prompts(
    data: Dict[str, List[str]],
    source_task: str | None = None,
) -> Dict[str, List[str]]:
    """Optionally force all tasks to use prompts from one source task."""
    if not source_task:
        return data
    if source_task not in data:
        raise ValueError(f"prompt source task '{source_task}' not found. Available: {sorted(data.keys())}")
    source_prompts = list(data[source_task])
    return {task: source_prompts for task in data}


# ---------------------------------------------------------------------------
# Open-loop worker (Poisson arrivals)
# ---------------------------------------------------------------------------

def task_worker_open_loop(
    task: str,
    client,
    client_label: str,
    prompts: List[str],
    duration: float,
    results: List,
    req_id_start: int,
    target_rps: float = 2.0,
):
    async def _run():
        await client.wait_ready()

        latencies: List[float] = []
        in_flight: List[asyncio.Task] = []
        deadline = time.time() + duration
        req_id = req_id_start
        inter_arrival = 1.0 / target_rps
        prompt_cycle = len(prompts)

        async def _fire(rid, prompt, send_time):
            try:
                await client.infer({
                    "req_id": rid,
                    "task": task,
                    "x": np.array([0.0], dtype=np.float32),
                    "question": prompt,
                })
                latencies.append((time.time() - send_time) * 1000)
            except Exception as e:
                print(f"[Worker:{task}] req {rid} error: {e}")

        while time.time() < deadline:
            t0 = time.time()
            prompt = prompts[req_id % prompt_cycle]
            in_flight.append(asyncio.create_task(_fire(req_id, prompt, t0)))
            req_id += 1
            sleep_s = np.random.exponential(inter_arrival)
            remaining = (t0 + sleep_s) - time.time()
            if remaining > 0:
                await asyncio.sleep(remaining)

        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)
        await client.close()
        return latencies

    latencies = asyncio.run(_run())
    n = len(latencies)
    avg_lat = sum(latencies) / n if n else 0.0
    results.append({
        "task": task,
        "device_url": client_label,
        "n_requests": n,
        "throughput_rps": n / duration,
        "avg_latency_ms": avg_lat,
    })
    print(
        f"[Worker:{task}] done — {n} reqs @ target={target_rps:.1f} "
        f"actual={n/duration:.2f} req/s, avg_lat={avg_lat:.1f} ms"
    )


def run_open_loop(
    task_client_map: Dict[str, tuple],
    data: Dict[str, List[str]],
    duration: float,
    target_rps_per_task: float,
) -> List[Dict]:
    results = []
    threads = []
    for i, (task, (client, label)) in enumerate(task_client_map.items()):
        t = threading.Thread(
            target=task_worker_open_loop,
            args=(task, client, label, data[task], duration, results, i * 10000),
            kwargs={"target_rps": target_rps_per_task},
            daemon=True,
        )
        threads.append(t)

    print(
        f"\n[Benchmark:open_loop] {len(threads)} tasks @ {target_rps_per_task:.1f} req/s each "
        f"({target_rps_per_task * len(threads):.1f} req/s total) for {duration}s..."
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=duration + 120)
    return results


# ---------------------------------------------------------------------------
# Closed-loop worker (fixed concurrency)
# ---------------------------------------------------------------------------

def task_worker_closed_loop(
    task: str,
    client,
    client_label: str,
    prompts: List[str],
    duration: float,
    results: List,
    req_id_start: int,
    concurrency: int = 4,
):
    async def _run():
        await client.wait_ready()

        latencies: List[float] = []
        semaphore = asyncio.Semaphore(concurrency)
        stop_event = asyncio.Event()
        req_id = req_id_start
        deadline = time.time() + duration
        prompt_cycle = len(prompts)

        async def _one_worker():
            nonlocal req_id
            while not stop_event.is_set():
                async with semaphore:
                    if stop_event.is_set():
                        break
                    rid = req_id
                    prompt = prompts[req_id % prompt_cycle]
                    req_id += 1
                    t0 = time.time()
                    try:
                        await client.infer({
                            "req_id": rid,
                            "task": task,
                            "x": np.array([0.0], dtype=np.float32),
                            "question": prompt,
                        })
                        latencies.append((time.time() - t0) * 1000)
                    except Exception as e:
                        print(f"[Worker:{task}] req {rid} error: {e}")

        async def _timer():
            remaining = deadline - time.time()
            if remaining > 0:
                await asyncio.sleep(remaining)
            stop_event.set()

        workers = [asyncio.create_task(_one_worker()) for _ in range(concurrency)]
        timer = asyncio.create_task(_timer())
        await asyncio.gather(timer, *workers, return_exceptions=True)
        await client.close()
        return latencies

    latencies = asyncio.run(_run())
    n = len(latencies)
    avg_lat = sum(latencies) / n if n else 0.0
    results.append({
        "task": task,
        "device_url": client_label,
        "n_requests": n,
        "throughput_rps": n / duration,
        "avg_latency_ms": avg_lat,
    })
    print(
        f"[Worker:{task}] done — {n} reqs, concurrency={concurrency}, "
        f"actual={n/duration:.2f} req/s, avg_lat={avg_lat:.1f} ms"
    )


def run_closed_loop(
    task_client_map: Dict[str, tuple],
    data: Dict[str, List[str]],
    duration: float,
    concurrency_per_task: int,
) -> List[Dict]:
    results = []
    threads = []
    for i, (task, (client, label)) in enumerate(task_client_map.items()):
        t = threading.Thread(
            target=task_worker_closed_loop,
            args=(task, client, label, data[task], duration, results, i * 10000),
            kwargs={"concurrency": concurrency_per_task},
            daemon=True,
        )
        threads.append(t)

    total_concurrency = concurrency_per_task * len(threads)
    print(
        f"\n[Benchmark:closed_loop] {len(threads)} tasks × {concurrency_per_task} workers "
        f"= {total_concurrency} concurrent requests, for {duration}s..."
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=duration + 120)
    return results


# ---------------------------------------------------------------------------
# Task-sharing process isolation helpers
# ---------------------------------------------------------------------------

async def _process_open_loop(runtime, task: str, prompts: List[str], duration: float, target_rps: float) -> List[float]:
    latencies: List[float] = []
    in_flight: List[asyncio.Task] = []
    deadline = time.time() + duration
    req_id = 0
    prompt_cycle = len(prompts)
    inter_arrival = 1.0 / target_rps

    async def _fire(rid: int, prompt: str, t0: float):
        await runtime.infer(rid, prompt)
        latencies.append((time.time() - t0) * 1000)

    while time.time() < deadline:
        t0 = time.time()
        prompt = prompts[req_id % prompt_cycle]
        in_flight.append(asyncio.create_task(_fire(req_id, prompt, t0)))
        req_id += 1
        sleep_s = np.random.exponential(inter_arrival)
        remaining = (t0 + sleep_s) - time.time()
        if remaining > 0:
            await asyncio.sleep(remaining)

    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=False)
    return latencies


async def _process_closed_loop(runtime, prompts: List[str], duration: float, concurrency: int) -> List[float]:
    latencies: List[float] = []
    stop_event = asyncio.Event()
    req_id = 0
    prompt_cycle = len(prompts)
    req_lock = asyncio.Lock()
    deadline = time.time() + duration

    async def _one_worker():
        nonlocal req_id
        while not stop_event.is_set():
            async with req_lock:
                rid = req_id
                req_id += 1
            prompt = prompts[rid % prompt_cycle]
            t0 = time.time()
            await runtime.infer(rid, prompt)
            latencies.append((time.time() - t0) * 1000)

    async def _timer():
        remaining = deadline - time.time()
        if remaining > 0:
            await asyncio.sleep(remaining)
        stop_event.set()

    workers = [asyncio.create_task(_one_worker()) for _ in range(concurrency)]
    timer = asyncio.create_task(_timer())
    await asyncio.gather(timer, *workers, return_exceptions=True)
    return latencies


def _task_process_entry(
    task: str,
    prompts: List[str],
    duration: float,
    backbone: str,
    device: str,
    model_config: dict,
    benchmark_mode: str,
    target_rps: float,
    concurrency: int,
    process_index: int,
    result_queue,
    load_barrier=None,
) -> None:
    label = f"local://process-{process_index}"
    try:
        import torch
        from device.vllm_runtime import VLLMRuntime

        runtime = VLLMRuntime()
        runtime.load(backbone=backbone, decoders=[], device=device, model_config=model_config)
        dev = torch.device(device)
        load_alloc_mb = 0.0
        if dev.type == "cuda":
            try:
                torch.cuda.synchronize(dev)
            except Exception:
                pass
            load_alloc_mb = torch.cuda.memory_allocated(dev) / (1024 ** 2)
            try:
                torch.cuda.reset_peak_memory_stats(dev)
            except Exception:
                pass

        # Signal parent that this process has loaded; wait for all others to load too
        if load_barrier is not None:
            load_barrier.wait()

        if benchmark_mode == "closed_loop":
            latencies = asyncio.run(_process_closed_loop(runtime, prompts, duration, concurrency))
        else:
            latencies = asyncio.run(_process_open_loop(runtime, task, prompts, duration, target_rps))
        peak_alloc_mb = load_alloc_mb
        if dev.type == "cuda":
            try:
                torch.cuda.synchronize(dev)
            except Exception:
                pass
            peak_alloc_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

        n = len(latencies)
        avg_lat = sum(latencies) / n if n else 0.0
        result_queue.put(
            {
                "task_result": {
                    "task": task,
                    "device_url": label,
                    "n_requests": n,
                    "throughput_rps": n / duration,
                    "avg_latency_ms": avg_lat,
                },
                "deployment_record": {
                    "endpoint": label,
                    "model_config": model_config,
                    "memory": runtime.memory_stats or {},
                },
                "cuda_alloc": {
                    "load_alloc_mb": round(load_alloc_mb, 3),
                    "peak_alloc_mb": round(peak_alloc_mb, 3),
                },
                "error": None,
            }
        )
    except Exception as e:
        result_queue.put(
            {
                "task_result": {
                    "task": task,
                    "device_url": label,
                    "n_requests": 0,
                    "throughput_rps": 0.0,
                    "avg_latency_ms": 0.0,
                },
                "deployment_record": {
                    "endpoint": label,
                    "model_config": model_config,
                    "memory": {},
                },
                "cuda_alloc": {
                    "load_alloc_mb": 0.0,
                    "peak_alloc_mb": 0.0,
                },
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
        )


def _aggregate_memory_from_records(records: List[dict]) -> dict:
    total_model_gb = 0.0
    total_reserved_gb = 0.0
    total_gpu_gb = 0.0
    for rec in records:
        mem = rec.get("memory", {})
        total_model_gb += float(mem.get("model_memory_gb", 0.0))
        total_reserved_gb += float(mem.get("reserved_gb", 0.0))
        total_gpu_gb = float(mem.get("total_gpu_gb", 0.0))  # same GPU across all records
    return {
        "model_memory_mb": round(total_model_gb * 1024),
        "total_reserved_mb": round(total_reserved_gb * 1024),
        "total_gpu_gb": total_gpu_gb,
    }


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

def run_task_sharing(
    active_tasks: List[str],
    duration: float,
    result_dir: Path,
    data: Dict[str, List[str]],
    backbone: str,
    device: str,
    uniform_max_new_tokens: int,
    benchmark_mode: str = "closed_loop",
    target_rps_per_task: float = 2.0,
    concurrency_per_task: int = 4,
) -> Dict:
    """N separate process-isolated vLLM runtimes, one per task."""
    result_dir.mkdir(parents=True, exist_ok=True)
    n_tasks = len(active_tasks)

    total_gpu_gb = 14.61
    weights_gb = 0.93
    # Cap total allocation to 85% of GPU to leave headroom for concurrent activation peaks
    safe_total_gb = total_gpu_gb * 0.85
    kv_budget_gb = max(0.3, (safe_total_gb - n_tasks * weights_gb) / n_tasks)
    gpu_mem_per_instance = (weights_gb + kv_budget_gb) / total_gpu_gb
    instance_model_config = {
        "max_model_len": 256,
        "gpu_memory_utilization": gpu_mem_per_instance,  
        "max_new_tokens": int(uniform_max_new_tokens),
    }

    print(f"[task_sharing] {n_tasks} process-isolated vLLM runtimes on {device}:")
    for i, task in enumerate(active_tasks):
        print(f"  {task:20s} → local://process-{i}")
    print(
        f"  gpu_memory_utilization per instance: {gpu_mem_per_instance:.3f} "
        f"({kv_budget_gb:.2f}GB KV), max_model_len=256"
    )
    print(f"  uniform max_new_tokens: {int(uniform_max_new_tokens)}")

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs: List[mp.Process] = []
    for i, task in enumerate(active_tasks):
        p = ctx.Process(
            target=_task_process_entry,
            args=(
                task,
                data[task],
                duration,
                backbone,
                device,
                instance_model_config,
                benchmark_mode,
                target_rps_per_task,
                concurrency_per_task,
                i,
                result_queue,
            ),
        )
        p.start()
        procs.append(p)
        time.sleep(10)

    with (result_dir / "deployment_info.json").open("w") as f:
        json.dump(
            {
                "strategy": "task_sharing",
                "backbone": backbone,
                "n_tasks": n_tasks,
                "runtime": "in_process_vllm",
                "benchmark_mode": benchmark_mode,
                "assignments": [
                    {"task": t, "url": f"local://process-{i}"}
                    for i, t in enumerate(active_tasks)
                ],
            },
            f,
            indent=2,
        )

    outputs: List[dict] = []
    try:
        for _ in active_tasks:
            outputs.append(result_queue.get(timeout=duration + 300))
    except queue.Empty:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
            else:
                print(f"[task_sharing] process pid={p.pid} exited with code {p.exitcode}")
        # raise RuntimeError("task_sharing timed out waiting for worker process results")

    for p in procs:
        p.join(timeout=2)
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)

    failures = [o for o in outputs if o.get("error")]
    if failures:
        for f in failures:
            print(f"[task_sharing] worker failure ({f['task_result']['task']}): {f['error']}")
            if f.get("traceback"):
                print(f["traceback"])
        print(f"[task_sharing] {len(failures)} worker(s) failed — continuing with {len(outputs) - len(failures)} successful")
    successful = [o for o in outputs if not o.get("error")]
    if not successful:
        raise RuntimeError("task_sharing: all workers failed, cannot produce results")

    # All task results (including failed) go to CSV with error column
    all_task_results = []
    for o in outputs:
        tr = dict(o["task_result"])
        if o.get("error"):
            tr["error"] = o["error"]
        all_task_results.append(tr)
    all_task_results_by_task = {r["task"]: r for r in all_task_results}
    task_results_for_csv = [all_task_results_by_task[t] for t in active_tasks if t in all_task_results_by_task]

    # Only successful outputs for aggregation
    task_results = [o["task_result"] for o in successful]
    deployment_records = [o["deployment_record"] for o in successful]
    _save_deployment_results(result_dir, deployment_records)
    mem_stats = _aggregate_memory_from_records(deployment_records)
    _save_task_results(result_dir, task_results_for_csv)

    total_rps = sum(r["throughput_rps"] for r in task_results)
    avg_lat = sum(r["avg_latency_ms"] for r in task_results) / len(task_results) if task_results else 0.0

    model_load_mb = sum(float(o.get("cuda_alloc", {}).get("load_alloc_mb", 0.0)) for o in outputs)
    peak_mb = sum(float(o.get("cuda_alloc", {}).get("peak_alloc_mb", 0.0)) for o in outputs)
    # Static: n_tasks instances each reserving gpu_mem_per_instance × actual GPU total
    static_peak_mb = n_tasks * gpu_mem_per_instance * mem_stats["total_gpu_gb"] * 1024

    print(
        f"  => load_mem={model_load_mb:.1f} MB peak_mem={peak_mb:.1f} MB "
        f"static_peak={static_peak_mb:.1f} MB "
        f"total_throughput={total_rps:.2f} req/s avg_lat={avg_lat:.1f} ms"
    )
    return {
        "strategy": "task_sharing",
        "backbone": backbone,
        "n_tasks": n_tasks,
        "n_instances": n_tasks,
        "gpu_memory_utilization": round(gpu_mem_per_instance, 4),
        "model_memory_mb": mem_stats["model_memory_mb"],
        "model_load_mem_mb": round(model_load_mb, 3),
        "peak_gpu_mem_mb": round(peak_mb, 3),
        "static_peak_gpu_mem_mb": round(static_peak_mb, 3),
        "throughput_rps": round(total_rps, 4),
        "avg_latency_ms": round(avg_lat, 3),
    }


def run_deploy_sharing(
    active_tasks: List[str],
    duration: float,
    result_dir: Path,
    data: Dict[str, List[str]],
    backbone: str,
    device: str,
    uniform_max_new_tokens: int,
    benchmark_mode: str = "closed_loop",
    target_rps_per_task: float = 2.0,
    concurrency_per_task: int = 4,
) -> Dict:
    """1 shared local vLLM runtime for all tasks."""
    result_dir.mkdir(parents=True, exist_ok=True)
    n_tasks = len(active_tasks)

    shared_model_config = {
        "max_model_len": 256,
        "gpu_memory_utilization": 0.85,
        "max_new_tokens": int(uniform_max_new_tokens),
    }

    print(f"[deploy_sharing] 1 in-process vLLM runtime on {device}, {n_tasks} tasks:")
    print(f"  tasks: {active_tasks}")
    print(f"  uniform max_new_tokens: {int(uniform_max_new_tokens)}")

    server = LLMInferenceServer(
        strategy="deploy_sharing",
        tasks=active_tasks,
        backbone=backbone,
        device=device,
        model_config=shared_model_config,
    )
    server.start()

    deployment_records = server.deployment_records()
    _save_deployment_results(result_dir, deployment_records)
    mem_stats = server.memory_summary()

    task_client_map = {task: (server.client(task), "local://instance-0") for task in active_tasks}

    with (result_dir / "deployment_info.json").open("w") as f:
        json.dump(
            {
                "strategy": "deploy_sharing",
                "backbone": backbone,
                "n_tasks": n_tasks,
                "runtime": "in_process_vllm",
                "benchmark_mode": benchmark_mode,
                "server": "local://instance-0",
                "tasks": active_tasks,
            },
            f,
            indent=2,
        )

    if benchmark_mode == "closed_loop":
        task_results = run_closed_loop(task_client_map, data, duration, concurrency_per_task)
    else:
        task_results = run_open_loop(task_client_map, data, duration, target_rps_per_task)

    _save_task_results(result_dir, task_results)
    total_rps = sum(r["throughput_rps"] for r in task_results)
    avg_lat = sum(r["avg_latency_ms"] for r in task_results) / len(task_results) if task_results else 0.0

    model_load_mb = server.model_load_mem_mb()
    peak_mb = server.peak_gpu_mem_mb()
    gpu_util = shared_model_config["gpu_memory_utilization"]
    total_gpu_gb = mem_stats.get("total_gpu_gb", 0.0)
    static_peak_mb = gpu_util * total_gpu_gb * 1024
    server.stop()

    print(
        f"  => load_mem={model_load_mb:.1f} MB peak_mem={peak_mb:.1f} MB "
        f"static_peak={static_peak_mb:.1f} MB "
        f"total_throughput={total_rps:.2f} req/s avg_lat={avg_lat:.1f} ms"
    )
    return {
        "strategy": "deploy_sharing",
        "backbone": backbone,
        "n_tasks": n_tasks,
        "n_instances": 1,
        "gpu_memory_utilization": gpu_util,
        "model_memory_mb": mem_stats["model_memory_mb"],
        "model_load_mem_mb": round(model_load_mb, 3),
        "peak_gpu_mem_mb": round(peak_mb, 3),
        "static_peak_gpu_mem_mb": round(static_peak_mb, 3),
        "throughput_rps": round(total_rps, 4),
        "avg_latency_ms": round(avg_lat, 3),
    }


def _save_deployment_results(result_dir: Path, records: List[dict]):
    with (result_dir / "model_deployment_results.json").open("w") as f:
        json.dump(records, f, indent=2)


def _save_task_results(result_dir: Path, task_results: List[Dict]):
    fields = ["task", "device_url", "n_requests", "throughput_rps", "avg_latency_ms", "error"]
    with (result_dir / "task_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(task_results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Motivation LLM Experiment — task_sharing vs deploy_sharing")
    parser.add_argument("--n-tasks", default=os.environ.get("N_TASKS", "10,8,6,2,1"))
    parser.add_argument("--duration", type=float, default=float(os.environ.get("PHASE_DURATION", "60")))
    parser.add_argument("--strategies", default=os.environ.get("STRATEGIES", "task_sharing,deploy_sharing"))
    parser.add_argument("--backbone", default=os.environ.get("BACKBONE", DEFAULT_BACKBONE))
    parser.add_argument("--cuda", default=os.environ.get("CUDA_DEVICE", "cuda:0"))
    parser.add_argument("--exp-dir", default=os.environ.get("EXP_DIR", "experiments/motivation1/llm/results"))
    parser.add_argument(
        "--benchmark-mode",
        default=os.environ.get("BENCHMARK_MODE", "closed_loop"),
        choices=["open_loop", "closed_loop", "total_rps"],
    )
    parser.add_argument(
        "--target-rps",
        type=float,
        default=float(os.environ.get("TARGET_RPS", "2.0")),
        help="[open_loop] Target arrival rate per task (req/s).",
    )
    parser.add_argument(
        "--total-rps",
        type=float,
        default=float(os.environ.get("TOTAL_RPS", "20.0")),
        help="[total_rps] Fixed total arrival rate across all tasks (req/s).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("CONCURRENCY", "4")),
        help="[closed_loop] Number of concurrent workers per task.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=int(os.environ.get("MAX_SAMPLES", "50")),
        help="Number of dataset samples to load per task.",
    )
    parser.add_argument(
        "--uniform-max-new-tokens",
        type=int,
        default=int(os.environ.get("UNIFORM_MAX_NEW_TOKENS", "256")),
        help="Uniform generation cap applied to all tasks.",
    )
    parser.add_argument(
        "--prompt-source-task",
        default=os.environ.get("PROMPT_SOURCE_TASK", "").strip(),
        help="If set, all tasks reuse prompts from this source task (e.g., ag_news).",
    )
    args = parser.parse_args()

    task_counts = [int(x.strip()) for x in args.n_tasks.split(",")]
    strategies_to_run = [s.strip() for s in args.strategies.split(",")]

    result_root = (SERVING_DIR / args.exp_dir).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    summary_path = result_root / "summary.csv"

    print("\n" + "═" * 65)
    print("  Motivation Experiment #1 — LLM Deploy Sharing vs Task Sharing")
    print("═" * 65)
    print(f"  Backbone        : {args.backbone} (vLLM, in-process)")
    print(f"  CUDA device     : {args.cuda}")
    print(f"  Strategies      : {strategies_to_run}")
    print(f"  N sweep         : {task_counts}")
    print(f"  Duration/run    : {args.duration}s")
    print(f"  Benchmark mode  : {args.benchmark_mode}")
    print(f"  Uniform max out : {args.uniform_max_new_tokens} tokens")
    print(f"  Prompt source   : {args.prompt_source_task or '(per-task datasets)'}")
    if args.benchmark_mode == "open_loop":
        print(f"  Target RPS/task : {args.target_rps} req/s")
    elif args.benchmark_mode == "total_rps":
        print(f"  Total RPS       : {args.total_rps} req/s")
    else:
        print(f"  Concurrency/task: {args.concurrency} workers")
    print(f"  Results         : {result_root}")
    print("═" * 65)

    rows: List[Dict] = []
    existing: set = set()
    if summary_path.exists():
        with summary_path.open() as f:
            for row in csv.DictReader(f):
                existing.add((int(row["n_tasks"]), row["strategy"]))
                rows.append(row)
        print(f"[INFO] Resuming — {len(existing)} cells already done")

    print("[INFO] Loading datasets...")
    data = build_data(max_samples=args.max_samples)
    data = homogenize_prompts(data, source_task=args.prompt_source_task or None)
    print(f"[INFO] Loaded {len(data)} LLM task datasets")

    for n_tasks in task_counts:
        active_tasks = TASK_ORDER[:n_tasks]
        for strategy in strategies_to_run:
            if strategy not in ("task_sharing", "deploy_sharing"):
                print(f"[WARN] Unknown strategy '{strategy}', skipping.")
                continue
            if (n_tasks, strategy) in existing:
                print(f"[SKIP] n_tasks={n_tasks} strategy={strategy} already done.")
                continue

            run_dir = result_root / strategy / f"{n_tasks}_tasks"
            print(f"\n{'─' * 65}")
            print(f"  strategy={strategy}  n_tasks={n_tasks}  mode={args.benchmark_mode}")
            print(f"  tasks={active_tasks}")

            if args.benchmark_mode == "total_rps":
                effective_rps = args.total_rps / n_tasks
                effective_mode = "open_loop"
                print(
                    f"  total_rps={args.total_rps} / n_tasks={n_tasks} "
                    f"→ {effective_rps:.2f} req/s per task"
                )
            else:
                effective_rps = args.target_rps
                effective_mode = args.benchmark_mode

            if strategy == "task_sharing":
                row = run_task_sharing(
                    active_tasks,
                    args.duration,
                    run_dir,
                    data,
                    backbone=args.backbone,
                    device=args.cuda,
                    uniform_max_new_tokens=args.uniform_max_new_tokens,
                    benchmark_mode=effective_mode,
                    target_rps_per_task=effective_rps,
                    concurrency_per_task=args.concurrency,
                )
            else:
                row = run_deploy_sharing(
                    active_tasks,
                    args.duration,
                    run_dir,
                    data,
                    backbone=args.backbone,
                    device=args.cuda,
                    uniform_max_new_tokens=args.uniform_max_new_tokens,
                    benchmark_mode=effective_mode,
                    target_rps_per_task=effective_rps,
                    concurrency_per_task=args.concurrency,
                )

            rows.append(row)
            existing.add((n_tasks, strategy))
            _write_summary(summary_path, rows)

    print(f"\n[INFO] Done. Summary: {summary_path}")
    _print_summary(rows)
    return 0


def _write_summary(path: Path, rows: List) -> None:
    if not rows:
        return
    fields = [
        "n_tasks",
        "strategy",
        "backbone",
        "n_instances",
        "gpu_memory_utilization",
        "model_memory_mb",
        "model_load_mem_mb",
        "peak_gpu_mem_mb",
        "static_peak_gpu_mem_mb",
        "throughput_rps",
        "avg_latency_ms",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _print_summary(rows: List) -> None:
    print("\n" + "─" * 125)
    print(
        f"  {'n_tasks':>8}  {'strategy':<20}  {'inst':>4}  {'gpu_util':>8}  "
        f"{'model_mb':>8}  {'load_mb':>8}  {'peak_mb':>8}  {'static_peak_mb':>14}  {'rps':>8}  {'avg_lat_ms':>10}"
    )
    print("─" * 125)
    for r in rows:
        print(
            f"  {int(r['n_tasks']):>8}  {r['strategy']:<20}  "
            f"{int(r.get('n_instances', 0)):>4}  "
            f"{float(r.get('gpu_memory_utilization', 0)):>8.3f}  "
            f"{int(r.get('model_memory_mb', 0)):>8}  "
            f"{float(r.get('model_load_mem_mb', 0)):>8.1f}  "
            f"{float(r.get('peak_gpu_mem_mb', 0)):>8.1f}  "
            f"{float(r.get('static_peak_gpu_mem_mb', 0)):>14.1f}  "
            f"{float(r['throughput_rps']):>8.3f}  {float(r['avg_latency_ms']):>10.1f}"
        )
    print("─" * 125)


if __name__ == "__main__":
    raise SystemExit(main())
