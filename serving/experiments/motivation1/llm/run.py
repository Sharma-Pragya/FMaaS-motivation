#!/usr/bin/env python3
"""Motivation Experiment — LLM: Task Sharing vs. Deploy Sharing with vLLM.

Simple closed-loop benchmark. No orchestrator or site_manager involved.

Strategies
----------
task_sharing   : N qwen2.5-0.5b vLLM servers, one per task (each on a separate port).
                 Each task gets its own isolated model instance — no sharing.

deploy_sharing : 1 qwen2.5-0.5b vLLM server for all tasks.
                 All tasks share one backbone; vLLM continuous batching
                 amortises the model across concurrent requests.

The two strategies differ in GPU memory footprint and throughput:
- task_sharing  : N × model memory, no cross-task batching.
- deploy_sharing: 1 × model memory, cross-task continuous batching.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

FMTK_SRC = Path("/project/pi_shenoy_umass_edu/hshastri/FMTK/src")
if str(FMTK_SRC) not in sys.path:
    sys.path.insert(0, str(FMTK_SRC))

import numpy as np

from site_manager.deployment_handler import shutdown_devices
from site_manager.grpc_client import EdgeRuntimeClient

# ---------------------------------------------------------------------------
# Hardware layout
# ---------------------------------------------------------------------------

DEVICE_HOST = "10.100.20.48"
CUDA_DEVICE  = "cuda:0"
BASE_PORT    = 8000   # task_sharing: ports 8000..8009; deploy_sharing: 8000 only

BACKBONE     = "qwen2.5-0.5b"

# ---------------------------------------------------------------------------
# Task library — 10 LLM tasks
# ---------------------------------------------------------------------------

TASK_ORDER: List[str] = [
    "ag_news",        # text classification
    "sst2",           # sentiment analysis
    "conll2003",      # named entity recognition
    "squad",          # question answering
    "cnn_dailymail",  # summarization
    "flores",         # translation
    "gsm8k",          # math reasoning
    "humaneval",      # code generation
    "fever",          # fact verification
    "hellaswag",      # reading comprehension
]

TASK_TYPE: Dict[str, str] = {
    "ag_news":       "text_classification",
    "sst2":          "sentiment",
    "conll2003":     "ner",
    "squad":         "qa",
    "cnn_dailymail": "summarization",
    "flores":        "translation",
    "gsm8k":         "math_reasoning",
    "humaneval":     "code_generation",
    "fever":         "fact_verification",
    "hellaswag":     "reading_comprehension",
}

# max_new_tokens per task type (generous upper bound for benchmarking)
MAX_NEW_TOKENS: Dict[str, int] = {
    "text_classification": 8,
    "sentiment":           8,
    "ner":                 64,
    "qa":                  64,
    "summarization":       128,
    "translation":         128,
    "math_reasoning":      256,
    "code_generation":     256,
    "fact_verification":   8,
    "reading_comprehension": 8,
}

# ---------------------------------------------------------------------------
# Dataset loader — returns one sample prompt per task
# ---------------------------------------------------------------------------

def build_data(max_samples: int = 50) -> Dict[str, List[str]]:
    """Return {task: [prompt_string, ...]} for each LLM task."""
    from torch.utils.data import DataLoader
    from fmtk.datasetloaders.ag_news      import AGNewsDataset
    from fmtk.datasetloaders.sst2         import SST2Dataset
    from fmtk.datasetloaders.conll2003    import CoNLL2003Dataset
    from fmtk.datasetloaders.squad        import SQuADDataset
    from fmtk.datasetloaders.cnn_dailymail import CNNDailyMailDataset
    from fmtk.datasetloaders.flores       import FLORESDataset
    from fmtk.datasetloaders.gsm8k        import GSM8KDataset
    from fmtk.datasetloaders.humaneval    import HumanEvalDataset
    from fmtk.datasetloaders.fever        import FEVERDataset
    from fmtk.datasetloaders.hellaswag    import HellaSwagDataset

    ds_cfg = {"max_samples": max_samples}
    collate = lambda batch: {"x": [i["x"] for i in batch], "y": [i["y"] for i in batch]}

    flores_task_cfg = {
        "task_type": "translation",
        "src_lang": "fra_Latn",
        "tgt_lang": "eng_Latn",
    }

    datasets = {
        "ag_news":       AGNewsDataset(ds_cfg,       {"task_type": "text_classification"}, "test"),
        "sst2":          SST2Dataset(ds_cfg,          {"task_type": "sentiment"},           "test"),
        "conll2003":     CoNLL2003Dataset(ds_cfg,     {"task_type": "ner"},                 "test"),
        "squad":         SQuADDataset(ds_cfg,         {"task_type": "qa"},                  "test"),
        "cnn_dailymail": CNNDailyMailDataset(ds_cfg,  {"task_type": "summarization"},        "test"),
        "flores":        FLORESDataset(ds_cfg,        flores_task_cfg,                       "test"),
        "gsm8k":         GSM8KDataset(ds_cfg,         {"task_type": "math_reasoning"},       "test"),
        "humaneval":     HumanEvalDataset(ds_cfg,     {"task_type": "code_generation"},      "test"),
        "fever":         FEVERDataset(ds_cfg,         {"task_type": "fact_verification"},    "test"),
        "hellaswag":     HellaSwagDataset(ds_cfg,     {"task_type": "reading_comprehension"},"test"),
    }

    result = {}
    for task, ds in datasets.items():
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
        result[task] = [batch["x"][0] for batch in loader]
    return result


# ---------------------------------------------------------------------------
# Deployment helpers
# ---------------------------------------------------------------------------

async def _deploy_one_with_retry(
    host: str, port: int, cuda: str,
    model_config: dict | None = None,
    retries: int = 5, retry_delay: float = 5.0,
) -> dict:
    """SSH-launch a vLLM device server and load the model."""
    from site_manager.deployment_handler import _parse_url, _ssh_start_server
    from site_manager.config import timeseries_env, username

    grpc_url = f"{host}:{port}"
    server_cmd = (
        f"python -u device/main.py --port {port} "
        f"--runtime-type vllm "
        f"--cuda {cuda} "
    )
    log_path = f"./device/logs/{host}_{cuda.replace(':', '')}_port{port}_{BACKBONE}.log"

    await _ssh_start_server(host, username, timeseries_env, server_cmd, log_path)

    config_payload = json.dumps({"backbone": BACKBONE, "decoders": [], "model_config": model_config or {}})

    from site_manager.deployment_handler import _send_control
    for attempt in range(retries):
        result = await _send_control(grpc_url, "load", config_payload)
        if isinstance(result, dict) and result.get("status", "").startswith("loaded_"):
            return {"grpc_url": grpc_url, "log_path": log_path, "logger_summary": result.get("logger_summary", "")}
        wait = retry_delay * (attempt + 1)
        print(f"[Deploy] {grpc_url} load attempt {attempt+1}/{retries} failed, retrying in {wait:.0f}s...")
        await asyncio.sleep(wait)

    raise RuntimeError(f"Failed to load model on {grpc_url} after {retries} attempts")


async def deploy_servers(server_specs: List[dict], model_config: dict | None = None) -> List[dict]:
    """Deploy all specs in parallel. Returns list of {grpc_url, log_path}."""
    results = await asyncio.gather(
        *[_deploy_one_with_retry(**s, model_config=model_config) for s in server_specs],
        return_exceptions=True,
    )
    deployed = []
    for spec, r in zip(server_specs, results):
        if isinstance(r, Exception):
            raise RuntimeError(f"Deployment failed for {spec}: {r}")
        deployed.append(r)
    return deployed


async def cleanup(host: str, ports: List[int]):
    """Kill device servers on the given ports."""
    specs = [{"device": f"{host}:{p}"} for p in ports]
    await shutdown_devices(specs)
    await asyncio.sleep(15)


# ---------------------------------------------------------------------------
# Open-loop worker (Poisson arrivals)
# ---------------------------------------------------------------------------

def task_worker_open_loop(
    task: str,
    device_url: str,
    prompts: List[str],
    duration: float,
    results: List,
    req_id_start: int,
    target_rps: float = 2.0,
):
    async def _run():
        client = EdgeRuntimeClient(device_url)
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
        "device_url": device_url,
        "n_requests": n,
        "throughput_rps": n / duration,
        "avg_latency_ms": avg_lat,
    })
    print(f"[Worker:{task}] done — {n} reqs @ target={target_rps:.1f} "
          f"actual={n/duration:.2f} req/s, avg_lat={avg_lat:.1f} ms")


def run_open_loop(
    task_device_map: Dict[str, str],
    data: Dict[str, List[str]],
    duration: float,
    target_rps_per_task: float,
) -> List[Dict]:
    results = []
    threads = []
    for i, (task, url) in enumerate(task_device_map.items()):
        t = threading.Thread(
            target=task_worker_open_loop,
            args=(task, url, data[task], duration, results, i * 10000),
            kwargs={"target_rps": target_rps_per_task},
            daemon=True,
        )
        threads.append(t)

    print(f"\n[Benchmark:open_loop] {len(threads)} tasks @ {target_rps_per_task:.1f} req/s each "
          f"({target_rps_per_task * len(threads):.1f} req/s total) for {duration}s...")
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
    device_url: str,
    prompts: List[str],
    duration: float,
    results: List,
    req_id_start: int,
    concurrency: int = 4,
):
    async def _run():
        client = EdgeRuntimeClient(device_url)
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
        "device_url": device_url,
        "n_requests": n,
        "throughput_rps": n / duration,
        "avg_latency_ms": avg_lat,
    })
    print(f"[Worker:{task}] done — {n} reqs, concurrency={concurrency}, "
          f"actual={n/duration:.2f} req/s, avg_lat={avg_lat:.1f} ms")


def run_closed_loop(
    task_device_map: Dict[str, str],
    data: Dict[str, List[str]],
    duration: float,
    concurrency_per_task: int,
) -> List[Dict]:
    results = []
    threads = []
    for i, (task, url) in enumerate(task_device_map.items()):
        t = threading.Thread(
            target=task_worker_closed_loop,
            args=(task, url, data[task], duration, results, i * 10000),
            kwargs={"concurrency": concurrency_per_task},
            daemon=True,
        )
        threads.append(t)

    total_concurrency = concurrency_per_task * len(threads)
    print(f"\n[Benchmark:closed_loop] {len(threads)} tasks × {concurrency_per_task} workers "
          f"= {total_concurrency} concurrent requests, for {duration}s...")
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=duration + 120)
    return results


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

async def run_task_sharing(
    active_tasks: List[str],
    duration: float,
    result_dir: Path,
    data: Dict[str, List[str]],
    benchmark_mode: str = "closed_loop",
    target_rps_per_task: float = 2.0,
    concurrency_per_task: int = 4,
) -> Dict:
    """N separate qwen2.5-0.5b vLLM servers, one per task."""
    result_dir.mkdir(parents=True, exist_ok=True)
    n_tasks = len(active_tasks)

    server_specs = [
        {"host": DEVICE_HOST, "port": BASE_PORT + i, "cuda": CUDA_DEVICE}
        for i in range(n_tasks)
    ]

    print(f"[task_sharing] {n_tasks} qwen2.5-0.5b servers on {CUDA_DEVICE}:")
    for i, task in enumerate(active_tasks):
        print(f"  {task:20s} → {DEVICE_HOST}:{BASE_PORT + i}")

    # Each instance shares the GPU — cap KV cache so N instances fit.
    # 14.6GB GPU, ~0.93GB weights per qwen2.5-0.5b instance.
    # Budget each instance: weights + (remaining / n_tasks) for KV cache.
    # gpu_memory_utilization is per-process: set to budget/total_gpu_gb.
    total_gpu_gb = 14.61
    weights_gb = 0.93
    kv_budget_gb = max(0.3, (total_gpu_gb - n_tasks * weights_gb) / n_tasks)
    gpu_mem_per_instance = min((weights_gb + kv_budget_gb) / total_gpu_gb, 0.85)
    instance_model_config = {"max_model_len": 4096, "gpu_memory_utilization": gpu_mem_per_instance}
    print(f"  gpu_memory_utilization per instance: {gpu_mem_per_instance:.3f} ({kv_budget_gb:.2f}GB KV), max_model_len=4096")
    deployed = await deploy_servers(server_specs, model_config=instance_model_config)
    _save_deployment_results(result_dir, deployed, instance_model_config)
    mem_stats = _aggregate_memory(deployed)
    task_device_map = {
        task: f"{DEVICE_HOST}:{BASE_PORT + i}"
        for i, task in enumerate(active_tasks)
    }

    with (result_dir / "deployment_info.json").open("w") as f:
        json.dump({
            "strategy": "task_sharing",
            "backbone": BACKBONE,
            "n_tasks": n_tasks,
            "benchmark_mode": benchmark_mode,
            "assignments": [{"task": t, "url": f"{DEVICE_HOST}:{BASE_PORT+i}"}
                             for i, t in enumerate(active_tasks)],
        }, f, indent=2)

    if benchmark_mode == "closed_loop":
        task_results = run_closed_loop(task_device_map, data, duration, concurrency_per_task)
    else:
        task_results = run_open_loop(task_device_map, data, duration, target_rps_per_task)

    _save_task_results(result_dir, task_results)
    total_rps = sum(r["throughput_rps"] for r in task_results)
    avg_lat = sum(r["avg_latency_ms"] for r in task_results) / len(task_results) if task_results else 0.0

    await cleanup(DEVICE_HOST, [BASE_PORT + i for i in range(n_tasks)])

    print(f"  => total_throughput={total_rps:.2f} req/s  avg_lat={avg_lat:.1f} ms")
    return {
        "strategy": "task_sharing",
        "backbone": BACKBONE,
        "n_tasks": n_tasks,
        "n_instances": n_tasks,
        "gpu_memory_utilization": round(gpu_mem_per_instance, 4),
        "model_memory_mb": mem_stats["model_memory_mb"],
        "throughput_rps": round(total_rps, 4),
        "avg_latency_ms": round(avg_lat, 3),
    }


async def run_deploy_sharing(
    active_tasks: List[str],
    duration: float,
    result_dir: Path,
    data: Dict[str, List[str]],
    benchmark_mode: str = "closed_loop",
    target_rps_per_task: float = 2.0,
    concurrency_per_task: int = 4,
) -> Dict:
    """1 qwen2.5-0.5b vLLM server for all tasks (continuous batching)."""
    result_dir.mkdir(parents=True, exist_ok=True)
    n_tasks = len(active_tasks)

    print(f"[deploy_sharing] 1 qwen2.5-0.5b server on {CUDA_DEVICE}, {n_tasks} tasks:")
    print(f"  tasks: {active_tasks}")

    shared_model_config = {"max_model_len": 4096, "gpu_memory_utilization": 0.85}
    deployed = await deploy_servers([
        {"host": DEVICE_HOST, "port": BASE_PORT, "cuda": CUDA_DEVICE}
    ], model_config=shared_model_config)
    _save_deployment_results(result_dir, deployed, shared_model_config)
    mem_stats = _aggregate_memory(deployed)
    task_device_map = {task: f"{DEVICE_HOST}:{BASE_PORT}" for task in active_tasks}

    with (result_dir / "deployment_info.json").open("w") as f:
        json.dump({
            "strategy": "deploy_sharing",
            "backbone": BACKBONE,
            "n_tasks": n_tasks,
            "benchmark_mode": benchmark_mode,
            "server": f"{DEVICE_HOST}:{BASE_PORT}",
            "tasks": active_tasks,
        }, f, indent=2)

    if benchmark_mode == "closed_loop":
        task_results = run_closed_loop(task_device_map, data, duration, concurrency_per_task)
    else:
        task_results = run_open_loop(task_device_map, data, duration, target_rps_per_task)

    _save_task_results(result_dir, task_results)
    total_rps = sum(r["throughput_rps"] for r in task_results)
    avg_lat = sum(r["avg_latency_ms"] for r in task_results) / len(task_results) if task_results else 0.0

    await cleanup(DEVICE_HOST, [BASE_PORT])

    print(f"  => total_throughput={total_rps:.2f} req/s  avg_lat={avg_lat:.1f} ms")
    return {
        "strategy": "deploy_sharing",
        "backbone": BACKBONE,
        "n_tasks": n_tasks,
        "n_instances": 1,
        "gpu_memory_utilization": shared_model_config["gpu_memory_utilization"],
        "model_memory_mb": mem_stats["model_memory_mb"],
        "throughput_rps": round(total_rps, 4),
        "avg_latency_ms": round(avg_lat, 3),
    }


def _aggregate_memory(deployed: List[dict]) -> dict:
    """Sum model_memory_gb and reserved_gb across all instances, convert to MB."""
    total_model_gb = 0.0
    total_reserved_gb = 0.0
    for d in deployed:
        try:
            mem = json.loads(d.get("logger_summary", "{}"))
            total_model_gb += mem.get("model_memory_gb", 0.0)
            total_reserved_gb += mem.get("reserved_gb", 0.0)
        except (json.JSONDecodeError, TypeError):
            pass
    return {
        "model_memory_mb": round(total_model_gb * 1024),
        "total_reserved_mb": round(total_reserved_gb * 1024),
    }


def _save_deployment_results(result_dir: Path, deployed: List[dict], model_config: dict):
    """Write model_deployment_results.json matching the tsfm experiment format."""
    records = []
    for d in deployed:
        mem = {}
        try:
            mem = json.loads(d.get("logger_summary", "{}"))
        except (json.JSONDecodeError, TypeError):
            mem = {"raw": d.get("logger_summary", "")}
        records.append({
            "grpc_url": d["grpc_url"],
            "model_config": model_config,
            "memory": mem,
        })
    with (result_dir / "model_deployment_results.json").open("w") as f:
        json.dump(records, f, indent=2)


def _save_task_results(result_dir: Path, task_results: List[Dict]):
    fields = ["task", "device_url", "n_requests", "throughput_rps", "avg_latency_ms"]
    with (result_dir / "task_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(task_results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STRATEGY_RUNNERS = {
    "task_sharing":   run_task_sharing,
    "deploy_sharing": run_deploy_sharing,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Motivation LLM Experiment — task_sharing vs deploy_sharing")
    parser.add_argument("--n-tasks", default=os.environ.get("N_TASKS", "10,8,6,2,1"),
                        help="Comma-separated task count sweep, e.g. '10,8,6,2,1'")
    parser.add_argument("--duration", type=float, default=float(os.environ.get("PHASE_DURATION", "60")))
    parser.add_argument("--exp-dir", default=os.environ.get("EXP_DIR", "experiments/motivation2_llm/results"))
    parser.add_argument("--strategies", default="task_sharing,deploy_sharing")
    parser.add_argument("--benchmark-mode", default=os.environ.get("BENCHMARK_MODE", "closed_loop"),
                        choices=["open_loop", "closed_loop", "total_rps"])
    parser.add_argument("--target-rps", type=float, default=float(os.environ.get("TARGET_RPS", "2.0")),
                        help="[open_loop] Target arrival rate per task (req/s).")
    parser.add_argument("--total-rps", type=float, default=float(os.environ.get("TOTAL_RPS", "20.0")),
                        help="[total_rps] Fixed total arrival rate across all tasks (req/s).")
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", "4")),
                        help="[closed_loop] Number of concurrent workers per task.")
    parser.add_argument("--max-samples", type=int, default=int(os.environ.get("MAX_SAMPLES", "50")),
                        help="Number of dataset samples to load per task (cycled during benchmark).")
    args = parser.parse_args()

    task_counts = [int(x.strip()) for x in args.n_tasks.split(",")]
    strategies_to_run = [s.strip() for s in args.strategies.split(",")]

    result_root = (SERVING_DIR / args.exp_dir).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    summary_path = result_root / "summary.csv"

    print("\n" + "═" * 65)
    print(f"  Motivation LLM Experiment — {args.benchmark_mode} benchmark")
    print("═" * 65)
    print(f"  Backbone        : {BACKBONE} (vLLM)")
    print(f"  Device          : {DEVICE_HOST} {CUDA_DEVICE}")
    print(f"  task_sharing    : N qwen2.5-0.5b servers (ports {BASE_PORT}..{BASE_PORT+9})")
    print(f"  deploy_sharing  : 1 qwen2.5-0.5b server, all tasks share via continuous batching")
    print(f"  Strategies      : {strategies_to_run}")
    print(f"  N sweep         : {task_counts}")
    print(f"  Duration/run    : {args.duration}s")
    print(f"  Benchmark mode  : {args.benchmark_mode}")
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
    print(f"[INFO] Loaded {len(data)} LLM task datasets")

    async def _sweep():
        for n_tasks in task_counts:
            active_tasks = TASK_ORDER[:n_tasks]
            for strategy in strategies_to_run:
                if strategy not in STRATEGY_RUNNERS:
                    print(f"[WARN] Unknown strategy '{strategy}', skipping.")
                    continue
                if (n_tasks, strategy) in existing:
                    print(f"[SKIP] n_tasks={n_tasks} strategy={strategy} already done.")
                    continue

                run_dir = result_root / strategy / f"{n_tasks}_tasks"
                print(f"\n{'─'*65}")
                print(f"  strategy={strategy}  n_tasks={n_tasks}  mode={args.benchmark_mode}")
                print(f"  tasks={active_tasks}")

                if args.benchmark_mode == "total_rps":
                    effective_rps = args.total_rps / n_tasks
                    effective_mode = "open_loop"
                    print(f"  total_rps={args.total_rps} / n_tasks={n_tasks} → {effective_rps:.2f} req/s per task")
                else:
                    effective_rps = args.target_rps
                    effective_mode = args.benchmark_mode

                row = await STRATEGY_RUNNERS[strategy](
                    active_tasks, args.duration, run_dir, data,
                    benchmark_mode=effective_mode,
                    target_rps_per_task=effective_rps,
                    concurrency_per_task=args.concurrency,
                )
                rows.append(row)
                existing.add((n_tasks, strategy))
                _write_summary(summary_path, rows)
                await asyncio.sleep(3)

    asyncio.run(_sweep())

    print(f"\n[INFO] Done. Summary: {summary_path}")
    _print_summary(rows)
    return 0


def _write_summary(path: Path, rows: List) -> None:
    if not rows:
        return
    fields = ["n_tasks", "strategy", "backbone", "n_instances", "gpu_memory_utilization",
              "model_memory_mb", "throughput_rps", "avg_latency_ms"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _print_summary(rows: List) -> None:
    print("\n" + "─" * 90)
    print(f"  {'n_tasks':>8}  {'strategy':<20}  {'inst':>4}  {'gpu_util':>8}  "
          f"{'model_mb':>8}  {'rps':>8}  {'avg_lat_ms':>10}")
    print("─" * 90)
    for r in rows:
        print(f"  {int(r['n_tasks']):>8}  {r['strategy']:<20}  "
              f"{int(r.get('n_instances', 0)):>4}  "
              f"{float(r.get('gpu_memory_utilization', 0)):>8.3f}  "
              f"{int(r.get('model_memory_mb', 0)):>8}  "
              f"{float(r['throughput_rps']):>8.3f}  {float(r['avg_latency_ms']):>10.1f}")
    print("─" * 90)


if __name__ == "__main__":
    raise SystemExit(main())
