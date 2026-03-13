#!/usr/bin/env python3
"""Motivation Experiment #1 — LLM in-process benchmark.

- task_sharing:   one process-isolated local vLLM runtime per task
- deploy_sharing: one shared local vLLM runtime across tasks
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

SERVING_DIR = Path(__file__).resolve().parents[3]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import numpy as np

from experiments.motivation1.llm.server import LLMInferenceServer

DEFAULT_BACKBONE = "qwen2.5-0.5b"

TASK_ORDER: List[str] = [
    "ag_news", "sst2", "conll2003", "squad", "cnn_dailymail",
    "flores", "gsm8k", "humaneval", "fever", "hellaswag",
]

TASK_TYPE: Dict[str, str] = {
    "ag_news": "text_classification", "sst2": "sentiment",
    "conll2003": "ner", "squad": "qa", "cnn_dailymail": "summarization",
    "flores": "translation", "gsm8k": "math_reasoning",
    "humaneval": "code_generation", "fever": "fact_verification",
    "hellaswag": "reading_comprehension",
}

MAX_NEW_TOKENS: Dict[str, int] = {
    "text_classification": 8, "sentiment": 8, "ner": 64, "qa": 64,
    "summarization": 128, "translation": 128, "math_reasoning": 256,
    "code_generation": 256, "fact_verification": 8, "reading_comprehension": 8,
}


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def build_data(max_samples: int = 50) -> Dict[str, List[str]]:
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

    ds_cfg  = {"max_samples": max_samples}
    collate = lambda batch: {"x": [i["x"] for i in batch], "y": [i["y"] for i in batch]}
    datasets = {
        "ag_news":       AGNewsDataset(ds_cfg, {"task_type": "text_classification"}, "test"),
        "sst2":          SST2Dataset(ds_cfg, {"task_type": "sentiment"}, "test"),
        "conll2003":     CoNLL2003Dataset(ds_cfg, {"task_type": "ner"}, "test"),
        "squad":         SQuADDataset(ds_cfg, {"task_type": "qa"}, "test"),
        "cnn_dailymail": CNNDailyMailDataset(ds_cfg, {"task_type": "summarization"}, "test"),
        "flores":        FLORESDataset(ds_cfg, {"task_type": "translation", "src_lang": "fra_Latn", "tgt_lang": "eng_Latn"}, "test"),
        "gsm8k":         GSM8KDataset(ds_cfg, {"task_type": "math_reasoning"}, "test"),
        "humaneval":     HumanEvalDataset(ds_cfg, {"task_type": "code_generation"}, "test"),
        "fever":         FEVERDataset(ds_cfg, {"task_type": "fact_verification"}, "test"),
        "hellaswag":     HellaSwagDataset(ds_cfg, {"task_type": "reading_comprehension"}, "test"),
    }
    result = {}
    for task, ds in datasets.items():
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
        result[task] = [batch["x"][0] for batch in loader]
    return result


def homogenize_prompts(data: Dict[str, List[str]], source_task: str | None = None) -> Dict[str, List[str]]:
    if not source_task:
        return data
    if source_task not in data:
        raise ValueError(f"prompt source task '{source_task}' not found")
    return {task: list(data[source_task]) for task in data}


# ---------------------------------------------------------------------------
# Core benchmark primitive — shared by both open-loop and closed-loop workers
# ---------------------------------------------------------------------------

async def _run_closed_loop(client, task: str, prompts: List[str],
                           duration: float, concurrency: int, req_id_start: int = 0) -> List[float]:
    """Run closed-loop inference: `concurrency` workers fire back-to-back for `duration` seconds."""
    latencies: List[float] = []
    stop      = asyncio.Event()
    req_id    = req_id_start
    lock      = asyncio.Lock()
    cycle     = len(prompts)

    async def _worker():
        nonlocal req_id
        while not stop.is_set():
            async with lock:
                rid = req_id; req_id += 1
            t0 = time.time()
            try:
                await client.infer({"req_id": rid, "task": task,
                                    "x": np.array([0.0], dtype=np.float32),
                                    "question": prompts[rid % cycle]})
                latencies.append((time.time() - t0) * 1000)
            except Exception as e:
                print(f"[Worker:{task}] req {rid} error: {e}")

    async def _timer():
        await asyncio.sleep(max(0, time.time() + duration - time.time()))
        stop.set()

    await client.wait_ready()
    await asyncio.gather(
        asyncio.create_task(_timer()),
        *[asyncio.create_task(_worker()) for _ in range(concurrency)],
        return_exceptions=True,
    )
    await client.close()
    return latencies


async def _run_open_loop(client, task: str, prompts: List[str],
                         duration: float, target_rps: float, req_id_start: int = 0) -> List[float]:
    """Run open-loop inference: Poisson arrivals at `target_rps` for `duration` seconds."""
    latencies: List[float] = []
    in_flight = []
    deadline  = time.time() + duration
    req_id    = req_id_start
    cycle     = len(prompts)

    async def _fire(rid, prompt, t0):
        try:
            await client.infer({"req_id": rid, "task": task,
                                "x": np.array([0.0], dtype=np.float32), "question": prompt})
            latencies.append((time.time() - t0) * 1000)
        except Exception as e:
            print(f"[Worker:{task}] req {rid} error: {e}")

    await client.wait_ready()
    while time.time() < deadline:
        t0 = time.time()
        in_flight.append(asyncio.create_task(_fire(req_id, prompts[req_id % cycle], t0)))
        req_id += 1
        remaining = (t0 + np.random.exponential(1.0 / target_rps)) - time.time()
        if remaining > 0:
            await asyncio.sleep(remaining)

    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=True)
    await client.close()
    return latencies


# ---------------------------------------------------------------------------
# Thread entry — one thread per task, calls the appropriate async runner
# ---------------------------------------------------------------------------

def _task_worker(task: str, client, label: str, prompts: List[str],
                 duration: float, results: List, req_id_start: int,
                 mode: str, target_rps: float, concurrency: int) -> None:
    if mode == "closed_loop":
        latencies = asyncio.run(_run_closed_loop(client, task, prompts, duration, concurrency, req_id_start))
        print(f"[Worker:{task}] {len(latencies)} reqs concurrency={concurrency} "
              f"actual={len(latencies)/duration:.2f} req/s avg_lat={sum(latencies)/len(latencies) if latencies else 0:.1f} ms")
    else:
        latencies = asyncio.run(_run_open_loop(client, task, prompts, duration, target_rps, req_id_start))
        print(f"[Worker:{task}] {len(latencies)} reqs @ target={target_rps:.1f} "
              f"actual={len(latencies)/duration:.2f} req/s avg_lat={sum(latencies)/len(latencies) if latencies else 0:.1f} ms")

    n = len(latencies)
    results.append({
        "task": task, "device_url": label, "n_requests": n,
        "throughput_rps": n / duration, "avg_latency_ms": sum(latencies) / n if n else 0.0,
    })


def _run_benchmark(task_client_map: Dict[str, tuple], data: Dict[str, List[str]],
                   duration: float, mode: str, target_rps: float, concurrency: int) -> List[Dict]:
    results, threads = [], []
    for i, (task, (client, label)) in enumerate(task_client_map.items()):
        t = threading.Thread(
            target=_task_worker,
            args=(task, client, label, data[task], duration, results, i * 10000),
            kwargs={"mode": mode, "target_rps": target_rps, "concurrency": concurrency},
            daemon=True,
        )
        threads.append(t)

    print(f"\n[Benchmark:{mode}] {len(threads)} tasks, {duration}s")
    for t in threads: t.start()
    for t in threads: t.join(timeout=duration + 120)
    return results


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

def _run_strategy(strategy: str, active_tasks: List[str], duration: float,
                  result_dir: Path, data: Dict[str, List[str]], backbone: str,
                  device: str, uniform_max_new_tokens: int, benchmark_mode: str,
                  target_rps_per_task: float, concurrency_per_task: int) -> Dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    n_tasks = len(active_tasks)

    # task_sharing: N separate processes on one GPU. Using 1/n_tasks is wrong
    # for separate processes — vLLM's memory profiler already sees sibling
    # processes' weights as non_torch_memory, so a low gpu_util leaves
    # zero room for KV cache. Use 0.90 so each process grabs KV blocks
    # from whatever real GPU memory remains after siblings.
    # deploy_sharing: 1 engine gets the full GPU.
    gpu_util = 0.85/n_tasks if strategy == "task_sharing" else 0.85
    model_config = {
        "max_model_len":          256,
        "gpu_memory_utilization": gpu_util,
        "max_new_tokens":         int(uniform_max_new_tokens),
        "duration":               duration,
        "concurrency":            concurrency_per_task,
        "prompts":                {task: data[task] for task in active_tasks},
    }

    print(f"[{strategy}] backbone={backbone} device={device} n_tasks={n_tasks}")
    server = LLMInferenceServer(strategy=strategy, tasks=active_tasks,
                                backbone=backbone, device=device, model_config=model_config)
    server.start()

    if strategy == "deploy_sharing":
        task_client_map = {task: (server.client(task), "local://instance-0") for task in active_tasks}
        task_results = _run_benchmark(task_client_map, data, duration, benchmark_mode,
                                      target_rps_per_task, concurrency_per_task)
    else:
        proc_results = server.task_sharing_results()
        failures = [r for r in proc_results if r.get("error")]
        for f in failures:
            print(f"[task_sharing] worker failure ({f['task']}): {f['error']}")
            if f.get("traceback"): print(f["traceback"])
        task_results = [
            {"task": r["task"], "device_url": r["label"], "n_requests": r["n_requests"],
             "throughput_rps": r["throughput_rps"], "avg_latency_ms": r["avg_latency_ms"]}
            for r in proc_results if not r.get("error")
        ]
        if not task_results:
            raise RuntimeError("task_sharing: all workers failed")

    _save_task_results(result_dir, task_results)
    mem_stats          = server.memory_summary()
    deployment_records = server.deployment_records()
    _save_deployment_results(result_dir, deployment_records)

    with (result_dir / "deployment_info.json").open("w") as f:
        json.dump({"strategy": strategy, "backbone": backbone, "n_tasks": n_tasks,
                   "runtime": "in_process_vllm", "benchmark_mode": benchmark_mode,
                   "tasks": active_tasks}, f, indent=2)

    server.stop()

    total_rps      = sum(r["throughput_rps"] for r in task_results)
    avg_lat        = sum(r["avg_latency_ms"] for r in task_results) / len(task_results) if task_results else 0.0
    gpu_util       = model_config["gpu_memory_utilization"]
    n_inst         = 1 if strategy == "deploy_sharing" else n_tasks
    static_peak_mb = n_inst * gpu_util * mem_stats["total_gpu_gb"] * 1024
    peak_gpu_mb    = sum(r.get("memory", {}).get("gpu_peak_mb", 0.0) for r in deployment_records)

    print(f"  => model_mem={mem_stats['model_memory_mb']:.0f} MB static_peak={static_peak_mb:.1f} MB "
          f"total_throughput={total_rps:.2f} req/s avg_lat={avg_lat:.1f} ms")
    return {
        "strategy": strategy, "backbone": backbone, "n_tasks": n_tasks,
        "n_instances": n_inst, "gpu_memory_utilization": round(gpu_util, 4),
        "model_memory_mb": mem_stats["model_memory_mb"],
        "model_load_mem_mb": mem_stats["model_memory_mb"],
        "peak_gpu_mem_mb": round(peak_gpu_mb, 3),
        "static_peak_gpu_mem_mb": round(static_peak_mb, 3),
        "throughput_rps": round(total_rps, 4),
        "avg_latency_ms": round(avg_lat, 3),
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_deployment_results(result_dir: Path, records: List[dict]) -> None:
    with (result_dir / "model_deployment_results.json").open("w") as f:
        json.dump(records, f, indent=2)


def _save_task_results(result_dir: Path, task_results: List[Dict]) -> None:
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
    parser.add_argument("--n-tasks",        default=os.environ.get("N_TASKS",         "10,8,6,2,1"))
    parser.add_argument("--duration",       type=float, default=float(os.environ.get("PHASE_DURATION", "60")))
    parser.add_argument("--strategies",     default=os.environ.get("STRATEGIES",      "task_sharing,deploy_sharing"))
    parser.add_argument("--backbone",       default=os.environ.get("BACKBONE",        DEFAULT_BACKBONE))
    parser.add_argument("--cuda",           default=os.environ.get("CUDA_DEVICE",     "cuda:0"))
    parser.add_argument("--exp-dir",        default=os.environ.get("EXP_DIR",         "experiments/motivation1/llm/results"))
    parser.add_argument("--benchmark-mode", default=os.environ.get("BENCHMARK_MODE",  "closed_loop"),
                        choices=["open_loop", "closed_loop", "total_rps"])
    parser.add_argument("--target-rps",     type=float, default=float(os.environ.get("TARGET_RPS",  "2.0")))
    parser.add_argument("--total-rps",      type=float, default=float(os.environ.get("TOTAL_RPS",   "20.0")))
    parser.add_argument("--concurrency",    type=int,   default=int(os.environ.get("CONCURRENCY",   "4")))
    parser.add_argument("--max-samples",    type=int,   default=int(os.environ.get("MAX_SAMPLES",   "50")))
    parser.add_argument("--uniform-max-new-tokens", type=int,
                        default=int(os.environ.get("UNIFORM_MAX_NEW_TOKENS", "64")))
    parser.add_argument("--prompt-source-task", default=os.environ.get("PROMPT_SOURCE_TASK", "").strip())
    args = parser.parse_args()

    task_counts       = [int(x.strip()) for x in args.n_tasks.split(",")]
    strategies_to_run = [s.strip() for s in args.strategies.split(",")]
    result_root       = (SERVING_DIR / args.exp_dir).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    summary_path      = result_root / "summary.csv"

    print("\n" + "═" * 65)
    print("  Motivation Experiment #1 — LLM Deploy Sharing vs Task Sharing")
    print("═" * 65)
    print(f"  Backbone        : {args.backbone}")
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
                effective_rps  = args.total_rps / n_tasks
                effective_mode = "open_loop"
                print(f"  total_rps={args.total_rps} / n_tasks={n_tasks} → {effective_rps:.2f} req/s per task")
            else:
                effective_rps  = args.target_rps
                effective_mode = args.benchmark_mode

            row = _run_strategy(
                strategy=strategy, active_tasks=active_tasks, duration=args.duration,
                result_dir=run_dir, data=data, backbone=args.backbone, device=args.cuda,
                uniform_max_new_tokens=args.uniform_max_new_tokens,
                benchmark_mode=effective_mode, target_rps_per_task=effective_rps,
                concurrency_per_task=args.concurrency,
            )
            rows.append(row)
            existing.add((n_tasks, strategy))
            _write_summary(summary_path, rows)
            time.sleep(3)

    print(f"\n[INFO] Done. Summary: {summary_path}")
    _print_summary(rows)
    return 0


def _write_summary(path: Path, rows: List) -> None:
    if not rows:
        return
    fields = ["n_tasks", "strategy", "backbone", "n_instances", "gpu_memory_utilization",
              "model_memory_mb", "model_load_mem_mb", "peak_gpu_mem_mb",
              "static_peak_gpu_mem_mb", "throughput_rps", "avg_latency_ms"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _print_summary(rows: List) -> None:
    print("\n" + "─" * 125)
    print(f"  {'n_tasks':>8}  {'strategy':<20}  {'inst':>4}  {'gpu_util':>8}  "
          f"{'model_mb':>8}  {'load_mb':>8}  {'peak_mb':>8}  {'static_peak_mb':>14}  {'rps':>8}  {'avg_lat_ms':>10}")
    print("─" * 125)
    for r in rows:
        print(f"  {int(r['n_tasks']):>8}  {r['strategy']:<20}  "
              f"{int(r.get('n_instances', 0)):>4}  {float(r.get('gpu_memory_utilization', 0)):>8.3f}  "
              f"{int(r.get('model_memory_mb', 0)):>8}  {float(r.get('model_load_mem_mb', 0)):>8.1f}  "
              f"{float(r.get('peak_gpu_mem_mb', 0)):>8.1f}  {float(r.get('static_peak_gpu_mem_mb', 0)):>14.1f}  "
              f"{float(r['throughput_rps']):>8.3f}  {float(r['avg_latency_ms']):>10.1f}")
    print("─" * 125)


if __name__ == "__main__":
    raise SystemExit(main())
