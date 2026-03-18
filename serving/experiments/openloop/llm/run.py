#!/usr/bin/env python3
"""motivation2/llm — Open-loop RPS benchmark against a running device gRPC server.

Assumes the device server is already running with vLLM runtime:
    python device/main.py --port 8000 --runtime-type vllm --cuda cuda:0

This script:
  1. Sends a Control(load) to load the LLM backbone on the device.
  2. Fires Poisson-arrival open-loop requests at each target RPS in --rps-sweep.
  3. Records sent_count, completed_count, latency per request.
  4. Sweeps across --n-tasks (tasks share one backbone/device).
  5. Saves per-run CSVs and a summary.csv.

Usage:
    python run.py \
        --device-url gpu01:8000 \
        --backbone qwen2.5-0.5b \
        --n-tasks 1,2,4 \
        --rps-sweep 2,5,10,20 \
        --duration 30
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
from typing import Dict, List

SERVING_DIR = Path(__file__).resolve().parents[3]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import numpy as np
from torch.utils.data import DataLoader

# Reuse site_manager gRPC client
from site_manager.grpc_client import EdgeRuntimeClient

# ---------------------------------------------------------------------------
# Task library (same as motivation1/llm)
# ---------------------------------------------------------------------------

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
    "text_classification": 8,  "sentiment": 8, "ner": 64, "qa": 64,
    "summarization": 128, "translation": 128, "math_reasoning": 256,
    "code_generation": 256, "fact_verification": 8, "reading_comprehension": 8,
}


# ---------------------------------------------------------------------------
# Dataset loading (reuse fmtk loaders, same as motivation1/llm)
# ---------------------------------------------------------------------------

def build_data(tasks: List[str], max_samples: int = 50) -> Dict[str, List[str]]:
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
    all_datasets = {
        "ag_news":       lambda: AGNewsDataset(ds_cfg, {"task_type": "text_classification"}, "test"),
        "sst2":          lambda: SST2Dataset(ds_cfg, {"task_type": "sentiment"}, "test"),
        "conll2003":     lambda: CoNLL2003Dataset(ds_cfg, {"task_type": "ner"}, "test"),
        "squad":         lambda: SQuADDataset(ds_cfg, {"task_type": "qa"}, "test"),
        "cnn_dailymail": lambda: CNNDailyMailDataset(ds_cfg, {"task_type": "summarization"}, "test"),
        "flores":        lambda: FLORESDataset(ds_cfg, {"task_type": "translation", "src_lang": "fra_Latn", "tgt_lang": "eng_Latn"}, "test"),
        "gsm8k":         lambda: GSM8KDataset(ds_cfg, {"task_type": "math_reasoning"}, "test"),
        "humaneval":     lambda: HumanEvalDataset(ds_cfg, {"task_type": "code_generation"}, "test"),
        "fever":         lambda: FEVERDataset(ds_cfg, {"task_type": "fact_verification"}, "test"),
        "hellaswag":     lambda: HellaSwagDataset(ds_cfg, {"task_type": "reading_comprehension"}, "test"),
    }
    result = {}
    for task in tasks:
        if task not in all_datasets:
            raise ValueError(f"Unknown task: {task}")
        ds = all_datasets[task]()
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
        result[task] = [batch["x"][0] for batch in loader]
        print(f"[Data] Loaded {task}: {len(result[task])} prompts")
    return result


# ---------------------------------------------------------------------------
# Deploy: Control(load) on already-running device server
# ---------------------------------------------------------------------------

async def deploy_backbone_async(device_url: str, backbone: str, model_config: dict) -> dict:
    """Send load control command to pre-running device server."""
    print(f"[Deploy] Connecting to {device_url} ...")
    client = EdgeRuntimeClient(device_url)
    try:
        await client.wait_ready()
        payload = json.dumps({"backbone": backbone, "decoders": [], **model_config})
        print(f"[Deploy] Sending Control(load) backbone={backbone} — this may take 30-120s for LLM ...")
        resp = await client.control("load", payload)
        print(f"[Deploy] Control(load) returned: {resp['status']}")
        return resp
    except Exception as e:
        print(f"[Deploy] ERROR: Control(load) failed: {e}")
        raise
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Open-loop Poisson sender
# ---------------------------------------------------------------------------

async def run_open_loop(
    device_url: str,
    tasks: List[str],
    data: Dict[str, List[str]],
    target_rps_per_task: float,
    duration: float,
) -> Dict:
    """
    Fire Poisson-arrival requests for each task concurrently.
    Returns dict with sent timestamps, latencies, errors per task.
    """
    client = EdgeRuntimeClient(device_url)
    await client.wait_ready()

    sent:      Dict[str, List[float]] = {t: [] for t in tasks}
    latencies: Dict[str, List[float]] = {t: [] for t in tasks}
    errors:    Dict[str, int]         = {t: 0  for t in tasks}

    req_timeout = max(60.0, duration * 2)

    async def _fire(task: str, req_id: int, prompt: str, t_send: float) -> None:
        try:
            resp = await asyncio.wait_for(client.infer({
                "req_id":   req_id,
                "task":     task,
                "x":        np.array([0.0], dtype=np.float32),
                "question": prompt,
            }), timeout=req_timeout)
            latencies[task].append(time.time() - t_send)
        except Exception:
            errors[task] += 1

    async def _task_sender(task: str, req_id_offset: int) -> None:
        prompts  = data[task]
        cycle    = len(prompts)
        deadline = time.time() + duration
        req_id   = req_id_offset
        in_flight = []
        while time.time() < deadline:
            t_send = time.time()
            sent[task].append(t_send)
            prompt = prompts[req_id % cycle]
            in_flight.append(asyncio.create_task(_fire(task, req_id, prompt, t_send)))
            req_id += 1
            gap = np.random.exponential(1.0 / target_rps_per_task)
            remaining = (t_send + gap) - time.time()
            if remaining > 0:
                await asyncio.sleep(remaining)
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

    await asyncio.gather(
        *[_task_sender(task, i * 1_000_000) for i, task in enumerate(tasks)],
        return_exceptions=True,
    )
    await client.close()

    for task in tasks:
        n_sent = len(sent[task])
        n_done = len(latencies[task])
        actual_rps = n_done / duration
        avg_lat = (sum(latencies[task]) / n_done * 1000) if n_done else 0.0
        print(f"  [{task}] target={target_rps_per_task:.1f} sent={n_sent} "
              f"completed={n_done} actual_rps={actual_rps:.2f} "
              f"avg_lat={avg_lat:.1f}ms errors={errors[task]}")

    return {"sent": sent, "latencies": latencies, "errors": errors}


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _save_run(result_dir: Path, task: str, target_rps: float,
              duration: float, sent: List[float], latencies: List[float],
              n_errors: int) -> Dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    n_sent = len(sent)
    n_completed = len(latencies)
    input_rps  = n_sent / duration
    output_rps = n_completed / duration
    avg_lat_ms = (sum(latencies) / n_completed * 1000) if n_completed else 0.0
    p50 = float(np.percentile(latencies, 50)) * 1000 if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) * 1000 if latencies else 0.0
    p99 = float(np.percentile(latencies, 99)) * 1000 if latencies else 0.0

    with (result_dir / f"{task}_requests.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["req_idx", "send_time", "latency_ms"])
        for i, (t, lat) in enumerate(zip(sent, latencies)):
            w.writerow([i, f"{t:.6f}", f"{lat*1000:.3f}"])

    return {
        "task":        task,
        "target_rps":  target_rps,
        "input_rps":   round(input_rps,  4),
        "output_rps":  round(output_rps, 4),
        "n_sent":      n_sent,
        "n_completed": n_completed,
        "n_errors":    n_errors,
        "avg_lat_ms":  round(avg_lat_ms, 3),
        "p50_lat_ms":  round(p50, 3),
        "p95_lat_ms":  round(p95, 3),
        "p99_lat_ms":  round(p99, 3),
    }


def _write_summary(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fields = ["n_tasks", "backbone", "task", "target_rps", "input_rps", "output_rps",
              "n_sent", "n_completed", "n_errors",
              "avg_lat_ms", "p50_lat_ms", "p95_lat_ms", "p99_lat_ms"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _print_summary(rows: List[Dict]) -> None:
    print("\n" + "─" * 100)
    print(f"  {'n_tasks':>7}  {'task':<14}  {'target':>7}  {'in_rps':>7}  "
          f"{'out_rps':>7}  {'sent':>6}  {'done':>6}  {'err':>4}  "
          f"{'avg_ms':>7}  {'p95_ms':>7}  {'p99_ms':>7}")
    print("─" * 100)
    for r in rows:
        print(f"  {int(r['n_tasks']):>7}  {r['task']:<14}  "
              f"{float(r['target_rps']):>7.1f}  {float(r['input_rps']):>7.2f}  "
              f"{float(r['output_rps']):>7.2f}  {int(r['n_sent']):>6}  "
              f"{int(r['n_completed']):>6}  {int(r['n_errors']):>4}  "
              f"{float(r['avg_lat_ms']):>7.1f}  {float(r['p95_lat_ms']):>7.1f}  "
              f"{float(r['p99_lat_ms']):>7.1f}")
    print("─" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="motivation2/llm — Open-loop LLM device overhead benchmark"
    )
    parser.add_argument("--device-url",   default=os.environ.get("DEVICE_URL",   "localhost:8000"))
    parser.add_argument("--backbone",     default=os.environ.get("BACKBONE",     "qwen2.5-0.5b"))
    parser.add_argument("--n-tasks",      default=os.environ.get("N_TASKS",      "1"),
                        help="Comma-separated task counts to sweep, e.g. 1,2,4")
    parser.add_argument("--rps-sweep",    default=os.environ.get("RPS_SWEEP",    "2,5,10,20"),
                        help="Comma-separated target RPS per task")
    parser.add_argument("--duration",     type=float, default=float(os.environ.get("DURATION",     "30")))
    parser.add_argument("--max-samples",  type=int,   default=int(os.environ.get("MAX_SAMPLES",   "50")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "64")),
                        help="Uniform max_new_tokens sent in model_config at load time")
    parser.add_argument("--max-model-len", type=int,  default=int(os.environ.get("MAX_MODEL_LEN", "256")))
    parser.add_argument("--gpu-util",     type=float, default=float(os.environ.get("GPU_UTIL",    "0.85")))
    parser.add_argument("--exp-dir",      default=os.environ.get("EXP_DIR",
                        "experiments/motivation2/llm/results"))
    parser.add_argument("--skip-load",    action="store_true",
                        default=os.environ.get("SKIP_LOAD", "").lower() in ("1", "true"),
                        help="Skip Control(load) — model already loaded on device")
    args = parser.parse_args()

    task_counts = [int(x.strip()) for x in args.n_tasks.split(",")]
    rps_values  = [float(x.strip()) for x in args.rps_sweep.split(",")]
    result_root = (SERVING_DIR / args.exp_dir).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    summary_path = result_root / "summary.csv"

    print("\n" + "═" * 70)
    print("  motivation2/llm — Open-loop device overhead experiment")
    print("═" * 70)
    print(f"  Device URL    : {args.device_url}")
    print(f"  Backbone      : {args.backbone}")
    print(f"  N-tasks       : {task_counts}")
    print(f"  RPS sweep     : {rps_values}")
    print(f"  Duration      : {args.duration}s per point")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  Skip load     : {args.skip_load}")
    print(f"  Results       : {result_root}")
    print("═" * 70)

    rows: List[Dict] = []
    existing: set = set()
    if summary_path.exists():
        with summary_path.open() as f:
            for row in csv.DictReader(f):
                existing.add((int(row["n_tasks"]), row["task"], float(row["target_rps"])))
                rows.append(row)
        print(f"[INFO] Resuming — {len(existing)} points already done")

    model_config = {
        "max_model_len":          args.max_model_len,
        "gpu_memory_utilization": args.gpu_util,
        "max_new_tokens":         args.max_new_tokens,
    }

    async def run_experiment(n_tasks: int, active_tasks: List[str], data: Dict,
                             rps_values: List[float]) -> List[Dict]:
        """Deploy once then sweep RPS — all in one event loop to avoid races."""
        exp_rows: List[Dict] = []

        if not args.skip_load:
            print(f"[Deploy] Loading {args.backbone} on {args.device_url} — waiting for load to complete...")
            resp = await deploy_backbone_async(args.device_url, args.backbone, model_config)
            if "error" in resp.get("status", "").lower():
                raise RuntimeError(f"Model load failed: {resp}")
            print(f"[Deploy] Load complete: {resp['status']}")
            await asyncio.sleep(1)

        for target_rps in rps_values:
            all_done = all((n_tasks, t, target_rps) in existing for t in active_tasks)
            if all_done:
                print(f"[SKIP] n_tasks={n_tasks} rps={target_rps} already done.")
                continue

            run_dir = result_root / f"{n_tasks}_tasks" / f"rps_{target_rps:.1f}"
            print(f"\n{'─' * 70}")
            print(f"  n_tasks={n_tasks}  target_rps={target_rps}  tasks={active_tasks}")

            result = await run_open_loop(
                device_url=args.device_url,
                tasks=active_tasks,
                data=data,
                target_rps_per_task=target_rps,
                duration=args.duration,
            )

            for task in active_tasks:
                if (n_tasks, task, target_rps) in existing:
                    continue
                row = _save_run(
                    run_dir, task, target_rps, args.duration,
                    result["sent"][task],
                    result["latencies"][task],
                    result["errors"][task],
                )
                row["n_tasks"]  = n_tasks
                row["backbone"] = args.backbone
                exp_rows.append(row)
                existing.add((n_tasks, task, target_rps))

            _write_summary(summary_path, rows + exp_rows)
            await asyncio.sleep(1)

        return exp_rows

    for n_tasks in task_counts:
        active_tasks = TASK_ORDER[:n_tasks]
        print(f"\n[INFO] Loading data for {n_tasks} task(s): {active_tasks}")
        data = build_data(active_tasks, max_samples=args.max_samples)
        new_rows = asyncio.run(run_experiment(n_tasks, active_tasks, data, rps_values))
        rows.extend(new_rows)

    print(f"\n[INFO] Done. Summary: {summary_path}")
    _print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
