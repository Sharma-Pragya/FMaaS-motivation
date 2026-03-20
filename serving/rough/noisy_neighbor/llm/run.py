#!/usr/bin/env python3
"""noisy_neighbor/llm — Cross-task interference experiment (LLM).

Fixes victim task at a low RPS and sweeps aggressor RPS to show how the
naive cross-task FIFO scheduler causes victim latency degradation.

Usage (from serving/):
    python experiments/noisy_neighbor/llm/run.py \
        --device-url localhost:8000 \
        --backbone qwen2.5-0.5b \
        --victim-task sst2 \
        --victim-rps 2 \
        --aggressor-task conll2003 \
        --aggressor-rps-sweep 0,2,5,10,20,30 \
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

from site_manager.grpc_client import EdgeRuntimeClient

# ---------------------------------------------------------------------------
# Task library
# ---------------------------------------------------------------------------

TASK_TYPE: Dict[str, str] = {
    "ag_news": "text_classification", "sst2": "sentiment",
    "conll2003": "ner", "squad": "qa", "cnn_dailymail": "summarization",
    "flores": "translation", "gsm8k": "math_reasoning",
    "humaneval": "code_generation", "fever": "fact_verification",
    "hellaswag": "reading_comprehension",
}


# ---------------------------------------------------------------------------
# Dataset loading
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
    collate = lambda batch: {"x": [i["x"] for i in batch]}
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
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            collate_fn=lambda batch: {"x": [i["x"] for i in batch]})
        result[task] = [batch["x"][0] for batch in loader]
        print(f"[Data] Loaded {task}: {len(result[task])} prompts")
    return result


# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

async def deploy_backbone_async(device_url: str, backbone: str, model_config: dict) -> dict:
    print(f"[Deploy] Connecting to {device_url} ...")
    client = EdgeRuntimeClient(device_url)
    try:
        await client.wait_ready()
        payload = json.dumps({"backbone": backbone, "decoders": [], **model_config})
        print(f"[Deploy] Sending Control(load) backbone={backbone} — may take 30-120s ...")
        resp = await client.control("load", payload)
        print(f"[Deploy] Control(load) returned: {resp['status']}")
        return resp
    except Exception as e:
        print(f"[Deploy] ERROR: {e}")
        raise
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Open-loop sender — per-task RPS
# ---------------------------------------------------------------------------

async def run_open_loop(
    device_url: str,
    task_rps: Dict[str, float],
    data: Dict[str, List[str]],
    duration: float,
    req_timeout: float,
) -> Dict:
    tasks = [t for t, rps in task_rps.items() if rps > 0]
    client = EdgeRuntimeClient(device_url)
    await client.wait_ready()

    sent:      Dict[str, List[float]] = {t: [] for t in tasks}
    latencies: Dict[str, List[float]] = {t: [] for t in tasks}
    errors:    Dict[str, int]         = {t: 0  for t in tasks}

    async def _fire(task: str, req_id: int, prompt: str, t_send: float) -> None:
        try:
            await asyncio.wait_for(client.infer({
                "req_id":   req_id,
                "task":     task,
                "x":        np.array([0.0], dtype=np.float32),
                "question": prompt,
            }), timeout=req_timeout)
            latencies[task].append(time.time() - t_send)
        except Exception:
            errors[task] += 1

    async def _task_sender(task: str, rps: float, req_id_offset: int) -> None:
        prompts  = data[task]
        cycle    = len(prompts)
        deadline = time.time() + duration
        req_id   = req_id_offset
        in_flight = []
        while time.time() < deadline:
            t_send = time.time()
            sent[task].append(t_send)
            in_flight.append(asyncio.create_task(
                _fire(task, req_id, prompts[req_id % cycle], t_send)))
            req_id += 1
            gap = np.random.exponential(1.0 / rps)
            remaining = (t_send + gap) - time.time()
            if remaining > 0:
                await asyncio.sleep(remaining)
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

    await asyncio.gather(
        *[_task_sender(t, task_rps[t], i * 1_000_000) for i, t in enumerate(tasks)],
        return_exceptions=True,
    )
    await client.close()

    for task in tasks:
        n_done = len(latencies[task])
        avg_lat = (sum(latencies[task]) / n_done * 1000) if n_done else 0.0
        print(f"  [{task}] rps={task_rps[task]:.1f} sent={len(sent[task])} "
              f"completed={n_done} actual={n_done/duration:.2f} "
              f"avg_lat={avg_lat:.1f}ms errors={errors[task]}")

    return {"sent": sent, "latencies": latencies, "errors": errors}


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _compute_stats(sent: List[float], latencies: List[float],
                   n_errors: int, duration: float) -> Dict:
    n_sent = len(sent)
    n_completed = len(latencies)
    return {
        "input_rps":   round(n_sent / duration, 4),
        "output_rps":  round(n_completed / duration, 4),
        "n_sent":      n_sent,
        "n_completed": n_completed,
        "n_errors":    n_errors,
        "avg_lat_ms":  round((sum(latencies) / n_completed * 1000) if n_completed else 0.0, 3),
        "p50_lat_ms":  round(float(np.percentile(latencies, 50)) * 1000 if latencies else 0.0, 3),
        "p95_lat_ms":  round(float(np.percentile(latencies, 95)) * 1000 if latencies else 0.0, 3),
        "p99_lat_ms":  round(float(np.percentile(latencies, 99)) * 1000 if latencies else 0.0, 3),
    }


def _save_run(result_dir: Path, task: str, sent: List[float],
              latencies: List[float], duration: float) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / f"{task}_requests.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["req_idx", "send_time", "latency_ms"])
        for i, (t, lat) in enumerate(zip(sent, latencies)):
            w.writerow([i, f"{t:.6f}", f"{lat*1000:.3f}"])


SUMMARY_FIELDS = [
    "backbone", "victim_task", "victim_rps",
    "aggressor_task", "aggressor_rps",
    "role", "task",
    "input_rps", "output_rps", "n_sent", "n_completed", "n_errors",
    "avg_lat_ms", "p50_lat_ms", "p95_lat_ms", "p99_lat_ms",
]


def _write_summary(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _print_summary(rows: List[Dict]) -> None:
    print("\n" + "─" * 110)
    print(f"  {'victim_task':<12}  {'aggressor_task':<14}  {'agg_rps':>7}  "
          f"{'role':<10}  {'avg_ms':>8}  {'p50_ms':>8}  {'p95_ms':>8}  {'p99_ms':>8}")
    print("─" * 110)
    for r in rows:
        print(f"  {r['victim_task']:<12}  {r['aggressor_task']:<14}  "
              f"{float(r['aggressor_rps']):>7.1f}  {r['role']:<10}  "
              f"{float(r['avg_lat_ms']):>8.1f}  {float(r['p50_lat_ms']):>8.1f}  "
              f"{float(r['p95_lat_ms']):>8.1f}  {float(r['p99_lat_ms']):>8.1f}")
    print("─" * 110)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="noisy_neighbor/llm — Cross-task interference experiment"
    )
    parser.add_argument("--device-url",           default=os.environ.get("DEVICE_URL", "localhost:8000"))
    parser.add_argument("--backbone",             default=os.environ.get("BACKBONE", "qwen2.5-0.5b"))
    parser.add_argument("--victim-task",          default=os.environ.get("VICTIM_TASK", "sst2"))
    parser.add_argument("--victim-rps",           type=float, default=float(os.environ.get("VICTIM_RPS", "2")))
    parser.add_argument("--aggressor-task",       default=os.environ.get("AGGRESSOR_TASK", "conll2003"))
    parser.add_argument("--aggressor-rps-sweep",  default=os.environ.get("AGGRESSOR_RPS_SWEEP", "0,2,5,10,20,30"))
    parser.add_argument("--duration",             type=float, default=float(os.environ.get("DURATION", "30")))
    parser.add_argument("--max-samples",          type=int,   default=int(os.environ.get("MAX_SAMPLES", "50")))
    parser.add_argument("--max-new-tokens",       type=int,   default=int(os.environ.get("MAX_NEW_TOKENS", "64")))
    parser.add_argument("--max-model-len",        type=int,   default=int(os.environ.get("MAX_MODEL_LEN", "256")))
    parser.add_argument("--gpu-util",             type=float, default=float(os.environ.get("GPU_UTIL", "0.85")))
    parser.add_argument("--exp-dir",              default=os.environ.get("EXP_DIR",
                                                  "experiments/noisy_neighbor/llm/results"))
    parser.add_argument("--skip-load",            action="store_true",
                        default=os.environ.get("SKIP_LOAD", "").lower() in ("1", "true"))
    args = parser.parse_args()

    aggressor_rps_values = [float(x.strip()) for x in args.aggressor_rps_sweep.split(",")]
    result_root = (SERVING_DIR / args.exp_dir).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    summary_path = result_root / "summary.csv"
    req_timeout = max(60.0, args.duration * 2)

    print("\n" + "═" * 70)
    print("  noisy_neighbor/llm — Cross-task interference experiment")
    print("═" * 70)
    print(f"  Device URL     : {args.device_url}")
    print(f"  Backbone       : {args.backbone}")
    print(f"  Victim task    : {args.victim_task} @ {args.victim_rps} rps (fixed)")
    print(f"  Aggressor task : {args.aggressor_task} @ {aggressor_rps_values} rps (sweep)")
    print(f"  Duration       : {args.duration}s per point")
    print(f"  Results        : {result_root}")
    print("═" * 70)

    rows: List[Dict] = []
    existing: set = set()
    if summary_path.exists():
        with summary_path.open() as f:
            for row in csv.DictReader(f):
                existing.add((row["victim_task"], row["aggressor_task"],
                               float(row["aggressor_rps"]), row["role"]))
                rows.append(row)
        print(f"[INFO] Resuming — {len(existing)//2} points already done")

    model_config = {
        "max_model_len":          args.max_model_len,
        "gpu_memory_utilization": args.gpu_util,
        "max_new_tokens":         args.max_new_tokens,
    }

    async def run_experiment(aggressor_rps_values: List[float]) -> List[Dict]:
        exp_rows: List[Dict] = []

        if not args.skip_load:
            print(f"[Deploy] Loading {args.backbone} on {args.device_url} ...")
            resp = await deploy_backbone_async(args.device_url, args.backbone, model_config)
            if "error" in resp.get("status", "").lower():
                raise RuntimeError(f"Model load failed: {resp}")
            await asyncio.sleep(1)

        for agg_rps in aggressor_rps_values:
            key_victim    = (args.victim_task, args.aggressor_task, agg_rps, "victim")
            key_aggressor = (args.victim_task, args.aggressor_task, agg_rps, "aggressor")
            if key_victim in existing and (agg_rps == 0 or key_aggressor in existing):
                print(f"[SKIP] aggressor_rps={agg_rps} already done.")
                continue

            print(f"\n{'─'*70}")
            print(f"  victim={args.victim_task}@{args.victim_rps}  "
                  f"aggressor={args.aggressor_task}@{agg_rps}")

            task_rps: Dict[str, float] = {args.victim_task: args.victim_rps}
            if agg_rps > 0:
                task_rps[args.aggressor_task] = agg_rps

            result = await run_open_loop(
                device_url=args.device_url,
                task_rps=task_rps,
                data=data,
                duration=args.duration,
                req_timeout=req_timeout,
            )

            run_dir = result_root / f"agg_rps_{agg_rps:.1f}"

            stats = _compute_stats(
                result["sent"].get(args.victim_task, []),
                result["latencies"].get(args.victim_task, []),
                result["errors"].get(args.victim_task, 0),
                args.duration,
            )
            _save_run(run_dir, args.victim_task,
                      result["sent"].get(args.victim_task, []),
                      result["latencies"].get(args.victim_task, []),
                      args.duration)
            exp_rows.append({
                "backbone":       args.backbone,
                "victim_task":    args.victim_task,
                "victim_rps":     args.victim_rps,
                "aggressor_task": args.aggressor_task,
                "aggressor_rps":  agg_rps,
                "role":           "victim",
                "task":           args.victim_task,
                **stats,
            })

            if agg_rps > 0:
                stats_agg = _compute_stats(
                    result["sent"].get(args.aggressor_task, []),
                    result["latencies"].get(args.aggressor_task, []),
                    result["errors"].get(args.aggressor_task, 0),
                    args.duration,
                )
                _save_run(run_dir, args.aggressor_task,
                          result["sent"].get(args.aggressor_task, []),
                          result["latencies"].get(args.aggressor_task, []),
                          args.duration)
                exp_rows.append({
                    "backbone":       args.backbone,
                    "victim_task":    args.victim_task,
                    "victim_rps":     args.victim_rps,
                    "aggressor_task": args.aggressor_task,
                    "aggressor_rps":  agg_rps,
                    "role":           "aggressor",
                    "task":           args.aggressor_task,
                    **stats_agg,
                })

            _write_summary(summary_path, rows + exp_rows)
            await asyncio.sleep(1)

        return exp_rows

    print(f"\n[INFO] Loading data for: {[args.victim_task, args.aggressor_task]}")
    data = build_data([args.victim_task, args.aggressor_task], args.max_samples)
    new_rows = asyncio.run(run_experiment(aggressor_rps_values))
    rows.extend(new_rows)

    print(f"\n[INFO] Done. Summary: {summary_path}")
    _print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
