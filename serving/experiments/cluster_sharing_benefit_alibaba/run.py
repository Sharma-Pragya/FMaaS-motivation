#!/usr/bin/env python3
"""cluster_sharing_benefit/run.py — sweep runner.

Loads deployments/N{n_apps}/<condition>.json (built by deployments/generate.py
from user_config), mirrors synthetic-task metadata under base-task entries,
generates a Poisson trace at user_config.per_app_rps per app, deploys, and runs.

Usage:
    python -u -m experiments.cluster_sharing_benefit.run \
        --n-apps 16 --condition sharing \
        --duration 180 \
        --output-dir experiments/cluster_sharing_benefit/results/N16/sharing
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

from client.runner import TraceRunner
from client.trace import generate_trace
from experiments.cluster_sharing_benefit_alibaba import user_config as cfg
from site_manager.local import LocalSiteManager


DEPLOYMENTS_DIR = Path(__file__).resolve().parent / "deployments"


def _load_plan(condition: str, n_apps: int) -> dict:
    path = DEPLOYMENTS_DIR / f"N{n_apps}" / f"{condition}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no plan at {path} — run deployments/generate.py first."
        )
    with open(path) as f:
        return json.load(f)


def _apply_runtime_defaults(plan: dict, task_rates: Dict[str, float]) -> None:
    from collections import Counter
    exp = cfg.experiment

    counts: Counter = Counter()
    for site in plan.get("sites", []):
        for dep in site.get("deployments", []):
            host = dep["device"].rsplit(":", 1)[0]
            counts[(host, dep.get("cuda", ""))] += 1

    for site in plan.get("sites", []):
        for dep in site.get("deployments", []):
            dep.setdefault("max_batch_size", exp.get("max_batch_size", 32))
            dep.setdefault("max_batch_wait_ms", exp.get("max_batch_wait_ms", 0))
            dep.setdefault("isolation_mode", exp.get("isolation_mode", "shared"))
            if exp.get("max_model_len") is not None:
                dep.setdefault("max_model_len", exp["max_model_len"])

            host = dep["device"].rsplit(":", 1)[0]
            n = counts.get((host, dep.get("cuda", "")), 1)
            dep.setdefault("gpu_memory_utilization", round(0.85 / n, 4))

            tasks_block = dep.get("tasks", {})
            for t in list(tasks_block.keys()):
                rate = float(task_rates.get(t, 0.0))
                tasks_block[t]["request_per_sec"] = rate
                tasks_block[t]["total_requested_workload"] = rate


def _plan_tasks(plan: dict) -> List[str]:
    seen = []
    for site in plan.get("sites", []):
        for dep in site.get("deployments", []):
            for t in dep.get("tasks", {}):
                if t not in seen:
                    seen.append(t)
    return seen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-apps", type=int, required=True)
    parser.add_argument("--condition", required=True,
                        help=f"one of {list(cfg.conditions)}")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exp = cfg.experiment
    duration = float(args.duration if args.duration is not None else exp["duration"])
    trace_type = exp["trace"]
    warmup_gap = float(exp.get("warmup_gap", 2.0))
    warmup_burst_secs = float(exp.get("warmup_burst_secs", 0.0))

    plan = _load_plan(args.condition, args.n_apps)

    # Mirror base-task metadata under synthetic task names so cfg.tasks lookups
    # (type, latency, seed, peak_workload) resolve.
    meta_path = DEPLOYMENTS_DIR / f"N{args.n_apps}" / "task_meta.json"
    if meta_path.is_file():
        mapping = json.loads(meta_path.read_text())
        for synth, base in mapping.items():
            if synth == base:
                continue
            if base not in cfg.tasks:
                raise KeyError(f"task_meta: base {base!r} not in user_config.tasks")
            if synth not in cfg.tasks:
                entry = dict(cfg.tasks[base])
                max_seed = max((t.get("seed", 0) for t in cfg.tasks.values()), default=0)
                entry["seed"] = max_seed + 100
                cfg.tasks[synth] = entry

    active_tasks = _plan_tasks(plan)
    print(f"[Runner] N={args.n_apps} condition={args.condition} tasks={active_tasks}")

    per_app = float(cfg.per_app_rps)
    task_rates = {t: per_app for t in active_tasks}
    print(f"[Runner] task_rates={task_rates}")

    _apply_runtime_defaults(plan, task_rates)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "deployment_plan.json"), "w") as f:
        json.dump(plan, f, indent=2)
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump({
            "n_apps": args.n_apps,
            "condition": args.condition,
            "duration": duration,
            "trace_type": trace_type,
            "seed": args.seed,
            "active_tasks": active_tasks,
            "task_rates": task_rates,
            "experiment": dict(exp),
        }, f, indent=2, default=str)

    sorted_active = sorted(active_tasks)
    rate_list = [task_rates[t] for t in sorted_active]
    tasks_dict = {t: {**cfg.tasks[t], "peak_workload": task_rates[t]}
                  for t in active_tasks}
    trace, _, _ = generate_trace(trace_type, rate_list, duration,
                                 tasks_dict, args.seed)
    print(f"[Runner] Generated {len(trace)} requests over {duration}s.")

    site_mgr = LocalSiteManager(output_dir)
    runner = None
    try:
        site_mgr.deploy(plan, output_dir)

        runner = TraceRunner(site_mgr.live_plan, trace, output_dir,
                             warmup_burst_secs=warmup_burst_secs)

        async def _go():
            await runner.warmup()
            start_epoch = time.time() + warmup_gap
            print(f"[Runner] Warmup done. Trace starts in {warmup_gap}s...")
            await asyncio.sleep(warmup_gap)
            await runner.run(start_epoch=start_epoch)

        asyncio.run(_go())
        runner.save_results()
    finally:
        site_mgr.cleanup()

    print(f"[Runner] Done. Results in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
