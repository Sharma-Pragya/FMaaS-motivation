#!/usr/bin/env python3
"""End-to-end real-world experiment runner.

Runs a single (regime, N, condition) scenario:
  1. Loads the pre-generated plan + slots + trace for {regime}_N{N}.
  2. Deploys all placed tasks at t=0 (no timeline events).
  3. Replays the trace for `duration` seconds.
  4. Saves response-time / throughput results.

Usage (from serving/):
    python -m experiments.end_to_end_realworld.run \
        --regime high --n 16 --condition fmaas \
        --output-dir experiments/end_to_end_realworld/results/high_N16/fmaas
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

from client.runner import TraceRunner
from experiments.end_to_end_realworld import user_config as cfg
from site_manager.local import LocalSiteManager

DEPLOYMENTS_DIR = Path(__file__).resolve().parent / "deployments"


def _load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing: {path}\n"
            "Run:  python -m experiments.end_to_end_realworld.deployments.generate"
        )
    return json.loads(path.read_text())


def _apply_runtime_defaults(plan: dict, task_rps: Dict[str, float]) -> None:
    exp = cfg.experiment
    counts: Counter = Counter()
    for site in plan.get("sites", []):
        for dep in site.get("deployments", []):
            host = dep["device"].rsplit(":", 1)[0]
            counts[(host, dep.get("cuda", ""))] += 1

    for site in plan.get("sites", []):
        for dep in site.get("deployments", []):
            dep.setdefault("max_batch_size",    exp.get("max_batch_size", 32))
            dep.setdefault("max_batch_wait_ms", exp.get("max_batch_wait_ms", 0))
            dep.setdefault("isolation_mode",    exp.get("isolation_mode", "shared"))
            if exp.get("max_model_len") is not None:
                dep.setdefault("max_model_len", exp["max_model_len"])
            host = dep["device"].rsplit(":", 1)[0]
            n = counts.get((host, dep.get("cuda", "")), 1)
            dep.setdefault("gpu_memory_utilization", round(0.85 / n, 4))
            for t, tentry in dep.get("tasks", {}).items():
                rate = float(task_rps[t])
                tentry["request_per_sec"]          = rate
                tentry["total_requested_workload"] = rate


def _mirror_synthetic_tasks(task_meta: Dict[str, str]) -> None:
    """Register synthetic task names (e.g. 'imgclass10__app1') in cfg.tasks."""
    max_seed = max((t.get("seed", 0) for t in cfg.tasks.values()), default=0)
    for synth, base in task_meta.items():
        if synth == base or synth in cfg.tasks:
            continue
        if base not in cfg.tasks:
            raise KeyError(f"task_meta references unknown base task {base!r}")
        entry = dict(cfg.tasks[base])
        max_seed += 100
        entry["seed"] = max_seed
        cfg.tasks[synth] = entry


def _placed_tasks(condition: str, slots: list) -> set:
    if condition == "fmaas":
        return {t["task"] for s in slots for t in s["tasks"]}
    return {s["task"] for s in slots}


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end real-world runner")
    parser.add_argument("--regime",     required=True, choices=cfg.experiment['load_regimes'])
    parser.add_argument("--n",          required=True, type=int,
                        help="Number of tasks (must match a value in n_tasks_sweep)")
    parser.add_argument("--condition",  required=True, choices=cfg.conditions)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration",   type=float, default=None,
                        help="Override experiment duration (s)")
    args = parser.parse_args()

    exp        = cfg.experiment
    duration   = float(args.duration or exp["duration"])
    warmup_gap = float(exp.get("warmup_gap", 2.0))

    scenario_dir = DEPLOYMENTS_DIR / f"{args.regime}_N{args.n}"
    plan         = _load_json(scenario_dir / f"{args.condition}.json")
    slots        = _load_json(scenario_dir / f"{args.condition}_slots.json")
    task_meta    = _load_json(scenario_dir / "task_meta.json")
    full_trace   = _load_json(scenario_dir / "trace.json")
    task_rps     = _load_json(scenario_dir / "task_rps.json")

    _mirror_synthetic_tasks(task_meta)

    placed = _placed_tasks(args.condition, slots)
    rejected = sorted(set(task_meta) - placed)
    if rejected:
        print(f"[Runner] {len(rejected)} tasks not placed by {args.condition}: "
              f"{rejected}")
    full_trace = [r for r in full_trace if r["task"] in placed]

    _apply_runtime_defaults(plan, task_rps)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "deployment_plan.json"), "w") as f:
        json.dump(plan, f, indent=2)
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump({
            "regime":        args.regime,
            "n_requested":   args.n,
            "n_placed":      len(placed),
            "placed_tasks":  sorted(placed),
            "rejected":      rejected,
            "condition":     args.condition,
            "duration":      duration,
            "task_rps":      task_rps,
            "experiment":    dict(exp),
        }, f, indent=2, default=str)

    print(f"[Runner] regime={args.regime}  N={args.n}  condition={args.condition}  "
          f"placed={len(placed)}  rejected={len(rejected)}  "
          f"trace={len(full_trace)} reqs")

    init_trace = sorted(
        [r for r in full_trace if float(r["req_time"]) < duration],
        key=lambda r: float(r["req_time"]),
    )

    site_mgr = LocalSiteManager(output_dir)
    runner   = None
    try:
        site_mgr.deploy(plan, output_dir)
        runner = TraceRunner(site_mgr.live_plan, init_trace, output_dir)

        async def _go():
            await runner.warmup()
            start_epoch = time.time() + warmup_gap
            runner.start_epoch = start_epoch
            print(f"[Runner] Warmup done. Trace starts in {warmup_gap:.1f}s…")
            await asyncio.sleep(warmup_gap)
            await runner.run(start_epoch=start_epoch, duration=duration)

        asyncio.run(_go())

    finally:
        if runner is not None:
            if hasattr(runner, "start_epoch"):
                rc_path = os.path.join(output_dir, "run_config.json")
                try:
                    with open(rc_path) as f:
                        rc = json.load(f)
                    rc["start_epoch"] = runner.start_epoch
                    with open(rc_path, "w") as f:
                        json.dump(rc, f, indent=2, default=str)
                except Exception:
                    pass
            runner.save_results()
        site_mgr.cleanup()

    print(f"[Runner] Done. Results in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
