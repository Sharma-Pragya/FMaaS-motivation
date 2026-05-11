#!/usr/bin/env python3
"""Long-horizon experiment runner.

Loads the pre-generated deployment plan and task timeline (from deployments/generate.py),
deploys the initially-active tasks, then drives per-task arrive/depart events while
a TraceRunner dispatches requests.

Arrive  — fmaas:               hot-attach decoder to already-running backbone.
Arrive  — no_sharing[_tpc]:    cold-start a new backbone process, then route to it.
Depart  — all conditions:      remove routing; requests for that task are dropped.

Usage (from serving/):
    python -m experiments.long_horizon.run --condition fmaas \
        --output-dir experiments/long_horizon/results/fmaas

    python -m experiments.long_horizon.run --condition no_sharing \
        --output-dir experiments/long_horizon/results/no_sharing
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
from experiments.long_horizon import user_config as cfg
from site_manager.deployment_handler import (
    _add_decoder_to_device, _remove_decoder_from_device, deploy_models, shutdown_devices,
)
from site_manager.local import LocalSiteManager

DEPLOYMENTS_DIR = Path(__file__).resolve().parent / "deployments"


# ── Artifact loading ──────────────────────────────────────────────────────────

def _load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing: {path}\n"
            "Run:  python -m experiments.long_horizon.deployments.generate"
        )
    return json.loads(path.read_text())


def _load_artifacts(condition: str):
    d = DEPLOYMENTS_DIR
    plan          = _load_json(d / f"{condition}.json")
    slots         = _load_json(d / f"{condition}_slots.json")
    task_timeline = _load_json(d / "task_timeline.json")
    task_meta     = _load_json(d / "task_meta.json")
    full_trace    = _load_json(d / "trace.json")
    task_rps      = _load_json(d / "task_rps.json")
    return plan, slots, task_timeline, task_meta, full_trace, task_rps


# ── Runtime defaults ──────────────────────────────────────────────────────────

def _apply_runtime_defaults(plan: dict, task_rps: Dict[str, float]) -> None:
    from collections import Counter
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
                tentry["request_per_sec"]         = rate
                tentry["total_requested_workload"] = rate


def _patch_spec_defaults(spec: dict) -> None:
    exp = cfg.experiment
    spec.setdefault("max_batch_size",    exp.get("max_batch_size", 32))
    spec.setdefault("max_batch_wait_ms", exp.get("max_batch_wait_ms", 0))
    spec.setdefault("isolation_mode",    exp.get("isolation_mode", "shared"))
    if exp.get("max_model_len") is not None:
        spec.setdefault("max_model_len", exp["max_model_len"])
    spec.setdefault("gpu_memory_utilization", 0.85)


# ── cfg.tasks helpers ─────────────────────────────────────────────────────────

def _mirror_synthetic_tasks(task_meta: Dict[str, str]) -> None:
    """Register every synthetic task name in cfg.tasks so trace generation works."""
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


# ── live_plan helpers ─────────────────────────────────────────────────────────

def _find_dep(live_plan: dict, device_url: str) -> dict | None:
    for site in live_plan.get("sites", []):
        for dep in site.get("deployments", []):
            if dep["device"] == device_url:
                return dep
    return None


def _insert_dep(live_plan: dict, spec: dict, site_id: str) -> None:
    for site in live_plan.get("sites", []):
        if site["id"] == site_id:
            site["deployments"].append(spec)
            return
    live_plan.setdefault("sites", []).append(
        {"id": site_id, "deployments": [spec]}
    )


def _remove_task(live_plan: dict, task_name: str) -> None:
    for site in live_plan.get("sites", []):
        for dep in site.get("deployments", []):
            dep.get("tasks", {}).pop(task_name, None)
            dep["decoders"] = [d for d in dep.get("decoders", [])
                               if d["task"] != task_name]


# ── Trace helpers ─────────────────────────────────────────────────────────────

def _slice_trace(full_trace: list, task_name: str,
                 t_start: float, t_end: float) -> list:
    """Return pre-generated requests for task_name in [t_start, t_end)."""
    return [r for r in full_trace
            if r["task"] == task_name
            and t_start <= float(r["req_time"]) < t_end]


# ── Timeline ──────────────────────────────────────────────────────────────────

async def _run_timeline(
    condition: str,
    slots,
    task_timeline: dict,
    live_plan: dict,
    runner: TraceRunner,
    start_epoch: float,
    duration: float,
    full_trace: list,
    task_rps: Dict[str, float],
) -> None:

    init_secs = float(cfg.timeline.get("initial_active_secs", 5.0))
    device_locks: dict[str, asyncio.Lock] = {}
    arrival_tasks: list[asyncio.Task] = []

    events: List[dict] = []
    for task, info in task_timeline.items():
        if init_secs <= info["arrive"] < duration:
            events.append({"t": info["arrive"], "action": "arrive", "task": task})
        if info["depart"] < duration:
            events.append({"t": info["depart"], "action": "depart", "task": task})
    events.sort(key=lambda e: e["t"])

    for ev in events:
        wait = ev["t"] - (time.time() - start_epoch)
        if wait > 0:
            await asyncio.sleep(wait)

        task = ev["task"]
        action = ev["action"]
        print(f"[Timeline] t={ev['t']:.1f}s  {action}  task={task}")

        if action == "arrive":
            t_end = min(task_timeline[task]["depart"], duration)
            # Fire-and-forget: cold-starts on one backbone must not block
            # hot-attaches on other already-running backbones.
            # _arrive gates request injection behind its own t_ready check,
            # so concurrent arrive tasks are safe.
            arrival_tasks.append(asyncio.create_task(_arrive(
                condition, task, slots, live_plan, runner,
                t_end, full_trace, start_epoch, task_timeline, task_rps,
                device_locks,
            )))
        elif action == "depart":
            await _depart(condition, task, slots, live_plan, runner)

    if arrival_tasks:
        results = await asyncio.gather(*arrival_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                print(f"[Timeline]   WARNING: arrival task failed: {result}")


async def _arrive(
    condition: str,
    task_name: str,
    slots,
    live_plan: dict,
    runner: TraceRunner,
    t_end: float,
    full_trace: list,
    start_epoch: float,
    task_timeline: dict,
    task_rps: Dict[str, float],
    device_locks: dict[str, asyncio.Lock],
) -> None:
    if condition == "fmaas":
        # slots is a list of per-backbone groups; find the one owning this task
        for group in slots:
            for t in group["tasks"]:
                if t["task"] != task_name:
                    continue
                device_url = group["device_url"]
                decoder = {
                    "task":      t["task"],
                    "base_task": t["base_task"],
                    "type":      cfg.tasks[t["base_task"]]["type"],
                    "path":      t["decoder_path"],
                }

                lock = device_locks.setdefault(device_url, asyncio.Lock())
                async with lock:
                    dep = _find_dep(live_plan, device_url)
                    if dep is None:
                        # Backbone not yet running — cold-start it first, then
                        # hot-attach the decoder. Re-check under the lock so
                        # concurrent arrivals for this device do not launch
                        # multiple servers on the same port.
                        spec = {k: v for k, v in group["deploy_spec"].items()}
                        _patch_spec_defaults(spec)
                        site_id = cfg.devices[group["device_name"]]["site_manager"]
                        print(f"[Timeline]   cold-start backbone {group['backbone']} → {device_url}")
                        results = await deploy_models([spec])
                        failures = [r for r in results
                                    if not isinstance(r, dict)
                                    or str(r.get("status", "")).startswith("error")]
                        if failures:
                            print(f"[Timeline]   WARNING: backbone cold-start failed: {failures}")
                            return
                        _insert_dep(live_plan, spec, site_id)
                        dep = _find_dep(live_plan, device_url)

                    print(f"[Timeline]   hot-attach {task_name} → {device_url}")
                    result = await _add_decoder_to_device(device_url, [decoder])
                    status = str(result.get("status", "")) if isinstance(result, dict) else ""
                    if status.startswith("error") or not status.startswith("added_"):
                        print(f"[Timeline]   WARNING: decoder attach failed for {task_name}: {result}")
                        return

                    if dep is not None:
                        rps = float(task_rps[task_name])
                        dep["tasks"][task_name] = {
                            "type": cfg.tasks[t["base_task"]]["type"],
                            "request_per_sec": rps,
                            "total_requested_workload": rps,
                        }
                        dep["decoders"].append(decoder)

    else:
        # no_sharing / no_sharing_tpc: cold-start one backbone process
        for sl in slots:
            if sl["task"] != task_name:
                continue
            spec = {k: v for k, v in sl["deploy_spec"].items()}
            _patch_spec_defaults(spec)
            site_id = cfg.devices[sl["device_name"]]["site_manager"]
            device_url = sl["device_url"]
            lock = device_locks.setdefault(device_url, asyncio.Lock())
            async with lock:
                print(f"[Timeline]   cold-start {task_name} on {device_url}")
                results = await deploy_models([spec])
                failures = [
                    r for r in results
                    if not isinstance(r, dict) or str(r.get("status", "")).startswith("error")
                ]
                if failures:
                    print(f"[Timeline]   WARNING: cold-start failed for {task_name}: {failures}")
                    return
                _insert_dep(live_plan, spec, site_id)
            break

    runner.invalidate_plan_cache()

    # Start from actual deployment-ready time — pre-generated requests before
    # this point were never dispatched (cold-start window) and appear as the
    # gap between offered load and completions in the plot.
    t_ready = time.time() - start_epoch
    t_arrive = float(task_timeline[task_name]["arrive"])
    runner.activation_ready[task_name] = {
        "t_arrive": t_arrive,
        "t_ready":  round(t_ready, 3),
        "latency_s": round(max(0.0, t_ready - t_arrive), 3),
    }
    t_start = max(t_ready, 0.0)
    print(f"[Timeline]   ready at t={t_ready:.1f}s; activation={max(0.0, t_ready-t_arrive):.2f}s  trace [{t_start:.1f}, {t_end:.1f})")

    new_trace = _slice_trace(full_trace, task_name, t_start, t_end)
    print(f"[Timeline]   {len(new_trace)} requests for {task_name}")
    if new_trace:
        runner.add_requests(new_trace)


async def _depart(
    condition: str,
    task_name: str,
    slots,
    live_plan: dict,
    runner: TraceRunner,
) -> None:
    _remove_task(live_plan, task_name)
    runner.invalidate_plan_cache()
    print(f"[Timeline]   routing removed for {task_name}")

    if condition == "fmaas":
        # Remove the decoder from the backbone process.
        # If this was the last task on that backbone, shut down the backbone.
        for group in slots:
            if not any(t["task"] == task_name for t in group["tasks"]):
                continue
            device_url = group["device_url"]
            await _remove_decoder_from_device(device_url, [task_name])
            print(f"[Timeline]   decoder removed: {task_name} from {device_url}")

            dep = _find_dep(live_plan, device_url)
            remaining = list(dep["tasks"].keys()) if dep else []
            if not remaining:
                print(f"[Timeline]   last task departed — shutting down backbone {group['backbone']} on {device_url}")
                await shutdown_devices([{"device": device_url}])
                # Remove the deployment from live_plan so it is not re-used
                for site in live_plan.get("sites", []):
                    site["deployments"] = [
                        d for d in site["deployments"] if d["device"] != device_url
                    ]
            break

    else:
        # no_sharing / no_sharing_tpc: each task owns one process.
        # Remove the decoder then kill the backbone process to free GPU memory.
        for sl in slots:
            if sl["task"] != task_name:
                continue
            device_url = sl["device_url"]
            await _remove_decoder_from_device(device_url, [task_name])
            print(f"[Timeline]   decoder removed: {task_name} from {device_url}")
            await shutdown_devices([{"device": device_url}])
            print(f"[Timeline]   backbone shut down: {device_url}")
            break


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Long-horizon experiment runner")
    parser.add_argument("--condition",   required=True, choices=cfg.conditions)
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--duration",    type=float, default=None,
                        help="Override experiment duration (s)")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    exp        = cfg.experiment
    duration   = float(args.duration or exp["duration"])
    warmup_gap = float(exp.get("warmup_gap", 2.0))
    warmup_burst_secs = float(exp.get("warmup_burst_secs", 0.0))

    plan, slots, task_timeline, task_meta, full_trace, task_rps = _load_artifacts(args.condition)
    _mirror_synthetic_tasks(task_meta)

    # Filter timeline + trace to tasks the scheduler actually placed.  Tasks
    # the scheduler couldn't fit (peak demand vs. capacity) have no slot, so
    # their arrive events would no-op and their requests would log "no route".
    if args.condition == "fmaas":
        placed_tasks = {t["task"] for s in slots for t in s["tasks"]}
    else:
        placed_tasks = {s["task"] for s in slots}
    unplaced = [t for t in task_timeline if t not in placed_tasks]
    if unplaced:
        print(f"[Runner] {len(unplaced)} tasks not placed by {args.condition} scheduler — "
              f"dropping from timeline/trace: {unplaced}")
    task_timeline = {t: info for t, info in task_timeline.items() if t in placed_tasks}
    full_trace    = [r for r in full_trace if r["task"] in placed_tasks]

    init_secs     = float(cfg.timeline.get("initial_active_secs", 5.0))
    initial_tasks = [t for t, info in task_timeline.items()
                     if info["arrive"] < init_secs]

    _apply_runtime_defaults(plan, task_rps)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "deployment_plan.json"), "w") as f:
        json.dump(plan, f, indent=2)
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump({
            "condition":     args.condition,
            "duration":      duration,
            "initial_tasks": initial_tasks,
            "task_rps":      task_rps,
            "task_timeline": task_timeline,
            "experiment":    dict(exp),
        }, f, indent=2, default=str)

    print(f"[Runner] condition={args.condition}  "
          f"initial tasks={len(initial_tasks)}  "
          f"dynamic tasks={len(task_timeline) - len(initial_tasks)}")

    # Build initial trace from pre-generated requests (same across all conditions).
    # Cap at duration: trace spans trace_duration but the experiment runs only duration secs.
    init_trace = sorted(
        [r for r in full_trace
         if r["task"] in initial_tasks and float(r["req_time"]) < duration],
        key=lambda r: float(r["req_time"]),
    )
    print(f"[Runner] {len(init_trace)} initial requests "
          f"({len(initial_tasks)} tasks, from pre-generated trace)")

    site_mgr = LocalSiteManager(output_dir)
    runner   = None
    try:
        site_mgr.deploy(plan, output_dir)

        runner = TraceRunner(
            site_mgr.live_plan, init_trace, output_dir,
            warmup_burst_secs=warmup_burst_secs,
        )

        async def _go():
            await runner.warmup()
            start_epoch = time.time() + warmup_gap
            runner.start_epoch = start_epoch  # saved to run_config after _go()
            print(f"[Runner] Warmup done. Trace starts in {warmup_gap:.1f}s…")
            await asyncio.sleep(warmup_gap)

            tl_task = asyncio.create_task(
                _run_timeline(
                    args.condition, slots, task_timeline,
                    site_mgr.live_plan, runner,
                    start_epoch, duration,
                    full_trace, task_rps,
                )
            )
            await runner.run(start_epoch=start_epoch, duration=duration)
            await tl_task

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
