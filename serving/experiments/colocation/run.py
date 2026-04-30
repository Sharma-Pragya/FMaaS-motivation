#!/usr/bin/env python3
"""colocation/run.py — Colocate ecgclass (tsfm/momentlarge) and nyudepth (vision/dinobase).

Conditions:
  single_ecgclass  — 1 device server, ecgclass on momentlarge
  single_nyudepth  — 1 device server, nyudepth on dinobase
  no_sharing       — 2 device servers (one per backbone) running concurrently

Each task is tied to its own backbone via --task-backbones (task:backbone,...).
All other task-set plumbing (dataset loaders, decoder paths, seeds) is shared with
experiments/sharing_benefit/tpc/run.py.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[3]
SERVING_DIR = REPO_ROOT / "serving"
for p in (str(REPO_ROOT), str(SERVING_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Reuse helpers from the tpc experiment
from experiments.sharing_benefit.tpc.run import (
    TASK_SETS,
    build_data,
    deploy,
    generate_traces,
    load_trace,
    save_trace,
    save_results,
    run_open_loop,
    run_warmup_burst,
)


# Task -> task_set mapping for this experiment
TASK_TO_SET = {
    "ecgclass": "tsfm",
    "nyudepth": "vision",
}


def _merged_cfg(tasks):
    """Merge types/decoder_paths/seeds across the task-sets the tasks belong to."""
    types, dec, seeds = {}, {}, {}
    for t in tasks:
        ts = TASK_TO_SET[t]
        cfg = TASK_SETS[ts]
        types[t] = cfg["types"][t]
        dec[t]   = cfg["decoder_paths"][t]
        seeds[t] = cfg["seeds"][t]
    return types, dec, seeds


def _build_data_mixed(tasks):
    """Build data dict covering tasks from potentially multiple task-sets."""
    data = {}
    by_set = {}
    for t in tasks:
        by_set.setdefault(TASK_TO_SET[t], []).append(t)
    for ts, ts_tasks in by_set.items():
        data.update(build_data(ts, ts_tasks))
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True,
                        help="Comma-separated tasks (e.g. 'ecgclass,nyudepth')")
    parser.add_argument("--task-backbones", required=True,
                        help="Comma-separated task:backbone pairs "
                             "(e.g. 'ecgclass:momentlarge,nyudepth:dinobase-patch')")
    parser.add_argument("--condition", required=True,
                        help="single_{task} or no_sharing")
    parser.add_argument("--device-urls", required=True,
                        help="Comma-separated device URLs, one per task for no_sharing")
    parser.add_argument("--rps",      type=float, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--warmup-secs", type=float, default=10.0)
    parser.add_argument("--warmup-burst-secs", type=float,
                        default=float(os.environ.get("WARMUP_BURST_SECS", "15.0")))
    parser.add_argument("--exp-dir",    required=True)
    parser.add_argument("--trace-file", default=None)
    args = parser.parse_args()

    all_tasks = [t.strip() for t in args.tasks.split(",")]
    for t in all_tasks:
        if t not in TASK_TO_SET:
            print(f"ERROR: unknown task '{t}' (expected one of {list(TASK_TO_SET)})",
                  file=sys.stderr)
            return 1

    task_backbone = {}
    for pair in args.task_backbones.split(","):
        k, v = pair.split(":", 1)
        task_backbone[k.strip()] = v.strip()
    for t in all_tasks:
        if t not in task_backbone:
            print(f"ERROR: no backbone mapping for task '{t}'", file=sys.stderr)
            return 1

    device_urls = [u.strip() for u in args.device_urls.split(",")]
    out_dir = (SERVING_DIR / args.exp_dir).resolve()

    types, dec_paths, seeds = _merged_cfg(all_tasks)

    print("=" * 65)
    print(f"  Colocation — condition={args.condition}")
    print(f"  Tasks     : {all_tasks}")
    print(f"  Backbones : {task_backbone}")
    print(f"  RPS/task  : {args.rps}")
    print(f"  Duration  : {args.duration}s")
    print(f"  Results   : {out_dir}")
    print("=" * 65)

    # Trace (shared across conditions at the same rps)
    if args.trace_file:
        trace_path = Path(args.trace_file)
        if not trace_path.is_absolute():
            trace_path = (SERVING_DIR / trace_path).resolve()
        if trace_path.exists():
            send_times = load_trace(trace_path)
            missing = [t for t in all_tasks if t not in send_times]
            if missing:
                extra = generate_traces(missing, seeds, args.rps, args.duration)
                send_times.update(extra)
                save_trace(send_times, trace_path)
        else:
            send_times = generate_traces(all_tasks, seeds, args.rps, args.duration)
            save_trace(send_times, trace_path)
    else:
        send_times = generate_traces(all_tasks, seeds, args.rps, args.duration)

    # Resolve active tasks + url mapping
    if args.condition.startswith("single_"):
        task = args.condition[len("single_"):]
        if task not in all_tasks:
            print(f"ERROR: task '{task}' not in {all_tasks}", file=sys.stderr)
            return 1
        tasks = [task]
        task_urls = {task: device_urls[0]}
    elif args.condition == "no_sharing":
        tasks = all_tasks
        if len(device_urls) < len(tasks):
            print(f"ERROR: need {len(tasks)} device URLs for no_sharing, got {len(device_urls)}",
                  file=sys.stderr)
            return 1
        task_urls = {t: device_urls[i] for i, t in enumerate(tasks)}
    else:
        print(f"ERROR: unknown condition '{args.condition}'", file=sys.stderr)
        return 1

    print(f"[INFO] Loading data for: {tasks}")
    data = _build_data_mixed(tasks)

    # Deploy — each task to its own server with its own backbone
    for t in tasks:
        url = task_urls[t]
        asyncio.run(deploy(url, task_backbone[t], [t], types, dec_paths))

    asyncio.run(asyncio.sleep(1))

    async def _warmup_then_run():
        warm_clients = None
        if args.warmup_burst_secs > 0:
            warm_clients = await run_warmup_burst(
                task_urls=task_urls,
                data=data,
                duration_s=args.warmup_burst_secs,
            )
        print(f"\n[Run] Starting open-loop send ({args.duration}s) ...")
        return await run_open_loop(
            task_urls=task_urls,
            data=data,
            send_times={t: send_times[t] for t in tasks},
            warm_clients=warm_clients,
        )

    records = asyncio.run(_warmup_then_run())

    save_results(records, out_dir, args.condition, args.duration, args.warmup_secs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
