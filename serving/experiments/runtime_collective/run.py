#!/usr/bin/env python3
"""Collective stepwise runtime experiment.

Each phase performs a fresh deployment for the cumulative task set and then
runs one combined trace for all active tasks in that phase.

This avoids overlapping independent runtime batches from previous events.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

from orchestrator.main import Orchestrator
from orchestrator.router import route_trace
from traces.deterministic import Request
from experiments.runtime.user_config import devices


FACTOR = 1.5
TASK_LIBRARY: Dict[str, Dict] = {
    "ecgclass": {
        "type": "classification",
        "latency": 3.86 * FACTOR,
        "metric": "accuracy",
        "value": 0.7,
    },
    "gestureclass": {
        "type": "classification",
        "latency": 3.88 * FACTOR,
        "metric": "accuracy",
        "value": 0.6,
    },
    "sysbp": {
        "type": "regression",
        "latency": 5.55 * FACTOR,
        "metric": "mae",
        "value": 100,
    },
    "diasbp": {
        "type": "regression",
        "latency": 5.58 * FACTOR,
        "metric": "mae",
        "value": 100,
    },
    "heartrate": {
        "type": "regression",
        "latency": 5.58 * FACTOR,
        "metric": "mae",
        "value": 100,
    },
}


def phase_specs() -> List[Tuple[str, Dict[str, float]]]:
    """Return cumulative workload for each phase."""
    return [
        ("phase0_base_ecgclass", {"ecgclass": 10.0}),
        ("phase1_add_gestureclass", {"ecgclass": 10.0, "gestureclass": 8.0}),
        ("phase2_ecgclass_plus5", {"ecgclass": 15.0, "gestureclass": 8.0}),
        ("phase3_ecgclass_plus10", {"ecgclass": 25.0, "gestureclass": 8.0}),
        ("phase4_ecgclass_plus5", {"ecgclass": 30.0, "gestureclass": 8.0}),
        ("phase5_add_sysbp", {"ecgclass": 30.0, "gestureclass": 8.0, "sysbp": 8.0}),
        ("phase6_add_diasbp", {"ecgclass": 30.0, "gestureclass": 8.0, "sysbp": 8.0, "diasbp": 20.0}),
        ("phase7_ecgclass_plus15", {"ecgclass": 45.0, "gestureclass": 8.0, "sysbp": 8.0, "diasbp": 20.0}),
    ]


def build_task_specs(active_rates: Dict[str, float]) -> Dict[str, Dict]:
    specs = {}
    for task_name, rate in active_rates.items():
        base = TASK_LIBRARY[task_name].copy()
        base["peak_workload"] = rate
        specs[task_name] = base
    return specs


def generate_collective_trace(
    active_rates: Dict[str, float],
    duration: float,
    req_id_offset: int = 0,
) -> List[Request]:
    """Generate one combined deterministic trace for the current phase.

    This builds a single global arrival stream at the aggregate phase rate,
    then assigns each arrival to a task using weighted fair scheduling based
    on the target per-task rates.

    Result:
    - exactly one global arrival timeline
    - no independent per-task periodic clocks
    - per-task request fractions track the requested rates over the phase
    """
    requests: List[Request] = []
    positive_rates = {
        task_name: rate
        for task_name, rate in active_rates.items()
        if rate > 0
    }
    if not positive_rates:
        return requests

    total_rate = sum(positive_rates.values())
    total_requests = int(total_rate * duration)
    if total_requests <= 0:
        return requests

    interval = 1.0 / total_rate
    task_names = sorted(positive_rates.keys())
    shares = {
        task_name: positive_rates[task_name] / total_rate
        for task_name in task_names
    }
    deficits = {task_name: 0.0 for task_name in task_names}

    for i in range(total_requests):
        for task_name in task_names:
            deficits[task_name] += shares[task_name]

        # Pick the task furthest behind its target share.
        chosen = max(task_names, key=lambda task_name: (deficits[task_name], -task_names.index(task_name)))
        deficits[chosen] -= 1.0

        req_time = (i + 1) * interval
        if req_time > duration:
            break
        requests.append(Request(req_id_offset, chosen, None, None, req_time))
        req_id_offset += 1

    return requests


def start_site_manager(site_log: Path) -> subprocess.Popen:
    with site_log.open("w") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "site_manager.main"],
            cwd=str(SERVING_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    deadline = time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"site_manager exited early; see {site_log}")
        if site_log.exists() and "ready. Entering MQTT loop" in site_log.read_text():
            return proc
        time.sleep(2)

    proc.terminate()
    raise TimeoutError("Timed out waiting for site_manager to become ready")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collective stepwise runtime experiment")
    parser.add_argument("--scheduler", default=os.environ.get("SCHEDULER", "fmaas_share"))
    parser.add_argument("--duration", type=float, default=float(os.environ.get("PHASE_DURATION", "20")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--exp-dir", default=os.environ.get("EXP_DIR", "experiments/runtime_collective/results"))
    args = parser.parse_args()

    result_root = (SERVING_DIR / args.exp_dir / args.scheduler / "10").resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    site_log = SERVING_DIR / "site_manager.log"

    print("\n" + "━" * 61)
    print("Collective Runtime Experiment")
    print("━" * 61)
    print(f"Scheduler: {args.scheduler}")
    print(f"Phase duration: {args.duration}s")
    print(f"Results root: {result_root}")

    site_proc = None
    try:
        site_proc = start_site_manager(site_log)
        print(f"[INFO] site_manager ready (pid={site_proc.pid})")

        for phase_name, active_rates in phase_specs():
            phase_dir = result_root / phase_name
            phase_dir.mkdir(parents=True, exist_ok=True)
            task_specs = build_task_specs(active_rates)
            trace = generate_collective_trace(active_rates, args.duration)

            print(f"\n[PHASE] {phase_name}")
            print(f"        Active rates: {active_rates}")
            print(f"        Requests: {len(trace)}")

            orchestrator = Orchestrator(devices, task_specs, args.scheduler, str(phase_dir))
            plan = orchestrator.run_deployment_plan(devices, task_specs, scheduler_name=args.scheduler, output_dir=str(phase_dir))
            routed_trace = route_trace(trace, plan, args.seed)
            orchestrator.initial_deployment(plan, routed_trace, output_dir=str(phase_dir))
            orchestrator.run_inference_requests()
            orchestrator.wait_for_inference_completion(timeout=max(300, int(args.duration) + 120))
            orchestrator.cleanup_only(plan)
            time.sleep(2)

        print("\n[INFO] All collective phases complete.")
        return 0

    finally:
        if site_proc and site_proc.poll() is None:
            site_proc.terminate()
            try:
                site_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                site_proc.kill()
                site_proc.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
