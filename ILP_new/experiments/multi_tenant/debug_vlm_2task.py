#!/usr/bin/env python3
"""Debug script to investigate VLM 2-task model selection."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from methods.multi_tenant import build_and_solve
from profiled_metadata.vlm_config import components, pipelines, latency as latency_data, metric as metric_data

# Tasks
TASKS = ["activity_recognition", "crowd_counting"]

# Device pool (same as run_scaling_experiments.py)
CANONICAL_DEVICES = {
    "10.100.20.16": {"type": "A6000", "vram_mb": 48000},
    "10.100.20.17": {"type": "A6000", "vram_mb": 48000},
    "10.100.20.18": {"type": "A6000", "vram_mb": 48000},
    "10.100.20.19": {"type": "A6000", "vram_mb": 48000},
    "10.100.20.20": {"type": "A16", "vram_mb": 16000},
    "10.100.20.21": {"type": "A16", "vram_mb": 16000},
    "10.100.20.22": {"type": "A16", "vram_mb": 16000},
    "10.100.20.23": {"type": "A16", "vram_mb": 16000},
}

# Reuse the build function from run_scaling_experiments.py
def build_ilp_inputs_vlm(task_list):
    """Simplified build function - same logic as experiment runner."""
    # Devices - use canonical fixed pool
    devices = {k: {"type": v["type"]} for k, v in CANONICAL_DEVICES.items()}
    vram_device = {"A6000": 48000.0, "A16": 16000.0}

    device_types = set(info["type"] for info in devices.values())

    # Task requirements
    task_slos = {t: 1500.0 for t in task_list}  # 1500ms latency SLO
    demands = {t: 1.0 for t in task_list}
    Amin = {t: 0.0 for t in task_list}

    # Models = unique backbones
    models = sorted({info["backbone"] for info in pipelines.values()})

    # Build reverse index: (backbone, task) -> pipeline_id
    backbone_task_to_pipeline = {}
    for pid, info in pipelines.items():
        key = (info["backbone"], info["task"])
        if key not in backbone_task_to_pipeline:
            backbone_task_to_pipeline[key] = pid
        else:
            existing_pid = backbone_task_to_pipeline[key]
            for device_type in device_types:
                if pid in latency_data and device_type in latency_data[pid]:
                    new_lat = latency_data[pid].get(device_type, float('inf'))
                    old_lat = latency_data.get(existing_pid, {}).get(device_type, float('inf'))
                    if new_lat < old_lat:
                        backbone_task_to_pipeline[key] = pid

    # Support
    support = {}
    for m in models:
        for t in task_list:
            support[(m, t)] = 1 if (m, t) in backbone_task_to_pipeline else 0

    # Accuracy
    accuracy = {}
    for (backbone, task), pid in backbone_task_to_pipeline.items():
        if task in task_list:
            accuracy[(task, backbone)] = float(metric_data.get(pid, 0.0))

    # Latency
    latency_ilp = {}
    for (backbone, task), pid in backbone_task_to_pipeline.items():
        if task not in task_list:
            continue
        for d, d_info in devices.items():
            device_type = d_info["type"]
            if pid in latency_data and device_type in latency_data[pid]:
                latency_ilp[(task, backbone, d)] = float(latency_data[pid][device_type])

    # Throughput
    Ptmd = {}
    for key, L_ms in latency_ilp.items():
        if L_ms > 0:
            Ptmd[key] = 1000.0 / L_ms

    # Memory
    vram_model = {}
    for m in models:
        total_mem = float(components.get(m, {}).get("mem", 0.0))
        for (backbone, task), pid in backbone_task_to_pipeline.items():
            if backbone != m or task not in task_list:
                continue
            info = pipelines[pid]
            decoder = info.get("decoder", "none")
            if decoder != "none":
                dec_key = f"{decoder}_{backbone}_{task}"
                dec_mem = float(components.get(dec_key, {}).get("mem", 0.0))
                total_mem += dec_mem
        vram_model[m] = total_mem

    return {
        "devices": devices,
        "models": models,
        "tasks": task_slos,
        "demands": demands,
        "support": support,
        "accuracy": accuracy,
        "latency": latency_ilp,
        "vram_model": vram_model,
        "vram_device": vram_device,
        "Ptmd": Ptmd,
        "Amin": Amin,
    }


def run_experiment(objective_mode):
    print(f"\n{'='*70}")
    print(f"Running: {objective_mode}")
    print('='*70)

    inputs = build_ilp_inputs_vlm(TASKS)

    result = build_and_solve(
        devices=inputs["devices"],
        models=inputs["models"],
        tasks=inputs["tasks"],
        demands=inputs["demands"],
        support=inputs["support"],
        accuracy=inputs["accuracy"],
        latency=inputs["latency"],
        vram_model=inputs["vram_model"],
        vram_device=inputs["vram_device"],
        Ptmd=inputs["Ptmd"],
        Amin=inputs["Amin"],
        minimize=objective_mode,
        time_limit=60,
        log_to_console=False,
    )

    print(f"Status: {result['status']}")
    print(f"\nObjective components:")
    for k, v in result.get("objective_components", {}).items():
        print(f"  {k}: {v}")

    # Print deployments
    print(f"\nDeployments (x):")
    deployed = [(m, d) for (m, d), val in result["x"].items() if val > 0]
    for m, d in deployed:
        vram = inputs["vram_model"].get(m, 0)
        print(f"  Model: {m:30s} on Device: {d:15s} (VRAM: {vram:.1f} MB)")

    # Print routing
    print(f"\nRouting (r):")
    for (t, m, d), frac in result["r"].items():
        if frac > 0.001:
            print(f"  Task: {t:25s} -> Model: {m:30s} on {d:15s} (frac: {frac:.3f})")

    # Calculate throughput
    from collections import defaultdict
    deployment_throughput = defaultdict(float)
    for (m, d) in deployed:
        routed_tasks = [t for t in inputs["tasks"]
                       if any((tt, mm, dd) == (t, m, d) and frac > 0.001
                             for (tt, mm, dd), frac in result["r"].items())]
        if routed_tasks:
            capacities = [inputs["Ptmd"].get((t, m, d), 0.0) for t in routed_tasks]
            capacities = [c for c in capacities if c > 0]
            deployment_throughput[(m, d)] = min(capacities) if capacities else 0.0

    total_throughput = sum(deployment_throughput.values())
    print(f"\nTotal throughput: {total_throughput:.2f} req/s")


if __name__ == "__main__":
    run_experiment("deployments_devices")
    run_experiment("deployments_devices_modelsize")
