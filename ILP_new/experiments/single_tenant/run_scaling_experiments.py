"""
Scaling experiments for single-tenant ILP: varying number of tasks.

This script runs ILP experiments with varying task counts (2, 4, 6, 8, 10, 12, 14, 16)
for VLM, TSFM, and Mixed workloads, testing all 4 objective modes.

Usage:
    python run_scaling_experiments.py                    # Run all experiments
    python run_scaling_experiments.py --workload vlm     # Run only VLM
    python run_scaling_experiments.py --tasks 2 4 6      # Run specific task counts
    python run_scaling_experiments.py --dry-run          # Show what would run

Results saved to: ./results/scaling/
"""

import os
import sys
import json
import argparse
from itertools import cycle

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from methods.single_tenant import build_and_solve
from experiments.config_loader import (
    build_devices_from_config,
    get_solver_settings,
)

# Fixed device pool for ALL scaling experiments (from mixed config)
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

# Available tasks for each workload type
VLM_TASKS = [
    "activity_recognition",
    "crowd_counting",
    "gesture_recognition",
    "image_classification",
    "object_detection",
    "ocr",
    "scene_classification",
    "traffic_classification",
    "vqa",
]

TSFM_TASKS = [
    "diasbp",
    "ecgclass",
    "elecfore",
    "etth1fore",
    "gestureclass",
    "heartrate",
    "ratefore",
    "sysbp",
    "trafficfore",
    "weatherfore",
]

# Default task requirements
DEFAULT_VLM_TASK_CONFIG = {
    "latency_slo_ms": 1500,
    "accuracy_min": 0.0,
    "demand_req_s": 1.0,
}

DEFAULT_TSFM_TASK_CONFIG = {
    "latency_slo_ms": 100,
    "accuracy_min": 0.0,
    "demand_req_s": 1.0,
}


def select_tasks_roundrobin(task_pool, count):
    """Select 'count' tasks from task_pool using round-robin."""
    if count == 0:
        return []

    selected = []
    task_cycle = cycle(task_pool)
    for _ in range(count):
        selected.append(next(task_cycle))

    return selected


def load_pipeline_data(data_source="vlm"):
    """Load pipeline data (components, pipelines, latency, metric)."""
    if data_source == "vlm":
        from profiled_metadata.vlm_config import components, pipelines, latency, metric
    elif data_source == "tsfm":
        from profiled_metadata.profiler import components, pipelines, latency, metric
    elif data_source == "mixed":
        # Load both and merge
        from profiled_metadata.vlm_config import (
            components as vlm_components,
            pipelines as vlm_pipelines,
            latency as vlm_latency,
            metric as vlm_metric
        )
        from profiled_metadata.profiler import (
            components as tsfm_components,
            pipelines as tsfm_pipelines,
            latency as tsfm_latency,
            metric as tsfm_metric
        )

        # Merge components
        components = {**vlm_components, **tsfm_components}

        # Merge pipelines with prefixes to avoid collision
        pipelines = {}
        for pid, info in vlm_pipelines.items():
            pipelines[f"vlm_{pid}"] = info
        for pid, info in tsfm_pipelines.items():
            pipelines[f"tsfm_{pid}"] = info

        # Merge latency
        latency = {}
        for pid, lat_info in vlm_latency.items():
            latency[f"vlm_{pid}"] = lat_info
        for pid, lat_info in tsfm_latency.items():
            latency[f"tsfm_{pid}"] = lat_info

        # Merge metric
        metric = {}
        for pid, met_val in vlm_metric.items():
            metric[f"vlm_{pid}"] = met_val
        for pid, met_val in tsfm_metric.items():
            metric[f"tsfm_{pid}"] = met_val
    else:
        from profiled_metadata.profiler import components, pipelines, latency, metric

    return components, pipelines, latency, metric


def build_ilp_inputs_vlm(task_list, components, pipelines, latency_data, metric_data):
    """Build ILP inputs for VLM workload with specified tasks."""
    # Devices - use canonical fixed pool
    devices = {k: {"type": v["type"]} for k, v in CANONICAL_DEVICES.items()}
    vram_device = {"A6000": 48000.0, "A16": 16000.0}

    device_types = set(info["type"] for info in devices.values())

    # Task requirements
    task_slos = {t: DEFAULT_VLM_TASK_CONFIG["latency_slo_ms"] for t in task_list}
    demands = {t: DEFAULT_VLM_TASK_CONFIG["demand_req_s"] for t in task_list}
    Amin = {t: DEFAULT_VLM_TASK_CONFIG["accuracy_min"] for t in task_list}

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
                total_mem += float(components.get(dec_key, {}).get("mem", 0.0))
            task_key = f"{task}_{backbone}_{decoder}"
            total_mem += float(components.get(task_key, {}).get("mem", 0.0))
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


def build_ilp_inputs_tsfm(task_list, components, pipelines, latency_data, metric_data):
    """Build ILP inputs for TSFM workload with specified tasks."""
    # Devices - use canonical fixed pool
    devices = {k: {"type": v["type"]} for k, v in CANONICAL_DEVICES.items()}
    vram_device = {"A6000": 48000.0, "A16": 16000.0}

    # Task requirements
    task_slos = {t: DEFAULT_TSFM_TASK_CONFIG["latency_slo_ms"] for t in task_list}
    demands = {t: DEFAULT_TSFM_TASK_CONFIG["demand_req_s"] for t in task_list}
    Amin = {t: DEFAULT_TSFM_TASK_CONFIG["accuracy_min"] for t in task_list}

    # Models = pipeline IDs
    models = sorted(pipelines.keys())

    # Support
    support = {}
    for pid in models:
        pipeline_task = pipelines[pid]["task"]
        for t in task_list:
            support[(pid, t)] = 1 if t == pipeline_task else 0

    # Accuracy
    accuracy = {}
    for pid, info in pipelines.items():
        t = info["task"]
        if t in task_list:
            accuracy[(t, pid)] = float(metric_data.get(pid, 0.0))

    # Latency
    latency_ilp = {}
    for pid, info in pipelines.items():
        t = info["task"]
        if t not in task_list:
            continue
        for d, d_info in devices.items():
            device_type = d_info["type"]
            if pid in latency_data and device_type in latency_data[pid]:
                latency_ilp[(t, pid, d)] = float(latency_data[pid][device_type])

    # Throughput
    Ptmd = {}
    for key, L_ms in latency_ilp.items():
        if L_ms > 0:
            Ptmd[key] = 1000.0 / L_ms

    # Memory
    vram_model = {}
    for pid, info in pipelines.items():
        backbone = info["backbone"]
        decoder = info["decoder"]
        task = info["task"]

        bb_mem = components.get(backbone, {}).get("mem", 0.0)
        dec_key = f"{decoder}_{backbone}_{task}"
        dec_mem = components.get(dec_key, {}).get("mem", 0.0)
        task_key = f"{task}_{backbone}_{decoder}"
        task_mem = components.get(task_key, {}).get("mem", 0.0)

        vram_model[pid] = float(bb_mem + dec_mem + task_mem)

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


def build_ilp_inputs_mixed(vlm_tasks, tsfm_tasks, components, pipelines, latency_data, metric_data):
    """Build ILP inputs for mixed workload with specified VLM and TSFM tasks."""
    # Devices - use canonical fixed pool
    devices = {k: {"type": v["type"]} for k, v in CANONICAL_DEVICES.items()}
    vram_device = {"A6000": 48000.0, "A16": 16000.0}
    device_types = set(info["type"] for info in devices.values())

    # Separate VLM and TSFM pipelines
    vlm_pipelines = {pid: info for pid, info in pipelines.items()
                     if info.get("decoder", "none") == "none"}
    tsfm_pipelines = {pid: info for pid, info in pipelines.items()
                      if info.get("decoder", "none") != "none"}

    # Task requirements
    task_list = vlm_tasks + tsfm_tasks
    task_slos = {}
    demands = {}
    Amin = {}
    for t in vlm_tasks:
        task_slos[t] = DEFAULT_VLM_TASK_CONFIG["latency_slo_ms"]
        demands[t] = DEFAULT_VLM_TASK_CONFIG["demand_req_s"]
        Amin[t] = DEFAULT_VLM_TASK_CONFIG["accuracy_min"]
    for t in tsfm_tasks:
        task_slos[t] = DEFAULT_TSFM_TASK_CONFIG["latency_slo_ms"]
        demands[t] = DEFAULT_TSFM_TASK_CONFIG["demand_req_s"]
        Amin[t] = DEFAULT_TSFM_TASK_CONFIG["accuracy_min"]

    # Models: VLM backbones + TSFM pipeline IDs
    vlm_backbones = sorted({info["backbone"] for info in vlm_pipelines.values()})
    tsfm_models = sorted(tsfm_pipelines.keys())
    models = vlm_backbones + tsfm_models

    # Build VLM backbone -> task -> pipeline mapping
    vlm_backbone_task_to_pipeline = {}
    for pid, info in vlm_pipelines.items():
        key = (info["backbone"], info["task"])
        if key not in vlm_backbone_task_to_pipeline:
            vlm_backbone_task_to_pipeline[key] = pid

    # Support
    support = {}
    for m in vlm_backbones:
        for t in task_list:
            support[(m, t)] = 1 if (m, t) in vlm_backbone_task_to_pipeline and t in vlm_tasks else 0
    for pid in tsfm_models:
        pipeline_task = tsfm_pipelines[pid]["task"]
        for t in task_list:
            support[(pid, t)] = 1 if t == pipeline_task else 0

    # Accuracy
    accuracy = {}
    for (backbone, task), pid in vlm_backbone_task_to_pipeline.items():
        if task in vlm_tasks:
            accuracy[(task, backbone)] = float(metric_data.get(pid, 0.0))
    for pid, info in tsfm_pipelines.items():
        t = info["task"]
        if t in tsfm_tasks:
            accuracy[(t, pid)] = float(metric_data.get(pid, 0.0))

    # Latency
    latency_ilp = {}
    # VLM
    for (backbone, task), pid in vlm_backbone_task_to_pipeline.items():
        if task not in vlm_tasks:
            continue
        for d, d_info in devices.items():
            device_type = d_info["type"]
            if pid in latency_data and device_type in latency_data[pid]:
                latency_ilp[(task, backbone, d)] = float(latency_data[pid][device_type])
    # TSFM
    for pid, info in tsfm_pipelines.items():
        t = info["task"]
        if t not in tsfm_tasks:
            continue
        for d, d_info in devices.items():
            device_type = d_info["type"]
            if pid in latency_data and device_type in latency_data[pid]:
                latency_ilp[(t, pid, d)] = float(latency_data[pid][device_type])

    # Throughput
    Ptmd = {}
    for key, L_ms in latency_ilp.items():
        if L_ms > 0:
            Ptmd[key] = 1000.0 / L_ms

    # Memory
    vram_model = {}
    # VLM backbones
    for m in vlm_backbones:
        total_mem = float(components.get(m, {}).get("mem", 0.0))
        for (backbone, task), pid in vlm_backbone_task_to_pipeline.items():
            if backbone != m or task not in vlm_tasks:
                continue
            task_key = f"{task}_{backbone}_none"
            total_mem += float(components.get(task_key, {}).get("mem", 0.0))
        vram_model[m] = total_mem
    # TSFM pipelines
    for pid, info in tsfm_pipelines.items():
        if info["task"] not in tsfm_tasks:
            continue
        backbone = info["backbone"]
        decoder = info["decoder"]
        task = info["task"]
        bb_mem = float(components.get(backbone, {}).get("mem", 0.0))
        dec_key = f"{decoder}_{backbone}_{task}"
        dec_mem = float(components.get(dec_key, {}).get("mem", 0.0))
        task_key = f"{task}_{backbone}_{decoder}"
        task_mem = float(components.get(task_key, {}).get("mem", 0.0))
        vram_model[pid] = bb_mem + dec_mem + task_mem

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


def collect_metrics(result, ilp_inputs):
    """Extract key metrics from ILP solution."""
    if result["status"] not in ("OPTIMAL", "TIME_LIMIT"):
        return None

    # Memory footprint: total model memory (O4)
    memory_mb = result.get("objective_components", {}).get("O4_total_mem_mb", 0.0)

    # Get deployed models
    x = result.get("x", {})
    deployed = [(m, d) for (m, d), v in x.items() if v > 0.5]

    # Total deployed capacity: sum of throughput capacities for all deployed model-device pairs
    Ptmd = ilp_inputs["Ptmd"]
    support = ilp_inputs["support"]
    tasks = list(ilp_inputs["tasks"].keys())
    demands = ilp_inputs["demands"]

    total_capacity = 0.0
    for (m, d) in deployed:
        # Find tasks this model can serve
        model_tasks = [t for t in tasks if support.get((m, t), 0) == 1]
        if model_tasks:
            # For each task this model serves, get its throughput capacity
            # Use the average capacity across tasks (models may have different speeds per task)
            capacities = [Ptmd.get((t, m, d), 0.0) for t in model_tasks]
            capacities = [c for c in capacities if c > 0]
            if capacities:
                # Sum capacities (each deployed model adds capacity)
                total_capacity += sum(capacities)

    # Total demand being served (input demand, all served if feasible)
    total_demand = sum(demands.values())

    # Average latency: weighted by routing fraction and demand
    r = result.get("r", {})
    latency_ilp = ilp_inputs["latency"]

    weighted_latency = 0.0
    weighted_demand = 0.0
    for (t, m, d), frac in r.items():
        if frac > 0.001:
            lat = latency_ilp.get((t, m, d), 0.0)
            dem = demands.get(t, 0.0)
            weighted_latency += lat * frac * dem
            weighted_demand += dem * frac

    avg_latency = weighted_latency / weighted_demand if weighted_demand > 0 else 0.0

    return {
        "memory_mb": float(memory_mb),
        "throughput_req_s": float(total_capacity),  # Total deployed capacity
        "total_demand_req_s": float(total_demand),  # Input demand being served
        "avg_latency_ms": float(avg_latency),
        "num_deployments": result.get("objective_components", {}).get("O1_deployments", 0),
        "num_devices": result.get("objective_components", {}).get("O2_devices", 0),
        "status": result["status"],
    }


def run_single_experiment(workload_type, num_tasks, objective_mode, components, pipelines, latency_data, metric_data):
    """Run a single ILP experiment."""
    # Select tasks
    if workload_type == "vlm":
        task_list = select_tasks_roundrobin(VLM_TASKS, num_tasks)
        ilp_inputs = build_ilp_inputs_vlm(task_list, components, pipelines, latency_data, metric_data)
    elif workload_type == "tsfm":
        task_list = select_tasks_roundrobin(TSFM_TASKS, num_tasks)
        ilp_inputs = build_ilp_inputs_tsfm(task_list, components, pipelines, latency_data, metric_data)
    elif workload_type == "mixed":
        # Split tasks: half VLM, half TSFM (tie-breaker: TSFM gets extra)
        num_vlm = num_tasks // 2
        num_tsfm = num_tasks - num_vlm
        vlm_task_list = select_tasks_roundrobin(VLM_TASKS, num_vlm)
        tsfm_task_list = select_tasks_roundrobin(TSFM_TASKS, num_tsfm)
        ilp_inputs = build_ilp_inputs_mixed(vlm_task_list, tsfm_task_list, components, pipelines, latency_data, metric_data)
        task_list = vlm_task_list + tsfm_task_list
    else:
        raise ValueError(f"Unknown workload type: {workload_type}")

    print(f"  Tasks ({len(task_list)}): {', '.join(task_list)}")

    # Run ILP
    result = build_and_solve(
        devices=ilp_inputs["devices"],
        models=ilp_inputs["models"],
        tasks=ilp_inputs["tasks"],
        demands=ilp_inputs["demands"],
        support=ilp_inputs["support"],
        accuracy=ilp_inputs["accuracy"],
        latency=ilp_inputs["latency"],
        vram_model=ilp_inputs["vram_model"],
        vram_device=ilp_inputs["vram_device"],
        Ptmd=ilp_inputs["Ptmd"],
        Amin=ilp_inputs["Amin"],
        minimize=objective_mode,
        time_limit=60,
        log_to_console=False,
    )

    # Collect metrics
    metrics = collect_metrics(result, ilp_inputs)

    return {
        "workload_type": workload_type,
        "num_tasks": num_tasks,
        "objective_mode": objective_mode,
        "task_list": task_list,
        "metrics": metrics,
        "ilp_result": {
            "status": result["status"],
            "objective": result.get("obj"),
            "objective_components": result.get("objective_components", {}),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Scaling experiments for single-tenant ILP")
    parser.add_argument(
        "--workload",
        type=str,
        nargs="+",
        default=["vlm", "tsfm", "mixed"],
        choices=["vlm", "tsfm", "mixed"],
        help="Workload types to run",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8, 10, 12, 14, 16],
        help="Task counts to test",
    )
    parser.add_argument(
        "--objective",
        type=str,
        nargs="+",
        default=["deployments", "deployments_devices", "deployments_devices_waste", "deployments_devices_waste_modelsize"],
        choices=["deployments", "deployments_devices", "deployments_devices_waste", "deployments_devices_waste_modelsize"],
        help="Objective modes to run",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")

    args = parser.parse_args()

    # Print experiment plan
    total_runs = len(args.workload) * len(args.tasks) * len(args.objective)
    print(f"Scaling Experiment Plan:")
    print(f"  Workloads: {args.workload}")
    print(f"  Task counts: {args.tasks}")
    print(f"  Objectives: {args.objective}")
    print(f"  Total runs: {total_runs}")
    print(f"  Device pool: {len(CANONICAL_DEVICES)} devices (4× A6000, 4× A16)")
    print()

    if args.dry_run:
        print("Dry-run mode: not executing experiments")
        return

    # Create results directory
    results_base = os.path.join(os.path.dirname(__file__), "results", "scaling")
    os.makedirs(results_base, exist_ok=True)
    for wl in args.workload:
        os.makedirs(os.path.join(results_base, wl), exist_ok=True)
    os.makedirs(os.path.join(results_base, "summary"), exist_ok=True)

    # Track results for summary
    all_results = {wl: {obj: {} for obj in args.objective} for wl in args.workload}

    # Run experiments
    run_num = 0
    for workload_type in args.workload:
        print(f"\n{'='*70}")
        print(f"Workload: {workload_type.upper()}")
        print(f"{'='*70}")

        # Load pipeline data
        print(f"Loading {workload_type.upper()} pipeline data...")
        components, pipelines, latency_data, metric_data = load_pipeline_data(workload_type)
        print(f"  Pipelines: {len(pipelines)}")

        for num_tasks in args.tasks:
            for objective_mode in args.objective:
                run_num += 1
                print(f"\n[{run_num}/{total_runs}] {workload_type} | {num_tasks} tasks | {objective_mode}")

                # Run experiment
                exp_result = run_single_experiment(
                    workload_type, num_tasks, objective_mode,
                    components, pipelines, latency_data, metric_data
                )

                # Print metrics
                if exp_result["metrics"]:
                    m = exp_result["metrics"]
                    print(f"  Status: {m['status']}")
                    print(f"  Deployments: {m['num_deployments']}, Devices: {m['num_devices']}")
                    print(f"  Memory: {m['memory_mb']:.1f} MB")
                    print(f"  Throughput: {m['throughput_req_s']:.2f} req/s")
                    print(f"  Avg Latency: {m['avg_latency_ms']:.2f} ms")
                else:
                    print(f"  Status: {exp_result['ilp_result']['status']} (no solution)")

                # Save individual result
                result_file = os.path.join(
                    results_base, workload_type,
                    f"{workload_type}_{num_tasks}tasks_{objective_mode}.json"
                )
                with open(result_file, 'w') as f:
                    json.dump(exp_result, f, indent=2)

                # Store for summary
                all_results[workload_type][objective_mode][num_tasks] = exp_result["metrics"]

    # Save summaries
    print(f"\n{'='*70}")
    print("Saving summaries...")
    print(f"{'='*70}")

    for workload_type in args.workload:
        summary = {
            "workload_type": workload_type,
            "task_counts": sorted(args.tasks),
            "objective_modes": args.objective,
            "results": all_results[workload_type],
        }

        summary_file = os.path.join(results_base, "summary", f"{workload_type}_scaling_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  {summary_file}")

    print(f"\nDone! Results saved to: {results_base}")


if __name__ == "__main__":
    main()
