# ILP_new/experiments/single_tenant/run_experiments.py
"""
Single-tenant ILP experiments using config file for inputs.

Usage:
    python run_experiments.py                              # Use default config
    python run_experiments.py --config ../my_config.json   # Use custom config

Outputs results to ./results/
"""

import os
import sys
import json
import argparse

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from methods.single_tenant import build_and_solve
from experiments.config_loader import (
    load_experiment_config,
    build_devices_from_config,
    build_tasks_from_config,
    get_solver_settings,
)

os.environ.setdefault("GRB_LICENSE_FILE",
    os.path.join(os.path.dirname(__file__), "../../gurobi/gurobi.lic"))


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


def build_ilp_inputs_vlm(config, components, pipelines, latency_data, metric_data):
    """
    Build ILP inputs for VLM (monolithic models).

    VLM models serve multiple tasks with the same weights.
    Models = backbones, one deployment serves all supported tasks.
    """
    # Get devices from config
    devices, vram_device = build_devices_from_config(config)

    # Get unique device types
    device_types = set(info["type"] for info in devices.values())

    # Get task requirements from config
    task_slos, demands, Amin = build_tasks_from_config(config)
    config_tasks = set(task_slos.keys())

    # Models = unique backbones
    models = sorted({info["backbone"] for info in pipelines.values()})

    # Build reverse index: (backbone, task) -> pipeline_id
    # For each device type, keep best latency pipeline
    backbone_task_to_pipeline = {}
    for pid, info in pipelines.items():
        key = (info["backbone"], info["task"])
        # Keep the one with best latency if multiple exist
        if key not in backbone_task_to_pipeline:
            backbone_task_to_pipeline[key] = pid
        else:
            existing_pid = backbone_task_to_pipeline[key]
            # Check latency across all device types
            for device_type in device_types:
                if pid in latency_data and device_type in latency_data[pid]:
                    new_lat = latency_data[pid].get(device_type, float('inf'))
                    old_lat = latency_data.get(existing_pid, {}).get(device_type, float('inf'))
                    if new_lat < old_lat:
                        backbone_task_to_pipeline[key] = pid

    # Support: backbone can serve task if there's a pipeline for it
    support = {}
    for m in models:
        for t in config_tasks:
            support[(m, t)] = 1 if (m, t) in backbone_task_to_pipeline else 0

    # Accuracy: (task, model) -> metric from best pipeline
    accuracy = {}
    for (backbone, task), pid in backbone_task_to_pipeline.items():
        if task in config_tasks:
            accuracy[(task, backbone)] = float(metric_data.get(pid, 0.0))

    # Latency: (task, model, device) -> ms
    latency_ilp = {}
    for (backbone, task), pid in backbone_task_to_pipeline.items():
        if task not in config_tasks:
            continue
        for d, d_info in devices.items():
            device_type = d_info["type"]
            if pid in latency_data and device_type in latency_data[pid]:
                latency_ilp[(task, backbone, d)] = float(latency_data[pid][device_type])

    # Throughput: Ptmd = 1000 / latency_ms
    Ptmd = {}
    for key, L_ms in latency_ilp.items():
        if L_ms > 0:
            Ptmd[key] = 1000.0 / L_ms

    # vram_model: backbone memory + decoder/task memory for all supported tasks
    vram_model = {}
    for m in models:
        # Start with backbone memory
        total_mem = float(components.get(m, {}).get("mem", 0.0))

        # Add decoder and task component memory for each task this backbone serves
        for (backbone, task), pid in backbone_task_to_pipeline.items():
            if backbone != m or task not in config_tasks:
                continue

            info = pipelines[pid]
            decoder = info.get("decoder", "none")

            # Decoder memory (TSFM format: decoder_backbone_task)
            if decoder != "none":
                dec_key = f"{decoder}_{backbone}_{task}"
                total_mem += float(components.get(dec_key, {}).get("mem", 0.0))

            # Task component memory (format: task_backbone_decoder)
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


def build_ilp_inputs_tsfm(config, components, pipelines, latency_data, metric_data):
    """
    Build ILP inputs for TSFM (pipeline-based with task-specific decoders).

    Each pipeline has unique decoder/task head weights.
    Models = pipelines, one deployment serves exactly one task.
    """
    # Get devices from config
    devices, vram_device = build_devices_from_config(config)

    # Get task requirements from config
    task_slos, demands, Amin = build_tasks_from_config(config)
    config_tasks = set(task_slos.keys())

    # Models = pipeline IDs
    models = sorted(pipelines.keys())

    # Support: each pipeline serves exactly its task
    support = {}
    for pid in models:
        pipeline_task = pipelines[pid]["task"]
        for t in config_tasks:
            support[(pid, t)] = 1 if t == pipeline_task else 0

    # Accuracy: (task, pipeline) -> metric
    accuracy = {}
    for pid, info in pipelines.items():
        t = info["task"]
        if t in config_tasks:
            accuracy[(t, pid)] = float(metric_data.get(pid, 0.0))

    # Latency: (task, pipeline, device) -> ms
    latency_ilp = {}
    for pid, info in pipelines.items():
        t = info["task"]
        if t not in config_tasks:
            continue
        for d, d_info in devices.items():
            device_type = d_info["type"]
            if pid in latency_data and device_type in latency_data[pid]:
                latency_ilp[(t, pid, d)] = float(latency_data[pid][device_type])

    # Throughput: Ptmd = 1000 / latency_ms
    Ptmd = {}
    for key, L_ms in latency_ilp.items():
        if L_ms > 0:
            Ptmd[key] = 1000.0 / L_ms

    # vram_model: full pipeline memory (backbone + decoder + task)
    vram_model = {}
    for pid, info in pipelines.items():
        backbone = info["backbone"]
        decoder = info["decoder"]
        task = info["task"]

        bb_mem = components.get(backbone, {}).get("mem", 0.0)

        # Decoder memory (TSFM format: decoder_backbone_task)
        dec_key = f"{decoder}_{backbone}_{task}"
        dec_mem = components.get(dec_key, {}).get("mem", 0.0)

        # Task component memory
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


def build_ilp_inputs_mixed(config, components, pipelines, latency_data, metric_data):
    """
    Build ILP inputs for mixed VLM + TSFM workloads.

    VLM pipelines: models = backbones (sharing allowed)
    TSFM pipelines: models = pipeline IDs (no sharing)
    """
    # Get devices from config
    devices, vram_device = build_devices_from_config(config)
    device_types = set(info["type"] for info in devices.values())

    # Get task requirements from config
    task_slos, demands, Amin = build_tasks_from_config(config)
    config_tasks = set(task_slos.keys())

    # Separate VLM and TSFM pipelines
    vlm_pipelines = {pid: info for pid, info in pipelines.items()
                     if info.get("decoder", "none") == "none"}
    tsfm_pipelines = {pid: info for pid, info in pipelines.items()
                      if info.get("decoder", "none") != "none"}

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

    # Support matrix
    support = {}
    # VLM: backbone can serve task if there's a pipeline
    for m in vlm_backbones:
        for t in config_tasks:
            support[(m, t)] = 1 if (m, t) in vlm_backbone_task_to_pipeline else 0
    # TSFM: pipeline serves exactly its task
    for pid in tsfm_models:
        pipeline_task = tsfm_pipelines[pid]["task"]
        for t in config_tasks:
            support[(pid, t)] = 1 if t == pipeline_task else 0

    # Accuracy
    accuracy = {}
    for (backbone, task), pid in vlm_backbone_task_to_pipeline.items():
        if task in config_tasks:
            accuracy[(task, backbone)] = float(metric_data.get(pid, 0.0))
    for pid, info in tsfm_pipelines.items():
        t = info["task"]
        if t in config_tasks:
            accuracy[(t, pid)] = float(metric_data.get(pid, 0.0))

    # Latency
    latency_ilp = {}
    # VLM
    for (backbone, task), pid in vlm_backbone_task_to_pipeline.items():
        if task not in config_tasks:
            continue
        for d, d_info in devices.items():
            device_type = d_info["type"]
            if pid in latency_data and device_type in latency_data[pid]:
                latency_ilp[(task, backbone, d)] = float(latency_data[pid][device_type])
    # TSFM
    for pid, info in tsfm_pipelines.items():
        t = info["task"]
        if t not in config_tasks:
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
            if backbone != m or task not in config_tasks:
                continue
            task_key = f"{task}_{backbone}_none"
            total_mem += float(components.get(task_key, {}).get("mem", 0.0))
        vram_model[m] = total_mem
    # TSFM pipelines
    for pid, info in tsfm_pipelines.items():
        if info["task"] not in config_tasks:
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


def build_ilp_inputs(config, components, pipelines, latency_data, metric_data, data_source="vlm"):
    """
    Build ILP inputs based on data source type.

    VLM: models = backbones, one deployment serves multiple tasks
    TSFM: models = pipelines, one deployment serves one task
    Mixed: VLM backbones + TSFM pipelines (no TSFM sharing)
    """
    if data_source == "vlm":
        return build_ilp_inputs_vlm(config, components, pipelines, latency_data, metric_data)
    elif data_source == "mixed":
        return build_ilp_inputs_mixed(config, components, pipelines, latency_data, metric_data)
    else:
        return build_ilp_inputs_tsfm(config, components, pipelines, latency_data, metric_data)


def main():
    parser = argparse.ArgumentParser(description="Single-tenant ILP experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config JSON (default: ../experiment_config_vlm.json or _tsfm based on --data)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="vlm",
        choices=["vlm", "tsfm", "mixed"],
        help="Pipeline data source",
    )
    args = parser.parse_args()

    # Resolve config path
    if args.config is None:
        # Auto-select based on data type
        config_path = os.path.join(os.path.dirname(__file__), f"../experiment_config_{args.data}.json")
    else:
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(__file__), config_path)

    # Load config
    print(f"Loading config from: {config_path}")
    config = load_experiment_config(config_path)
    print(f"Experiment: {config.get('name', 'unnamed')}")

    # Load pipeline data
    print(f"Loading {args.data.upper()} pipeline data...")
    components, pipelines, latency_data, metric_data = load_pipeline_data(args.data)
    print(f"  Pipelines: {len(pipelines)}")

    # Build ILP inputs
    ilp_inputs = build_ilp_inputs(config, components, pipelines, latency_data, metric_data, args.data)

    print(f"\nILP Setup:")
    print(f"  Devices: {len(ilp_inputs['devices'])}")
    print(f"  Tasks: {len(ilp_inputs['tasks'])}")
    print(f"  Models (backbones): {len(ilp_inputs['models'])}")

    # Get solver settings
    solver_settings = get_solver_settings(config)

    # Run ILP
    print(f"\n--- Running ILP (time_limit={solver_settings['time_limit']}s) ---")

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
        minimize="deployments",
        time_limit=solver_settings["time_limit"],
        log_to_console=solver_settings["log_to_console"],
    )

    # Print results
    print(f"\n--- Results ---")
    print(f"Status: {result['status']}")
    print(f"Objective (deployments): {result.get('obj')}")

    if result['status'] in ('OPTIMAL', 'TIME_LIMIT'):
        x = result.get('x', {})
        deployed = [(m, d) for (m, d), v in x.items() if v > 0.5]
        print(f"Deployed pipelines: {len(deployed)}")

        # Group by device
        by_device = {}
        for m, d in deployed:
            by_device.setdefault(d, []).append(m)

        print(f"Devices used: {len(by_device)}")

        # Show deployments
        print("\nDeployments:")
        for d in sorted(by_device.keys()):
            for model_id in by_device[d]:
                # Handle both VLM (backbone names) and TSFM (pipeline IDs)
                if model_id in pipelines:
                    # TSFM mode: model_id is pipeline ID
                    task = pipelines[model_id]['task']
                    backbone = pipelines[model_id]['backbone'].split('/')[-1][:25]
                    print(f"  {d}: {model_id} ({backbone}) -> {task}")
                else:
                    # VLM mode: model_id is backbone name
                    # Find all tasks this backbone can serve
                    tasks_served = [info['task'] for info in pipelines.values()
                                   if info['backbone'] == model_id]
                    backbone_short = model_id.split('/')[-1][:25]
                    print(f"  {d}: {backbone_short} -> {len(tasks_served)} tasks")

        # Save results
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)

        results_file = os.path.join(results_dir, f"{config.get('name', 'result')}.json")
        with open(results_file, 'w') as f:
            json.dump({
                "config": config,
                "status": result["status"],
                "objective": result.get("obj"),
                "deployments": len(deployed),
                "devices_used": len(by_device),
                "deployed_pipelines": [{"pipeline": m, "device": d} for m, d in deployed],
            }, f, indent=2)
        print(f"\nSaved results to: {results_file}")

        # Save deployment plan in serving format
        r = result.get('r', {})
        deployment_entries = []

        for model_id, device in deployed:
            # Calculate request rates for each task routed to this deployment
            task_routing = {}
            for (task, model, dev), fraction in r.items():
                if model == model_id and dev == device and fraction > 0.001:
                    demand = float(ilp_inputs['demands'].get(task, 1.0))
                    req_per_sec = demand * fraction
                    task_routing[task] = {
                        "type": "classification",  # Default type
                        "total_requested_workload": demand,
                        "request_per_sec": req_per_sec
                    }

            # Get device type from devices dict
            device_type = ilp_inputs["devices"][device]["type"]

            if model_id in pipelines:
                # TSFM mode: model_id is pipeline ID
                info = pipelines[model_id]
                entry = {
                    "device": device,
                    "device_type": device_type,
                    "backbone": info['backbone'],
                    "decoders": [{
                        "task": info['task'],
                        "type": "classification",
                        "path": f"{info['task']}_{info['backbone']}_{info['decoder']}"
                    }],
                    "tasks": task_routing
                }
            else:
                # VLM mode: model_id is backbone name
                entry = {
                    "device": device,
                    "device_type": device_type,
                    "backbone": model_id,
                    "decoders": [],
                    "tasks": task_routing
                }
            deployment_entries.append(entry)

        deployment_plan = {
            "sites": [{
                "id": "site1",
                "deployments": deployment_entries
            }]
        }

        # Save deployment plan to results directory
        plan_file = os.path.join(results_dir, f"deployment_plan_{args.data}.json")
        with open(plan_file, 'w') as f:
            json.dump(deployment_plan, f, indent=2)
        print(f"Saved deployment plan to: {plan_file}")

    elif result['status'] == 'INFEASIBLE':
        print("\nModel is INFEASIBLE!")
        if result.get('tasks_with_no_feasible_triple'):
            print(f"Tasks with no feasible options: {result['tasks_with_no_feasible_triple']}")


if __name__ == "__main__":
    main()
