# ILP_new/experiments/multi_tenant/run_experiments.py
"""
Multi-tenant ILP experiments using config file for inputs.

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

from methods.multi_tenant import build_and_solve
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
    parser = argparse.ArgumentParser(description="Multi-tenant ILP experiment")
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
    parser.add_argument(
        "--minimize",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "deployments", "deployments_devices",
                 "deployments_devices_waste", "deployments_devices_waste_modelsize"],
        help="Objective mode(s) to run (default: all)",
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

    # Update experiment name for multi-tenant
    original_name = config.get('name', 'unnamed')
    config['name'] = original_name.replace('single_tenant', 'multi_tenant')
    if 'multi_tenant' not in config['name']:
        config['name'] = f"{original_name}_multi_tenant"

    print(f"Experiment: {config['name']}")

    # Expand "all" into all minimize modes
    if "all" in args.minimize:
        minimize_modes = [
            "deployments",
            "deployments_devices",
            "deployments_devices_waste",
            "deployments_devices_waste_modelsize"
        ]
    else:
        minimize_modes = args.minimize

    print(f"Will run {len(minimize_modes)} objective mode(s): {', '.join(minimize_modes)}")

    # Load pipeline data
    print(f"Loading {args.data.upper()} pipeline data...")
    components, pipelines, latency_data, metric_data = load_pipeline_data(args.data)
    print(f"  Pipelines: {len(pipelines)}")

    # Build ILP inputs
    ilp_inputs = build_ilp_inputs(config, components, pipelines, latency_data, metric_data, args.data)

    print(f"\nILP Setup:")
    print(f"  Devices: {len(ilp_inputs['devices'])}")
    print(f"  Tasks: {len(ilp_inputs['tasks'])}")
    print(f"  Models (backbones/pipelines): {len(ilp_inputs['models'])}")

    # Get solver settings
    solver_settings = get_solver_settings(config)

    # Store results for all modes
    all_results = {}

    # Loop through each minimize mode
    for minimize_mode in minimize_modes:
        print(f"\n{'='*70}")
        print(f"Running Multi-Tenant ILP with minimize='{minimize_mode}'")
        print(f"{'='*70}")

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
            minimize=minimize_mode,
            time_limit=solver_settings["time_limit"],
            log_to_console=solver_settings["log_to_console"],
        )

        # Store result
        all_results[minimize_mode] = result

        # Print results
        print(f"\n--- Results ({minimize_mode}) ---")
        print(f"Status: {result['status']}")
        print(f"Objective Value: {result.get('obj')}")

        # Print objective breakdown
        if "objective_components" in result:
            comp = result["objective_components"]
            print(f"  O1 (Deployments):     {comp['O1_deployments']}")
            print(f"  O2 (Devices Used):    {comp['O2_devices']}")
            print(f"  O3 (Wasted Memory):   {comp['O3_waste_mb']:.1f} MB")
            print(f"  O4 (Total Model Mem): {comp['O4_total_mem_mb']:.1f} MB")

        if result['status'] in ('OPTIMAL', 'TIME_LIMIT'):
            x = result.get('x', {})
            deployed = [(m, d) for (m, d), v in x.items() if v > 0.5]
            print(f"Deployed models: {len(deployed)}")

            # Group by device
            by_device = {}
            for m, d in deployed:
                by_device.setdefault(d, []).append(m)

            print(f"Devices used: {len(by_device)}")

            # Show deployments
            print("\nDeployments:")
            for d in sorted(by_device.keys()):
                device_type = ilp_inputs["devices"][d]["type"]
                models_on_device = by_device[d]

                # Calculate total memory on this device
                total_mem = sum(ilp_inputs["vram_model"].get(m, 0) for m in models_on_device)
                device_cap = ilp_inputs["vram_device"].get(device_type, 0)

                print(f"  {d} ({device_type}): {len(models_on_device)} models, {total_mem:.1f}/{device_cap:.1f} MB")

                for model_id in models_on_device:
                    # Handle both VLM (backbone names) and TSFM (pipeline IDs)
                    if model_id in pipelines:
                        # TSFM mode: model_id is pipeline ID
                        task = pipelines[model_id]['task']
                        backbone = pipelines[model_id]['backbone'].split('/')[-1][:20]
                        mem = ilp_inputs["vram_model"].get(model_id, 0)
                        print(f"    - {model_id}: {backbone} -> {task} ({mem:.1f} MB)")
                    else:
                        # VLM mode: model_id is backbone name
                        tasks_served = [info['task'] for info in pipelines.values()
                                       if info['backbone'] == model_id]
                        backbone_short = model_id.split('/')[-1][:20]
                        mem = ilp_inputs["vram_model"].get(model_id, 0)
                        print(f"    - {backbone_short} -> {len(tasks_served)} tasks ({mem:.1f} MB)")

            # Save results
            results_dir = os.path.join(os.path.dirname(__file__), "results")
            os.makedirs(results_dir, exist_ok=True)

            results_file = os.path.join(results_dir, f"{config['name']}_{minimize_mode}.json")
            result_data = {
                "config": config,
                "minimize_mode": minimize_mode,
                "status": result["status"],
                "objective": result.get("obj"),
                "deployments": len(deployed),
                "devices_used": len(by_device),
                "deployed_models": [{"model": m, "device": d} for m, d in deployed],
            }

            # Add objective breakdown if available
            if "objective_components" in result:
                result_data["objective_components"] = result["objective_components"]

            with open(results_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            print(f"\nSaved results to: {results_file}")

            # Save deployment plan in serving format
            r = result.get('r', {})
            deployment_entries = []

            for device in by_device:
                models_on_device = by_device[device]
                device_type = ilp_inputs["devices"][device]["type"]

                for model_id in models_on_device:
                    # Calculate request rates for each task routed to this deployment
                    task_routing = {}
                    for (task, model, dev), fraction in r.items():
                        if model == model_id and dev == device and fraction > 0.001:
                            demand = float(ilp_inputs['demands'].get(task, 1.0))
                            req_per_sec = demand * fraction
                            task_routing[task] = {
                                "type": "classification",
                                "total_requested_workload": demand,
                                "request_per_sec": req_per_sec
                            }

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
            plan_file = os.path.join(results_dir, f"deployment_plan_{args.data}_{minimize_mode}.json")
            with open(plan_file, 'w') as f:
                json.dump(deployment_plan, f, indent=2)
            print(f"Saved deployment plan to: {plan_file}")

        elif result['status'] == 'INFEASIBLE':
            print("\nModel is INFEASIBLE!")
            if result.get('tasks_with_no_feasible_triple'):
                print(f"Tasks with no feasible options: {result['tasks_with_no_feasible_triple']}")

    # Print summary comparison table
    print(f"\n{'='*70}")
    print("SUMMARY: Comparison Across All Objective Modes")
    print(f"{'='*70}")

    if all_results:
        # Print header
        print(f"{'Mode':<40} | {'O1':>4} | {'O2':>4} | {'O3 (MB)':>10} | {'O4 (MB)':>12} | {'Status':<10}")
        print("-" * 95)

        # Print each mode's results
        for mode in minimize_modes:
            result = all_results.get(mode, {})
            status = result.get('status', 'N/A')
            comp = result.get('objective_components', {})
            O1 = comp.get('O1_deployments', 0)
            O2 = comp.get('O2_devices', 0)
            O3 = comp.get('O3_waste_mb', 0.0)
            O4 = comp.get('O4_total_mem_mb', 0.0)

            print(f"{mode:<40} | {O1:>4} | {O2:>4} | {O3:>10.1f} | {O4:>12.1f} | {status:<10}")

        # Save summary
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        summary_file = os.path.join(results_dir, f"{config['name']}_summary.json")
        summary_data = {
            "config_name": config.get('name', 'unnamed'),
            "data_source": args.data,
            "modes_run": minimize_modes,
            "results_by_mode": {}
        }

        for mode, result in all_results.items():
            comp = result.get('objective_components', {})
            summary_data["results_by_mode"][mode] = {
                "status": result.get('status'),
                "objective_total": result.get('obj'),
                "O1_deployments": comp.get('O1_deployments', 0),
                "O2_devices": comp.get('O2_devices', 0),
                "O3_waste_mb": comp.get('O3_waste_mb', 0.0),
                "O4_total_mem_mb": comp.get('O4_total_mem_mb', 0.0)
            }

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"\nSaved summary to: {summary_file}")
    else:
        print("No results to summarize.")


if __name__ == "__main__":
    main()
