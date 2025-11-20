# ILP_new/experiments/config_loader.py
"""
Load experiment configuration from JSON file.
"""

import json
import os


def load_experiment_config(config_path):
    """
    Load experiment config from JSON file.

    Returns:
        dict with keys:
            - name: experiment name
            - devices: dict with device specs
            - tasks: dict with task requirements
            - solver: solver settings
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def build_devices_from_config(config):
    """
    Build devices dict for ILP from config.

    Returns:
        devices: {device_id: {"type": device_type}}
        vram_device: {device_type: vram_mb}
    """
    # Check for new sites-based structure
    if "sites" in config:
        devices = {}
        vram_device = {}

        for site_id, site_info in config["sites"].items():
            for device_ip, device_info in site_info["devices"].items():
                dev_type = device_info.get("type", "A6000")
                vram_mb = device_info.get("vram_mb", 48000)

                devices[device_ip] = {"type": dev_type, "site": site_id}
                vram_device[dev_type] = float(vram_mb)

        return devices, vram_device

    # Fallback to old flat structure
    dev_config = config["devices"]

    count = dev_config.get("count", 1)
    dev_type = dev_config.get("type", "A6000")
    vram_mb = dev_config.get("vram_mb", 48000)

    devices = {f"gpu{i}": {"type": dev_type} for i in range(count)}
    vram_device = {dev_type: float(vram_mb)}

    return devices, vram_device


def build_tasks_from_config(config):
    """
    Build task dicts for ILP from config.

    Returns:
        tasks: {task: latency_slo_ms}
        demands: {task: demand_req_s}
        Amin: {task: accuracy_min} or None
    """
    task_config = config["tasks"]

    tasks = {}
    demands = {}
    Amin = {}

    for task_name, reqs in task_config.items():
        tasks[task_name] = float(reqs.get("latency_slo_ms", 1e9))
        demands[task_name] = float(reqs.get("demand_req_s", 1.0))
        acc_min = reqs.get("accuracy_min", 0.0)
        if acc_min > 0:
            Amin[task_name] = float(acc_min)

    # Return None if no accuracy requirements
    if not Amin:
        Amin = None

    return tasks, demands, Amin


def get_solver_settings(config):
    """Get solver settings from config."""
    solver = config.get("solver", {})
    return {
        "time_limit": solver.get("time_limit_s", 60),
        "log_to_console": solver.get("log_to_console", False),
    }
