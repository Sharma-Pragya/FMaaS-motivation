# VLMParser.py
# Usage: python vlm_parser.py [path_to_unified_metrics.csv]
# Reads VLM profiling data and writes config.py in the SAME format as TSFM:
#   - components: Memory for each model (backbone)
#   - pipelines: p1, p2, ... -> {backbone, decoder, task}
#   - latency: p1, p2, ... -> {device: latency_ms}
#   - metric: p1, p2, ... -> accuracy
#
# VLM models are treated as "backbones" with decoder='none' (no sharing).

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import csv
import os
import sys

# Default CSV path (relative to this script's location)
DEFAULT_CSV = "../../vqa/updated/unified_metrics.csv"

def extract_task_name(dataset_path: str) -> str:
    """Extract task name from dataset path like '/home/.../dataset/crowd_counting' -> 'crowd_counting'"""
    path = dataset_path.rstrip('/')
    task = os.path.basename(path)
    return task.lower()

def normalize_model_name(model_name: str) -> str:
    """Normalize model name for use as backbone identifier"""
    return model_name.strip()

def extract_device_type(gpu_name: str) -> str:
    """Extract device type from GPU name like 'NVIDIA RTX A6000' -> 'A6000'"""
    gpu_name = gpu_name.upper()
    if 'A6000' in gpu_name:
        return 'A6000'
    elif 'A100' in gpu_name:
        return 'A100'
    elif 'A16' in gpu_name:
        return 'A16'
    elif '3090' in gpu_name:
        return 'RTX3090'
    elif '4090' in gpu_name:
        return 'RTX4090'
    else:
        parts = gpu_name.split()
        return parts[-1] if parts else 'UNKNOWN'

def parse_float(value: str) -> Optional[float]:
    """Safely parse a float value"""
    try:
        return float(value) if value else None
    except (ValueError, TypeError):
        return None

def write_dict(name: str, data: dict, f, inline_values: bool = False):
    """Write dictionary in a readable format"""
    f.write(f"{name}={{\n")
    for k, v in data.items():
        if inline_values and isinstance(v, dict):
            f.write(f"    {repr(k)}:{repr(v)},\n")
        else:
            f.write(f"    {repr(k)}: {repr(v)},\n")
    f.write("}\n\n")

def main():
    # Determine CSV path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, DEFAULT_CSV)

    csv_path = os.path.abspath(csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    print(f"[VLMParser] Reading from: {csv_path}")

    # Read CSV
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty")

    # Data structures
    # components: backbone_name -> {'mem': MB}
    components: Dict[str, Dict[str, float]] = {}

    # For VLM, we also need task-specific components (like TSFM has decoder and task components)
    # Format: task_backbone_none -> {'mem': peak_inference_mem}

    # pipelines: p1, p2, ... -> {'backbone': ..., 'decoder': ..., 'task': ...}
    pipelines: Dict[str, Dict[str, str]] = {}

    # latency: p1, p2, ... -> {device: latency_ms}
    latency: Dict[str, Dict[str, float]] = {}

    # metric: p1, p2, ... -> accuracy
    metric: Dict[str, float] = {}

    # Track unique (backbone, task) combinations to create pipelines
    pipeline_map: Dict[Tuple[str, str], str] = {}  # (backbone, task) -> pipeline_id
    pipeline_counter = 1

    # First pass: collect backbone memory and identify unique pipelines
    backbone_mem: Dict[str, float] = {}
    pipeline_data: Dict[Tuple[str, str], Dict] = {}  # (backbone, task) -> aggregated data

    for row in rows:
        model_name = row.get('model_name', '').strip()
        dataset_path = row.get('dataset_name', '').strip()
        gpu_name = row.get('gpu_name', '').strip()

        if not model_name or not dataset_path:
            continue

        backbone = normalize_model_name(model_name)
        task = extract_task_name(dataset_path)
        device = extract_device_type(gpu_name)

        # Skip invalid entries
        latency_val = parse_float(row.get('avg_latency_ms', ''))
        if latency_val is None or latency_val <= 0:
            continue

        # Backbone memory (model load memory)
        load_mem = parse_float(row.get('gpu_load_memory_mb', ''))
        if load_mem and load_mem > 0:
            if backbone not in backbone_mem:
                backbone_mem[backbone] = load_mem
            else:
                backbone_mem[backbone] = max(backbone_mem[backbone], load_mem)

        # Aggregate pipeline data
        key = (backbone, task)
        if key not in pipeline_data:
            pipeline_data[key] = {
                'latency': {},  # device -> min latency
                'accuracy': [],  # list of accuracy values to average
                'peak_mem': 0.0,  # max peak mem
            }

        # Latency per device (keep minimum)
        if device not in pipeline_data[key]['latency']:
            pipeline_data[key]['latency'][device] = latency_val
        else:
            pipeline_data[key]['latency'][device] = min(pipeline_data[key]['latency'][device], latency_val)

        # Accuracy (collect for averaging)
        accuracy = parse_float(row.get('accuracy', ''))
        if accuracy is not None:
            pipeline_data[key]['accuracy'].append(accuracy)

        # Peak inference memory
        peak_mem = parse_float(row.get('avg_gpu_memory_usage_mb', ''))
        if peak_mem and peak_mem > 0:
            pipeline_data[key]['peak_mem'] = max(pipeline_data[key]['peak_mem'], peak_mem)

    # Build components dictionary
    # 1. Backbone components (model memory)
    for backbone, mem in sorted(backbone_mem.items()):
        components[backbone] = {'mem': round(mem, 6)}

    # 2. Decoder components (none for VLM)
    components['none'] = {'mem': 0.0}

    # 3. Task-specific components (peak inference memory)
    # Format follows TSFM: task_backbone_decoder
    for (backbone, task), data in sorted(pipeline_data.items()):
        task_key = f"{task}_{backbone}_none"
        components[task_key] = {'mem': round(data['peak_mem'], 6)}

    # Build pipelines, latency, and metric dictionaries
    for (backbone, task), data in sorted(pipeline_data.items()):
        pid = f"p{pipeline_counter}"
        pipeline_map[(backbone, task)] = pid
        pipeline_counter += 1

        # Pipeline definition
        pipelines[pid] = {
            'backbone': backbone,
            'decoder': 'none',
            'task': task
        }

        # Latency per device
        latency[pid] = {dev: round(lat, 5) for dev, lat in sorted(data['latency'].items())}

        # Metric (average accuracy)
        if data['accuracy']:
            metric[pid] = round(sum(data['accuracy']) / len(data['accuracy']), 2)
        else:
            metric[pid] = 0.0

    # Write output file
    output_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(output_dir, "vlm_config.py")

    with open(config_path, "w", encoding="utf-8") as f:
        f.write("# THIS FILE IS AUTO-GENERATED by vlm_parser.py - DO NOT EDIT\n")
        f.write("# VLM profiling data in TSFM pipeline format.\n")
        f.write("# VLM models are backbones with decoder='none' (no sharing).\n\n")

        write_dict('components', components, f)
        write_dict('pipelines', pipelines, f)
        write_dict('latency', latency, f)
        write_dict('metric', metric, f)

    # Print summary
    print(f"[VLMParser] Wrote {config_path}")
    print(f"  components: {len(components)} (backbones={len(backbone_mem)}, task_components={len(pipeline_data)})")
    print(f"  pipelines: {len(pipelines)}")
    print(f"  devices: {sorted(set(d for lat in latency.values() for d in lat.keys()))}")

if __name__ == "__main__":
    main()
