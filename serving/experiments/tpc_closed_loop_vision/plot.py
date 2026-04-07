import json
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Directory containing results
RESULTS_DIR = Path(__file__).parent / "results"

def extract_model_device_metrics():
    """
    Extract TPC count and mean_service_time_ms grouped by model and device.
    Returns a dict: {model: {device: {tpc_count: mean_service_time_ms}}}
    """
    metrics = {}
    
    for device_dir in RESULTS_DIR.iterdir():
        if not device_dir.is_dir() or device_dir.name == "logs":
            continue
            
        device_name = device_dir.name
        
        # Look for model directories within device directory
        for model_dir in device_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            
            # Recursively search for tpc_* directories
            for root, dirs, files in os.walk(model_dir):
                if "summary.json" in files:
                    path_parts = Path(root).parts
                    # Find tpc_<digits> in the path
                    tpc_dirs = [p for p in path_parts if re.match(r"^tpc_\d+$", p)]
                    if tpc_dirs:
                        tpc_count = int(tpc_dirs[0].split("_")[1])
                        summary_path = Path(root) / "summary.json"
                        
                        with open(summary_path) as f:
                            data = json.load(f)
                            mean_service_time = data.get("mean_service_time_ms")
                            if mean_service_time is not None:
                                if model_name not in metrics:
                                    metrics[model_name] = {}
                                if device_name not in metrics[model_name]:
                                    metrics[model_name][device_name] = {}
                                if tpc_count not in metrics[model_name][device_name]:
                                    metrics[model_name][device_name][tpc_count] = mean_service_time
    
    return metrics

def plot_model_results(model_name, device_metrics):
    """
    Plot mean_service_time_ms vs TPC count for a specific model across all devices.
    device_metrics: {device: {tpc_count: mean_service_time_ms}}
    """
    
    num_devices = len(device_metrics)
    fig, axes = plt.subplots(1, num_devices, figsize=(6*num_devices, 5))
    
    # Handle single device case
    if num_devices == 1:
        axes = [axes]
    
    for ax, (device_name, tpc_data) in zip(axes, sorted(device_metrics.items())):
        if not tpc_data:
            continue
            
        tpc_counts = sorted(tpc_data.keys())
        service_times = [tpc_data[tpc] for tpc in tpc_counts]
        
        ax.plot(tpc_counts, service_times, marker='o', linewidth=2, markersize=8, color='steelblue')
        ax.set_xlabel("Number of TPC", fontsize=12)
        ax.set_ylabel("Mean Service Time (ms)", fontsize=12)
        ax.set_title(f"{model_name} - Device: {device_name}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(tpc_counts)
    
    plt.tight_layout()
    filename = f"tpc_service_time_{model_name}.png"
    plt.savefig(Path(__file__).parent / filename, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {filename}")
    plt.show()

if __name__ == "__main__":
    all_metrics = extract_model_device_metrics()
    
    if all_metrics:
        print(f"Found models: {list(all_metrics.keys())}\n")
        
        # Plot each model separately
        for model_name in sorted(all_metrics.keys()):
            print(f"Plotting {model_name}...")
            plot_model_results(model_name, all_metrics[model_name])
    else:
        print("❌ No results found in:", RESULTS_DIR)
