import json
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Directory containing results
RESULTS_DIR = Path(__file__).parent / "results"
MODEL = "momentlarge"  # Used for l4/t4 filtering

def extract_tpc_metrics():
    """
    Extract TPC count and mean_service_time_ms from summary.json files.
    Looks in model subdirs (e.g., momentlarge/tpc_*)
    Returns a dict: {device: {tpc_count: mean_service_time_ms}}
    """
    metrics = {}
    
    for device_dir in RESULTS_DIR.iterdir():
        if not device_dir.is_dir() or device_dir.name == "logs":
            continue
            
        device_name = device_dir.name
        metrics[device_name] = {}
        
        # All devices have model subdirs (e.g., momentlarge/tpc_*)
        for root, dirs, files in os.walk(device_dir):
            if "summary.json" in files and MODEL in root:
                path_parts = Path(root).parts
                tpc_dirs = [p for p in path_parts if re.match(r"^tpc_\d+$", p)]
                if tpc_dirs:
                    tpc_count = int(tpc_dirs[0].split("_")[1])
                    summary_path = Path(root) / "summary.json"
                    
                    with open(summary_path) as f:
                        data = json.load(f)
                        mean_service_time = data.get("mean_service_time_ms")
                        if mean_service_time is not None:
                            if tpc_count not in metrics[device_name]:
                                metrics[device_name][tpc_count] = mean_service_time
    
    return metrics

def plot_results(metrics):
    """Plot mean_service_time_ms vs TPC count for each device"""
    
    num_devices = len(metrics)
    fig, axes = plt.subplots(1, num_devices, figsize=(6*num_devices, 5))
    
    # Handle single device case
    if num_devices == 1:
        axes = [axes]
    
    for ax, (device_name, tpc_data) in zip(axes, sorted(metrics.items())):
        if not tpc_data:
            continue
            
        tpc_counts = sorted(tpc_data.keys())
        service_times = [tpc_data[tpc] for tpc in tpc_counts]
        
        ax.plot(tpc_counts, service_times, marker='o', linewidth=2, markersize=8, color='steelblue')
        ax.set_xlabel("Number of TPC", fontsize=12)
        ax.set_ylabel("Mean Service Time (ms)", fontsize=12)
        ax.set_title(f"Device: {device_name}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(tpc_counts)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "tpc_service_time_plot.png", dpi=300, bbox_inches='tight')
    print("✓ Plot saved to: tpc_service_time_plot.png")
    plt.show()

if __name__ == "__main__":
    metrics = extract_tpc_metrics()
    if metrics:
        print(f"Extracted data for devices: {list(metrics.keys())}")
        plot_results(metrics)
    else:
        print(f"❌ No results found for model '{MODEL}' in:", RESULTS_DIR)
