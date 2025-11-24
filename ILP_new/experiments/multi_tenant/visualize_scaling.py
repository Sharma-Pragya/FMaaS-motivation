"""
Visualization script for scaling experiments.

Generates 9 plots (3 metrics × 3 workload types):
- Memory Footprint (MB) vs. Number of Tasks
- Throughput (req/s) vs. Number of Tasks
- Average Latency (ms) vs. Number of Tasks

Each plot shows 4 lines (one per objective mode).

Usage:
    python visualize_scaling.py                    # Generate all 9 plots
    python visualize_scaling.py --workload vlm     # Only VLM plots
    python visualize_scaling.py --metric memory    # Only memory plots
    python visualize_scaling.py --format pdf       # Save as PDF instead of PNG

Output: ./results/scaling/plots/
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Styling
COLORS = {
    "deployments": "#1f77b4",  # blue
    "deployments_devices": "#ff7f0e",  # orange
    "deployments_devices_modelsize": "#9467bd",  # purple
    "deployments_devices_waste": "#2ca02c",  # green
    "deployments_devices_waste_modelsize": "#d62728",  # red
}

LABELS = {
    "deployments": "Deployments",
    "deployments_devices": "Dep+Dev",
    "deployments_devices_modelsize": "Dep+Dev+Size",
    "deployments_devices_waste": "Dep+Dev+Waste",
    "deployments_devices_waste_modelsize": "Dep+Dev+Waste+Size",
}

MARKERS = {
    "deployments": "o",
    "deployments_devices": "s",
    "deployments_devices_modelsize": "v",
    "deployments_devices_waste": "^",
    "deployments_devices_waste_modelsize": "D",
}

LINESTYLES = {
    "deployments": "-",
    "deployments_devices": "-",
    "deployments_devices_modelsize": "-",
    "deployments_devices_waste": "--",
    "deployments_devices_waste_modelsize": ":",
}


def load_summary(workload_type, results_dir):
    """Load scaling summary for a workload type."""
    summary_file = os.path.join(results_dir, "summary", f"{workload_type}_scaling_summary.json")
    if not os.path.exists(summary_file):
        print(f"Warning: {summary_file} not found")
        return None

    with open(summary_file, 'r') as f:
        return json.load(f)


def plot_metric(workload_type, metric_name, summary_data, output_dir, output_format="png"):
    """
    Generate a single plot for a given workload and metric.

    Args:
        workload_type: "vlm", "tsfm", or "mixed"
        metric_name: "memory", "throughput", or "latency"
        summary_data: loaded summary JSON
        output_dir: directory to save plot
        output_format: "png" or "pdf"
    """
    if summary_data is None:
        return

    task_counts = sorted(summary_data["task_counts"])
    objective_modes = summary_data["objective_modes"]
    results = summary_data["results"]

    # Metric config
    metric_configs = {
        "memory": {
            "key": "memory_mb",
            "ylabel": "Memory Footprint (MB)",
            "title": f"{workload_type.upper()}: Memory Footprint vs. Number of Tasks",
        },
        "throughput": {
            "key": "throughput_req_s",
            "ylabel": "Total System Throughput (req/s)",
            "title": f"{workload_type.upper()}: Throughput vs. Number of Tasks",
        },
        "latency": {
            "key": "avg_latency_ms",
            "ylabel": "Average Latency (ms)",
            "title": f"{workload_type.upper()}: Average Latency vs. Number of Tasks",
        },
    }

    if metric_name not in metric_configs:
        raise ValueError(f"Unknown metric: {metric_name}")

    config = metric_configs[metric_name]
    metric_key = config["key"]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each objective mode
    for obj_mode in objective_modes:
        if obj_mode not in results:
            continue

        x_vals = []
        y_vals = []

        for num_tasks in task_counts:
            # JSON stores keys as strings, so convert to string
            task_key = str(num_tasks)
            if task_key in results[obj_mode]:
                metrics = results[obj_mode][task_key]
                if metrics and metric_key in metrics:
                    x_vals.append(num_tasks)
                    y_vals.append(metrics[metric_key])

        if x_vals:
            ax.plot(
                x_vals, y_vals,
                marker=MARKERS[obj_mode],
                color=COLORS[obj_mode],
                linestyle=LINESTYLES.get(obj_mode, "-"),
                label=LABELS[obj_mode],
                linewidth=2,
                markersize=8,
            )

    # Formatting
    ax.set_xlabel("Number of Tasks", fontsize=12, fontweight='bold')
    ax.set_ylabel(config["ylabel"], fontsize=12, fontweight='bold')
    ax.set_title(config["title"], fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set x-axis to show only integer ticks
    ax.set_xticks(task_counts)

    plt.tight_layout()

    # Save
    output_file = os.path.join(output_dir, f"{workload_type}_{metric_name}.{output_format}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize scaling experiment results")
    parser.add_argument(
        "--workload",
        type=str,
        nargs="+",
        default=["vlm", "tsfm", "mixed"],
        choices=["vlm", "tsfm", "mixed"],
        help="Workload types to visualize",
    )
    parser.add_argument(
        "--metric",
        type=str,
        nargs="+",
        default=["memory", "throughput", "latency"],
        choices=["memory", "throughput", "latency"],
        help="Metrics to visualize",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (default: ./results/scaling)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf"],
        help="Output format",
    )

    args = parser.parse_args()

    # Resolve results directory
    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(__file__), "results", "scaling")

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        print("Run run_scaling_experiments.py first to generate data.")
        return

    # Create output directory
    output_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating plots...")
    print(f"  Results dir: {args.results_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Workloads: {args.workload}")
    print(f"  Metrics: {args.metric}")
    print()

    # Generate plots
    plot_count = 0
    for workload in args.workload:
        print(f"\n{workload.upper()}:")

        # Load summary
        summary = load_summary(workload, args.results_dir)
        if summary is None:
            continue

        # Generate plots for each metric
        for metric in args.metric:
            plot_metric(workload, metric, summary, output_dir, args.format)
            plot_count += 1

    print(f"\nDone! Generated {plot_count} plots in: {output_dir}")


if __name__ == "__main__":
    main()
