"""
Visualization script for request rate scaling experiments.

Generates 9 plots (3 metrics × 3 workload types):
- Memory Footprint (MB) vs. Request Rate
- Throughput (req/s) vs. Request Rate
- Average Latency (ms) vs. Request Rate

Each plot shows 4 lines (one per objective mode).

Usage:
    python visualize_reqrate.py                    # Generate all 9 plots
    python visualize_reqrate.py --workload vlm     # Only VLM plots
    python visualize_reqrate.py --metric memory    # Only memory plots

Output: ./results/reqrate/plots/
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

COLORS = {
    "deployments": "#ff7f0e",
    "deployments_devices": "#1f77b4",
    "deployments_devices_modelsize": "#2ca02c",
}

LABELS = {
    "deployments": "O1: Dep",
    "deployments_devices": "O1+O2: Dep+Dev",
    "deployments_devices_modelsize": "O1+O2+O4: Dep+Dev+Size",
}

MARKERS = {
    "deployments": "s",
    "deployments_devices": "^",
    "deployments_devices_modelsize": "o",
}

LINESTYLES = {
    "deployments": "-",
    "deployments_devices": "-",
    "deployments_devices_modelsize": "-",
}


def load_summary(workload_type, results_dir):
    """Load request rate summary for a workload type."""
    summary_file = os.path.join(results_dir, "summary", f"{workload_type}_reqrate_summary.json")
    if not os.path.exists(summary_file):
        print(f"Warning: {summary_file} not found")
        return None
    with open(summary_file, 'r') as f:
        return json.load(f)


def plot_metric(workload_type, metric_name, summary_data, output_dir, output_format="png"):
    """Generate a single plot for a given workload and metric."""
    if summary_data is None:
        return

    req_rates = sorted(summary_data["req_rates"])
    objective_modes = summary_data["objective_modes"]
    results = summary_data["results"]
    num_tasks = summary_data.get("num_tasks", 4)

    metric_configs = {
        "memory": {
            "key": "memory_mb",
            "ylabel": "Memory Footprint (MB)",
            "title": f"{workload_type.upper()}: Memory Footprint vs. Request Rate ({num_tasks} tasks)",
        },
        "throughput": {
            "key": "throughput_req_s",
            "ylabel": "Total System Throughput (req/s)",
            "title": f"{workload_type.upper()}: Throughput vs. Request Rate ({num_tasks} tasks)",
        },
        "latency": {
            "key": "avg_latency_ms",
            "ylabel": "Average Latency (ms)",
            "title": f"{workload_type.upper()}: Average Latency vs. Request Rate ({num_tasks} tasks)",
        },
    }

    if metric_name not in metric_configs:
        raise ValueError(f"Unknown metric: {metric_name}")

    config = metric_configs[metric_name]
    metric_key = config["key"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for obj_mode in objective_modes:
        if obj_mode not in results:
            continue

        x_vals = []
        y_vals = []

        for req_rate in req_rates:
            # JSON stores keys as strings for floats too
            rate_key = str(req_rate) if str(req_rate) in results[obj_mode] else str(int(req_rate))
            if rate_key in results[obj_mode]:
                metrics = results[obj_mode][rate_key]
                if metrics and metric_key in metrics:
                    x_vals.append(req_rate)
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

    ax.set_xlabel("Request Rate per Task (req/s)", fontsize=12, fontweight='bold')
    ax.set_ylabel(config["ylabel"], fontsize=12, fontweight='bold')
    ax.set_title(config["title"], fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(req_rates)

    plt.tight_layout()

    output_file = os.path.join(output_dir, f"{workload_type}_{metric_name}_reqrate.{output_format}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize request rate scaling experiment results")
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
        help="Results directory (default: ./results/reqrate)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf"],
        help="Output format",
    )

    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(__file__), "results", "reqrate")

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        print("Run run_reqrate_experiments.py first to generate data.")
        return

    output_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating request rate plots...")
    print(f"  Results dir: {args.results_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Workloads: {args.workload}")
    print(f"  Metrics: {args.metric}")
    print()

    plot_count = 0
    for workload in args.workload:
        print(f"\n{workload.upper()}:")
        summary = load_summary(workload, args.results_dir)
        if summary is None:
            continue

        for metric in args.metric:
            plot_metric(workload, metric, summary, output_dir, args.format)
            plot_count += 1

    print(f"\nDone! Generated {plot_count} plots in: {output_dir}")


if __name__ == "__main__":
    main()
