"""
Cross-workload comparison visualization.

Generates 3 plots comparing VLM, TSFM, and Mixed workloads:
- Memory Footprint vs. Request Rate
- Throughput vs. Request Rate
- Latency vs. Request Rate

Each plot shows 6 lines (3 workloads × 2 objectives):
- O1+O2: Deployments + Devices
- O1+O2+O4: Deployments + Devices + ModelSize
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Styling
COLORS = {
    "vlm": "#1f77b4",      # blue
    "tsfm": "#2ca02c",     # green
    "mixed": "#d62728",    # red
}

LABELS = {
    "vlm": "VLM",
    "tsfm": "TSFM",
    "mixed": "Mixed",
}

OBJECTIVE_STYLES = {
    "deployments_devices": {
        "linestyle": "-",
        "linewidth": 2.5,
        "label_suffix": ": Dep+Dev",
    },
    "deployments_devices_modelsize": {
        "linestyle": "--",
        "linewidth": 3.5,
        "label_suffix": ": Dep+Dev+Size",
    },
}

MARKERS = {
    "vlm": "o",
    "tsfm": "s",
    "mixed": "^",
}


def load_summary(workload_type, results_dir):
    """Load reqrate summary for a workload type."""
    summary_file = os.path.join(results_dir, "summary", f"{workload_type}_reqrate_summary.json")
    if not os.path.exists(summary_file):
        print(f"Warning: {summary_file} not found")
        return None

    with open(summary_file, 'r') as f:
        return json.load(f)


def plot_cross_workload_metric(metric_name, summaries, output_dir, output_format="png"):
    """
    Generate a single cross-workload plot for a metric.

    Args:
        metric_name: "memory", "throughput", or "latency"
        summaries: dict of {workload_type: summary_data}
        output_dir: directory to save plot
        output_format: "png" or "pdf"
    """
    # Metric config
    metric_configs = {
        "memory": {
            "key": "memory_mb",
            "ylabel": "Memory Footprint (MB)",
            "title": "Cross-Workload: Memory Footprint vs. Request Rate",
        },
        "throughput": {
            "key": "throughput_req_s",
            "ylabel": "Total System Throughput (req/s)",
            "title": "Cross-Workload: Throughput vs. Request Rate",
        },
        "latency": {
            "key": "avg_latency_ms",
            "ylabel": "Average Latency (ms)",
            "title": "Cross-Workload: Average Latency vs. Request Rate",
        },
    }

    if metric_name not in metric_configs:
        raise ValueError(f"Unknown metric: {metric_name}")

    config = metric_configs[metric_name]
    metric_key = config["key"]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each workload × objective combination
    for workload_type, summary in summaries.items():
        if summary is None:
            continue

        req_rates = sorted(summary["req_rates"])

        # Plot both O1+O2 and O1+O2+O4
        for obj_mode in ["deployments_devices", "deployments_devices_modelsize"]:
            if obj_mode not in summary["results"]:
                continue

            x_vals = []
            y_vals = []

            for num_tasks in req_rates:
                task_key = str(num_tasks)
                if task_key in summary["results"][obj_mode]:
                    metrics = summary["results"][obj_mode][task_key]
                    if metrics and metric_key in metrics:
                        x_vals.append(num_tasks)
                        y_vals.append(metrics[metric_key])

            if x_vals:
                obj_style = OBJECTIVE_STYLES[obj_mode]
                label = LABELS[workload_type] + obj_style["label_suffix"]

                ax.plot(
                    x_vals, y_vals,
                    marker=MARKERS[workload_type],
                    color=COLORS[workload_type],
                    linestyle=obj_style["linestyle"],
                    label=label,
                    linewidth=2.5,
                    markersize=9,
                    alpha=0.8,
                )

    # Formatting
    ax.set_xlabel("Request Rate per Task (req/s)", fontsize=13, fontweight='bold')
    ax.set_ylabel(config["ylabel"], fontsize=13, fontweight='bold')
    ax.set_title(config["title"], fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save
    output_file = os.path.join(output_dir, f"cross_workload_{metric_name}.{output_format}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results", "reqrate")
    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    print("Generating cross-workload comparison plots...")
    print(f"  Results dir: {results_dir}")
    print(f"  Output dir: {output_dir}")
    print()

    # Load all workload summaries
    summaries = {}
    for workload in ["vlm", "tsfm", "mixed"]:
        print(f"Loading {workload.upper()} summary...")
        summaries[workload] = load_summary(workload, results_dir)

    print()

    # Generate plots for each metric
    for metric in ["memory", "throughput", "latency"]:
        print(f"Plotting {metric}...")
        plot_cross_workload_metric(metric, summaries, output_dir)

    print(f"\nDone! Generated 3 cross-workload plots in: {output_dir}")


if __name__ == "__main__":
    main()
