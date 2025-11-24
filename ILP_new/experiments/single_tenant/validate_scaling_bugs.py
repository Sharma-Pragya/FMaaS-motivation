"""
Validate and demonstrate the scaling experiment bugs.
"""
import json
import os


def analyze_vlm_scaling():
    """Analyze VLM scaling results to show bugs."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(script_dir, "results", "scaling")
    summary_file = f"{base}/summary/vlm_scaling_summary.json"

    with open(summary_file, 'r') as f:
        data = json.load(f)

    print("="*80)
    print("VLM SCALING EXPERIMENT BUG ANALYSIS")
    print("="*80)

    # Bug #1: Throughput Overcounting
    print("\n### BUG #1: THROUGHPUT OVERCOUNTING ###\n")
    print("Objective: deployments (O1 only)")
    print(f"{'Tasks':>6} {'Demand':>8} {'Throughput':>12} {'Ratio':>8} {'Deployments':>12}")
    print("-" * 55)

    results = data["results"]["deployments"]
    for task_count in data["task_counts"]:
        metrics = results.get(str(task_count))
        if metrics:
            demand = metrics["total_demand_req_s"]
            throughput = metrics["throughput_req_s"]
            ratio = throughput / demand if demand > 0 else 0
            deployments = metrics["num_deployments"]
            print(f"{task_count:6} {demand:8.1f} {throughput:12.2f} {ratio:8.1f}x {deployments:12}")

    print("\nPROBLEM: Throughput increases linearly with tasks!")
    print("- 2 tasks: 4.76 req/s capacity")
    print("- 10 tasks: 143.85 req/s capacity (30x increase!)")
    print("- But only 2 deployments (2x increase)")
    print("\nThis is because throughput is summed across ALL tasks a model supports.")
    print("If a model supports 6 tasks @ 10 req/s each, it counts as 60 req/s total!")

    # Bug #2: Duplicate Tasks
    print("\n\n### BUG #2: DUPLICATE TASK HANDLING ###\n")
    print(f"{'Tasks':>6} {'Unique':>8} {'Demand':>8} {'Status':>15}")
    print("-" * 45)

    vlm_dir = f"{base}/vlm"
    for task_count in data["task_counts"]:
        result_file = f"{vlm_dir}/vlm_{task_count}tasks_deployments.json"
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result = json.load(f)
            task_list = result["task_list"]
            unique_tasks = len(set(task_list))
            demand = result["metrics"]["total_demand_req_s"]

            status = "OK" if unique_tasks == task_count else "DUPLICATE!"
            print(f"{task_count:6} {unique_tasks:8} {demand:8.1f} {status:>15}")

    print("\nPROBLEM: VLM has only 9 unique tasks!")
    print("- Tasks 10-16 all show demand = 9.0 (not 10, 12, 14, 16)")
    print("- Round-robin creates duplicate task names")
    print("- ILP treats duplicate names as single task")
    print("- Results plateau and are identical for 10, 12, 14, 16 tasks")

    # Bug #3: Expected Trends
    print("\n\n### BUG #3: UNEXPECTED TRENDS ###\n")
    print("As tasks increase, we expect:")
    print("  Memory: INCREASE (more/larger models needed)")
    print("  Latency: INCREASE or FLAT (system more loaded)")
    print("  Throughput: FLAT or SLIGHT INCREASE (more deployments)")
    print()
    print(f"{'Tasks':>6} {'Memory(GB)':>12} {'Latency(ms)':>13} {'Trend':>20}")
    print("-" * 60)

    prev_memory = None
    prev_latency = None

    for task_count in [2, 4, 6, 8, 10]:
        metrics = results.get(str(task_count))
        if metrics:
            memory_gb = metrics["memory_mb"] / 1024
            latency = metrics["avg_latency_ms"]

            trend = ""
            if prev_memory and prev_latency:
                mem_change = "↑" if memory_gb > prev_memory else ("↓" if memory_gb < prev_memory else "→")
                lat_change = "↑" if latency > prev_latency else ("↓" if latency < prev_latency else "→")
                trend = f"Mem:{mem_change} Lat:{lat_change}"

            print(f"{task_count:6} {memory_gb:12.1f} {latency:13.1f} {trend:>20}")

            prev_memory = memory_gb
            prev_latency = latency

    print("\nPROBLEM: Latency DECREASES as tasks increase!")
    print("- 2 tasks: 422ms average latency")
    print("- 6 tasks: 133ms average latency (3.2x faster!?)")
    print("\nThis could be due to:")
    print("- ILP selecting faster models when more tasks present")
    print("- More optimization flexibility with more tasks")
    print("- OR a bug in latency calculation")

    # Compare objectives
    print("\n\n### COMPARISON ACROSS OBJECTIVES ###\n")
    print("Memory footprint at 8 tasks:")
    for obj in data["objective_modes"]:
        metrics = data["results"][obj].get("8")
        if metrics:
            memory_gb = metrics["memory_mb"] / 1024
            deployments = metrics["num_deployments"]
            print(f"  {obj:45s}: {memory_gb:6.1f} GB ({deployments} deployments)")

    print("\nPROBLEM: Huge jump for waste+modelsize objectives!")
    print("- O1 (deployments only): 20.1 GB")
    print("- O1+O2+O3+O4 (all objectives): 56.4 GB (2.8x more!)")
    print("\nWhy does optimizing for waste choose such huge models?")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("1. Throughput is OVERCOUNTED by summing across all supported tasks")
    print("2. Duplicate tasks are not handled - results plateau at 9 tasks for VLM")
    print("3. Latency DECREASES with more tasks (counterintuitive)")
    print("4. Waste optimization leads to huge memory footprints")
    print("\nAll plots and conclusions based on these experiments are INVALID.")
    print("="*80)


if __name__ == "__main__":
    analyze_vlm_scaling()
