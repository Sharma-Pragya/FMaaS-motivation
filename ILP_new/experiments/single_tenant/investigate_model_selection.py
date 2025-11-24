"""
Investigate model selection decisions to understand memory and latency trends.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from profiled_metadata.vlm_config import components, pipelines, latency as latency_config, metric


def analyze_deployment(result_file):
    """Analyze what models are deployed and their characteristics."""
    with open(result_file, 'r') as f:
        data = json.load(f)

    num_tasks = data['num_tasks']
    task_list = data['task_list']
    objective = data['objective_mode']
    metrics = data['metrics']

    print(f"\n{'='*80}")
    print(f"Tasks: {num_tasks} | Objective: {objective}")
    print(f"{'='*80}")
    print(f"Task list: {task_list}")
    print(f"Unique tasks: {len(set(task_list))}")
    print()

    # We need to re-run the ILP to get the actual deployed models
    # Since the result files don't contain x and r variables
    # Let's at least analyze what models COULD be selected

    # Find all models (backbones) that can serve these tasks
    unique_tasks = set(task_list)

    # Build backbone -> tasks mapping
    backbone_tasks = {}
    for pid, info in pipelines.items():
        backbone = info['backbone']
        task = info['task']
        if task in unique_tasks:
            if backbone not in backbone_tasks:
                backbone_tasks[backbone] = []
            backbone_tasks[backbone].append(task)

    print(f"Available backbones that can serve these tasks:")
    print(f"{'Backbone':<40} {'Tasks Covered':<10} {'Memory (MB)':<15} {'Avg Latency (ms)'}")
    print("-" * 90)

    backbone_info = []
    for backbone, tasks in backbone_tasks.items():
        tasks_covered = len(set(tasks))
        mem = components.get(backbone, {}).get('mem', 0)

        # Calculate average latency across tasks on A6000
        latencies = []
        for task in set(tasks):
            # Find pipeline for this backbone+task
            for pid, info in pipelines.items():
                if info['backbone'] == backbone and info['task'] == task:
                    lat = latency_config.get(pid, {}).get('A6000', None)
                    if lat:
                        latencies.append(lat)
                    break

        avg_lat = sum(latencies) / len(latencies) if latencies else 0

        backbone_info.append({
            'backbone': backbone,
            'tasks_covered': tasks_covered,
            'memory': mem,
            'avg_latency': avg_lat,
            'can_serve_all': tasks_covered >= len(unique_tasks)
        })

        serve_all = "✓ ALL" if tasks_covered >= len(unique_tasks) else ""
        print(f"{backbone:<40} {tasks_covered:<10} {mem:<15.1f} {avg_lat:<15.1f} {serve_all}")

    # Sort by different criteria to see trade-offs
    print(f"\n{'─'*80}")
    print("BEST OPTIONS BY DIFFERENT CRITERIA:")
    print(f"{'─'*80}")

    # Models that can serve all tasks
    all_serving = [b for b in backbone_info if b['can_serve_all']]

    if all_serving:
        print("\n1. SMALLEST MEMORY (among models that serve all tasks):")
        best_mem = min(all_serving, key=lambda x: x['memory'])
        print(f"   {best_mem['backbone']}: {best_mem['memory']:.1f} MB, {best_mem['avg_latency']:.1f} ms")

        print("\n2. LOWEST LATENCY (among models that serve all tasks):")
        best_lat = min(all_serving, key=lambda x: x['avg_latency'])
        print(f"   {best_lat['backbone']}: {best_lat['memory']:.1f} MB, {best_lat['avg_latency']:.1f} ms")

        print("\n3. BEST MEMORY-LATENCY TRADE-OFF:")
        # Normalize and combine
        min_mem = min(b['memory'] for b in all_serving)
        max_mem = max(b['memory'] for b in all_serving)
        min_lat = min(b['avg_latency'] for b in all_serving)
        max_lat = max(b['avg_latency'] for b in all_serving)

        for b in all_serving:
            norm_mem = (b['memory'] - min_mem) / (max_mem - min_mem) if max_mem > min_mem else 0
            norm_lat = (b['avg_latency'] - min_lat) / (max_lat - min_lat) if max_lat > min_lat else 0
            b['score'] = norm_mem + norm_lat

        best_tradeoff = min(all_serving, key=lambda x: x['score'])
        print(f"   {best_tradeoff['backbone']}: {best_tradeoff['memory']:.1f} MB, {best_tradeoff['avg_latency']:.1f} ms")

    # What the ILP actually chose
    print(f"\n{'─'*80}")
    print("ACTUAL ILP SOLUTION:")
    print(f"{'─'*80}")
    print(f"Memory footprint: {metrics['memory_mb']:.1f} MB")
    print(f"Average latency: {metrics['avg_latency_ms']:.1f} ms")
    print(f"Num deployments: {metrics['num_deployments']}")
    print(f"Num devices: {metrics['num_devices']}")

    return {
        'num_tasks': num_tasks,
        'unique_tasks': len(unique_tasks),
        'memory': metrics['memory_mb'],
        'latency': metrics['avg_latency_ms'],
        'deployments': metrics['num_deployments'],
        'best_single_memory': best_mem['memory'] if all_serving else None,
        'best_single_latency': best_lat['avg_latency'] if all_serving else None,
    }


def compare_across_task_counts():
    """Compare model selection across different task counts."""
    base = os.path.join(os.path.dirname(__file__), "results", "scaling", "vlm")

    print("\n" + "="*80)
    print("COMPARING MODEL SELECTION ACROSS TASK COUNTS")
    print("Objective: deployments (O1 only)")
    print("="*80)

    summaries = []
    for task_count in [2, 4, 6, 8, 10]:
        result_file = f"{base}/vlm_{task_count}tasks_deployments.json"
        if os.path.exists(result_file):
            summary = analyze_deployment(result_file)
            summaries.append(summary)

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Tasks':<8} {'Unique':<8} {'Memory(MB)':<12} {'Latency(ms)':<13} {'Deps':<6} {'Best Single (MB)':<18}")
    print("-" * 80)

    for s in summaries:
        best_single = f"{s['best_single_memory']:.1f}" if s['best_single_memory'] else "N/A"
        print(f"{s['num_tasks']:<8} {s['unique_tasks']:<8} {s['memory']:<12.1f} {s['latency']:<13.1f} {s['deployments']:<6} {best_single:<18}")

    # Analyze trends
    print("\n" + "="*80)
    print("TREND ANALYSIS")
    print("="*80)

    print("\n1. MEMORY TREND:")
    print("   2 tasks: 16889 MB (one large model)")
    print("   4 tasks:  7154 MB (one smaller model) ← DECREASES!")
    print("   6 tasks:  8668 MB (one model)")
    print("   8 tasks: 20585 MB (TWO models)")
    print()
    print("   EXPLANATION: With more tasks, ILP has more flexibility.")
    print("   - 2 tasks: Limited options, may need large model")
    print("   - 4 tasks: More models can cover 4 tasks, picks smaller one")
    print("   - 8 tasks: Needs 2 deployments, total memory increases")

    print("\n2. LATENCY TREND:")
    print("   2 tasks: 421.7 ms")
    print("   4 tasks: 141.9 ms ← MUCH FASTER!")
    print("   6 tasks: 132.6 ms")
    print("   8 tasks: 128.6 ms")
    print()
    print("   EXPLANATION: ILP prioritizes minimizing deployments.")
    print("   - With 2 tasks, might pick a large/slow model that can serve both")
    print("   - With 4-6 tasks, picks faster models (same priority, more options)")
    print("   - Latency is not in the objective, so ILP doesn't optimize for it")
    print("   - But having more tasks gives ILP more flexibility to pick faster models")


if __name__ == "__main__":
    compare_across_task_counts()
