"""Analyze ILP solutions to debug scaling experiments."""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from profiled_metadata.vlm_config import components, pipelines, latency, metric


def analyze_solution(result_file):
    """Analyze a single ILP solution."""
    with open(result_file, 'r') as f:
        data = json.load(f)

    print(f"\n{'='*80}")
    print(f"File: {os.path.basename(result_file)}")
    print(f"{'='*80}")
    print(f"Tasks ({len(data['task_list'])}): {data['task_list']}")
    print(f"Unique tasks: {len(set(data['task_list']))}")

    # Check if full result exists
    full_result_file = result_file.replace('.json', '_full.json')
    if not os.path.exists(full_result_file):
        print("\nNo full result file found - need to check experiment output")
        print(f"Metrics summary:")
        for k, v in data['metrics'].items():
            print(f"  {k}: {v}")
        return

    with open(full_result_file, 'r') as f:
        full_data = json.load(f)

    # Analyze deployments (x variables)
    x = full_data.get('x', {})
    deployed = [(eval(k)[0], eval(k)[1]) for k, v in x.items() if v > 0.5]

    print(f"\nDeployed models ({len(deployed)}):")
    for m, d in deployed:
        print(f"  Model: {m}, Device: {d}")

    # Analyze routing (r variables)
    r = full_data.get('r', {})
    print(f"\nRouting decisions:")
    for k, v in r.items():
        if v > 0.01:
            t, m, d = eval(k)
            print(f"  Task {t} -> Model {m} on {d}: {v:.3f}")

    # Recompute throughput correctly
    print(f"\n--- THROUGHPUT ANALYSIS ---")
    print(f"Reported throughput: {data['metrics']['throughput_req_s']:.2f} req/s")
    print(f"Total demand: {data['metrics']['total_demand_req_s']:.2f} req/s")

    # What SHOULD be calculated: actual capacity being used, not total possible capacity


if __name__ == "__main__":
    base = "ILP_new/experiments/single_tenant/results/scaling/vlm"

    # Analyze a few key cases
    files = [
        f"{base}/vlm_2tasks_deployments.json",
        f"{base}/vlm_6tasks_deployments.json",
        f"{base}/vlm_10tasks_deployments.json",
    ]

    for f in files:
        if os.path.exists(f):
            analyze_solution(f)
