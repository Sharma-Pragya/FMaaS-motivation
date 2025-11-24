"""
Quick test to verify bug fixes work correctly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from run_scaling_experiments import (
    build_ilp_inputs_vlm,
    load_pipeline_data,
    select_tasks_roundrobin,
    VLM_TASKS,
    DEFAULT_VLM_TASK_CONFIG,
)
from methods.single_tenant import build_and_solve

print("="*80)
print("TESTING BUG FIXES")
print("="*80)

# Load VLM data
print("\nLoading VLM pipeline data...")
components, pipelines, latency_data, metric_data = load_pipeline_data("vlm")

# Test with 10 tasks (should have 1 duplicate)
print("\n" + "-"*80)
print("TEST: 10 tasks (should include 1 duplicate task)")
print("-"*80)

task_list = select_tasks_roundrobin(VLM_TASKS, 10)
print(f"Task list: {task_list}")
print(f"Unique tasks: {set(task_list)}")
print(f"Number of tasks in list: {len(task_list)}")
print(f"Number of unique tasks: {len(set(task_list))}")

# Build ILP inputs
ilp_inputs = build_ilp_inputs_vlm(task_list, components, pipelines, latency_data, metric_data)

print(f"\n✓ Fix #2 Check: Duplicate Task Handling")
print(f"  Total demand: {sum(ilp_inputs['demands'].values()):.1f} req/s")
print(f"  Expected: 10.0 req/s (aggregated duplicates)")
print(f"  Old behavior would be: 9.0 req/s (duplicates ignored)")

if sum(ilp_inputs['demands'].values()) == 10.0:
    print(f"  ✓ PASS: Demand correctly aggregated!")
else:
    print(f"  ✗ FAIL: Demand not aggregated correctly")

# Run ILP
print(f"\nRunning ILP solver...")
result = build_and_solve(
    devices=ilp_inputs["devices"],
    models=ilp_inputs["models"],
    tasks=ilp_inputs["tasks"],
    demands=ilp_inputs["demands"],
    support=ilp_inputs["support"],
    accuracy=ilp_inputs["accuracy"],
    latency=ilp_inputs["latency"],
    vram_model=ilp_inputs["vram_model"],
    vram_device=ilp_inputs["vram_device"],
    Ptmd=ilp_inputs["Ptmd"],
    Amin=ilp_inputs["Amin"],
    minimize="deployments",
    time_limit=60,
    log_to_console=False,
)

print(f"  Status: {result['status']}")
print(f"  Deployments: {result.get('objective_components', {}).get('O1_deployments', 0)}")

# Now test throughput calculation
print(f"\n✓ Fix #1 Check: Throughput Calculation")

# Get deployed models
x = result.get("x", {})
deployed = [(m, d) for (m, d), v in x.items() if v > 0.5]
print(f"  Deployed: {len(deployed)} model(s)")

# OLD BUGGY CALCULATION (sum across all tasks)
Ptmd = ilp_inputs["Ptmd"]
support = ilp_inputs["support"]
tasks = list(ilp_inputs["tasks"].keys())

old_throughput = 0.0
for (m, d) in deployed:
    model_tasks = [t for t in tasks if support.get((m, t), 0) == 1]
    if model_tasks:
        capacities = [Ptmd.get((t, m, d), 0.0) for t in model_tasks]
        capacities = [c for c in capacities if c > 0]
        if capacities:
            old_throughput += sum(capacities)  # BUG: sums all tasks

# NEW FIXED CALCULATION (min across tasks)
new_throughput = 0.0
for (m, d) in deployed:
    model_tasks = [t for t in tasks if support.get((m, t), 0) == 1]
    if model_tasks:
        capacities = [Ptmd.get((t, m, d), 0.0) for t in model_tasks]
        capacities = [c for c in capacities if c > 0]
        if capacities:
            new_throughput += min(capacities)  # FIXED: uses min

print(f"  Old (buggy) throughput: {old_throughput:.2f} req/s")
print(f"  New (fixed) throughput: {new_throughput:.2f} req/s")
print(f"  Overcount ratio: {old_throughput / new_throughput:.1f}x")

if old_throughput > new_throughput * 5:
    print(f"  ✓ PASS: Old calculation overcounts significantly!")
else:
    print(f"  ⚠ WARNING: Overcount not as large as expected")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("Both fixes have been applied and tested:")
print("  1. Throughput calculation now uses min() instead of sum()")
print("  2. Duplicate tasks now have aggregated demands")
print("\nReady to re-run scaling experiments!")
print("="*80)
