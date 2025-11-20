# Test to compare old our_singletenant.py with new single_tenant.py
# Uses the old config/profiler data (1 task) to verify consistency

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

os.environ.setdefault("GRB_LICENSE_FILE",
    os.path.join(os.path.dirname(__file__), "../../gurobi/gurobi.lic"))

# Import old data
import config
import profiler

# Import both solvers
from methods.our_singletenant import build_and_solve as solve_old
from methods.single_tenant import build_and_solve as solve_new


def run_test(demand_level):
    # Use old config data
    devices = config.devices
    models = config.models
    tasks = config.tasks  # {'dataset/val2014': 1120}
    task_list = list(tasks.keys())

    demands = {t: demand_level for t in task_list}

    # Old solver inputs
    latency = profiler.latency_tmd
    accuracy = profiler.accuracy_tm
    vram_model = profiler.model_memory
    vram_device = profiler.memory_device
    Pmd = profiler.throughput_capacity
    support = profiler.can_serve

    # Amin threshold
    Amin = {t: 0.5 for t in task_list}

    # Run OLD solver
    result_old = solve_old(
        devices=devices,
        models=models,
        tasks=tasks,
        demands=demands,
        support=support,
        accuracy=accuracy,
        latency=latency,
        vram_model=vram_model,
        vram_device=vram_device,
        Pmd=Pmd,
        Amin=Amin,
        redundancy=1,
        minimize="deployments",
        time_limit=60,
        log_to_console=False,
    )

    # Run NEW solver - Convert Pmd to Ptmd format
    Ptmd = {}
    for (m, d), cap in Pmd.items():
        for t in task_list:
            Ptmd[(t, m, d)] = cap

    result_new = solve_new(
        devices=devices,
        models=models,
        tasks=tasks,
        demands=demands,
        support=support,
        accuracy=accuracy,
        latency=latency,
        vram_model=vram_model,
        vram_device=vram_device,
        Ptmd=Ptmd,
        Amin=Amin,
        minimize="deployments",
        time_limit=60,
        log_to_console=False,
    )

    return result_old, result_new


def main():
    print("=" * 60)
    print("TEST: Comparing old vs new single-tenant ILP")
    print("=" * 60)
    print(f"Tasks: {list(config.tasks.keys())}")
    print(f"Models: {len(config.models)}")
    print(f"Devices: {list(config.devices.keys())}")

    all_pass = True

    for demand_level in [5.0, 10.0, 20.0]:
        print(f"\n--- Demand: {demand_level} req/s ---")

        result_old, result_new = run_test(demand_level)

        obj_old = result_old.get('obj')
        obj_new = result_new.get('obj')

        x_old = result_old.get('x', {})
        x_new = result_new.get('x', {})
        deployed_old = len([(m, d) for (m, d), v in x_old.items() if v > 0.5])
        deployed_new = len([(m, d) for (m, d), v in x_new.items() if v > 0.5])

        print(f"OLD: {result_old['status']}, obj={obj_old}, deployments={deployed_old}")
        print(f"NEW: {result_new['status']}, obj={obj_new}, deployments={deployed_new}")

        if obj_old == obj_new and result_old['status'] == result_new['status']:
            print("✓ MATCH")
        else:
            print("✗ MISMATCH")
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("✓ ALL TESTS PASSED - New ILP matches old ILP")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
