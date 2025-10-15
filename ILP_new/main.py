from ILP_new.methods.our_linear import build_and_solve
from ILP_new import config, profiler
import os
from collections import defaultdict

os.environ["GRB_LICENSE_FILE"] = "gurobi/gurobi.lic"

def fmt(x, nd=3):
    if x is None: return "–"
    return f"{x:.{nd}f}"

def summarize_solution(result, devices, models, tasks, demands, Pmd, latency_tmd, accuracy_tm, vram_model, vram_device):
    print("\n================== SOLUTION SUMMARY ==================")
    print(f"Status: {result.get('status','?')}")
    obj = result.get('obj', None)
    if obj is not None:
        print(f"Objective (min deployments): {fmt(obj,0)}")

    # Early exit for infeasible
    if result.get('status') not in ('OPTIMAL','TIME_LIMIT'):
        twft = result.get('tasks_with_no_feasible_triple', [])
        if twft:
            print(f"\nNo feasible (task, model, device) for {len(twft)} tasks:")
            for t in twft[:10]:
                print(f"  - {t}")
            if len(twft) > 10:
                print(f"  ... and {len(twft)-10} more")
        print("======================================================\n")
        return

    x = result.get('x', {})
    r = result.get('r', {})
    # u = result.get('u', {})  # (optional) “home” endpoints, not needed for printing

    # ----------------- Selected deployments -----------------
    selected = [(m,d) for (m,d),val in x.items() if val > 0.5]
    print("\nSelected deployments (x=1):")
    if not selected:
        print("  (none)")
    else:
        for (m,d) in selected:
            print(f"  - {m}  @  {d}")

    # ----------------- Device VRAM usage -----------------
    vram_used = defaultdict(float)
    for (m,d),val in x.items():
        if val > 0.5:
            vram_used[d] += float(vram_model.get(m, 0.0))

    print("\nPer-device VRAM usage:")
    for d in devices:
        dtype = devices[d]["type"]
        used = vram_used[d]
        cap  = float(vram_device.get(dtype, 0.0))
        pct  = (used / cap * 100.0) if cap > 0 else 0.0
        print(f"  - {d:<8} used {fmt(used,1)} MB / cap {fmt(cap,1)} MB  ({fmt(pct,1)}%)")

    # ----------------- Per-task routing -----------------
    print("\nPer-task routing (fractions r, loads, util, latency, accuracy):")
    for t in tasks:
        print(f"  Task: {t}")
        task_routes = []
        for (tt,m,d), frac in r.items():
            if tt == t and frac > 1e-9:
                load = demands[t] * frac  # req/s sent to this endpoint
                cap  = float(Pmd.get((m,d), 0.0))
                util = (load / cap) if cap > 0 else 0.0
                lat  = latency_tmd.get((t,m,d), None)
                acc  = accuracy_tm.get((t,m), None)
                task_routes.append((m,d,frac,load,cap,util,lat,acc))

        if not task_routes:
            print("    (no routing)")
            continue

        # Sort by contribution (largest load first)
        task_routes.sort(key=lambda z: z[3], reverse=True)
        for (m,d,frac,load,cap,util,lat,acc) in task_routes:
            print(f"    - {m} @ {d}: r={fmt(frac,3)}, load={fmt(load,3)} req/s, cap={fmt(cap,3)} "
                  f"(util={fmt(100*util,1)}%), L={fmt(lat,1)} ms, A={fmt(acc,3)}")

        # sanity: sum r = 1.0
        rsum = sum(frac for *_, frac, __ in [(*x[:2], x[2], x[3]) for x in task_routes])  # or simply sum(frac)
        print(f"    sum r = {fmt(sum(fr for *_, fr, __ in task_routes),6)}")

    # ----------------- Endpoint slack summary -----------------
    print("\nEndpoint capacity slack (aggregated over tasks):")
    # aggregate loads per (m,d)
    load_md = defaultdict(float)
    for (t,m,d), frac in r.items():
        if frac <= 1e-9: continue
        load_md[(m,d)] += demands[t] * frac
    for (m,d) in selected:
        cap = float(Pmd.get((m,d), 0.0))
        ld  = load_md.get((m,d), 0.0)
        slack = cap - ld
        util = (ld / cap * 100.0) if cap > 0 else 0.0
        print(f"  - {m} @ {d}: load={fmt(ld,3)} / cap={fmt(cap,3)}  (util={fmt(util,1)}%, slack={fmt(slack,3)} req/s)")

    print("\n======================================================\n")

def main():
    devices = config.devices
    models = config.models
    tasks = config.tasks

    # Demand and accuracy thresholds
    demands = {t: 55.0 for t in tasks}   # req/s per task
    Amin = {t: 0.5 for t in tasks}       # accuracy thresholds

    # Use data from profiler
    latency_tmd = profiler.latency_tmd
    accuracy_tm = profiler.accuracy_tm
    vram_model  = profiler.model_memory
    vram_device = profiler.memory_device
    Pmd         = profiler.throughput_capacity

    print(f"[sanity] latency entries: {len(latency_tmd)}")

    result = build_and_solve(
        devices=devices,
        models=models,
        tasks=tasks,
        demands=demands,
        support=profiler.can_serve,
        accuracy=accuracy_tm,
        latency=latency_tmd,
        vram_model=vram_model,
        vram_device=vram_device,
        Pmd=Pmd,               
        Amin=Amin,
        redundancy=1,
        minimize="deployments",
        time_limit=60,
    )

    summarize_solution(
        result=result,
        devices=devices,
        models=models,
        tasks=tasks,
        demands=demands,
        Pmd=Pmd,
        latency_tmd=latency_tmd,
        accuracy_tm=accuracy_tm,
        vram_model=vram_model,
        vram_device=vram_device
    )

if __name__ == "__main__":
    main()