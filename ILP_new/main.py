# ILP_new/main.py
import os
import argparse
from collections import defaultdict

# point to the license (override from environment if already set)
os.environ.setdefault("GRB_LICENSE_FILE", "gurobi/gurobi.lic")

from ILP_new import config, profiler

# import both solvers; choose at runtime
from ILP_new.methods.our_linear import build_and_solve as solve_multi
from ILP_new.methods.our_singletenant import build_and_solve as solve_single


def fmt(x, nd=3):
    if x is None:
        return "–"
    try:
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def summarize_solution(tag, result, devices, demands, Pmd, latency_tmd, accuracy_tm, vram_model, vram_device):
    print(f"\n================== SOLUTION SUMMARY [{tag}] ==================")
    print(f"Status: {result.get('status','?')}")
    obj = result.get('obj', None)
    if obj is not None:
        print(f"Objective (min deployments): {fmt(obj,0)}")

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

    # Selected deployments
    selected = [(m,d) for (m,d),val in x.items() if val > 0.5]
    print("\nSelected deployments (x=1):")
    if not selected:
        print("  (none)")
    else:
        for (m,d) in selected:
            print(f"  - {m}  @  {d}")

    # Device VRAM usage
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

    # Per-task routing
    print("\nPer-task routing (r, load, util, latency, accuracy):")
    for t in config.tasks:
        print(f"  Task: {t}")
        task_routes = []
        for (tt,m,d), frac in r.items():
            if tt == t and frac > 1e-9:
                load = demands[t] * frac
                cap  = float(Pmd.get((m,d), 0.0))
                util = (load / cap) if cap > 0 else 0.0
                lat  = latency_tmd.get((t,m,d), None)
                acc  = accuracy_tm.get((t,m), None)
                task_routes.append((m,d,frac,load,cap,util,lat,acc))

        if not task_routes:
            print("    (no routing)")
            continue

        task_routes.sort(key=lambda z: z[3], reverse=True)
        for (m,d,frac,load,cap,util,lat,acc) in task_routes:
            print(f"    - {m} @ {d}: r={fmt(frac,3)}, load={fmt(load,3)} req/s, cap={fmt(cap,3)} "
                  f"(util={fmt(100*util,1)}%), L={fmt(lat,1)} ms, A={fmt(acc,3)}")
        print(f"    sum r = {fmt(sum(fr for *_ , fr, __ in task_routes),6)}")

    # Endpoint capacity slack
    print("\nEndpoint capacity slack (aggregated over tasks):")
    load_md = defaultdict(float)
    for (t,m,d), frac in r.items():
        if frac <= 1e-9:
            continue
        load_md[(m,d)] += demands[t] * frac
    for (m,d) in selected:
        cap = float(Pmd.get((m,d), 0.0))
        ld  = load_md.get((m,d), 0.0)
        slack = cap - ld
        util = (ld / cap * 100.0) if cap > 0 else 0.0
        print(f"  - {m} @ {d}: load={fmt(ld,3)} / cap={fmt(cap,3)}  (util={fmt(util,1)}%, slack={fmt(slack,3)} req/s)")

    print("\n======================================================\n")


def run_once(policy: str, demands, Amin, redundancy=1, time_limit=60, log_to_console=True):
    """
    policy: 'multi' uses our_linear; 'single' uses our_singletenant
    """
    devices = config.devices
    models  = config.models
    tasks   = config.tasks

    latency_tmd = profiler.latency_tmd
    accuracy_tm = profiler.accuracy_tm
    vram_model  = profiler.model_memory
    vram_device = profiler.memory_device
    Pmd         = profiler.throughput_capacity

    if policy == "single":
        solver = solve_single
    elif policy == "multi":
        solver = solve_multi
    else:
        raise ValueError("policy must be 'multi' or 'single'")

    result = solver(
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
        redundancy=redundancy,
        minimize="deployments",
        time_limit=time_limit,
        log_to_console=log_to_console,
    )
    return result


def main():
    ap = argparse.ArgumentParser(description="Run ILP with multi-tenant or single-tenant solver.")
    ap.add_argument("--policy", choices=["multi", "single"], help="Choose tenancy policy to run once.")
    ap.add_argument("--both", action="store_true", help="Run both policies back-to-back for comparison.")
    ap.add_argument("--lambda", dest="lam", type=float, default=10.0, help="Demand (req/s) for the single task.")
    ap.add_argument("--amin", type=float, default=0.5, help="Accuracy threshold for the single task.")
    ap.add_argument("--redundancy", type=int, default=1, help="Redundancy K.")
    ap.add_argument("--time-limit", type=int, default=60, help="Gurobi time limit (s).")
    ap.add_argument("--quiet", action="store_true", help="Silence solver logs.")
    args = ap.parse_args()

    tasks = config.tasks
    # single-task assumption here; if you add more tasks later, this still works.
    demands = {t: float(args.lam) for t in tasks}
    Amin    = {t: float(args.amin) for t in tasks}

    print(f"[sanity] latency entries: {len(profiler.latency_tmd)}")

    if args.both:
        for policy in ("multi", "single"):
            res = run_once(
                policy=policy,
                demands=demands,
                Amin=Amin,
                redundancy=args.redundancy,
                time_limit=args.time_limit,
                log_to_console=not args.quiet
            )
            summarize_solution(
                tag=policy,
                result=res,
                devices=config.devices,
                demands=demands,
                Pmd=profiler.throughput_capacity,
                latency_tmd=profiler.latency_tmd,
                accuracy_tm=profiler.accuracy_tm,
                vram_model=profiler.model_memory,
                vram_device=profiler.memory_device
            )
    else:
        policy = args.policy or "multi"
        res = run_once(
            policy=policy,
            demands=demands,
            Amin=Amin,
            redundancy=args.redundancy,
            time_limit=args.time_limit,
            log_to_console=not args.quiet
        )
        summarize_solution(
            tag=policy,
            result=res,
            devices=config.devices,
            demands=demands,
            Pmd=profiler.throughput_capacity,
            latency_tmd=profiler.latency_tmd,
            accuracy_tm=profiler.accuracy_tm,
            vram_model=profiler.model_memory,
            vram_device=profiler.memory_device
        )


if __name__ == "__main__":
    main()