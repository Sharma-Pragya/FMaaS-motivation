# ILP_new/experiments_minimal.py
"""
Minimal run plan for the new ILP:
  1) Lambda sweep (demand per task)
  2) Headroom sweep (rho capacity margin)
  3) Policy ablation (multi-tenant vs single-tenant)

Outputs a tidy CSV table to stdout with one row per run.
"""

from ILP_new.methods.our_linear import build_and_solve
from ILP_new import config, profiler
from collections import defaultdict
import os
import math

# Ensure gurobi license path (adapt if yours differs)
os.environ.setdefault("GRB_LICENSE_FILE", "gurobi/gurobi.lic")

# --- helpers -----------------------------------------------------------------

def devices_used_from_x(x):
    used = set()
    for (m,d), v in x.items():
        if v > 0.5:
            used.add(d)
    return used

def kpis_from_solution(result, demands, Pmd, latency_tmd, accuracy_tm, vram_model):
    """Compute compact KPIs from the solver result dict."""
    kpis = {}
    kpis["status"] = result.get("status")
    obj = result.get("obj")
    if obj is not None:
        kpis["objective"] = float(obj)
    else:
        kpis["objective"] = math.nan

    x = result.get("x", {})
    r = result.get("r", {})

    # deployments & devices
    deployments = sum(1 for v in x.values() if v > 0.5)
    kpis["deployments"] = deployments
    used_devs = devices_used_from_x(x)
    kpis["devices_used"] = len(used_devs)

    # summarize capacity & load on selected endpoints
    load_md = defaultdict(float)
    for (t,m,d), frac in r.items():
        if frac <= 1e-12:
            continue
        load_md[(m,d)] += demands[t] * frac

    total_routed = sum(load_md.values())
    kpis["total_routed_reqps"] = total_routed

    total_demand = sum(demands.values())
    kpis["total_demand_reqps"] = total_demand

    # capacities (selected & all)
    cap_selected = 0.0
    cap_all = 0.0
    peak_util = 0.0
    min_slack = float("inf")
    for (m,d), cap in Pmd.items():
        cap_all += cap
        if x.get((m,d), 0) > 0.5:
            cap_selected += cap
            ld = load_md.get((m,d), 0.0)
            util = ld / cap if cap > 0 else 0.0
            peak_util = max(peak_util, util)
            min_slack = min(min_slack, cap - ld)
    if deployments == 0:
        min_slack = float("nan")

    kpis["cap_selected_reqps"] = cap_selected
    kpis["cap_all_reqps"] = cap_all
    kpis["peak_util_selected"] = peak_util
    kpis["min_slack_selected"] = min_slack

    return kpis

def vram_caps_single_tenant(vram_device, vram_model, devices):
    """
    Enforce single-tenant (≤1 model per device) by capping device VRAM to the
    maximum single-model VRAM footprint. Then ∑ C_m x_{m,d} ≤ cap ⇒ at most one model fits.
    """
    if len(vram_model) == 0:
        return vram_device  # fallback

    max_model_mem = max(vram_model.values())
    # Slightly less than (2 * smallest model) isn't robust; the hard cap at max_model_mem does the job.
    # Build a per-type map, then per-id cap is looked up by its type.
    single_caps_by_type = {}
    for d, meta in devices.items():
        dtype = meta["type"]
        single_caps_by_type[dtype] = max_model_mem
    return single_caps_by_type

def scale_Pmd_by_rho(Pmd, rho):
    """Apply headroom factor rho (e.g., 0.7) to every endpoint capacity."""
    if rho is None:
        return Pmd
    return {k: rho * v for k, v in Pmd.items()}

# --- experiment driver --------------------------------------------------------

def run_once(demands, Amin, rho, policy, redundancy=1, time_limit=60):
    """
    policy: 'multi' or 'single' (tenancy policy)
    rho: headroom factor in (0,1], scales all P_{m,d}
    """
    devices = config.devices
    models = config.models
    tasks  = config.tasks

    # Base inputs
    latency_tmd = profiler.latency_tmd
    accuracy_tm = profiler.accuracy_tm
    vram_model  = profiler.model_memory
    vram_device = profiler.memory_device
    Pmd_base    = profiler.throughput_capacity

    # Apply headroom
    Pmd = scale_Pmd_by_rho(Pmd_base, rho)

    # Apply tenancy policy
    if policy == "single":
        # Replace device caps with single-tenant caps
        vram_device_eff = vram_caps_single_tenant(vram_device, vram_model, devices)
    else:
        vram_device_eff = vram_device

    # Solve
    result = build_and_solve(
        devices=devices,
        models=models,
        tasks=tasks,
        demands=demands,
        support=profiler.can_serve,
        accuracy=accuracy_tm,
        latency=latency_tmd,
        vram_model=vram_model,
        vram_device=vram_device_eff,
        Pmd=Pmd,
        Amin=Amin,
        redundancy=redundancy,
        minimize="deployments",
        time_limit=time_limit,
    )

    # KPIs
    k = kpis_from_solution(result, demands, Pmd, latency_tmd, accuracy_tm, vram_model)
    k["policy"] = policy
    k["rho"] = rho if rho is not None else 1.0
    return result, k

def main():
    # One task; use your configured SLO/accuracy thresholds
    # Keep a realistic Amin; or set to None to disable accuracy gating.
    tasks = list(config.tasks.keys())
    Amin  = {t: 0.5 for t in tasks}  # adjust if you want stricter thresholds

    # Sweeps
    lambda_values = [1, 2, 5, 10, 15, 20]     # demand (req/s) sweep
    rhos          = [1.0, 0.7, 0.5]           # headroom sweep
    policies      = ["multi", "single"]       # policy ablation

    # CSV header
    header = [
        "lambda", "rho", "policy", "status", "objective",
        "deployments", "devices_used",
        "total_demand_reqps", "total_routed_reqps",
        "cap_selected_reqps", "cap_all_reqps",
        "peak_util_selected", "min_slack_selected",
    ]
    print(",".join(header))

    # Run grid
    for lam in lambda_values:
        demands = {t: float(lam) for t in tasks}
        for rho in rhos:
            for policy in policies:
                _, k = run_once(demands=demands, Amin=Amin, rho=rho, policy=policy)
                # print CSV row
                row = [
                    lam,
                    k.get("rho", ""),
                    k.get("policy", ""),
                    k.get("status", ""),
                    k.get("objective", ""),
                    k.get("deployments", ""),
                    k.get("devices_used", ""),
                    k.get("total_demand_reqps", ""),
                    k.get("total_routed_reqps", ""),
                    k.get("cap_selected_reqps", ""),
                    k.get("cap_all_reqps", ""),
                    k.get("peak_util_selected", ""),
                    k.get("min_slack_selected", ""),
                ]
                # format nicely
                row_fmt = []
                for v in row:
                    if isinstance(v, float):
                        row_fmt.append(f"{v:.6g}")
                    else:
                        row_fmt.append(str(v))
                print(",".join(row_fmt))

if __name__ == "__main__":
    main()