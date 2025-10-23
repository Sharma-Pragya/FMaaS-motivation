# ILP_new/minimal_exps.py
"""
Minimal run plan for the new ILP:
  1) Lambda sweep (demand per task)
  2) Headroom sweep (rho capacity margin)
  3) Policy ablation (multi-tenant vs single-tenant)
Outputs CSV + figures saved to ./figures/
"""

import os
import math
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

from ILP_new import config, profiler

# import both solvers
from ILP_new.methods.our_linear import build_and_solve as solve_multi
from ILP_new.methods.our_singletenant import build_and_solve as solve_single

os.environ.setdefault("GRB_LICENSE_FILE", "gurobi/gurobi.lic")


# ----------------- helpers -----------------
def devices_used_from_x(x):
    return {d for (m, d), v in x.items() if v > 0.5}


def kpis_from_solution(result, demands, Pmd):
    k = {}
    k["status"] = result.get("status")
    obj = result.get("obj")
    k["objective"] = float(obj) if obj is not None else math.nan

    x, r = result.get("x", {}), result.get("r", {})
    deployments = sum(1 for v in x.values() if v > 0.5)
    k["deployments"] = deployments
    k["devices_used"] = len(devices_used_from_x(x))

    load_md = defaultdict(float)
    for (t, m, d), frac in r.items():
        if frac > 1e-12:
            load_md[(m, d)] += demands[t] * frac

    k["total_routed_reqps"] = sum(load_md.values())
    k["total_demand_reqps"] = sum(demands.values())

    cap_selected = 0.0
    cap_all = 0.0
    peak_util = 0.0
    min_slack = float("inf")
    for (m, d), cap in Pmd.items():
        cap_all += cap
        if x.get((m, d), 0) > 0.5:
            cap_selected += cap
            ld = load_md.get((m, d), 0.0)
            util = ld / cap if cap > 0 else 0.0
            peak_util = max(peak_util, util)
            min_slack = min(min_slack, cap - ld)
    if deployments == 0:
        min_slack = float("nan")

    k.update(
        {
            "cap_selected_reqps": cap_selected,
            "cap_all_reqps": cap_all,
            "peak_util_selected": peak_util,
            "min_slack_selected": min_slack,
        }
    )
    return k


def scale_Pmd_by_rho(Pmd, rho):
    if rho is None:
        return Pmd
    return {k: rho * v for k, v in Pmd.items()}


# ----------------- experiment core -----------------
def run_once(demands, Amin, rho, policy, redundancy=1, time_limit=60):
    devices = config.devices
    models = config.models
    tasks = config.tasks

    latency_tmd = profiler.latency_tmd
    accuracy_tm = profiler.accuracy_tm
    vram_model = profiler.model_memory
    vram_device = profiler.memory_device
    Pmd_base = profiler.throughput_capacity
    Pmd = scale_Pmd_by_rho(Pmd_base, rho)

    solver = solve_single if policy == "single" else solve_multi

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
        log_to_console=False,
    )

    k = kpis_from_solution(result, demands, Pmd)
    k["policy"] = policy
    k["rho"] = rho if rho is not None else 1.0
    return result, k


def main():
    TASKS = list(config.tasks.keys())
    Amin = {t: 0.5 for t in TASKS}

    lambda_values = [1, 2, 5, 10, 15, 20]
    rhos = [1.0, 0.7, 0.5]
    policies = ["multi", "single"]

    all_rows = []
    for lam in lambda_values:
        demands = {t: float(lam) for t in TASKS}
        for rho in rhos:
            for policy in policies:
                _, k = run_once(demands=demands, Amin=Amin, rho=rho, policy=policy)
                k["lambda"] = lam
                all_rows.append(k)

    df = pd.DataFrame(all_rows)
    os.makedirs("figures", exist_ok=True)
    df.to_csv("figures/minimal_exps.csv", index=False)
    print("Saved results to figures/minimal_exps.csv")

    # ------------------ FIGURES ------------------
    plt.style.use("seaborn-v0_8")

    # 1. Lambda sweep
    fig, ax = plt.subplots()
    for policy in policies:
        subset = df[(df["policy"] == policy) & (df["rho"] == 1.0)]
        ax.plot(subset["lambda"], subset["deployments"], "o-", label=f"{policy} deployments")
    ax.set_xlabel("λ (requests per second)")
    ax.set_ylabel("Deployments")
    ax.set_title("Lambda Sweep — Deployments vs Demand")
    ax.legend()
    fig.savefig("figures/lambda_sweep_deployments.png", dpi=300)
    plt.close(fig)

    # 2. Lambda sweep utilization
    fig, ax = plt.subplots()
    for policy in policies:
        subset = df[(df["policy"] == policy) & (df["rho"] == 1.0)]
        ax.plot(subset["lambda"], subset["peak_util_selected"], "o-", label=f"{policy} peak util")
    ax.set_xlabel("λ (requests per second)")
    ax.set_ylabel("Peak Utilization")
    ax.set_title("Lambda Sweep — Utilization vs Demand")
    ax.legend()
    fig.savefig("figures/lambda_sweep_util.png", dpi=300)
    plt.close(fig)

    # 3. Headroom sweep (rho vs deployments)
    fig, ax = plt.subplots()
    for policy in policies:
        subset = df[(df["policy"] == policy) & (df["lambda"] == 10)]
        ax.plot(subset["rho"], subset["deployments"], "o-", label=f"{policy} deployments")
    ax.set_xlabel("ρ (headroom factor)")
    ax.set_ylabel("Deployments")
    ax.set_title("Headroom Sweep — Deployments vs ρ")
    ax.legend()
    fig.savefig("figures/headroom_sweep.png", dpi=300)
    plt.close(fig)

    # 4. Policy ablation summary (multi vs single)
    fig, ax = plt.subplots()
    width = 0.35
    lams = sorted(df["lambda"].unique())
    multi_vals = [df[(df["lambda"] == l) & (df["policy"] == "multi")]["deployments"].mean() for l in lams]
    single_vals = [df[(df["lambda"] == l) & (df["policy"] == "single")]["deployments"].mean() for l in lams]
    ax.bar([x - width / 2 for x in range(len(lams))], multi_vals, width, label="Multi-tenant")
    ax.bar([x + width / 2 for x in range(len(lams))], single_vals, width, label="Single-tenant")
    ax.set_xticks(range(len(lams)))
    ax.set_xticklabels(lams)
    ax.set_xlabel("λ (requests per second)")
    ax.set_ylabel("Deployments")
    ax.set_title("Policy Ablation — Multi vs Single Tenant")
    ax.legend()
    fig.savefig("figures/policy_ablation.png", dpi=300)
    plt.close(fig)

    print("Saved all plots to figures/")


if __name__ == "__main__":
    main()