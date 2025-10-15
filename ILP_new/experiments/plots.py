# ILP_new/plots_minimal.py
"""
Plots for the minimal run plan:
  - A1: Deployments vs lambda
  - A2: Devices used vs lambda
  - A3: Peak utilization vs lambda
  - A4: Routing composition at selected lambdas (stacked bars)
  - B1: lambda_max vs rho
  - B2: Deployments at lambda_max(rho)
  - C1: Delta deployments vs lambda (single - multi)
  - C2: Per-device VRAM packing at representative lambdas

This script reuses the same sweep grid as experiments_minimal.py,
re-runs the solver (so results are consistent), and saves PNGs.

Usage:
  cd ILP_new
  python plots_minimal.py
"""

import os
import math
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt
import pandas as pd

from ILP_new.experiments.minimal_exps import run_once   # uses your build_and_solve
from ILP_new import config, profiler

# Ensure Gurobi license path (adjust if needed)
os.environ.setdefault("GRB_LICENSE_FILE", "gurobi/gurobi.lic")

# ------------------------ sweep configuration ------------------------

# One-task setup:
TASKS = list(config.tasks.keys())
Amin  = {t: 0.5 for t in TASKS}  # keep consistent with your experiments_minimal

lambda_values = [10, 20, 30, 40, 50, ]
rhos          = [1.0, 0.8, 0.7, 0.5]
policies      = ["multi", "single"]  # tenancy ablation

# For routing/packing visuals we’ll pick a few representative lambdas:
routing_lambdas = [5, 10, 20]

# ------------------------ run grid & collect KPIs ------------------------

rows = []
results_cache = {}  # (lam, rho, policy) -> (result, KPIs)

for lam in lambda_values:
    demands = {t: float(lam) for t in TASKS}
    for rho in rhos:
        for policy in policies:
            result, k = run_once(demands=demands, Amin=Amin, rho=rho, policy=policy)
            k["lambda"] = lam
            rows.append(k)
            results_cache[(lam, rho, policy)] = (result, k)

df = pd.DataFrame(rows)

# Save the raw KPIs (nice to have for the paper appendix)
df.to_csv("plots_minimal_kpis.csv", index=False)

# ------------------------ helper: per-endpoint routing table ------------------------

def routing_table(result, demands, Pmd, latency_tmd, accuracy_tm):
    """Return a DataFrame with routing and capacity per (m,d) for the single task."""
    r = result.get("r", {})
    x = result.get("x", {})

    # aggregate load per endpoint
    load_md = defaultdict(float)
    for (t, m, d), frac in r.items():
        if frac <= 1e-12:
            continue
        load_md[(m, d)] += demands[t] * frac

    rows = []
    for (m, d), load in load_md.items():
        cap = float(Pmd.get((m, d), 0.0))
        util = load / cap if cap > 0 else 0.0

        # pick the single task key (we only have one task here)
        t = TASKS[0]
        lat = latency_tmd.get((t, m, d), None)
        acc = accuracy_tm.get((t, m), None)

        rows.append(OrderedDict([
            ("model", m),
            ("device", d),
            ("load_reqps", load),
            ("cap_reqps", cap),
            ("util", util),
            ("latency_ms", lat),
            ("accuracy", acc),
            ("selected", 1 if x.get((m, d), 0) > 0.5 else 0),
        ]))
    return pd.DataFrame(rows).sort_values("load_reqps", ascending=False)

# ------------------------ A1: Deployments vs lambda ------------------------

for rho in rhos:
    plt.figure()
    for policy in policies:
        sub = df[(df["rho"] == rho) & (df["policy"] == policy)].sort_values("lambda")
        plt.plot(sub["lambda"], sub["deployments"], marker="o", label=policy)
    plt.xlabel("lambda (req/s)")
    plt.ylabel("deployments")
    plt.title(f"Deployments vs lambda (rho={rho})")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(f"fig_A1_deployments_vs_lambda_rho{str(rho).replace('.','')}.png", dpi=200)

# ------------------------ A2: Devices used vs lambda ------------------------

for rho in rhos:
    plt.figure()
    for policy in policies:
        sub = df[(df["rho"] == rho) & (df["policy"] == policy)].sort_values("lambda")
        plt.plot(sub["lambda"], sub["devices_used"], marker="o", label=policy)
    plt.xlabel("lambda (req/s)")
    plt.ylabel("devices used")
    plt.title(f"Devices used vs lambda (rho={rho})")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(f"fig_A2_devices_vs_lambda_rho{str(rho).replace('.','')}.png", dpi=200)

# ------------------------ A3: Peak utilization vs lambda ------------------------

for rho in rhos:
    plt.figure()
    for policy in policies:
        sub = df[(df["rho"] == rho) & (df["policy"] == policy)].sort_values("lambda")
        plt.plot(sub["lambda"], sub["peak_util_selected"], marker="o", label=policy)
    plt.xlabel("lambda (req/s)")
    plt.ylabel("peak endpoint utilization")
    plt.title(f"Peak utilization vs lambda (rho={rho})")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(f"fig_A3_peak_util_vs_lambda_rho{str(rho).replace('.','')}.png", dpi=200)

# ------------------------ A4: Routing composition (stacked bars) ------------------------

# Use the multi-tenant policy and rho=1.0 by default for readability
default_policy = "multi"
default_rho = 1.0

for lam in routing_lambdas:
    key = (lam, default_rho, default_policy)
    if key not in results_cache:
        continue
    result, _k = results_cache[key]
    Pmd = profiler.throughput_capacity if default_rho == 1.0 else {k: default_rho*v for k, v in profiler.throughput_capacity.items()}
    demands = {t: float(lam) for t in TASKS}
    rt = routing_table(
        result=result,
        demands=demands,
        Pmd=Pmd,
        latency_tmd=profiler.latency_tmd,
        accuracy_tm=profiler.accuracy_tm
    )

    # Keep top-K endpoints by load, group the rest as "other"
    K = 6
    rt_top = rt.head(K).copy()
    other_load = rt["load_reqps"][K:].sum()
    if other_load > 0:
        rt_top.loc[len(rt_top)] = {
            "model": "other",
            "device": "",
            "load_reqps": other_load,
            "cap_reqps": float('nan'),
            "util": float('nan'),
            "latency_ms": float('nan'),
            "accuracy": float('nan'),
            "selected": 1,
        }

    plt.figure()
    plt.bar(rt_top["model"], rt_top["load_reqps"])
    plt.xlabel("endpoint (model)")
    plt.ylabel("load (req/s)")
    plt.title(f"Routing composition at lambda={lam} (policy={default_policy}, rho={default_rho})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"fig_A4_routing_lambda{lam}.png", dpi=200)

# ------------------------ B1: lambda_max vs rho ------------------------

# For each rho & policy, find the largest feasible lambda in our grid
rows_b1 = []
for rho in rhos:
    for policy in policies:
        feasible_lams = df[(df["rho"] == rho) & (df["policy"] == policy) & (df["status"] == "OPTIMAL")]["lambda"]
        lam_max = feasible_lams.max() if not feasible_lams.empty else float("nan")
        rows_b1.append({"rho": rho, "policy": policy, "lambda_max": lam_max})

b1 = pd.DataFrame(rows_b1)

plt.figure()
for policy in policies:
    sub = b1[b1["policy"] == policy].sort_values("rho")
    plt.plot(sub["rho"], sub["lambda_max"], marker="o", label=policy)
plt.xlabel("rho (headroom factor)")
plt.ylabel("max feasible lambda")
plt.title("lambda_max vs rho")
plt.legend()
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("fig_B1_lambda_max_vs_rho.png", dpi=200)

# ------------------------ B2: Deployments at lambda_max(rho) ------------------------

plt.figure()
x = []
height_multi = []
height_single = []

for rho in rhos:
    # Multi
    sub_m = df[(df["rho"] == rho) & (df["policy"] == "multi") & (df["status"] == "OPTIMAL")]
    lam_m = sub_m["lambda"].max() if not sub_m.empty else float("nan")
    dep_m = sub_m[sub_m["lambda"] == lam_m]["deployments"].max() if not sub_m.empty else float("nan")

    # Single
    sub_s = df[(df["rho"] == rho) & (df["policy"] == "single") & (df["status"] == "OPTIMAL")]
    lam_s = sub_s["lambda"].max() if not sub_s.empty else float("nan")
    dep_s = sub_s[sub_s["lambda"] == lam_s]["deployments"].max() if not sub_s.empty else float("nan")

    x.append(str(rho))
    height_multi.append(dep_m)
    height_single.append(dep_s)

bar_w = 0.35
xs = range(len(x))
plt.bar([i - bar_w/2 for i in xs], height_multi, width=bar_w, label="multi")
plt.bar([i + bar_w/2 for i in xs], height_single, width=bar_w, label="single")
plt.xticks(list(xs), x)
plt.xlabel("rho")
plt.ylabel("deployments at lambda_max(rho)")
plt.title("Deployments at the feasibility frontier")
plt.legend()
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("fig_B2_deployments_at_lambda_max.png", dpi=200)

# ------------------------ C1: Delta deployments vs lambda (single - multi) ------------------------

plt.figure()
for rho in rhos:
    sub_m = df[(df["rho"] == rho) & (df["policy"] == "multi")].sort_values("lambda")
    sub_s = df[(df["rho"] == rho) & (df["policy"] == "single")].sort_values("lambda")
    merged = pd.merge(sub_s[["lambda","deployments"]],
                      sub_m[["lambda","deployments"]],
                      on="lambda", suffixes=("_single","_multi"))
    merged["delta"] = merged["deployments_single"] - merged["deployments_multi"]
    plt.plot(merged["lambda"], merged["delta"], marker="o", label=f"rho={rho}")
plt.xlabel("lambda (req/s)")
plt.ylabel("delta deployments (single - multi)")
plt.title("Benefit of multi-tenant vs single-tenant")
plt.legend()
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("fig_C1_delta_deployments_vs_lambda.png", dpi=200)

# ------------------------ C2: Per-device VRAM packing (at selected lambdas) ------------------------

def vram_packing_table(result, vram_model, vram_device, devices):
    x = result.get("x", {})
    # Accumulate VRAM per device id (note: device ids carry 'type' in config.devices[d]['type'])
    usage = defaultdict(float)
    for (m, d), val in x.items():
        if val > 0.5:
            usage[d] += float(vram_model.get(m, 0.0))
    rows = []
    for d in devices:
        dtype = devices[d]["type"]
        rows.append({
            "device": d,
            "device_type": dtype,
            "used_mb": usage.get(d, 0.0),
            "cap_mb": float(vram_device.get(dtype, 0.0))
        })
    return pd.DataFrame(rows)

devices = config.devices
vram_model = profiler.model_memory
vram_device = profiler.memory_device

for lam in routing_lambdas:
    fig, ax = plt.subplots()
    for policy in policies:
        key = (lam, 1.0, policy)
        if key not in results_cache: 
            continue
        result, _k = results_cache[key]
        pack = vram_packing_table(result, vram_model, vram_device, devices)
        # bar positions offset by policy
        xs = range(len(pack))
        if policy == "multi":
            pos = [i - 0.15 for i in xs]
        else:
            pos = [i + 0.15 for i in xs]
        ax.bar(pos, pack["used_mb"], width=0.3, label=policy)
    ax.set_xticks(list(range(len(devices))))
    ax.set_xticklabels(list(devices.keys()), rotation=45, ha="right")
    ax.set_xlabel("device id")
    ax.set_ylabel("VRAM used (MB)")
    ax.set_title(f"Per-device VRAM packing at lambda={lam}")
    ax.legend()
    ax.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(f"fig_C2_vram_packing_lambda{lam}.png", dpi=200)

print("Saved figures:")
for f in [
    *[f"fig_A1_deployments_vs_lambda_rho{str(r).replace('.','')}.png" for r in rhos],
    *[f"fig_A2_devices_vs_lambda_rho{str(r).replace('.','')}.png" for r in rhos],
    *[f"fig_A3_peak_util_vs_lambda_rho{str(r).replace('.','')}.png" for r in rhos],
    *[f"fig_A4_routing_lambda{lam}.png" for lam in routing_lambdas],
    "fig_B1_lambda_max_vs_rho.png",
    "fig_B2_deployments_at_lambda_max.png",
    "fig_C1_delta_deployments_vs_lambda.png",
    *[f"fig_C2_vram_packing_lambda{lam}.png" for lam in routing_lambdas],
]:
    print(" -", f)