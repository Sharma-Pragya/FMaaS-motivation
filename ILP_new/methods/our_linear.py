# methods/our_linear.py
# A linear Gurobi ILP.
# - Deploy endpoints (m,d)
# - Count redundancy "homes" per task
# - Split load fractions across endpoints
# - Enforce endpoint capacity, device VRAM, and SLO/accuracy via masking
#
# Usage (example):
#   from ILP.methods.our_linear import build_and_solve
#   sol = build_and_solve(devices, models, tasks, demands, support, accuracy, latency, vram_model, vram_device, Pmd,
#                         redundancy=1, minimize="deployments", time_limit=60, log_to_console=True)

from typing import Dict, Iterable, Tuple, List, Optional
from gurobipy import Model, GRB, quicksum

def build_and_solve(
    devices: Dict[str, Dict],               # e.g., {"d1": {"type":"A16"}, ...}
    models: Iterable[str],                  # e.g., ["llama3","gpt4o-mini",...]
    tasks: Dict[str, float],                # task -> Lmax (ms). If you already mask on latency externally, this can be {}.
    demands: Dict[str, float],              # task -> lambda_t (req/s)
    support: Dict[Tuple[str,str], int],     # (m,t) -> 0/1  (B_{m,t})
    accuracy: Dict[Tuple[str,str], float],  # (t,m) -> A_{t,m}  (set high value if you don't use accuracy)
    latency: Dict[Tuple[str,str,str], float],# (t,m,d) -> L_{t,m,d} (ms)
    vram_model: Dict[str, float],           # m -> C_m
    vram_device: Dict[str, float],          # device_type -> M_d^cap
    Pmd: Dict[Tuple[str,str], float],       # (m,d) -> SLO-feasible sustained throughput (req/s) for endpoint (m,d)
    Amin: Optional[Dict[str, float]] = None,# task -> A^{min}_t (if None, accuracy not checked)
    redundancy: int = 1,                    # K: min #homes per task
    minimize: str = "deployments",          # "deployments" or "devices"
    time_limit: Optional[int] = None,
    log_to_console: bool = True,
):
    """
    Returns a dict with model/solution info:
      {
        "status": str,
        "obj": float|None,
        "x": {(m,d): 0/1, ...},
        "u": {(t,m,d): 0/1, ...},
        "r": {(t,m,d): float in [0,1], ...},
        "used_devices": {d: 0/1, ...}   # only when minimize == "devices"
      }
    """

    # ---------- 0) Basic sets ----------
    D = list(devices.keys())
    M = list(models)
    T = list(demands.keys())  # ensure we optimize for tasks that have demand (req/s)

    def dev_type(d: str) -> str:
        return devices[d]["type"]

    # ---------- 1) Mask infeasible (t,m,d) triples ----------
    # A triple is feasible iff:
    #   - support[(m,t)] == 1
    #   - (accuracy[t,m] >= Amin[t]) if Amin is provided
    #   - latency[(t,m,d)] <= Lmax[t] if tasks contains an Lmax for t
    feasible = {}
    for t in T:
        Lmax_t = tasks.get(t, float("inf"))  # if tasks dict lacks t, no latency filtering
        for m in M:
            if support.get((m, t), 0) != 1:
                for d in D: feasible[(t,m,d)] = False
                continue
            # Accuracy check (optional)
            if Amin is not None:
                if accuracy.get((t,m), float("-inf")) < Amin.get(t, float("-inf")):
                    for d in D: feasible[(t,m,d)] = False
                    continue
            # Latency check (if provided)
            for d in D:
                L_tmd = latency.get((t,m,d), float("inf"))
                feasible[(t,m,d)] = (L_tmd <= Lmax_t)

    # Quick infeasibility probe: each t should have at least one feasible (m,d)
    t_with_no_feasible = [t for t in T if not any(feasible[(t,m,d)] for m in M for d in D)]

    # ---------- 2) Build model ----------
    m = Model("tidy_linear_ilp")
    if not log_to_console:
        m.Params.OutputFlag = 0
    if time_limit is not None:
        m.Params.TimeLimit = time_limit

    # ---------- 3) Decision variables ----------
    # x_{m,d} ∈ {0,1} : deploy endpoint (m,d)
    x = m.addVars([(mm, dd) for mm in M for dd in D], vtype=GRB.BINARY, name="x")

    # u_{t,m,d} ∈ {0,1} : (t) has (m,d) as a "home" (counts toward redundancy)
    u_keys = [(tt, mm, dd) for tt in T for mm in M for dd in D if feasible[(tt,mm,dd)]]
    u = m.addVars(u_keys, vtype=GRB.BINARY, name="u")

    # r_{t,m,d} ∈ [0,1] : fraction of t's demand routed to (m,d)
    r = m.addVars(u_keys, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="r")

    # Optional: w_d if minimizing number of devices
    w = {}
    if minimize == "devices":
        w = m.addVars(D, vtype=GRB.BINARY, name="w")

    # ---------- 4) Objective ----------
    if minimize == "deployments":
        # Minimize total endpoints deployed
        m.setObjective(quicksum(x[mm,dd] for mm in M for dd in D), GRB.MINIMIZE)
    elif minimize == "devices":
        # Minimize number of used devices
        # Need linking x_{m,d} ≤ w_d (see constraint group G below)
        m.setObjective(quicksum(w[dd] for dd in D), GRB.MINIMIZE)
    else:
        raise ValueError("minimize must be 'deployments' or 'devices'")

    # ---------- 5) Constraints ----------

    # (A) Device VRAM budget:
    #     sum_m C_m * x_{m,d} <= M_d^cap   ∀ d
    #     (Multi-tenant by memory; relaxes "one model per device" by policy.)
    for dd in D:
        m.addConstr(
            quicksum(vram_model[mm] * x[mm,dd] for mm in M) <= vram_device[dev_type(dd)],
            name=f"A_VRAM_{dd}"
        )

    # (B) Linking: a "home" u_{t,m,d} can only exist if endpoint (m,d) is deployed
    #     u_{t,m,d} <= x_{m,d}      ∀ feasible (t,m,d)
    m.addConstrs((u[tt,mm,dd] <= x[mm,dd] for (tt,mm,dd) in u_keys), name="B_u_le_x")

    # (C) Routing only to declared homes:
    #     r_{t,m,d} <= u_{t,m,d}    ∀ feasible (t,m,d)
    m.addConstrs((r[tt,mm,dd] <= u[tt,mm,dd] for (tt,mm,dd) in u_keys), name="C_r_le_u")

    # (D) Redundancy floor per task:
    #     sum_{m,d feasible} u_{t,m,d} >= K      ∀ t
    for tt in T:
        m.addConstr(
            quicksum(u[tt,mm,dd] for mm in M for dd in D if feasible[(tt,mm,dd)]) >= redundancy,
            name=f"D_redundancy_{tt}"
        )

    # (E) Demand split (all demand must be allocated):
    #     sum_{m,d feasible} r_{t,m,d} = 1       ∀ t
    for tt in T:
        m.addConstr(
            quicksum(r[tt,mm,dd] for mm in M for dd in D if feasible[(tt,mm,dd)]) == 1.0,
            name=f"E_split_{tt}"
        )

    # (F) Endpoint capacity:
    #     sum_t lambda_t * r_{t,m,d} <= P_{m,d} * x_{m,d}     ∀ (m,d)
    #     (Interpretation: SLO-feasible sustained throughput limit of endpoint (m,d).)
    for mm in M:
        for dd in D:
            lhs = quicksum(demands[tt] * r[tt,mm,dd] for tt in T if feasible[(tt,mm,dd)])
            cap = Pmd.get((mm,dd), 0.0)
            m.addConstr(lhs <= cap * x[mm,dd], name=f"F_cap_{mm}_{dd}")

    # (G) Device usage if minimizing devices:
    #     x_{m,d} <= w_d     ∀ (m,d)    (activating an endpoint implies the device is "on")
    if minimize == "devices":
        for dd in D:
            for mm in M:
                m.addConstr(x[mm,dd] <= w[dd], name=f"G_use_dev_{mm}_{dd}")

    # ---------- 6) Solve ----------
    # This model is linear (no bilinear / big-M rows needed), so no NonConvex param.
    m.optimize()

    # ---------- 7) Extract solution ----------
    status = m.Status
    status_str = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
    }.get(status, str(status))

    sol = {
        "status": status_str,
        "obj": None if m.SolCount == 0 else m.objVal,
        "x": {},
        "u": {},
        "r": {},
        "used_devices": {},
        "tasks_with_no_feasible_triple": t_with_no_feasible,
    }

    if m.SolCount > 0:
        sol["x"] = {(mm,dd): int(round(x[mm,dd].X)) for mm in M for dd in D}
        sol["u"] = {(tt,mm,dd): int(round(u[tt,mm,dd].X)) for (tt,mm,dd) in u_keys}
        sol["r"] = {(tt,mm,dd): r[tt,mm,dd].X for (tt,mm,dd) in u_keys}
        if minimize == "devices":
            sol["used_devices"] = {dd: int(round(w[dd].X)) for dd in D}

    return sol