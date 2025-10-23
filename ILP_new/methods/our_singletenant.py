# ILP_new/methods/our_singletenant.py
# Enforces one-device, one-model (single-tenant) policy:
#   ∑_m x[m,d] ≤ 1  for every device d
#
# API is aligned with our_linear.build_and_solve so you can swap easily.

from gurobipy import Model, GRB, quicksum

def build_and_solve(
    *,
    devices,            # dict: device_id -> {"type": <device_type>}
    models,             # list of model names
    tasks,              # dict: task -> Lmax_t (ms)
    demands,            # dict: task -> lambda_t (req/s)
    support,            # dict: (model, task) -> 1/0
    accuracy,           # dict: (task, model) -> A_{t,m}  (device-agnostic)
    latency,            # dict: (task, model, device) -> L_{t,m,d} (ms)
    vram_model,         # dict: model -> C_m (MB)
    vram_device,        # dict: device_type -> Mcap_d (MB)
    Pmd,                # dict: (model, device) -> capacity (req/s)
    Amin=None,          # dict: task -> A_min_t   (or None to ignore)
    redundancy=1,       # K homes per task (integer ≥1)
    minimize="deployments",  # "deployments" or "devices" (if you add device binaries)
    time_limit=60,
    log_to_console=True,
):
    """
    Decision vars
      x[m,d] ∈ {0,1}        deploy model m on device d  (SINGLE-TENANT enforced)
      u[t,m,d] ∈ {0,1}      task t has endpoint (m,d) as a "home" (counts redundancy)
      r[t,m,d] ∈ [0,1]      fraction of t's demand routed to (m,d)

    Constraints
      0) Feasibility mask: only create vars for triples (t,m,d) that pass support + SLO + (optional) accuracy
      1) Linking: u[t,m,d] ≤ x[m,d],  r[t,m,d] ≤ u[t,m,d]
      2) Redundancy: ∑_{m,d} u[t,m,d] ≥ K,   ∀ t
      3) Demand split: ∑_{m,d} r[t,m,d] = 1, ∀ t
      4) Endpoint capacity: ∑_t λ_t r[t,m,d] ≤ P_{m,d} x[m,d],  ∀ (m,d)
      5) Device VRAM: ∑_m C_m x[m,d] ≤ M_cap[type(d)],         ∀ d
      6) SINGLE-TENANT: ∑_m x[m,d] ≤ 1,                         ∀ d

    Objective
      minimize ∑_{m,d} x[m,d]
    """

    m = Model("tidy_linear_ilp_singletenant")
    if not log_to_console:
        m.Params.OutputFlag = 0
    if time_limit is not None:
        m.Params.TimeLimit = time_limit

    # ---------- Sets & helpers ----------
    D = list(devices.keys())
    M = list(models)
    T = list(tasks.keys())

    def feasible_triple(t, mm, d):
        # Support check
        if support.get((mm, t), 0) != 1:
            return False
        # Latency SLO
        Lmax = tasks[t]
        L = latency.get((t, mm, d), None)
        if L is None or L > Lmax:
            return False
        # Accuracy (optional)
        if Amin is not None and t in Amin:
            A = accuracy.get((t, mm), None)
            if A is None or A < Amin[t]:
                return False
        return True

    # Build the list of feasible (t,m,d)
    F = [(t, mm, d) for t in T for mm in M for d in D if feasible_triple(t, mm, d)]

    # Track any task with no feasible triple
    tasks_with_no_feasible_triple = [t for t in T if all((t != tt) for (tt, _, _) in F)]

    # ---------- Decision variables ----------
    # Deployments: x[m,d]
    x = {(mm, d): m.addVar(vtype=GRB.BINARY, name=f"x[{mm},{d}]") for mm in M for d in D}

    # Homes: u[t,m,d]  (only for feasible triples)
    u = {(t, mm, d): m.addVar(vtype=GRB.BINARY, name=f"u[{t},{mm},{d}]") for (t, mm, d) in F}

    # Routing fractions: r[t,m,d] ∈ [0,1]  (only for feasible triples)
    r = {(t, mm, d): m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"r[{t},{mm},{d}]") for (t, mm, d) in F}

    m.update()

    # ---------- Constraints ----------

    # (1) Linking: u ≤ x and r ≤ u (only feasible triples exist)
    for (t, mm, d) in F:
        m.addConstr(u[(t, mm, d)] <= x[(mm, d)], name=f"link_u_x[{t},{mm},{d}]")
        m.addConstr(r[(t, mm, d)] <= u[(t, mm, d)], name=f"link_r_u[{t},{mm},{d}]")

    # (2) Redundancy: Sum of homes ≥ K per task (but if no feasible triples, model will be infeasible)
    for t in T:
        m.addConstr(
            quicksum(u[(tt, mm, d)] for (tt, mm, d) in F if tt == t) >= max(redundancy, 1),
            name=f"redundancy[{t}]"
        )

    # (3) Demand split: sum r = 1 per task (route all demand of the single task)
    for t in T:
        m.addConstr(
            quicksum(r[(tt, mm, d)] for (tt, mm, d) in F if tt == t) == 1.0,
            name=f"demand_split[{t}]"
        )

    # (4) Endpoint capacity: sum_t λ_t r_{t,m,d} ≤ P_{m,d} x_{m,d}
    for mm in M:
        for d in D:
            Rt = quicksum(demands[t] * r[(t, mm, d)] for (t, mmm, dd) in F if mmm == mm and dd == d)
            cap = Pmd.get((mm, d), 0.0)
            m.addConstr(Rt <= cap * x[(mm, d)], name=f"capacity[{mm},{d}]")

    # (5) Device VRAM: ∑_m C_m x[m,d] ≤ M_cap(d)
    for d in D:
        dtype = devices[d]["type"]
        cap_mb = float(vram_device.get(dtype, 0.0))
        m.addConstr(
            quicksum(float(vram_model.get(mm, 0.0)) * x[(mm, d)] for mm in M) <= cap_mb,
            name=f"vram[{d}]"
        )

    # (6) SINGLE-TENANT: at most one model per device
    for d in D:
        m.addConstr(
            quicksum(x[(mm, d)] for mm in M) <= 1,
            name=f"single_tenant[{d}]"
        )

    # ---------- Objective ----------
    if minimize == "deployments":
        obj = quicksum(x[(mm, d)] for mm in M for d in D)
    else:
        # default to deployments if other modes aren’t provided
        obj = quicksum(x[(mm, d)] for mm in M for d in D)

    m.setObjective(obj, GRB.MINIMIZE)

    # ---------- Solve ----------
    m.optimize()

    # ---------- Extract solution ----------
    status_code = m.Status
    status = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
    }.get(status_code, f"STATUS_{status_code}")

    result = {
        "status": status,
        "obj": None,
        "x": {},
        "u": {},
        "r": {},
        "used_devices": {},
        "tasks_with_no_feasible_triple": tasks_with_no_feasible_triple,
    }

    if status in ("OPTIMAL", "TIME_LIMIT"):
        try:
            result["obj"] = m.ObjVal
        except Exception:
            pass

        # binaries
        for (mm, d), var in x.items():
            val = var.X if var.X is not None else 0.0
            result["x"][(mm, d)] = int(round(val))

        for (t, mm, d), var in u.items():
            val = var.X if var.X is not None else 0.0
            result["u"][(t, mm, d)] = int(round(val))

        # routing
        for (t, mm, d), var in r.items():
            val = var.X if var.X is not None else 0.0
            result["r"][(t, mm, d)] = float(val)

        # used devices
        used = {}
        for d in D:
            used[d] = 1 if any(result["x"][(mm, d)] > 0.5 for mm in M) else 0
        result["used_devices"] = used

    elif status == "INFEASIBLE":
        try:
            m.computeIIS()
            m.write("singletenant_ilp.iis")
        except Exception:
            pass

    return result