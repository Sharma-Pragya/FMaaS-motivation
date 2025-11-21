# ILP_new/methods/our_multitenant.py
# Multi-tenant ILP (many models per device) aligned with the current non-sharing ILP:
#   - SLO pre-filtering over (t, m, d)
#   - Device-level compute capacity with task-specific throughput P_{t,m,d}
#   - Device memory capacity
#   - Multiple objective modes:
#       * "deployments"
#       * "deployments_devices"
#       * "deployments_devices_waste"
#       * "deployments_devices_waste_modelsize"
#
# This is the multi-tenant specialization of the non-sharing ILP in the paper.

from gurobipy import Model, GRB, quicksum


def build_and_solve_from_pipelines(
    *,
    components,         # dict: component_name -> {'mem': MB}
    pipelines,          # dict: pid -> {'backbone': ..., 'decoder': ..., 'task': ...}
    latency_config,     # dict: pid -> {device_type: latency_ms}
    metric_config,      # dict: pid -> accuracy/metric value
    device_type="A6000",
    device_name="gpu0",
    vram_device_cap_mb=48000.0,
    task_latency_slo_ms=None,   # dict: task -> L_max_ms; if None, derive as 1.5x max
    task_accuracy_slo=None,     # dict: task -> A_min; if None, ignore accuracy
    task_demands=None,          # dict: task -> lambda (req/s); if None, default 1.0
    minimize="deployments",
    time_limit=60,
    log_to_console=True,
):
    """
    Wrapper that accepts VLM/TSFM pipeline format and converts to ILP format.
    Each pipeline becomes a "model" in the ILP (no architectural sharing here).

    This calls build_and_solve() with converted inputs.
    """

    # 1. Models = pipeline IDs
    models = sorted(pipelines.keys())

    # 2. Collect unique tasks
    task_set = {info["task"] for info in pipelines.values()}
    tasks_list = sorted(task_set)

    # 3. Devices
    devices = {device_name: {"type": device_type}}
    vram_device = {device_type: float(vram_device_cap_mb)}

    # 4. Tasks dict (per-task latency SLOs)
    tasks = {}
    if task_latency_slo_ms is not None:
        for t in tasks_list:
            tasks[t] = float(task_latency_slo_ms.get(t, 1e9))
    else:
        # Derive per-task SLO as 1.5x max observed latency on that task
        task_lat_values = {t: [] for t in tasks_list}
        for pid, info in pipelines.items():
            t = info["task"]
            if pid in latency_config and device_type in latency_config[pid]:
                task_lat_values[t].append(latency_config[pid][device_type])
        for t in tasks_list:
            if task_lat_values[t]:
                tasks[t] = 1.5 * max(task_lat_values[t])
            else:
                tasks[t] = 1e9

    # 5. Demands
    demands = {}
    if task_demands is not None:
        for t in tasks_list:
            demands[t] = float(task_demands.get(t, 1.0))
    else:
        for t in tasks_list:
            demands[t] = 1.0

    # 6. Support: each pipeline serves exactly its own task
    support = {}
    for pid in models:
        pipeline_task = pipelines[pid]["task"]
        for t in tasks_list:
            support[(pid, t)] = 1 if t == pipeline_task else 0

    # 7. Accuracy: (task, pipeline) -> metric
    accuracy = {}
    for pid, info in pipelines.items():
        t = info["task"]
        accuracy[(t, pid)] = float(metric_config.get(pid, 0.0))

    # 8. Amin: per-task minimum accuracy
    if task_accuracy_slo is not None:
        Amin = {t: float(task_accuracy_slo[t]) for t in tasks_list}
    else:
        Amin = None

    # 9. Latency: (task, pipeline, device) -> ms
    latency_ilp = {}
    for pid, info in pipelines.items():
        t = info["task"]
        if pid in latency_config and device_type in latency_config[pid]:
            latency_ilp[(t, pid, device_name)] = float(latency_config[pid][device_type])

    # 10. Throughput: Ptmd = 1000 / latency_ms (batch size 1 approximation)
    Ptmd = {}
    for key, L_ms in latency_ilp.items():
        if L_ms > 0:
            Ptmd[key] = 1000.0 / L_ms

    # 11. vram_model: full pipeline memory (backbone + decoder + task component)
    vram_model = {}
    for pid, info in pipelines.items():
        backbone = info["backbone"]
        decoder = info["decoder"]
        task = info["task"]

        # Backbone memory
        bb_mem = components.get(backbone, {}).get("mem", 0.0)

        # Decoder memory - allow for multiple naming patterns
        dec_mem = 0.0
        # TSFM-like: decoder_backbone_task
        dec_key1 = f"{decoder}_{backbone}_{task}"
        # Direct decoder entry
        dec_key2 = decoder

        if dec_key1 in components:
            dec_mem = components[dec_key1].get("mem", 0.0)
        elif dec_key2 in components:
            dec_mem = components[dec_key2].get("mem", 0.0)

        # Task component memory in VLM-style: task_backbone_decoder
        task_key = f"{task}_{backbone}_{decoder}"
        task_mem = components.get(task_key, {}).get("mem", 0.0)

        vram_model[pid] = float(bb_mem + dec_mem + task_mem)

    # Call the core ILP builder
    return build_and_solve(
        devices=devices,
        models=models,
        tasks=tasks,
        demands=demands,
        support=support,
        accuracy=accuracy,
        latency=latency_ilp,
        vram_model=vram_model,
        vram_device=vram_device,
        Ptmd=Ptmd,
        Amin=Amin,
        minimize=minimize,
        time_limit=time_limit,
        log_to_console=log_to_console,
    )


def build_and_solve(
    *,
    devices,            # dict: device_id -> {"type": <device_type>}
    models,             # list of model names
    tasks,              # dict: task -> Lmax_t (ms)
    demands,            # dict: task -> lambda_t (req/s)
    support,            # dict: (model, task) -> 1/0
    accuracy,           # dict: (task, model) -> A_{t,m} (device-agnostic)
    latency,            # dict: (task, model, device) -> L_{t,m,d} (ms)
    vram_model,         # dict: model -> C_m (MB)
    vram_device,        # dict: device_type -> M_cap(d) (MB)
    Ptmd,               # dict: (task, model, device) -> P_{t,m,d} (req/s)
    Amin=None,          # dict: task -> A_min_t   (or None to ignore accuracy)
    minimize="deployments",  # objective mode
    time_limit=60,
    log_to_console=True,
):
    """
    Multi-tenant non-sharing ILP.

    Decision variables
      x[m,d] ∈ {0,1}        model m is deployed on device d
      z[d]   ∈ {0,1}        device d is used by any model
      r[t,m,d] ∈ [0,1]      fraction of task t's demand routed to (m,d)

    SLO pre-filtering:
      We only create routing variables for triples (t,m,d) that satisfy:
        - support(m,t) = 1
        - latency L_{t,m,d} ≤ Lmax_t
        - (optional) accuracy A_{t,m} ≥ A_min_t
        - throughput Ptmd[(t,m,d)] is defined and > 0

    Constraints
      (1) Demand conservation:
            ∑_{(m,d) : (t,m,d)∈F} r[t,m,d] = 1                  ∀ t
      (2) Routing only via deployed endpoints:
            r[t,m,d] ≤ x[m,d]                                   ∀ (t,m,d) ∈ F
      (3) Device-level compute capacity:
            ∑_{(t,m) : (t,m,d)∈F} (λ_t r[t,m,d] / P_{t,m,d}) ≤ 1 ∀ d
      (4) Device memory capacity:
            ∑_m C_m x[m,d] ≤ M_cap(d)                          ∀ d
      (5) Device usage:
            x[m,d] ≤ z[d]                                      ∀ m,d

    Objectives (weighted lexicographic style)
      O1: Minimize deployments  ∑_{m,d} x[m,d]
      O2: Minimize devices      ∑_d z[d]
      O3: Minimize wasted memory ∑_d (z[d] M_cap(d) - ∑_m C_m x[m,d])
      O4: Minimize total model memory ∑_{m,d} C_m x[m,d]

      Combined objective:
        O = α O1 + β O2 + γ O3 + δ O4

      with weights chosen according to `minimize`:
        - "deployments"
        - "deployments_devices"
        - "deployments_devices_waste"
        - "deployments_devices_waste_modelsize"
    """

    m = Model("nonsharing_ilp_multitenant")
    if not log_to_console:
        m.Params.OutputFlag = 0
    if time_limit is not None:
        m.Params.TimeLimit = time_limit

    # ---------- Sets ----------
    D = list(devices.keys())
    M = list(models)
    T = list(tasks.keys())

    # Per-device capacity (MB) derived from device type
    cap_d = {}
    for d in D:
        dtype = devices[d]["type"]
        cap_d[d] = float(vram_device.get(dtype, 0.0))

    # ---------- Feasibility predicate ----------
    def feasible_triple(t, mm, d):
        # support(m,t) must be 1
        if support.get((mm, t), 0) != 1:
            return False

        # latency SLO
        Lmax = tasks[t]
        L = latency.get((t, mm, d), None)
        if L is None or L > Lmax:
            return False

        # optional accuracy SLO
        if Amin is not None and t in Amin:
            A = accuracy.get((t, mm), None)
            if A is None or A < Amin[t]:
                return False

        # throughput must be defined and positive
        P = Ptmd.get((t, mm, d), None)
        if P is None or P <= 0:
            return False

        return True

    # Set of feasible triples F
    F = [(t, mm, d) for t in T for mm in M for d in D if feasible_triple(t, mm, d)]

    # Tasks with no feasible triple (diagnostic only)
    tasks_with_no_feasible_triple = [t for t in T if not any(tt == t for (tt, _, _) in F)]

    # ---------- Decision variables ----------
    # Deployment: x[m,d]
    x = {(mm, d): m.addVar(vtype=GRB.BINARY, name=f"x[{mm},{d}]") for mm in M for d in D}

    # Device usage: z[d]
    z = {d: m.addVar(vtype=GRB.BINARY, name=f"z[{d}]") for d in D}

    # Routing fractions: r[t,m,d] only for (t,m,d) in F
    r = {
        (t, mm, d): m.addVar(
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS,
            name=f"r[{t},{mm},{d}]",
        )
        for (t, mm, d) in F
    }

    m.update()

    # ---------- Constraints ----------

    # (1) Demand conservation: route all of each task's demand across feasible endpoints
    for t in T:
        m.addConstr(
            quicksum(r[(tt, mm, d)] for (tt, mm, d) in F if tt == t) == 1.0,
            name=f"demand_conservation[{t}]",
        )

    # (2) Routing only via deployed endpoints
    for (t, mm, d) in F:
        m.addConstr(
            r[(t, mm, d)] <= x[(mm, d)],
            name=f"route_implies_deploy[{t},{mm},{d}]",
        )

    # (3) Device-level compute capacity (normalized utilization ≤ 1)
    for d in D:
        utilization_terms = []
        for (t, mm, dd) in F:
            if dd != d:
                continue
            lam = float(demands.get(t, 0.0))
            P = float(Ptmd[(t, mm, d)])
            # contribution: λ_t r[t,m,d] / P_{t,m,d}
            utilization_terms.append((lam / P, (t, mm, d)))

        if utilization_terms:
            m.addConstr(
                quicksum(coeff * r[key] for coeff, key in utilization_terms) <= 1.0,
                name=f"compute_capacity[{d}]",
            )

    # (4) Device memory capacity
    for d in D:
        m.addConstr(
            quicksum(float(vram_model.get(mm, 0.0)) * x[(mm, d)] for mm in M) <= cap_d[d],
            name=f"vram[{d}]",
        )

    # (5) Device usage indicator: x[m,d] implies z[d]
    for d in D:
        # Lower bound: if any x[m,d] is 1 then z[d] must be 1
        for mm in M:
            m.addConstr(
                x[(mm, d)] <= z[d],
                name=f"device_used_lb[{mm},{d}]",
            )
        # Upper bound: if no model is deployed then z[d] must be 0
        m.addConstr(
            z[d] <= quicksum(x[(mm, d)] for mm in M),
            name=f"device_used_ub[{d}]",
        )

    # ---------- Objective ----------

    # Base components
    O1 = quicksum(x[(mm, d)] for mm in M for d in D)  # deployments
    O2 = quicksum(z[d] for d in D)                    # devices

    # Wasted memory: z[d]*cap_d[d] - sum_m C_m x[m,d]
    O3_terms = []
    for d in D:
        used_cap = quicksum(float(vram_model.get(mm, 0.0)) * x[(mm, d)] for mm in M)
        O3_terms.append(z[d] * cap_d[d] - used_cap)
    O3 = quicksum(O3_terms)

    # Total model memory deployed
    O4 = quicksum(float(vram_model.get(mm, 0.0)) * x[(mm, d)] for mm in M for d in D)

    # Weights (α » β » γ » δ) to approximate lexicographic priority
    alpha = 1e6
    beta = 1e3
    gamma = 1.0
    delta = 1e-3

    if minimize == "deployments":
        obj = alpha * O1
    elif minimize == "deployments_devices":
        obj = alpha * O1 + beta * O2
    elif minimize == "deployments_devices_waste":
        obj = alpha * O1 + beta * O2 + gamma * O3
    elif minimize == "deployments_devices_waste_modelsize":
        obj = alpha * O1 + beta * O2 + gamma * O3 + delta * O4
    else:
        # Fallback to deployments only
        obj = alpha * O1

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
        "z": {},
        "r": {},
        "used_devices": {},
        "tasks_with_no_feasible_triple": tasks_with_no_feasible_triple,
    }

    if status in ("OPTIMAL", "TIME_LIMIT"):
        try:
            result["obj"] = m.ObjVal
        except Exception:
            pass

        # deployment decisions
        for (mm, d), var in x.items():
            val = var.X if var.X is not None else 0.0
            result["x"][(mm, d)] = int(round(val))

        # device usage
        for d, var in z.items():
            val = var.X if var.X is not None else 0.0
            result["z"][d] = int(round(val))

        # routing decisions
        for key, var in r.items():
            val = var.X if var.X is not None else 0.0
            result["r"][key] = float(val)

        # used devices: directly from z
        used = {}
        for d in D:
            used[d] = result["z"].get(d, 0)
        result["used_devices"] = used

        # Calculate objective components from solution
        O1_val = sum(result["x"][(mm, d)] for mm in M for d in D)
        O2_val = sum(result["z"][d] for d in D)

        # O3: wasted memory
        O3_val = 0.0
        for d in D:
            used_mem = sum(float(vram_model.get(mm, 0.0)) * result["x"][(mm, d)] for mm in M)
            O3_val += cap_d[d] * result["z"][d] - used_mem

        # O4: total model memory
        O4_val = sum(float(vram_model.get(mm, 0.0)) * result["x"][(mm, d)] for mm in M for d in D)

        result["objective_components"] = {
            "O1_deployments": int(O1_val),
            "O2_devices": int(O2_val),
            "O3_waste_mb": float(O3_val),
            "O4_total_mem_mb": float(O4_val)
        }

    elif status == "INFEASIBLE":
        try:
            m.computeIIS()
            m.write("multitenant_ilp.iis")
        except Exception:
            pass

    return result