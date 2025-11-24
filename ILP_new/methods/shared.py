# ILP_new/methods/our_shared_multitenant.py
# Shared multi tenant ILP with explicit backbone and decoder sharing:
#   - SLO pre filtering over (t, p, d) where p is a pipeline (backbone, decoder, task)
#   - Device level compute capacity with task specific throughput P_{t,p,d}
#   - Device memory capacity with shared components:
#         * backbones shared across all pipelines that use them
#         * decoders shared across all pipelines that use them
#         * per pipeline task head memory (not shared)
#   - Multiple objective modes:
#         * "deployments"
#         * "deployments_devices"
#         * "deployments_devices_waste"
#         * "deployments_devices_waste_modelsize"
#
# Compared to the non sharing multitenant ILP, this file splits memory into
# backbone, decoder, and per pipeline head components, and adds deployment
# variables for backbones and decoders on each device.


from gurobipy import Model, GRB, quicksum


def build_and_solve_from_pipelines(
    *,
    components,         # dict: component_name -> {"mem": MB}
    pipelines,          # dict: pid -> {"backbone": ..., "decoder": ..., "task": ...}
    latency_config,     # dict: pid -> {device_type: latency_ms}
    metric_config,      # dict: pid -> accuracy or metric value
    device_type="A6000",
    device_name="gpu0",
    vram_device_cap_mb=48000.0,
    task_latency_slo_ms=None,   # dict: task -> L_max_ms; if None, derive as 1.5x max
    task_accuracy_slo=None,     # dict: task -> A_min; if None, ignore accuracy SLO
    task_demands=None,          # dict: task -> lambda (req/s); if None, default 1.0
    minimize="deployments",
    time_limit=60,
    log_to_console=True,
):
    """
    Wrapper that accepts VLM or TSFM style pipeline format and converts to shared ILP format.

    Each pipeline id "pid" corresponds to a triple (backbone, decoder, task).

    Memory is decomposed as:
      - vram_backbone[b]     shared across all pipelines using backbone b
      - vram_decoder[c]      shared across all pipelines using decoder c
      - vram_head[pid]       per pipeline task head and any pipeline specific
                             components (not shared)
    """

    # 1. Pipelines and tasks
    pipelines_ids = sorted(pipelines.keys())
    task_set = {info["task"] for info in pipelines.values()}
    tasks_list = sorted(task_set)

    # 2. Backbones and decoders
    backbones = sorted({info["backbone"] for info in pipelines.values()})
    decoders = sorted({info["decoder"] for info in pipelines.values()})

    # 3. Devices
    devices = {device_name: {"type": device_type}}
    vram_device = {device_type: float(vram_device_cap_mb)}

    # 4. Tasks dict (latency SLOs)
    tasks = {}
    if task_latency_slo_ms is not None:
        for t in tasks_list:
            tasks[t] = float(task_latency_slo_ms.get(t, 1e9))
    else:
        # Derive per task SLO as 1.5 times max observed latency for that task
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
    for pid in pipelines_ids:
        pipeline_task = pipelines[pid]["task"]
        for t in tasks_list:
            support[(pid, t)] = 1 if t == pipeline_task else 0

    # 7. Accuracy: (task, pipeline) -> metric
    accuracy = {}
    for pid, info in pipelines.items():
        t = info["task"]
        accuracy[(t, pid)] = float(metric_config.get(pid, 0.0))

    # 8. Amin accuracy SLO
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

    # 10. Throughput: P_{t,p,d} = 1000 / latency_ms (batch size 1)
    Ptmd = {}
    for key, L_ms in latency_ilp.items():
        if L_ms > 0:
            Ptmd[key] = 1000.0 / L_ms

    # 11. Memory decomposition

    # Shared backbone memory
    vram_backbone = {}
    for b in backbones:
        vram_backbone[b] = float(components.get(b, {}).get("mem", 0.0))

    # Shared decoder memory
    # We treat components[decoder] as the shareable decoder block.
    # Any pipeline specific entries like "decoder_backbone_task" or
    # "task_backbone_decoder" are counted as per pipeline head memory.
    vram_decoder = {}
    for c in decoders:
        vram_decoder[c] = float(components.get(c, {}).get("mem", 0.0))

    # Per pipeline head memory (task specific and any fused extras)
    vram_head = {}
    for pid, info in pipelines.items():
        backbone = info["backbone"]
        decoder = info["decoder"]
        task = info["task"]

        # Fused decoder block that depends on backbone and task
        # for example "decoder_backbone_task"
        dec_extra_mem = 0.0
        dec_key_fused = f"{decoder}_{backbone}_{task}"
        if dec_key_fused in components:
            dec_extra_mem += components[dec_key_fused].get("mem", 0.0)

        # Task specific component in VLM format "task_backbone_decoder"
        task_key = f"{task}_{backbone}_{decoder}"
        task_mem = components.get(task_key, {}).get("mem", 0.0)

        vram_head[pid] = float(dec_extra_mem + task_mem)

    # Call core shared ILP
    return build_and_solve(
        devices=devices,
        pipelines_ids=pipelines_ids,
        backbones=backbones,
        decoders=decoders,
        tasks=tasks,
        demands=demands,
        pipelines=pipelines,
        support=support,
        accuracy=accuracy,
        latency=latency_ilp,
        vram_backbone=vram_backbone,
        vram_decoder=vram_decoder,
        vram_head=vram_head,
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
    pipelines_ids,      # list of pipeline ids (models)
    backbones,          # list of backbone names
    decoders,           # list of decoder names
    tasks,              # dict: task -> Lmax_t (ms)
    demands,            # dict: task -> lambda_t (req/s)
    pipelines,          # dict: pid -> {"backbone": ..., "decoder": ..., "task": ...}
    support,            # dict: (pipeline, task) -> 1 or 0
    accuracy,           # dict: (task, pipeline) -> A_{t,p}
    latency,            # dict: (task, pipeline, device) -> L_{t,p,d} (ms)
    vram_backbone,      # dict: backbone -> C_backbone (MB)
    vram_decoder,       # dict: decoder -> C_decoder (MB)
    vram_head,          # dict: pipeline -> C_head (MB)
    vram_device,        # dict: device_type -> M_cap(d) (MB)
    Ptmd,               # dict: (task, pipeline, device) -> P_{t,p,d} (req/s)
    Amin=None,          # dict: task -> A_min_t (or None to ignore accuracy)
    minimize="deployments",
    time_limit=60,
    log_to_console=True,
):
    """
    Shared multi tenant non sharing ILP with backbone and decoder sharing.

    Decision variables
      x_b[b,d] ∈ {0,1}      backbone b is deployed on device d
      x_c[c,d] ∈ {0,1}      decoder c is deployed on device d
      x_p[p,d] ∈ {0,1}      pipeline p's task head is deployed on device d
      z[d]     ∈ {0,1}      device d is used by any component
      r[t,p,d] ∈ [0,1]      fraction of task t's demand routed to pipeline p on device d

    SLO pre filtering:
      We only create routing variables for triples (t,p,d) that satisfy:
        - support(p,t) = 1
        - latency L_{t,p,d} ≤ Lmax_t
        - optional accuracy A_{t,p} ≥ A_min_t
        - throughput P_{t,p,d} is defined and positive

    Constraints
      (1) Demand conservation:
            ∑_{(p,d) : (t,p,d)∈F} r[t,p,d] = 1                  ∀ t
      (2) Routing only via deployed pipeline heads:
            r[t,p,d] ≤ x_p[p,d]                                 ∀ (t,p,d) ∈ F
      (3) Pipeline implies backbone and decoder:
            x_p[p,d] ≤ x_b[b_p,d]                               ∀ p,d
            x_p[p,d] ≤ x_c[c_p,d]                               ∀ p,d
      (4) Device level compute capacity:
            ∑_{(t,p) : (t,p,d)∈F} (λ_t r[t,p,d] / P_{t,p,d}) ≤ 1 ∀ d
      (5) Device memory capacity:
            ∑_b C_backbone[b] x_b[b,d]
          + ∑_c C_decoder[c]   x_c[c,d]
          + ∑_p C_head[p]      x_p[p,d]
          ≤ M_cap(d)                                              ∀ d
      (6) Device usage indicator:
            x_b[b,d] ≤ z[d],  x_c[c,d] ≤ z[d],  x_p[p,d] ≤ z[d]   ∀ b,c,p,d
            z[d] ≤ ∑_b x_b[b,d] + ∑_c x_c[c,d] + ∑_p x_p[p,d]     ∀ d

    Objective components
      O1: number of deployments (all components)
            O1 = ∑_{b,d} x_b[b,d] + ∑_{c,d} x_c[c,d] + ∑_{p,d} x_p[p,d]
      O2: number of devices used
            O2 = ∑_d z[d]
      O3: wasted memory on used devices
            O3 = ∑_d (z[d] M_cap(d) - used_mem[d])
      O4: total deployed memory
            O4 = ∑_d used_mem[d]

      Combined objective:
        O = α O1 + β O2 + γ O3 + δ O4

      with α » β » γ » δ controlled by `minimize` mode:
        - "deployments"
        - "deployments_devices"
        - "deployments_devices_waste"
        - "deployments_devices_waste_modelsize"
    """

    m = Model("shared_ilp_multitenant")
    if not log_to_console:
        m.Params.OutputFlag = 0
    if time_limit is not None:
        m.Params.TimeLimit = time_limit

    # ---------- Sets ----------
    D = list(devices.keys())
    P_ids = list(pipelines_ids)
    B = list(backbones)
    C = list(decoders)
    T = list(tasks.keys())

    # Per device capacity (MB)
    cap_d = {}
    for d in D:
        dtype = devices[d]["type"]
        cap_d[d] = float(vram_device.get(dtype, 0.0))

    # ---------- Feasibility predicate ----------
    def feasible_triple(t, p, d):
        # pipeline must support task
        if support.get((p, t), 0) != 1:
            return False

        # latency SLO
        Lmax = tasks[t]
        L = latency.get((t, p, d), None)
        if L is None or L > Lmax:
            return False

        # optional accuracy SLO
        if Amin is not None and t in Amin:
            A = accuracy.get((t, p), None)
            if A is None or A < Amin[t]:
                return False

        # throughput must be defined and positive
        P = Ptmd.get((t, p, d), None)
        if P is None or P <= 0:
            return False

        return True

    # Set of feasible triples F over tasks, pipelines, devices
    F = [(t, p, d) for t in T for p in P_ids for d in D if feasible_triple(t, p, d)]

    # Tasks with no feasible triple (diagnostic)
    tasks_with_no_feasible_triple = [t for t in T if not any(tt == t for (tt, _, _) in F)]

    # ---------- Decision variables ----------

    # Backbone deployment: x_b[b,d]
    x_b = {
        (b, d): m.addVar(vtype=GRB.BINARY, name=f"x_b[{b},{d}]")
        for b in B for d in D
    }

    # Decoder deployment: x_c[c,d]
    x_c = {
        (c, d): m.addVar(vtype=GRB.BINARY, name=f"x_c[{c},{d}]")
        for c in C for d in D
    }

    # Pipeline head deployment: x_p[p,d]
    x_p = {
        (p, d): m.addVar(vtype=GRB.BINARY, name=f"x_p[{p},{d}]")
        for p in P_ids for d in D
    }

    # Device usage: z[d]
    z = {d: m.addVar(vtype=GRB.BINARY, name=f"z[{d}]") for d in D}

    # Routing fractions: r[t,p,d] only for feasible triples
    r = {
        (t, p, d): m.addVar(
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS,
            name=f"r[{t},{p},{d}]",
        )
        for (t, p, d) in F
    }

    m.update()

    # ---------- Constraints ----------

    # (1) Demand conservation
    for t in T:
        m.addConstr(
            quicksum(r[(tt, p, d)] for (tt, p, d) in F if tt == t) == 1.0,
            name=f"demand_conservation[{t}]",
        )

    # (2) Routing only via deployed pipeline heads
    for (t, p, d) in F:
        m.addConstr(
            r[(t, p, d)] <= x_p[(p, d)],
            name=f"route_implies_pipeline[{t},{p},{d}]",
        )

    # (3) Pipeline implies backbone and decoder deployment on the device
    for p in P_ids:
        b_p = pipelines[p]["backbone"]
        c_p = pipelines[p]["decoder"]
        for d in D:
            m.addConstr(
                x_p[(p, d)] <= x_b[(b_p, d)],
                name=f"pipeline_implies_backbone[{p},{d}]",
            )
            m.addConstr(
                x_p[(p, d)] <= x_c[(c_p, d)],
                name=f"pipeline_implies_decoder[{p},{d}]",
            )

    # (4) Device compute capacity: normalized utilization ≤ 1
    for d in D:
        utilization_terms = []
        for (t, p, dd) in F:
            if dd != d:
                continue
            lam = float(demands.get(t, 0.0))
            P_val = float(Ptmd[(t, p, d)])
            utilization_terms.append((lam / P_val, (t, p, d)))

        if utilization_terms:
            m.addConstr(
                quicksum(coeff * r[key] for coeff, key in utilization_terms) <= 1.0,
                name=f"compute_capacity[{d}]",
            )

    # (5) Device memory capacity
    for d in D:
        mem_backbones = quicksum(
            float(vram_backbone.get(b, 0.0)) * x_b[(b, d)] for b in B
        )
        mem_decoders = quicksum(
            float(vram_decoder.get(c, 0.0)) * x_c[(c, d)] for c in C
        )
        mem_heads = quicksum(
            float(vram_head.get(p, 0.0)) * x_p[(p, d)] for p in P_ids
        )

        m.addConstr(
            mem_backbones + mem_decoders + mem_heads <= cap_d[d],
            name=f"vram[{d}]",
        )

    # (6) Device usage indicator: any deployment implies z[d] = 1
    for d in D:
        # Lower bounds
        for b in B:
            m.addConstr(
                x_b[(b, d)] <= z[d],
                name=f"device_used_lb_backbone[{b},{d}]",
            )
        for c in C:
            m.addConstr(
                x_c[(c, d)] <= z[d],
                name=f"device_used_lb_decoder[{c},{d}]",
            )
        for p in P_ids:
            m.addConstr(
                x_p[(p, d)] <= z[d],
                name=f"device_used_lb_pipeline[{p},{d}]",
            )

        # Upper bound: if nothing is deployed then z[d] is forced to 0
        m.addConstr(
            z[d] <= quicksum(x_b[(b, d)] for b in B)
                 + quicksum(x_c[(c, d)] for c in C)
                 + quicksum(x_p[(p, d)] for p in P_ids),
            name=f"device_used_ub[{d}]",
        )

    # ---------- Objective ----------

    # Objective components

    # O1: total deployments of all components
    O1 = (
        quicksum(x_b[(b, d)] for b in B for d in D)
        + quicksum(x_c[(c, d)] for c in C for d in D)
        + quicksum(x_p[(p, d)] for p in P_ids for d in D)
    )

    # O2: number of devices used
    O2 = quicksum(z[d] for d in D)

    # O3: wasted memory on used devices
    O3_terms = []
    for d in D:
        used_mem_d = (
            quicksum(float(vram_backbone.get(b, 0.0)) * x_b[(b, d)] for b in B)
            + quicksum(float(vram_decoder.get(c, 0.0)) * x_c[(c, d)] for c in C)
            + quicksum(float(vram_head.get(p, 0.0)) * x_p[(p, d)] for p in P_ids)
        )
        O3_terms.append(z[d] * cap_d[d] - used_mem_d)
    O3 = quicksum(O3_terms)

    # O4: total deployed memory
    O4 = quicksum(
        float(vram_backbone.get(b, 0.0)) * x_b[(b, d)] for b in B for d in D
    ) + quicksum(
        float(vram_decoder.get(c, 0.0)) * x_c[(c, d)] for c in C for d in D
    ) + quicksum(
        float(vram_head.get(p, 0.0)) * x_p[(p, d)] for p in P_ids for d in D
    )

    # Weights (α » β » γ » δ)
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
        "x_backbone": {},
        "x_decoder": {},
        "x_pipeline": {},
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

        # backbone deployments
        for (b, d), var in x_b.items():
            val = var.X if var.X is not None else 0.0
            result["x_backbone"][(b, d)] = int(round(val))

        # decoder deployments
        for (c, d), var in x_c.items():
            val = var.X if var.X is not None else 0.0
            result["x_decoder"][(c, d)] = int(round(val))

        # pipeline deployments
        for (p, d), var in x_p.items():
            val = var.X if var.X is not None else 0.0
            result["x_pipeline"][(p, d)] = int(round(val))

        # device usage
        for d, var in z.items():
            val = var.X if var.X is not None else 0.0
            result["z"][d] = int(round(val))

        # routing
        for key, var in r.items():
            val = var.X if var.X is not None else 0.0
            result["r"][key] = float(val)

        # used devices derived from z
        used = {}
        for d in D:
            used[d] = result["z"].get(d, 0)
        result["used_devices"] = used

        # compute objective components numerically from solution
        O1_val = (
            sum(result["x_backbone"][(b, d)] for b in B for d in D)
            + sum(result["x_decoder"][(c, d)] for c in C for d in D)
            + sum(result["x_pipeline"][(p, d)] for p in P_ids for d in D)
        )
        O2_val = sum(result["z"][d] for d in D)

        O3_val = 0.0
        O4_val = 0.0
        for d in D:
            used_mem_d = 0.0
            for b in B:
                used_mem_d += float(vram_backbone.get(b, 0.0)) * result["x_backbone"][(b, d)]
            for c in C:
                used_mem_d += float(vram_decoder.get(c, 0.0)) * result["x_decoder"][(c, d)]
            for p in P_ids:
                used_mem_d += float(vram_head.get(p, 0.0)) * result["x_pipeline"][(p, d)]

            O3_val += cap_d[d] * result["z"][d] - used_mem_d
            O4_val += used_mem_d

        result["objective_components"] = {
            "O1_deployments_all": int(O1_val),
            "O2_devices": int(O2_val),
            "O3_waste_mb": float(O3_val),
            "O4_total_mem_mb": float(O4_val),
        }

    elif status == "INFEASIBLE":
        try:
            m.computeIIS()
            m.write("shared_ilp_multitenant.iis")
        except Exception:
            pass

    return result
