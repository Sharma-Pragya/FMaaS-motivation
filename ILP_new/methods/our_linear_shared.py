# our_linear_shared.py

# NOTE: This file preserves the same overall structure and naming style
# (x / u / r) as a typical multi-tenant ILP like your our_linear.py.
# Only the necessary decision variables and constraints are added to support
# co-located Backbone (B) + Head/Decoder (H) sharing alongside Monolithic (M).

from typing import Dict, List, Any, Optional

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    raise RuntimeError("gurobipy is required to run this ILP.") from e


def _v(dct, *keys, default=0.0):
    """Safe nested dict getter returning default when missing."""
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_model(data: Dict[str, Any], params: Optional[Dict[str, Any]] = None):
    """
    Build the multi-tenant ILP with Backbone+Head sharing (co-located) and Monoliths.

    Expected keys in `data` (mirrors a standard multi-tenant ILP, extended for B/H):
      devices: [d...]
      tasks:   [t...]

      # Monolithic models
      M: [m...]
      mM[m]: memory
      cM[m][d]: per-request compute on device d
      feasM[(t,m,d)] in {0,1}

      # Backbones and Heads
      B: [b...]
      H: [h...]
      b_of_h[h] -> b
      mB[b], mH[h]
      cB[b][d], cH[h][d]
      feasBH[(t,h,d)] in {0,1}  # implies using b_of_h[h] on same device

      # Workload & capacity
      lambda_t[t]
      R_t[t]  (redundancy, >=1)
      Mem[d], Cap[d], rho (headroom multiplier)

      # Optional placement costs
      alphaM[m], betaB[b], gammaH[h]
      minimize_devices (bool), device_cost[d]

    Returns:
      model: gp.Model
      vars_: dict of Gurobi tupledicts for x, u, r (and w if added)
    """
    if params is None:
        params = {}

    # Sets
    D: List[str] = list(data.get("devices", []))
    T: List[str] = list(data.get("tasks", []))
    M: List[str] = list(data.get("M", []))
    B: List[str] = list(data.get("B", []))
    H: List[str] = list(data.get("H", []))
    b_of_h: Dict[str, str] = dict(data.get("b_of_h", {}))

    # Workload / capacities
    lam = dict(data.get("lambda_t", {}))
    R_t = dict(data.get("R_t", {}))
    Mem = dict(data.get("Mem", {}))
    Cap = dict(data.get("Cap", {}))
    rho = float(data.get("rho", 1.0))
    Cap_eff = {d: rho * Cap.get(d, 0.0) for d in D}

    # Profiles
    mM = dict(data.get("mM", {}))
    mB = dict(data.get("mB", {}))
    mH = dict(data.get("mH", {}))

    cM = dict(data.get("cM", {}))  # cM[m][d]
    cB = dict(data.get("cB", {}))  # cB[b][d]
    cH = dict(data.get("cH", {}))  # cH[h][d]

    # Feasibility masks
    feasM = dict(data.get("feasM", {}))    # keyed (t,m,d)
    feasBH = dict(data.get("feasBH", {}))  # keyed (t,h,d), for co-located b(h) on same d

    # Optional costs
    alphaM = {m: data.get("alphaM", {}).get(m, 1.0) for m in M}
    betaB  = {b: data.get("betaB", {}).get(b, 1.0)  for b in B}
    gammaH = {h: data.get("gammaH", {}).get(h, 1.0) for h in H}
    device_cost = {d: data.get("device_cost", {}).get(d, 1.0) for d in D}

    # Solver params
    time_limit = params.get("time_limit", 300)
    mip_gap = params.get("mip_gap", 0.0)
    minimize_devices = params.get("minimize_devices", False)
    include_util = params.get("include_util_in_objective", False)
    util_weight = params.get("util_weight", 0.0)
    log_to_console = params.get("log_to_console", True)

    model = gp.Model("our_linear_shared")
    model.Params.OutputFlag = 1 if log_to_console else 0
    model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    # ============================================================
    # Decision variables — keep x/u/r naming (extend index domains)
    # ============================================================

    # Placement x:
    #   x[('M', m, d)] — monolithic m on device d
    #   x[('B', b, d)] — backbone b on device d
    #   x[('H', h, d)] — head h on device d (must satisfy co-location with its backbone)
    x = gp.tupledict()

    for m in M:
        for d in D:
            x[('M', m, d)] = model.addVar(vtype=GRB.BINARY, name=f"x[M,{m},{d}]")

    for b in B:
        for d in D:
            x[('B', b, d)] = model.addVar(vtype=GRB.BINARY, name=f"x[B,{b},{d}]")

    for h in H:
        for d in D:
            x[('H', h, d)] = model.addVar(vtype=GRB.BINARY, name=f"x[H,{h},{d}]")

    # Homes u for redundancy (like original):
    #   u[('M', t, m, d)]
    #   u[('H', t, h, d)]  (no separate homes for backbone-only)
    u = gp.tupledict()
    for t in T:
        for m in M:
            for d in D:
                u[('M', t, m, d)] = model.addVar(vtype=GRB.BINARY, name=f"u[M,{t},{m},{d}]")
        for h in H:
            for d in D:
                u[('H', t, h, d)] = model.addVar(vtype=GRB.BINARY, name=f"u[H,{t},{h},{d}]")

    # Routing r:
    #   r[('M', t, m, d)]
    #   r[('H', t, h, d)]
    r = gp.tupledict()
    for t in T:
        for m in M:
            for d in D:
                r[('M', t, m, d)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                                 name=f"r[M,{t},{m},{d}]")
        for h in H:
            for d in D:
                r[('H', t, h, d)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                                 name=f"r[H,{t},{h},{d}]")

    # Optional device-usage if minimizing devices
    if minimize_devices:
        w = model.addVars(D, vtype=GRB.BINARY, name="w")
    else:
        w = None

    model.update()

    # =================
    # Core constraints
    # =================

    # Demand conservation: sum over monolith paths + head paths == 1
    for t in T:
        model.addConstr(
            gp.quicksum(r[('M', t, m, d)] for m in M for d in D if feasM.get((t, m, d), 0) == 1) +
            gp.quicksum(r[('H', t, h, d)] for h in H for d in D if feasBH.get((t, h, d), 0) == 1)
            == 1.0,
            name=f"demand[{t}]"
        )

    # Routing ⇒ homes ⇒ placement, and feasibility masks
    for t in T:
        for m in M:
            for d in D:
                feas = feasM.get((t, m, d), 0)
                model.addConstr(r[('M', t, m, d)] <= u[('M', t, m, d)], name=f"r_to_u_M[{t},{m},{d}]")
                model.addConstr(u[('M', t, m, d)] <= feas,              name=f"u_feas_M[{t},{m},{d}]")
                model.addConstr(u[('M', t, m, d)] <= x[('M', m, d)],    name=f"u_to_x_M[{t},{m},{d}]")

        for h in H:
            for d in D:
                feas = feasBH.get((t, h, d), 0)
                model.addConstr(r[('H', t, h, d)] <= u[('H', t, h, d)], name=f"r_to_u_H[{t},{h},{d}]")
                model.addConstr(u[('H', t, h, d)] <= feas,              name=f"u_feas_H[{t},{h},{d}]")
                model.addConstr(u[('H', t, h, d)] <= x[('H', h, d)],    name=f"u_to_x_H[{t},{h},{d}]")

    # Co-location: a head can be placed only if its backbone is placed on the same device
    for h in H:
        b = b_of_h[h]
        for d in D:
            model.addConstr(x[('H', h, d)] <= x[('B', b, d)], name=f"head_coloc[{h},{d}]")

    # Redundancy: total distinct homes across (M-sites + H-sites) >= R_t
    for t in T:
        Rt = int(R_t.get(t, 1))
        model.addConstr(
            gp.quicksum(u[('M', t, m, d)] for m in M for d in D) +
            gp.quicksum(u[('H', t, h, d)] for h in H for d in D)
            >= Rt,
            name=f"redundancy[{t}]"
        )

    # Memory per device: pay once per placed component
    for d in D:
        model.addConstr(
            gp.quicksum(mM.get(m, 0.0) * x[('M', m, d)] for m in M) +
            gp.quicksum(mB.get(b, 0.0) * x[('B', b, d)] for b in B) +
            gp.quicksum(mH.get(h, 0.0) * x[('H', h, d)] for h in H)
            <= Mem.get(d, 0.0),
            name=f"mem[{d}]"
        )

    # Compute capacity per device (headroom-aware).
    # Each monolith request uses cM[m][d].
    # Each BH request uses cB[b(h)][d] + cH[h][d] on the SAME device (co-located).
    for d in D:
        util_monolith = gp.quicksum(
            lam.get(t, 0.0) * _v(cM, m, d) * r[('M', t, m, d)]
            for t in T for m in M
        )
        util_bh = gp.quicksum(
            lam.get(t, 0.0) * (_v(cB, b_of_h.get(h, ""), d) + _v(cH, h, d)) * r[('H', t, h, d)]
            for t in T for h in H
        )
        model.addConstr(util_monolith + util_bh <= Cap_eff.get(d, 0.0), name=f"cap[{d}]")

    # Optional device usage linking
    if minimize_devices:
        for d in D:
            for m in M:
                model.addConstr(x[('M', m, d)] <= w[d], name=f"xM_le_w[{m},{d}]")
            for b in B:
                model.addConstr(x[('B', b, d)] <= w[d], name=f"xB_le_w[{b},{d}]")
            for h in H:
                model.addConstr(x[('H', h, d)] <= w[d], name=f"xH_le_w[{h},{d}]")

    # =========
    # Objective
    # =========
    place_cost = (
        gp.quicksum(alphaM[m] * x[('M', m, d)] for m in M for d in D) +
        gp.quicksum(betaB[b]  * x[('B', b, d)] for b in B for d in D) +
        gp.quicksum(gammaH[h] * x[('H', h, d)] for h in H for d in D)
    )

    device_term = gp.LinExpr(0.0)
    if minimize_devices:
        device_term = gp.quicksum(device_cost[d] * w[d] for d in D)

    util_term = gp.LinExpr(0.0)
    if include_util and util_weight != 0.0:
        util_term = util_weight * gp.quicksum(
            lam.get(t, 0.0) * (
                gp.quicksum(_v(cM, m, d) * r[('M', t, m, d)] for m in M) +
                gp.quicksum((_v(cB, b_of_h.get(h, ""), d) + _v(cH, h, d)) * r[('H', t, h, d)] for h in H)
            )
            for t in T for d in D
        )

    model.setObjective(place_cost + device_term + util_term, GRB.MINIMIZE)

    # Return model and all core variables with the same naming surface (x/u/r)
    vars_ = {"x": x, "u": u, "r": r}
    if minimize_devices:
        vars_["w"] = w

    return model, vars_


def solve(data: Dict[str, Any], params: Optional[Dict[str, Any]] = None):
    """
    Build and solve; returns (model, vars_, sol) where sol packs objective/status and non-zero vars.
    """
    model, vars_ = build_model(data, params)
    model.optimize()

    sol = {"obj": None, "status": model.Status}
    if model.SolCount > 0 and model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        sol["obj"] = model.ObjVal

        # Extract compact non-zero solution
        def _dump_tupledict(td, bin_thresh=0.5, cont_thresh=1e-9):
            out = {}
            for k, v in td.items():
                if v.VType in (GRB.BINARY, GRB.INTEGER):
                    val = 1 if v.X >= bin_thresh else 0
                    if val != 0:
                        out[k] = val
                else:
                    val = float(v.X)
                    if abs(val) >= cont_thresh:
                        out[k] = val
            return out

        for name in ("x", "u", "r"):
            sol[name] = _dump_tupledict(vars_[name])

        if "w" in vars_:
            sol["w"] = _dump_tupledict(vars_["w"])

    return model, vars_, sol