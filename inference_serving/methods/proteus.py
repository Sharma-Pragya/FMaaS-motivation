import gurobipy as gp
from gurobipy import GRB
from inference_serving.config import *
from inference_serving.profiler import *

def proteus():
    # Create model
    m = gp.Model("Proteus_MIQP")
    m.setParam("OutputFlag", 0)
    m.setParam("NonConvex", 2)

    # Variables
    x = m.addVars(devices, models, vtype=GRB.BINARY, name="x")                 # model placement
    y = m.addVars(devices, tasks, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="y")  # query assignment ratio
    z = m.addVars(devices, tasks, lb=0, vtype=GRB.CONTINUOUS, name="z")        # query throughput

    # Constraint 1: each device hosts at most one model
    for d in devices:
        m.addConstr(gp.quicksum(x[d, m] for m in models) <= 1, f"c1_{d}")

    # Constraint 2: total ratio for query type routed ≤ 1
    for q in tasks:
        m.addConstr(gp.quicksum(y[d, q] for d in devices) <= 1, f"c2_{q}")

    # Constraint 3: assignment only if hosted model supports query type
    for q in tasks:
        lhs = gp.quicksum(
            can_serve[m_name, q] * x[d, m_name] * y[d, q]
            for d in devices for m_name in models if (m_name, q) in can_serve
        )
        rhs = gp.quicksum(y[d, q] for d in devices)
        m.addConstr(lhs == rhs, f"c3_{q}")

    # Constraint 4: z[d,q] ≤ y[d,q] * s[q]
    for d in devices:
        m.addConstr(
            gp.quicksum(z[d, q] for q in tasks)
            <= gp.quicksum(y[d, q] * s[q] for q in tasks),
            f"c4_{d}"
        )

    # Constraint 5: z[d,q] ≤ sum throughput of hosted model variant
    for d in devices:
        for q in tasks:
            m.addConstr(
                z[d, q] <= gp.quicksum(
                    throughput_capacity[d, m_name, q] * x[d, m_name]
                    for m_name in models if (d, m_name, q) in throughput_capacity
                ), f"c5_{d}_{q}"
            )

    # Constraint 6: total served = demand
    for q in tasks:
        m.addConstr(
            gp.quicksum(z[d, q] for d in devices) == s[q], f"c6_{q}"
        )

    # Objective: maximize effective accuracy
    m.setObjective(
        gp.quicksum(
            accuracy[m_name] * x[d, m_name] * z[d, q]
            for d in devices for m_name in models for q in tasks if (m_name, q) in can_serve
        ),
        GRB.MAXIMIZE
    )

    # Solve
    m.optimize()

    # Results
    results = {
        "Status": m.status,
        "Total Accuracy": m.objVal if m.status == GRB.OPTIMAL else None,
        "Model Placements": {(d, m_name): x[d, m_name].X for d in devices for m_name in models},
        "Query Assignments": {(d, q): y[d, q].X for d in devices for q in tasks},
        "Device QPS Served": {(d, q): z[d, q].X for d in devices for q in tasks},
    }
    return results
