import gurobipy as gp
from gurobipy import GRB
from ..config import *
from ..profiler import *

def our():
    # Create model
    m = gp.Model("Our_MIQP")
    m.setParam("OutputFlag", 0)
    m.setParam("NonConvex", 2)

    # Variables
    x = m.addVars(devices, models, vtype=GRB.BINARY, name="x") # model placement
    y = m.addVars(devices, tasks, vtype=GRB.BINARY, name="y")  # task assignment

    # Constraint 1: each device hosts at most one model
    for d in devices:
        m.addConstr(gp.quicksum(x[d, m] for m in models) <= 1, f"c1_{d}")

    # Constraint 2: total ratio for query type routed ≤ 1
    for q in tasks:
        m.addConstr(gp.quicksum(y[d, q] for d in devices) == 1, f"c2_{q}")

    # Constraint 3: assignment only if hosted model supports query type
    for q in tasks:
        lhs = gp.quicksum(
            can_serve[m_name, q] * x[d, m_name] * y[d, q]
            for d in devices for m_name in models if (m_name, q) in can_serve
        )
        rhs = gp.quicksum(y[d, q] for d in devices)
        m.addConstr(lhs == rhs, f"c3_{q}")
    
    # Constraint 4: Memory constraint
    for d in devices:
        m.addConstr(
            gp.quicksum(
                model_memory[m_name] * x[d, m_name]
                for m_name in models
            ) <= memory_device[d],
            f"c4_{d}"
        )



    # Objective: maximize effective accuracy
    m.setObjective(
        gp.quicksum(
            x[d, m_name]
            for d in devices for m_name in models
        ),
        GRB.MINIMIZE
    )

    # Solve
    m.optimize()

    results = {"Status": m.status}
    if m.status == GRB.INFEASIBLE:
        print("Model is infeasible. Generating IIS...")
        m.computeIIS()
        m.write("model.ilp")   # Human-readable summary
    if m.status == GRB.OPTIMAL or m.status == GRB.SUBOPTIMAL:
        results.update({
            "Number of Models": m.objVal,
            "Model Placements": {(d, m_name): x[d, m_name].X for d in devices for m_name in models if x[d, m_name].X ==1},
            "Query Assignments": {(d, q): y[d, q].X for d in devices for q in tasks if y[d, q].X ==1},
        })
    else:
        results.update({
            "Number of Models": None,
            "Model Placements": {},
            "Query Assignments": {}
        })
    return results
