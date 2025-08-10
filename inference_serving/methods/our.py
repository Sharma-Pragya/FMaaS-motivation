import gurobipy as gp
from gurobipy import GRB
from inference_serving.profiler import *

def our(devices,models,tasks,redundancy=1):
    # Create model
    m = gp.Model("Our_MIQP")
    m.setParam("OutputFlag", 0)
    m.setParam("NonConvex", 2)

    # Variables
    x = m.addVars(devices.keys(), models, vtype=GRB.BINARY, name="x") # model placement
    y = m.addVars(devices.keys(), tasks.keys(), vtype=GRB.BINARY, name="y")  # task assignment

    # Constraint 1: each device hosts at most one model
    for d in devices.keys():
        m.addConstr(gp.quicksum(x[d, m] for m in models) <= 1, f"c1_{d}")

    # Constraint 2: total ratio for query type routed ≤ 1
    for q in tasks.keys():
        m.addConstr(gp.quicksum(y[d, q] for d in devices) == redundancy, f"c2_{q}")

    # Constraint 3: assignment only if hosted model supports query type
    for q in tasks.keys():
        lhs = gp.quicksum(
            can_serve[m_name, q] * x[d, m_name] * y[d, q]
            for d in devices for m_name in models if (m_name, q) in can_serve
        )
        rhs = gp.quicksum(y[d, q] for d in devices)
        m.addConstr(lhs == rhs, f"c3_{q}")
    
    # Constraint 4: Memory constraint
    for d in devices.keys():
        m.addConstr(
            gp.quicksum(
                model_memory[m_name] * x[d, m_name]
                for m_name in models
            ) <= memory_device[devices[d]['type']],
            f"c4_{d}"
        )
    Q=100
    #Constraint 4: Latency constraint
    for d in devices.keys():
        for q in tasks.keys():
            for m_name in models:
                if (m_name, q) in can_serve:
                    lhs=latency[devices[d]['type'], m_name, q] * can_serve[m_name, q] * x[d, m_name] * y[d, q]
                    rhs=tasks[q] + Q*(1 - can_serve[m_name, q] * x[d, m_name] * y[d, q])
                    m.addConstr(
                        lhs<= rhs,
                        f"c5_{d}_{m_name}_{q}"
                    )

    # Objective: maximize effective accuracy
    m.setObjective(
        gp.quicksum(
            x[d, m_name]
            for d in devices for m_name in models
        ),
        GRB.MINIMIZE
    )

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
            "Query Assignments": {(d, q): y[d, q].X for d in devices for q in tasks.keys() if y[d, q].X ==1},
            "Runtime":  m.Runtime
        })
    else:
        results.update({
            "Number of Models": None,
            "Model Placements": {},
            "Query Assignments": {}
        })
    return results
