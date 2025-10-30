import json
import numpy as np
from request import Request

def parse_plan(plan_json):
    """Returns mappings:
    task_routes = {task: [(site_manager, device, requests_per_sec)]}
    task_totals = {task: total rate}
    """
    task_routes, task_totals = {}, {}
    for site in plan_json["sites"]:
        site_manager = site["id"]
        for deploy in site["deployments"]:
            device = deploy["device"]
            for dec in deploy["decoders"]:
                task = dec["task"]
                rate = deploy["tasks"][task]['request_per_sec']
                task_routes.setdefault(task, []).append((site_manager, device, rate))
                task_totals[task] = task_totals.get(task, 0.0) + rate
    return task_routes, task_totals


def route_trace(trace_requests, plan_json, seed=42):
    """Reroute an existing unified trace (from gamma.py) to devices as per deployment plan."""
    np.random.seed(seed)
    task_routes, task_totals = parse_plan(plan_json)

    routed = []
    for req in trace_requests:
        task = req.task
        if task not in task_routes:
            # if task isn't in plan, keep it unchanged
            routed.append(req)
            continue

        routes = task_routes[task]
        total_task_rate = task_totals[task]
        probs = np.array([r[2] for r in routes]) / total_task_rate
        idx = np.random.choice(len(routes), p=probs)
        site, device, _ = routes[idx]

        routed.append(Request(req.req_id, task, site, device, req.req_time))

    return routed

#168
#84 : route 1 device1
#84 route 2 device2
#np.random.choice(2, p=[0.5, 0.5])

