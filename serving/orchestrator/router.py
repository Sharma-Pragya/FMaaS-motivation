import numpy as np
from orchestrator.request import Request


def parse_plan(plan_json):
    """Parse deployment plan into per-task routing tables.

    Returns:
        task_routes: {task: [(site_manager, device, backbone, requests_per_sec)]}
        task_totals: {task: total_rate}
    """
    task_routes, task_totals = {}, {}
    for site in plan_json["sites"]:
        site_manager = site["id"]
        for deploy in site["deployments"]:
            device = deploy["device"]
            backbone = deploy["backbone"]
            for dec in deploy["decoders"]:
                task = dec["task"]
                rate = deploy["tasks"][task]["request_per_sec"]
                task_routes.setdefault(task, []).append((site_manager, device, backbone, rate))
                task_totals[task] = task_totals.get(task, 0.0) + rate
    return task_routes, task_totals


def route_trace(trace_requests, plan_json, seed=42):
    """Route a unified trace to devices according to the deployment plan.

    Assigns each request to a device probabilistically, proportional to
    each device's allocated request rate for that task.

    No requests are dropped — generate_trace() already produces exactly
    the right count for the planned rate, so acceptance sampling is not
    needed and would incorrectly drop spike requests (which are generated
    at the delta rate but routed against the pre-update plan rate).
    """
    np.random.seed(seed)
    task_routes, task_totals = parse_plan(plan_json)

    incoming_counts = {}
    for req in trace_requests:
        incoming_counts[req.task] = incoming_counts.get(req.task, 0) + 1

    print("task_routes:", task_routes)
    print("task_totals:", task_totals)
    print("incoming_counts:", incoming_counts)

    routed = []
    for req in trace_requests:
        task = req.task
        if task not in task_routes:
            routed.append(req)
            continue

        routes = task_routes[task]
        total_task_rate = task_totals[task]

        probs = np.array([r[3] for r in routes]) / total_task_rate
        idx = np.random.choice(len(routes), p=probs)
        site, device, backbone, _ = routes[idx]

        routed.append(Request(req.req_id, task, site, device, backbone, req.req_time))

    print(len(routed))
    return routed
