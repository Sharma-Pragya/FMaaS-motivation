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

    Probabilistically assigns each request to a device proportional to
    its allocated request rate. Drops requests if incoming rate exceeds
    planned capacity.
    """
    np.random.seed(seed)
    task_routes, task_totals = parse_plan(plan_json)

    incoming_counts = {}
    for req in trace_requests:
        incoming_counts[req.task] = incoming_counts.get(req.task, 0) + 1

    if trace_requests:
        min_time = min(req.req_time for req in trace_requests)
        max_time = max(req.req_time for req in trace_requests)
        trace_duration = max_time - min_time if max_time > min_time else 1.0
    else:
        trace_duration = 1.0

    print("task_routes:", task_routes)
    print("task_totals:", task_totals)
    print("incoming_counts:", incoming_counts)
    print("trace_duration:", trace_duration)

    routed = []
    for req in trace_requests:
        task = req.task
        if task not in task_routes:
            routed.append(req)
            continue

        routes = task_routes[task]
        total_task_rate = task_totals[task]
        incoming_count = incoming_counts[task]

        expected_count = total_task_rate * trace_duration
        if incoming_count > expected_count:
            accept_prob = expected_count / incoming_count
            if np.random.random() > accept_prob:
                continue

        probs = np.array([r[3] for r in routes]) / total_task_rate
        idx = np.random.choice(len(routes), p=probs)
        site, device, backbone, _ = routes[idx]

        routed.append(Request(req.req_id, task, site, device, backbone, req.req_time))

    print(len(routed))
    return routed
