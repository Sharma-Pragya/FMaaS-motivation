"""Trace generation — builds a workload trace from raw trace generators.

Sits between traces/ (statistical generators) and client/runner.py (dispatcher).
Used by both local mode and MQTT mode (orchestrator/server.py).

req_rate can be:
  - float : total req/s across all tasks (split equally)
  - list  : per-task req/s, req_rate[i] maps to sorted(tasks_dict.keys())[i]

Trace Request objects contain only req_id, task, req_time.
Routing (device, backbone, site_manager) is resolved lazily at dispatch time.
"""

from typing import Dict, List, Union


def generate_trace(
    trace_type: str,
    req_rate: Union[float, List[float]],
    duration: float,
    tasks_dict: Dict,
    seed: int,
    req_id_offset: int = 0,
) -> tuple:
    task_names = sorted(tasks_dict.keys())

    if trace_type == 'lmsyschat':
        from traces.lmsyschat import generate_requests
        return generate_requests(req_rate, duration, task_names, seed, req_id_offset)
    elif trace_type == 'gamma':
        from traces.gamma import generate_requests
        return generate_requests(req_rate, duration, task_names, seed=seed, req_id_offset=req_id_offset)
    elif trace_type == 'chatbotarena':
        from traces.chatbotarena import generate_requests
        return generate_requests(req_rate, duration, task_names, seed, req_id_offset)
    elif trace_type == 'deterministic':
        from traces.deterministic import generate_requests
        return generate_requests(req_rate, duration, task_names, seed, req_id_offset)
    elif trace_type in ('poisson', 'poisson_per_task'):
        from traces.poisson_per_task import generate_requests
        return generate_requests(req_rate, duration, task_names, seed, req_id_offset)
    else:
        raise ValueError(f"Unknown trace type: {trace_type}")
