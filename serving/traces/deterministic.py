import math
from typing import Dict, List, Union


class Request:
    def __init__(self, req_id, task, req_time):
        self.req_id = req_id
        self.task = task
        self.req_time = req_time

    def __repr__(self):
        return f"req_id={self.req_id}, task={self.task}, req_time={self.req_time}"

    def to_dict(self):
        return {
            "req_id": self.req_id,
            "task": self.task,
            "req_time": self.req_time,
        }


def generate_requests(
    req_rate: Union[float, List[float]],
    duration: float,
    task_names: List[str],
    seed: int = 42,
    req_id_offset: int = 0,
) -> tuple:
    """Generate a perfectly deterministic (D/D/1) workload trace.

    If req_rate is a float: total rate split equally across all tasks,
    arrivals interleaved by time.
    If req_rate is a list: req_rate[i] maps to task_names[i], each task
    generates its own evenly-spaced arrivals independently, then merged.

    Args:
        req_rate:      Total req/s (float) or per-task req/s (list).
        duration:      Experiment duration in seconds.
        task_names:    List of task name strings.
        seed:          Ignored (deterministic trace needs no RNG).
        req_id_offset: Starting request ID.

    Returns:
        (requests, mean_rps_per_task, peak_rps_per_task)
    """
    if isinstance(req_rate, list):
        per_task_rate = {task_names[i]: req_rate[i] for i in range(len(task_names))}
    else:
        k = len(task_names)
        per_task_rate = {t: req_rate / k for t in task_names}

    events = []
    for task_name, rate in per_task_rate.items():
        if rate <= 0:
            continue
        interval = 1.0 / rate
        n = int(rate * duration)
        for i in range(n):
            events.append(((i + 1) * interval, task_name))

    events.sort(key=lambda x: x[0])

    requests: List[Request] = []
    counts: Dict[str, int] = {}
    bins: Dict[str, Dict[int, int]] = {}
    for i, (req_time, task_name) in enumerate(events):
        requests.append(Request(req_id_offset + i, task_name, req_time))
        counts[task_name] = counts.get(task_name, 0) + 1
        sec = int(math.floor(req_time))
        bins.setdefault(task_name, {})[sec] = bins.get(task_name, {}).get(sec, 0) + 1

    mean_rps = {t: cnt / float(duration) for t, cnt in counts.items()}
    peak_rps = {t: float(max(v.values())) for t, v in bins.items()} if bins else {}

    return requests, mean_rps, peak_rps
