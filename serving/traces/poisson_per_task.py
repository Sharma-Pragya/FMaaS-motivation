import hashlib
import math
from collections import defaultdict
from typing import Dict, List, Union

import numpy as np


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


def _seed_for_task(task_name: str, seed: int) -> int:
    digest = hashlib.sha256(task_name.encode("utf-8")).hexdigest()
    task_hash = int(digest[:8], 16)
    return (seed + task_hash) % (2**32)


def generate_requests(
    req_rate: Union[float, List[float]],
    duration: float,
    task_names: List[str],
    seed: int = 42,
    req_id_offset: int = 0,
) -> tuple:
    """Generate independent Poisson traces for each task.

    If req_rate is a float: total rate split equally across all tasks.
    If req_rate is a list: req_rate[i] maps to task_names[i].

    Each task uses a fixed task-specific RNG seed so its trace is
    reproducible and independent of other tasks.

    Args:
        req_rate:      Total req/s (float) or per-task req/s (list).
        duration:      Experiment duration in seconds.
        task_names:    List of task name strings.
        seed:          Base RNG seed.
        req_id_offset: Starting request ID.

    Returns:
        (requests, mean_rps_per_task, peak_rps_per_task)
    """
    if isinstance(req_rate, list):
        per_task_rate = {task_names[i]: req_rate[i] for i in range(len(task_names))}
    else:
        n_tasks = len(task_names)
        if n_tasks == 0 or req_rate <= 0 or duration <= 0:
            return [], {}, {}
        per_task_rate = {t: req_rate / float(n_tasks) for t in task_names}

    events = []
    counts: Dict[str, int] = defaultdict(int)
    bins: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for task_name, rate in per_task_rate.items():
        if rate <= 0:
            continue
        rng = np.random.default_rng(_seed_for_task(task_name, seed))
        t = 0.0
        while t < duration:
            events.append((t, task_name))
            counts[task_name] += 1
            bins[task_name][int(math.floor(t))] += 1
            t += float(rng.exponential(1.0 / rate))

    events.sort(key=lambda x: x[0])

    requests: List[Request] = []
    for i, (req_time, task_name) in enumerate(events):
        requests.append(Request(req_id_offset + i, task_name, req_time))

    mean_rps = {task: cnt / float(duration) for task, cnt in counts.items()}
    peak_rps = {task: float(max(v.values())) for task, v in bins.items()}

    return requests, mean_rps, peak_rps
