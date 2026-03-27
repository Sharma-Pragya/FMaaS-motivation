import math
from typing import Dict, List, Union

import numpy as np
from datasets import load_dataset


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
    """Generate a trace from the LMSys chat dataset.

    Distinct models are mapped round-robin to task names. Timestamps are
    rescaled to [0, duration].

    If req_rate is a float: total requests = req_rate * duration, split
    equally across tasks via round-robin model assignment.
    If req_rate is a list: req_rate[i] maps to task_names[i]; total
    requests = sum(req_rate) * duration.

    Args:
        req_rate:      Total req/s (float) or per-task req/s (list).
        duration:      Experiment duration in seconds.
        task_names:    List of task name strings.
        seed:          RNG seed.
        req_id_offset: Starting request ID.

    Returns:
        (requests, mean_rps_per_task, peak_rps_per_task)
    """
    rng = np.random.default_rng(seed)

    total_rate = sum(req_rate) if isinstance(req_rate, list) else req_rate
    tot_req = int(total_rate * duration)

    tok = open("../hf-token.txt").read().strip()
    K = max(tot_req * 3, tot_req + 1000)
    ds = load_dataset("lmsys/lmsys-chat-1m", split=f"train[:{K}]", token=tok)

    raw = []
    for i, ex in enumerate(ds):
        raw.append((ex["model"], float(i)))
        if len(raw) >= tot_req:
            break

    if len(raw) < tot_req:
        raise ValueError(f"not enough rows: have {len(raw)} need {tot_req}.")

    raw.sort(key=lambda x: x[1])
    t0, t1 = raw[0][1], raw[-1][1]
    span = max(t1 - t0, 1e-12)
    times = [(t - t0) / span * duration for _, t in raw]

    k = len(task_names)
    model_to_task = {}
    assign_ptr = 0
    reqs: List[Request] = []
    for i, ((m, _), ts) in enumerate(zip(raw, times)):
        if m not in model_to_task:
            model_to_task[m] = task_names[assign_ptr % k]
            assign_ptr += 1
        reqs.append(Request(req_id_offset + i, model_to_task[m], float(ts)))

    counts_per_task: Dict[str, int] = {}
    bins_per_task: Dict[str, Dict[int, int]] = {}
    for r in reqs:
        counts_per_task[r.task] = counts_per_task.get(r.task, 0) + 1
        sec = int(math.floor(r.req_time))
        d = bins_per_task.setdefault(r.task, {})
        d[sec] = d.get(sec, 0) + 1

    mean_rps: Dict[str, float] = {t: cnt / float(duration) for t, cnt in counts_per_task.items()}
    peak_rps: Dict[str, float] = {t: float(max(v.values())) for t, v in bins_per_task.items()} if bins_per_task else {}

    return reqs, mean_rps, peak_rps
