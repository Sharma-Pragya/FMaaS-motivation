import math
from typing import Dict, List, Union

import numpy as np
import pandas as pd


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
    """Generate a trace from the Chatbot Arena dataset.

    Distinct models are mapped round-robin to task names. Timestamps are
    rescaled to [0, duration].

    If req_rate is a float: total requests = req_rate * duration.
    If req_rate is a list: total requests = sum(req_rate) * duration.

    Args:
        req_rate:      Total req/s (float) or per-task req/s (list).
        duration:      Experiment duration in seconds.
        task_names:    List of task name strings.
        seed:          RNG seed.
        req_id_offset: Starting request ID.

    Returns:
        (requests, mean_rps_per_task, peak_rps_per_task)
    """
    np.random.seed(seed)

    total_rate = sum(req_rate) if isinstance(req_rate, list) else req_rate
    tot_req = int(total_rate * duration)

    df = pd.read_parquet("traces/data/train-00000-of-00001-cced8514c7ed782a.parquet")

    time_col = 'tstamp'
    df = df[[time_col, 'model_a', 'model_b']].dropna(subset=[time_col])
    left = df[[time_col, 'model_a']].rename(columns={'model_a': 'model'})
    right = df[[time_col, 'model_b']].dropna().rename(columns={'model_b': 'model'})
    dfm = pd.concat([left, right], ignore_index=True).dropna(subset=['model'])
    dfm = dfm.sort_values(time_col).reset_index(drop=True)

    if len(dfm) < tot_req:
        raise ValueError(f"not enough rows: have {len(dfm)} need {tot_req}")

    samp = dfm.sample(n=tot_req, replace=False, random_state=seed).sort_values(time_col).reset_index(drop=True)
    t0 = float(samp[time_col].iloc[0])
    t1 = float(samp[time_col].iloc[-1])
    span = max(t1 - t0, 1e-12)
    samp['t'] = (samp[time_col].astype(float) - t0) / span * duration

    k = len(task_names)
    uniq_models = samp['model'].drop_duplicates().tolist()
    model_to_task = {m: task_names[i % k] for i, m in enumerate(uniq_models)}

    reqs: List[Request] = []
    for i, row in samp.iterrows():
        reqs.append(Request(req_id_offset + i, model_to_task[row['model']], float(row['t'])))

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
