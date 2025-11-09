from typing import List, Tuple, Dict, Any
import math
import numpy as np
import pandas as pd

class Request:
    def __init__(self, req_id, task, site_manager, device, backbone, req_time):
        self.req_id = req_id
        self.task = task
        self.site_manager = site_manager
        self.device = device
        self.backbone=backbone
        self.req_time = req_time
    def __repr__(self):
        return f"req_id={self.req_id}, task={self.task}, site_manager={self.site_manager}, device={self.device}, backbone={self.backbone}, req_time={self.req_time}"
    def to_dict(self):
        return {"req_id": self.req_id, "task": self.task, "site_manager": self.site_manager, "device": self.device, "backbone":self.backbone, "req_time": self.req_time}

def generate_requests(req_rate:float | None, duration: float | None,
                                 tasks: List[Tuple[str,str,str]],
                                 seed: int = 42) -> Tuple[List[Request], Dict[str,float], Dict[str,float]]:
    np.random.seed(seed)
    tot_req = int(req_rate * duration)

    # Use Chatbot Arena Conversations (has timestamps)
    df = pd.read_parquet("traces/data/train-00000-of-00001-cced8514c7ed782a.parquet")

    # Pick columns
    time_col = 'tstamp'
    model_a  = 'model_a'
    model_b  = 'model_b'
    # Expand to one row per (model, timestamp)
    cols = [time_col, model_a] + ([model_b] if model_b else [])
    df = df[cols].dropna(subset=[time_col])
    left = df[[time_col, model_a]].rename(columns={model_a: "model"})
    parts = [left]
    if model_b:
        right = df[[time_col, model_b]].dropna().rename(columns={model_b: "model"})
        parts.append(right)
    dfm = pd.concat(parts, ignore_index=True).dropna(subset=["model"])
    dfm = dfm.sort_values(time_col).reset_index(drop=True)

    print(f"Total rows after expansion: {len(dfm)}")
    if len(dfm) < tot_req:
        raise ValueError(f"not enough rows after filtering: have {len(dfm)} need {tot_req}")

    # Sample N, sort by time, rescale to [0, duration]
    samp = dfm.sample(n=tot_req, replace=False, random_state=seed).sort_values(time_col).reset_index(drop=True)
    t0 = float(samp[time_col].iloc[0]); t1 = float(samp[time_col].iloc[-1])
    span = max(t1 - t0, 1e-12)
    samp["t"] = (samp[time_col].astype(float) - t0) / span * duration

    # Map distinct models -> tasks round-robin
    uniq_models = samp["model"].drop_duplicates().tolist()
    k = len(tasks)
    model_to_task = {m: tasks[i % k] for i, m in enumerate(uniq_models)}

    # Build requests
    reqs: List[Request] = []
    for i, row in samp.iterrows():
        task_name, site_manager, device, backbone = model_to_task[row["model"]]
        reqs.append(Request(i, task_name, site_manager, device, backbone, float(row["t"])))

    # Per-task mean & peak workload (RPS using 1s bins over [0, duration])
    counts_per_task: Dict[str, int] = {}
    bins_per_task: Dict[str, Dict[int, int]] = {}
    for r in reqs:
        counts_per_task[r.task] = counts_per_task.get(r.task, 0) + 1
        sec = int(math.floor(r.req_time))
        d = bins_per_task.setdefault(r.task, {})
        d[sec] = d.get(sec, 0) + 1

    mean_rps_per_task = {task: cnt / float(duration) for task, cnt in counts_per_task.items()}
    peak_rps_per_task = {task: float(max(sec_counts.values())) for task, sec_counts in bins_per_task.items()} if bins_per_task else {}

    return reqs, mean_rps_per_task, peak_rps_per_task