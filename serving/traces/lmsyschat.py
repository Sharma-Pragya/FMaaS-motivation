from datasets import load_dataset
import numpy as np
from typing import List, Tuple, Any,Dict
import math
class Request:
    def __init__(self, req_id, task, site_manager, device, req_time):
        self.req_id = req_id
        self.task = task
        self.site_manager = site_manager
        self.device = device
        self.req_time = req_time

    def __repr__(self):
        return f"req_id={self.req_id}, task={self.task}, site_manager={self.site_manager}, device={self.device}, req_time={self.req_time}"

    def to_dict(self):
        return {
            "req_id": self.req_id,
            "task": self.task,
            "site_manager": self.site_manager,
            "device": self.device,
            "req_time": self.req_time,
        }

def generate_requests(req_rate, duration, tasks, seed=42) -> List[Request]:
    rng = np.random.default_rng(seed)
    tot_req = int(req_rate * duration)

    tok = open("../hf-token.txt").read().strip()
    K = max(tot_req * 3, tot_req + 1000)  # grab a small prefix to stay fast & avoid OOM
    ds = load_dataset("lmsys/lmsys-chat-1m", split=f"train[:{K}]", token=tok)

    cols = ds.column_names
    model_col = "model" if "model" in cols else None
    time_col = "timestamp" if "timestamp" in cols else None
    if model_col is None:
        raise ValueError(f"missing model column; have {cols}")

    raw = []
    for i, ex in enumerate(ds):
        m = ex.get(model_col, None)
        if m is None:
            continue
        t = ex.get(time_col, None)
        t = float(t) if t is not None else float(i)
        raw.append((m, t))
        if len(raw) >= tot_req:
            break

    if len(raw) < tot_req:
        raise ValueError(f"not enough rows in prefix: have {len(raw)} need {tot_req}. Increase K.")

    raw.sort(key=lambda x: x[1])
    t0, t1 = raw[0][1], raw[-1][1]
    span = max(t1 - t0, 1e-12)
    times = [(t - t0) / span * duration for _, t in raw]

    model_to_task = {}
    assign_ptr = 0
    k =  len(tasks)
    reqs: List[Request] = []
    for i, ((m, _), ts) in enumerate(zip(raw, times)):
        if m not in model_to_task:
            model_to_task[m] = tasks[assign_ptr % k]
            assign_ptr += 1
        task_name, site_manager, device = model_to_task[m]
        reqs.append(Request(i, task_name, site_manager, device, float(ts)))
    # --- per-task mean and peak RPS (1-second bins) ---
    counts_per_task: Dict[str, int] = {}
    bins_per_task: Dict[str, Dict[int, int]] = {}

    for r in reqs:
        counts_per_task[r.task] = counts_per_task.get(r.task, 0) + 1
        sec = int(math.floor(r.req_time))
        d = bins_per_task.setdefault(r.task, {})
        d[sec] = d.get(sec, 0) + 1

    mean_rps_per_task: Dict[str, float] = {task: cnt / float(duration) for task, cnt in counts_per_task.items()}
    peak_rps_per_task: Dict[str, float] = {task: float(max(sec_counts.values())) for task, sec_counts in bins_per_task.items()} if bins_per_task else {}

    return reqs, mean_rps_per_task, peak_rps_per_task
