import argparse
import hashlib
import os
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


DEFAULT_TRACE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "alibaba_gentd26",
    "lora_request_trace.csv",
)

DEFAULT_GROUP_BY = "checkpoint_model_version_id"


class Request:
    def __init__(self, req_id, task, req_time):
        self.req_id = req_id
        self.task = task
        self.req_time = req_time

    def __repr__(self):
        return f"req_id={self.req_id}, task={self.task}, req_time={self.req_time}"

    def to_dict(self):
        return {"req_id": self.req_id, "task": self.task, "req_time": self.req_time}


def load_trace(path: str = DEFAULT_TRACE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["predict_status"] == "SUCCEED"].copy()
    df["gmt_create"] = pd.to_datetime(df["gmt_create"])
    df = df.sort_values("gmt_create").reset_index(drop=True)
    return df


def top_k_tasks(
    df: pd.DataFrame,
    k: int,
    group_by: str = DEFAULT_GROUP_BY,
) -> List[str]:
    return df[group_by].value_counts().head(k).index.tolist()


def _interarrivals_for_task(
    df: pd.DataFrame, task_name: str, group_by: str
) -> np.ndarray:
    t = df.loc[df[group_by] == task_name, "gmt_create"].sort_values()
    if len(t) < 2:
        return np.array([], dtype=float)
    return np.diff(t.astype("int64").to_numpy()) / 1e9


def generate_requests(
    req_rate: Union[float, List[float]],
    duration: float,
    task_names: List[str],
    seed: int = 42,
    req_id_offset: int = 0,
    tasks_dict: Optional[Dict] = None,
    trace_path: str = DEFAULT_TRACE_PATH,
    group_by: str = DEFAULT_GROUP_BY,
) -> tuple:
    """Replay the Alibaba GenTD26 per-model interarrival shape for each task.

    Mapping (Clockwork-style K-streams): each user task is assigned a block of
    K_STREAMS=9 consecutive Alibaba models from the pool (ranked by request
    count).  App at pool position P gets models [P*K, P*K+1, ..., P*K+K-1]
    (wrapping).  Each stream fires at rate/K and their arrivals merge under the
    same task label, superposing into one arrival process per task.

    Stability across N: because pool position is derived from the __appN suffix
    (or original rank) rather than sorted task name order, every task's model
    block is fixed regardless of how many other tasks exist.  Adding __appX
    clones never shifts the models assigned to the original tasks.

    CoV stability: by Palm-Khintchine, superposing K independent renewal
    processes reduces CoV compared to any single bursty stream.  Since every
    task always gets the same K-stream block, the merged CoV is identical for
    all N — eliminating the spurious latency decrease as N grows.

    Time-axis rescaling: each stream's interarrival sequence is scaled by
    (real_mean_ia * rate_k) so the mean RPS across K streams equals the target.
    Burst shape and autocorrelation are preserved.

    Args:
        req_rate:      Total req/s (float, split equally) or per-task list.
        duration:      Experiment duration in seconds.
        task_names:    User task identifiers (any strings; mapped positionally).
        seed:          RNG seed for per-task start offsets.
        req_id_offset: Starting request ID.
        tasks_dict:    Unused; accepted for signature parity.
        trace_path:    Path to lora_request_trace.csv.
        group_by:      Column to treat as "task" (model id or group id).

    Returns:
        (requests, mean_rps_per_task, peak_rps_per_task)
    """
    del tasks_dict  # unused, kept for signature parity

    if isinstance(req_rate, list):
        if len(req_rate) != len(task_names):
            raise ValueError("req_rate list length must match task_names")
        per_task_rate = dict(zip(task_names, req_rate))
    else:
        n = len(task_names)
        if n == 0 or req_rate <= 0 or duration <= 0:
            return [], {}, {}
        per_task_rate = {t: req_rate / float(n) for t in task_names}

    df = load_trace(trace_path)

    vc = df[group_by].value_counts()
    models_by_rank = vc[vc >= 2].index.tolist()
    if not models_by_rank:
        raise ValueError(f"trace has no tasks with >=2 requests under group_by={group_by!r}")

    ia_cache: Dict[str, np.ndarray] = {}

    def _ia(model_id: str) -> np.ndarray:
        if model_id not in ia_cache:
            ia_cache[model_id] = _interarrivals_for_task(df, model_id, group_by)
        return ia_cache[model_id]

    # K alibaba streams are merged per app (Clockwork-style).  Each stream fires
    # at rate/K and they share the same user task label, so their arrivals merge
    # into one superposed process.  The superposition of K independent renewal
    # streams converges toward Poisson (Palm-Khintchine), reducing CoV compared
    # to any single bursty stream.  Crucially, every app always gets the *same*
    # K consecutive models from the pool regardless of N, so the merged CoV is
    # identical for all N — fixing the spurious latency decrease across N.
    K_STREAMS = 9  # streams merged per app; covers the 74-model pool evenly

    events: List[tuple] = []
    counts: Dict[str, int] = defaultdict(int)
    bins: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    assignment: Dict[str, str] = {}

    # Assign pool positions by stable index, not sorted alphabetical order.
    # Clone tasks carry their pool index in the __appN suffix; original tasks
    # are ranked by their position among the sorted originals.  This keeps the
    # assignment stable across N: adding __appX clones never shifts the models
    # assigned to any original task, and each clone gets its own distinct block.
    originals_sorted = sorted(t for t in per_task_rate if "__app" not in t)
    orig_rank = {t: i for i, t in enumerate(originals_sorted)}

    def _pool_idx(task: str) -> int:
        if "__app" in task:
            return int(task.split("__app")[1])
        return orig_rank[task]

    def _stream_rng(task: str, k: int) -> np.random.Generator:
        # Deterministic per-(task, stream-index) seed so start offsets are
        # independent of how many other tasks or streams exist.
        key = f"{task}__stream{k}"
        digest = hashlib.sha256(key.encode()).hexdigest()
        task_seed = (seed + int(digest[:8], 16)) % (2 ** 32)
        return np.random.default_rng(task_seed)

    sorted_tasks = sorted(per_task_rate.keys())
    for user_task in sorted_tasks:
        rate = per_task_rate[user_task]
        if rate <= 0:
            continue

        base_idx = _pool_idx(user_task) * K_STREAMS
        # Record which model block this task is assigned to (first stream).
        assignment[user_task] = models_by_rank[base_idx % len(models_by_rank)]

        rate_k = rate / K_STREAMS
        for k in range(K_STREAMS):
            model_id = models_by_rank[(base_idx + k) % len(models_by_rank)]
            ia = _ia(model_id)
            if ia.size == 0:
                raise ValueError(
                    f"model {model_id!r} has <2 requests in trace "
                    f"(user task={user_task!r}, stream={k}, group_by={group_by!r})"
                )
            scale = ia.mean() * rate_k
            if scale <= 0:
                continue
            scaled_ia = ia / scale  # mean inter-arrival = 1/rate_k
            n_ia = len(scaled_ia)
            start = int(_stream_rng(user_task, k).integers(0, n_ia))

            t = 0.0
            idx = 0
            while t < duration:
                events.append((t, user_task))
                counts[user_task] += 1
                bins[user_task][int(t)] += 1
                t += float(scaled_ia[(start + idx) % n_ia])
                idx += 1

    events.sort(key=lambda x: x[0])
    requests = [
        Request(req_id_offset + i, name, rt) for i, (rt, name) in enumerate(events)
    ]

    mean_rps = {task: cnt / float(duration) for task, cnt in counts.items()}
    peak_rps = {task: float(max(v.values())) for task, v in bins.items()}
    return requests, mean_rps, peak_rps


def analyze(
    trace_path: str = DEFAULT_TRACE_PATH,
    group_by: str = DEFAULT_GROUP_BY,
    top_k: int = 10,
) -> None:
    df = load_trace(trace_path)
    dur_s = (df["gmt_create"].max() - df["gmt_create"].min()).total_seconds()
    print(f"trace: {trace_path}")
    print(f"rows (SUCCEED only): {len(df)}")
    print(f"time span: {df['gmt_create'].min()} -> {df['gmt_create'].max()}  ({dur_s/86400:.2f} days)")
    print(f"distinct tasks ({group_by}): {df[group_by].nunique()}")
    print(f"overall avg rps: {len(df)/dur_s:.4f}")
    print()
    print(f"Top-{top_k} tasks by {group_by}:")
    print(f"  {'task':<40} {'n_req':>8} {'rps':>10} {'mean_ia(s)':>12} {'cov_ia':>8}")
    for name in top_k_tasks(df, top_k, group_by):
        ia = _interarrivals_for_task(df, name, group_by)
        n = len(ia) + 1
        rps = n / dur_s
        mean_ia = ia.mean() if ia.size else float("nan")
        cov = ia.std() / mean_ia if ia.size and mean_ia > 0 else float("nan")
        print(f"  {str(name):<40} {n:>8d} {rps:>10.4f} {mean_ia:>12.2f} {cov:>8.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--trace-path", default=DEFAULT_TRACE_PATH)
    p.add_argument("--group-by", default=DEFAULT_GROUP_BY,
                   choices=["checkpoint_model_version_id", "groupId"])
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    if args.analyze:
        analyze(args.trace_path, args.group_by, args.top_k)
    else:
        p.print_help()
