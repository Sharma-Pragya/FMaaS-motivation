"""Alibaba GenTD26 trace adapter for the long-horizon experiment.

One Alibaba model per task (vs K_STREAMS=9 in the cluster-sharing variant).
The assigned model determines both the task's activation window (arrive/depart)
and the inter-arrival burstiness within that window.

Two public entry points
-----------------------
extract_timeline(...)
    Called once by deployments/generate.py to select N models from the trace
    and compute time-compressed arrive/depart times for each task.

generate_requests(...)
    Called by _partial_trace() in run.py for each active window.
    Reads each task's 'model_id' from tasks_dict and replays that model's
    real inter-arrival shape, scaled to the target per_app_rps.
"""

import hashlib
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from traces.alibaba_gentd26 import (
    DEFAULT_GROUP_BY,
    DEFAULT_TRACE_PATH,
    Request,
    _interarrivals_for_task,
    load_trace,
)
from collections import defaultdict


def extract_timeline(
    n_tasks: int,
    duration: float,
    idle_timeout_real_s: float = 300.0,
    min_window_s: float = 30.0,
    seed: int = 42,
    trace_path: str = DEFAULT_TRACE_PATH,
    group_by: str = DEFAULT_GROUP_BY,
) -> List[Dict]:
    """Select n_tasks Alibaba models and compute time-compressed arrive/depart.

    Models are selected by spreading evenly across the first-invocation
    distribution so the result naturally mixes initially-active tasks (models
    that appear early in the trace) with dynamically-arriving ones (models that
    first appear mid-trace).

    min_window_s: minimum compressed active window in seconds.  Models whose
        compressed [arrive, depart) span is shorter than this are excluded from
        the timing pool before linspace selection.  This filters out single-burst
        artifacts (models with 1–2 requests clustered in a few real seconds) that
        would produce near-empty task windows after time compression.
        Set to 0 to disable filtering.

    Returns a list of n_tasks dicts sorted by arrive time:
        {'model_id': str, 'arrive': float, 'depart': float}
    where arrive/depart are in seconds within [0, duration].
    """
    df = load_trace(trace_path)

    stats = []
    for model_id, grp in df.groupby(group_by):
        times = grp["gmt_create"].sort_values()
        if len(times) < 2:
            continue
        stats.append(
            {
                "model_id": model_id,
                "first": times.iloc[0],
                "last": times.iloc[-1],
                "n_req": len(times),
            }
        )

    # Sort by first invocation time so index position ≈ arrival order.
    stats.sort(key=lambda s: s["first"])

    # Compute compression scale up-front so we can filter by compressed window.
    trace_start = stats[0]["first"]
    trace_end   = stats[-1]["last"]
    trace_span  = (trace_end - trace_start).total_seconds()
    if trace_span <= 0:
        raise ValueError("Trace has zero time span.")
    scale            = duration / trace_span
    idle_compressed  = idle_timeout_real_s * scale

    # Filter: exclude models whose compressed active window is too short.
    if min_window_s > 0:
        filtered = []
        for s in stats:
            arrive = (s["first"] - trace_start).total_seconds() * scale
            depart = min(
                (s["last"] - trace_start).total_seconds() * scale + idle_compressed,
                duration,
            )
            if depart - arrive >= min_window_s:
                filtered.append(s)
        stats = filtered

    if len(stats) == 0:
        raise ValueError("No Alibaba models remain after min_window_s filter")

    total = len(stats)

    rng = np.random.default_rng(seed)

    if total >= n_tasks:
        # Original logic: select unique models
        base_idx = np.linspace(0, total - 1, n_tasks).round().astype(int)
        jitter = rng.integers(-2, 3, size=n_tasks)
        candidates = np.clip(base_idx + jitter, 0, total - 1).tolist()

        seen: set = set()
        selected_idx: List[int] = []
        for i in candidates:
            while i in seen and i < total - 1:
                i += 1
            seen.add(i)
            selected_idx.append(i)
    else:
        # New logic: allow duplicates when there are too few unique models
        print(f"[extract_timeline] only {total} unique models available, "
              f"repeating models to reach n_tasks={n_tasks}")
        repeats = int(np.ceil(n_tasks / total))
        selected_idx = np.tile(np.arange(total), repeats)[:n_tasks].tolist()
        rng.shuffle(selected_idx)

    selected = [stats[i] for i in selected_idx]
    selected.sort(key=lambda s: s["first"])

    usage_counts: Dict[str, int] = defaultdict(int)
    result = []
    for s in selected:
        arrive = (s["first"] - trace_start).total_seconds() * scale
        depart = min(
            (s["last"] - trace_start).total_seconds() * scale + idle_compressed,
            duration,
        )

        if usage_counts[s["model_id"]] > 0:
            # Add jitter for duplicate models
            max_shift = max(1.0, min_window_s * 0.25)
            shift = float(rng.uniform(-max_shift, max_shift))
            arrive = max(0.0, min(arrive + shift, duration - min_window_s))
            depart = max(arrive + min_window_s,
                         min(depart + shift, duration))

        usage_counts[s["model_id"]] += 1

        result.append(
            {
                "model_id": s["model_id"],
                "arrive": round(arrive, 2),
                "depart": round(depart, 2),
            }
        )

    return result


def compress_model_requests(
    task_entries: List[Dict],
    compressed_duration: float,
    req_id_offset: int = 0,
    trace_path: str = DEFAULT_TRACE_PATH,
    group_by: str = DEFAULT_GROUP_BY,
) -> List[dict]:
    """Generate requests by directly compressing real Alibaba timestamps.

    Each task entry must have keys: task, model_id, arrive, depart.
    Real request timestamps are mapped to [0, compressed_duration] using the
    same global scale factor as extract_timeline, then filtered to
    [task.arrive, task.depart).

    Returns list of {req_id, task, req_time} dicts sorted by req_time.
    """
    df = load_trace(trace_path)

    # Compute global compression params (same formula as extract_timeline)
    all_first: list = []
    all_last:  list = []
    for _, grp in df.groupby(group_by):
        times = grp["gmt_create"].sort_values()
        if len(times) < 2:
            continue
        all_first.append(times.iloc[0])
        all_last.append(times.iloc[-1])
    if not all_first:
        return []

    trace_start = min(all_first)
    trace_end   = max(all_last)
    trace_span  = (trace_end - trace_start).total_seconds()
    if trace_span <= 0:
        return []
    scale = compressed_duration / trace_span

    events: List[dict] = []
    req_id = req_id_offset

    for entry in task_entries:
        task_name = entry["task"]
        model_id  = entry["model_id"]
        arrive    = float(entry["arrive"])
        depart    = float(entry["depart"])

        model_times = df.loc[df[group_by] == model_id, "gmt_create"].sort_values()
        for t_real in model_times:
            t_comp = (t_real - trace_start).total_seconds() * scale
            if arrive <= t_comp < depart:
                events.append({
                    "req_id":   req_id,
                    "task":     task_name,
                    "req_time": round(t_comp, 4),
                })
                req_id += 1

    events.sort(key=lambda r: r["req_time"])
    return events


def windowed_replay_requests(
    task_entries: List[Dict],
    compressed_duration: float,
    req_id_offset: int = 0,
    seed: int = 42,
    trace_path: str = DEFAULT_TRACE_PATH,
    group_by: str = DEFAULT_GROUP_BY,
) -> List[dict]:
    """Generate requests by replaying req_model_id inter-arrivals within each task's window.

    Each task entry must have keys: task, req_model_id, arrive, depart.
    Real inter-arrivals from req_model_id are multiplied by the global
    compression scale (compressed_duration / trace_span) and replayed
    cyclically within [arrive, depart), starting at a task-seeded random
    offset in the inter-arrival array.

    Returns list of {req_id, task, req_time} dicts sorted by req_time.
    """
    df = load_trace(trace_path)

    # Global compression scale (same formula as extract_timeline)
    all_first: list = []
    all_last:  list = []
    for _, grp in df.groupby(group_by):
        times = grp["gmt_create"].sort_values()
        if len(times) < 2:
            continue
        all_first.append(times.iloc[0])
        all_last.append(times.iloc[-1])
    if not all_first:
        return []

    trace_start = min(all_first)
    trace_end   = max(all_last)
    trace_span  = (trace_end - trace_start).total_seconds()
    if trace_span <= 0:
        return []
    ia_scale = compressed_duration / trace_span  # real seconds → compressed seconds

    ia_cache: Dict[str, np.ndarray] = {}

    def _ia(model_id: str) -> np.ndarray:
        if model_id not in ia_cache:
            ia_cache[model_id] = _interarrivals_for_task(df, model_id, group_by)
        return ia_cache[model_id]

    def _task_rng(task: str) -> np.random.Generator:
        digest = hashlib.sha256(f"{task}__lh".encode()).hexdigest()
        task_seed = (seed + int(digest[:8], 16)) % (2**32)
        return np.random.default_rng(task_seed)

    events: List[dict] = []
    req_id = req_id_offset

    for entry in task_entries:
        task_name = entry["task"]
        model_id  = entry.get("req_model_id") or entry.get("model_id")
        arrive    = float(entry["arrive"])
        depart    = float(entry["depart"])

        if model_id is None:
            continue

        real_ia = _ia(model_id)
        if real_ia.size == 0:
            continue

        compressed_ia = real_ia * ia_scale
        n_ia = len(compressed_ia)
        start = int(_task_rng(task_name).integers(0, n_ia))

        t = arrive
        idx = 0
        while t < depart:
            events.append({
                "req_id":   req_id,
                "task":     task_name,
                "req_time": round(t, 4),
            })
            req_id += 1
            t += float(compressed_ia[(start + idx) % n_ia])
            idx += 1

    events.sort(key=lambda r: r["req_time"])
    return events


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
    """Generate a trace using one Alibaba model per task.

    Each task's 'model_id' must be present in tasks_dict[task]['model_id'].
    Inter-arrival times are scaled to hit the target rate while preserving
    the model's real burst shape and autocorrelation.

    Returns (requests, mean_rps_per_task, peak_rps_per_task).
    """
    tasks_dict = tasks_dict or {}

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
    ia_cache: Dict[str, np.ndarray] = {}

    def _ia(model_id: str) -> np.ndarray:
        if model_id not in ia_cache:
            ia_cache[model_id] = _interarrivals_for_task(df, model_id, group_by)
        return ia_cache[model_id]

    def _task_rng(task: str) -> np.random.Generator:
        digest = hashlib.sha256(f"{task}__lh".encode()).hexdigest()
        task_seed = (seed + int(digest[:8], 16)) % (2**32)
        return np.random.default_rng(task_seed)

    events: List[tuple] = []
    counts: Dict[str, int] = {}
    bins: Dict[str, Dict[int, int]] = {}

    for task in sorted(task_names):
        rate = per_task_rate.get(task, 0.0)
        if rate <= 0:
            continue

        model_id = (tasks_dict.get(task) or {}).get("model_id")
        if model_id is None:
            raise ValueError(
                f"Task '{task}' has no 'model_id' in tasks_dict. "
                "Run deployments/generate.py first to assign Alibaba models."
            )

        ia = _ia(model_id)
        if ia.size == 0:
            raise ValueError(
                f"Model '{model_id}' has <2 requests in trace (task={task!r})"
            )

        scale = ia.mean() * rate  # mean inter-arrival becomes 1/rate
        if scale <= 0:
            continue
        scaled_ia = ia / scale
        n_ia = len(scaled_ia)
        start = int(_task_rng(task).integers(0, n_ia))

        task_counts = 0
        task_bins: Dict[int, int] = {}
        t = 0.0
        idx = 0
        while t < duration:
            events.append((t, task))
            task_counts += 1
            task_bins[int(t)] = task_bins.get(int(t), 0) + 1
            t += float(scaled_ia[(start + idx) % n_ia])
            idx += 1

        counts[task] = task_counts
        bins[task] = task_bins

    events.sort(key=lambda x: x[0])
    requests = [
        Request(req_id_offset + i, name, rt) for i, (rt, name) in enumerate(events)
    ]

    mean_rps = {t: cnt / float(duration) for t, cnt in counts.items()}
    peak_rps = {t: float(max(v.values())) if v else 0.0 for t, v in bins.items()}
    return requests, mean_rps, peak_rps
