"""Microsoft Azure Functions 2019 trace adapter for the long-horizon experiment.

One Azure function per task.  The assigned function determines both the task's
activation window (arrive/depart) and the per-minute invocation counts within
that window.

Data layout
-----------
Download from:
  https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz

Extract the 14 invocations CSV files into serving/traces/maf/:
  invocations_per_function_md.anon.d01.csv  ...  d14.csv

CSV columns: HashOwner, HashApp, HashFunction, Trigger, 1, 2, ..., 1440
  - Each row is one function.
  - Columns 1–1440 are invocation counts for each minute of that day.
  - A function may appear in multiple day files.

Trace span: 14 days = 1,209,600 seconds.

Performance design
------------------
extract_timeline  — only needs (first_s, last_s, n_req) per function.
                    Uses a fast stats-only pass: vectorised numpy over the
                    wide (n_funcs × 1440) matrix, NO melt, NO per-row Python.

compress_function_requests — only needs per-minute counts for the N≈16
                    selected functions.  Loads all 14 day files but keeps
                    only those rows, so the working set stays tiny.

Both functions cache their intermediate results at module level so repeated
calls (e.g. from generate.py running both build_task_list and build_full_trace)
pay the I/O cost only once.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "azurefunctions",
)

DAYS              = 14
MINUTES_PER_DAY   = 1440
SECONDS_PER_MINUTE = 60
TRACE_SPAN_S      = float(DAYS * MINUTES_PER_DAY * SECONDS_PER_MINUTE)  # 1,209,600 s


def _trace_span_s(n_days: int) -> float:
    return float(n_days * MINUTES_PER_DAY * SECONDS_PER_MINUTE)

_MINUTE_COLS = [str(m) for m in range(1, MINUTES_PER_DAY + 1)]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _day_file(data_dir: str, day: int) -> Path:
    return Path(data_dir) / f"invocations_per_function_md.anon.d{day:02d}.csv"


def _cache_path(data_dir: str, day: int) -> Path:
    return Path(data_dir) / "cache" / f"day_{day:02d}.parquet"


def _load_day_df(data_dir: str, day: int) -> pd.DataFrame:
    """Return the day's DataFrame with minute columns as int32.

    First call per day parses the CSV (slow) and writes a parquet cache.
    Subsequent calls read parquet (fast).
    """
    cache = _cache_path(data_dir, day)
    if cache.is_file():
        return pd.read_parquet(cache)

    print(f"[maf] day {day:02d}: parsing CSV (first time, slow)…", flush=True)
    import time as _t
    t0 = _t.time()

    # Fast path: pyarrow C++ CSV reader, multi-threaded.
    try:
        import pyarrow.csv as _pv
        import pyarrow as _pa
        col_types = {c: _pa.int32() for c in _MINUTE_COLS}
        for c in ("HashOwner", "HashApp", "HashFunction", "Trigger"):
            col_types[c] = _pa.string()
        table = _pv.read_csv(
            str(_day_file(data_dir, day)),
            convert_options=_pv.ConvertOptions(
                column_types=col_types,
                null_values=[""],
                strings_can_be_null=True,
            ),
        )
        out = table.to_pandas()
        # Fill NaNs in minute cols (rare but possible) with 0 and ensure int32.
        present = [c for c in _MINUTE_COLS if c in out.columns]
        if present:
            out[present] = out[present].fillna(0).astype(np.int32, copy=False)
    except ImportError:
        # Fallback: pandas CSV + per-column to_numeric (slow but works).
        df = pd.read_csv(_day_file(data_dir, day), dtype=str, low_memory=False)
        present = [c for c in _MINUTE_COLS if c in df.columns]
        parts = [df[[c for c in df.columns if c not in present]]]
        if present:
            nums = df[present].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.int32)
            parts.append(nums)
        out = pd.concat(parts, axis=1).copy()

    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, compression="snappy")
    print(f"[maf] day {day:02d}: parsed in {_t.time() - t0:.1f}s → {cache}", flush=True)
    return out


def preprocess_all(data_dir: str = DEFAULT_DATA_DIR) -> None:
    """Parse every day's CSV once and cache as parquet.

    Run this once to make subsequent load_stats / load_function_data calls fast:
        python -m traces.maf preprocess
    """
    _check_files(data_dir)
    for day in range(1, DAYS + 1):
        cache = _cache_path(data_dir, day)
        if cache.is_file():
            print(f"[maf] day {day:02d}: cached ({cache.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"[maf] day {day:02d}: preprocessing…", flush=True)
        _load_day_df(data_dir, day)
        print(f"[maf] day {day:02d}: saved → {cache}")


def _check_files(data_dir: str) -> None:
    missing = [_day_file(data_dir, d) for d in range(1, DAYS + 1)
               if not _day_file(data_dir, d).is_file()]
    if missing:
        raise FileNotFoundError(
            f"MAF data not found in {data_dir!r}.\n"
            "Download from:\n"
            "  https://azurepublicdatasettraces.blob.core.windows.net/"
            "azurepublicdatasetv2/azurefunctions_dataset2019/"
            "azurefunctions-dataset2019.tar.xz\n"
            f"Extract invocations_per_function_md.anon.d*.csv into {data_dir!r}."
        )


# ── Fast stats pass (no melt) ─────────────────────────────────────────────────

# Module-level cache keyed by (data_dir, group_by, n_days) so multiple
# calls in the same process are free after the first.
_stats_cache: Dict[Tuple[str, str, int], List[dict]] = {}


def load_stats(
    data_dir: str = DEFAULT_DATA_DIR,
    group_by: str = "HashFunction",
    n_days: int = DAYS,
) -> List[dict]:
    """Return per-group stats without melting any minute columns.

    group_by selects the grouping column: "HashFunction" (default) or
    "HashOwner".  When grouping by HashOwner, multiple rows per day are
    summed via the cross-day accumulator (min first_s, max last_s, sum n_req).

    Uses fully vectorised numpy over the wide (n_groups × 1440) matrix.
    Result is sorted by first_s.

    Each entry: {function_id, first_s, last_s, n_req}  (function_id holds
    the group identifier — kept named for backward compatibility).
    """
    cache_key = (data_dir, group_by, n_days)
    if cache_key in _stats_cache:
        return _stats_cache[cache_key]

    _check_files(data_dir)
    n_days = max(1, min(int(n_days), DAYS))

    # Accumulate across days and across same-day rows of the same group:
    # group_id → [first_s, last_s, n_req]
    acc: Dict[str, list] = {}

    for day in range(1, n_days + 1):
        day_offset_s = float((day - 1) * MINUTES_PER_DAY * SECONDS_PER_MINUTE)

        df = _load_day_df(data_dir, day)
        func_ids = df[group_by].values

        # Minute columns are already int32 (preprocessed parquet).
        present = [c for c in _MINUTE_COLS if c in df.columns]
        m_arr = df[present].values.astype(np.int32, copy=False)

        # Fill any missing minute columns with zeros on the right.
        if m_arr.shape[1] < MINUTES_PER_DAY:
            pad = np.zeros((m_arr.shape[0], MINUTES_PER_DAY - m_arr.shape[1]), dtype=np.int32)
            m_arr = np.hstack([m_arr, pad])

        totals = m_arr.sum(axis=1)          # (n_funcs,)
        active = np.where(totals > 0)[0]
        if active.size == 0:
            continue

        # argmax on bool matrix gives first True per row (0-indexed minute).
        bool_arr = m_arr[active] > 0
        first_mins = np.argmax(bool_arr, axis=1)                        # 0-indexed
        last_mins  = (MINUTES_PER_DAY - 1) - np.argmax(bool_arr[:, ::-1], axis=1)

        first_s_arr = day_offset_s + first_mins.astype(float) * SECONDS_PER_MINUTE
        last_s_arr  = day_offset_s + last_mins.astype(float)  * SECONDS_PER_MINUTE

        for i, idx in enumerate(active):
            fid = func_ids[idx]
            fs  = float(first_s_arr[i])
            ls  = float(last_s_arr[i])
            tot = int(totals[idx])
            if fid not in acc:
                acc[fid] = [fs, ls, tot]
            else:
                if fs < acc[fid][0]:
                    acc[fid][0] = fs
                if ls > acc[fid][1]:
                    acc[fid][1] = ls
                acc[fid][2] += tot

    stats = [
        {"function_id": fid, "first_s": v[0], "last_s": v[1], "n_req": v[2]}
        for fid, v in acc.items()
    ]
    stats.sort(key=lambda s: s["first_s"])
    _stats_cache[cache_key] = stats
    return stats


# ── Lazy per-group data (only for selected groups) ────────────────────────────

# Cache: (data_dir, group_by, n_days) → {group_id → DataFrame with columns [day, minute, count]}
_func_data_cache: Dict[Tuple[str, str, int], Dict[str, pd.DataFrame]] = {}


def load_function_data(
    function_ids: List[str],
    data_dir: str = DEFAULT_DATA_DIR,
    group_by: str = "HashFunction",
    n_days: int = DAYS,
) -> Dict[str, pd.DataFrame]:
    """Return per-minute counts for a small set of selected groups.

    group_by selects the grouping column: "HashFunction" (default) or
    "HashOwner".  When grouping by HashOwner, per-minute counts across all
    functions belonging to that owner are summed.

    Scans all 14 day files but keeps only rows matching the requested ids,
    so the working set is tiny regardless of how large the full trace is.

    Returns {group_id → DataFrame(day, minute, count)} with count > 0.
    """
    _check_files(data_dir)
    n_days = max(1, min(int(n_days), DAYS))

    cache = _func_data_cache.setdefault((data_dir, group_by, n_days), {})
    needed = [fid for fid in function_ids if fid not in cache]
    if not needed:
        return {fid: cache[fid] for fid in function_ids}

    needed_set = set(needed)
    frames: Dict[str, List[pd.DataFrame]] = {fid: [] for fid in needed}

    for day in range(1, n_days + 1):
        df = _load_day_df(data_dir, day)
        df = df[df[group_by].isin(needed_set)]
        if df.empty:
            continue

        present = [c for c in _MINUTE_COLS if c in df.columns]
        m_df = df[[group_by] + present]

        # When grouping by HashOwner, multiple rows may share the same owner —
        # sum their per-minute counts before melting so an owner's trace is the
        # union of all its functions.  Aggregate via numpy to avoid the
        # fragmented-DataFrame slow path on 1440 columns.
        if group_by != "HashFunction":
            ids = m_df[group_by].values
            arr = m_df[present].values.astype(np.int32, copy=False)
            uniq, inv = np.unique(ids, return_inverse=True)
            summed = np.zeros((len(uniq), arr.shape[1]), dtype=np.int32)
            np.add.at(summed, inv, arr)
            m_df = pd.DataFrame(summed, columns=present)
            m_df.insert(0, group_by, uniq)

        melted = m_df.melt(id_vars=group_by, var_name="minute", value_name="count")
        melted = melted[melted["count"] > 0].copy()
        melted["day"]    = day
        melted["minute"] = melted["minute"].astype(int)

        for fid, grp in melted.groupby(group_by):
            if fid in frames:
                frames[fid].append(grp[["day", "minute", "count"]])

    for fid in needed:
        parts = frames[fid]
        cache[fid] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["day", "minute", "count"]
        )

    return {fid: cache[fid] for fid in function_ids}


# ── Timeline extraction ───────────────────────────────────────────────────────

def extract_timeline(
    n_tasks: int,
    duration: float,
    idle_timeout_real_s: float = 300.0,
    min_invocations: int = 50,
    seed: int = 42,
    data_dir: str = DEFAULT_DATA_DIR,
    group_by: str = "HashFunction",
    n_days: int = DAYS,
) -> List[Dict]:
    """Select n_tasks Azure groups and compute time-compressed arrive/depart.

    group_by selects the grouping column: "HashFunction" (default) or
    "HashOwner".  For HashOwner, each task corresponds to one owner (a
    coarser, higher-volume aggregate than a single function).

    Uses the fast stats-only pass (no melt).  Functions with fewer than
    min_invocations total are excluded.  Selection uses evenly-spaced indices
    across the first-invocation distribution (same strategy as
    alibaba_long_horizon.extract_timeline).

    Returns a list of n_tasks dicts sorted by arrive time:
        {'function_id': str, 'arrive': float, 'depart': float}
    where arrive/depart are in seconds within [0, duration].
    """
    stats = load_stats(data_dir, group_by=group_by, n_days=n_days)
    stats = [s for s in stats if s["n_req"] >= min_invocations]

    if len(stats) < n_tasks:
        raise ValueError(
            f"Only {len(stats)} MAF {group_by} groups have >= {min_invocations} "
            f"invocations, but n_tasks={n_tasks} requested. "
            "Lower min_invocations or reduce n_tasks."
        )

    scale             = duration / _trace_span_s(n_days)
    idle_compressed   = idle_timeout_real_s * scale
    total             = len(stats)

    rng       = np.random.default_rng(seed)
    base_idx  = np.linspace(0, total - 1, n_tasks).round().astype(int)
    jitter    = rng.integers(-2, 3, size=n_tasks)
    candidates = np.clip(base_idx + jitter, 0, total - 1).tolist()

    seen: set = set()
    selected_idx: List[int] = []
    for i in candidates:
        while i in seen and i < total - 1:
            i += 1
        seen.add(i)
        selected_idx.append(i)

    selected = [stats[i] for i in selected_idx]
    selected.sort(key=lambda s: s["first_s"])

    result = []
    for s in selected:
        arrive = s["first_s"] * scale
        depart = min(s["last_s"] * scale + idle_compressed, duration)
        result.append({
            "function_id": s["function_id"],
            "arrive":      round(arrive, 2),
            "depart":      round(depart, 2),
        })
    return result


# ── Request generation (only for selected functions) ─────────────────────────

def _function_timestamps(func_df: pd.DataFrame, rng_seed: int) -> np.ndarray:
    """Reconstruct per-invocation real timestamps from per-minute counts.

    Vectorised: repeats each minute's start time by its count, then adds
    uniform offsets within the 60-second window.  After compression a
    60-second real window shrinks to ~0.02 s so sub-minute placement is
    negligible, but we keep it for faithfulness.
    """
    if func_df.empty:
        return np.array([], dtype=float)

    t_starts = (
        (func_df["day"].values   - 1) * MINUTES_PER_DAY * SECONDS_PER_MINUTE
        + (func_df["minute"].values - 1) * SECONDS_PER_MINUTE
    ).astype(float)
    counts = func_df["count"].values.astype(int)

    repeated_starts = np.repeat(t_starts, counts)
    rng     = np.random.default_rng(rng_seed)
    offsets = rng.uniform(0.0, float(SECONDS_PER_MINUTE), size=repeated_starts.size)
    return np.sort(repeated_starts + offsets)


def compress_function_requests(
    task_entries: List[Dict],
    compressed_duration: float,
    req_id_offset: int = 0,
    data_dir: str = DEFAULT_DATA_DIR,
    group_by: str = "HashFunction",
    n_days: int = DAYS,
) -> List[dict]:
    """Generate requests by time-compressing real MAF invocation timestamps.

    Analogous to alibaba_long_horizon.compress_model_requests.

    Each task entry must have keys: task, function_id, arrive, depart.
    Only loads per-minute data for the selected functions — O(N_selected),
    not O(all functions in trace).

    Returns list of {req_id, task, req_time} dicts sorted by req_time.
    """
    function_ids = [e["function_id"] for e in task_entries]
    func_data    = load_function_data(function_ids, data_dir,
                                      group_by=group_by, n_days=n_days)
    scale        = compressed_duration / _trace_span_s(n_days)

    events: List[dict] = []
    req_id = req_id_offset

    for entry in task_entries:
        task_name   = entry["task"]
        function_id = entry["function_id"]
        arrive      = float(entry["arrive"])
        depart      = float(entry["depart"])

        seed = int(hashlib.sha256(function_id.encode()).hexdigest()[:8], 16) % (2**32)
        real_times = _function_timestamps(func_data[function_id], seed)

        for t_real in real_times:
            t_comp = t_real * scale
            if arrive <= t_comp < depart:
                events.append({
                    "req_id":   req_id,
                    "task":     task_name,
                    "req_time": round(t_comp, 4),
                })
                req_id += 1

    events.sort(key=lambda r: r["req_time"])
    return events


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "preprocess":
        data_dir = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_DATA_DIR
        preprocess_all(data_dir)
    else:
        print("Usage:\n  python -m traces.maf preprocess [data_dir]")
        sys.exit(1)
