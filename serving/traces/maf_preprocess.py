"""One-shot preprocessing of the Azure Functions trace into a compact
per-HashOwner per-minute matrix for a chosen day.

Run once:
    cd serving
    python -m traces.maf_preprocess        # defaults to day=1

Output (under traces/azurefunctions/preprocessed/):
    hashowner_day01.npz
        owner_ids : (n_owners,)            U64 string array
        minutes   : (n_owners, 1440)       int32 — invocations per minute
        n_req     : (n_owners,)            int64 — row sum of `minutes`

The experiment loads this with `load_hashowner_minutes(day)` — no CSV parse,
no parquet, no per-row Python.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = str(THIS_DIR / "azurefunctions")

MINUTES_PER_DAY = 1440
_MINUTE_COLS = [str(m) for m in range(1, MINUTES_PER_DAY + 1)]


def _csv_path(data_dir: str, day: int) -> Path:
    return Path(data_dir) / f"invocations_per_function_md.anon.d{day:02d}.csv"


def _npz_path(data_dir: str, day: int) -> Path:
    return Path(data_dir) / "preprocessed" / f"hashowner_day{day:02d}.npz"


def preprocess_day(day: int = 1, data_dir: str = DEFAULT_DATA_DIR) -> Path:
    """Aggregate one day's CSV to per-HashOwner per-minute counts → .npz."""
    csv = _csv_path(data_dir, day)
    if not csv.is_file():
        raise FileNotFoundError(f"missing {csv}")

    out = _npz_path(data_dir, day)
    if out.is_file():
        print(f"[preprocess] day {day:02d}: already cached → {out}")
        return out

    t0 = time.time()
    print(f"[preprocess] day {day:02d}: parsing {csv.name}…", flush=True)
    df = pd.read_csv(csv, dtype=str, low_memory=False)
    print(f"[preprocess]   read CSV in {time.time() - t0:.1f}s; rows={len(df)}", flush=True)

    t1 = time.time()
    present = [c for c in _MINUTE_COLS if c in df.columns]
    # Vectorised string → int32 over all minute columns at once.
    arr = df[present].fillna("0").replace("", "0").to_numpy()
    nums = arr.astype(np.int32)
    if nums.shape[1] < MINUTES_PER_DAY:
        pad = np.zeros((nums.shape[0], MINUTES_PER_DAY - nums.shape[1]), dtype=np.int32)
        nums = np.hstack([nums, pad])
    print(f"[preprocess]   converted to int32 in {time.time() - t1:.1f}s", flush=True)

    t2 = time.time()
    owners = df["HashOwner"].to_numpy()
    uniq, inv = np.unique(owners, return_inverse=True)
    summed = np.zeros((len(uniq), MINUTES_PER_DAY), dtype=np.int32)
    np.add.at(summed, inv, nums)
    n_req = summed.sum(axis=1, dtype=np.int64)
    print(f"[preprocess]   aggregated {len(uniq)} owners in {time.time() - t2:.1f}s", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    uniq_u = np.asarray(uniq, dtype="U64")   # fixed-width strings; npz-safe
    np.savez_compressed(out, owner_ids=uniq_u, minutes=summed, n_req=n_req)
    print(f"[preprocess] day {day:02d}: total {time.time() - t0:.1f}s; "
          f"saved → {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def load_hashowner_minutes(day: int = 1, data_dir: str = DEFAULT_DATA_DIR):
    """Load the cached per-owner per-minute matrix.

    Returns dict: {owner_ids, minutes, n_req}.
    """
    npz = _npz_path(data_dir, day)
    if not npz.is_file():
        raise FileNotFoundError(
            f"missing {npz}\nRun: python -m traces.maf_preprocess  (day={day})"
        )
    z = np.load(npz, allow_pickle=False)
    return {
        "owner_ids": z["owner_ids"],
        "minutes":   z["minutes"],
        "n_req":     z["n_req"],
    }


if __name__ == "__main__":
    day = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
    data_dir = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_DATA_DIR
    preprocess_day(day, data_dir)
