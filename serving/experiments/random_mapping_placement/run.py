#!/usr/bin/env python3
"""Random workload-to-backbone placement sweep.

This is a placement-only experiment. It reuses the real-world MAF owner
selection and real placement schedulers, but it does not write deployment plans
for serving and does not start device servers.

Modes:
  fixed-n     randomly map N tasks, then measure how many tasks place
  admission   draw one task at a time and stop after both methods fail

Output:
  experiments/random_mapping_placement/outputs/run_<timestamp>/
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

from experiments.end_to_end_realworld_mix import user_config as uc
from experiments.end_to_end_realworld_mix.deployments import generate as gen

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
CONDITIONS = ("fmaas", "no_sharing")


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_task_counts(args) -> List[int]:
    if args.task_counts:
        return [int(x) for x in _parse_csv_list(args.task_counts)]
    return list(range(args.min_tasks, args.max_tasks + 1, args.step_tasks))


def _unique_pool_indices() -> List[int]:
    seen = set()
    out = []
    for i, entry in enumerate(uc.task_pool):
        backbone = entry["backbone"]
        if backbone in seen:
            continue
        seen.add(backbone)
        out.append(i)
    return out


def _build_random_task_list(owners: List[dict], duration: float,
                            seed: int) -> List[dict]:
    rng = np.random.default_rng(seed)
    candidates = _unique_pool_indices()
    base_counts: Dict[str, int] = defaultdict(int)
    tasks_out: List[dict] = []

    for owner in owners:
        bi = int(rng.choice(candidates))
        pool_entry = uc.task_pool[bi]
        base = pool_entry["task"]
        count = base_counts[base]
        base_counts[base] += 1

        bb_short = pool_entry["backbone"].replace("-patch", "")
        default_dec = f"{base}_{bb_short}_mlp"
        dec_path = pool_entry.get("decoder_path", default_dec)
        task_name = base if count == 0 else f"{base}__app{count}"

        tasks_out.append({
            "task": task_name,
            "base_task": base,
            "backbone": pool_entry["backbone"],
            "tier": pool_entry.get("tier", "small"),
            "decoder_path": dec_path,
            "arrive": 0.0,
            "depart": float(duration),
            "backbone_idx": bi,
            "function_id": owner["function_id"],
            "model_id": None,
            "req_model_id": None,
            "owner_n_req": owner["n_req"],
            "owner_rate_real": owner["rate_real"],
            "regime": owner.get("regime"),
            "window_cv": owner.get("window_cv"),
            "_minutes": owner["minutes"],
        })
    return tasks_out


def _task_rps_from_tasks(tasks: List[dict]) -> Dict[str, float]:
    return {
        t["task"]: max(float(t.get("owner_rate_real", 0.0)), 0.1)
        for t in tasks
    }


def _make_synthetic_owner_stream(regime: str, n_tasks: int,
                                 seed: int) -> Tuple[List[dict], dict]:
    bands = uc.experiment.get("rate_bands_req_per_s", {})
    if regime not in bands:
        raise ValueError(f"unknown regime {regime!r}; expected one of {list(bands)}")
    lo, hi = float(bands[regime][0]), float(bands[regime][1])
    duration = float(uc.experiment["duration"])
    n_minutes = max(1, int(duration // 60))
    min_count = max(1, int(math.floor(lo * duration)) + 1)
    max_count = max(min_count, int(math.floor(hi * duration)))
    rng = np.random.default_rng(seed)

    owners = []
    for i in range(n_tasks):
        n_req = int(rng.integers(min_count, max_count + 1))
        base = n_req // n_minutes
        rem = n_req % n_minutes
        minutes = np.full(n_minutes, base, dtype=np.int32)
        if rem:
            minutes[:rem] += 1
        rate = float(n_req / duration)
        owners.append({
            "function_id": f"synthetic_{regime}_{seed}_{i}",
            "minutes": minutes,
            "n_req": n_req,
            "rate_real": rate,
            "regime": regime,
            "window_cv": 0.0,
            "synthetic": True,
        })
    return owners, {
        "source": "synthetic",
        "regime": regime,
        "n_tasks": n_tasks,
        "rate_band_req_per_s": [lo, hi],
        "seed": seed,
    }


def _with_owner_source(window_info: dict, source: str) -> dict:
    info = dict(window_info)
    info["source"] = source
    return info


def _fallback_to_synthetic(regime: str, n_tasks: int, seed: int,
                           exc: Exception) -> Tuple[List[dict], dict]:
    print("[random-placement] real owner selection failed "
          f"({exc}); falling back to synthetic owners",
          flush=True)
    owners, window_info = _make_synthetic_owner_stream(regime, n_tasks, seed)
    window_info["fallback_reason"] = str(exc)
    return owners, window_info


def _make_fixed_owner_set(regime: str, n_tasks: int, seed: int,
                          owner_source: str) -> Tuple[List[dict], dict]:
    if owner_source == "synthetic":
        return _make_synthetic_owner_stream(regime, n_tasks, seed)

    try:
        owners, window_info = gen._select_owners_windowed_mix({regime: n_tasks})
        return owners, _with_owner_source(window_info, "real")
    except ValueError as exc:
        if owner_source != "auto":
            raise
        return _fallback_to_synthetic(regime, n_tasks, seed, exc)


def _make_owner_stream(regime: str, n_tasks: int, owner_pool_size: int,
                       seed: int, owner_source: str) -> Tuple[List[dict], dict]:
    """Sample a stream of owners from one load regime, with replacement."""
    if owner_source == "synthetic":
        return _make_synthetic_owner_stream(regime, n_tasks, seed)

    pool_size = max(1, min(owner_pool_size, n_tasks))
    try:
        owners, window_info = gen._select_owners_windowed_mix({regime: pool_size})
        window_info = _with_owner_source(window_info, "real")
    except ValueError as exc:
        if owner_source != "auto":
            raise
        return _fallback_to_synthetic(regime, n_tasks, seed, exc)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(owners), size=n_tasks)

    stream = []
    for pos, i in enumerate(idx):
        owner = dict(owners[int(i)])
        owner["function_id"] = f"{owner['function_id']}__draw{pos}"
        stream.append(owner)
    return stream, window_info


def _placed_names(condition: str, slots: list) -> set:
    if condition == "fmaas":
        return {t["task"] for s in slots for t in s["tasks"]}
    return {s["task"] for s in slots}


def _run_placement(condition: str, tasks: List[dict],
                   task_rps: Dict[str, float]) -> Tuple[set, str]:
    builders = {
        "fmaas": lambda: gen.build_fmaas_place(tasks, task_rps),
        "no_sharing": lambda: gen.build_clipper_place(tasks, task_rps),
    }
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _, slots = builders[condition]()
        return _placed_names(condition, slots), ""
    except Exception as e:
        return set(), str(e)


def _ci95(vals: List[float]) -> dict:
    if not vals:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    arr = np.asarray(vals, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    half = 1.96 * std / math.sqrt(arr.size) if arr.size > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _summarize(rows: List[dict]) -> Dict[str, dict]:
    summary = {}
    for condition in CONDITIONS:
        cur = [r for r in rows if r["condition"] == condition and not r["error"]]
        placed = [float(r["placed_count"]) for r in cur]
        rates = [float(r["placement_rate"]) for r in cur]
        summary[condition] = {
            "successful_trials": len(cur),
            "placed_count": _ci95(placed),
            "placement_rate": _ci95(rates),
            "min_placed": min(placed) if placed else 0.0,
            "max_placed": max(placed) if placed else 0.0,
        }
    return summary


def _write_trials_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "regime", "num_tasks", "trial", "seed", "condition", "requested",
        "placed_count", "rejected_count", "placement_rate", "error",
        "backbone_counts",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["backbone_counts"] = json.dumps(row["backbone_counts"],
                                                sort_keys=True)
            writer.writerow(row)


def _write_aggregate_summary(path: Path, scenario_summaries: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "regime", "num_tasks", "condition", "successful_trials",
        "placed_mean", "placed_std", "placed_ci95_low", "placed_ci95_high",
        "rate_mean", "rate_std", "rate_ci95_low", "rate_ci95_high",
        "min_placed", "max_placed",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scenario in scenario_summaries:
            for condition, stats in scenario["conditions"].items():
                placed = stats["placed_count"]
                rate = stats["placement_rate"]
                writer.writerow({
                    "regime": scenario["regime"],
                    "num_tasks": scenario["num_tasks"],
                    "condition": condition,
                    "successful_trials": stats["successful_trials"],
                    "placed_mean": placed["mean"],
                    "placed_std": placed["std"],
                    "placed_ci95_low": placed["ci95_low"],
                    "placed_ci95_high": placed["ci95_high"],
                    "rate_mean": rate["mean"],
                    "rate_std": rate["std"],
                    "rate_ci95_low": rate["ci95_low"],
                    "rate_ci95_high": rate["ci95_high"],
                    "min_placed": stats["min_placed"],
                    "max_placed": stats["max_placed"],
                })


def _write_admission_steps(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "regime", "trial", "seed", "prefix_n", "condition", "placed_count",
        "all_placed", "first_failure_n", "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_admission_trials(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "regime", "trial", "seed", "condition", "admitted_before_failure",
        "first_failure_n", "placed_at_failure", "censored", "max_tasks",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_admission_aggregate(path: Path, summaries: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "regime", "condition", "successful_trials", "capacity_mean",
        "capacity_std", "capacity_ci95_low", "capacity_ci95_high",
        "min_capacity", "max_capacity", "censored_trials",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scenario in summaries:
            for condition, stats in scenario["conditions"].items():
                cap = stats["admitted_before_failure"]
                writer.writerow({
                    "regime": scenario["regime"],
                    "condition": condition,
                    "successful_trials": stats["successful_trials"],
                    "capacity_mean": cap["mean"],
                    "capacity_std": cap["std"],
                    "capacity_ci95_low": cap["ci95_low"],
                    "capacity_ci95_high": cap["ci95_high"],
                    "min_capacity": stats["min_capacity"],
                    "max_capacity": stats["max_capacity"],
                    "censored_trials": stats["censored_trials"],
                })


def _run_scenario(regime: str, num_tasks: int, trials: int, seed: int,
                  owner_source: str,
                  out_dir: Path) -> Tuple[List[dict], dict]:
    mix = {regime: num_tasks}
    duration = float(uc.experiment["duration"])

    scenario_dir = out_dir / regime / f"ntasks_{num_tasks}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    print(f"[random-placement] regime={regime} ntasks={num_tasks} "
          f"trials={trials} owner_source={owner_source}")
    owners, window_info = _make_fixed_owner_set(
        regime, num_tasks, seed, owner_source)

    rows = []
    for trial in range(trials):
        trial_seed = int(seed + trial)
        tasks = _build_random_task_list(owners, duration, trial_seed)
        task_rps = _task_rps_from_tasks(tasks)
        requested = len(tasks)
        backbone_counts = defaultdict(int)
        for t in tasks:
            backbone_counts[t["backbone"]] += 1
        clean_tasks = [{k: v for k, v in t.items() if k != "_minutes"}
                       for t in tasks]

        for condition in CONDITIONS:
            placed, error = _run_placement(condition, clean_tasks, task_rps)
            placed_count = len(placed)
            rows.append({
                "regime": regime,
                "num_tasks": num_tasks,
                "trial": trial,
                "seed": trial_seed,
                "condition": condition,
                "requested": requested,
                "placed_count": placed_count,
                "rejected_count": requested - placed_count,
                "placement_rate": placed_count / requested if requested else 0.0,
                "error": error,
                "backbone_counts": dict(sorted(backbone_counts.items())),
            })
            print(f"  trial={trial:03d} {condition}: "
                  f"placed {placed_count}/{requested}"
                  f"{' ERROR: ' + error if error else ''}")

    summary = {
        "regime": regime,
        "num_tasks": num_tasks,
        "trials": trials,
        "seed": seed,
        "mix": mix,
        "owner_source": owner_source,
        "resolved_owner_source": window_info.get("source", owner_source),
        "window_info": window_info,
        "random_pool": [
            {
                "task": uc.task_pool[i]["task"],
                "backbone": uc.task_pool[i]["backbone"],
                "tier": uc.task_pool[i].get("tier"),
            }
            for i in _unique_pool_indices()
        ],
        "conditions": _summarize(rows),
    }

    _write_trials_csv(scenario_dir / "placement_trials.csv", rows)
    (scenario_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return rows, summary


def _summarize_admission(rows: List[dict]) -> Dict[str, dict]:
    summary = {}
    for condition in CONDITIONS:
        cur = [r for r in rows if r["condition"] == condition]
        vals = [float(r["admitted_before_failure"]) for r in cur]
        summary[condition] = {
            "successful_trials": len(cur),
            "admitted_before_failure": _ci95(vals),
            "min_capacity": min(vals) if vals else 0.0,
            "max_capacity": max(vals) if vals else 0.0,
            "censored_trials": sum(1 for r in cur if r["censored"]),
        }
    return summary


_INC_PLACERS = {
    "fmaas": gen.place_task_fmaas_incremental,
    "no_sharing": gen.place_task_clipper_incremental,
}


def _make_inc_state_makers(fmaas_latency_factor: float) -> dict:
    return {
        "fmaas": lambda: gen.make_fmaas_incremental_state(fmaas_latency_factor),
        "no_sharing": gen.make_clipper_incremental_state,
    }


def _run_admission_scenario(regime: str, trials: int, seed: int,
                            max_tasks: int, owner_pool_size: int,
                            owner_source: str, fmaas_latency_factor: float,
                            out_dir: Path) -> Tuple[List[dict], List[dict], dict]:
    duration = float(uc.experiment["duration"])
    scenario_dir = out_dir / "admission" / regime
    scenario_dir.mkdir(parents=True, exist_ok=True)

    print(f"[random-placement/admission] regime={regime} "
          f"trials={trials} max_tasks={max_tasks} owner_source={owner_source}")

    step_rows = []
    trial_rows = []
    window_info = None

    for trial in range(trials):
        trial_seed = int(seed + trial)
        owner_stream, window_info = _make_owner_stream(
            regime, max_tasks, owner_pool_size, trial_seed, owner_source)
        task_stream = _build_random_task_list(
            owner_stream, duration, trial_seed + 10_000_000)

        # One fresh incremental state per condition for this trial.
        inc_state_makers = _make_inc_state_makers(fmaas_latency_factor)
        inc_states = {c: inc_state_makers[c]() for c in CONDITIONS}

        failed = {condition: False for condition in CONDITIONS}
        first_failure = {condition: None for condition in CONDITIONS}
        placed_at_failure = {condition: None for condition in CONDITIONS}
        placed_counts = {condition: 0 for condition in CONDITIONS}

        for prefix_n in range(1, max_tasks + 1):
            new_task = {k: v for k, v in task_stream[prefix_n - 1].items()
                        if k != "_minutes"}
            task_rps_single = {new_task["task"]: max(
                float(new_task.get("owner_rate_real", 0.0)), 0.1)}

            for condition in CONDITIONS:
                if failed[condition]:
                    continue

                scheduler, config, state = inc_states[condition]
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    was_placed = _INC_PLACERS[condition](
                        scheduler, config, state, new_task, task_rps_single)

                if was_placed:
                    placed_counts[condition] += 1
                placed_count = placed_counts[condition]
                all_placed = placed_count == prefix_n
                if not all_placed:
                    failed[condition] = True
                    first_failure[condition] = prefix_n
                    placed_at_failure[condition] = placed_count
                step_rows.append({
                    "regime": regime,
                    "trial": trial,
                    "seed": trial_seed,
                    "prefix_n": prefix_n,
                    "condition": condition,
                    "placed_count": placed_count,
                    "all_placed": all_placed,
                    "first_failure_n": first_failure[condition] or "",
                    "error": "",
                })

            if prefix_n % 50 == 0:
                status = []
                for condition in CONDITIONS:
                    if first_failure[condition] is None:
                        status.append(f"{condition}=ok")
                    else:
                        status.append(
                            f"{condition}=failed@{first_failure[condition]}"
                        )
                print(f"  progress regime={regime} trial={trial:03d}: "
                      f"placed {prefix_n} tasks ({', '.join(status)})",
                      flush=True)

            if all(failed.values()):
                break

        for condition in CONDITIONS:
            failure_n = first_failure[condition]
            censored = failure_n is None
            trial_rows.append({
                "regime": regime,
                "trial": trial,
                "seed": trial_seed,
                "condition": condition,
                "admitted_before_failure": max_tasks if censored else failure_n - 1,
                "first_failure_n": "" if censored else failure_n,
                "placed_at_failure": "" if censored else placed_at_failure[condition],
                "censored": censored,
                "max_tasks": max_tasks,
            })
            cap = trial_rows[-1]["admitted_before_failure"]
            print(f"  trial={trial:03d} {condition}: capacity={cap}"
                  f"{' (censored)' if censored else ''}")

    summary = {
        "regime": regime,
        "trials": trials,
        "seed": seed,
        "max_tasks": max_tasks,
        "owner_pool_size": owner_pool_size,
        "owner_source": owner_source,
        "fmaas_latency_factor": fmaas_latency_factor,
        "resolved_owner_source": (
            window_info.get("source", owner_source) if window_info else owner_source
        ),
        "window_info": window_info,
        "conditions": _summarize_admission(trial_rows),
    }
    _write_admission_steps(scenario_dir / "admission_steps.csv", step_rows)
    _write_admission_trials(scenario_dir / "admission_trials.csv", trial_rows)
    (scenario_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return step_rows, trial_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fixed-n", "admission"],
                        default="admission")
    parser.add_argument("--regimes", default="low,medium,high",
                        help="Comma-separated regimes to sweep.")
    parser.add_argument("--task-counts", default='1500',
                        help="Comma-separated task counts, e.g. 8,16,32.")
    parser.add_argument("--min-tasks", type=int, default=8)
    parser.add_argument("--max-tasks", type=int, default=80)
    parser.add_argument("--step-tasks", type=int, default=8)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--admission-max-tasks", type=int, default=2000)
    parser.add_argument("--owner-pool-size", type=int, default=80)
    parser.add_argument("--owner-source", choices=["real", "synthetic", "auto"],
                        default="auto",
                        help=("Owner source: real uses only MAF traces, synthetic "
                              "uses generated steady owners, auto tries real first "
                              "and falls back to synthetic only if real selection "
                              "cannot find a valid window."))
    parser.add_argument("--fmaas-latency-factor", type=float, default=0.9,
                        help=("Multiplicative factor applied to profiler latency "
                              "when computing FMaaS utilisation (admission mode only). "
                              "E.g. 0.9 models a 10%% latency reduction from batching, "
                              "allowing more tasks to be packed per device. "
                              "Default 1.0 (no scaling). Has no effect on no_sharing."))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-label", default=None,
                        help="Output subdirectory name. Defaults to timestamp.")
    args = parser.parse_args()

    regimes = _parse_csv_list(args.regimes)
    task_counts = _parse_task_counts(args)
    run_label = args.run_label or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "mode": args.mode,
        "regimes": regimes,
        "task_counts": task_counts,
        "trials": args.trials,
        "seed": args.seed,
        "admission_max_tasks": args.admission_max_tasks,
        "owner_pool_size": args.owner_pool_size,
        "owner_source": args.owner_source,
        "fmaas_latency_factor": args.fmaas_latency_factor,
        "output_dir": str(out_dir),
        "conditions": list(CONDITIONS),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    if args.mode == "admission":
        all_step_rows: List[dict] = []
        all_trial_rows: List[dict] = []
        summaries: List[dict] = []
        for i, regime in enumerate(regimes):
            scenario_seed = int(args.seed + i * 1_000_000)
            step_rows, trial_rows, summary = _run_admission_scenario(
                regime, args.trials, scenario_seed, args.admission_max_tasks,
                args.owner_pool_size, args.owner_source,
                args.fmaas_latency_factor, out_dir)
            all_step_rows.extend(step_rows)
            all_trial_rows.extend(trial_rows)
            summaries.append(summary)
        _write_admission_steps(out_dir / "admission_all_steps.csv", all_step_rows)
        _write_admission_trials(out_dir / "admission_all_trials.csv",
                                all_trial_rows)
        _write_admission_aggregate(out_dir / "admission_aggregate_summary.csv",
                                   summaries)
        (out_dir / "admission_aggregate_summary.json").write_text(json.dumps({
            "config": config,
            "scenarios": summaries,
        }, indent=2))
    else:
        all_rows: List[dict] = []
        summaries: List[dict] = []
        scenario_idx = 0
        for regime in regimes:
            for num_tasks in task_counts:
                scenario_seed = int(args.seed + scenario_idx * 1_000_000)
                rows, summary = _run_scenario(
                    regime, num_tasks, args.trials, scenario_seed,
                    args.owner_source, out_dir)
                all_rows.extend(rows)
                summaries.append(summary)
                scenario_idx += 1
        _write_trials_csv(out_dir / "all_placement_trials.csv", all_rows)
        _write_aggregate_summary(out_dir / "aggregate_summary.csv", summaries)
        (out_dir / "aggregate_summary.json").write_text(json.dumps({
            "config": config,
            "scenarios": summaries,
        }, indent=2))
    print(f"[random-placement] wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
