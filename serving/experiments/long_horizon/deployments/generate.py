#!/usr/bin/env python3
"""Generate deployment plans for the long-horizon experiment.

Reads user_config.py and the Alibaba trace to build, for each condition:
    deployments/<condition>.json         initial deployment plan
    deployments/<condition>_slots.json   full lookup: task → device/backbone/spec

Also writes (condition-independent):
    deployments/task_timeline.json   task_name → {arrive, depart, model_id, base_task}
    deployments/task_meta.json       task_name → base_task  (for cfg.tasks lookups)

Timeline derivation
-------------------
Each task instance is assigned one Alibaba model.  The model's first/last
invocation time (time-compressed to experiment duration) becomes the task's
arrive/depart.  Backbone types are assigned round-robin in arrive-time order
so each backbone gets a natural mix of early and late starters.

Tasks with arrive < timeline['initial_active_secs'] are deployed at t=0;
the rest are cold-started (no_sharing) or hot-attached (fmaas) at runtime.

Usage:
    cd serving
    python -m experiments.long_horizon.deployments.generate
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import fnmatch

from experiments.long_horizon import user_config as uc
from traces.alibaba_long_horizon import extract_timeline as _alibaba_extract_timeline

# ── Repeated-task profile wrapper ─────────────────────────────────────────────
# Loads client/repeated_task.py (wildcard → base-task map) so synthetic instance
# names like "imgclass10__app1" resolve to the correct profiler pipeline.

def _load_repeated_map() -> Dict[str, str]:
    p = Path(__file__).resolve().parents[3] / "client" / "repeated_task.py"
    import json
    return json.loads(p.read_text())

def _base_task_name(task: str, repeated: Dict[str, str]) -> str:
    """Return the profiler base task name for a (possibly synthetic) task name."""
    for pattern, base in repeated.items():
        if fnmatch.fnmatch(task, pattern):
            return base
    return task

def _make_profile(repeated: Dict[str, str]):
    """Return a ProfileData subclass whose find_pipeline_id falls back to the
    base task name when the instance name has no direct profiler entry."""
    from planner import ProfileData
    from planner.parser.profiler import components, pipelines, latency, metric

    class _RepeatedProfileData(ProfileData):
        def find_pipeline_id(self, task, backbone):
            pid = super().find_pipeline_id(task, backbone)
            if pid is None:
                base = _base_task_name(task, repeated)
                if base != task:
                    pid = super().find_pipeline_id(base, backbone)
            return pid

    return _RepeatedProfileData(components, pipelines, latency, metric)

OUT_DIR   = Path(__file__).resolve().parent
BASE_PORT = 8000
PORT_STEP = 10


# ── Timeline from trace ───────────────────────────────────────────────────────

def _extract_timeline_for_trace(exp: dict, tl: dict, n_total: int,
                                 trace_duration: float) -> List[dict]:
    """Dispatch to the right trace adapter based on exp['trace'].

    Returns a list of dicts with keys: arrive, depart, and either
    'model_id' (alibaba_lh) or 'function_id' (maf_lh).
    """
    trace = exp.get('trace', 'alibaba_lh')
    idle  = float(tl.get('idle_timeout_real_s', 300.0))

    if trace == 'maf_lh':
        from traces.maf import extract_timeline as _maf_extract_timeline
        return _maf_extract_timeline(
            n_tasks             = n_total,
            duration            = trace_duration,
            idle_timeout_real_s = idle,
            min_invocations     = int(exp.get('maf_min_invocations', 50)),
            seed                = 42,
        )
    else:  # default: alibaba_lh
        return _alibaba_extract_timeline(
            n_tasks             = n_total,
            duration            = trace_duration,
            idle_timeout_real_s = idle,
            min_window_s        = float(exp.get('timing_min_window_s', 30.0)),
            seed                = 42,
        )


def _get_req_model_pool(
    trace_duration: float,
    idle_timeout_real_s: float = 300.0,
    min_rps: float = 1.0,
    min_window_s: float = 50.0,
) -> List[str]:
    """Return Alibaba model IDs whose compressed RPS >= min_rps and window >= min_window_s.

    These are used as the request-generation pool (separate from the timing pool).
    RPS is estimated from the model's real mean inter-arrival scaled by the same
    compression factor as extract_timeline.
    """
    from traces.alibaba_gentd26 import (
        DEFAULT_TRACE_PATH, DEFAULT_GROUP_BY,
        load_trace, _interarrivals_for_task,
    )

    df       = load_trace(DEFAULT_TRACE_PATH)
    group_by = DEFAULT_GROUP_BY

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

    trace_start     = min(all_first)
    trace_end       = max(all_last)
    trace_span      = (trace_end - trace_start).total_seconds()
    scale           = trace_duration / trace_span
    idle_compressed = idle_timeout_real_s * scale

    valid = []
    for model_id, grp in df.groupby(group_by):
        times = grp["gmt_create"].sort_values()
        if len(times) < 2:
            continue

        arrive = (times.iloc[0] - trace_start).total_seconds() * scale
        depart = min(
            (times.iloc[-1] - trace_start).total_seconds() * scale + idle_compressed,
            trace_duration,
        )
        if depart - arrive < min_window_s:
            continue

        ia = _interarrivals_for_task(df, model_id, group_by)
        if ia.size == 0:
            continue

        compressed_rps = 1.0 / (float(ia.mean()) * scale)
        if compressed_rps >= min_rps:
            valid.append(model_id)

    return sorted(valid)


def build_task_list() -> List[dict]:
    """Return flat list of task dicts, one per task instance, sorted by arrive.

    Each dict:
        task, base_task, backbone, tier, decoder_path,
        model_id (alibaba_lh) or function_id (maf_lh), arrive, depart, backbone_idx
    """
    exp     = uc.experiment
    tl      = uc.timeline
    n_bb    = len(uc.task_pool)
    n_total = int(exp.get('n_tasks', n_bb))

    duration       = float(exp['duration'])
    trace_duration = float(exp.get('trace_duration', exp['duration']))
    entries = _extract_timeline_for_trace(exp, tl, n_total, trace_duration)

    # Continuous-replay window: when duration > trace_duration, arrivals stay in
    # [0, trace_duration] but departs shift into the last trace_duration window
    # so the IA pattern replays cyclically across the full experiment.
    trace        = exp.get('trace', 'alibaba_lh')
    n_cycles     = max(int(duration // trace_duration), 1)
    depart_shift = (n_cycles - 1) * trace_duration if trace == 'alibaba_lh' else 0.0
    if depart_shift > 0:
        print(f"[gen] continuous replay: {n_cycles} cycles × {trace_duration:.0f}s "
              f"= {duration:.0f}s; depart_shift=+{depart_shift:.0f}s")

    # For alibaba_lh: build the request-model pool (high-RPS models for replay).
    req_pool: List[str] = []
    if trace == 'alibaba_lh':
        req_pool = _get_req_model_pool(
            trace_duration      = trace_duration,
            idle_timeout_real_s = float(tl.get('idle_timeout_real_s', 300.0)),
            min_rps             = float(exp.get('req_model_min_rps', 1.0)),
            min_window_s        = float(exp.get('req_model_min_window_s', 50.0)),
        )
        if not req_pool:
            raise ValueError(
                "No Alibaba models meet the req_model_min_rps / req_model_min_window_s "
                "criteria.  Lower the thresholds in user_config.experiment."
            )
        print(f"[gen] req_model_pool ({len(req_pool)} models): {req_pool}")

    # Assign backbone types round-robin in arrive-time order so each backbone
    # gets a natural mix of early and late starters.
    # Use a global counter per base-task name so that two backbone types
    # sharing the same base task (e.g. swintiny and swinsmall both → imgclass10)
    # produce globally unique instance names across the whole task list.
    base_counts: Dict[str, int] = defaultdict(int)
    tasks: List[dict] = []

    for i, entry in enumerate(entries):
        pool_entry = uc.task_pool[i % n_bb]
        base       = pool_entry['task']
        count      = base_counts[base]
        base_counts[base] += 1

        bb_short    = pool_entry['backbone'].replace('-patch', '')
        default_dec = f"{base}_{bb_short}_mlp"
        dec_path    = (pool_entry['decoder_path']
                       if 'decoder_path' in pool_entry
                       else default_dec)

        task_name = base if count == 0 else f"{base}__app{count}"

        new_depart = min(entry['depart'] + depart_shift, duration)
        task_dict = {
            'task':         task_name,
            'base_task':    base,
            'backbone':     pool_entry['backbone'],
            'tier':         pool_entry.get('tier', 'small'),
            'decoder_path': dec_path,
            'arrive':       entry['arrive'],
            'depart':       round(new_depart, 2),
            'backbone_idx': i % n_bb,
        }
        # Store trace-specific source identifiers.
        if 'function_id' in entry:
            task_dict['function_id']  = entry['function_id']
            task_dict['model_id']     = None
            task_dict['req_model_id'] = None
        else:
            task_dict['model_id']     = entry['model_id']
            task_dict['function_id']  = None
            # Assign req_model_id cyclically from high-RPS pool.
            task_dict['req_model_id'] = req_pool[i % len(req_pool)] if req_pool else None
        tasks.append(task_dict)

    return tasks


# ── Device helpers ────────────────────────────────────────────────────────────

def _dev_meta(name: str) -> dict:
    if name not in uc.devices:
        raise KeyError(f"device {name!r} not in user_config.devices")
    return uc.devices[name]

def _dev_names() -> List[str]:
    return list(uc.devices.keys())

def _base_dep(device_name: str, port: int, backbone: str, scheduler: str,
              tpc_partition: List[int] = None) -> dict:
    meta = _dev_meta(device_name)
    d = {
        'device':           f"{meta['ip']}:{port}",
        'device_name':      device_name,
        'device_type':      meta['type'],
        'backbone':         backbone,
        'cuda':             meta['cuda'],
        'scheduler_policy': scheduler,
        'worker_mode':      'threaded',
        'decoders':         [],
        'tasks':            {},
    }
    if tpc_partition is not None:
        d['tpc_mode']      = 'libsmctrl'
        d['tpc_partition'] = tpc_partition
    return d

def _wrap_sites(deps: List[dict]) -> dict:
    by_site: Dict[str, List[dict]] = {}
    for d in deps:
        sid = _dev_meta(d['device_name'])['site_manager']
        by_site.setdefault(sid, []).append(d)
    return {'sites': [{'id': sid, 'deployments': dps} for sid, dps in by_site.items()]}

def _decoder_spec(t: dict) -> dict:
    return {
        'task':      t['task'],
        'base_task': t['base_task'],
        'type':      uc.tasks[t['base_task']]['type'],
        'path':      t['decoder_path'],
    }

def _task_entry(t: dict, rps: float) -> dict:
    return {'type': uc.tasks[t['base_task']]['type'], 'request_per_sec': float(rps)}

def _is_initial(t: dict) -> bool:
    return t['arrive'] < float(uc.timeline.get('initial_active_secs', 5.0))


# ── FMaaS plan ────────────────────────────────────────────────────────────────

def build_fmaas(tasks: List[dict], task_rps: Dict[str, float]) -> tuple:
    """One backbone process per backbone type.  Initially-active tasks get
    decoders attached at t=0; late-arriving tasks are hot-attached at runtime.

    Returns (initial_plan, slots_list).
    slots_list[bi] = {backbone, device_name, device_url, tasks: [all task dicts for backbone bi]}
    """
    dev_names = _dev_names()
    n_bb = len(uc.task_pool)
    if len(dev_names) < n_bb:
        raise ValueError(
            f"FMaaS needs >= {n_bb} devices (one per backbone type), "
            f"but only {len(dev_names)} in user_config.devices."
        )

    by_backbone: Dict[int, List[dict]] = defaultdict(list)
    for t in tasks:
        by_backbone[t['backbone_idx']].append(t)

    deps  = []
    slots = []

    for bi in range(n_bb):
        bb_tasks = by_backbone.get(bi, [])
        if not bb_tasks:
            continue
        dev_name = dev_names[bi]
        backbone = uc.task_pool[bi]['backbone']
        port     = BASE_PORT

        d = _base_dep(dev_name, port, backbone, 'stfq')
        for t in bb_tasks:
            if _is_initial(t):
                d['decoders'].append(_decoder_spec(t))
                d['tasks'][t['task']] = _task_entry(t, task_rps[t['task']])
        deps.append(d)

        slots.append({
            'backbone':    backbone,
            'device_name': dev_name,
            'device_url':  f"{_dev_meta(dev_name)['ip']}:{port}",
            'tasks':       bb_tasks,
        })

    return _wrap_sites(deps), slots


# ── fmaas_place plan ─────────────────────────────────────────────────────────

def build_fmaas_place(
    tasks: List[dict],
    task_rps: Dict[str, float],
) -> tuple:
    """Placement decided by FMaaSPlacementScheduler over all devices/tasks.

    Tasks are placed iteratively in arrival-time order so that early-arriving
    tasks claim resources first, matching actual runtime behavior.
    """
    from planner import SchedulerConfig
    from planner.schedulers.fmaas_place import FMaaSPlacementScheduler
    from planner.state import DeploymentState

    repeated = _load_repeated_map()
    tasks_slo = {}
    task_arrive = {}
    for t in tasks:
        base_info = uc.tasks[t['base_task']]
        tasks_slo[t['task']] = {
            'backbone':      t['backbone'],
            'type':          base_info['type'],
            'peak_workload': float(task_rps[t['task']]),
            'latency':       base_info.get('latency', 50),
            'metric':        base_info.get('metric', 'mae'),
            'value':         base_info.get('value', 0),
        }
        task_arrive[t['task']] = t['arrive']

    profile   = _make_profile(repeated)
    config    = SchedulerConfig()
    scheduler = FMaaSPlacementScheduler(profile, config, batch_profile=None)

    # Iterative placement in arrival order
    servers = scheduler._create_servers(uc.devices)
    state   = DeploymentState(servers)
    for task_name, task_spec in sorted(tasks_slo.items(), key=lambda x: task_arrive[x[0]]):
        task = scheduler._create_task_spec(task_name, task_spec)
        temp_plan, demand_left = scheduler._deploy_task(state, task)
        if demand_left is not None and demand_left > config.demand_epsilon:
            print(f"FMaaSPlacement: task '{task_name}' has {demand_left:.4f} rps "
                  f"unsatisfied demand out of {task.peak_workload:.4f} rps")
        if temp_plan:
            for deployment in temp_plan.values():
                key = (deployment.server_name, deployment.backbone)
                existing = state.get_deployment(deployment.server_name, deployment.backbone)
                if existing:
                    if ':' not in deployment.ip:
                        port = state.get_next_port(
                            deployment.ip, config.base_port, config.port_increment)
                        deployment.ip = f"{deployment.ip}:{port}"
                    state._deployments[key] = deployment
                    state._sync_server_utilization(deployment.server_name, deployment.util)
                else:
                    state.add_deployment(deployment, config.base_port, config.port_increment)
    deployments = state.get_all_deployments()

    task_map = {t['task']: t for t in tasks}

    deps  = []
    slots = []
    for d in deployments:
        placed_names = list(d.task_info.keys())
        bb_tasks = [task_map[n] for n in placed_names if n in task_map]
        if not bb_tasks:
            continue

        dep = {
            'device':           d.ip,
            'device_name':      d.server_name,
            'device_type':      d.device_type,
            'backbone':         d.backbone,
            'cuda':             d.cuda,
            'scheduler_policy': 'stfq',
            'worker_mode':      'threaded',
            'decoders':         [],
            'tasks':            {},
        }

        # deploy_spec: bare backbone spec used for cold-start at runtime when
        # the first task for this backbone arrives (no decoders/tasks yet).
        deploy_spec = {k: v for k, v in dep.items()}
        deploy_spec['decoders'] = []
        deploy_spec['tasks']    = {}

        initial_tasks = [t for t in bb_tasks if _is_initial(t)]
        for t in initial_tasks:
            dep['decoders'].append(_decoder_spec(t))
            dep['tasks'][t['task']] = _task_entry(t, task_rps[t['task']])

        if initial_tasks:
            deps.append(dep)

        slots.append({
            'backbone':    d.backbone,
            'device_name': d.server_name,
            'device_url':  d.ip,
            'tasks':       bb_tasks,
            'deploy_spec': deploy_spec,
        })

    placed = {t['task'] for s in slots for t in s['tasks']}
    unplaced = [t['task'] for t in tasks if t['task'] not in placed]
    if unplaced:
        print(f"[gen] fmaas_place: {len(unplaced)}/{len(tasks)} tasks not placed: {unplaced}")

    return _wrap_sites(deps), slots


# ── clipper_place plan ────────────────────────────────────────────────────────

def build_clipper_place(
    tasks: List[dict],
    task_rps: Dict[str, float],
) -> tuple:
    """Placement decided by ClipperPlacementScheduler. One task per instance.

    Tasks are placed iteratively in arrival-time order so that early-arriving
    tasks claim resources first, matching actual runtime behavior.
    """
    from planner import SchedulerConfig
    from planner.schedulers.clipper_place import ClipperPlacementScheduler
    from planner.state import DeploymentState

    repeated = _load_repeated_map()
    tasks_slo = {}
    task_arrive = {}
    for t in tasks:
        base_info = uc.tasks[t['base_task']]
        tasks_slo[t['task']] = {
            'backbone':      t['backbone'],
            'type':          base_info['type'],
            'peak_workload': float(task_rps[t['task']]),
            'latency':       base_info.get('latency', 50),
            'metric':        base_info.get('metric', 'mae'),
            'value':         base_info.get('value', 0),
        }
        task_arrive[t['task']] = t['arrive']

    profile   = _make_profile(repeated)
    config    = SchedulerConfig()
    scheduler = ClipperPlacementScheduler(profile, config, batch_profile=None)

    # Iterative placement in arrival order
    servers = scheduler._create_servers(uc.devices)
    state   = DeploymentState(servers)
    for task_name, task_spec in sorted(tasks_slo.items(), key=lambda x: task_arrive[x[0]]):
        task = scheduler._create_task_spec(task_name, task_spec)
        temp_plan, demand_left, bottleneck = scheduler._deploy_task(state, task)
        if demand_left is not None and demand_left > config.demand_epsilon:
            print(f"ClipperPlacement: task '{task_name}' has {demand_left:.4f} rps "
                  f"unsatisfied demand out of {task.peak_workload:.4f} rps "
                  f"(bottleneck: {bottleneck})")
        if temp_plan:
            for deployment in temp_plan.values():
                state.add_deployment(deployment, config.base_port, config.port_increment)
    deployments = state.get_all_deployments()

    # deployments use backbone key = "<backbone>__clipper__<task>"
    task_map = {t['task']: t for t in tasks}
    slots: List[dict] = []

    for d in deployments:
        if '__clipper__' in d.backbone:
            real_backbone = d.backbone.split('__clipper__')[0]
            task_name     = d.backbone.split('__clipper__')[1]
        else:
            real_backbone = d.backbone
            task_name     = next(iter(d.task_info), None)

        t = task_map.get(task_name)
        if t is None:
            continue

        spec = {
            'device':            d.ip,
            'device_name':       d.server_name,
            'device_type':       d.device_type,
            'backbone':          real_backbone,
            'cuda':              d.cuda,
            'scheduler_policy':  'fifo',
            'worker_mode':       'inline',
            'decoders':          [_decoder_spec(t)],
            'tasks':             {task_name: _task_entry(t, task_rps[task_name])},
        }
        slots.append({
            **t,
            'device_name':   d.server_name,
            'device_url':    d.ip,
            'tpc_partition': None,
            'deploy_spec':   spec,
        })

    by_device_port: Dict[str, dict] = {}
    for sl in slots:
        if not _is_initial(sl):
            continue
        key = sl['device_url']
        if key not in by_device_port:
            by_device_port[key] = sl['deploy_spec'].copy()
            by_device_port[key]['decoders'] = [_decoder_spec(sl)]
            by_device_port[key]['tasks']    = {sl['task']: _task_entry(sl, task_rps[sl['task']])}
        else:
            by_device_port[key]['decoders'].append(_decoder_spec(sl))
            by_device_port[key]['tasks'][sl['task']] = _task_entry(sl, task_rps[sl['task']])

    placed = {sl['task'] for sl in slots}
    unplaced = [t['task'] for t in tasks if t['task'] not in placed]
    if unplaced:
        print(f"[gen] clipper_place: {len(unplaced)}/{len(tasks)} tasks not placed: {unplaced}")

    return _wrap_sites(list(by_device_port.values())), slots


# ── Workload metrics from trace ───────────────────────────────────────────────

def compute_mean_rps(full_trace: list, tasks: List[dict]) -> Dict[str, float]:
    """Mean RPS per task over its active window [arrive, depart).

    Used both as the runtime request_per_sec (routing weight) and, after
    temporal scaling, as the scheduler's peak_workload.
    """
    arrive = {t['task']: float(t['arrive']) for t in tasks}
    depart = {t['task']: float(t['depart']) for t in tasks}
    counts: Dict[str, int] = defaultdict(int)
    for r in full_trace:
        counts[r["task"]] += 1

    rps: Dict[str, float] = {}
    for task in arrive:
        window = max(depart[task] - arrive[task], 1.0)
        rps[task] = max(counts.get(task, 0) / window, 0.1)
    return rps


# ── Offered-load trace ────────────────────────────────────────────────────────

def build_full_trace(tasks: List[dict], trace_duration: float) -> List[dict]:
    """Pre-generate the workload trace by time-compressing real trace timestamps.

    Condition-independent: all conditions share this trace as the offered load.
    Dispatches to the right trace adapter based on experiment['trace']:
      'alibaba_lh' — compress real Alibaba per-model timestamps
      'maf_lh'     — compress real MAF per-function per-minute counts

    Deduplicates by task name across backbone types.
    """
    trace = uc.experiment.get('trace', 'alibaba_lh')

    seen: set = set()
    task_entries: List[dict] = []
    for t in sorted(tasks, key=lambda t: t['arrive']):
        if t['task'] not in seen:
            seen.add(t['task'])
            entry = {'task': t['task'], 'arrive': t['arrive'], 'depart': t['depart']}
            if trace == 'maf_lh':
                entry['function_id'] = t['function_id']
            else:
                entry['model_id']     = t['model_id']
                entry['req_model_id'] = t['req_model_id']
            task_entries.append(entry)

    if trace == 'maf_lh':
        from traces.maf import compress_function_requests
        return compress_function_requests(task_entries, trace_duration)
    else:
        from traces.alibaba_long_horizon import windowed_replay_requests
        return windowed_replay_requests(task_entries, trace_duration)


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_all() -> None:
    tasks = build_task_list()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    task_timeline = {
        t['task']: {
            'arrive':       t['arrive'],
            'depart':       t['depart'],
            'model_id':     t['model_id'],
            'req_model_id': t.get('req_model_id'),
            'function_id':  t['function_id'],
            'base_task':    t['base_task'],
            'backbone':     t['backbone'],
        }
        for t in tasks
    }
    (OUT_DIR / 'task_timeline.json').write_text(json.dumps(task_timeline, indent=2))
    print(f'[gen] task_timeline → {OUT_DIR}/task_timeline.json')

    task_meta = {t['task']: t['base_task'] for t in tasks}
    (OUT_DIR / 'task_meta.json').write_text(json.dumps(task_meta, indent=2))
    print(f'[gen] task_meta → {OUT_DIR}/task_meta.json')

    trace_duration = float(uc.experiment.get('trace_duration', uc.experiment['duration']))
    full_trace = build_full_trace(tasks, trace_duration)
    (OUT_DIR / 'trace.json').write_text(json.dumps(full_trace, indent=2))
    print(f'[gen] trace → {OUT_DIR}/trace.json ({len(full_trace)} requests, '
          f'trace_duration={trace_duration}s)')

    task_rps = compute_mean_rps(full_trace, tasks)
    (OUT_DIR / 'task_rps.json').write_text(json.dumps(task_rps, indent=2))
    print(f'[gen] task_rps → {OUT_DIR}/task_rps.json')
    print(f'[gen] mean RPS per task (over [arrive, depart)):')
    for task, rps in sorted(task_rps.items(), key=lambda x: x[1], reverse=True):
        print(f'  {task:<25}  {rps:.2f} req/s')

    print(f'[gen] sum(mean rps) = {sum(task_rps.values()):.2f} req/s')

    init_secs = float(uc.timeline.get('initial_active_secs', 5.0))
    n_init    = sum(1 for t in tasks if t['arrive'] < init_secs)
    print(f'[gen] {len(tasks)} tasks: {n_init} initially active, '
          f'{len(tasks) - n_init} arrive dynamically')
    for t in sorted(tasks, key=lambda x: x['arrive']):
        tag = 'init' if t['arrive'] < init_secs else f"t={t['arrive']:.1f}s"
        req_m = t.get('req_model_id') or t.get('function_id') or t.get('model_id')
        print(f"  {t['task']:<22} backbone={t['backbone']:<14} "
              f"{tag:>12}  depart={t['depart']:.1f}s  "
              f"timing={t['model_id']}  req={req_m}")

    builders = {
        'fmaas':      lambda: build_fmaas_place(tasks, task_rps),
        'no_sharing': lambda: build_clipper_place(tasks, task_rps),
    }

    for cond in uc.conditions:
        if cond not in builders:
            print(f'[gen] {cond}: unknown condition, skipping')
            continue
        try:
            plan, slots = builders[cond]()
        except (ValueError, KeyError) as e:
            print(f'[gen] {cond}: SKIPPED — {e}')
            continue
        (OUT_DIR / f'{cond}.json').write_text(json.dumps(plan, indent=2))
        (OUT_DIR / f'{cond}_slots.json').write_text(json.dumps(slots, indent=2))
        print(f'[gen] {cond}: plan → {cond}.json  slots → {cond}_slots.json')


if __name__ == '__main__':
    generate_all()
