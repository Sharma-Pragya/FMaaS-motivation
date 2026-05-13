#!/usr/bin/env python3
"""Generate deployment plans for the end-to-end real-world experiment.

For each (load_regime, N) pair:
  1. Select N Azure Functions HashOwners (top-N by rate for 'high',
     bottom-N for 'low').
  2. Build a flat task list with arrive=0, depart=duration — every task
     is live for the whole experiment.
  3. Compress per-owner invocations into a trace covering [0, duration).
  4. Compute task_rps (mean over [0, duration)) per task.
  5. Run FMaaSPlacementScheduler ('fmaas') and ClipperPlacementScheduler
     ('no_sharing') over all N tasks; record what each places vs. rejects.

Outputs per scenario (under deployments/{regime}_N{N}/):
  fmaas.json        no_sharing.json
  fmaas_slots.json  no_sharing_slots.json
  task_timeline.json  task_meta.json  task_rps.json  trace.json
  placement_summary.json   # {method: {placed: [...], rejected: [...]}}

Usage (from serving/):
    python -m experiments.end_to_end_realworld.deployments.generate
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Make every task count as "initial" inside long_horizon's builders so the
# returned plan deploys all placed tasks at t=0.  We are reusing the
# long_horizon builders verbatim; only this knob needs adjusting.
import experiments.end_to_end_realworld.user_config as _lh_uc
_lh_uc.timeline['initial_active_secs'] = float('inf')

import hashlib
import numpy as np
import fnmatch

from experiments.end_to_end_realworld import user_config as uc
from traces.maf_preprocess import load_hashowner_minutes

OUT_DIR = Path(__file__).resolve().parent

MINUTES_PER_DAY = 1440
SECONDS_PER_MINUTE = 60
DAY_SPAN_S = float(MINUTES_PER_DAY * SECONDS_PER_MINUTE)   # 86,400

# Module-level cache: filtered view of the preprocessed npz.
_filtered_cache: dict = {}

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


def _load_filtered():
    """Load preprocessed day-1 owner matrix, filtered by min_invocations.

    Returns dict with rate-sorted (descending) arrays:
        owner_ids: (k,) U64
        minutes:   (k, 1440) int32
        n_req:     (k,) int64
        rate:      (k,) float  (req / active-window-seconds)
    """
    if 'view' in _filtered_cache:
        return _filtered_cache['view']

    min_inv = int(uc.experiment.get('maf_min_invocations', 200))
    n_days  = int(uc.experiment.get('maf_n_days', 1))

    # Each (owner, day) becomes its own candidate row.  This gives roughly
    # n_days × (#unique owners) rows, expanding band coverage without any
    # synthetic jitter.  function_id is suffixed with __d{day} so downstream
    # task naming and trace seeding treat them as distinct.
    owner_ids_list, minutes_list, n_req_list = [], [], []
    for day in range(1, n_days + 1):
        try:
            data = load_hashowner_minutes(day=day)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"day {day} not preprocessed. Run: "
                f"python -m traces.maf_preprocess {day}"
            ) from e
        oid = data['owner_ids']
        if n_days > 1:
            oid = np.asarray([f"{x}__d{day}" for x in oid], dtype=oid.dtype)
        owner_ids_list.append(oid)
        minutes_list.append(data['minutes'])
        n_req_list.append(data['n_req'])

    owner_ids = np.concatenate(owner_ids_list)
    minutes   = np.concatenate(minutes_list, axis=0)
    n_req     = np.concatenate(n_req_list)

    mask = n_req >= min_inv
    owner_ids = owner_ids[mask]
    minutes   = minutes[mask]
    n_req     = n_req[mask]

    nonzero = minutes > 0
    first   = np.argmax(nonzero, axis=1)
    last    = (MINUTES_PER_DAY - 1) - np.argmax(nonzero[:, ::-1], axis=1)
    window_s = np.maximum((last - first + 1).astype(np.float64) * SECONDS_PER_MINUTE,
                          float(SECONDS_PER_MINUTE))
    rate    = n_req.astype(np.float64) / window_s

    # order = np.argsort(rate)[::-1]   
    order = np.argsort(rate)  # low to high
    view = {
        'owner_ids': owner_ids[order],
        'minutes':   minutes[order],
        'n_req':     n_req[order],
        'rate':      rate[order],
    }
    _filtered_cache['view'] = view
    print(f"[gen] preprocessed day-1: {len(view['owner_ids'])} owners "
          f"with ≥{min_inv} invocations  "
          f"(rate range: {view['rate'][-1]:.4f}–{view['rate'][0]:.4f} req/s)")
    return view


def _select_owners(regime: str, n: int) -> List[dict]:
    """Return n owner-stats entries whose real rate falls in the regime's
    (lo, hi] band.  View is already sorted by rate desc, so we pick the
    rate-densest contiguous slice within the band.
    """
    bands = uc.experiment.get('rate_bands_req_per_s', {})
    if regime not in bands:
        raise ValueError(f"unknown regime {regime!r}; expected one of {list(bands)}")
    lo, hi = float(bands[regime][0]), float(bands[regime][1])

    view = _load_filtered()
    rate = view['rate']
    in_band = np.where((rate > lo) & (rate <hi))[0]
    if in_band.size < n:
        raise ValueError(
            f"regime={regime!r} has only {in_band.size} HashOwners with rate "
            f"in ({lo}, {hi}] req/s, but N={n} requested.  Increase "
            f"maf_n_days, widen the band, lower min_invocations, or reduce N."
        )

    # View is rate-desc.  Take the first N (highest-rate owners within the
    # band) so the task set is NESTED across N values: tasks at N=K are a
    # subset of tasks at N=K' for K' > K.  Combined with rate-desc placement
    # order, this makes placed(N) monotonic in N — the first K placements at
    # N=K' are identical to N=K, so any newly-failing tasks are strictly the
    # extras, never the originals.
    idx = in_band[:n]

    return [{
        'function_id': str(view['owner_ids'][i]),
        'minutes':     view['minutes'][i],
        'n_req':       int(view['n_req'][i]),
        'rate_real':   float(view['rate'][i]),
    } for i in idx]

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
    print(f"[gen] FMaaSPlacement: placing {len(tasks)} tasks over {len(servers)} devices ")
    state   = DeploymentState(servers)
    for task_name, task_spec in sorted(tasks_slo.items(), key=lambda x: task_arrive[x[0]]):
        task = scheduler._create_task_spec(task_name, task_spec)
        temp_plan, demand_left, bottleneck = scheduler._deploy_task(state, task)
        if demand_left is not None and demand_left > config.demand_epsilon:
            print(f"FMaaSPlacement: task '{task_name}' has {demand_left:.4f} rps "
                  f"unsatisfied demand out of {task.peak_workload:.4f} rps "
                  f"(bottleneck: {bottleneck})")
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


# ── Task list ─────────────────────────────────────────────────────────────────

def _build_task_list(owners: List[dict], duration: float) -> List[dict]:
    """One task per owner, with arrive=0 / depart=duration.

    Backbone assignment round-robins over uc.task_pool, identical to
    long_horizon so identical infrastructure handles either experiment.
    """
    n_bb = len(uc.task_pool)
    base_counts: Dict[str, int] = defaultdict(int)
    tasks_out: List[dict] = []

    for i, owner in enumerate(owners):
        pool_entry = uc.task_pool[i % n_bb]
        base       = pool_entry['task']
        count      = base_counts[base]
        base_counts[base] += 1

        bb_short    = pool_entry['backbone'].replace('-patch', '')
        default_dec = f"{base}_{bb_short}_mlp"
        dec_path    = pool_entry.get('decoder_path', default_dec)

        task_name = base if count == 0 else f"{base}__app{count}"
        tasks_out.append({
            'task':         task_name,
            'base_task':    base,
            'backbone':     pool_entry['backbone'],
            'tier':         pool_entry.get('tier', 'small'),
            'decoder_path': dec_path,
            'arrive':       0.0,
            'depart':       float(duration),
            'backbone_idx': i % n_bb,
            'function_id':  owner['function_id'],
            'model_id':     None,
            'req_model_id': None,
            'owner_n_req':  owner['n_req'],
            'owner_rate_real': owner['rate_real'],
            '_minutes':     owner['minutes'],         # used downstream, not serialised
        })
    return tasks_out


# ── Trace + RPS ───────────────────────────────────────────────────────────────

def _build_trace(tasks: List[dict], duration: float) -> List[dict]:
    """Take a real-time `duration`-long slice of each owner's day, starting
    from their first invocation, and replay it at the natural rate.

    No temporal compression — served rate == owner's real rate within the
    chosen window, matching the regime's rate band.
    """
    n_minutes = int(np.ceil(duration / SECONDS_PER_MINUTE))
    chunks = []  # (task_name, sorted real-time timestamps in [0, duration))
    for t in tasks:
        minute_row = t['_minutes']
        active = np.where(minute_row > 0)[0]
        if active.size == 0:
            continue
        start_min = int(active[0])
        end_min   = min(start_min + n_minutes, MINUTES_PER_DAY)
        window    = minute_row[start_min:end_min]
        nz        = np.where(window > 0)[0]
        if nz.size == 0:
            continue
        counts  = window[nz]
        starts  = nz.astype(np.float64) * SECONDS_PER_MINUTE
        seed    = int(hashlib.sha256(t['function_id'].encode()).hexdigest()[:8], 16) % (2**32)
        rng     = np.random.default_rng(seed)
        total   = int(counts.sum())
        offsets = rng.uniform(0.0, SECONDS_PER_MINUTE, size=total)
        real_t  = np.repeat(starts, counts) + offsets
        real_t  = real_t[(real_t >= 0.0) & (real_t < duration)]
        chunks.append((t['task'], np.sort(real_t)))

    # Interleave into one sorted, req_id-numbered list.
    if not chunks:
        return []
    times = np.concatenate([c[1] for c in chunks])
    names = np.concatenate([
        np.full(c[1].size, c[0], dtype=object) for c in chunks
    ])
    order = np.argsort(times, kind='stable')
    times = times[order]
    names = names[order]
    return [
        {'req_id': i, 'task': str(names[i]), 'req_time': float(round(times[i], 4))}
        for i in range(len(times))
    ]


def _compute_rps(full_trace: List[dict], tasks: List[dict],
                 duration: float) -> Dict[str, float]:
    counts: Dict[str, int] = defaultdict(int)
    for r in full_trace:
        counts[r['task']] += 1
    return {t['task']: max(counts[t['task']] / duration, 0.1) for t in tasks}


# ── Diagnostics capture ───────────────────────────────────────────────────────

import io
import re
import contextlib

_FMAAS_RE = re.compile(
    r"FMaaSPlacement: task '([^']+)' has ([\d.]+) rps unsatisfied demand "
    r"out of ([\d.]+) rps \(bottleneck: ([^)]+)\)"
)
_CLIPPER_RE = re.compile(
    r"ClipperPlacement: task '([^']+)' has ([\d.]+) rps unsatisfied demand "
    r"out of ([\d.]+) rps \(bottleneck: ([^)]+)\)"
)


def _run_with_diagnostics(builder) -> Tuple[dict, list, Dict[str, dict]]:
    """Call a builder while capturing scheduler stdout, parsing per-task
    rejection reasons.  Re-emits captured lines so users still see them.

    Returns (plan, slots, per_task_info) where per_task_info maps task name
    to {demand_left, peak_workload, bottleneck} for any task the scheduler
    reported on.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        plan, slots = builder()
    captured = buf.getvalue()
    if captured:
        sys.stdout.write(captured)

    per_task: Dict[str, dict] = {}
    for line in captured.splitlines():
        m = _CLIPPER_RE.search(line)
        if m:
            name, dl, peak, bn = m.groups()
            per_task[name] = {
                'demand_left':   float(dl),
                'peak_workload': float(peak),
                'bottleneck':    bn.strip(),
            }
            continue
        m = _FMAAS_RE.search(line)
        if m:
            name, dl, peak, bn = m.groups()
            per_task[name] = {
                'demand_left':   float(dl),
                'peak_workload': float(peak),
                'bottleneck':    bn.strip(),
            }
    return plan, slots, per_task


# ── Per-scenario driver ───────────────────────────────────────────────────────

def _generate_scenario(regime: str, n: int) -> None:
    scenario_dir = OUT_DIR / f"{regime}_N{n}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    duration = float(uc.experiment['duration'])
    trace_duration = float(uc.experiment.get('trace_duration', duration))

    print(f"\n[gen] === regime={regime}  N={n} ===")
    owners = _select_owners(regime, n)
    tasks  = _build_task_list(owners, duration)

    trace    = _build_trace(tasks, trace_duration)
    task_rps = _compute_rps(trace, tasks, duration)

    (scenario_dir / 'trace.json').write_text(json.dumps(trace, indent=2))
    (scenario_dir / 'task_rps.json').write_text(json.dumps(task_rps, indent=2))
    (scenario_dir / 'task_meta.json').write_text(
        json.dumps({t['task']: t['base_task'] for t in tasks}, indent=2))
    (scenario_dir / 'task_timeline.json').write_text(json.dumps({
        t['task']: {
            'arrive':       t['arrive'],
            'depart':       t['depart'],
            'function_id':  t['function_id'],
            'model_id':     None,
            'req_model_id': None,
            'base_task':    t['base_task'],
            'backbone':     t['backbone'],
        } for t in tasks
    }, indent=2))

    print(f"[gen] {len(tasks)} tasks; trace={len(trace)} reqs; "
          f"Σrps={sum(task_rps.values()):.2f}")
    for t in tasks:
        print(f"  {t['task']:<22} backbone={t['backbone']:<14} "
              f"rps={task_rps[t['task']]:.3f}  owner={t['function_id'][:12]}…")

    # Strip the numpy minutes array before handing tasks to builders / writers —
    # it's only needed for trace generation.
    for t in tasks:
        t.pop('_minutes', None)

    builders = {
        'fmaas':      lambda: build_fmaas_place(tasks, task_rps),
        'no_sharing': lambda: build_clipper_place(tasks, task_rps),
    }

    summary: Dict[str, dict] = {}
    for cond in uc.conditions:
        if cond not in builders:
            continue
        try:
            plan, slots, per_task = _run_with_diagnostics(builders[cond])
        except (ValueError, KeyError) as e:
            print(f"[gen] {cond}: SKIPPED — {e}")
            summary[cond] = {'error': str(e), 'placed': [], 'rejected': []}
            continue

        if cond == 'fmaas':
            placed = sorted({t['task'] for s in slots for t in s['tasks']})
        else:
            placed = sorted({s['task'] for s in slots})
        rejected = sorted({t['task'] for t in tasks} - set(placed))

        # Annotate per-task with placed status.
        for name in placed:
            per_task.setdefault(name, {})['placed'] = True
        for name in rejected:
            per_task.setdefault(name, {'demand_left': None, 'peak_workload': task_rps.get(name),
                                       'bottleneck': None})['placed'] = False

        bottleneck_counts: Dict[str, int] = defaultdict(int)
        for name in rejected:
            info = per_task.get(name, {})
            bn = info.get('bottleneck') or ('unknown_fmaas' if cond == 'fmaas' else 'unknown')
            bottleneck_counts[bn] += 1

        (scenario_dir / f'{cond}.json').write_text(json.dumps(plan, indent=2))
        (scenario_dir / f'{cond}_slots.json').write_text(json.dumps(slots, indent=2))
        summary[cond] = {
            'placed':           placed,
            'rejected':         rejected,
            'placed_count':     len(placed),
            'requested':        len(tasks),
            'per_task':         per_task,
            'bottleneck_counts': dict(bottleneck_counts),
        }
        print(f"[gen] {cond}: placed {len(placed)}/{len(tasks)}; "
              f"rejected={rejected}")
        if bottleneck_counts:
            print(f"[gen] {cond}: rejection bottlenecks → {dict(bottleneck_counts)}")

    (scenario_dir / 'placement_summary.json').write_text(json.dumps(summary, indent=2))


def generate_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for regime in uc.experiment['load_regimes']:
        for n in uc.experiment['n_tasks_sweep']:
            _generate_scenario(regime, n)


if __name__ == '__main__':
    generate_all()
