"""Cluster sharing-benefit experiment config.

Single source of truth for: hardware inventory, canonical task pool, sweep
parameters (N values, app pool, TPC layouts, conditions), and runtime knobs.

Sweep flow
----------
deployments/generate.py reads the sweep section below and emits
    deployments/N{N}/<condition>.json
    deployments/N{N}/assignment.json
    deployments/N{N}/task_meta.json
for every N in `n_apps_list`. run.py loads them via --n-apps + --condition.

Pool entries (one per app)
--------------------------
    {"task":         <unique task name; used in plan + trace>,
     "backbone":     <backbone name>,
     "base_task":    <optional; metadata fallback in `tasks` below>,
     "decoder_path": <optional; defaults to "{base_task}_{backbone_short}_mlp">}

If the pool is cycled or two entries reuse the same `task`, deployment
generation auto-suffixes later copies as `<task>__app<idx>` so each logical app
gets its own task stream and its own loaded decoder instance. Repeated apps
reuse the same trained decoder checkpoint path by default unless `decoder_path`
is overridden explicitly. Add a matching alias in client/repeated_task.py so
the client runner reuses the base task's dataset, e.g. {"ecgclass*": "ecgclass"}.
"""

# ── Runtime knobs ────────────────────────────────────────────────────
experiment = {
    'trace':                 'alibaba_gentd26', #alibaba_gentd26,poisson_per_task
    'duration':              180,
    'max_batch_wait_ms':     0,
    'max_batch_size':        32,
    'isolation_mode':        'shared',
    'warmup_gap':            2.0,
    'warmup_burst_secs':     15.0,
    'max_model_len':         256,
    'batch_mode':            'fixedpoint',
}

# ── Hardware inventory ───────────────────────────────────────────────
# `tpcs` is the total #TPCs available for partitioning on this device.
devices = {
    'device1': {'type': 'NVIDIA A2', 'mem': 15356, 'ip': '192.168.245.191',
                'site_manager': 'site2', 'cuda': 'cuda:0', 'tpcs': 5},
    'device2': {'type': 'NVIDIA A2', 'mem': 15356, 'ip': '192.168.245.193',
                'site_manager': 'site2', 'cuda': 'cuda:0', 'tpcs': 5},
    'device3': {'type': 'NVIDIA A2', 'mem': 15356, 'ip': '192.168.245.194',
                'site_manager': 'site2', 'cuda': 'cuda:0', 'tpcs': 5},
    'device4': {'type': 'NVIDIA A2', 'mem': 15356, 'ip': '192.168.245.195',
                'site_manager': 'site2', 'cuda': 'cuda:0', 'tpcs': 5},
}

# ── Canonical task metadata ──────────────────────────────────────────
factor = 1.5
tasks = {
    'heartrate':    {'type': 'regression',     'peak_workload': 50, 'latency': 5.58*factor, 'metric': 'mae',      'value': 100,  'backbone': 'momentlarge', 'seed': 100},
    'sysbp':        {'type': 'regression',     'peak_workload': 50, 'latency': 5.55*factor, 'metric': 'mae',      'value': 100,  'backbone': 'momentlarge', 'seed': 200},
    'diasbp':       {'type': 'regression',     'peak_workload': 50, 'latency': 5.58*factor, 'metric': 'mae',      'value': 100,  'backbone': 'momentlarge', 'seed': 300},
    'ecgclass':     {'type': 'classification', 'peak_workload': 50, 'latency': 3.86*factor, 'metric': 'accuracy', 'value': 0.7,  'backbone': 'momentlarge', 'seed': 400},
    'gestureclass': {'type': 'classification', 'peak_workload': 50, 'latency': 3.88*factor, 'metric': 'accuracy', 'value': 0.6,  'backbone': 'momentlarge', 'seed': 500},
    'etth1fore':    {'type': 'forecasting',    'peak_workload': 50, 'latency': 5.58*factor, 'metric': 'mae',      'value': 5.0,  'backbone': 'momentlarge', 'seed': 600},
    'weatherfore':  {'type': 'forecasting',    'peak_workload': 50, 'latency': 5.58*factor, 'metric': 'mae',      'value': 5.0,  'backbone': 'momentlarge', 'seed': 700},
    'exchangefore': {'type': 'forecasting',    'peak_workload': 50, 'latency': 5.58*factor, 'metric': 'mae',      'value': 5.0,  'backbone': 'momentlarge', 'seed': 800},
    'eclfore':      {'type': 'forecasting',    'peak_workload': 50, 'latency': 5.58*factor, 'metric': 'mae',      'value': 5.0,  'backbone': 'momentlarge', 'seed': 900},
    'trafficfore':  {'type': 'forecasting',    'peak_workload': 50, 'latency': 5.58*factor, 'metric': 'mae',      'value': 5.0,  'backbone': 'momentlarge', 'seed': 1000},
    'nyudepth':     {'type': 'monocular',      'peak_workload': 50, 'latency': 20.0,        'metric': 'rmse',     'value': 0.5,  'backbone': 'dinobase',    'seed': 1100},
    'vocseg':       {'type': 'linear_seg',     'peak_workload': 50, 'latency': 20.0,        'metric': 'miou',     'value': 0.5,  'backbone': 'dinobase',    'seed': 1200},
    'imgclass10': {'type': 'linear_classification', 'peak_workload': 50, 'latency': 10.0,        'metric': 'accuracy', 'value': 0.7,  'backbone': 'dinobase',    'seed': 1300},
    'eurosatclass': {'type': 'linear_classification', 'peak_workload': 50, 'latency': 10.0,        'metric': 'accuracy', 'value': 0.7,  'backbone': 'dinobase',    'seed': 1400},
    # 'crowdcount':   {'type': 'spatial_count',  'peak_workload': 50, 'latency': 20.0,        'metric': 'mae',      'value': 50.0, 'backbone': 'dinobase',    'seed': 1500},
}

# ── Sweep parameters ─────────────────────────────────────────────────
# Subset of devices to use. Defaults to all GPUs.
gpus = list(devices.keys())

n_apps_list = [8,16,32,64,128]

# Ordered pool. Cycled when N > len(pool). One entry = one app.
# Repeated entries are auto-renamed to "<task>__app<app_idx>" at generation time.
task_pool = [
    # Vision — dino (patch variants for dense-prediction decoders)
    {"task": "nyudepth",     "backbone": "dinosmall", "decoder_path": "nyudepth_dinosmall_monocular"},
    {"task": "heartrate",    "backbone": "momentbase"},
    {"task": "imgclass10",       "backbone": "swinsmall",  "decoder_path": None},
    {"task": "diasbp",       "backbone": "papageissvri"},
    {"task": "vocseg", "backbone": "dinosmall",        "decoder_path": None},
    {"task": "sysbp",        "backbone": "momentbase"},
    {"task": "eurosatclass", "backbone": "swinsmall",        "decoder_path": None},
    {"task": "etth1fore",    "backbone": "papageissvri"},
]

per_app_rps = 2.0

assignment = "round_robin"      # or "custom"
custom_assignment: dict = {}    # {N: [[app_idx,...per gpu0], [...gpu1], ...]}

# no_sharing_tpc: TPCs per app, per N. (Consecutive, wrapping if oversubscribed.)
tpc_per_app = {8: 2, 16: 1, 24: 1, 32: 1, 40:1, 48:1, 56:1, 64:1, 128:1}

# no_sharing_tpc: optional per-GPU uneven TPC split. Overrides tpc_per_app[N]
# when the (N, gpu_name) entry is present. Each list gives one TPC count per
# app on that GPU, in assignment order. Example for 4 apps on a 5-TPC GPU:
# {16: {"device1": [1, 1, 1, 2]}}
tpc_per_app_split = {8: {"device1": [2,3],"device2": [2,3],"device3": [2,3],"device4": [2,3]},
                    16: {"device1": [1, 1, 1, 2],"device2": [1, 1, 1, 2],"device3": [1, 1, 1, 2],"device4": [1, 1, 1, 2]}}

# sharing: explicit TPC split per GPU per N when multiple backbones co-locate.
# {N: {gpu_name: [tpc_count_per_backbone_group, ...]}} ordered by first appearance.
# Omit a GPU key when only one backbone lands there (full GPU, no TPC field).
sharing_tpc_split: dict = {
    # # N=16: each GPU hosts 4 distinct backbones on 5 TPCs (one backbone gets 2).
    # 16: {
    #     "device1": [2, 1, 1, 1],
    #     "device2": [2, 1, 1, 1],
    #     "device3": [2, 2, 1],
    #     "device4": [2, 1, 1, 1],
    # },
}

conditions = ["no_sharing", "sharing", "no_sharing_tpc"]

# ── Active tasks (kept for compatibility with anything that imports it) ─
active_tasks = sorted({e["task"] for e in task_pool})
