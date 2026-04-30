"""Case 1 — same backbone (momentlarge) on both devices.

Initial state:
  device1 (momentlarge, :8000): 4 tasks
  device2 (momentlarge, :8000): 1 task

Adaptation:
  At t = bump_at, device1's victim task RPS jumps. At t = attach_at, the
  same task's decoder is hot-attached to device2's existing momentlarge
  backbone, and routing weights split the inflated load across both.

Schema mirrors experiments/cluster_sharing_benefit/user_config.py so the
shared generator can reuse the same builders (sharing condition).
"""

# ── Runtime knobs ────────────────────────────────────────────────────
experiment = {
    'trace':                 'poisson_per_task',
    'duration':              420,
    'max_batch_wait_ms':     0,
    'max_batch_size':        32,
    'isolation_mode':        'shared',
    'warmup_gap':            2.0,
    'pretrace_warmup_secs':  15.0,
    'max_model_len':         256,
    'batch_mode':            'fixedpoint',
}

# ── Hardware inventory ───────────────────────────────────────────────
devices = {
    'device1': {'type': 'NVIDIA A2', 'mem': 15360, 'ip': '192.168.245.191',
                'site_manager': 'site2', 'cuda': 'cuda:0', 'tpcs': 5},
    'device2': {'type': 'NVIDIA A2', 'mem': 15360, 'ip': '192.168.245.194',
                'site_manager': 'site2', 'cuda': 'cuda:0', 'tpcs': 5},
}

# ── Canonical task metadata ──────────────────────────────────────────
factor = 1.5
tasks = {
    'trafficfore':  {'type': 'forecasting',    'peak_workload': 50, 'latency': 5.58*factor, 'metric': 'mae',      'value': 5.0,  'backbone': 'momentlarge', 'seed': 1000},
    'exchangefore': {'type': 'forecasting',    'peak_workload': 50, 'latency': 5.58*factor, 'metric': 'mae',      'value': 5.0,  'backbone': 'momentlarge', 'seed': 800},
    'heartrate':    {'type': 'regression',     'peak_workload': 50, 'latency': 5.58*factor, 'metric': 'mae',      'value': 100,  'backbone': 'momentlarge', 'seed': 100},
}

# ── Sweep parameters ─────────────────────────────────────────────────
gpus = ['device1', 'device2']

# Single-point sweep — one base plan per case.
n_apps_list = [3]

# Pool order matters: indices 0..3 land on device1 via custom_assignment,
# index 4 lands on device2.
task_pool = [
    {"task": "trafficfore",  "backbone": "momentlarge"},
    {"task": "exchangefore", "backbone": "momentlarge"},
    {"task": "heartrate",    "backbone": "momentlarge"},
]

per_app_rps = 5.0

assignment = "custom"
custom_assignment = {3: [[0, 1], [2]]}

# Single backbone per device → no TPC split needed; full GPU per backbone.
tpc_per_app = {}
tpc_per_app_split = {}
sharing_tpc_split = {}

conditions = ["sharing"]

# ── Workload-adaptation specific ─────────────────────────────────────
# The "victim" task whose RPS gets bumped and is then offloaded.
adaptation = {
    "victim_task": "trafficfore",                      # task name in plan
    "victim_base_task": "trafficfore",
    "victim_decoder_path": "trafficfore_momentlarge_mlp",
    "victim_decoder_type": "forecasting",
    "victim_origin_device": "device1",                 # currently serves it
    "offload_target_device": "device2",                # absorbs new load
    "offload_target_backbone": "momentlarge",          # already running on device2:8000
    "offload_target_port": 8000,
    # Mirror decoder on device2 uses the SAME task name as on device1.
    # The router auto-splits requests proportional to request_per_sec across
    # all deployments that list this task — that IS the weighted split.
    "mirror_task_name": "trafficfore",

    # Timeline
    "bump_at_s":      30.0,
    "attach_at_s":    30.0,
    "stop_at_s":      60.0,

    # RPS profile
    "baseline_rps":   5.0,                             # pre-bump victim RPS
    "bumped_rps":     40.0,                            # post-bump victim RPS
    # Of the bumped_rps, fraction routed to the mirror on device2.
    "mirror_share_post_attach": 0.50,
    # Pre-attach (between bump_at and attach_at) all bumped traffic still hits
    # device1; that's the spike the figure shows being absorbed.
}

active_tasks = sorted({e["task"] for e in task_pool})
