"""Case 3 — single-task decoder-attach (simplest version of case 1).

Initial state:
  device1 (momentlarge, :8000): trafficfore (the victim)
  device2 (momentlarge, :8000): a single placeholder task to keep the
                                 backbone deployed and warm

Adaptation:
  At t = bump_at, trafficfore RPS jumps. Decoder is hot-attached to
  device2's existing momentlarge backbone, and routing splits.

The placeholder task on device2 has very low RPS and runs throughout —
just enough to keep the deployment alive so we don't conflate cold-start
latency with decoder-attach latency. Use case4 for cold-start.
"""

# ── Runtime knobs ────────────────────────────────────────────────────
experiment = {
    'trace':                 'poisson_per_task',
    'duration':              60,
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
    'trafficfore':  {'type': 'forecasting', 'peak_workload': 50, 'latency': 5.58*factor, 'metric': 'mae', 'value': 5.0,  'backbone': 'momentlarge', 'seed': 1000},
    # Placeholder on device2 — kept at very low RPS so it doesn't confound
    # the trafficfore measurement post-attach.
    'heartrate':    {'type': 'regression',  'peak_workload': 50, 'latency': 5.58*factor, 'metric': 'mae', 'value': 100,  'backbone': 'momentlarge', 'seed': 100},
}

# ── Sweep parameters ─────────────────────────────────────────────────
gpus = ['device1', 'device2']
n_apps_list = [2]

# device1 hosts trafficfore, device2 hosts the placeholder.
task_pool = [
    {"task": "trafficfore", "backbone": "momentlarge"},
    {"task": "heartrate",   "backbone": "momentlarge"},
]

per_app_rps = 5.0   # default for both; adaptation overrides trafficfore live

assignment = "custom"
custom_assignment = {2: [[0], [1]]}

tpc_per_app = {}
tpc_per_app_split = {}
sharing_tpc_split = {}   # single backbone per device → no split needed

conditions = ["sharing"]

# ── Workload-adaptation specific ─────────────────────────────────────
adaptation = {
    "victim_task": "trafficfore",
    "victim_base_task": "trafficfore",
    "victim_decoder_path": "trafficfore_momentlarge_mlp",
    "victim_decoder_type": "forecasting",
    "victim_origin_device": "device1",
    "offload_target_device": "device2",
    "offload_target_backbone": "momentlarge",
    "offload_target_port": 8000,            # already running on :8000
    "mirror_task_name": "trafficfore",      # same task name → router weighted-splits

    # Timeline
    "bump_at_s":      60.0,
    "attach_at_s":    60.0,                 # bump + attach simultaneous
    "stop_at_s":      180.0,

    # RPS profile
    "baseline_rps":   15.0,
    "bumped_rps":     70.0,
    "mirror_share_post_attach": 0.5,        # 50/50 split
}

active_tasks = sorted({e["task"] for e in task_pool})
