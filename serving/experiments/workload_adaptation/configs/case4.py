"""Case 4 — single-task backbone cold-start (simplest version of case 2).

Initial state:
  device1 (momentlarge, :8000): trafficfore (the victim)
  device2: nothing — no backbone running

Adaptation:
  At t = bump_at, trafficfore RPS jumps; device1 alone absorbs it.
  At t = spinup_at, momentlarge is started cold on device2:8000 via SSH.
  At t = attach_at, mirror decoder is attached to the new backbone and
  routing splits.

The gap (spinup_at, attach_at] is the cold-start latency window — much
larger than the decoder-attach latency in case 3. This is the experiment's
key finding: hot-attach is sub-second, cold-start is tens of seconds.
"""

# ── Runtime knobs ────────────────────────────────────────────────────
experiment = {
    'trace':                 'poisson_per_task',
    'duration':              120,
    'max_batch_wait_ms':     0,
    'max_batch_size':        32,
    'isolation_mode':        'shared',
    'warmup_gap':            2.0,
    'pretrace_warmup_secs':  15.0,
    'max_model_len':         256,
    'batch_mode':            'fixedpoint',
}

# ── Hardware inventory ───────────────────────────────────────────────
# Both devices are listed so the runner can SSH-deploy device2 at runtime.
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
}

# ── Sweep parameters ─────────────────────────────────────────────────
# Only device1 is in the base plan. device2 is created at runtime.
gpus = ['device1']
n_apps_list = [1]

task_pool = [
    {"task": "trafficfore", "backbone": "momentlarge"},
]

per_app_rps = 5.0

assignment = "custom"
custom_assignment = {1: [[0]]}

tpc_per_app = {}
tpc_per_app_split = {}
sharing_tpc_split = {}

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
    "offload_target_port": 8000,         # fresh server, default port
    "mirror_task_name": "trafficfore",

    # Timeline — leaves room for momentlarge cold-start (~30-60s on A2).
    # bump_at and spinup_at are simultaneous; attach_at is after spinup is
    # expected to finish. The runner's cold-start step blocks on _deploy_one
    # until the backbone is loaded, so attach_at being "early" just means
    # the await chain progresses as fast as the spinup completes.
    "bump_at_s":      60.0,
    "spinup_at_s":    60.0,
    "attach_at_s":    60.0,                # actual attach happens after spinup returns
    "stop_at_s":     180.0,

    # # RPS profile
    # "baseline_rps":   15.0,
    # "bumped_rps":     70.0,
    # "mirror_share_post_attach": 0.5,
    # # # RPS profile
    # "baseline_rps":   15.0,
    # "bumped_rps":     30.0,
    # "mirror_share_post_attach": 0.5
    # # RPS profile
    "baseline_rps":   15.0,
    "bumped_rps":     50.0, 
    "mirror_share_post_attach": 0.5    

}

active_tasks = sorted({e["task"] for e in task_pool})
