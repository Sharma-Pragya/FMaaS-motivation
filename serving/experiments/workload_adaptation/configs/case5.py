"""Case 2 — mixed backbones, momentlarge cold-start on device2.

Initial state:
  device1: momentlarge (:8000) with 3 tasks  +  dinosmall (:8010) with 1 task
  device2: dinosmall (:8000) with 2 tasks   (no momentlarge yet)

Adaptation:
  At t = bump_at, victim task (on device1's momentlarge) RPS jumps.
  At t = spinup_at, momentlarge is started on device2:8010 (TPCs 10..19;
  dinosmall keeps 0..9). At t = attach_at, mirror decoder is attached
  to that new backbone and routing splits.

Co-located backbones on one host follow the cluster_sharing_benefit
:8000 / :8010 convention. TPC partitions are set at backbone-start time
(no live repartition); device2's dinosmall is already pinned to 0..9 in
the base plan, leaving 10..19 free for momentlarge to claim on spinup.
"""

# ── Runtime knobs ────────────────────────────────────────────────────
experiment = {
    'trace':                 'poisson_per_task',
    'duration':              480,
    'max_batch_wait_ms':     0,
    'max_batch_size':        3,
    'isolation_mode':        'shared',
    'warmup_gap':            2.0,
    'pretrace_warmup_secs':  15.0,
    'max_model_len':         256,
    'batch_mode':            'fixedpoint',
}

# ── Hardware inventory ───────────────────────────────────────────────
devices = {
    'device1': {'type': 'NVIDIA A2', 'mem': 15360, 'ip': '192.168.245.193',
                'site_manager': 'site2', 'cuda': 'cuda:0', 'tpcs': 5},
    'device2': {'type': 'NVIDIA A2', 'mem': 15360, 'ip': '192.168.245.194',
                'site_manager': 'site2', 'cuda': 'cuda:0', 'tpcs': 5},
}

# ── Canonical task metadata ──────────────────────────────────────────
factor = 1.5
tasks = {
    'trafficfore':  {'type': 'forecasting', 'peak_workload': 50, 'latency': 5.58*factor, 'metric': 'mae',      'value': 5.0,  'backbone': 'momentlarge', 'seed': 1000},
    'exchangefore': {'type': 'forecasting',    'peak_workload': 50, 'latency': 5.58*factor, 'metric': 'mae',      'value': 5.0,  'backbone': 'momentlarge', 'seed': 800},
    'nyudepth':     {'type': 'monocular',   'peak_workload': 50, 'latency': 20.0,        'metric': 'rmse',     'value': 0.5,  'backbone': 'dinosmall',   'seed': 1100},
}

# ── Sweep parameters ─────────────────────────────────────────────────
gpus = ['device1', 'device2']

n_apps_list = [3]

# device1 hosts apps [0,1,2,3]: 1 momentlarge tasks + 1 dinosmall.
# device2 hosts apps [4,5]: 1 dinosmall tasks.
task_pool = [
    {"task": "trafficfore",  "backbone": "momentlarge"},
    {"task": "exchangefore", "backbone": "momentlarge"},
    {"task": "nyudepth",     "backbone": "dinosmall",
     "decoder_path": "nyudepth_dinosmall_monocular"},
]

per_app_rps = 5.0

assignment = "custom"
custom_assignment = {3: [[0, 1], [2]]}

# device1 has two backbones → split TPCs 10/10. device2 has one backbone
# initially (dinosmall) but we still split 10/10 so dinosmall is pinned to
# 0..9, leaving 10..19 reserved for the momentlarge that case2 spins up
# at runtime. This avoids needing a live TPC repartition.
tpc_per_app = {}
tpc_per_app_split = {}
sharing_tpc_split = {
    3: {
        "device2": [2,3],   # dinosmall on 0..9, reserved 10..19 for cold-start momentlarge
    },
}

# For device2 the second slot has no apps initially. The generator only
# emits a deployment when there are apps in a backbone bucket, so we use
# a sentinel "phantom" backbone in the pool to force the slot. Simpler:
# leave device2 single-backbone in the base plan and accept that dinosmall
# *might* opportunistically use all 20 TPCs until the cold start. The
# spinup action explicitly passes tpc_partition=[10..19] for momentlarge
# regardless.

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
    "offload_target_port": 8010,                  # second port on device2 host
    # device2 dinosmall sits on TPCs [0,1] (first chunk of [2,3] split),
    # so the cold-started momentlarge takes the remaining [2,3,4].
    "offload_target_tpc_partition": [2, 3, 4],
    # Same task name as on device1 → router weighted-splits by request_per_sec.
    "mirror_task_name": "trafficfore",

    # Timeline
    "bump_at_s":      30,
    "spinup_at_s":    30,                      # start backbone (cold)
    "attach_at_s":    30,                      # backbone warm enough; attach decoder
    "stop_at_s":      120,

    # RPS profile
    "baseline_rps":   5.0,
    "bumped_rps":     50.0,
    "mirror_share_post_attach": 0.5,
}

active_tasks = sorted({e["task"] for e in task_pool})
