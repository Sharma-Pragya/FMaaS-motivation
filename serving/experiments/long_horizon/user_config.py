"""Long-horizon experiment — single source of truth.

Edit ONLY this file to configure the experiment.
Everything else (generate.py, run.py) reads from here.

FMaaS needs exactly len(task_pool) devices (one per backbone type).
no_sharing placement is decided by ClipperPlacementScheduler; tasks that
don't fit on the available devices are simply not placed.

Full 14-backbone experiment: swap in the commented-out task_pool below and
add 14+ devices to `devices`.
"""

# ── Hardware ──────────────────────────────────────────────────────────────────
# One entry per physical T4.  Add more to scale up.
devices = {
    'device1': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.40.38',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device2': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.34.115', 'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device3': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.39.15',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device4': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.46.11',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device5': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.35.65',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device6': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.32.69',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device7': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.34.179', 'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device8': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.36.225', 'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    # add more devices here for the full experiment
}

# ── Task pool — one entry per task/backbone pair ──────────────────────────────
# Add entries to increase tasks. FMaaS places one backbone process per entry.
# Duplicate backbone entries (e.g. two swintiny rows) are fine — each gets its
# own Alibaba model and arrive/depart window.
# task_pool = [
#     # TSFM — papagei family
#     {'task': 'sysbp',     'backbone': 'momentlarge',  'decoder_path': 'sysbp_momentlarge_mlp'},
#     {'task': 'diasbp',    'backbone': 'momentsmall',  'decoder_path': 'diasbp_momentsmall_mlp'},
#     # Vision — swin family (linear probes: no separate decoder checkpoint)
#     {'task': 'imgclass10','backbone': 'dinolarge',  'decoder_path': None},
#     {'task': 'imgclass10','backbone': 'swinlarge',  'decoder_path': None},
# ]

# ── Full 14-backbone experiment ───────────────────────────────────────────────
# Uncomment and add 14+ devices to run the large-scale version.
task_pool = [
    # TSFM — moment family
    {'task': 'ecgclass',     'backbone': 'momentlarge',    'tier': 'large',  'decoder_path': 'ecgclass_momentlarge_mlp'},
    {'task': 'heartrate',    'backbone': 'momentbase',     'tier': 'large',  'decoder_path': 'heartrate_momentbase_mlp'},
    {'task': 'gestureclass', 'backbone': 'momentsmall',    'tier': 'medium', 'decoder_path': 'gestureclass_momentsmall_mlp'},
    # TSFM — papagei family
    {'task': 'sysbp',        'backbone': 'papageip',       'tier': 'small',  'decoder_path': 'sysbp_papageip_mlp'},
    {'task': 'diasbp',       'backbone': 'papageis',       'tier': 'small',  'decoder_path': 'diasbp_papageis_mlp'},
    {'task': 'etth1fore',    'backbone': 'papageissvri',   'tier': 'small',  'decoder_path': 'etth1fore_papageissvri_mlp'},
    # Vision — swin family
    {'task': 'imgclass10',   'backbone': 'swintiny',       'tier': 'small',  'decoder_path': None},
    {'task': 'imgclass10',   'backbone': 'swinsmall',      'tier': 'small',  'decoder_path': None},
    {'task': 'eurosatclass', 'backbone': 'swinbase',       'tier': 'medium', 'decoder_path': None},
    {'task': 'eurosatclass', 'backbone': 'swinlarge',      'tier': 'medium', 'decoder_path': None},
    # Vision — dino-patch family
    {'task': 'vocseg',       'backbone': 'dinosmall','tier': 'medium', 'decoder_path': None},
    {'task': 'vocseg',       'backbone': 'dinobase', 'tier': 'medium', 'decoder_path': None},
    {'task': 'nyudepth',     'backbone': 'dinolarge','tier': 'large',  'decoder_path': None},
    {'task': 'nyudepth',     'backbone': 'dinogiant','tier': 'large',  'decoder_path': None},
]

# ── Task metadata (for trace generation) ─────────────────────────────────────
tasks = {
    'ecgclass':     {'type': 'classification',        'peak_workload': 10, 'latency': 50,  'metric': 'accuracy', 'value': 0.7,  'seed': 100},
    'heartrate':    {'type': 'regression',            'peak_workload': 10, 'latency': 50,  'metric': 'mae',      'value': 100,  'seed': 200},
    'gestureclass': {'type': 'classification',        'peak_workload': 10, 'latency': 50,  'metric': 'accuracy', 'value': 0.6,  'seed': 300},
    'sysbp':        {'type': 'regression',            'peak_workload': 10, 'latency': 50,  'metric': 'mae',      'value': 100,  'seed': 400},
    'diasbp':       {'type': 'regression',            'peak_workload': 10, 'latency': 50,  'metric': 'mae',      'value': 100,  'seed': 500},
    'etth1fore':    {'type': 'forecasting',           'peak_workload': 10, 'latency': 50,  'metric': 'mae',      'value': 5.0,  'seed': 600},
    'imgclass10':   {'type': 'linear_classification', 'peak_workload': 10, 'latency': 50,  'metric': 'accuracy', 'value': 0.5,  'seed': 700},
    'eurosatclass': {'type': 'linear_classification', 'peak_workload': 10, 'latency': 50,  'metric': 'accuracy', 'value': 0.5,  'seed': 800},
    'vocseg':       {'type': 'linear_seg',            'peak_workload': 10, 'latency': 100, 'metric': 'miou',     'value': 0.3,  'seed': 900},
    'nyudepth':     {'type': 'monocular',             'peak_workload': 10, 'latency': 100, 'metric': 'rmse',     'value': 0.5,  'seed': 1000},
}

# ── Experiment knobs ──────────────────────────────────────────────────────────
experiment = {
    'trace':              'alibaba_lh',   # 'alibaba_lh': real Alibaba inter-arrivals (sparse for dynamic tasks)
                                          # 'maf_lh':     real Azure Functions per-minute counts (denser dynamic tasks)
    'duration':           900,           # total experiment seconds
    'trace_duration':     1800,            # compression window: arrivals in [0, trace_duration],
                                          # departs in [duration - trace_duration, duration],
                                          # IA pattern replays cyclically in between
    'n_tasks':            100,              # total task instances; task_pool entries repeat round-robin
    'maf_min_invocations': 50,            # maf_lh only: min total invocations a function must have to be selected
    'max_batch_wait_ms':  0,
    'max_batch_size':     32,
    'isolation_mode':     'shared',
    'warmup_gap':         2.0,
    'max_model_len':      256,
    'batch_mode':         'fixedpoint',
    # Trace decoupling: timing pool vs request-generation pool
    'timing_min_window_s':    30.0,  # min compressed window for timing models
    'req_model_min_rps':       1.0,  # min compressed RPS for request-generation models
    'req_model_min_window_s': 50.0,  # min compressed window for request-generation models
}

# ── Timeline — Alibaba trace driven ──────────────────────────────────────────
# arrive/depart times for each task are derived from real Alibaba model
# first/last invocation times, time-compressed to experiment['duration'].
#
# idle_timeout_real_s: seconds of inactivity (real trace time) before a model
#   is considered departed.  300 s = 5-min serverless keep-alive.
# initial_active_secs: tasks whose compressed arrive time is below this
#   threshold are deployed at t=0 (initially active).  The rest are
#   cold-started / hot-attached when their arrive event fires.
timeline = {
    'idle_timeout_real_s': 300.0,
    'initial_active_secs': 0.0,
}

conditions = ['no_sharing','fmaas']
