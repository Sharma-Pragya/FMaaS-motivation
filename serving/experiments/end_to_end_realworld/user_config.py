"""End-to-end real-world experiment — config.

Each task = one Azure Functions HashOwner.  For a given (load_regime, N),
N hashowners are selected and handed to each scheduling method.  Methods
report how many they place vs. reject; the trace is then replayed against
the placed tasks to measure response time and throughput.

No arrivals/departures: every placed task is live for the full duration.

Hardware and task_pool / tasks definitions are reused from long_horizon
so the comparison is apples-to-apples.
"""

devices = {
    'device1': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.40.38',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device2': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.34.115', 'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device3': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.39.15',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device4': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.46.11',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device5': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.35.65',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device6': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.32.69',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device7': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.34.179', 'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device8': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.36.225', 'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device9': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.40.31',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device10': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.34.111', 'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device11': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.39.13',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device12': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.46.10',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device13': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.35.60',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device14': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.32.60',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device15': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.34.170', 'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device16': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.36.20', 'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    # add more devices here for the full experiment
}
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
    # Vision — dino family
    {'task': 'vocseg',       'backbone': 'dinosmall','tier': 'medium', 'decoder_path': None},
    {'task': 'vocseg',       'backbone': 'dinobase', 'tier': 'medium', 'decoder_path': None},
    {'task': 'nyudepth',     'backbone': 'dinolarge','tier': 'large',  'decoder_path': None},
    # {'task': 'nyudepth',     'backbone': 'dinogiant','tier': 'large',  'decoder_path': None},
]

experiment = {
    'trace':                 'maf_lh',
    'group_by':              'HashOwner',
    'maf_n_days':            14,          # use only day-1 of the MAF trace
    'duration':              600,        # experiment seconds (per scenario)
    'trace_duration':        600,        # compress MAF span into this many seconds
    'n_tasks_sweep': [ 8, 24, 48, 96, 192, 256, 512],  # number of tasks to place (must match the scenarios)

    'load_regimes':          ['low', 'medium', 'high'],
    # Real-time rate bands (req / active-window-second) defining each regime.
    # Owners with mean rate in (lo, hi] are eligible; the experiment picks N
    # of them per scenario.
    'rate_bands_req_per_s':  {
        'low':    (0.1,  1),
        'medium': (1.0,  10.0),
        'high':   (10.0, 30.0),
    },
    'maf_min_invocations':   10,        # owners must have >= this many total invocations
    'max_batch_wait_ms':     200,
    'max_batch_size':        32,
    'isolation_mode':        'shared',
    'warmup_gap':            2.0,
    'max_model_len':         256,
    'batch_mode':            'fixedpoint',
}

# All tasks are live from t=0 → duration.  Reuses long_horizon's threshold
# semantics: arrive < initial_active_secs ⇒ deployed at t=0.
timeline = {
    'idle_timeout_real_s': 300.0,
    'initial_active_secs': 1.0,         # any positive value works; we set arrive=0 everywhere
}

conditions = ['no_sharing', 'fmaas']
