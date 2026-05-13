"""End-to-end real-world (mix) experiment — config.

Same setup as end_to_end_realworld, but each scenario draws a *mix* of owners
across regimes instead of a single regime. Define each scenario in
`mix_sweep` as a dict mapping regime → count, e.g. {'low': 8, 'medium': 8,
'high': 8}. Scenario directory name is derived as mix_L{low}_M{medium}_H{high}.
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
    'device9': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.44.130',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device10': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.33.84', 'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device11': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.35.104',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device12': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.43.153',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device13': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.47.61',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device14': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.39.252',  'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device15': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.36.14', 'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
    'device16': {'type': 'TESLA T4', 'mem': 15360, 'ip': '172.31.34.252', 'site_manager': 'site1', 'cuda': 'cuda:0', 'tpcs': 20},
}

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

# # Original uniform task_pool — every backbone exactly once. With 60 owners
# # this gives ~5 tasks per backbone, diluting the FMaaS sharing benefit
# # across 13 small co-tenant sets. Kept here for ablations; swap with the
# # heavy-biased pool below by commenting/uncommenting.
# task_pool = [
#     {'task': 'ecgclass',     'backbone': 'momentlarge',    'tier': 'large',  'decoder_path': 'ecgclass_momentlarge_mlp'},
#     {'task': 'heartrate',    'backbone': 'momentbase',     'tier': 'large',  'decoder_path': 'heartrate_momentbase_mlp'},
#     {'task': 'gestureclass', 'backbone': 'momentsmall',    'tier': 'medium', 'decoder_path': 'gestureclass_momentsmall_mlp'},
#     {'task': 'sysbp',        'backbone': 'papageip',       'tier': 'small',  'decoder_path': 'sysbp_papageip_mlp'},
#     {'task': 'diasbp',       'backbone': 'papageis',       'tier': 'small',  'decoder_path': 'diasbp_papageis_mlp'},
#     {'task': 'etth1fore',    'backbone': 'papageissvri',   'tier': 'small',  'decoder_path': 'etth1fore_papageissvri_mlp'},
#     {'task': 'imgclass10',   'backbone': 'swintiny',       'tier': 'small',  'decoder_path': None},
#     {'task': 'imgclass10',   'backbone': 'swinsmall',      'tier': 'small',  'decoder_path': None},
#     {'task': 'eurosatclass', 'backbone': 'swinbase',       'tier': 'medium', 'decoder_path': None},
#     {'task': 'eurosatclass', 'backbone': 'swinlarge',      'tier': 'medium', 'decoder_path': None},
#     {'task': 'vocseg',       'backbone': 'dinosmall','tier': 'medium', 'decoder_path': None},
#     {'task': 'vocseg',       'backbone': 'dinobase', 'tier': 'medium', 'decoder_path': None},
#     {'task': 'nyudepth',     'backbone': 'dinolarge','tier': 'large',  'decoder_path': None},
# ]

# Heavy-biased task_pool: full coverage but heavy backbones duplicated so
# more owners hash onto them. Hash assignment in generate.py makes
# P(owner → backbone) proportional to that backbone's count below.
#
#   dinolarge   ×4   ≈ 18%   (heaviest vision — biggest mem/process win)
#   momentlarge ×3   ≈ 14%   (heaviest tsfm)
#   swinlarge   ×3   ≈ 14%
#   dinobase    ×2   ≈  9%
#   momentbase  ×2   ≈  9%
#   swinbase    ×2   ≈  9%
#   dinosmall, momentsmall, swinsmall, swintiny,
#   papageip, papageis, papageissvri    ×1 each  (~4.5%)
#
# Mixed-regime workloads still work: regime decides which owners are
# picked; this pool decides which backbone each runs on. Orthogonal.
task_pool = [
    # dinolarge ×4
    {'task': 'nyudepth',     'backbone': 'dinolarge',      'tier': 'large',  'decoder_path': None},
    {'task': 'nyudepth',     'backbone': 'dinolarge',      'tier': 'large',  'decoder_path': None},
    {'task': 'nyudepth',     'backbone': 'dinolarge',      'tier': 'large',  'decoder_path': None},
    {'task': 'nyudepth',     'backbone': 'dinolarge',      'tier': 'large',  'decoder_path': None},
    # momentlarge ×3
    {'task': 'ecgclass',     'backbone': 'momentlarge',    'tier': 'large',  'decoder_path': 'ecgclass_momentlarge_mlp'},
    {'task': 'ecgclass',     'backbone': 'momentlarge',    'tier': 'large',  'decoder_path': 'ecgclass_momentlarge_mlp'},
    {'task': 'ecgclass',     'backbone': 'momentlarge',    'tier': 'large',  'decoder_path': 'ecgclass_momentlarge_mlp'},
    # swinlarge ×3
    {'task': 'eurosatclass', 'backbone': 'swinlarge',      'tier': 'medium', 'decoder_path': None},
    {'task': 'eurosatclass', 'backbone': 'swinlarge',      'tier': 'medium', 'decoder_path': None},
    {'task': 'eurosatclass', 'backbone': 'swinlarge',      'tier': 'medium', 'decoder_path': None},
    # dinobase ×2
    {'task': 'vocseg',       'backbone': 'dinobase',       'tier': 'medium', 'decoder_path': None},
    {'task': 'vocseg',       'backbone': 'dinobase',       'tier': 'medium', 'decoder_path': None},
    # momentbase ×2
    {'task': 'heartrate',    'backbone': 'momentbase',     'tier': 'large',  'decoder_path': 'heartrate_momentbase_mlp'},
    {'task': 'heartrate',    'backbone': 'momentbase',     'tier': 'large',  'decoder_path': 'heartrate_momentbase_mlp'},
    # swinbase ×2
    {'task': 'eurosatclass', 'backbone': 'swinbase',       'tier': 'medium', 'decoder_path': None},
    {'task': 'eurosatclass', 'backbone': 'swinbase',       'tier': 'medium', 'decoder_path': None},
    # cheap backbones — singletons (low mem/process pressure; sharing
    # benefit here is small, but coverage matters for realism)
    {'task': 'vocseg',       'backbone': 'dinosmall',      'tier': 'medium', 'decoder_path': None},
    {'task': 'gestureclass', 'backbone': 'momentsmall',    'tier': 'medium', 'decoder_path': 'gestureclass_momentsmall_mlp'},
    {'task': 'imgclass10',   'backbone': 'swinsmall',      'tier': 'small',  'decoder_path': None},
    {'task': 'imgclass10',   'backbone': 'swintiny',       'tier': 'small',  'decoder_path': None},
    {'task': 'sysbp',        'backbone': 'papageip',       'tier': 'small',  'decoder_path': 'sysbp_papageip_mlp'},
    {'task': 'diasbp',       'backbone': 'papageis',       'tier': 'small',  'decoder_path': 'diasbp_papageis_mlp'},
    {'task': 'etth1fore',    'backbone': 'papageissvri',   'tier': 'small',  'decoder_path': 'etth1fore_papageissvri_mlp'},
]

experiment = {
    'trace':                 'maf_lh',
    'group_by':              'HashOwner',
    'maf_n_days':            14,
    'duration':              600,
    'trace_duration':        600,

    'load_regimes':          ['low', 'medium', 'high'],
    'rate_bands_req_per_s':  {
        'low':    (0.1,  1),
        'medium': (1.0,  10.0),
        'high':   (10.0, 30.0),
    },

    # Each entry is a dict mapping regime → count. Scenario directory name
    # is mix_L{low}_M{medium}_H{high}. Missing regimes default to 0.
    'mix_sweep': [
        # {'low': 20,  'medium': 20,  'high': 20},
        # {'low': 40,  'medium': 40,  'high': 40},
        {'low': 35, 'medium': 30, 'high': 20}
        # {'low': 20,  'medium': 0,  'high': 0},
        # {'low': 60,  'medium': 0,  'high': 0},
        # {'low': 100,  'medium': 0,  'high': 0},
        # {'low': 20, 'medium': 20,  'high': 20},
    ],

    'maf_min_invocations':   10,
    'window_cv_max':         1.0,   # owners with within-600s CV >= this are excluded
    'max_batch_wait_ms':     0,
    'max_batch_size':        32,
    'isolation_mode':        'shared',
    'warmup_gap':            2.0,
    'max_model_len':         256,
    'batch_mode':            'fixedpoint',

    # Workload-to-backbone mapping policy used by deployments/generate.py.
    #
    # "latency_benefit" is the response-time experiment: medium-rate owners
    # can map to heavier shared backbones so FMaaS can batch and amortize the
    # expensive backbone work, similar to the TPC experiment as RPS increases.
    #
    # "fit_benefit" is the older placement/capacity stress mapping: low-rate
    # owners go to heavy backbones and high-rate owners go to cheap backbones.
    # That is useful for showing consolidation/fit, but it tends to hide the
    # runtime latency win because the expensive models see only low RPS.
    'mapping_policy':        'latency_benefit',
}

timeline = {
    'idle_timeout_real_s': 300.0,
    'initial_active_secs': 1.0,
}

conditions = ['fmaas','no_sharing']


def mix_label(mix: dict) -> str:
    """Canonical scenario directory name for a mix dict."""
    parts = []
    for r in experiment['load_regimes']:
        parts.append(f"{r[0].upper()}{int(mix.get(r, 0))}")
    return "mix_" + "_".join(parts)
