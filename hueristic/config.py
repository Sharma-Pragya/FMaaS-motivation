components={
    'MOMENT-1-large':{'mem':1462.480384},
    'mlp_momentlarge_ecgclass':{'mem':0.527872},
    'mlp_momentlarge_diasbp':{'mem':0.525824},
    'mlp_momentlarge_sysbp':{'mem':0.525824},
    'mlp_momentlarge_hrbp':{'mem':0.525824},
    'ecg_classification':{'mem':22.215650816}, #inference memory
    'diastolicBP':{'mem':55.3127836098}, #inference memory
    'systolicBP':{'mem':55.3127836098},
    'heartrate_prediction':{'mem':55.3127836098}
}
pipelines={
    'p1':{'architecture':['MOMENT-1-large','mlp_momentlarge_ecgclass','ecg_classification'],'latency':0.02587,'metric':93.43},
    'p2':{'architecture':['MOMENT-1-large','mlp_momentlarge_diasbp','diastolicBP'],'latency':0.03216,'metric':9.39},
    'p3':{'architecture':['MOMENT-1-large','mlp_momentlarge_sysbp','systolicBP'],'latency':0.03284,'metric':15.55},
    'p4':{'architecture':['MOMENT-1-large','mlp_momentlarge_hrbp','heartrate_prediction'],'latency':0.03117,'metric':5.62},
}

tasks={
    'heartrate_prediction':
    {
    'peak_workload':10,
    'latency':10,
    'metric':0.9
    },
    'systolicBP':
    {
    'peak_workload':5,
    'latency':10,
    'metric':0.9        
    },
    'diastolicBP':
    {
    'peak_workload':5,
    'latency':10,
    'metric':0.9            
    },
    # 'ecg_classification':
    # {
    # 'peak_workload':5,
    # 'latency':10,
    # 'metric':0.9         
    # }
}