devices = {
  'A100': {
    'type': 'A100',
    'mem': 80000  # in MB
  },
  'A16': {
    'type': 'A16',
    'mem': 16000  # in MB
  },
  'A163': {
    'type': 'A16',
    'mem': 16000  # in MB
  },
    'A162': {
    'type': 'A16',
    'mem': 16000  # in MB
  },
    'A161': {
    'type': 'A16',
    'mem': 16000  # in MB
  },
  'A6000': {
    'type': 'A6000',
    'mem': 40000  # in MB
  }
}

# servers=[['A100',80000],['A16',16000],['A6000',40000]]

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
    'p1':['MOMENT-1-large','mlp_momentlarge_ecgclass','ecg_classification'],
    'p2':['MOMENT-1-large','mlp_momentlarge_diasbp','diastolicBP'],
    'p3':['MOMENT-1-large','mlp_momentlarge_sysbp','systolicBP'],
    'p4':['MOMENT-1-large','mlp_momentlarge_hrbp','heartrate_prediction'],
}

latency={
    'p1':{'A100':0.02587},
    'p2':{'A100':0.03216},
    'p3':{'A100':0.03284},
    'p4':{'A100':0.03117},
}
accuracy={
    'p1':93.43,
    'p2':9.39,
    'p3':15.55,    
    'p4':5.62,
}

tasks={
    'heartrate_prediction':
    {
    'peak_workload':100,
    'latency':10,
    'metric':'mae',
    'value':6

    },
    'systolicBP':
    {
    'peak_workload':5,
    'latency':10,
    'metric':'mae',
    'value': 16   
    },
    'diastolicBP':
    {
    'peak_workload':5,
    'latency':10,
    'metric':'mae',
    'value':10         
    },
    'ecg_classification':
    {
    'peak_workload':5,
    'latency':10,
    'metric':'accuracy',
    'value':91        
    }
}