experiment = {
    # float -> split equally across all tasks
    # list  -> per-task rates in sorted task order:
    # ['diasbp', 'ecgclass', 'eclfore', 'etth1fore', 'exchangefore',
    #  'gestureclass', 'heartrate', 'sysbp', 'trafficfore', 'weatherfore']
    # Example: [2,2,2,2,2,2,2,2,2,2] for 2 req/s on every task.
    'req_rate': [2,2,2,2,2,2,2,2,2,2],
    'trace': 'poisson_per_task',
    'duration': 180,
    'max_batch_wait_ms': 0,
    'isolation_mode': 'shared',
    'warmup_gap': 2.0,
    'max_model_len': 256,
    'batch_mode': 'fixedpoint',
}

devices = {
  'device1': {
    'type': 'NVIDIA L4',
    'mem': 23034,  # in MB
    'ip': '172.31.40.135',
    'site_manager':'site2',
    'cuda': 'cuda:0',
  },
  'device2': {
    'type': 'NVIDIA L4',
    'mem': 23034,  # in MB
    'ip': '172.31.42.84',
    'site_manager':'site2',
    'cuda': 'cuda:0',
  },
  'device3': {
    'type': 'NVIDIA L4',
    'mem': 23034,  # in MB
    'ip': '172.31.36.15',
    'site_manager':'site1',
    'cuda': 'cuda:0',
  },
  'device4': {
    'type': 'NVIDIA L4',
    'mem': 23034,  # in MB
    'ip': '172.31.47.59',
    'site_manager':'site1',
    'cuda': 'cuda:0',
  },
}
factor=1.5
tasks={
    'heartrate':
    {
    'type':'regression',
    'peak_workload':50,
    'latency':5.58*factor, #(5.58,239.15)
    'metric':'mae',
    'value':100,
    'backbone': 'momentlarge',
    'seed': 100,
    },
    'sysbp':
    {
    'type':'regression',
    'peak_workload':50,
    'latency':5.55*factor, #(5.55,239.08)
    'metric':'mae',
    'value': 100,
    'backbone': 'momentlarge',
    'seed': 200,
    },
    'diasbp':
    {
    'type':'regression',
    'peak_workload':50,
    'latency':5.58*factor,#(5.58,238.95)
    'metric':'mae',
    'value':100,
    'backbone': 'momentlarge',
    'seed': 300,
    },
    'ecgclass':
    {
    'type':'classification',
    'peak_workload':50,
    'latency':3.86*factor, #(3.86,86.34)
    'metric':'accuracy',
    'value':0.7,
    'backbone': 'momentlarge',
    'seed': 400,
    },
    'gestureclass':
    {
    'type':'classification',
    'peak_workload':50,
    'latency':3.88*factor, #(3.88,86.53)
    'metric':'accuracy',
    'value':0.6,
    'backbone': 'momentlarge',
    'seed': 500,
    },
    'etth1fore':{
    'type':'forecasting',
    'peak_workload':50,
    'latency':5.58*factor, #(5.58,239.15)
    'metric':'mae',
    'value': 5.0,
    'backbone': 'momentlarge',
    'seed': 600,
    },
    'weatherfore':{
    'type':'forecasting',
    'peak_workload':50,
    'latency':5.58*factor, #(5.58,239.15)
    'metric':'mae',
    'value': 5.0,
    'backbone': 'momentlarge',
    'seed': 700,
    },
    'exchangefore':{
    'type':'forecasting',
    'peak_workload':50,
    'latency':5.58*factor, #(5.58,239.15)
    'metric':'mae',
    'value': 5.0,
    'backbone': 'momentlarge',
    'seed': 800,
    },
    'eclfore':{
    'type':'forecasting',
    'peak_workload':50,
    'latency':5.58*factor, #(5.58,239.15)
    'metric':'mae',
    'value': 5.0,
    'backbone': 'momentlarge',
    'seed': 900,
    },
    'trafficfore':{
    'type':'forecasting',
    'peak_workload':50,
    'latency':5.58*factor, #(5.58,239.15)
    'metric':'mae',
    'value': 5.0,
    'backbone': 'momentlarge',
    'seed': 1000,
    },

}