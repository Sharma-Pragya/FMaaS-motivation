experiment = {
    'req_rate': 100,          # float → same rate for all tasks; list → per-task (sorted by name)
    'trace': 'poisson_per_task',
    'duration': 20,
    'max_batch_wait_ms': 0,
    'isolation_mode': 'shared',
    'warmup_gap': 2.0,
    'max_model_len': 256,
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
    'ip': '172.31.36.15',
    'site_manager':'site2',
    'cuda': 'cuda:0',
  },
  # 'device3': {
  #   'type': 'NVIDIA A16',
  #   'mem': 16000,  # in MB
  #   'ip': '10.100.20.50',
  #   'site_manager':'site2',
  #   'cuda': 'cuda:2',
  # },
  # 'device4': {
  #   'type': 'NVIDIA A16',
  #   'mem': 16000,  # in MB
  #   'ip': '10.100.20.51',
  #   'site_manager':'site1',
  #   'cuda': 'cuda:0',
  # },
  # 'device4': {
  #   'type': 'A16',
  #   'mem': 16000,  # in MB
  #   'ip': 'http://10.100.20.16',
  #   'site_manager':'site1'
  # },
}
factor=1.5
tasks={
    # 'heartrate':
    # {
    # 'type':'regression',
    # 'peak_workload':50,
    # 'latency':5.58*factor, #(5.58,239.15)
    # 'metric':'mae',
    # 'value':100,
    # 'backbone': 'momentbase',
    # 'seed': 100,
    # },
    # 'sysbp':
    # {
    # 'type':'regression',
    # 'peak_workload':50,
    # 'latency':5.55*factor, #(5.55,239.08)
    # 'metric':'mae',
    # 'value': 100,
    # 'backbone': 'momentbase',
    # 'seed': 200,
    # },
    # 'diasbp':
    # {
    # 'type':'regression',
    # 'peak_workload':50,
    # 'latency':5.58*factor,#(5.58,238.95)
    # 'metric':'mae',
    # 'value':100,
    # 'backbone': 'momentbase',
    # 'seed': 300,
    # },
    'ecgclass':
    {
    'type':'classification',
    'peak_workload':50,
    'latency':3.86*factor, #(3.86,86.34)
    'metric':'accuracy',
    'value':0.7,
    'backbone': 'momentbase',
    'seed': 400,
    },
    'gestureclass':
    {
    'type':'classification',
    'peak_workload':50,
    'latency':3.88*factor, #(3.88,86.53)
    'metric':'accuracy',
    'value':0.6,
    'backbone': 'momentbase',
    'seed': 500,
    },
    'etth1fore':{
    'type':'forecasting',
    'peak_workload':50,
    'latency':5.58*factor, #(5.58,239.15)
    'metric':'mae',
    'value': 5.0,
    'backbone': 'momentbase',
    'seed': 600,
    },
    'weatherfore':{
    'type':'forecasting',
    'peak_workload':50,
    'latency':5.58*factor, #(5.58,239.15)
    'metric':'mae',
    'value': 5.0,
    'backbone': 'momentbase',
    'seed': 700,
    },
    'exchangefore':{
    'type':'forecasting',
    'peak_workload':50,
    'latency':5.58*factor, #(5.58,239.15)
    'metric':'mae',
    'value': 5.0,
    'backbone': 'momentbase',
    'seed': 800,
    },
    'eclfore':{
    'type':'forecasting',
    'peak_workload':50,
    'latency':5.58*factor, #(5.58,239.15)
    'metric':'mae',
    'value': 5.0,
    'backbone': 'momentbase',
    'seed': 900,
    },
    'trafficfore':{
    'type':'forecasting',
    'peak_workload':50,
    'latency':5.58*factor, #(5.58,239.15)
    'metric':'mae',
    'value': 5.0,
    'backbone': 'momentbase',
    'seed': 1000,
    },

}