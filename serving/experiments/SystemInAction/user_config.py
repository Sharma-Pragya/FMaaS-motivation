experiment = {
    'req_rate': 95.0,          # float → same rate for all tasks; list → per-task (sorted by name)
    'trace': 'poisson_per_task',
    'duration': 20,
    'seed': 42,
    'max_batch_size': 5,
    'max_batch_wait_ms': 0,
    'isolation_mode': 'shared',
    'warmup_gap': 2.0,
    'max_model_len': 256,
}

devices = {
  'device1': {
    'type': 'NVIDIA A16',
    'mem': 16000,  # in MB
    'ip': '10.100.20.51',
    'site_manager':'site2',
    'cuda': 'cuda:0',
  },
  'device2': {
    'type': 'NVIDIA A16',
    'mem': 16000,  # in MB
    'ip': '10.100.20.52',
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
    # 'value':100
    # },
    # 'sysbp':
    # {
    # 'type':'regression',
    # 'peak_workload':50,
    # 'latency':5.55*factor, #(5.55,239.08)
    # 'metric':'mae',
    # 'value': 100   
    # },
    # 'diasbp':
    # {
    # 'type':'regression',
    # 'peak_workload':50,
    # 'latency':5.58*factor,#(5.58,238.95)
    # 'metric':'mae',
    # 'value':100       
    # },
    'ecgclass':
    {
    'type':'classification',
    'peak_workload':50,
    'latency':3.86*factor, #(3.86,86.34)
    'metric':'accuracy',
    'value':0.7,
    'backbone': 'momentbase'  
    },
    'gestureclass':
    {
    'type':'classification',
    'peak_workload':50,
    'latency':3.88*factor, #(3.88,86.53)
    'metric':'accuracy',
    'value':0.6,
    'backbone': 'momentbase'          
    },
    # 'etth1fore':{
    # 'type':'forecasting',
    # 'peak_workload':50,
    # 'latency':5.58*factor, #(5.58,239.15)
    # 'metric':'mae',
    # 'value': 5.0,
    # 'backbone': 'momentbase'  
    # },
    # 'weatherfore':{
    # 'type':'forecasting',
    # 'peak_workload':50,
    # 'latency':5.58*factor, #(5.58,239.15)
    # 'metric':'mae',
    # 'value': 5.0,
    # 'backbone': 'momentbase'  
    # },
    # 'exchangefore':{
    # 'type':'forecasting',
    # 'peak_workload':50,
    # 'latency':5.58*factor, #(5.58,239.15)
    # 'metric':'mae',
    # 'value': 5.0,
    # 'backbone': 'momentbase'  
    # },
    # 'eclfore':{
    # 'type':'forecasting',
    # 'peak_workload':50,
    # 'latency':5.58*factor, #(5.58,239.15)
    # 'metric':'mae',
    # 'value': 5.0,
    # 'backbone': 'momentbase'  

    # },
    # 'trafficfore':{
    # 'type':'forecasting',
    # 'peak_workload':50,
    # 'latency':5.58*factor, #(5.58,239.15)
    # 'metric':'mae',
    # 'value': 5.0,
    # 'backbone': 'momentbase'  
    # },

}