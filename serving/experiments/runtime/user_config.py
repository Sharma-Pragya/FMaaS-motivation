
devices = {
  'device1': {
    'type': 'NVIDIA GEFORCE RTX 2080 TI',
    'mem': 16000,  # in MB
    'ip': '10.100.40.30',
    'site_manager':'site2',
    'cuda': 'cuda:0',
  },
  'device2': {
    'type': 'NVIDIA GEFORCE RTX 2080 TI',
    'mem': 16000,  # in MB
    'ip': '10.100.40.30',
    'site_manager':'site2',
    'cuda': 'cuda:1',
  },
  # 'device3': {
  #   'type': 'NVIDIA A16',
  #   'mem': 16000,  # in MB
  #   'ip': 'http://10.100.20.52',
  #   'site_manager':'site2',
  # },
  # 'device3': {
  #   'type': 'A16',
  #   'mem': 16000,  # in MB
  #   'ip': 'http://10.100.20.15',
  #   'site_manager':'site1'
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
    'ecgclass':            # added at runtime via EVENT 1 (add-task)
    {
    'type':'classification',
    'peak_workload':50,
    'latency':3.86*factor, #(3.86,86.34)
    'metric':'accuracy',
    'value':0.7
    },
    # 'gestureclass':        # added at runtime via EVENT 3 (add-task)
    # {
    # 'type':'classification',
    # 'peak_workload':50,
    # 'latency':3.88*factor, #(3.88,86.53)
    # 'metric':'accuracy',
    # 'value':0.6
    # },

}