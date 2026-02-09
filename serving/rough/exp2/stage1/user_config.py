
devices = {
  'device1': {
    'type': 'A16',
    'mem': 16000,  # in MB
    'ip': 'http://10.100.20.54',
    'site_manager':'site2'
  },
  'device2': {
    'type': 'A16',
    'mem': 16000,  # in MB
    'ip': 'http://10.100.20.48',
    'site_manager':'site2',
  },
  'device3': {
    'type': 'A16',
    'mem': 16000,  # in MB
    'ip': 'http://10.100.20.53',
    'site_manager':'site2',
  },
  'device3': {
    'type': 'A16',
    'mem': 16000,  # in MB
    'ip': 'http://10.100.20.15',
    'site_manager':'site1'
  },
  'device4': {
    'type': 'A16',
    'mem': 16000,  # in MB
    'ip': 'http://10.100.20.16',
    'site_manager':'site1'
  },
}

tasks={
    'heartrate':
    {
    'type':'regression',
    'peak_workload':10,
    'latency':10,
    'metric':'mae',
    'value':100

    },
    'sysbp':
    {
    'type':'regression',
    'peak_workload':10,
    'latency':10,
    'metric':'mae',
    'value': 100   
    },
    'diasbp':
    {
    'type':'regression',
    'peak_workload':10,
    'latency':10,
    'metric':'mae',
    'value':100       
    },
    'ecgclass':
    {
    'type':'classification',
    'peak_workload':10,
    'latency':10,
    'metric':'accuracy',
    'value':0.7        
    },
    'gestureclass':
    {
    'type':'classification',
    'peak_workload':10,
    'latency':10,
    'metric':'accuracy',
    'value':0.6        
    },

}