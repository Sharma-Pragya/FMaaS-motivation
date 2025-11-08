
devices = {
  'device1': {
    'type': 'A16',
    'mem': 16000,  # in MB
    'ip': 'http://10.100.20.13',
    'site_manager':'site1'
  },
  'device2': {
    'type': 'A16',
    'mem': 16000,  # in MB
    'ip': 'http://10.100.20.14',
    'site_manager':'site1',
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
    'peak_workload':168,
    'latency':10,
    'metric':'mae',
    'value':100

    },
    'sysbp':
    {
    'type':'regression',
    'peak_workload':5,
    'latency':10,
    'metric':'mae',
    'value': 100   
    },
    'diasbp':
    {
    'type':'regression',
    'peak_workload':5,
    'latency':10,
    'metric':'mae',
    'value':10       
    },
    'ecgclass':
    {
    'type':'classification',
    'peak_workload':5,
    'latency':10,
    'metric':'accuracy',
    'value':0.7        
    }
}