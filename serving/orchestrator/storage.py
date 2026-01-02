import os
def write_data_to_file(site,payload):   
    site_requests = {}  
    with open('site_requests.csv', 'r') as f:
        next(f)  # skip header
        for line in f:
            site_manager, device, backbone, req_id, task, req_time = line.strip().split(',')
            if site_manager not in site_requests:
                site_requests[site_manager] = {}
            site_requests[site_manager][int(req_id)] = {
                'device': device,
                'backbone': backbone,
                'task':task,
                'req_time': float(req_time)
            }
            
    latency_data=payload.get('latency',[])
    if latency_data:
        filename='request_latency_results.csv'
        file_exists = os.path.isfile(filename)
        with open(filename, 'a') as f:
            if not file_exists:
                f.write('req_id,req_time,site_manager,device,backbone,task,end_to_end_latency(ms),proc_time(ms),swap_time(ms),pred,true\n')
            for record in latency_data:
                req_id=record[0]
                req_time = site_requests[site][req_id]['req_time']
                site_manager=site
                device = site_requests[site][req_id]['device']
                backbone=site_requests[site][req_id]['backbone']
                task= site_requests[site][req_id]['task']
                latency=record[2]*1000  # convert to milliseconds
                proc_time=record[3]*1000  # convert to milliseconds
                swap_time=record[4]*1000  # convert to milliseconds
                pred=record[5]
                true=record[6]
                f.write(f'{req_id},{req_time},{site_manager},{device},{backbone},{task},{latency},{proc_time},{swap_time},{pred},{true}\n')
            