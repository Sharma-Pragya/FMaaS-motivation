import heapq
import json
import copy
from parser.profiler import *
from user_config import devices, tasks
global servers, deployments
deployments=[]
servers = [
    {'name': name, 'type': d['type'], 'mem': d['mem'], 'ip': d['ip'], 
        'site_manager': d['site_manager'], 'util': 0} 
    for name, d in devices.items()
]

def get_pipelines_task(task_name):
    components_map = {}
    for id, pipeline in pipelines.items():
        if pipeline['task'] == task_name:
            components_map[id] = pipeline
    return components_map

def get_pipeline_task_accuracy(id):
    return metric[id]

def get_pipeline_task_device_latency(id, device_type):
    return latency[id][device_type]

def get_pipeline_components_mem(pipeline):
    backbone_name = f"{pipeline['backbone']}"
    decoder_name = f"{pipeline['decoder']}_{pipeline['backbone']}_{pipeline['task']}"
    task_name = f"{pipeline['task']}_{pipeline['backbone']}_{pipeline['decoder']}"
    return {
        backbone_name: components[backbone_name]['mem'],
        decoder_name: components[decoder_name]['mem'],
        task_name: components[task_name]['mem']
    }

def commit_plan(temp_plan):
    """Syncs the new total utilization across all ports on the same physical device."""
    for endpoint, plan in temp_plan.items():
        server_name=endpoint[0]
        new_total_util = plan['util']
        found=False
        max_port=0
        for deployed in deployments:
            deployed_server_backbone_name,deployed_plan=list(deployed.items())[0]
            if deployed_server_backbone_name==endpoint:
                deployed_plan['components'].update(plan['components'])
                deployed_plan['task_info'].update(plan['task_info'])
                return 
            elif server_name==deployed_server_backbone_name[0]:
                base_ip,deployed_port = deployed_plan['ip'].split(':')
                max_port=max(max_port,int(deployed_port))
                found=True
        if not found:
            for server in servers:
                if server['name']==server_name:
                    base_ip=server['ip']
                    break
            max_port=7990
        new_ip=f"{base_ip}:{max_port+10}"
        plan['ip']=new_ip
        deployments.append({endpoint:plan})

        # Sync total device utilization to ALL ports on this physical IP
        for deployed in deployments:
            _, p = list(deployed.items())[0]
            if p['ip'].startswith(base_ip):
                p['util'] = new_total_util

def delete_plan(endpoint):
    for deployed in deployments:
        deployed_server_backbone_name,_=list(deployed.items())[0]
        if deployed_server_backbone_name==endpoint:
            deployments.remove(deployed_server_backbone_name)
    #
def decrease_backbone_size(b):
    for name, data in components.items():
        if data.get('mem', float('inf')) < components[b]['mem'] and data.get('type') == components[b].get('type'):
            new_backbone = name
            return new_backbone
    return None
   

def fit(task):
    change = []
    demand_left= task[1]['peak_workload']
    sorted_deployments = sorted(deployments,key=lambda d: len(next(iter(d.values()))['task_info']))
    m=None
    for deployed in sorted_deployments: 
        if demand_left <= 10e-9:
            break
        server_backbone_name, redeploy_plan = list(deployed.items())[0]
        s,b=server_backbone_name
        b_new = decrease_backbone_size(b) 
        if b_new:
            change.append(redeploy_plan)
            
            delete_plan((s, b))

            new_components = {b_new: components[b_new]['mem']}
            new_util_contribution = 0
            s_type = redeploy_plan['type']
            for t_name, t_info in redeploy_plan['task_info'].items():
                for pid, p_val in pipelines.items():
                    if p_val['backbone'] == b_new and p_val['task'] == t_name:
                        d_name, ct_name = f"{p_val['decoder']}_{b_new}_{t_name}", f"{t_name}_{b_new}_{p_val['decoder']}"
                        new_components.update({d_name: components[d_name]['mem'], ct_name: components[ct_name]['mem']})
                        new_util_contribution += (t_info['request_per_sec'] * latency[pid][s_type]) / 1000.0
                        break
                        
            # Update this specific plan's components
            redeploy_plan['components'] = new_components
            
            m, demand_left = deploy_task(task,do_fit=False)
            #the change did not work need to roll back change
            if demand_left==None:
                print("Rolling back")
                delete_plan((s, b_new))
                for p in change:
                    commit_plan(p)
                continue
    return m, demand_left 

def distribute_demand(task, active_backbones_endpoints):
    task_name = task[0]
    total_requested_workload= task[1]['peak_workload']
    task_demand =total_requested_workload
    temp_plan = {}
    
    while task_demand > 1e-9 and active_backbones_endpoints:
        s, b = heapq.heappop(active_backbones_endpoints) 
        # get pid based on backbone and task_name
        pipeline = None
        for pid, p_data in pipelines.items():
            if p_data['backbone'] == b and p_data['task'] == task_name:
                pipeline = p_data
                break
        
        if not pipeline:
            continue

        for server in servers:
            if server['name']==s:
                s_util=server['util']
                s_type=server['type']
                s_mem=server['mem']
                s_site_manager=server['site_manager']


        latency_val = latency[pid][s_type]
        total_util = s_util
        left_cap = 1.0 - total_util
        
        if left_cap > 1e-6:
            task_cap_needed = task_demand * latency_val / 1000.0
            allocated_cap = min(left_cap, task_cap_needed)
            allocated_demand = allocated_cap * 1000.0 / latency_val
            
            task_demand -= allocated_demand

            # Assuming 'endpoint' refers to server name or a unique ID
            endpoint = (s,b)
            temp_plan[endpoint] = {
                'name': s,      
                'backbone': b,      
                'site_manager': s_site_manager,
                'type': s_type,
                'mem': s_mem,
                'components': get_pipeline_components_mem(pipeline),
                'task_info': {
                    task_name: {
                        'type': task[1]['type'],
                        'total_requested_workload': total_requested_workload,
                        'request_per_sec': allocated_demand
                    }
                },
                'util': allocated_cap + total_util
            }            
    return temp_plan,max(0, task_demand)

def find_active_deployments(backbone_name):
    active_backbones_endpoints = []
    for deployment in deployments:
        server_backbone_name, plan = list(deployment.items())[0]
        if server_backbone_name[1] == backbone_name:
            if plan['util'] < 0.8:
                active_backbones_endpoints.append(server_backbone_name)
    return active_backbones_endpoints
            
def deploy_model(task, backbone):
    active_backbones_endpoints = find_active_deployments(backbone)
    m, demand_left = distribute_demand(task, active_backbones_endpoints)
    
    if demand_left == 0:
        return m, demand_left
        
    for s in servers:
        if s['mem']>=components[backbone]['mem']:
            active_backbones_endpoints.append((s['name'], backbone))
            m, demand_left = distribute_demand(task, active_backbones_endpoints)
            if demand_left == 0:
                return m, demand_left
    return m, demand_left

def sort_based_on_accuracy(task):
    task_name=task[0]
    task_type=task[1]['type']
    pid_backbones={}
    for pid, pipeline in pipelines.items():
        if pipeline['task']==task_name:
            pid_backbones[pid]=pipeline['backbone']
    backbones=[]
    #get list of backbones after sorting based on accuracy metric
    for pid, backbone in pid_backbones.items():
        if task_type=='classification':
            heapq.heappush(backbones,(-metric[pid],backbone))
        else:
            heapq.heappush(backbones,(metric[pid],backbone))

    return [backbone[1] for backbone in backbones]

def deploy_task(task, do_fit=True):
    # based on latency and metric slo get pipelines
    sorted_backbones=sort_based_on_accuracy(task)
    for b in sorted_backbones:
        m, demand_left = deploy_model(task, b)
        if demand_left == 0:
            return m, demand_left
    
    if do_fit:
        print("Running fit")
        m, demand_left = fit(task)
        if demand_left==0:
            return m, demand_left 
    print("Nothing worked")
    return m, None

def shared_packing(tasks):
    global servers
    # iterate over tasks based on peak_workload, high workload first
    sorted_tasks = sorted(tasks.items(), key=lambda x: x[1].get('peak_workload', 0), reverse=True)
    for t_data in sorted_tasks:
        res = deploy_task(t_data)
        if res[0]:
            commit_plan(res[0])
        # update server utility
        for s in servers:
            for p in deployments:
                deployed_server_backbone_name,deployed_plan=list(p.items())[0]
                if s['name'] ==deployed_server_backbone_name[0]:
                    s['util'] = deployed_plan['util']
        servers = [s for s in servers if s['util'] < 0.999]

def build_final_json(deployment_plans):
    sites = {}
    for deployment in deployment_plans:
        server_backbone_name, plan = list(deployment.items())[0]
        backbone=server_backbone_name[1]
        decoders=[]
        site_id=plan['site_manager']
        for comp in plan['components']:
            for v in pipelines.values():
                d_key = f"{v['decoder']}_{v['backbone']}_{v['task']}"
                if comp == d_key and v['backbone'] == backbone and v['task'] in plan['task_info']:
                    decoders.append({"task": v['task'], "type": plan['task_info'][v['task']]['type'], "path": f"{v['task']}_{v['backbone']}_{v['decoder']}"})
                    break
        deployment = {"device": plan["ip"], "device_name": plan["name"], "device_type": plan["type"], "backbone": backbone, "decoders": decoders, "tasks": plan["task_info"], "util": round(plan["util"], 6)}
        if site_id not in sites: 
            sites[site_id] = {"id": site_id, "deployments": []}
        sites[site_id]["deployments"].append(deployment)
    return {"sites": list(sites.values())}

if __name__ == "__main__":
    shared_packing(tasks)
    print(deployments)
    final_json = build_final_json(deployments)
    with open("deployment_plan.json", "w") as f:
        json.dump(final_json, f, indent=2)
    
# Input: Task 𝜏, backbone 𝑏, deployments 𝑑, Servers S.
# Output: Deployment 𝑑, mapping 𝑚.
# 1 𝑑′ ←find_active_deployments(𝜏, 𝑏, S)// Find
# deployments that have 𝑏.
# 2 𝑑.append(𝑑′);
# 3 𝑚 ←distribute_demand(𝜏, 𝑑)// Distribute
# load across current deployments.
# 4 if satisfied(𝜏, 𝑑, 𝑚) then
# 5 return 𝑑, 𝑚
# 6 for 𝑠 ∈ S do // Iterate over servers.
# 7 if 𝑠.fits(𝑏) then
# 8 𝑑.append((s, b));
# 9 𝑚 ←distribute_demand(𝜏, 𝑑)// Distribute
# load across current deployments.
# 10 if satisfied(𝜏, 𝑑, 𝑚) then
# 11 return 𝑑, 𝑚
# 12 return 𝑑, 𝑁𝑜𝑛𝑒


# Input: Task 𝜏, Server S, Fit 𝑓 .
# Output: Deployment 𝑑, mapping 𝑚, Change 𝑐.
# 1 𝑑 ← [ ];
# 2 for 𝑏 ∈ 𝜏 .𝑏𝑎𝑐𝑘𝑏𝑜𝑛𝑒𝑠 do // Iterate over
# backbones best accuracy first.
# 3 𝑑, 𝑚 ←Deploy_Model_Walid(𝜏, 𝑏, 𝑑, S);
# 4 if 𝑚 ≠ None then
# 5 return 𝑑, 𝑚, 𝑁𝑜𝑛𝑒
# 6 if f then
# 7 𝑑, 𝑚, 𝑐 ← Fit(𝜏, 𝑆);
# 8 if 𝑚 ≠ None then
# 9 return 𝑑, 𝑚, 𝑐
# 10 return None

# Input: Task 𝜏, Server S.
# Output: Deployment 𝑑, mapping 𝑚, Change 𝑐.
# 1 𝑐 ← [ ];
# 2 repeat
# 3 𝑏, 𝑠 ← select_deployment() // select a
# deployment with the minimum number
# of tasks
# 4 𝑏′ ← decrease_backbone_size(𝑏)
# // decrease size of backbone
# 5 𝑐.append((𝑏, 𝑏′, 𝑠));
# 6 𝑑, 𝑚, _ ←Deploy_Task_Walid(𝜏, 𝑆, 𝐹𝑎𝑙𝑠𝑒)
# 7 until 𝑚 ≠ 𝑁𝑜𝑛𝑒;
# 8 𝑐 ← try_restore(𝑐);
# 9 return 𝑑, 𝑚, 𝑐