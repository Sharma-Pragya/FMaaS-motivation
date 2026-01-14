import heapq
import json
import copy
from parser.profiler import *

# --- Helpers ---

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

def get_next_available_port(device_ip, deployment_plans, base_port=8001):
    """Checks existing deployment_plans for ports assigned to this physical IP."""
    existing_ports = [
        int(p['ip'].split(':')[-1]) 
        for p in deployment_plans if p['ip'].startswith(device_ip)
    ]
    if not existing_ports:
        return base_port
    return max(existing_ports) + 10

def get_backbone_from_port(plan):
    for comp in plan['components']:
        if any(v['backbone'] == comp for v in pipelines.values()):
            return comp
    return None

def distribute_demand(task_name, task, active_deployments):
    total_requested_workload = task['peak_workload']
    task_demand = total_requested_workload
    temp_plan = {}
    
    working_heap = list(active_deployments)
    heapq.heapify(working_heap)
    
    while task_demand > 1e-9 and working_heap:
        # Tuple: (priority, latency, total_device_util, endpoint, pid, plan_ref)
        item = heapq.heappop(working_heap)
        latency_val, total_util, endpoint, pid, plan_ref = item[1], item[2], item[3], item[4], item[5]
        
        left_cap = 1.0 - total_util
        if left_cap > 1e-6:
            task_cap_needed = task_demand * latency_val / 1000.0
            allocated_cap = min(left_cap, task_cap_needed)
            allocated_demand = allocated_cap * 1000.0 / latency_val
            
            task_demand -= allocated_demand

            temp_plan[endpoint] = {
                'name': plan_ref['name'],  
                'ip': endpoint,            
                'site_manager': plan_ref['site_manager'],
                'type': plan_ref['type'],
                'mem': plan_ref['mem'],
                'components': get_pipeline_components_mem(pipelines[pid]),
                'task_info': {task_name:{
                'type': task['type'],
                'total_requested_workload': total_requested_workload,
                'request_per_sec': allocated_demand
                }
            },
                'util': total_util + allocated_cap 
            }            

    return max(0, task_demand), temp_plan

def commit_plan(temp_plan, deployment_plans):
    """Syncs the new total utilization across all ports on the same physical device."""
    for endpoint, plan_data in temp_plan.items():
        base_ip = endpoint.split(':')[0]
        new_total_util = plan_data['util']
        
        found = False
        for existing in deployment_plans:
            if existing['ip'] == endpoint:
                existing['components'].update(plan_data['components'])
                # If task already exists, increment it; else add new
                for t_name, t_info in plan_data['task_info'].items():
                    if t_name in existing['task_info']:
                        existing['task_info'][t_name]['request_per_sec'] += t_info['request_per_sec']
                    else:
                        existing['task_info'][t_name] = t_info
                found = True
                break
        if not found:
            deployment_plans.append(plan_data)
        
        # Sync total device utilization to ALL ports on this physical IP
        for p in deployment_plans:
            if p['ip'].startswith(base_ip):
                p['util'] = new_total_util

def deployment_plan(task_name, task, task_pipelines, servers, deployment_plans):
    """Algorithm 1: Place_Task"""
    d = []
    for plan in deployment_plans:
        for pid, pipeline in task_pipelines.items():
            if pipeline['backbone'] in plan['components']:
                p_metric, p_latency = metric[pid], latency[pid][plan['type']]
                if (task['metric'] == 'accuracy' and p_metric >= task['value']) or \
                   (task['metric'] == 'mae' and p_metric <= task['value']):
                    if p_latency <= task['latency']:
                        priority = -p_metric if task['metric'] == 'accuracy' else p_metric
                        d.append((priority, p_latency, plan['util'], plan['ip'], pid, plan))

    task_demand, temp_plan = distribute_demand(task_name, task, d)
    if task_demand <= 1e-9:
        commit_plan(temp_plan, deployment_plans)
        return

    seen_endpoints = {item[3] for item in d}
    for pid, pipeline in task_pipelines.items():
        if task_demand <= 1e-9 or not servers: break
        s = min(servers, key=lambda x: x['util'])        
        backbone_name = pipeline['backbone']
        existing_port = next((p['ip'] for p in deployment_plans if p['ip'].startswith(s['ip']) and backbone_name in p['components']), None)
        endpoint = existing_port if existing_port else f"{s['ip']}:{get_next_available_port(s['ip'], deployment_plans)}"

        if endpoint in seen_endpoints: 
            continue

        p_metric, p_latency = metric[pid], latency[pid][s['type']]
        if (task['metric'] == 'accuracy' and p_metric >= task['value']) or (task['metric'] == 'mae' and p_metric <= task['value']):
            if p_latency <= task['latency']:
                priority = -p_metric if task['metric'] == 'accuracy' else p_metric
                new_cand = (priority, p_latency, s['util'], endpoint, pid, 
                            {'name': s['name'], 'ip': endpoint, 'site_manager': s['site_manager'], 'type': s['type'], 'mem': s['mem'], 'util': s['util']})
                d.append(new_cand)
                seen_endpoints.add(endpoint)
                task_demand, temp_plan = distribute_demand(task_name, task, d)

    if task_demand > 1e-9:
        deployment_plans, success = fit(deployment_plans)
        if success:
            print(f"Re-fitting backbones to satisfy {task_name}")
            return deployment_plan(task_name, task, task_pipelines, servers, deployment_plans)
        else:
            print(f"CRITICAL: Resource limit reached for {task_name}")
            commit_plan(temp_plan, deployment_plans)
    else:
        commit_plan(temp_plan, deployment_plans)


def fit(deployment_plans):
    """Algorithm 2: fit()"""
    if not deployment_plans: return deployment_plans, False
    
    # 1. Select a port with minimum tasks to redeploy
    redeploy_plan = min(deployment_plans, key=lambda x: len(x['task_info']))
    deployed_backbone = get_backbone_from_port(redeploy_plan)
    if not deployed_backbone: return deployment_plans, False

    # 2. Find a smaller backbone of the same type
    new_backbone = None
    current_mem = components[deployed_backbone]['mem']
    b_type = components[deployed_backbone].get('type')
    
    for name, data in components.items():
        if data.get('mem', float('inf')) < current_mem and data.get('type') == b_type:
            new_backbone = name
            break

    if not new_backbone: return deployment_plans, False

    # 3. Apply change and recalculate utility for this port
    new_components = {new_backbone: components[new_backbone]['mem']}
    new_util_contribution = 0
    s_type = redeploy_plan['type']
    
    for t_name, t_info in redeploy_plan['task_info'].items():
        for pid, p_val in pipelines.items():
            if p_val['backbone'] == new_backbone and p_val['task'] == t_name:
                d_name, ct_name = f"{p_val['decoder']}_{new_backbone}_{t_name}", f"{t_name}_{new_backbone}_{p_val['decoder']}"
                new_components.update({d_name: components[d_name]['mem'], ct_name: components[ct_name]['mem']})
                new_util_contribution += (t_info['request_per_sec'] * latency[pid][s_type]) / 1000.0
                break
                
    # Update this specific plan's components
    redeploy_plan['components'] = new_components
    
    # Recalculate global device utility for all ports on this physical device
    base_ip = redeploy_plan['ip'].split(':')[0]
    # Sum up contribution of all other ports + new contribution of this port
    other_ports_util = sum(
        sum((t['request_per_sec'] * latency[pid][p['type']]) / 1000.0 
            for t_name, t in p['task_info'].items() 
            for pid, p_val in pipelines.items() 
            if p_val['task'] == t_name and p_val['backbone'] == get_backbone_from_port(p))
        for p in deployment_plans if p['ip'].startswith(base_ip) and p['ip'] != redeploy_plan['ip']
    )
    
    total_new_util = other_ports_util + new_util_contribution
    for p in deployment_plans:
        if p['ip'].startswith(base_ip):
            p['util'] = total_new_util

    return deployment_plans, True

def shared_packing(devices, tasks):
    #servers in form of list
    servers = [{'name': name, 'type': d['type'], 'mem': d['mem'], 'ip': d['ip'], 'site_manager': d['site_manager'], 'util': 0} for name, d in devices.items()]
    deployment_plans = []

    #iterate over tasks based on peak_workload, high workload first
    sorted_tasks = sorted(tasks.items(), key=lambda x: x[1].get('peak_workload', 0), reverse=True)
    for t_name, t_data in sorted_tasks:
        t_pipelines = get_pipelines_task(t_name)
        deployment_plan(t_name, t_data, t_pipelines, servers, deployment_plans)
        
        #deployment plan has util attach to device
        for s in servers:
            for p in deployment_plans:
                if s['name']==p['name']:
                    s['util']=p['util']

        servers = [s for s in servers if s['util'] < 0.999]
    return deployment_plans

def build_final_json(deployment_plans):
    sites = {}
    for plan in deployment_plans:
        site_id, backbone, decoders = plan["site_manager"], get_backbone_from_port(plan), []
        for comp in plan['components']:
            for v in pipelines.values():
                d_key = f"{v['decoder']}_{v['backbone']}_{v['task']}"
                if comp == d_key and v['backbone'] == backbone and v['task'] in plan['task_info']:
                    decoders.append({"task": v['task'], "type": plan['task_info'][v['task']]['type'], "path": f"{v['task']}_{v['backbone']}_{v['decoder']}"})
                    break
        deployment = {"device": plan["ip"], "device_name": plan["name"], "device_type": plan["type"], "backbone": backbone, "decoders": decoders, "tasks": plan["task_info"], "util": round(plan["util"], 6)}
        if site_id not in sites: sites[site_id] = {"id": site_id, "deployments": []}
        sites[site_id]["deployments"].append(deployment)
    return {"sites": list(sites.values())}

if __name__ == "__main__":
    from user_config import devices, tasks
    task_manifest = shared_packing(devices, tasks)
    final_json = build_final_json(task_manifest)
    with open("../deployment_plan.json", "w") as f:
        json.dump(final_json, f, indent=2)