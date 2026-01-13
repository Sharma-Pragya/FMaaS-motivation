import heapq
import json
import copy
from hueristic.parser.profiler import *

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

def select_active_backbones(task, task_pipelines, deployment_plans):
    """Algorithm 1 - Line 1: Select active backbone/server pairs."""
    active_plan = []
    for plan in deployment_plans:
        deployed_components = plan['components']
        for id, pipeline in task_pipelines.items():
            if pipeline['backbone'] in deployed_components:
                pipeline_metric = get_pipeline_task_accuracy(id)
                pipeline_latency = get_pipeline_task_device_latency(id, plan['type'])
                # Check SLOs
                if task['metric'] == 'accuracy':
                    if pipeline_metric >= task['value'] and pipeline_latency <= task['latency']:
                        heapq.heappush(active_plan, (-pipeline_metric, pipeline_latency, plan['util'], plan['name'], id, plan))
                elif task['metric'] == 'mae':
                    if pipeline_metric <= task['value'] and pipeline_latency <= task['latency']:
                        heapq.heappush(active_plan, (pipeline_metric, pipeline_latency, plan['util'], plan['name'], id, plan))
    return active_plan

def distribute_demand(task_name, task, active_deployments):
    """Algorithm 1 - Line 2 & 6: Distribute demand across sorted candidates."""
    total_requested_workload = task['peak_workload']
    task_demand = total_requested_workload
    task_type = task['type']
    temp_plan = {}
    
    # We work on a copy of the heap to avoid polluting the caller's state
    working_heap = list(active_deployments)
    heapq.heapify(working_heap)
    
    while task_demand > 1e-9 and working_heap:
        plan_tuple = heapq.heappop(working_heap)
        pipeline_latency = plan_tuple[1]
        current_util = plan_tuple[2]
        pipeline_id = plan_tuple[4]
        plan_ref = plan_tuple[5]
        server_name = plan_ref['name']
        
        left_cap = 1.0 - current_util
        if left_cap > 0:
            # Capacity needed for 1 request/sec: latency/1000
            task_cap_needed = task_demand * pipeline_latency / 1000.0
            allocated_cap = min(left_cap, task_cap_needed)
            allocated_demand = allocated_cap * 1000.0 / pipeline_latency
            
            task_demand -= allocated_demand

            temp_plan.update({plan_ref['name']:{
                    'name': plan_ref['name'],
                    'ip': plan_ref.get('ip'),
                    'site_manager': plan_ref.get('site_manager'),
                    'type': plan_ref['type'],
                    'mem': plan_ref['mem'],
                    'components': get_pipeline_components_mem(pipelines[pipeline_id]),
                    'task_info': {
                        task_name: {
                            'type': task_type,
                            'total_requested_workload': total_requested_workload,
                            'request_per_sec': allocated_demand
                        }
                    },
                    'util': current_util + allocated_cap
                }
            }
            )

    return max(0, task_demand), temp_plan

def select_lowutil_server(servers):
    if not servers: return None
    return min(servers, key=lambda x: x['util'])

def commit_plan(temp_plan, deployment_plans):
    """Helper to merge the temp_plan into the main deployment_plans."""
    for server_name, plan_data in temp_plan.items():
        found = False
        for existing in deployment_plans:
            if existing['name'] == server_name:
                existing['components'].update(plan_data['components'])
                existing['task_info'].update(plan_data['task_info'])
                # # If task already exists on this server (from a different backbone), sum it
                # for t_name, t_info in plan_data['task_info'].items():
                #     if t_name in existing['task_info']:
                #         existing['task_info'][t_name]['request_per_sec'] += t_info['request_per_sec']
                #     else:
                #         existing['task_info'][t_name] = t_info
                existing['util'] = plan_data['util']
                found = True
                break
        if not found:
            deployment_plans.append(plan_data)

def deployment_plan(task_name, task, task_pipelines, servers, deployment_plans):
    """Algorithm 1: Place_Task"""
    # 1. Select active backbones and try to distribute
    d = select_active_backbones(task, task_pipelines, deployment_plans)
    task_demand, temp_plan = distribute_demand(task_name, task, d)
    
    if task_demand <= 1e-9:
        commit_plan(temp_plan, deployment_plans)
        return
    # To prevent duplicates, keep track of (pipeline_id, server_name) already in 'd'
    seen_candidates = set()
    for item in d:
        # item: (priority, latency, util, server_name, pipeline_id, plan_ref)
        seen_candidates.add((item[4], item[3]))
    
    # 3. Iterate over backbones & servers if demand not satisfied
    for id, pipeline in task_pipelines.items():
        if task_demand <= 1e-9:
            break
            
        if not servers:
            break
            
        s = select_lowutil_server(servers)
        pipeline_metric = get_pipeline_task_accuracy(id)
        pipeline_latency = get_pipeline_task_device_latency(id, s['type'])
        if (id, s['name']) in seen_candidates:
            continue
        # Check if this pipeline/server combo is valid
        valid = False
        if task['metric'] == 'accuracy' and pipeline_metric >= task['value'] and pipeline_latency <= task['latency']:
            valid = True
            priority = -pipeline_metric
        elif task['metric'] == 'mae' and pipeline_metric <= task['value'] and pipeline_latency <= task['latency']:
            valid = True
            priority = pipeline_metric
            
        if valid:
            # 5. Append to candidates
            new_candidate = (priority, pipeline_latency, s['util'], s['name'], id, 
                            {'name': s['name'], 'site_manager': s['site_manager'], 'ip': s['ip'], 
                             'type': s['type'], 'mem': s['mem'],'components':get_pipeline_components_mem(pipeline),'task_info':{},'util': s['util']})
            
            d.append(new_candidate)
            
            # 6. Redistribute
            task_demand, temp_plan = distribute_demand(task_name, task, d)
    print("\n")
    print(task_name,task_demand,temp_plan)
    # 7. Final Check
    if task_demand > 1e-9:
        # 8. fit()
        deployment_plans, success = fit(deployment_plans)
        if success:
            print(f"Replan: Replaning previous deployment to fit {task_name}")
            # Recursively try again
            return deployment_plan(task_name, task, task_pipelines, servers, deployment_plans)
        else:
            print(f"CRITICAL: Unable to satisfy {task_name}. Demand left: {task_demand}")
            # Even if it failed, commit what we could
            commit_plan(temp_plan, deployment_plans)
    else:
        commit_plan(temp_plan, deployment_plans)

def fit(deployment_plans):
    """Algorithm 2: fit()"""
    if not deployment_plans:
        return deployment_plans, False
        
    # 1. Select backbone with minimum tasks
    redeploy_plan = min(deployment_plans, key=lambda x: len(x['task_info']))
    
    deployed_backbone = None
    for component in redeploy_plan['components'].keys():
        for pipeline in pipelines.values():
            if pipeline['backbone'] == component:
                deployed_backbone = component
                break
    
    if not deployed_backbone:
        return deployment_plans, False

    # 2. Decrease backbone size
    new_backbone = None
    current_mem = components[deployed_backbone]['mem']
    for component_name, comp_data in components.items():
        #try and except as for every component did not write the type
        try:
            # Heuristic: Find a backbone that is smaller but of the same 'type' (e.g., Chronos)
            if comp_data['mem'] < current_mem and comp_data['type']==components[deployed_backbone]['type']:
                new_backbone = component_name
                break
        except:
            pass
    if new_backbone is None:
        return deployment_plans, False

    # 3. Apply change
    new_components = {new_backbone: components[new_backbone]['mem']}
    new_util = 0
    deployed_server_type = redeploy_plan['type']
    for task_name, t_info in redeploy_plan['task_info'].items():
        for id, pipeline in pipelines.items():
            if pipeline['backbone'] == new_backbone and pipeline['task'] == task_name:
                decoder_name = f"{pipeline['decoder']}_{new_backbone}_{task_name}"
                comp_task_name = f"{task_name}_{new_backbone}_{pipeline['decoder']}"
                new_components.update({
                    decoder_name: components[decoder_name]['mem'],
                    comp_task_name: components[comp_task_name]['mem']
                })
                new_util += t_info['request_per_sec'] * latency[id][deployed_server_type] / 1000.0
                
    redeploy_plan['components'] = new_components
    redeploy_plan['util'] = new_util
    return deployment_plans, True

def shared_packing(devices, tasks):
    servers = [{'name': name, 'type': device['type'], 'mem': device['mem'], 
                'ip': device['ip'], 'site_manager': device['site_manager'], 'util': 0} 
               for name, device in devices.items()]
    deployment_plans = []
    
    # Sort tasks by peak workload (Descending for better packing)
    sorted_tasks = sorted(tasks.items(), key=lambda x: x[1].get('peak_workload', 0), reverse=True)
    
    for task_name, task in sorted_tasks:
        task_pipelines = get_pipelines_task(task_name)
        deployment_plan(task_name, task, task_pipelines, servers, deployment_plans)
        
        # Update server utilities from the plans
        for plan in deployment_plans:
            for s in servers:
                if s['name'] == plan['name']:
                    s['util'] = plan['util']
        
        # Filter out fully saturated servers
        servers = [s for s in servers if s['util'] < 0.999]
    print(deployment_plans)
    return deployment_plans

def build_final_json(device_list):
    """
    Build deployment JSON grouped by site_manager.
    Splits multiple backbones on a single device into separate deployment entries.
    """
    sites = {}
    
    # Pre-compute lookups from the global 'pipelines' metadata
    # decoder_key: e.g., "mlp_chronostiny_ecgclass"
    decoder_to_task = {f"{v['decoder']}_{v['backbone']}_{v['task']}": v['task'] for v in pipelines.values()}
    decoder_to_fulltask = {f"{v['decoder']}_{v['backbone']}_{v['task']}": f"{v['task']}_{v['backbone']}_{v['decoder']}" for v in pipelines.values()}
    decoder_to_backbone = {f"{v['decoder']}_{v['backbone']}_{v['task']}": v['backbone'] for v in pipelines.values()}
    
    # We also need a way to map which task belongs to which backbone
    task_to_backbones = {} 
    for v in pipelines.values():
        if v['task'] not in task_to_backbones:
            task_to_backbones[v['task']] = set()
        task_to_backbones[v['task']].add(v['backbone'])

    base_port = 8001

    for d in device_list:
        site_id = d["site_manager"]
        device_type = d['type']
        device_ip = d['ip']
        all_components = d["components"]
        all_tasks = d["task_info"]
        device_util=d['util']
        
        # 1. Identify all unique backbones present on this device
        # We find components that act as backbones (keys that are backbones in pipelines)
        backbones_on_device = [comp for comp in all_components if any(v['backbone'] == comp for v in pipelines.values())]
        
        # If a device has multiple backbones, we split them
        for i, backbone in enumerate(backbones_on_device):
            # Assign a unique port for each backbone on this device
            deployment_port = base_port + i*10
            device_url = f"{device_ip}:{deployment_port}"
            
            # 2. Filter decoders that belong ONLY to this backbone
            current_decoders_list = []
            backbone_tasks = {}
            backbone_util = 0.0
            
            # We iterate through decoders found in components
            for comp_name in all_components:
                if comp_name in decoder_to_backbone and decoder_to_backbone[comp_name] == backbone:
                    task_name = decoder_to_task[comp_name]
                    full_path = decoder_to_fulltask[comp_name]
                    
                    current_decoders_list.append({
                        "task": task_name,
                        "type": all_tasks[task_name]['type'],
                        "path": full_path
                    })
                    
                    # 3. Associate the task workload with this backbone
                    if task_name in all_tasks:
                        backbone_tasks[task_name] = all_tasks[task_name]

                        # 4. Calculate this backbone's contribution to utility
                        # Since util = (demand * latency) / 1000
                        # We find the specific pipeline ID for this task+backbone
                        for pid, p_val in pipelines.items():
                            if p_val['backbone'] == backbone and p_val['task'] == task_name:
                                latency_val = latency[pid][device_type]
                                backbone_util += (all_tasks[task_name]['request_per_sec'] * latency_val) / 1000.0
            
            # Build the specific deployment entry for this backbone
            deployment = {
                "device": device_url,
                "device_type": device_type,
                "backbone": backbone,
                "decoders": current_decoders_list,
                "tasks": backbone_tasks,
                "util": round(device_util, 6) # Rounding for cleaner JSON
            }

            # Initialize site entry and append
            if site_id not in sites:
                sites[site_id] = {
                    "id": site_id,
                    "deployments": []
                }
            sites[site_id]["deployments"].append(deployment)

    return {"sites": list(sites.values())}

if __name__ == "__main__":
    from user_config import devices, tasks
    task_manifest=shared_packing(devices, tasks)
    final_json = build_final_json(task_manifest)
    output_path = "../deployment_plan.json"
    with open(output_path, "w") as f:
        json.dump(final_json, f, indent=2)