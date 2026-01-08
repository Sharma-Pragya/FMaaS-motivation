import heapq
from parser.profiler import *


def get_pipelines_task(task_name):
    components={}
    for id,pipeline in pipelines.items():
        if pipeline['task'] == task_name:
            components[id]=pipeline
    return components

def get_pipeline_task_accuracy(id):
    return metric[id]

def get_pipeline_task_device_latency(id, device_type):
    return latency[id][device_type]


def get_pipeline_components_mem(pipeline):
    backbone_name=f"{pipeline['backbone']}"
    decoder_name=f'{pipeline['decoder']}_{pipeline['backbone']}_{pipeline['task']}'
    task_name=f'{pipeline['task']}_{pipeline['backbone']}_{pipeline['decoder']}'
    return {backbone_name:components[backbone_name]['mem'],decoder_name:components[decoder_name]['mem'],task_name:components[task_name]['mem']}

def select_active_backbones(task,pipelines,deployment_plans):
    active_plan=[]
    for plan in deployment_plans:
        deployed_components=plan['components']
        for id,pipeline in pipelines.items():
            if pipeline['backbone'] in deployed_components:
                pipeline_metric=get_pipeline_task_accuracy(id)
                pipeline_latency=get_pipeline_task_device_latency(id, plan['type'])
                if task['metric']=='accuracy':
                    if pipeline_metric >= task['value'] and pipeline_latency <= task['latency']: 
                        heapq.heappush(active_plan, (-pipeline_metric, pipeline_latency,id, plan))
                elif task['metric']=='mae':
                    if pipeline_metric <= task['value'] and pipeline_latency <= task['latency']: 
                        heapq.heappush(active_plan, (pipeline_metric, pipeline_latency,id, plan))
    return active_plan

def distribute_demand(task_name,task,active_deployments):
    total_requested_workload=task['peak_workload']
    task_demand=total_requested_workload
    task_type=task['type']
    temp_plan={}
    while task_demand and active_deployments:
        plan=heapq.heappop(active_deployments)
        pipeline_latency=plan[1]
        pipeline_id=plan[2]
        cap = plan[3]['util']
        left_cap=1-cap
        if left_cap > 0:
            task_cap=task_demand*pipeline_latency/1000
            allocated_cap=min(left_cap, task_cap)
            allocated_demand=allocated_cap*1000/pipeline_latency
            task_demand -= allocated_demand
            temp_plan.update({plan[3]['name']:{'name': plan[3]['name'], 'type': plan[3]['type'], 'mem': plan[3]['mem'],'components':get_pipeline_components_mem(pipelines[pipeline_id]),'task_info':{task_name: {'type':task_type,'total_requested_workload':total_requested_workload,'request_per_sec':allocated_demand}},'util': cap+allocated_cap}})
    if task_demand==0:
        return 0,temp_plan
    return task_demand,None

def select_lowutil_server(servers):
    # Select the server with the lowest utilization
    min_util = float('inf')
    selected_server = None
    for server in servers:
        if server['util'] < min_util:
            min_util = server['util']
            selected_server = server
    return selected_server

def deployment_plan(task_name,task, pipelines, servers, deployment_plans):
    task_active_deployments=select_active_backbones(task, pipelines,deployment_plans)
    task_demand,temp_plan=distribute_demand(task_name,task,task_active_deployments)
    for id,pipeline in pipelines.items():
        if task_demand!=0 and len(servers)>0:
            s = select_lowutil_server(servers)
            #only consider pipeline which satisfy task latency slo and accuracy requirement
            pipeline_metric=get_pipeline_task_accuracy(id)
            pipeline_latency=get_pipeline_task_device_latency(id, s['type'])
            if task['metric']=='accuracy':
                if pipeline_metric >= task['value'] and pipeline_latency <= task['latency']: 
                    heapq.heappush(task_active_deployments, (-pipeline_metric,pipeline_latency, id, {'name': s['name'], 'type': s['type'],'mem':s['mem'],'ip':s['ip'],'site_manager':s['site_manager'],'components':get_pipeline_components_mem(pipeline),'tasks': [],'util': s['util']}))  
                    task_demand,temp_plan=distribute_demand(task_name,task,task_active_deployments)

            elif task['metric']=='mae':
                if pipeline_metric <= task['value'] and pipeline_latency <= task['latency']: 
                    heapq.heappush(task_active_deployments, (-pipeline_metric,pipeline_latency, id, {'name': s['name'], 'type': s['type'],'mem':s['mem'],'ip':s['ip'],'site_manager':s['site_manager'],'components':get_pipeline_components_mem(pipeline),'tasks': [],'util': s['util']}))  
                    task_demand,temp_plan=distribute_demand(task_name,task,task_active_deployments)

    if task_demand!=0:
        print(f"Unable to satisfy {task_name} task demand {task_demand} with current backbones and servers.")

    else:
        flag=False
        for server_name, plan in temp_plan.items():
            for existing_plan in deployment_plans:
                if existing_plan['name'] == server_name:
                    existing_plan['components'].update(plan['components'])
                    existing_plan['task_info'].update(plan['task_info'])
                    existing_plan['util'] =  plan['util']
                    flag= True
                    break
            if not flag:
                deployment_plans.append(plan)


def shared_packing(devices, tasks):
    servers=[{'name':name, 'type': device['type'], 'mem': device['mem'],'ip': device['ip'],'site_manager': device['site_manager'],'util':0} for name, device in devices.items()]
    deployment_plans = []
    for task_name, task in tasks.items():
        pipelines=get_pipelines_task(task_name)
        deployment_plan(task_name, task, pipelines, servers, deployment_plans)
        #update util for each server based on the deployment plan
        for plan in deployment_plans:
            for server in servers:
                if server['name'] == plan['name']:
                    server['util'] = plan['util']
                if server['util'] ==1:
                    ##remove server from consideration for future tasks
                    servers.remove(server)
    print("Final Deployment Plan:")
    print(deployment_plans)
if __name__ == "__main__":
    from user_config import devices, tasks
    shared_packing(devices, tasks)