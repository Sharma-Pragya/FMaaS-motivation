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
    decoder_name=f"{pipeline['decoder']}_{pipeline['backbone']}_{pipeline['task']}"
    task_name=f"{pipeline['task']}_{pipeline['backbone']}_{pipeline['decoder']}"
    return {backbone_name:components[backbone_name]['mem'],decoder_name:components[decoder_name]['mem'],task_name:components[task_name]['mem']}

def select_active_backbones(task,task_pipelines,deployment_plans):
    active_plan=[]
    for plan in deployment_plans:
        deployed_components=plan['components']
        for id,pipeline in task_pipelines.items():
            if pipeline['backbone'] in deployed_components:
                pipeline_metric=get_pipeline_task_accuracy(id)
                pipeline_latency=get_pipeline_task_device_latency(id, plan['type'])
                plan_util=plan['util']
                plan_server=plan['name']
                print(-pipeline_metric, pipeline_latency,plan_util,id, plan)
                if task['metric']=='accuracy':
                    if pipeline_metric >= task['value'] and pipeline_latency <= task['latency']: 
                        heapq.heappush(active_plan, (-pipeline_metric, pipeline_latency,plan_util,plan_server,id, plan))
                elif task['metric']=='mae':
                    if pipeline_metric <= task['value'] and pipeline_latency <= task['latency']: 
                        heapq.heappush(active_plan, (pipeline_metric, pipeline_latency,plan_util,plan_server,id, plan))
    return active_plan

def distribute_demand(task_name,task,active_deployments):
    total_requested_workload=task['peak_workload']
    task_demand=total_requested_workload
    task_type=task['type']
    temp_plan={}
    while task_demand and active_deployments:
        plan=heapq.heappop(active_deployments)
        pipeline_latency=plan[1]
        cap = plan[2]
        pipeline_id=plan[4]
        left_cap=1-cap
        if left_cap > 0:
            task_cap=task_demand*pipeline_latency/1000
            allocated_cap=min(left_cap, task_cap)
            allocated_demand=allocated_cap*1000/pipeline_latency
            task_demand -= allocated_demand
            temp_plan.update({plan[5]['name']:{'name': plan[5]['name'], 'type': plan[5]['type'], 'mem': plan[5]['mem'],'components':get_pipeline_components_mem(pipelines[pipeline_id]),'task_info':{task_name: {'type':task_type,'total_requested_workload':total_requested_workload,'request_per_sec':allocated_demand}},'util': cap+allocated_cap}})
    return task_demand,temp_plan

def select_lowutil_server(servers):
    # Select the server with the lowest utilization
    min_util = float('inf')
    selected_server = None
    for server in servers:
        if server['util'] < min_util:
            min_util = server['util']
            selected_server = server
    return selected_server

def deployment_plan(task_name,task, task_pipelines, servers, deployment_plans):
    task_active_deployments=select_active_backbones(task, task_pipelines,deployment_plans)
    task_demand,temp_plan=distribute_demand(task_name,task,task_active_deployments)
    for id,pipeline in task_pipelines.items():
        if task_demand!=0 and len(servers)>0:
            s = select_lowutil_server(servers)
            #only consider pipeline which satisfy task latency slo and accuracy requirement
            pipeline_metric=get_pipeline_task_accuracy(id)
            pipeline_latency=get_pipeline_task_device_latency(id, s['type'])
            if task['metric']=='accuracy':
                if pipeline_metric >= task['value'] and pipeline_latency <= task['latency']: 
                    heapq.heappush(task_active_deployments, (-pipeline_metric,pipeline_latency,s['util'],s['name'],id, {'name': s['name'], 'type': s['type'],'mem':s['mem'],'ip':s['ip'],'site_manager':s['site_manager'],'components':get_pipeline_components_mem(pipeline),'tasks': [],'util': s['util']}))  
                    task_demand,temp_plan=distribute_demand(task_name,task,task_active_deployments)

            elif task['metric']=='mae':
                if pipeline_metric <= task['value'] and pipeline_latency <= task['latency']: 
                    heapq.heappush(task_active_deployments, (-pipeline_metric,pipeline_latency,s['util'],s['name'], id, {'name': s['name'], 'type': s['type'],'mem':s['mem'],'ip':s['ip'],'site_manager':s['site_manager'],'components':get_pipeline_components_mem(pipeline),'tasks': [],'util': s['util']}))  
                    task_demand,temp_plan=distribute_demand(task_name,task,task_active_deployments)

    if task_demand!=0:
        deployment_plans,redeploy_flag=fit(deployment_plans)
        if redeploy_flag==True:
            print("Trying redeployment")
            deployment_plan(task_name,task, task_pipelines, servers, deployment_plans)
        else:
            print(f"Unable to satisfy {task_name} task demand {task_demand} with current backbones and servers.")
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

def fit(deployment_plans):
    #select a backbone with minimum number of task deployed on a server
    #[{'name': 'device1', 'type': 'A16', 'mem': 16000, 'components': {'chronosmini': 57.957376, 'mlp_chronosmini_heartrate': 0.198144, 'heartrate_chronosmini_mlp': 84.86956956, 'mlp_chronosmini_sysbp': 0.198144, 'sysbp_chronosmini_mlp': 84.86956956, 'mlp_chronosmini_ecgclass': 0.200192, 'ecgclass_chronosmini_mlp': 28.13260789}, 'task_info': {'heartrate': {'type': 'regression', 'total_requested_workload': 25, 'request_per_sec': 25.0}, 'sysbp': {'type': 'regression', 'total_requested_workload': 25, 'request_per_sec': 25.0}, 'ecgclass': {'type': 'classification', 'total_requested_workload': 25, 'request_per_sec': 22.741736416325516}}, 'util': 1.0}, {'name': 'device2', 'type': 'A16', 'mem': 16000, 'components': {'chronostiny': 33.828864, 'mlp_chronostiny_diasbp': 0.132608, 'diasbp_chronostiny_mlp': 43.068878049999995, 'mlp_chronostiny_ecgclass': 0.134656, 'ecgclass_chronostiny_mlp': 16.75775989, 'mlp_chronostiny_gestureclass': 0.137216, 'gestureclass_chronostiny_mlp': 16.75903986}, 'task_info': {'diasbp': {'type': 'regression', 'total_requested_workload': 25, 'request_per_sec': 25.0}, 'ecgclass': {'type': 'classification', 'total_requested_workload': 25, 'request_per_sec': 2.258263583674484}, 'gestureclass': {'type': 'classification', 'total_requested_workload': 25, 'request_per_sec': 25.0}}, 'util': 0.4561525863576539}]
    min_task=100000000000
    for plan in deployment_plans:
        #minimum task deployment
        #size of dictionary plan['task_info']
        if len(plan['task_info'])<min_task:
            redeploy_plan=plan
            min_task=len(plan['task_info'])
    for component in redeploy_plan['components'].keys():
        for pipeline in pipelines.values():
            if pipeline['backbone']==component:
                deployed_backbone = component
    deployed_server_type=redeploy_plan['type']

    new_backbone=None
    #decrease size of backbone
    for component_name,component in components.items():
        try:
            if component['mem']<components[deployed_backbone]['mem'] and component['type']==components[deployed_backbone]['type']:
                new_backbone=component_name
        except:
            pass
    print(new_backbone)
    #new backbone not found then end program saying not any smaller backbone found
    if new_backbone is None:
        print("No smaller backbone found. Cannot redeploy to a smaller backbone.")
        return deployment_plans, False

    #try to deploy task
    #find all the pipelines with new_backbone for all tasks
    new_components={new_backbone:components[new_backbone]['mem']}
    new_util=0
    for task_name, task in redeploy_plan['task_info'].items():
        for id,pipeline in pipelines.items():
            if pipeline['backbone']==new_backbone and pipeline['task']==task_name:
                decoder_name=f"{pipeline['decoder']}_{new_backbone}_{task_name}"
                component_task_name=f"{task_name}_{new_backbone}_{pipeline['decoder']}"
                new_components.update({decoder_name:components[decoder_name]['mem'],component_task_name:components[component_task_name]['mem']})
                new_util+=task['request_per_sec']*latency[id][deployed_server_type]/1000
    deployment_plans.remove(redeploy_plan)
    redeploy_plan['components']=new_components
    redeploy_plan['util']=new_util
    deployment_plans.append(redeploy_plan)
    return deployment_plans, True


def shared_packing(devices, tasks):
    servers=[{'name':name, 'type': device['type'], 'mem': device['mem'],'ip': device['ip'],'site_manager': device['site_manager'],'util':0} for name, device in devices.items()]
    deployment_plans = []
    for task_name, task in tasks.items():
        task_pipelines=get_pipelines_task(task_name)
        deployment_plan(task_name, task, task_pipelines, servers, deployment_plans)
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