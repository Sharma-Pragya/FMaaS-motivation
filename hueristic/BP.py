from config import *
import math
def first_fit_binpack(capacity):
    servers=[]
    for task,task_info in tasks.items():
        for p,pipeline in pipelines.items():
            if task in pipeline['architecture']:
                task_pipeline=pipeline
                task_latency=pipeline['latency']

        cap=task_latency*task_info['peak_workload']
        if cap>1:
            redundant_servers=math.ceil(cap)
            for i in range(redundant_servers):
                for component in task_pipeline['architecture']:
                    servers.append({component:components[component]['mem']})
        else:
            for component in task_pipeline['architecture']:
                servers.append({component:components[component]['mem']})
    return servers

def packing():
    P=16000
    servers=first_fit_binpack(P)
    print(servers)


packing()