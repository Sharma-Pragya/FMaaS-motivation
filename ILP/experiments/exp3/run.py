from ILP.config import *
from ILP.profiler import *
from ILP.methods import proteus, our
import os
os.environ["GRB_LICENSE_FILE"] = "../../gurobi/gurobi.lic"
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import time

def generate_task(num_of_task:int,label='known'):
    if label =='known':
        keys = np.random.choice(list(tasks.keys()), num_of_known_task, replace=False)
        random_tasks = {str(k): tasks[k] for k in keys}
    elif label =='unknown':
        keys = np.random.choice(list(unknown_task.keys()), num_of_unknown_task, replace=False)
        random_tasks={str(k): unknown_task[k] for k in keys}
    return random_tasks

unopt_time=[]
opt_time=[]
opt_num_of_models=[]
unopt_num_of_models=[]
i=5
for i in range(1,len(tasks),1):
    print(i)
    num_of_unknown_task=int(i/3)
    num_of_known_task=i-num_of_unknown_task
    random_known_tasks=generate_task(num_of_known_task,'known')
    result_known=our.our(devices,models,random_known_tasks,1) 

    random_unknown_tasks=generate_task(num_of_unknown_task,'unknown')
    random_tasks={}
    random_tasks.update(random_known_tasks)
    random_tasks.update(random_unknown_tasks)
    result_unopt=our.our(devices,models,random_tasks,1) 
    unopt_time.append(result_unopt['Runtime'])
    unopt_num_of_models.append(result_unopt['Number of Models'])

    st=time.time()
    for task in random_tasks.keys():
        for device,t in result_known['Query Assignments'].keys():
            if task==t:
                continue
        else:
            found=False
            for d,model in result_known['Model Placements'].keys():
                if (model,task) in can_serve:
                    result_known['Query Assignments'].update({(d,task):1})
                    found=True
            if found is False:
                devices_used = {d for d, t in result_known['Query Assignments'].keys()}
                devices_not_used = list(set(devices.keys()) - devices_used)

                for model, t in can_serve.keys():
                    if task==t:
                        result_known['Number of Models']+=1
                        result_known['Model Placements'].update({(devices_not_used[0],model):1})
                        result_known['Query Assignments'].update({(devices_not_used[0],task):1})
                        break
    et=time.time()
    opt_time.append(et-st)
    opt_num_of_models.append(result_known['Number of Models'])

plt.figure()
fig, ax1 = plt.subplots()
ax1.plot(range(1,len(tasks),1),unopt_time, color='blue', label='Run again')
ax1.plot(range(1,len(tasks),1),opt_time, color='red', label='Incremental')
ax1.set_xlabel('#tasks')
ax1.set_ylabel('Time to adapt (sec)')
ax1.legend()
fig.tight_layout() # Adjust layout to prevent labels from overlapping
plt.savefig("exp3.png")

plt.figure()
fig, ax1 = plt.subplots()
ax1.plot(range(1,len(tasks),1),unopt_num_of_models, color='blue', label='Run again ILP')
ax1.plot(range(1,len(tasks),1),opt_num_of_models, color='red', label='Incremental')
ax1.set_xlabel('#tasks')
ax1.set_ylabel('Number of active deployment')
ax1.legend()
fig.tight_layout() # Adjust layout to prevent labels from overlapping
plt.savefig("exp4.png")