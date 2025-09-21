from inference_serving.config import *
from inference_serving.methods import proteus, our
import os
os.environ["GRB_LICENSE_FILE"] = "../../gurobi/gurobi.lic"
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def generate_task(num_of_task:int):
    keys = np.random.choice(list(tasks.keys()), num_of_task, replace=False)
    random_tasks = {str(k): tasks[k] for k in keys}
    return random_tasks

slo_violations=[]
i=5
for i in range(1,len(tasks),1):
    random_tasks=generate_task(i)
    result=our.our(devices,models,random_tasks,2)   
    print(result)

    task_to_devices = defaultdict(list)
    for device, task in result['Query Assignments']:
        task_to_devices[task].append(device)

    # Step 2: Bipartite matching: assign each task to ONE device
    device_assigned_tasks = defaultdict(list)
    assigned_device = {}  # task → device
    used = {}  # device → task (only one task per device)

    def can_assign(task, visited):
        for device in task_to_devices[task]:
            if device in visited:
                continue
            visited.add(device)

            # If device is free OR can reassign its current task
            if device not in used or can_assign(used[device], visited):
                assigned_device[task] = device
                used[device] = task  # this device now has this task
                return True
        return False

    slo_violation=0
    # Try assigning all tasks
    for task in task_to_devices:
        visited = set()
        if not can_assign(task, visited):
            slo_violation+=1
    slo_violations.append(slo_violation)

plt.figure()
fig, ax1 = plt.subplots()
ax1.plot(range(1,len(tasks),1),slo_violations, color='blue', label='Primary Data')
ax1.set_xlabel('#tasks')
ax1.set_ylabel('#SLO violation', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
fig.tight_layout() # Adjust layout to prevent labels from overlapping
plt.savefig("exp2.png")