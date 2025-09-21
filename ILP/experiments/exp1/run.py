from ILP.config import *
from ILP.methods import proteus, our
import os
os.environ["GRB_LICENSE_FILE"] = "../../gurobi/gurobi.lic"
import random
import matplotlib.pyplot as plt

import numpy as np
def generate_task(num_of_task:int):
    keys = np.random.choice(list(tasks.keys()), num_of_task, replace=False)
    random_tasks = {str(k): tasks[k] for k in keys}
    return random_tasks

number_of_models=[]
percent_sharing=[]
redundancy=2
for i in range(1,len(tasks),1):
    random_tasks=generate_task(i)
    result=our.our(devices,models,random_tasks,redundancy=redundancy)   
    number_of_models.append(result['Number of Models'])
    percent_sharing.append(result['Number of Models']*100/(i*redundancy))
    print('Number of Models:',result['Number of Models'])
    print('Percent sharing:',result['Number of Models']*100/(i*redundancy))

plt.figure()
fig, ax1 = plt.subplots()
ax1.plot(range(1,len(tasks),1),number_of_models, color='blue', label='Primary Data')
ax1.set_xlabel('#tasks')
ax1.set_ylabel('#active deployment', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(range(1,len(tasks),1),percent_sharing, color='red', label='Secondary Data')
ax2.set_ylabel('%percent sharing', color='red')
ax2.tick_params(axis='y', labelcolor='red')

fig.tight_layout() # Adjust layout to prevent labels from overlapping
plt.savefig("exp1.png")
