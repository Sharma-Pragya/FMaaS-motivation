from collections import Counter
import json
import logging
from itertools import groupby
import numpy as np
import pickle
from typing import List, Tuple, Any, Dict

import numpy as np

class Request:
    def __init__(self, req_id, task, site_manager, device, req_time):
        self.req_id = req_id
        self.task = task
        # self.dataloader = dataloader
        self.site_manager=site_manager
        self.device=device
        self.req_time = req_time
        self.backbone = None


    def __repr__(self):
        return f"req_id={self.req_id}, " \
               f"task={self.task}, "  \
               f"site_manager={self.site_manager}, " \
               f"device={self.device}, " \
               f"req_time={self.req_time}"

    def to_dict(self):
        return {
            'req_id': self.req_id,
            'task': self.task,
            'site_manager': self.site_manager,
            'device': self.device,
            'backbone': self.backbone,
            'req_time': self.req_time
        }

def generate_requests(num_tasks, alpha, req_rate, cv, duration,
                      tasks,
                      seed=42, req_id_offset=0)-> List[Request]:
    np.random.seed(seed)

    tot_req = int(req_rate * duration)

    # generate adapter id
    probs = np.random.power(alpha, tot_req)
    ind = (probs * num_tasks).astype(int)

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / req_rate
    # intervals = np.random.exponential(1.0 / req_rate, tot_req)
    intervals = np.random.gamma(shape, scale, tot_req)
    for i in range(tot_req):
        tic += intervals[i]
        requests.append(Request(req_id_offset + i, tasks[ind[i]][0], tasks[ind[i]][1], tasks[ind[i]][2], tic))
    #get per task mean rate per second per task
    task_counts = Counter([req.task for req in requests])
    mean_rps_per_task: Dict[str, float] = {task: cnt / float(duration) for task, cnt in task_counts.items()}

    return requests, mean_rps_per_task, None


