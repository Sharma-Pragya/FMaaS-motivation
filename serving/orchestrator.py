#access the workload and send it to the site managers based on the deployment strategy

import argparse
import asyncio
import csv
import json
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from typing import List, Tuple

import aiohttp
from traces.gamma import generate_requests
from torch.utils.data import DataLoader
import requests
from timeseries.datasets.etth1 import ETTh1Dataset
from timeseries.datasets.weather import WeatherDataset
from timeseries.datasets.exchange import ExchangeDataset
from timeseries.datasets.ecg5000 import ECG5000Dataset
from timeseries.datasets.uwavegesture import UWaveGestureLibraryALLDataset
from timeseries.datasets.ppg import PPGDataset
from timeseries.datasets.illness import IllnessDataset
from timeseries.datasets.ecl import ECLDataset
from timeseries.datasets.traffic import TrafficDataset
import argparse
import base64
REQUEST_LATENCY=[]
#{
#'server_id':[]
#}
async def send_request(i, server,payload):
    st = time.time()
    try:
        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(server, json=payload) as resp:
                data = await resp.json()
                et = time.time()
    except Exception as e:
        et = time.time()
        print(f"Request {i} failed: {e}")  
    REQUEST_LATENCY.append((i, (et - st)))
    return (i,(et - st))

def encode_raw(arr) -> dict:
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),  # "float32"
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
    }

def to_dict(config):
    num_tasks, alpha, req_rate, cv, duration = config
    return {"num_tasks": num_tasks,
            "alpha": alpha,
            "req_rate": req_rate,
            "cv": cv,
            "duration": duration}
    
async def benchmark(
    input_requests: List[Tuple[str, str, int, int]],
    debug=True,
) -> None:
    start = time.time()
    tasks: List[asyncio.Task] = []
    for req in input_requests:
        if debug:
            print(f"{req.req_id} {req.req_time:.5f} wait {start + req.req_time - time.time():.5f} "
                  f"{req.task}")
        await asyncio.sleep(start + req.req_time - time.time())
        payload = {
            "req_id":req.req_id,
            "task": req.task,
            'device': req.device,
        } 
        task = asyncio.create_task(send_request(req.req_id,req.site_manager,payload))
        tasks.append(task)
    latency = await asyncio.gather(*tasks)
    return latency


def get_res_stats(per_req_latency, benchmark_time):
    # get throughput
    per_req_latency = [i for i in per_req_latency]
    throughput = len(per_req_latency) / benchmark_time

    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {throughput:.2f} requests/s")
    avg_latency = np.mean([latency for _, latency in per_req_latency])

    result = {"total_time": benchmark_time, 
              "throughput": throughput,
              "avg_latency": avg_latency}
    #create a dict from config

    res = {"config": to_dict(config), "result": result}
    
    return res


def run_exp(server, config, seed=42):
    num_tasks, alpha, req_rate, cv, duration = config

    inference_config = {'batch_size': 1, 'shuffle': False}
    #come from greedy or ILP
    siteManager_H='http://10.100.1.2:8000/predict'
    deviceIP="http://10.100.115.7:8000/predict"

    #task_name, siteManager, deviceIP
    tasks=[('hr',siteManager_H,deviceIP),
        ('diasbp',siteManager_H,deviceIP),
        ('sysbp',siteManager_H,deviceIP),
        ('ecg_class',siteManager_H,deviceIP),
        ('gesture_class',siteManager_H,deviceIP),
        ('ecl',siteManager_H,deviceIP),
        ('traffic',siteManager_H,deviceIP),
        ('etth1',siteManager_H,deviceIP),
        ('weather',siteManager_H,deviceIP),
        ('rate',siteManager_H,deviceIP),
    ]
    # generate requests
    requests = generate_requests(num_tasks, alpha, req_rate, cv, duration, tasks, seed)

    # benchmark
    benchmark_start_time = time.time()
    per_req_latency = asyncio.run(benchmark(requests))
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time
    res = get_res_stats(per_req_latency, benchmark_time)
    output = "result/results_client.jsonl"
    with open(output, "a") as f:
        f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--server", type=str, default="http://0.0.0.0:8000/predict")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    alpha,cv,duration = 1,1,120
    req_rate=6
    num_tasks=1
    config = (num_tasks, alpha, req_rate, cv, duration)
    _ = run_exp(args.server, config, args.seed)
