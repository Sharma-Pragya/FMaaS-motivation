# access dataset make the request and send it to site devices

import asyncio
import time
import threading
from typing import List, Optional, Tuple, Literal
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from tqdm import tqdm
import uvicorn 
import numpy as np
import io
import base64
from fastapi import Request
import aiohttp
from timeseries.datasets.etth1 import ETTh1Dataset
from timeseries.datasets.weather import WeatherDataset
from timeseries.datasets.exchange import ExchangeDataset
from timeseries.datasets.ecg5000 import ECG5000Dataset
from timeseries.datasets.uwavegesture import UWaveGestureLibraryALLDataset
from timeseries.datasets.ppg import PPGDataset
from timeseries.datasets.illness import IllnessDataset
from timeseries.datasets.ecl import ECLDataset
from timeseries.datasets.traffic import TrafficDataset

app = FastAPI()
request_queue: asyncio.Queue = asyncio.Queue()  # async request queue

def initialize_dataloaders():
    """Initialize all dataloaders - called on startup"""
    global DATASET_LOADERS
    
    inference_config = {'batch_size': 1, 'shuffle': False}
    dir = ".."
    
    task_cfg = {'task_type': 'classification'}
    dataset_cfg = {'dataset_path': f'{dir}/dataset/ECG5000'}
    dataloader_ecg_test = DataLoader(ECG5000Dataset(dataset_cfg, task_cfg, split='test'),
                                     batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    
    task_cfg = {'task_type': 'classification'}
    dataset_cfg = {'dataset_path': f'{dir}/dataset/UWaveGestureLibraryAll'}
    dataloader_gesture_test = DataLoader(UWaveGestureLibraryALLDataset(dataset_cfg, task_cfg, split='test'),
                                     batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    
    task_cfg = {'task_type': 'regression', 'label': 'hr'}
    dataset_cfg = {'dataset_path': f'{dir}/dataset/PPG-data'}
    dataloader_hr_test = DataLoader(PPGDataset(dataset_cfg, task_cfg, split='test'),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    
    task_cfg = {'task_type': 'regression', 'label': 'diasbp'}
    dataset_cfg = {'dataset_path': f'{dir}/dataset/PPG-data'}
    dataloader_diasbp_test = DataLoader(PPGDataset(dataset_cfg, task_cfg, split='test'),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])

    task_cfg = {'task_type': 'regression', 'label': 'sysbp'}
    dataset_cfg = {'dataset_path': f'{dir}/dataset/PPG-data'}
    dataloader_sysbp_test = DataLoader(PPGDataset(dataset_cfg, task_cfg, split='test'),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])

    task_cfg = {'task_type': 'forecasting'}
    dataset_cfg = {'dataset_path': f'{dir}/dataset/ElectricityLoad-data'}
    dataloader_ecl_test = DataLoader(ECLDataset(dataset_cfg, task_cfg, split='test'),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    
    task_cfg = {'task_type': 'forecasting'}
    dataset_cfg = {'dataset_path': f'{dir}/dataset/Traffic'}
    dataloader_traffic_test = DataLoader(TrafficDataset(dataset_cfg, task_cfg, split='test'),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    
    task_cfg = {'task_type': 'forecasting'}
    dataset_cfg = {'dataset_path': f'{dir}/dataset/ILLNESS'}
    dataloader_illness_test = DataLoader(IllnessDataset(dataset_cfg, task_cfg, split='test', forecast_horizon=192),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])

    task_cfg = {'task_type': 'forecasting'}
    dataset_cfg = {'dataset_path': f'{dir}/dataset/ETTh1'}
    dataloader_etth1_test = DataLoader(ETTh1Dataset(dataset_cfg, task_cfg, split='test'),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    
    task_cfg = {'task_type': 'forecasting'}
    dataset_cfg = {'dataset_path': f'{dir}/dataset/Weather'}       
    dataloader_weather_test = DataLoader(WeatherDataset(dataset_cfg, task_cfg, split='test'),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])

    task_cfg = {'task_type': 'forecasting'}
    dataset_cfg = {'dataset_path': f'{dir}/dataset/Exchange'}
    dataloader_rate_test = DataLoader(ExchangeDataset(dataset_cfg, task_cfg, split='test'),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    
    DATASET_LOADERS = {
        "ecg_class": dataloader_ecg_test,
        "gesture_class": dataloader_gesture_test,
        "hr": dataloader_hr_test,
        "diasbp": dataloader_diasbp_test,
        "sysbp": dataloader_sysbp_test,
        "etth1": dataloader_etth1_test,
        "weather": dataloader_weather_test,
        "rate": dataloader_rate_test,
        "ecl": dataloader_ecl_test,
        "traffic": dataloader_traffic_test,
        "illness": dataloader_illness_test,
    }
    print(f"Initialized {len(DATASET_LOADERS)} dataloaders")

@app.on_event("startup")
async def startup_event():
    """Called when the FastAPI app starts"""
    initialize_dataloaders()
    # Start the GPU worker
    asyncio.create_task(_gpu_worker())

def load_dataloader(task):
    if task not in DATASET_LOADERS:
        raise ValueError(f"Unknown task: {task}")
    return DATASET_LOADERS[task]

class PredictRequest(BaseModel):
    req_id: int
    task: Literal["etth1", "weather", "rate", "hr", "ecg_class", "gesture_class", "diasbp", "sysbp", "ecl", "traffic", "illness"]
    device: str
    return_pred: bool = False

class PredictResponse(BaseModel):
    req_id: int
    y_pred: Optional[List] = None
    site_manager_arrival_time: float
    site_manager_total_time: float
    device_wait_time: float
    device_infer_time: float

async def send_request(i, server, payload):
    print("sending request ahead")
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
    site_manager_total_time=et - st
    return (i, site_manager_total_time,data)

def encode_raw(arr) -> dict:
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),  # "float32"
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
    }

async def _gpu_worker():
    while True:
        fut, req, dataloader, arrival_time, server_start = await request_queue.get()
        batch_reqs = [(fut, req, dataloader, arrival_time, server_start)]

        tasks = []
        
        for _, req, dataloader, _, _ in batch_reqs:
            batch = next(iter(dataloader))
            if len(batch) == 3:
                payload = {
                    "task": req.task,
                    'req_id': req.req_id,
                    "x": encode_raw(batch[0].numpy()),
                    "mask": encode_raw(batch[1].numpy()),
                    "y": encode_raw(batch[2].numpy()),
                }
            else:
                payload = {
                    "task": req.task,
                    'req_id': req.req_id,
                    "x": encode_raw(batch[0].numpy()),
                    "mask": None,
                    "y": encode_raw(batch[1].numpy()),
                } 
            task = asyncio.create_task(send_request(req.req_id, req.device, payload))
            tasks.append(task)
        latency = await asyncio.gather(*tasks)
        i, site_manager_total_time, data = latency[0]

        resp = {
            'req_id': i,
            'site_manager_arrival_time': arrival_time,
            'site_manager_total_time': site_manager_total_time,
            # If data contains device_wait_time and device_infer_time:
            'device_wait_time': data.get('device_wait_time'),
            'device_infer_time': data.get('device_infer_time'),
        }
        fut.set_result(resp)
        request_queue.task_done()


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    server_start = time.time()   # when FastAPI has parsed JSON into Pydantic

    fut = asyncio.get_event_loop().create_future()
    arrival = time.time()

    dataloader = load_dataloader(req.task)
    await request_queue.put((fut, req, dataloader, arrival, server_start))
    resp = await fut

    server_end = time.time()
    resp["server_total_time"] = (server_end - server_start)
    return resp

@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__": 
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=False)
    parser.add_argument('--workers', type=int, required=False)
    args = parser.parse_args()
    port = args.port if args.port else 8000
    workers = args.workers if args.workers else 1

    print(f"Starting site manager on port {port} with {workers} workers")
    uvicorn.run("sitemanager:app", host="0.0.0.0", port=port, workers=1)