# site_manager/runtime_executor.py
import asyncio, time, base64, aiohttp, numpy as np
from typing import List, Optional, Literal
from torch.utils.data import DataLoader
from site_manager.config import DATASET_DIR, DEFAULT_BATCH_SIZE

# from timeseries.datasets.etth1 import ETTh1Dataset
# from timeseries.datasets.weather import WeatherDataset
# from timeseries.datasets.exchange import ExchangeDataset
# from timeseries.datasets.ecg5000 import ECG5000Dataset
# from timeseries.datasets.uwavegesture import UWaveGestureLibraryALLDataset
from timeseries.datasets.ppg import PPGDataset
# from timeseries.datasets.illness import IllnessDataset
# from timeseries.datasets.ecl import ECLDataset
# from timeseries.datasets.traffic import TrafficDataset
from timeseries.datasets.vqa import VQADataset


DATASET_LOADERS = {}

#based on type of tasks in requests in needs to initialize dataloaders
def initialize_dataloaders():
    global DATASET_LOADERS
    inference_config = {"batch_size": DEFAULT_BATCH_SIZE, "shuffle": False}
    d = DATASET_DIR
    # create test dataloaders
    DATASET_LOADERS = {
        # "ecg_class": DataLoader(ECG5000Dataset({"dataset_path": f"{d}/ECG5000"}, {"task_type": "classification"}, "test"), **inference_config),
        # "gesture_class": DataLoader(UWaveGestureLibraryALLDataset({"dataset_path": f"{d}/UWaveGestureLibraryAll"}, {"task_type": "classification"}, "test"), **inference_config),
        "hr": DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data"}, {"task_type": "regression","label":"hr"}, "test"), **inference_config),
        # "diasbp": DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data"}, {"task_type": "regression","label":"diasbp"}, "test"), **inference_config),
        # "sysbp": DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data"}, {"task_type": "regression","label":"sysbp"}, "test"), **inference_config),
        # "ecl": DataLoader(ECLDataset({"dataset_path": f"{d}/ElectricityLoad-data"}, {"task_type": "forecasting"}, "test"), **inference_config),
        # "traffic": DataLoader(TrafficDataset({"dataset_path": f"{d}/Traffic"}, {"task_type": "forecasting"}, "test"), **inference_config),
        # "illness": DataLoader(IllnessDataset({"dataset_path": f"{d}/ILLNESS"}, {"task_type": "forecasting"}, "test", forecast_horizon=192), **inference_config),
        # "etth1": DataLoader(ETTh1Dataset({"dataset_path": f"{d}/ETTh1"}, {"task_type": "forecasting"}, "test"), **inference_config),
        # "weather": DataLoader(WeatherDataset({"dataset_path": f"{d}/Weather"}, {"task_type": "forecasting"}, "test"), **inference_config),
        # "rate": DataLoader(ExchangeDataset({"dataset_path": f"{d}/Exchange"}, {"task_type": "forecasting"}, "test"), **inference_config),
        # 'vqa': DataLoader(VQADataset({"dataset_path": f"{d}/val2014"}, {"task_type": "forecasting"}, "test") ,  **inference_config)
    }
    print(f"[RuntimeExecutor] Initialized {len(DATASET_LOADERS)} dataloaders.")

def load_dataloader(task: str):
    if task not in DATASET_LOADERS:
        raise ValueError(f'Unknown task: {task}')
    return DATASET_LOADERS[task]

def encode_raw(arr) -> dict:
    if isinstance(arr, np.ndarray):
        return {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
        }
    elif isinstance(arr, str):
        return {
            "type": "text",
            "data": arr,
        }
    elif isinstance(arr, (list, tuple)) and all(isinstance(x, str) for x in arr):
        return {
            "type": "text_list",
            "data": arr,
        }

async def send_request(i, server, payload):
    st = time.time()
    try:
        timeout = aiohttp.ClientTimeout(total=3*3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(server, json=payload) as resp:
                data = await resp.json()
                et = time.time()
    except Exception as e:
        print(f"Request {i} failed: {e}")
        data = {}
        et = time.time()
    return (i, et - st, data)

async def handle_runtime_request(req:dict):
    st=time.time()
    dataloader=load_dataloader(req['task'])
    batch=next(iter(dataloader))
    ## need to work on this for VLM and timeseries
    # payload = {
    #     "task": req.task, "req_id": req.req_id,
    #     "x": encode_raw(batch[0].numpy()),
    #     "mask": encode_raw(batch[1].numpy()) if len(batch)==3 else None,
    #     "y": encode_raw(batch[-1].numpy())
    # }
    if 'question' in batch:
        payload = {
            "task": req['task'], 
            "req_id": req['req_id'],
            "x": encode_raw(batch['x'].numpy()),
            "question": encode_raw(batch['question']),
            "y": encode_raw(batch['y'])
        }
    #timeseries with mask and without
    else:
        payload = {
            "task": req['task'], 
            "req_id": req['req_id'],
            "x": encode_raw(batch['x'].numpy()),
            "mask": encode_raw(batch['mask'].numpy()) if len(batch)==3 else None,
            "y": encode_raw(batch['y'].numpy())
        }       
    print(f"[RuntimeExecutor] Executing {req['task']} on device {req['device']}")
    i, total_time, data = await send_request(req['req_id'], req['device'], payload)
    print(f"[RuntimeExecutor] Completed {req['task']} ({i}) in {total_time}")
    return {'req_id':i,"latency":total_time,'device_info':data}