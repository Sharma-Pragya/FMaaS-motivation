from fmtk.datasets.etth1 import ETTh1Dataset
from fmtk.datasets.weather import WeatherDataset
from fmtk.datasets.exchange import ExchangeDataset
from fmtk.datasets.ecg5000 import ECG5000Dataset
from fmtk.datasets.uwavegesture import UWaveGestureLibraryALLDataset
from fmtk.datasets.ppg import PPGDataset
from fmtk.datasets.illness import IllnessDataset
from fmtk.datasets.ecl import ECLDataset
from fmtk.datasets.traffic import TrafficDataset
from fmtk.datasets.vqa import VQADataset
import asyncio
import time
import numpy as np
from typing import List
from pytriton.client import ModelClient
from torch.utils.data import DataLoader
from urllib.parse import urlparse
from config import *
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput

DATASET_LOADERS = {}
dataloaders={}
#based on type of tasks in requests in needs to initialize dataloaders
def initialize_dataloaders():
    global DATASET_LOADERS
    inference_config = {"batch_size": DEFAULT_BATCH_SIZE, "shuffle": False}
    d = DATASET_DIR
    # create test dataloaders
    DATASET_LOADERS = {
        "ecgclass": DataLoader(ECG5000Dataset({"dataset_path": f"{d}/ECG5000"}, {"task_type": "classification"}, "test"), **inference_config),
        # "heartrate": DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data"}, {"task_type": "regression","label":"hr"}, "test"), **inference_config),
        # "diasbp": DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data"}, {"task_type": "regression","label":"diasbp"}, "test"), **inference_config),
        # "sysbp": DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data"}, {"task_type": "regression","label":"sysbp"}, "test"), **inference_config),
        "gestureclass": DataLoader(UWaveGestureLibraryALLDataset({"dataset_path": f"{d}/UWaveGestureLibraryAll"}, {"task_type": "classification"}, "test"), **inference_config),
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

CLIENT_CACHE = {}
def get_client(url: str):
    if url not in CLIENT_CACHE:
        try:
            client = grpcclient.InferenceServerClient(url=url, verbose=False)
            
            if not client.is_server_live():
                print(f"Warning: Server at {url} is not live yet.")
                return None
                
            CLIENT_CACHE[url] = client
            print(f"Connected to gRPC server at {url}")
            return client # Make sure to return the new client here
            
        except Exception as e:
            print(f"Error connecting to {url}: {e}")
            return None
            
    return CLIENT_CACHE[url]

async def send_request(req_id, device_url, inputs_dict, ouputs_dict):
    # 1. Get the raw gRPC client
    st = time.time()
    client = get_client(device_url) 
    if client is None:
        return

    # 2. Prepare Inputs List
    triton_inputs = []
    
    for name, data in inputs_dict.items():
        # Determine format based on data type
        # Strings/Bytes need "BYTES", Floats need "FP32"
        if data.dtype.kind in {'U', 'S', 'O'}: # String/Object/Bytes
            dtype = "BYTES"
        else:
            dtype = "FP32" # Or "INT32", etc. based on your data

        # Create the InferInput object
        # shape is required
        inp = InferInput(name, data.shape, dtype)
        inp.set_data_from_numpy(data)
        triton_inputs.append(inp)

    # 3. Send Request (Thread-safe!)
    # logic: model_name is the one you defined in bind()
    try:
        response = client.infer(model_name="edge_infer", inputs=triton_inputs)
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    # 4. Get Output
    # Assuming your output is named "output"
    # You might need to check your model config for the exact output name
    result = response.as_numpy("output")
    proc_time = response.as_numpy("proc_time")
    swap_time = response.as_numpy("swap_time")
    et = time.time()
    return req_id, device_url, et - st, proc_time.item(), swap_time.item(), result.item(), ouputs_dict.get('y').item()
    


async def handle_runtime_request(reqs: dict, mode: str = 'trace'):
    if mode == 'trace':
        # Existing logic for trace-based requests
        tasks: List[asyncio.Task] = []
        start = time.time()
        
        for req in reqs:
            # Simulate Arrival Time
            wait_time = (start + req['req_time']) - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            dataloader = load_dataloader(req['task'])
            batch = next(iter(dataloader))

            # --- Prepare Inputs for PyTriton ---
            # PyTriton expects NumPy arrays.
            inputs = {}
            outputs = {}

            # 1. X (Data)
            inputs['x'] = batch['x'].numpy().astype(np.float32)

            #2. Y) (Labels)
            outputs['y'] = batch['y'].numpy().astype(np.float32)

            # 2. Task (Bytes)
            # Must be shape (Batch,) or (1,) depending on your batching strategy
            # Even for a single item, it expects an array.
            task_str = req['task']
            inputs['task'] = np.array([[task_str.encode('utf-8')]], dtype=object)

            # 3. Optional: Mask
            if 'mask' in batch:
                mask_np = batch['mask'].numpy().astype(np.float32)
                inputs['mask'] = np.expand_dims(mask_np, axis=-1)

            # 4. Optional: Question
            if 'question' in batch:
                q_data = batch['question']
                inputs['question'] = np.array([[q_data.encode('utf-8')]], dtype=object)

            # Schedule the request
            task = asyncio.create_task(send_request(req['req_id'], req['device'], inputs, outputs))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results
    
    elif mode == 'accuracy':
        task = reqs[0]['task']
        device = reqs[0]['device']
        
        dataloader = load_dataloader(task)
        
        results = []        
        for batch_idx, batch in enumerate(dataloader):
            # Prepare inputs (same as trace mode)
            inputs = {}
            outputs={}
            inputs['x'] = batch['x'].numpy().astype(np.float32)
            outputs['y'] = batch['y'].numpy().astype(np.float32)
            inputs['task'] = np.array([[task.encode('utf-8')]], dtype=object)
            if 'mask' in batch:
                mask_np = batch['mask'].numpy().astype(np.float32)
                inputs['mask'] = np.expand_dims(mask_np, axis=-1)
            if 'question' in batch:
                q_data = batch['question']
                inputs['question'] = np.array([[q_data.encode('utf-8')]], dtype=object)
            
            result = await send_request(batch_idx, device, inputs, outputs)
            results.append(result)
        return results
            
    else:
        raise ValueError(f"Unknown mode: {mode}")
