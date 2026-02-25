from fmtk.datasetloaders.etth1 import ETTh1Dataset
from fmtk.datasetloaders.weather import WeatherDataset
from fmtk.datasetloaders.exchange import ExchangeDataset
from fmtk.datasetloaders.ecg5000 import ECG5000Dataset
from fmtk.datasetloaders.uwavegesture import UWaveGestureLibraryALLDataset
from fmtk.datasetloaders.ppg import PPGDataset
from fmtk.datasetloaders.illness import IllnessDataset
from fmtk.datasetloaders.ecl import ECLDataset
from fmtk.datasetloaders.traffic import TrafficDataset
from fmtk.datasetloaders.vqa import VQADataset
import asyncio
import time
import numpy as np
from typing import List
from pytriton.client import ModelClient
from torch.utils.data import DataLoader
from urllib.parse import urlparse
from site_manager.config import *
from site_manager.storage import get_new_requests, is_task_deploying
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput
import tritonclient.grpc.aio as grpc_aio

DATASET_LOADERS = {}
DATA={}
#based on type of tasks in requests in needs to initialize dataloaders
def initialize_dataloaders():
    global DATASET_LOADERS
    inference_config = {"batch_size": DEFAULT_BATCH_SIZE, "shuffle": False}
    d = DATASET_DIR
    # create test dataloaders
    DATASET_LOADERS = {
        "ecgclass": DataLoader(ECG5000Dataset({"dataset_path": f"{d}/ECG5000"}, {"task_type": "classification"}, "test"), **inference_config),
        "heartrate": DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data"}, {"task_type": "regression","label":"hr"}, "test"), **inference_config),
        "diasbp": DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data"}, {"task_type": "regression","label":"diasbp"}, "test"), **inference_config),
        "sysbp": DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data"}, {"task_type": "regression","label":"sysbp"}, "test"), **inference_config),
        "gestureclass": DataLoader(UWaveGestureLibraryALLDataset({"dataset_path": f"{d}/UWaveGestureLibraryAll"}, {"task_type": "classification"}, "test"), **inference_config),
        # "ecl": DataLoader(ECLDataset({"dataset_path": f"{d}/ElectricityLoad-data"}, {"task_type": "forecasting"}, "test"), **inference_config),
        # "traffic": DataLoader(TrafficDataset({"dataset_path": f"{d}/Traffic"}, {"task_type": "forecasting"}, "test"), **inference_config),
        # "illness": DataLoader(IllnessDataset({"dataset_path": f"{d}/ILLNESS"}, {"task_type": "forecasting"}, "test", forecast_horizon=192), **inference_config),
        # "etth1": DataLoader(ETTh1Dataset({"dataset_path": f"{d}/ETTh1"}, {"task_type": "forecasting"}, "test"), **inference_config),
        # "weather": DataLoader(WeatherDataset({"dataset_path": f"{d}/Weather"}, {"task_type": "forecasting"}, "test"), **inference_config),
        # "rate": DataLoader(ExchangeDataset({"dataset_path": f"{d}/Exchange"}, {"task_type": "forecasting"}, "test"), **inference_config),
        # 'vqa': DataLoader(VQADataset({"dataset_path": f"{d}/val2014"}, {"task_type": "forecasting"}, "test") ,  **inference_config)
    }
    global DATA
    DATA = {task: next(iter(loader)) for task, loader in DATASET_LOADERS.items()}
    print(f"[RuntimeExecutor] Initialized {len(DATASET_LOADERS)} dataloaders.")

def load_dataloader(task: str):
    if task not in DATASET_LOADERS:
        raise ValueError(f'Unknown task: {task}')
    return DATASET_LOADERS[task]

CLIENT_CACHE = {}

async def close_clients():
    """Close all cached gRPC clients and clear the cache.
    
    Must be called at the start of each asyncio.run() session because
    grpc_aio clients are bound to the event loop that created them.
    When asyncio.run() closes that loop, the cached clients become stale.
    """
    n = len(CLIENT_CACHE)
    for url, client in list(CLIENT_CACHE.items()):
        try:
            await client.close()
        except Exception:
            pass
    CLIENT_CACHE.clear()
    if n > 0:
        print(f"[RuntimeExecutor] Closed and cleared {n} cached gRPC client(s).")


async def get_client(url: str):
    if url not in CLIENT_CACHE:
        try:
            client = grpc_aio.InferenceServerClient(url=url, verbose=False)

            if not await client.is_server_live():
                # Server not ready yet — don't cache, caller will retry next poll
                return None

            CLIENT_CACHE[url] = client
            print(f"Connected to gRPC server at {url}")
            return client

        except Exception as e:
            print(f"Error connecting to {url}: {e}")
            return None

    return CLIENT_CACHE[url]

async def send_request(req_id, device_url, inputs_dict, ouputs_dict):
    # 1. Get the raw gRPC client
    st = time.time()
    client = await get_client(device_url) 
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
    try:
        response = await client.infer(model_name="edge_infer", inputs=triton_inputs)
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    # 4. Get Output
    # Assuming your output is named "output"
    # You might need to check your model config for the exact output name
    result = response.as_numpy("output")
    device_start_time = response.as_numpy("start_time")/10**9
    device_end_time = response.as_numpy("end_time")/10**9
    proc_time = response.as_numpy("proc_time")/10**9
    swap_time = response.as_numpy("swap_time")/10**9
    decoder_time = response.as_numpy("decoder_time")/10**9
    et = time.time()
    return req_id, device_url, st, device_start_time.item(), device_end_time.item(), et - st, proc_time.item(), swap_time.item(), decoder_time.item(), result.item(), ouputs_dict.get('y').item()
    


async def handle_runtime_request_continuous():
    """Continuous inference mode that dynamically picks up new requests.

    This mode is used when tasks can be added at runtime. It continuously
    polls for new requests and processes them according to their timestamps.

    Uses a priority queue to ensure requests are sent in timestamp order,
    even when they arrive in different batches at runtime.
    """
    import heapq

    await close_clients()

    tasks: List[asyncio.Task] = []
    start = time.time()
    pending_queue = []  # Min-heap priority queue: [(req_time, counter, req), ...]
    processed_count = 0
    req_counter = 0  # Unique counter for tiebreaking in heap

    print(f"[RuntimeExecutor] Starting continuous inference mode (start_time={start:.2f})...")

    # Keep running until we detect no new requests for a while
    MAX_IDLE = 10  # seconds without new requests after last request scheduled
    last_request_received_time = start

    poll_iteration = 0
    while True:
        # Get any new requests that arrived via MQTT
        new_reqs = get_new_requests()
        poll_iteration += 1

        if new_reqs:
            last_request_received_time = time.time()
            print(f"[RuntimeExecutor] Received {len(new_reqs)} new request(s) at t={time.time()-start:.1f}s (poll #{poll_iteration})")

            # Add to priority queue (sorted by req_time, with counter as tiebreaker)
            for req in new_reqs:
                heapq.heappush(pending_queue, (req['req_time'], req_counter, req))
                req_counter += 1

            # Show timestamp range of newly added requests
            if new_reqs:
                min_time = min(r['req_time'] for r in new_reqs)
                max_time = max(r['req_time'] for r in new_reqs)
                print(f"[RuntimeExecutor] New requests span t={min_time:.1f}s to t={max_time:.1f}s")

            print(f"[RuntimeExecutor] Pending queue now has {len(pending_queue)} requests")
        elif poll_iteration % 100 == 0:
            # Log every 100 polls to confirm loop is running
            print(f"[RuntimeExecutor] Poll #{poll_iteration}: queue={len(pending_queue)}, processed={processed_count}")

        # Process requests whose scheduled time has arrived
        current_time = time.time()
        sent_this_iteration = 0
        deferred_this_iteration = 0
        while pending_queue:
            req_time, counter, req = pending_queue[0]  # Peek at earliest request
            target_time = start + req_time

            if current_time >= target_time:
                heapq.heappop(pending_queue)

                # Check if this (task, device) is still deploying — defer if so
                if is_task_deploying(req['task'], req['device']):
                    # Push back with a small delay so we retry soon
                    heapq.heappush(pending_queue, (req_time + 0.5, req_counter, req))
                    req_counter += 1
                    deferred_this_iteration += 1
                    continue

                batch = DATA.get(req['task'])

                # Prepare inputs
                inputs = {}
                outputs = {}
                inputs['x'] = batch['x'].numpy().astype(np.float32)
                outputs['y'] = batch['y'].numpy().astype(np.float32)
                inputs['task'] = np.array([[req['task'].encode('utf-8')]], dtype=object)

                if 'mask' in batch:
                    inputs['mask'] = batch['mask'].numpy().astype(np.float32)
                if 'question' in batch:
                    inputs['question'] = np.array([[batch['question'].encode('utf-8')]], dtype=object)

                # Schedule the request
                task = asyncio.create_task(send_request(req['req_id'], req['device'], inputs, outputs))
                tasks.append(task)
                processed_count += 1
                sent_this_iteration += 1
            else:
                # Next request is in the future, stop processing for now
                next_wait = target_time - current_time
                if sent_this_iteration > 0:
                    print(f"[RuntimeExecutor] Sent {sent_this_iteration} requests, next in {next_wait:.2f}s (at t={req_time:.1f}s)")
                break

        if deferred_this_iteration > 0:
            print(f"[RuntimeExecutor] Deferred {deferred_this_iteration} request(s) — deployment in progress")

        # Check if we should exit
        idle_time = time.time() - last_request_received_time
        if processed_count > 0 and idle_time >= MAX_IDLE and not pending_queue:
            print(f"[RuntimeExecutor] No new requests for {idle_time:.1f}s and queue empty, completing...")
            break

        # Sleep briefly before checking again
        # Use shorter sleep if we have pending requests with near-future timestamps
        # Keep sleep short (0.1s max) to frequently check for newly arrived requests
        if pending_queue:
            next_req_time, _, _ = pending_queue[0]  # Unpack with counter
            next_target = start + next_req_time
            time_until_next = next_target - time.time()
            sleep_time = min(0.1, max(0.01, time_until_next))  # Max 0.1s to check for new reqs
        else:
            sleep_time = 0.1  # Check for new requests every 100ms

        await asyncio.sleep(sleep_time)

    # Wait for all pending tasks to complete
    if tasks:
        print(f"[RuntimeExecutor] Waiting for {len(tasks)} pending requests to complete...")
        results = await asyncio.gather(*tasks)
        return results
    else:
        return []


# async def handle_runtime_request(reqs: dict, mode: str = 'trace'):
#     # Clear stale gRPC clients from previous asyncio.run() sessions.
#     # Each asyncio.run() creates a new event loop, so cached clients
#     # bound to the old (closed) loop would cause "Event loop is closed".
#     await close_clients()

#     if mode == 'trace':
#         # Existing logic for trace-based requests
#         tasks: List[asyncio.Task] = []
#         start = time.time()

#         for req in reqs:
#             # Simulate Arrival Time
#             wait_time = (start + req['req_time']) - time.time()
#             if wait_time > 0:
#                 await asyncio.sleep(wait_time)

#             batch = DATA.get(req['task'])

#             # --- Prepare Inputs for PyTriton ---
#             # PyTriton expects NumPy arrays.
#             inputs = {}
#             outputs = {}

#             # 1. X (Data)
#             inputs['x'] = batch['x'].numpy().astype(np.float32)

#             #2. Y) (Labels)
#             outputs['y'] = batch['y'].numpy().astype(np.float32)

#             # 2. Task (Bytes)
#             # Must be shape (Batch,) or (1,) depending on your batching strategy
#             # Even for a single item, it expects an array.
#             task_str = req['task']
#             inputs['task'] = np.array([[task_str.encode('utf-8')]], dtype=object)

#             # 3. Optional: Mask
#             if 'mask' in batch:
#                 mask_np = batch['mask'].numpy().astype(np.float32)
#                 # inputs['mask'] = np.expand_dims(mask_np, axis=-1)
#                 inputs['mask']=mask_np
                
#             # 4. Optional: Question
#             if 'question' in batch:
#                 q_data = batch['question']
#                 inputs['question'] = np.array([[q_data.encode('utf-8')]], dtype=object)

#             # Schedule the request
#             task = asyncio.create_task(send_request(req['req_id'], req['device'], inputs, outputs))
#             tasks.append(task)

#         results = await asyncio.gather(*tasks)
#         return results
    
#     elif mode == 'accuracy':
#         task = reqs[0]['task']
#         device = reqs[0]['device']
        
#         dataloader = load_dataloader(task)
        
#         results = []        
#         for batch_idx, batch in enumerate(dataloader):
#             # Prepare inputs (same as trace mode)
#             inputs = {}
#             outputs={}
#             inputs['x'] = batch['x'].numpy().astype(np.float32)
#             outputs['y'] = batch['y'].numpy().astype(np.float32)
#             inputs['task'] = np.array([[task.encode('utf-8')]], dtype=object)
#             if 'mask' in batch:
#                 mask_np = batch['mask'].numpy().astype(np.float32)
#                 # inputs['mask'] = np.expand_dims(mask_np, axis=-1)
#                 inputs['mask']=mask_np
#             if 'question' in batch:
#                 q_data = batch['question']
#                 inputs['question'] = np.array([[q_data.encode('utf-8')]], dtype=object)
            
#             result = await send_request(batch_idx, device, inputs, outputs)
#             results.append(result)
#         return results
            
#     else:
#         raise ValueError(f"Unknown mode: {mode}")
