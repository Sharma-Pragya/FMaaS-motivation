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
from torch.utils.data import DataLoader
from site_manager.config import *
from site_manager.storage import get_new_requests, wait_for_new_requests
from site_manager.grpc_client import EdgeRuntimeClient

DATASET_LOADERS = {}
DATA = {}


# based on type of tasks in requests in needs to initialize dataloaders
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
        # "etth1fore": DataLoader(ETTh1Dataset({"dataset_path": f"{d}/ETTh1"}, {"task_type": "forecasting"}, "test"), **inference_config),
        # "weatherfore": DataLoader(WeatherDataset({"dataset_path": f"{d}/Weather"}, {"task_type": "forecasting"}, "test"), **inference_config),
        # "trafficfore": DataLoader(TrafficDataset({"dataset_path": f"{d}/Traffic"}, {"task_type": "forecasting"}, "test"), **inference_config),
        # "eclfore": DataLoader(ECLDataset({"dataset_path": f"{d}/ElectricityLoad-data"}, {"task_type": "forecasting"}, "test"), **inference_config),
        # "exchangefore": DataLoader(ExchangeDataset({"dataset_path": f"{d}/Exchange"}, {"task_type": "forecasting"}, "test"), **inference_config),
        # "illness": DataLoader(IllnessDataset({"dataset_path": f"{d}/ILLNESS"}, {"task_type": "forecasting"}, "test", forecast_horizon=192), **inference_config),
        # 'vqa': DataLoader(VQADataset({"dataset_path": f"{d}/val2014"}, {"task_type": "forecasting"}, "test") ,  **inference_config)
    }
    global DATA
    DATA = {task: next(iter(loader)) for task, loader in DATASET_LOADERS.items()}
    print(f"[RuntimeExecutor] Initialized {len(DATASET_LOADERS)} dataloaders.")


def load_dataloader(task: str):
    if task not in DATASET_LOADERS:
        raise ValueError(f"Unknown task: {task}")
    return DATASET_LOADERS[task]


CLIENT_CACHE = {}


async def close_clients():
    """Close all cached gRPC clients and clear the cache.

    Must be called at the start of each asyncio.run() session because
    grpc clients are bound to the event loop that created them.
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
            client = EdgeRuntimeClient(url)
            await client.wait_ready()
            CLIENT_CACHE[url] = client
            print(f"Connected to gRPC server at {url}")
            return client
        except Exception as e:
            print(f"Error connecting to {url}: {e}")
            return None

    return CLIENT_CACHE[url]


async def send_request(req_id, device_url, inputs_dict, ouputs_dict):
    # 1. Get the raw gRPC client
    request_start_time = time.time()
    client = await get_client(device_url)
    if client is None:
        return

    # 2. Send Request
    infer_submit_time = time.time()
    try:
        response = await client.infer(
            {
                "req_id": req_id,
                "task": inputs_dict["task"],
                "x": inputs_dict["x"],
                "mask": inputs_dict.get("mask"),
                "question": inputs_dict.get("question"),
            }
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    # 3. Get Output
    result = response["output"]
    device_start_time = response["start_time_ns"] / 10**9
    device_end_time = response["end_time_ns"] / 10**9
    proc_time = response["proc_time_ns"] / 10**9
    swap_time = response["swap_time_ns"] / 10**9
    decoder_time = response["decoder_time_ns"] / 10**9
    response_recv_time = time.time()
    pred = result.item() if getattr(result, "size", 1) == 1 else result.flatten().tolist()
    true_val = ouputs_dict.get("y")
    true_val = true_val.item() if true_val.size == 1 else true_val.flatten().tolist()
    return (
        req_id,
        device_url,
        request_start_time,
        infer_submit_time,
        device_start_time,
        device_end_time,
        response_recv_time,
        response_recv_time - request_start_time,
        proc_time,
        swap_time,
        decoder_time,
        pred,
        true_val,
    )


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
    completed_results = []
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
                heapq.heappush(pending_queue, (req["req_time"], req_counter, req))
                req_counter += 1

            # Show timestamp range of newly added requests
            if new_reqs:
                min_time = min(r["req_time"] for r in new_reqs)
                max_time = max(r["req_time"] for r in new_reqs)
                print(f"[RuntimeExecutor] New requests span t={min_time:.1f}s to t={max_time:.1f}s")

            print(f"[RuntimeExecutor] Pending queue now has {len(pending_queue)} requests")
        elif poll_iteration % 100 == 0:
            # Log every 100 polls to confirm loop is running
            print(f"[RuntimeExecutor] Poll #{poll_iteration}: queue={len(pending_queue)}, processed={processed_count}")

        # Process requests whose scheduled time has arrived.
        # Re-check time each dispatch so requests keep their intended spacing
        # instead of being flushed in a 10-100ms burst.
        sent_this_iteration = 0
        while pending_queue:
            req_time, counter, req = pending_queue[0]  # Peek at earliest request
            target_time = start + req_time
            current_time = time.time()

            if current_time < target_time:
                if sent_this_iteration > 0:
                    next_wait = target_time - current_time
                    print(f"[RuntimeExecutor] Sent {sent_this_iteration} requests, next in {next_wait:.3f}s (at t={req_time:.3f}s)")
                break

            heapq.heappop(pending_queue)

            batch = DATA.get(req["task"])
            if batch is None:
                print(f"[RuntimeExecutor] WARNING: No dataloader for task '{req['task']}', skipping req {req['req_id']}")
                processed_count += 1
                sent_this_iteration += 1
                continue

            # Prepare inputs
            inputs = {}
            outputs = {}
            inputs["x"] = batch["x"].numpy().astype(np.float32)
            outputs["y"] = batch["y"].numpy().astype(np.float32)
            inputs["task"] = req["task"]

            if "mask" in batch:
                inputs["mask"] = batch["mask"].numpy().astype(np.float32)
            if "question" in batch:
                inputs["question"] = batch["question"]

            # Schedule the request
            task = asyncio.create_task(send_request(req["req_id"], req["device"], inputs, outputs))
            tasks.append(task)
            processed_count += 1
            sent_this_iteration += 1

        # Harvest completed tasks so this list does not grow for the whole run.
        if tasks:
            active_tasks = []
            for task in tasks:
                if task.done():
                    try:
                        completed_results.append(task.result())
                    except Exception as e:
                        print(f"[RuntimeExecutor] Task failed: {e}")
                        completed_results.append(None)
                else:
                    active_tasks.append(task)
            tasks = active_tasks

        # Check if we should exit
        idle_time = time.time() - last_request_received_time
        if processed_count > 0 and idle_time >= MAX_IDLE and not pending_queue:
            print(f"[RuntimeExecutor] No new requests for {idle_time:.1f}s and queue empty, completing...")
            break

        # Wait until either the next known request is due or a new request chunk
        # arrives that may introduce an earlier deadline.
        if pending_queue:
            next_req_time, _, _ = pending_queue[0]
            next_target = start + next_req_time
            wait_timeout = max(0.0, next_target - time.time())
        else:
            wait_timeout = 0.1

        if wait_timeout > 0:
            await asyncio.to_thread(wait_for_new_requests, wait_timeout)

    # Wait for all pending tasks to complete
    if tasks:
        print(f"[RuntimeExecutor] Waiting for {len(tasks)} pending requests to complete...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                print(f"[RuntimeExecutor] Task failed: {result}")
                completed_results.append(None)
            else:
                completed_results.append(result)

    return completed_results
