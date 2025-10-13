import asyncio
import time
import threading
from typing import List, Optional, Tuple, Literal
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from torch.utils.data import DataLoader, TensorDataset
from timeseries.pipeline import Pipeline
from timeseries.components.backbones.moment import MomentModel
import torch.nn as nn
from tqdm import tqdm
import uvicorn 
import numpy as np
import io
import base64
from fastapi import Request

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI()
request_queue: asyncio.Queue = asyncio.Queue()  # async request queue

class EncodedArray(BaseModel):
    shape: List[int]
    dtype: str
    data: str   # base64 string

class PredictRequest(BaseModel):
    task: Literal["etth1", "weather", "rate", "hr","ecg_class","gesture_class","diasbp","sysbp","ecl","traffic","illness"]
    req_id: int
    x: EncodedArray
    mask: Optional[EncodedArray] = None
    y: EncodedArray
    return_pred: bool = False

class PredictResponse(BaseModel):
    req_id: int
    y_pred: Optional[List] = None
    device_wait_time: float
    device_infer_time: float


def _build_pipeline_and_decoders(device: torch.device):
    P = Pipeline(MomentModel(device, "large"))
    
    from timeseries.components.decoders.regression.mlp import MLPDecoder
    print(torch.cuda.max_memory_allocated(device))
    d5 = P.add_decoder(
        MLPDecoder(device=device, cfg={'input_dim':1024,'output_dim':1,'hidden_dim':128}),
        load=True, trained=True, path="heartrate_momentlarge_mlp"
    )
    print(torch.cuda.max_memory_allocated(device))
    d3 = P.add_decoder(
        MLPDecoder(device=device, cfg={'input_dim':1024,'output_dim':1,'hidden_dim':128}),
        load=True, trained=True, path="sysbp_momentlarge_mlp"
    )
    print(torch.cuda.max_memory_allocated(device))
    d4 = P.add_decoder(
        MLPDecoder(device=device, cfg={'input_dim':1024,'output_dim':1,'hidden_dim':128}),
        load=True, trained=True, path="diasbp_momentlarge_mlp"
    )

    print(torch.cuda.max_memory_allocated(device))
    from timeseries.components.decoders.classification.mlp import MLPDecoder
    d1 = P.add_decoder(
        MLPDecoder(device=device, cfg={'input_dim':1024,'output_dim':5,'hidden_dim':128}),
        load=True, trained=True, path="ecgclass_momentlarge_mlp"
    )
    print(torch.cuda.max_memory_allocated(device)) 
    d2= P.add_decoder(
        MLPDecoder(device=device, cfg={'input_dim':1024,'output_dim':10,'hidden_dim':128}),
        load=True, trained=True, path="gestureclass_momentlarge_mlp"
    )
    
    from timeseries.components.decoders.forecasting.mlp import MLPDecoder
    print(torch.cuda.max_memory_allocated(device))
    d9 = P.add_decoder(
        MLPDecoder(device=device, cfg={'input_dim':64*1024,'output_dim':192,'dropout':0.1}),
        load=True, trained=True, path="ecl_fore_momentlarge_mlp"
    )
    print(torch.cuda.max_memory_allocated(device))
    d10= P.add_decoder(
        MLPDecoder(device=device, cfg={'input_dim':64*1024,'output_dim':192,'dropout':0.1}),
        load=True, trained=True, path="traffic_fore_momentlarge_mlp"
    )
    print(torch.cuda.max_memory_allocated(device))
    d6= P.add_decoder(
        MLPDecoder(device=device, cfg={'input_dim':64*1024,'output_dim':192,'dropout':0.1}),
        load=True, trained=True, path="etth1_fore_momentlarge_mlp"
    )
    print(torch.cuda.max_memory_allocated(device))
    d7 = P.add_decoder(
        MLPDecoder(device=device, cfg={'input_dim':64*1024,'output_dim':192,'dropout':0.1}),
        load=True, trained=True, path="weather_fore_momentlarge_mlp"
    )
    print(torch.cuda.max_memory_allocated(device))
    d8 = P.add_decoder(
        MLPDecoder(device=device, cfg={'input_dim':64*1024,'output_dim':192,'dropout':0.1}),
        load=True, trained=True, path="exchange_fore_momentlarge_mlp"
    )
    print(torch.cuda.max_memory_allocated(device))
 

    return P, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10

def _predict_one_batch(pipeline, tasks: list ,req_ids: list ,bx: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    feats=pipeline.model_instance.forward(bx, mask)
    results={}
    for i,task in enumerate(tasks):
        if task == "ecg_class":
            pipeline.load_decoder(_ecg_class_decoder,swap=False)
        elif task == "gesture_class":
            pipeline.load_decoder(_gesture_class_decoder,swap=False)
        elif task == "hr":
            pipeline.load_decoder(_hr_decoder,swap=False)
        elif task == "diasbp":
            pipeline.load_decoder(_diasbp_decoder,swap=False)
        elif task == "sysbp":
            pipeline.load_decoder(_sysbp_decoder,swap=False)
        elif task == "etth1":
            pipeline.load_decoder(_etth1_decoder,swap=False)
        elif task == "weather":
            pipeline.load_decoder(_weather_decoder,swap=False)
        elif task == "rate":
            pipeline.load_decoder(_rate_decoder,swap=False)
        elif task == "ecl":
            pipeline.load_decoder(_ecl_class_decoder,swap=False)
        elif task == "traffic":
            pipeline.load_decoder(_traffic_decoder,swap=False)
        logits = pipeline.active_decoder.forward((feats[i].unsqueeze(0)))
        if isinstance(pipeline.active_decoder.criterion, (nn.CrossEntropyLoss)):
            logits = torch.argmax(logits, dim=1)
        # Problem RevIN normalizer computed in the model instance forward is batch size N but then we are
        # denormalizing one at a time here which causes a size mismatch 

        # if (hasattr(pipeline.active_decoder, "requires_model") and pipeline.active_decoder.requires_model and hasattr(pipeline.model_instance.model, "normalizer")):
        #     print(logits.shape, y_out.shape)
        #     logits = pipeline.model_instance.model.normalizer(x=logits, mode="denorm")
        #     print("yooooo")
        #     print(logits.shape, y_out.shape)
        
        results[req_ids[i]] = {
            "pred": logits,
        }
    return results

@app.on_event("startup")
def _startup():
    global _pipeline, _etth1_decoder, _weather_decoder, _rate_decoder,_ecg_class_decoder,_gesture_class_decoder, _hr_decoder, _diasbp_decoder, _sysbp_decoder, _ecl_class_decoder,_traffic_decoder
    _pipeline, _ecg_class_decoder,_gesture_class_decoder, _hr_decoder, _diasbp_decoder, _sysbp_decoder, _etth1_decoder, _weather_decoder, _rate_decoder, _ecl_class_decoder, _traffic_decoder  = _build_pipeline_and_decoders(_device)
    asyncio.create_task(_gpu_worker())

def decode_raw(obj: dict) -> torch.Tensor:
    raw = base64.b64decode(obj["data"])
    arr = np.frombuffer(raw, dtype=np.dtype(obj["dtype"]))
    arr = arr.reshape(obj["shape"])
    return arr


async def _gpu_worker():
    while True:
        fut, req, arrival_time, server_start = await request_queue.get()
        batch_reqs=[(fut, req, arrival_time, server_start)]
        batch_start=time.time()
        try:
            # while len(batch_reqs)<1 and (time.time()-batch_start)<0.01:
            while len(batch_reqs)<1:
                fut, req, arrival_time, server_start = await asyncio.wait_for(request_queue.get(), timeout=0.01)
                batch_reqs.append((fut, req, arrival_time, server_start))
        except asyncio.TimeoutError:
            pass
        start_infer = time.time()
        bxs, bys, masks, tasks, req_ids = [], [], [], [], []
        for _, req, _, _ in batch_reqs:
            bxs.append(torch.from_numpy(decode_raw(req.x.model_dump())))
            bys.append(torch.from_numpy(decode_raw(req.y.model_dump())))
            if req.mask is not None:
                # masks.append(torch.from_numpy(decode_raw(req.mask.model_dump())))
                masks.append(None)
            else:
                masks.append(None)
            tasks.append(req.task)
            req_ids.append(req.req_id)
        bx = torch.cat(bxs, dim=0)
        mask = torch.cat([m for m in masks if m is not None], dim=0) if any(masks) else None
        decode_finish = time.time()
        results = _predict_one_batch(_pipeline, tasks,req_ids, bx, mask if mask is not None else None)
        end_infer = time.time()

        for batch_req in batch_reqs:
            fut, req, arrival_time, server_start = batch_req
            task_results = results[req.req_id]

            resp = {
                "req_id": req.req_id,
                "device_wait_time": start_infer - arrival_time,
                "device_infer_time": end_infer - decode_finish,
            }
            fut.set_result(resp)
            request_queue.task_done()

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    server_start = time.time()
    response = await call_next(request)
    server_end = time.time()
    # send server total time back as a header
    response.headers["X-Server-Total-Time"] = str((server_end - server_start) * 1000)
    return response

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    server_start = time.time()   # when FastAPI has parsed JSON into Pydantic

    fut = asyncio.get_event_loop().create_future()
    arrival = time.time()
    await request_queue.put((fut, req, arrival, server_start))
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

    print(f"Starting server on port {port} with {workers} workers")
    uvicorn.run("server:app", host="0.0.0.0", port=port, workers=1)

