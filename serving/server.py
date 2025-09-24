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
    task: Literal["etth1", "weather", "rate", "hr","ecg_class","diasbp","sysbp"]
    x: EncodedArray
    mask: Optional[EncodedArray] = None
    y: EncodedArray
    return_pred: bool = False

class PredictResponse(BaseModel):
    task: str
    y_pred: Optional[List] = None
    wait_time: float
    decode_time: float
    arrival_time: float
    infer_time: float
    server_compute_time: float
    server_total_time: float

def _build_pipeline_and_decoders(device: torch.device):
    P = Pipeline(MomentModel(device, "large"))
    from timeseries.components.decoders.classification.mlp import MLPDecoder
    print(torch.cuda.max_memory_allocated(device))
    d1 = P.add_decoder(
        MLPDecoder(device=device, cfg={'input_dim':1024,'output_dim':5,'hidden_dim':128}),
        load=True, trained=True, path="ecgclass_momentlarge_mlp"
    )
    print(torch.cuda.max_memory_allocated(device))
    from timeseries.components.decoders.regression.mlp import MLPDecoder
    d2 = P.add_decoder(
        MLPDecoder(device=device, cfg={'input_dim':1024,'output_dim':1,'hidden_dim':128}),
        load=True, trained=True, path="sysbp_momentlarge_mlp"
    )
    print(torch.cuda.max_memory_allocated(device))
    d3 = P.add_decoder(
        MLPDecoder(device=device, cfg={'input_dim':1024,'output_dim':1,'hidden_dim':128}),
        load=True, trained=True, path="diasbp_momentlarge_mlp"
    )
    print(torch.cuda.max_memory_allocated(device))
    d4 = P.add_decoder(
        MLPDecoder(device=device, cfg={'input_dim':1024,'output_dim':1,'hidden_dim':128}),
        load=True, trained=True, path="heartrate_momentlarge_mlp"
    )
    print(torch.cuda.max_memory_allocated(device))
    return P, d1, d2, d3, d4

def _predict_one_batch(pipeline, decoder_name: str, bx: torch.Tensor, by: torch.Tensor,mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
    if mask!=None:
        batch=(bx,mask,by)
    else:
        batch=(bx,by)
    pipeline.load_decoder(decoder_name)
    y_test, y_pred = pipeline.predict_one_batch(batch)
    return y_test, y_pred

@app.on_event("startup")
def _startup():
    global _pipeline, _etth1_decoder, _weather_decoder, _rate_decoder,_ecg_class_decoder, _hr_decoder, _diasbp_decoder, _sysbp_decoder
    _pipeline, _ecg_class_decoder, _hr_decoder, _diasbp_decoder, _sysbp_decoder = _build_pipeline_and_decoders(_device)
    asyncio.create_task(_gpu_worker())

def decode_raw(obj: dict) -> torch.Tensor:
    raw = base64.b64decode(obj["data"])
    arr = np.frombuffer(raw, dtype=np.dtype(obj["dtype"]))
    arr = arr.reshape(obj["shape"])
    return arr


async def _gpu_worker():
    while True:
        fut, req, arrival_time, server_start = await request_queue.get()
        start_infer = time.time()

        bx = torch.from_numpy(decode_raw(req.x.model_dump()))
        if req.mask is not None:
            mask = torch.from_numpy(decode_raw(req.mask.model_dump()))
        else:
            mask = None
        by = torch.from_numpy(decode_raw(req.y.model_dump()))

        decode_finish = time.time()
        if req.task == "ecg_class":
            y_test, y_pred = _predict_one_batch(_pipeline, _ecg_class_decoder, bx, by, mask)
        elif req.task == "hr":
            y_test, y_pred = _predict_one_batch(_pipeline, _hr_decoder, bx, by)
        elif req.task == "diasbp":
            y_test, y_pred = _predict_one_batch(_pipeline, _diasbp_decoder, bx, by)
        elif req.task == "sysbp":
            y_test, y_pred = _predict_one_batch(_pipeline, _sysbp_decoder, bx, by)
        end_infer = time.time()

        resp = {
            "task": req.task,
            "y_pred": y_pred.detach().cpu().tolist() if req.return_pred else None,
            "arrival_time": arrival_time,
            "wait_time": start_infer - arrival_time,
            "decode_time": decode_finish - start_infer,
            "infer_time": end_infer - decode_finish,
            "server_compute_time": end_infer - start_infer,
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
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1)
