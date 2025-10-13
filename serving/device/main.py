# device/main.py
import asyncio, time, uvicorn
from fastapi import FastAPI, Request
from typing import List, Optional, Literal
from pydantic import BaseModel
from device.model_loader import load_models, unload_models
from device.inference_engine import request_queue, _gpu_worker
from device.config import DEVICE

app = FastAPI()

# ---------- Schemas ----------
class DecoderSpec(BaseModel):
    task: str
    type: Literal["regression", "classification", "forecasting"]
    path: str

class DeployRequest(BaseModel):
    backbone: str
    decoders: List[DecoderSpec]

class EncodedArray(BaseModel):
    shape: List[int]
    dtype: str
    data: str

class PredictRequest(BaseModel):
    req_id: int
    task: str
    x: EncodedArray
    mask: Optional[EncodedArray] = None
    y: Optional[EncodedArray] = None

class PredictResponse(BaseModel):
    req_id: int
    device_wait_time: float
    device_infer_time: float

# ---------- FastAPI Events ----------
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_gpu_worker())
    print("[Device] Ready for deployment and runtime inference.")

# ---------- Deployment Endpoints ----------
@app.post("/load_model")
async def load_model(req: DeployRequest):
    load_models(req.backbone, [d.model_dump() for d in req.decoders])
    return {"status": "loaded", "decoders": len(req.decoders)}

@app.post("/unload_model")
async def unload_model_endpoint():
    unload_models()
    return {"status": "unloaded"}

# ---------- Runtime Endpoints ----------
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    fut = asyncio.get_event_loop().create_future()
    arrival = time.time()
    await request_queue.put((fut, req, arrival, time.time()))
    return await fut

@app.get("/health")
def health():
    return {"ok": True, "device": str(DEVICE)}

# @app.middleware("http")
# async def timing_middleware(request: Request, call_next):
#     server_start = time.time()
#     response = await call_next(request)
#     server_end = time.time()
#     response.headers["X-Server-Total-Time"] = str((server_end - server_start) * 1000)
#     return response

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("device.main:app", host="0.0.0.0", port=args.port, workers=1)
