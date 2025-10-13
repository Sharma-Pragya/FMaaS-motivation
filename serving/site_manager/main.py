# site_manager/main.py
import asyncio, time, uvicorn
from fastapi import FastAPI
from site_manager.runtime_executor import (
    PredictRequest, PredictResponse,
    request_queue, _gpu_worker,
    initialize_dataloaders, load_dataloader
)
from site_manager.deployment_handler import deploy_models, DeploySpec
from site_manager.health_monitor import heartbeat
from site_manager.config import DEFAULT_PORT

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    initialize_dataloaders()
    asyncio.create_task(_gpu_worker())
    # asyncio.create_task(heartbeat("http://orchestrator:8000", "site_1"))
    print("[SiteManager] Startup complete.")

@app.post("/deploy")
async def deploy(specs: list[DeploySpec]):
    return await deploy_models(specs)

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    fut = asyncio.get_event_loop().create_future()
    arrival = time.time()
    dataloader = load_dataloader(req.task)
    await request_queue.put((fut, req, dataloader, arrival, time.time()))
    return await fut

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    uvicorn.run("site_manager.main:app", host="0.0.0.0", port=args.port, workers=1)
