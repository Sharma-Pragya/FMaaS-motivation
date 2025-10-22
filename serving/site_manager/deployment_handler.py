# site_manager/deployment_handler.py
from typing import List
from pydantic import BaseModel,Field
import aiohttp, asyncio

class DecoderSpec(BaseModel):
    task: str
    type: str
    path: str

class DeploySpec(BaseModel):
    device: str               # device endpoint (e.g. http://10.100.115.7:8000)
    backbone: str             # e.g. "moment_large"
    decoders: List[DecoderSpec] = Field(default_factory=list) 

async def deploy_models(specs: List[DeploySpec]):
    """Forward model deployment specs to each device"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for s in specs:
            print(s)
            payload = {
                "backbone": s['backbone'],
                "decoders": s['decoders'],
            }
            print(f"[SiteManager] Deploying to device {s['device']} ...")
            t = session.post(f"{s['device']}/load_model", json=payload)
            tasks.append(t)
        responses = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"[SiteManager] Deployment complete for {len(specs)} devices.")
    return {"status": "ok"}



# # site_manager/device_manager.py
# import subprocess, os

# DEVICE_PROCESSES = {}

# def launch_model(device: str, model_path: str, port: int):
#     """Launch a model server process on a device"""
#     #ssh into unity
#     #sattach into that device
#     cmd = [
#         "ssh", device,
#         f"nohup python3 -m device.inference_server --model {model_path} --port {port} > /tmp/device_{device}.log 2>&1 &"
#     ]
#     print(f"[DeviceManager] Launching model on {device}: {model_path}")
#     subprocess.Popen(" ".join(cmd), shell=True)
#     DEVICE_PROCESSES[device] = {"model": model_path, "port": port}

# def stop_model(device: str):
#     """Stop model process on a device"""
#     print(f"[DeviceManager] Stopping model on {device}")
#     subprocess.Popen(f"ssh {device} pkill -f inference_server", shell=True)
#     DEVICE_PROCESSES.pop(device, None)
