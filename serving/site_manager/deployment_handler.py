# site_manager/deployment_handler.py
from typing import List
from pydantic import BaseModel
import aiohttp, asyncio

class DecoderSpec(BaseModel):
    task: str
    type: str
    path: str

class DeploySpec(BaseModel):
    device: str               # device endpoint (e.g. http://10.100.115.7:8000)
    backbone: str             # e.g. "moment_large"
    decoders: List[DecoderSpec]

async def deploy_models(specs: List[DeploySpec]):
    """Forward model deployment specs to each device"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for s in specs:
            payload = {
                "backbone": s.backbone,
                "decoders": [d.model_dump() for d in s.decoders],
            }
            print(f"[SiteManager] Deploying to device {s.device} ...")
            t = session.post(f"{s.device}/load_model", json=payload)
            tasks.append(t)
        responses = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"[SiteManager] Deployment complete for {len(specs)} devices.")
    return {"status": "ok"}
