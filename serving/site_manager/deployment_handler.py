# # site_manager/deployment_handler.py
# from typing import List
# from pydantic import BaseModel,Field
# import aiohttp, asyncio

# class DecoderSpec(BaseModel):
#     task: str
#     type: str
#     path: str

# class DeploySpec(BaseModel):
#     device: str               # device endpoint (e.g. http://10.100.115.7:8000)
#     backbone: str             # e.g. "moment_large"
#     decoders: List[DecoderSpec] = Field(default_factory=list) 

# async def deploy_models(specs: List[DeploySpec]):
#     """Forward model deployment specs to each device"""
#     async with aiohttp.ClientSession() as session:
#         tasks = []
#         for s in specs:
#             if s['backbone']=='llava':
#                 # ssh into that s['device']
#                 # conda activate benchmark-foundation-vqa
#                 # run main.py --vision-language port 
#                 # hearbeat 
#             elif s['backbone']=='moment_large': 
#                 # ssh into that s['device']
#                 # conda activate fmtk
#                 # run main.py --timeseries port 
#                 # hearbeat 
#             payload = {
#                 "backbone": s['backbone'],
#                 "decoders": s['decoders'],
#             }
#             print(f"[SiteManager] Deploying to device {s['device']} ...")
#             t = session.post(f"{s['device']}/load_model", json=payload)
#             tasks.append(t)
#         responses = await asyncio.gather(*tasks, return_exceptions=True)
#     print(f"[SiteManager] Deployment complete for {len(specs)} devices.")
#     return {"status": "ok"}

# site_manager/deployment_handler.py
from typing import List
from pydantic import BaseModel, Field
import asyncio, asyncssh, aiohttp
from urllib.parse import urlparse
from config import custom_pythonpath

class DecoderSpec(BaseModel):
    task: str
    type: str
    path: str

class DeploySpec(BaseModel):
    device: str               # HTTP endpoint, e.g. "http://gpu048:8000" or "http://10.100.20.48:8000"
    backbone: str             # "moment_large" or "llava"
    decoders: List[DecoderSpec] = Field(default_factory=list)

def _parse_hosts(device_url: str) -> tuple[str, str]:
    """
    Returns (ssh_host, api_base).
    device_url may be "http://gpu048:8000" or "http://10.100.20.48:8000".
    """
    p = urlparse(device_url)
    if p.scheme and p.hostname:
        ssh_host = p.hostname                # e.g. "gpu048" or "10.100.20.48"
        api_base = f"{p.scheme}://{p.netloc}"  # e.g. "http://gpu048:8000"
    else:
        ssh_host = device_url
        api_base = f"http://{device_url}:8000"
    return ssh_host, api_base

async def _ssh_start_server(ssh_host: str, conda_env: str, cmd: str,log_path:str):
    """Run remote command on gpu node via SSH (agent forwarding must be enabled)."""
    try:
        async with asyncssh.connect(
            ssh_host,
            username="hshastri_umass_edu",
            agent_forwarding=True,
        ) as conn:
            remote_cmd = (
                f"bash -lc 'cd /project/pi_shenoy_umass_edu/hshastri/FMaaS-motivation/serving && "
                f"module load conda/latest && "
                f"conda activate {conda_env} && "
                f"export PYTHONPATH={custom_pythonpath} && "
                f"nohup {cmd}> {log_path} 2>&1 &'"
            )

            # print(f"[SSH] {ssh_host}: {remote_cmd}")
            # await conn.run(remote_cmd, check=False)
            print(f"[SSH] Launching on {ssh_host}: {remote_cmd}")
            proc = await conn.create_process(remote_cmd)
            await asyncio.sleep(1)   # small delay to ensure command sent
            proc.exit_status  # don’t wait; just trigger cleanup
            print(f"[SSH] {ssh_host}: detached.")

    except Exception as e:
        print(f"[SSH] Error on {ssh_host}: {e}")
        raise

async def _deploy_one(s: DeploySpec, session: aiohttp.ClientSession):
    ssh_host, api_base = _parse_hosts(s['device'])

    # choose env + server command
    if s['backbone'] == "llava":
        conda_env = "benchmark-foundation-vqa"
        server_cmd = f"python device/main.py"
    elif s['backbone'] == "moment_large":
        conda_env = "fmtk"
        server_cmd = f"python device/main.py"
    else:
        print(f"[WARN] Unknown backbone {s['backbone']}; skipping {s['device']}")
        return

    log_path = f"./device/{ssh_host}_{s['backbone']}.log"

    # 1) SSH: start/ensure the per-node server is running
    await _ssh_start_server(ssh_host, conda_env, server_cmd, log_path)
    
    # (tiny grace period for bind)
    await asyncio.sleep(60)
    print('Server Started on device')

    # 2) HTTP: send /load_model payload to that server
    payload = {
        "backbone": s['backbone'],
        "decoders": s['decoders'],
    }
    print(f"[SiteManager] Deploying to device {api_base}")
    try:
        async with session.post(f"{api_base}/load_model", json=payload, total=600) as resp:
            txt = await resp.text()
            print(f"[HTTP] {api_base}/load_model -> {resp.status} {txt[:200]}")
    except asyncio.TimeoutError:
        print(f"[HTTP] Timeout while waiting for {api_base}/load_model. Probably still loading.")
    except Exception as e:
        print(f"[HTTP] Error posting to {api_base}/load_model: {e}")

async def deploy_models(specs: List[DeploySpec]):
    """SSH into each node to start its server, then POST /load_model to that node."""
    async with aiohttp.ClientSession() as session:
        tasks = [_deploy_one(s, session) for s in specs]
        await asyncio.gather(*tasks, return_exceptions=False)
    print(f"[SiteManager] Deployment complete for {len(specs)} devices.")
    return {"status": "ok"}
