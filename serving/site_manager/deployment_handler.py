from typing import List
from pydantic import BaseModel, Field
import asyncio, asyncssh, aiohttp
from urllib.parse import urlparse
from config import *

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
    return ssh_host, api_base,p.port

async def _ssh_start_server(ssh_host: str, conda_env: str, cmd: str,log_path:str):
    """Run remote command on gpu node via SSH (agent forwarding must be enabled)."""
    try:
        async with asyncssh.connect(
            ssh_host,
            username="hshastri_umass_edu",
            agent_forwarding=True,
        ) as conn:
            remote_cmd=(f"bash -lc '{cmds} && {activate_env} {conda_env} && nohup {cmd}> {log_path} 2>&1 &'")
       
            # print(f"[SSH] {ssh_host}: {remote_cmd}")
            # await conn.run(remote_cmd, check=False)
            print(f"[SSH] Launching on {ssh_host}: {remote_cmd}")
            proc = await conn.create_process(remote_cmd)
            await asyncio.sleep(10)   # small delay to ensure command sent
            proc.exit_status  # don’t wait; just trigger cleanup
            print(f"[SSH] {ssh_host}: detached.")

    except Exception as e:
        print(f"[SSH] Error on {ssh_host}: {e}")
        raise

async def _deploy_one(s: DeploySpec, session: aiohttp.ClientSession):
    ssh_host, api_base, port = _parse_hosts(s['device'])
    # choose env + server command
    if s['backbone'] == "llava":
        conda_env = vlm_env
        server_cmd = f"python device/main.py --port {port} "
    elif s['backbone'] in ["momentlarge","momentbase",'momentsmall',"chronostiny"] :
        conda_env = timeseries_env
        server_cmd = f"python device/main.py --port {port} "
    else:
        print(f"[WARN] Unknown backbone {s['backbone']}; skipping {s['device']}")
        return

    log_path = f"./device/logs/{ssh_host}_{s['backbone']}.log"

    await _ssh_start_server(ssh_host, conda_env, server_cmd, log_path)
    
    await asyncio.sleep(60)
    print('Server Started on device')

    payload = {
        "backbone": s['backbone'],
        "decoders": s['decoders'],
    }
    print(f"[SiteManager] Deploying to device {api_base}")
    try:
        async with session.post(f"{api_base}/load_model", json=payload) as resp:
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
