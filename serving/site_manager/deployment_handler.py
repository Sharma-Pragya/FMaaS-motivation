import json
import asyncio
import asyncssh
import numpy as np
from typing import List
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from pytriton.client import ModelClient
from config import * 

class DecoderSpec(BaseModel):
    task: str
    type: str
    path: str

class DeploySpec(BaseModel):
    device: str
    backbone: str
    decoders: List[DecoderSpec] = Field(default_factory=list)

def _parse_url(device_url: str) -> tuple[str, str]:
    """
    Returns (ssh_host, triton_url)
    triton_url is formatted as 'host:port' for ModelClient.
    """
    print(device_url)
    p = urlparse(device_url)
    print(p)
    if p.scheme and p.path:
        ssh_host = p.scheme
        #http port for control so +1
        port = int(p.path) if p.path else 8000
        triton_url = f"{p.scheme}:{port+1}"
    else:
        ssh_host = device_url
        port=8000
        triton_url = f"{device_url}:{port+1}"
    return ssh_host, triton_url, port

async def _ssh_start_server(ssh_host: str, conda_env: str, cmd: str,log_path:str):
    """Run remote command on gpu node via SSH (agent forwarding must be enabled)."""
    try:
        async with asyncssh.connect(
            ssh_host,
            username="hshastri_umass_edu",
            agent_forwarding=True,
        ) as conn:
            remote_cmd=(f"bash -lc '{cmds} && {activate_env} {conda_env} && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib && nohup {cmd}> {log_path} 2>&1 &'")

            print(f"[SSH] Launching on {ssh_host}: {remote_cmd}")
            proc = await conn.create_process(remote_cmd)
            await asyncio.sleep(10)   # small delay to ensure command sent
            proc.exit_status  # don’t wait; just trigger cleanup
            print(f"[SSH] {ssh_host}: detached.")

    except Exception as e:
        print(f"[SSH] Error on {ssh_host}: {e}")
        raise
    
def _send_control(ssh_host, port, cmd_data, payload_data):
    try:
        with ModelClient(f"{ssh_host}:{port}", "edge_control", init_timeout_s=120) as client:
            print(f"[SiteManager] Connected to control plane at {ssh_host}:{port}")
            resp = client.infer_batch(command=cmd_data, payload=payload_data)
            status = resp["status"][0].decode("utf-8")
            logger_summary = resp["logger_summary"][0].decode("utf-8")
            print(f"[PyTriton] {ssh_host}:{port} Status: {status}")
            return {"status": status, "logger_summary": logger_summary}
        
    except Exception as e:
        print(f"[PyTriton] Failed to deploy to {ssh_host}:{port}: {e}")
        return False
        
async def _deploy_one(s: DeploySpec):
    ssh_host, triton_url, grpc_port = _parse_url(s['device'])
    print(ssh_host,grpc_port,triton_url)
    # choose env + server command
    if s['backbone'] == "llava":
        conda_env = vlm_env
        server_cmd = f"python -u device/main.py --port {grpc_port} "
    elif s['backbone'] in ["momentlarge","momentbase",'momentsmall',"chronostiny","chronossmall","chronosbase","chronosmini","chronoslarge","papageip","papageis","papageisvri"] :
        conda_env = timeseries_env
        server_cmd = f"python -u device/main.py --port {grpc_port} "
    else:
        print(f"[WARN] Unknown backbone {s['backbone']}; skipping {s['device']}")
        return

    log_path = f"./device/logs/{ssh_host}_{s['backbone']}.log"
    # 1. Start Server via SSH
    await _ssh_start_server(ssh_host, conda_env, server_cmd, log_path)
    
    # 2. Send Deployment Config via Control Plane
    config_payload = {
        "backbone": s['backbone'],
        "decoders": [d for d in s['decoders']],
    }
    config_str = json.dumps(config_payload)
    cmd_data = np.array([[b"load"]], dtype=object)
    payload_data = np.array([[config_str.encode("utf-8")]], dtype=object)
    deployment_status = await asyncio.to_thread(_send_control, ssh_host, grpc_port+1, cmd_data, payload_data)
    return deployment_status

async def deploy_models(specs: List[DeploySpec]):
    results=await asyncio.gather(*[_deploy_one(s) for s in specs])
    print(f"[SiteManager] Deployment complete for {len(specs)} devices.")
    return results