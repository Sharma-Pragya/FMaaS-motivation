import json
import asyncio
import asyncssh
import numpy as np
from urllib.parse import urlparse
from pytriton.client import ModelClient
from site_manager.config import cmds, activate_env, vlm_env, timeseries_env, username


def _parse_url(device_url: str) -> tuple[str, str, int]:
    """
    Returns (ssh_host, triton_url)
    triton_url is formatted as 'host:port' for ModelClient.
    """
    p = urlparse(device_url)
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

async def _ssh_start_server(ssh_host: str, username: str, conda_env: str, cmd: str, log_path: str):
    """Run remote command on gpu node via SSH (agent forwarding must be enabled)."""
    try:
        async with asyncssh.connect(
            ssh_host,
            username=username,
            agent_forwarding=True,
        ) as conn:
            remote_cmd=(f"bash -lc '{cmds} && {activate_env} {conda_env} && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib && nohup {cmd}> {log_path} 2>&1 &'")

            print(f"[SSH] Launching on {ssh_host}: {remote_cmd}")
            proc = await conn.create_process(remote_cmd)
            await asyncio.sleep(3)   # small delay to ensure nohup submitted before SSH closes
            proc.exit_status  # don't wait; just trigger cleanup
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


async def _deploy_one(s: dict):
    ssh_host, triton_url, grpc_port = _parse_url(s['device'])
    print(ssh_host, grpc_port, triton_url)
    # choose env + server command
    if s['backbone'] == "llava":
        conda_env = vlm_env
        server_cmd = f"python -u device/main.py --port {grpc_port} "
    elif s['backbone'] in ["momentlarge","momentbase",'momentsmall',"chronostiny","chronossmall","chronosbase","chronosmini","chronoslarge","papageip","papageis","papageissvri"] :
        conda_env = timeseries_env
        server_cmd = f"python -u device/main.py --port {grpc_port} "
    else:
        print(f"[WARN] Unknown backbone {s['backbone']}; skipping {s['device']}")
        return

    cuda_device = s.get("cuda", None)
    if cuda_device:
        server_cmd += f"--cuda {cuda_device} "

    log_path = f"./device/logs/{ssh_host}_{s['backbone']}.log"

    # 1. Start Server via SSH
    await _ssh_start_server(ssh_host, username, conda_env, server_cmd, log_path)
    
    # 2. Send Deployment Config via Control Plane
    config_payload = {
        "backbone": s['backbone'],
        "decoders": s['decoders'],
    }
    config_str = json.dumps(config_payload)
    cmd_data = np.array([[b"load"]], dtype=object)
    payload_data = np.array([[config_str.encode("utf-8")]], dtype=object)
    deployment_status = await asyncio.to_thread(_send_control, ssh_host, grpc_port+1, cmd_data, payload_data)
    return deployment_status

async def _add_decoder_to_device(device_url: str, decoders: list) -> dict:
    """Hot-add decoders to a running device server (no SSH needed).

    The device already has a backbone loaded and a PyTriton server running.
    This sends an "add_decoder" control command via gRPC to load new
    decoder heads on the existing backbone.

    Args:
        device_url: Device endpoint string (e.g. "gpu-node:8000").
        decoders: List of {"task": str, "type": str, "path": str} dicts.

    Returns:
        Status dict from the device, or False on failure.
    """
    ssh_host, _, grpc_port = _parse_url(device_url)
    config_payload = {"decoders": decoders}
    config_str = json.dumps(config_payload)
    cmd_data = np.array([[b"add_decoder"]], dtype=object)
    payload_data = np.array([[config_str.encode("utf-8")]], dtype=object)

    # _send_control is synchronous; run in thread to keep async context
    result = await asyncio.to_thread(
        _send_control, ssh_host, grpc_port + 1, cmd_data, payload_data
    )
    return result


async def _swap_backbone_on_device(device_url: str, new_backbone: str, decoders: list) -> dict:
    """Send a swap_backbone control command to a running device server.

    The device frees the old backbone from GPU memory and loads the new one
    in-process — no SSH kill/start needed.

    Args:
        device_url: Device endpoint string (e.g. "gpu-node:8000").
        new_backbone: Name of the new backbone to load (e.g. "chronosbase").
        decoders: List of {"task": str, "type": str, "path": str} dicts.

    Returns:
        Status dict from the device, or False on failure.
    """
    ssh_host, _, grpc_port = _parse_url(device_url)
    config_payload = {"backbone": new_backbone, "decoders": decoders}
    config_str = json.dumps(config_payload)
    cmd_data = np.array([[b"swap_backbone"]], dtype=object)
    payload_data = np.array([[config_str.encode("utf-8")]], dtype=object)

    result = await asyncio.to_thread(
        _send_control, ssh_host, grpc_port + 1, cmd_data, payload_data
    )
    return result


async def deploy_models(specs: list):
    results=await asyncio.gather(*[_deploy_one(s) for s in specs])
    print(f"[SiteManager] Deployment complete for {len(specs)} devices.")
    return results


# ── Device cleanup ───────────────────────────────────────────────────────────

async def _ssh_kill_server(ssh_host: str, username: str, grpc_port: int):
    """Gracefully kill a Triton server on a remote host, then clean up resources.

    Steps:
      1. SIGTERM — give PyTriton/Triton time to release GPU mem, shared mem, temp dirs.
      2. Wait 5s for graceful shutdown.
      3. SIGKILL — force-kill anything still lingering.
      4. Clean up leaked shared-memory segments and PyTriton workspace dirs.
      5. Wait 5s for CUDA driver to fully release GPU memory.
    """
    try:
        async with asyncssh.connect(
            ssh_host,
            username=username,
            agent_forwarding=True,
        ) as conn:
            ports = [grpc_port, grpc_port + 1, grpc_port + 2]

            kill_cmd = (
                # 1. Graceful SIGTERM
                f"for p in {' '.join(str(p) for p in ports)}; do "
                f"  fuser -TERM $p/tcp 2>/dev/null; "
                f"done; "
                # 2. Wait for graceful shutdown
                f"sleep 5; "
                # 3. Force-kill survivors
                f"for p in {' '.join(str(p) for p in ports)}; do "
                f"  fuser -k $p/tcp 2>/dev/null; "
                f"done; "
                # 4. Clean up shared memory and PyTriton temp dirs
                f"rm -f /dev/shm/*triton* 2>/dev/null; "
                f"rm -rf $HOME/.cache/pytriton/workspace_* 2>/dev/null; "
                f"rm -rf /tmp/folder* 2>/dev/null;"
                # 5. Wait for CUDA driver to reclaim GPU memory
                f"sleep 2; "
                f"true"
            )
            result = await conn.run(kill_cmd)
            print(f"[SSH] Killed Triton on {ssh_host}:{grpc_port} "
                  f"(exit={result.exit_status})")

    except Exception as e:
        print(f"[SSH] Error killing server on {ssh_host}:{grpc_port}: {e}")


async def shutdown_devices(specs: list):
    """Kill all Triton servers launched for the given deployment specs.
    
    Extracts (host, port) from each deployment spec and SSHes in to
    kill the processes. Safe to call even if servers are already dead.
    
    Args:
        specs: List of deployment spec dicts (same format as received
               from the orchestrator via MQTT).
    """
    # Collect unique (host, port) pairs to avoid killing the same server twice
    seen = set()
    tasks = []
    for s in specs:
        ssh_host, _, grpc_port = _parse_url(s['device'])
        key = (ssh_host, grpc_port)
        if key not in seen:
            seen.add(key)
            tasks.append(_ssh_kill_server(ssh_host, username, grpc_port))

    if tasks:
        await asyncio.gather(*tasks)
    print(f"[SiteManager] Shutdown complete for {len(seen)} device server(s).")