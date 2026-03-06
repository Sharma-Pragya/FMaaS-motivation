import asyncio
import json
import os
from urllib.parse import urlparse

import asyncssh

from site_manager.config import activate_env, cmds, timeseries_env, username, vlm_env
from site_manager.grpc_client import EdgeRuntimeClient
from site_manager.storage import get_output_dir


def _parse_url(device_url: str) -> tuple[str, str, int]:
    """
    Returns (ssh_host, grpc_url, grpc_port)
    grpc_url is formatted as 'host:port' for the custom gRPC client.
    """
    parsed = urlparse(device_url)
    if parsed.scheme and parsed.path:
        ssh_host = parsed.scheme
        port = int(parsed.path) if parsed.path else 8000
        grpc_url = f"{parsed.scheme}:{port}"
    else:
        ssh_host = device_url
        port = 8000
        grpc_url = f"{device_url}:{port}"
    return ssh_host, grpc_url, port


async def _ssh_start_server(ssh_host: str, username: str, conda_env: str, cmd: str, log_path: str):
    """Run remote command on gpu node via SSH (agent forwarding must be enabled)."""
    try:
        async with asyncssh.connect(
            ssh_host,
            username=username,
            agent_forwarding=True,
            agent_path=os.environ.get("SSH_AUTH_SOCK"),
            known_hosts=None,
        ) as conn:
            remote_cmd = (
                f"bash -lc '{cmds} && {activate_env} {conda_env} "
                f"&& export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib "
                f"&& nohup {cmd}> {log_path} 2>&1 &'"
            )

            print(f"[SSH] Launching on {ssh_host}: {remote_cmd}")
            proc = await conn.create_process(remote_cmd)
            await asyncio.sleep(3)
            proc.exit_status
            print(f"[SSH] {ssh_host}: detached.")

    except Exception as exc:
        print(f"[SSH] Error on {ssh_host}: {exc}")
        raise RuntimeError(f"ssh_start_failed[{ssh_host}]: {exc}") from exc


async def _send_control(grpc_url: str, command: str, payload_json: str):
    client = EdgeRuntimeClient(grpc_url)
    try:
        print(f"[SiteManager] Connected to control plane at {grpc_url}")
        resp = await client.control(command, payload_json)
        print(f"[CustomGRPC] {grpc_url} Status: {resp['status']}")
        return resp
    except Exception as exc:
        print(f"[CustomGRPC] Failed to deploy to {grpc_url}: {exc}")
        return False
    finally:
        await client.close()


async def _deploy_one(spec: dict):
    ssh_host, grpc_url, grpc_port = _parse_url(spec["device"])
    print(ssh_host, grpc_port, grpc_url)
    if spec["backbone"] == "llava":
        conda_env = vlm_env
        server_cmd = f"python -u device/main.py --port {grpc_port} "
    elif spec["backbone"] in [
        "momentlarge",
        "momentbase",
        "momentsmall",
        "chronostiny",
        "chronossmall",
        "chronosbase",
        "chronosmini",
        "chronoslarge",
        "papageip",
        "papageis",
        "papageissvri",
    ]:
        conda_env = timeseries_env
        server_cmd = f"python -u device/main.py --port {grpc_port} "
    else:
        print(f"[WARN] Unknown backbone {spec['backbone']}; skipping {spec['device']}")
        return

    cuda_device = spec.get("cuda", None)
    if cuda_device:
        server_cmd += f"--cuda {cuda_device} "

    output_dir = get_output_dir()
    if output_dir:
        server_cmd += f"--output-dir {output_dir} "

    log_path = f"./device/logs/{ssh_host}_{spec['backbone']}.log"

    await _ssh_start_server(ssh_host, username, conda_env, server_cmd, log_path)

    config_payload = {
        "backbone": spec["backbone"],
        "decoders": spec["decoders"],
    }
    deployment_status = await _send_control(grpc_url, "load", json.dumps(config_payload))
    return deployment_status


async def _add_decoder_to_device(device_url: str, decoders: list) -> dict:
    """Hot-add decoders to a running device server (no SSH needed)."""
    _, grpc_url, _ = _parse_url(device_url)
    config_payload = {"decoders": decoders}
    return await _send_control(grpc_url, "add_decoder", json.dumps(config_payload))


async def _swap_backbone_on_device(device_url: str, new_backbone: str, decoders: list) -> dict:
    """Send a swap_backbone control command to a running device server."""
    _, grpc_url, _ = _parse_url(device_url)
    config_payload = {"backbone": new_backbone, "decoders": decoders}
    return await _send_control(grpc_url, "swap_backbone", json.dumps(config_payload))


async def deploy_models(specs: list):
    results = await asyncio.gather(*[_deploy_one(s) for s in specs], return_exceptions=True)
    normalized = []
    for spec, result in zip(specs, results):
        if isinstance(result, Exception):
            normalized.append(
                {
                    "status": "error",
                    "device": spec.get("device"),
                    "backbone": spec.get("backbone"),
                    "error": str(result),
                }
            )
        else:
            normalized.append(result)
    print(f"[SiteManager] Deployment complete for {len(specs)} devices.")
    return normalized


async def _ssh_kill_server(ssh_host: str, username: str, grpc_port: int):
    """Gracefully kill a device server on a remote host."""
    try:
        async with asyncssh.connect(
            ssh_host,
            username=username,
            agent_forwarding=True,
            agent_path=os.environ.get("SSH_AUTH_SOCK"),
            known_hosts=None,
        ) as conn:
            kill_cmd = (
                f"fuser -TERM {grpc_port}/tcp 2>/dev/null; "
                f"sleep 2; "
                f"fuser -k {grpc_port}/tcp 2>/dev/null; "
                f"true"
            )
            result = await conn.run(kill_cmd)
            print(f"[SSH] Killed device server on {ssh_host}:{grpc_port} (exit={result.exit_status})")

    except Exception as exc:
        print(f"[SSH] Error killing server on {ssh_host}:{grpc_port}: {exc}")


async def shutdown_devices(specs: list):
    """Kill all device servers launched for the given deployment specs."""
    seen = set()
    tasks = []
    for spec in specs:
        ssh_host, _, grpc_port = _parse_url(spec["device"])
        key = (ssh_host, grpc_port)
        if key not in seen:
            seen.add(key)
            tasks.append(_ssh_kill_server(ssh_host, username, grpc_port))

    if tasks:
        await asyncio.gather(*tasks)
    print(f"[SiteManager] Shutdown complete for {len(seen)} device server(s).")
