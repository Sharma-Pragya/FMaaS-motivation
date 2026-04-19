import asyncio
import json
import os
from turtle import delay
from urllib.parse import urlparse

import asyncssh

from site_manager.config import activate_env, cmds, ssh_key, timeseries_env, username, vlm_env
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


def _split_grpc_url(grpc_url: str) -> tuple[str, int]:
    host, port = grpc_url.rsplit(":", 1)
    return host, int(port)


async def _ssh_start_server(ssh_host: str, username: str, conda_env: str, cmd: str, log_path: str,
                            cuda_visible: str | None = None):
    """Run remote command on gpu node via SSH (agent forwarding must be enabled)."""
    try:
        ssh_kwargs = dict(
            username=username,
            agent_forwarding=True,
            known_hosts=None,
        )
        if ssh_key:
            ssh_kwargs["client_keys"] = [ssh_key]
        else:
            ssh_kwargs["agent_path"] = os.environ.get("SSH_AUTH_SOCK")
        async with asyncssh.connect(ssh_host, **ssh_kwargs) as conn:
            cuda_env = f"export CUDA_VISIBLE_DEVICES={cuda_visible} && " if cuda_visible is not None else ""
            # Re-activate conda inside the nohup'd shell — `bash -lc` resets PATH,
            # so the outer activation does not carry into this inner login shell.
            launch_cmd = (
                "nohup bash -lc "
                "\"echo \\\"[Launcher] START ts=$(date -Is) shell_pid=$$\\\"; "
                f"{cmds} && {activate_env} {conda_env} "
                "&& export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib; "
                f"{cuda_env}{cmd}; "
                "rc=$?; "
                "echo \\\"[Launcher] EXIT ts=$(date -Is) rc=${rc}\\\"\" "
                f"> {log_path} 2>&1 &"
            )
            remote_cmd = (
                f"bash -lc '{cmds} && {activate_env} {conda_env} "
                f"&& export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib "
                f"&& {launch_cmd}'"
            )

            print(f"[SSH] Launching on {ssh_host}: {remote_cmd}")
            proc = await conn.create_process(remote_cmd)
            await asyncio.sleep(3)
            proc.exit_status
            print(f"[SSH] {ssh_host}: detached.")

    except Exception as exc:
        print(f"[SSH] Error on {ssh_host}: {exc}")
        raise RuntimeError(f"ssh_start_failed[{ssh_host}]: {exc}") from exc


async def _send_control(grpc_url: str, command: str, payload_json: str,
                        max_retries: int = 10, retry_delay: float = 5.0):
    """Send a control command to a device, retrying until the server is ready.

    The device process may take several seconds to start (especially when
    multiple instances share a GPU). Retrying avoids silently dropping the
    load command and leaving pipeline=None.
    """
    for attempt in range(1, max_retries + 1):
        client = EdgeRuntimeClient(grpc_url)
        try:
            print(f"[SiteManager] Sending '{command}' to {grpc_url} (attempt {attempt}/{max_retries})")
            timeout_s = 180.0 if command in ("load", "swap_backbone") else 60.0
            resp = await client.control(command, payload_json,timeout_s=timeout_s)
            print(f"[CustomGRPC] {grpc_url} Status: {resp['status']}")
            return resp
        except Exception as exc:
            print(f"[CustomGRPC] Failed to reach {grpc_url}: {exc}")
            if attempt < max_retries:
                print(f"[CustomGRPC] Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            else:
                print(f"[CustomGRPC] Giving up after {max_retries} attempts for {grpc_url}")
                return False
        finally:
            await client.close()


async def _deploy_one(spec: dict):
    ssh_host, grpc_url, grpc_port = _parse_url(spec["device"])
    print(ssh_host, grpc_port, grpc_url)
    if spec["backbone"] in ("qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-7b"):
        conda_env = vlm_env
        server_cmd = f"python -u device/main.py --port {grpc_port} --runtime-type vllm "
    elif spec["backbone"] == "llava" or spec["backbone"] in ("phi-3.5-vision-instruct", "phi") or spec["backbone"].startswith("qwen2.5"):
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

    scheduler_policy = spec.get("scheduler_policy", "stfq")
    server_cmd += f"--scheduler-policy {scheduler_policy} "

    max_batch_size = spec.get("max_batch_size", 5)
    max_batch_wait_ms = spec.get("max_batch_wait_ms", 0)
    server_cmd += f"--max-batch-size {max_batch_size} --max-batch-wait-ms {max_batch_wait_ms} "

    isolation_mode = spec.get("isolation_mode", "shared")
    server_cmd += f"--isolation-mode {isolation_mode} "

    gpu_memory_utilization = spec.get("gpu_memory_utilization", None)
    if gpu_memory_utilization is not None:
        server_cmd += f"--gpu-memory-utilization {gpu_memory_utilization} "

    max_model_len = spec.get("max_model_len", None)
    if max_model_len is not None:
        server_cmd += f"--max-model-len {max_model_len} "

    tpc_mode = spec.get("tpc_mode", None)
    tpc_partition = spec.get("tpc_partition", None)
    if tpc_mode and tpc_mode != "none" and tpc_partition:
        part_str = " ".join(str(p) for p in tpc_partition)
        server_cmd += f"--tpc-mode {tpc_mode} --tpc-partition {part_str} "

    worker_mode = spec.get("worker_mode", None)
    if worker_mode:
        server_cmd += f"--worker-mode {worker_mode} "

    # Build --task-rates from tasks dict if present
    tasks_dict = spec.get("tasks", {})
    if tasks_dict and scheduler_policy in ("stfq", "wfq", "token_bucket", "saba", "deadline_split"):
        rates_str = ",".join(f"{t}:{info['request_per_sec']:.4f}" for t, info in tasks_dict.items())
        server_cmd += f"--task-rates {rates_str} "

    cuda_suffix = spec.get("cuda", "").replace(":", "")
    log_dir = output_dir if output_dir else "./device/logs"
    log_path = f"{log_dir}/{ssh_host}_{cuda_suffix}_{spec['backbone']}_port{grpc_port}.log"

    # Extract GPU index from "cuda:N" for CUDA_VISIBLE_DEVICES so vLLM picks the right GPU
    cuda_visible = None
    if cuda_device:
        # "cuda:0" -> "0", "cuda:1" -> "1"
        cuda_visible = cuda_device.split(":")[-1] if ":" in cuda_device else None

    await _ssh_start_server(ssh_host, username, conda_env, server_cmd, log_path,
                            cuda_visible=cuda_visible)

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
    # Group specs by cuda device so deployments on the same physical GPU are
    # serialized. Loading two models concurrently on the same GPU causes the
    # second load to block on the device lock, making _send_control time out.
    # Deployments on different GPUs still run in parallel.
    from collections import defaultdict
    groups = defaultdict(list)  # cuda_key -> [spec, ...]
    for spec in specs:
        cuda_key = (spec.get("device", "").rsplit(":", 1)[0], spec.get("cuda", ""))
        groups[cuda_key].append(spec)

    async def _deploy_group(group_specs):
        results = []
        for spec in group_specs:
            try:
                result = await _deploy_one(spec)
                results.append(result)
            except Exception as exc:
                results.append({
                    "status": "error",
                    "device": spec.get("device"),
                    "backbone": spec.get("backbone"),
                    "error": str(exc),
                })
        return results

    group_results = await asyncio.gather(*[_deploy_group(g) for g in groups.values()])
    normalized = [r for group in group_results for r in group]
    print(f"[SiteManager] Deployment complete for {len(specs)} devices.")
    return normalized


async def _ssh_kill_server(ssh_host: str, username: str, grpc_port: int):
    """Gracefully kill a device server on a remote host."""
    try:
        ssh_kwargs = dict(
            username=username,
            agent_forwarding=True,
            known_hosts=None,
        )
        if ssh_key:
            ssh_kwargs["client_keys"] = [ssh_key]
        else:
            ssh_kwargs["agent_path"] = os.environ.get("SSH_AUTH_SOCK")
        async with asyncssh.connect(ssh_host, **ssh_kwargs) as conn:
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
