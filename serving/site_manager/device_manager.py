# site_manager/device_manager.py
import subprocess, os

DEVICE_PROCESSES = {}

def launch_model(device: str, model_path: str, port: int):
    """Launch a model server process on a device"""
    #ssh into unity
    #sattach into that device
    cmd = [
        "ssh", device,
        f"nohup python3 -m device.inference_server --model {model_path} --port {port} > /tmp/device_{device}.log 2>&1 &"
    ]
    print(f"[DeviceManager] Launching model on {device}: {model_path}")
    subprocess.Popen(" ".join(cmd), shell=True)
    DEVICE_PROCESSES[device] = {"model": model_path, "port": port}

def stop_model(device: str):
    """Stop model process on a device"""
    print(f"[DeviceManager] Stopping model on {device}")
    subprocess.Popen(f"ssh {device} pkill -f inference_server", shell=True)
    DEVICE_PROCESSES.pop(device, None)
