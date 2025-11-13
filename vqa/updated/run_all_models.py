import subprocess
import torch
import csv
import time
import importlib.util
import re

import os
from config import models_directory

os.environ["HF_HOME"] = models_directory
os.environ["HUGGINGFACE_HUB_CACHE"] = models_directory
os.environ["TRANSFORMERS_CACHE"] = models_directory

CONFIG_PATH = "config.py"

def get_config_vars():
    # Load variables from config.py
    spec = importlib.util.spec_from_file_location("config", CONFIG_PATH)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.model_name, config.log_file

def update_config_model(model_name_str):
    with open(CONFIG_PATH, "r") as f:
        lines = f.readlines()

    with open(CONFIG_PATH, "w") as f:
        for line in lines:
            if line.strip().startswith("model_name"):
                f.write(f'model_name="{model_name_str}"\n')
            else:
                f.write(line)

def restore_original_model_list(model_list):
    # Restore model_name = [...] at the end
    with open(CONFIG_PATH, "r") as f:
        lines = f.readlines()

    with open(CONFIG_PATH, "w") as f:
        for line in lines:
            if line.strip().startswith("model_name"):
                f.write(f"model_name = {repr(model_list)}\n")
            else:
                f.write(line)

def get_gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

def inject_gpu_name_in_log(log_file, gpu_name):
    try:
        with open(log_file, "r") as f:
            rows = list(csv.DictReader(f))
            fieldnames = rows[0].keys() if rows else []
    except FileNotFoundError:
        print(f"⚠️ Log file {log_file} not found. Creating a new one.")
        rows = []
        fieldnames = []

    # If log exists but is empty, try to detect fieldnames from standard structure
    if not fieldnames and rows:
        fieldnames = rows[0].keys()

    # Add gpu_name field if it doesn't exist
    if "gpu_name" not in fieldnames:
        fieldnames = list(fieldnames) + ["gpu_name"]

    # Inject gpu_name into the last row
    if rows:
        rows[-1]["gpu_name"] = gpu_name

    # Rewrite file with updated rows
    with open(log_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    model_list, log_file = get_config_vars()
    gpu_name = get_gpu_name()

    for model in model_list:
        print(f"\nRunning model: {model}")
        subprocess.run([
            "python", "gpu_run_model_inference_scene.py",
            "--model_name", model,
            "--log_file", log_file
        ], check=True)
        inject_gpu_name_in_log(log_file, gpu_name)
        time.sleep(1)

    print("\nAll models run. Restoring original config...")
    restore_original_model_list(model_list)
    print("config.py restored. Done!")

if __name__ == "__main__":
    main()
