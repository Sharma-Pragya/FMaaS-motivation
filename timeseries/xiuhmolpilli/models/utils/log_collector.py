import time
import threading
import os
import json
from pynvml import *

LOG_INTERVAL = 0.001  # Logging interval in seconds
stop_logging = threading.Event()  # Stop signal

def init_nvml():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)  # assuming single-GPU
    return handle

def get_gpu_memory_and_util(handle):
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    util = nvmlDeviceGetUtilizationRates(handle)
    return {
        "gpu_mem_used_mb": mem_info.used / 1024**2,
        "gpu_util_percent": util.gpu,
    }

def get_cpu_memory_and_util():
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / 1024**2  # in MB
    cpu_util = psutil.cpu_percent(interval=1)     # in %
    return {
        "cpu_mem_used_mb": cpu_mem,
        "cpu_util_percent": cpu_util,
    }

# def write_log(folder_name, filename, data):
#     with open(f"{folder_name}/{filename}", "a") as f:
#         f.write(data + "\n")

# # Store metrics for averaging
# metrics = {
#     "cpu_usage": [],
#     "cpu_temp": [],
#     "gpu_usage": [],
#     "gpu_temp": [],
#     "memory_usage": [],
#     "swap_usage": [],
#     "gpu_mem_used": []
# }

# def average_metric(name):
#     values = metrics[name]
#     return sum(values) / len(values) if values else None

# def get_stat_jetson(folder_name):
#     from jtop import jtop
#     os.makedirs(folder_name, exist_ok=True)

#     with jtop() as jetson:
#         while not stop_logging.is_set() and jetson.ok():
#             timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
#             stats = jetson.stats

#             try:
#                 cpu_usage = sum(stats[f'CPU{i}'] for i in range(1, 7)) / 6
#                 gpu_usage = stats['GPU']
#                 mem_usage = round(stats['RAM'] * 100, 2)
#                 swap_usage = round(stats['SWAP'] * 100, 2)
#                 cpu_temp = stats['Temp CPU']
#                 gpu_temp = stats['Temp GPU']
#             except KeyError:
#                 continue

#             metrics['cpu_usage'].append(cpu_usage)
#             metrics['gpu_usage'].append(gpu_usage)
#             metrics['memory_usage'].append(mem_usage)
#             metrics['swap_usage'].append(swap_usage)
#             metrics['cpu_temp'].append(cpu_temp)
#             metrics['gpu_temp'].append(gpu_temp)

#             write_log(folder_name, "cpu_usage.log", f"{timestamp} - {cpu_usage:.2f}%")
#             write_log(folder_name, "gpu_usage.log", f"{timestamp} - {gpu_usage:.2f}%")
#             write_log(folder_name, "memory_usage.log", f"{timestamp} - {mem_usage:.2f}%")
#             write_log(folder_name, "swap_memory.log", f"{timestamp} - {swap_usage:.2f}%")
#             write_log(folder_name, "cpu_temp.log", f"{timestamp} - {cpu_temp:.2f}°C")
#             write_log(folder_name, "gpu_temp.log", f"{timestamp} - {gpu_temp:.2f}°C")

#             # time.sleep(LOG_INTERVAL)

# def get_stat_mac(folder_name):
#     import subprocess
#     os.makedirs(folder_name, exist_ok=True)

#     process = subprocess.Popen(
#         "macmon pipe",
#         shell=True,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         text=True
#     )

#     try:
#         for line in process.stdout:
#             if stop_logging.is_set():
#                 break

#             timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
#             try:
#                 data = json.loads(line.strip())
#                 ecpu_usage = data['ecpu_usage'][1]
#                 pcpu_usage = data['pcpu_usage'][1]
#                 cpu_usage = (ecpu_usage + pcpu_usage) / 2
#                 gpu_usage = data['gpu_usage'][1]
#                 mem_usage = data['memory']['ram_usage']
#                 swap_usage = data['memory']['swap_usage']
#                 cpu_temp = data['temp']['cpu_temp_avg']
#                 gpu_temp = data['temp']['gpu_temp_avg']

#                 metrics['cpu_usage'].append(cpu_usage)
#                 metrics['gpu_usage'].append(gpu_usage)
#                 metrics['memory_usage'].append(mem_usage)
#                 metrics['swap_usage'].append(swap_usage)
#                 metrics['cpu_temp'].append(cpu_temp)
#                 metrics['gpu_temp'].append(gpu_temp)

#                 write_log(folder_name, "cpu_usage.log", f"{timestamp} - {cpu_usage:.2f}%")
#                 write_log(folder_name, "gpu_usage.log", f"{timestamp} - {gpu_usage:.2f}%")
#                 write_log(folder_name, "memory_usage.log", f"{timestamp} - {mem_usage}")
#                 write_log(folder_name, "swap_memory.log", f"{timestamp} - {swap_usage}")
#                 write_log(folder_name, "cpu_temp.log", f"{timestamp} - {cpu_temp}")
#                 write_log(folder_name, "gpu_temp.log", f"{timestamp} - {gpu_temp}")

#             except Exception as e:
#                 print(f"Error collecting stats: {e}")
#                 continue

#     finally:
#         process.terminate()

# def get_stat_nvidia_server(folder_name):
#     import subprocess
#     import psutil

#     os.makedirs(folder_name, exist_ok=True)

#     while not stop_logging.is_set():
#         timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
#         try:
#             # GPU query
#             result = subprocess.run(
#                 ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,memory.used,memory.total",
#                  "--format=csv,noheader,nounits"],
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True
#             )
#             output = result.stdout.strip().split("\n")[0]
#             gpu_util, mem_util, gpu_temp, mem_used, mem_total = map(int, output.split(", "))

#             # CPU
#             cpu_usage = psutil.cpu_percent(interval=None)
#             cpu_temp = None
#             try:
#                 temps = psutil.sensors_temperatures()
#                 if "coretemp" in temps:
#                     cpu_temp = sum([t.current for t in temps["coretemp"]]) / len(temps["coretemp"])
#             except Exception:
#                 pass

#             mem = psutil.virtual_memory()
#             swap = psutil.swap_memory()

#             metrics['cpu_usage'].append(cpu_usage)
#             if cpu_temp: metrics['cpu_temp'].append(cpu_temp)
#             metrics['gpu_usage'].append(gpu_util)
#             metrics['gpu_temp'].append(gpu_temp)
#             metrics['memory_usage'].append(mem.percent)
#             metrics['swap_usage'].append(swap.percent)
#             metrics['gpu_mem_used'].append(mem_used)

#             write_log(folder_name, "cpu_usage.log", f"{timestamp} - {cpu_usage:.2f}%")
#             if cpu_temp:
#                 write_log(folder_name, "cpu_temp.log", f"{timestamp} - {cpu_temp:.2f}°C")
#             write_log(folder_name, "gpu_usage.log", f"{timestamp} - {gpu_util}%")
#             write_log(folder_name, "gpu_temp.log", f"{timestamp} - {gpu_temp}°C")
#             write_log(folder_name, "memory_usage.log", f"{timestamp} - {mem.percent:.2f}%")
#             write_log(folder_name, "swap_memory.log", f"{timestamp} - {swap.percent:.2f}%")
#             write_log(folder_name, "gpu_mem_usage.log", f"{timestamp} - {mem_used}MB / {mem_total}MB")

#         except Exception as e:
#             print(f"Error collecting NVIDIA server stats: {e}")

#         # time.sleep(LOG_INTERVAL)

# def start_logging(folder_name, device_type):
#     """Start logging based on device type."""
#     if device_type == 'jetson':
#         logging_thread = threading.Thread(target=get_stat_jetson, args=(folder_name,), daemon=True)
#     elif device_type == 'mac':
#         logging_thread = threading.Thread(target=get_stat_mac, args=(folder_name,), daemon=True)
#     elif device_type == 'nvidia_server':
#         logging_thread = threading.Thread(target=get_stat_nvidia_server, args=(folder_name,), daemon=True)
#     else:
#         print("Unsupported device type. Choose 'jetson', 'mac', or 'nvidia_server'.")
#         return

#     logging_thread.start()
#     return logging_thread

# def stop_logging_thread():
#     """Stop logging and return average metrics collected."""
#     stop_logging.set()
#     time.sleep(LOG_INTERVAL + 0.1)  # Give time for thread to finish

#     avg = {key: round(average_metric(key), 2) for key in metrics if metrics[key]}
#     return avg
