import time
import threading
import os
import json
LOG_INTERVAL = 1  # Logging interval in seconds
stop_logging = threading.Event()  # Stop signal

def write_log(folder_name, filename, data):
    """Append logs to the appropriate file."""
    with open(f"{folder_name}/{filename}", "a") as f:
        f.write(data + "\n")

def get_stat_jetson(folder_name):
    """Collect system statistics for Jetson devices."""
    from jtop import jtop
    
    os.makedirs(folder_name, exist_ok=True)  # Ensure log directory exists
    
    with jtop() as jetson:
        while not stop_logging.is_set() and jetson.ok():
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            stats = jetson.stats
            
            try:
                cpu_usage = sum(stats[f'CPU{i}'] for i in range(1, 7)) / 6  # Avg CPU usage
                gpu_usage = stats['GPU']
                mem_usage = round(stats['RAM'] * 100, 2)  # Convert fraction to percentage
                swap_usage = round(stats['SWAP'] * 100, 2)  # Convert fraction to percentage
                cpu_temp = stats['Temp CPU']
                gpu_temp = stats['Temp GPU']
            except KeyError:
                continue  # If stats are missing, retry in the next interval
            
            # Write logs
            write_log(folder_name, "cpu_usage.log", f"{timestamp} - {cpu_usage:.2f}%")
            write_log(folder_name, "gpu_usage.log", f"{timestamp} - {gpu_usage:.2f}%")
            write_log(folder_name, "memory_usage.log", f"{timestamp} - {mem_usage:.2f}%")
            write_log(folder_name, "swap_memory.log", f"{timestamp} - {swap_usage:.2f}%")
            write_log(folder_name, "cpu_temp.log", f"{timestamp} - {cpu_temp:.2f}°C")
            write_log(folder_name, "gpu_temp.log", f"{timestamp} - {gpu_temp:.2f}°C")
            
            time.sleep(LOG_INTERVAL)

def get_stat_mac(folder_name):
    """Collect system statistics for Mac devices."""
    import subprocess
    
    os.makedirs(folder_name, exist_ok=True)  # Ensure log directory exists
    process = subprocess.Popen(
        "macmon pipe", 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )
    try:
        for line in process.stdout:
            if stop_logging.is_set():
                break  # Stop logging when requested

            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            try:
                data = json.loads(line.strip())
                ecpu_usage = data['ecpu_usage'][1]
                ecpu_freq = data['ecpu_usage'][0]
                pcpu_usage = data['pcpu_usage'][1]
                pcpu_freq = data['pcpu_usage'][0]
                gpu_usage = data['gpu_usage'][1]
                gpu_freq = data['gpu_usage'][0]
                mem_usage =  data['memory']['ram_usage']
                swap_usage = data['memory']['swap_usage']
                cpu_temp = data['temp']['cpu_temp_avg']
                gpu_temp = data['temp']['gpu_temp_avg']

                write_log(folder_name, "ecpu_usage.log", f"{timestamp} - {ecpu_usage:}%")
                write_log(folder_name, "ecpu_freq.log", f"{timestamp} - {ecpu_freq:}")
                write_log(folder_name, "pcpu_usage.log", f"{timestamp} - {pcpu_usage:}%")
                write_log(folder_name, "pcpu_freq.log", f"{timestamp} - {pcpu_freq:}")
                write_log(folder_name, "gpu_usage.log", f"{timestamp} - {gpu_usage}")
                write_log(folder_name, "gpu_freq.log", f"{timestamp} - {gpu_freq}")
                write_log(folder_name, "memory_usage.log", f"{timestamp} - {mem_usage:}")
                write_log(folder_name, "swap_memory.log", f"{timestamp} - {swap_usage:}")
                write_log(folder_name, "cpu_temp.log", f"{timestamp} - {cpu_temp}")
                write_log(folder_name, "gpu_temp.log", f"{timestamp} - {gpu_temp}")

    
            except Exception as e:
                print(f"Error collecting stats: {e}")
                continue  # Skip this cycle if an error occurs



    except Exception as e:
        print(f"Error collecting stats: {e}")

    finally:
        process.terminate()  # Ensure process cleanup

def start_logging(folder_name, device_type):
    """Start logging based on device type."""
    if device_type == 'jetson':
        logging_thread = threading.Thread(target=get_stat_jetson, args=(folder_name,), daemon=True)
    elif device_type == 'mac':
        logging_thread = threading.Thread(target=get_stat_mac, args=(folder_name,), daemon=True)
    else:
        print("Unsupported device type. Choose 'jetson' or 'mac'.")
        return

    logging_thread.start()
    return logging_thread

def stop_logging_thread():
    """Stop the logging process."""
    stop_logging.set()