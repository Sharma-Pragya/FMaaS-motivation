import time
import psutil
import torch
import csv
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# GPU monitoring (skip if not available)
GPU_MONITOR_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITOR_AVAILABLE = True
except (ImportError, OSError, pynvml.NVMLError):
    print("[INFO] NVML not available — GPU monitoring will be disabled.")
    GPU_MONITOR_AVAILABLE = False

def measure_latency(func, *args, **kwargs):
    start = time.perf_counter()
    output = func(*args, **kwargs)
    end = time.perf_counter()
    latency_ms = (end - start) * 1000
    return output, latency_ms

def main(model_name, dataset_name, split="test[:100]", output_csv="metrics_log_mac.csv"):
    # Set device
    device = "mps"  # Force MPS on MacBook
    process = psutil.Process()

    # Model loading
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    load_end = time.perf_counter()
    model_load_duration = load_end - load_start

    # Memory after load
    memory_after_load_MB = process.memory_info().rss / (1024 * 1024)

    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    total_latency = []
    correct_predictions = 0
    total_prompt_tokens = 0

    # CPU and GPU usage monitoring
    cpu_usages = []
    gpu_usages = []

    print(f"Running inference on {len(dataset)} samples...")

    inference_memory_peak_MB = memory_after_load_MB

    for example in dataset:
        inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.tensor([example["label"]]).to(device)



        # Count prompt tokens
        prompt_tokens = inputs['input_ids'].shape[1]
        total_prompt_tokens += prompt_tokens

        # CPU usage snapshot
        cpu_usage = psutil.cpu_percent(interval=None)
        cpu_usages.append(cpu_usage)

        # GPU usage snapshot (skip if not available)
        if GPU_MONITOR_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_usages.append(util.gpu)
            except Exception:
                gpu_usages.append(None)

        # Inference and latency
        with torch.no_grad():
            outputs, latency_ms = measure_latency(model, **inputs)
            logits = outputs.logits
            predictions = logits.argmax(dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_latency.append(latency_ms)

        # Update memory usage
        mem_now_MB = process.memory_info().rss / (1024 * 1024)
        inference_memory_peak_MB = max(inference_memory_peak_MB, mem_now_MB)

    total_inference_time_sec = sum(total_latency) / 1000
    accuracy = correct_predictions / len(dataset)
    avg_latency_ms = sum(total_latency) / len(total_latency)

    # Throughput calculations
    prompt_eval_rate = total_prompt_tokens / total_inference_time_sec  # tokens/sec

    # No TTFT or generation in classification, set to None
    ttft_ms = None
    total_generated_tokens = 0
    throughput_tokens_per_sec = None

    # CPU and GPU stats
    avg_cpu_usage = sum(cpu_usages) / len(cpu_usages) if cpu_usages else None
    avg_gpu_usage = (sum(gpu_usages) / len([g for g in gpu_usages if g is not None])) if gpu_usages and any(gpu_usages) else None

    # Log everything into a dictionary
    results = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "device": device,
        "model_load_duration_sec": model_load_duration,
        "memory_after_load_mb": memory_after_load_MB,
        "peak_memory_usage_mb": inference_memory_peak_MB,
        "avg_cpu_usage_percent": avg_cpu_usage,
        "avg_gpu_usage_percent": avg_gpu_usage,
        "total_prompt_tokens": total_prompt_tokens,
        "prompt_eval_rate_tokens_per_sec": prompt_eval_rate,
        "ttft_ms": ttft_ms,
        "total_generated_tokens": total_generated_tokens,
        "throughput_tokens_per_sec": throughput_tokens_per_sec,
        "accuracy": accuracy,
        "average_latency_ms": avg_latency_ms,
        "total_inference_time_sec": total_inference_time_sec,
        "num_samples": len(dataset)
    }

    # Save to CSV
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

    print(f"\nMetrics written to {output_csv}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python single_run_metrics_mac.py <model_name> <dataset_name>")
        print("Example: python single_run_metrics_mac.py bert-base-uncased imdb")
        sys.exit(1)

    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    main(model_name, dataset_name)
