import os
import time
import json
import torch
import psutil
import tracemalloc
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import model_name, dataset, dataset_json, models_directory
import csv

# ============ Setup ============
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32  # FP16 not fully supported on MPS for all ops

# ============ Load Model ============
def load_model(model_id):
    # Load token from file
    with open("../../hf-token.txt", "r") as f:
        hf_token = f.read().strip()
    login(token=hf_token)

    start_time = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=models_directory, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, cache_dir=models_directory, token=hf_token)
        model = model.to(device)
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_id}: {e}")
    load_duration = time.time() - start_time
    return model, tokenizer, load_duration

# ============ Load Dataset ============
def load_dataset(json_path):
    with open(json_path, "r") as f:
        raw_data = json.load(f)
    return list(raw_data.values())


# ============ Metrics ============
def get_memory_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def run_inference(model, tokenizer, data):
    model.eval()
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_latency = 0
    correct = 0

    tracemalloc.start()
    start_time = time.time()

    for i, item in enumerate(data):
        print("Running inference")
        question = item["question"]
        gt_answer = item.get("answer", "").strip().lower()

        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_tokens = inputs["input_ids"].shape[1]
        total_prompt_tokens += input_tokens

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
        latency = time.time() - start
        total_latency += latency

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("Answer:")[-1].strip().lower()
        if gt_answer in answer:
            correct += 1
        print(f"[{i}] Q: {question}")
        print(f"    GT: {gt_answer}")
        print(f"    →  Predicted: {answer}")
        generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        total_generated_tokens += generated_tokens

    total_inference_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    accuracy = correct / len(data) if data else 0

    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_generated_tokens": total_generated_tokens,
        "ttft_ms": 0,
        "avg_latency_ms": (total_latency / len(data)) * 1000,
        "throughput_tps": total_generated_tokens / total_inference_time,
        "accuracy": accuracy,
        "total_time": total_inference_time,
        "memory_after_load_mb": get_memory_mb(),
        "peak_memory_mb": peak / 1024 / 1024,
        "cpu_percent": get_cpu_usage(),
        "gpu_percent": None,  # MPS doesn't expose this directly
        "num_samples": len(data),
    }

# ============ Main ============
def main():
    model, tokenizer, load_duration = load_model(model_name)
    data = load_dataset(dataset_json)
    data = data[:100]
    results = run_inference(model, tokenizer, data)

    metrics = {
        "model_name": model_name,
        "dataset_name": dataset,
        "device": device,
        "model_load_duration_sec": load_duration,
        "memory_after_load_mb": results["memory_after_load_mb"],
        "peak_memory_usage_mb": results["peak_memory_mb"],
        "avg_cpu_usage_percent": results["cpu_percent"],
        "avg_gpu_usage_percent": results["gpu_percent"],
        "total_prompt_tokens": results["total_prompt_tokens"],
        "prompt_eval_rate_tokens_per_sec": results["total_prompt_tokens"] / results["total_time"],
        "ttft_ms": results["ttft_ms"],
        "total_generated_tokens": results["total_generated_tokens"],
        "throughput_tokens_per_sec": results["throughput_tps"],
        "accuracy": results["accuracy"],
        "average_latency_ms": results["avg_latency_ms"],
        "total_inference_time_sec": results["total_time"],
        "num_samples": results["num_samples"],
    }

    with open("metrics_log_mac.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    print("Inference completed. Metrics saved to metrics_log_mac.csv")

if __name__ == "__main__":
    main()