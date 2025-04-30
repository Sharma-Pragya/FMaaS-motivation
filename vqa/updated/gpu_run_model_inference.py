import os
import time
import json
import torch
import psutil
import tracemalloc
from huggingface_hub import login
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from config import model_name, dataset, dataset_json, models_directory
import csv
from pynvml import *

# ============ Setup ============
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# ============ Load Model ============
def load_model(model_id):
    with open("../../hf-token.txt", "r") as f:
        hf_token = f.read().strip()
    login(token=hf_token)

    start_time = time.time()
    try:
        if "llava" in model_name:
            from transformers import LlavaForConditionalGeneration, LlavaProcessor
            processor = LlavaProcessor.from_pretrained(model_id)
            model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        elif "moondream" in model_name:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cuda")
        else:
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, token=hf_token, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, cache_dir=models_directory, token=hf_token)
            model = model.to(device)
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_id}: {e}")
    load_duration = time.time() - start_time
    memory_after_load_mb = psutil.Process().memory_info().rss / 1024 / 1024
    return model, processor, load_duration, memory_after_load_mb

# ============ Load Dataset ============
def load_dataset(json_path):
    with open(json_path, "r") as f:
        raw_data = json.load(f)
    return list(raw_data.values())

# ============ Metrics ============

def init_nvml():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)  # assuming single-GPU
    return handle

def get_gpu_memory_and_util(handle):
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    util = nvmlDeviceGetUtilizationRates(handle)
    return {
        "gpu_mem_used_mb": mem_info.used / 1024**2,
        "gpu_util_percent": util.gpu
    }

def run_inference(model, processor, data):
    model.eval()
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_latency = 0
    correct = 0
    evaluated = 0
    total_gpu_util = 0
    total_gpu_mem = 0
    total_cpu_percent = 0

    tracemalloc.start()
    handle = init_nvml()
    start_time = time.time()

    is_moondream = "moondream" in model_name

    for i, item in enumerate(data):
        print("Running inference")
        question = item["question"]
        gt_answer = item.get("answer", "").strip().lower()
        image_id = item["image_id"]
        image_name = f"COCO_val2014_{image_id:012d}.jpg"
        image_path = os.path.join("dataset", "val2014", image_name)

        if not os.path.exists(image_path):
            print(f"Skipping [{i}] - Image not found: {image_path, image_name}")
            continue
        print(f"✅ Using [{i}] - Found image: {image_name}")

        image = Image.open(image_path).convert("RGB")

        # conversation = [{"role": "user", "content": question}]
        # prompt = processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        # inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        gpu_before = get_gpu_memory_and_util(handle)
        start = time.time()
        with torch.no_grad():
            if is_moondream:
                image_embeds = model.encode_image(image)
                answer = model.answer_question(image_embeds=image_embeds, question=question+ " Please answer in one word.")
                input_tokens = len(question.split())
                generated_tokens = len(answer.split())
            else:
                prompt = f"USER: <image>\n{question}\nPlease answer in one word.\nASSISTANT:"
                inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
                input_tokens = inputs["input_ids"].shape[1]
                outputs = model.generate(**inputs, max_new_tokens=10)
                response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                answer = response.split("ASSISTANT:")[-1].strip().split()[0]
                generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        latency = time.time() - start

        total_prompt_tokens += input_tokens
        total_generated_tokens += generated_tokens
        total_latency += latency

        total_cpu_percent += psutil.cpu_percent(interval=None)
        gpu_after = get_gpu_memory_and_util(handle)
        gpu_mem_delta = gpu_after["gpu_mem_used_mb"] - gpu_before["gpu_mem_used_mb"]
        print(f"[{i}] GPU util: {gpu_after['gpu_util_percent']}%, Mem used: Δ{gpu_mem_delta:.2f} MB")
        total_gpu_util += gpu_after["gpu_util_percent"]
        total_gpu_mem += gpu_after["gpu_mem_used_mb"]

        evaluated += 1
        if gt_answer.lower() in answer.lower():
            correct += 1
        print(f"[{i}] Q: {question}")
        print(f"    GT: {gt_answer.lower()}")
        print(f"    →  Predicted: {answer.lower()}")
        print(f"[{i}] Accuracy so far: {correct}/{evaluated} = {correct / evaluated:.2f}")

    total_inference_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    accuracy = correct / len(data) if data else 0
    avg_gpu_util = total_gpu_util / len(data)
    avg_gpu_mem = total_gpu_mem / len(data)
    avg_cpu_percent = total_cpu_percent / len(data)

    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_generated_tokens": total_generated_tokens,
        "ttft_ms": 0,
        "avg_latency_ms": (total_latency / len(data)) * 1000,
        "throughput_tps": total_generated_tokens / total_inference_time,
        "accuracy": accuracy,
        "total_time": total_inference_time,
        "peak_memory_mb": peak / 1024 / 1024,
        "cpu_percent": avg_cpu_percent,
        "gpu_percent": avg_gpu_util,
        "mem_util": avg_gpu_mem,
        "num_samples": len(data),
    }

# ============ Main ============
def main():
    model, processor, load_duration, memory_after_load_mb = load_model(model_name)
    data = load_dataset(dataset_json)
    data = data[:100]
    results = run_inference(model, processor, data)

    metrics = {
        "model_name": model_name,
        "dataset_name": dataset,
        "device": device,
        "model_load_duration_sec": load_duration,  # seconds
        "memory_after_load_mb": memory_after_load_mb,  # MB (CPU)
        "peak_memory_usage_mb": results["peak_memory_mb"],  # MB (CPU peak)
        "avg_cpu_usage_percent": results["cpu_percent"],  # %
        "avg_gpu_usage_percent": results["gpu_percent"],  # %
        "avg_gpu_mem_usage_mb": results["mem_util"],  # MB (GPU memory used)
        "total_prompt_tokens": results["total_prompt_tokens"],  # tokens
        "prompt_eval_rate_tokens_per_sec": results["total_prompt_tokens"] / results["total_time"],  # tokens/sec
        "ttft_ms": results["ttft_ms"],  # ms (placeholder, currently 0)
        "total_generated_tokens": results["total_generated_tokens"],  # tokens
        "throughput_tokens_per_sec": results["throughput_tps"],  # tokens/sec
        "accuracy": results["accuracy"],  # [0,1]
        "average_latency_ms": results["avg_latency_ms"],  # ms
        "total_inference_time_sec": results["total_time"],  # seconds
        "num_samples": results["num_samples"]  # count
    }

    csv_path = "metrics_log_linux.csv"
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    print("Inference completed. Metrics saved to metrics_log_linux.csv")

if __name__ == "__main__":
    main()