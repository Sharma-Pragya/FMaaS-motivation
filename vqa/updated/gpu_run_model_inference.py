import os
import time
import json
import torch
import psutil
import tracemalloc
from huggingface_hub import login
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from config import model_name, dataset, dataset_json, models_directory
import csv, argparse
from pynvml import *
import argparse

# ============ Parse CLI Arguments ============
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, help="Model name to run")
    parser.add_argument("--log_file", type=str, default="metrics_log_linux.csv", help="Output CSV file")
    return parser.parse_args()

args = parse_args()
model_name = args.model_name if args.model_name else model_name
log_file = args.log_file

# ============ Setup ============
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(device)

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

def get_cpu_memory_and_util():
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / 1024**2  # in MB
    cpu_util = psutil.cpu_percent()     # in %
    return {
        "cpu_mem_used_mb": cpu_mem,
        "cpu_util_percent": cpu_util,
    }

# ============ Load Model ============
def load_model(model_id, handle):
    with open("../../hf-token.txt", "r") as f:
        hf_token = f.read().strip()
    login(token=hf_token)

    load_memory_before = get_gpu_memory_and_util(handle)["gpu_mem_used_mb"]
    start_time = time.time()
    try:
        if "llava-1.5" in model_name:
            from transformers import LlavaForConditionalGeneration, LlavaProcessor
            processor = LlavaProcessor.from_pretrained(model_id, cache_dir=models_directory)
            model = LlavaForConditionalGeneration.from_pretrained(model_id, cache_dir=models_directory, torch_dtype=torch.float16).to("cuda")
        elif "llava-v1.6" in model_name or "llava-next" in model_name:
            from transformers import LlavaNextForConditionalGeneration, AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True)
            model = LlavaNextForConditionalGeneration.from_pretrained(model_id, cache_dir=models_directory, torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True, use_flash_attention_2=True).to("cuda")
        elif "moondream" in model_name:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True).to("cuda")
        elif "qwen" in model_name.lower():
            from transformers import AutoModelForVision2Seq, AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(model_id, cache_dir=models_directory, torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
        elif "molmo" in model_name.lower():
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, torch_dtype=dtype).to("cuda")
        elif "phi" in model_name.lower():
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, attn_implementation="eager")
            model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, torch_dtype=dtype, attn_implementation="eager").to("cuda")
        elif "smolvlm" in model_name.lower():
            from transformers import AutoModelForVision2Seq, AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(model_id, cache_dir=models_directory, torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
        elif "llama" in model_name:
            from transformers import MllamaForConditionalGeneration, AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True)
            model = MllamaForConditionalGeneration.from_pretrained(model_id, cache_dir=models_directory, torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
        else:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, token=hf_token, use_fast=True, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, cache_dir=models_directory, token=hf_token, trust_remote_code=True)
            model = model.to(device)
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_id}: {e}")
    load_duration = time.time() - start_time
    memory_after_load_mb = get_gpu_memory_and_util(handle)["gpu_mem_used_mb"] - load_memory_before
    return model, processor, load_duration, memory_after_load_mb

# ============ Load Dataset ============
def load_dataset(json_path):
    with open(json_path, "r") as f:
        raw_data = json.load(f)
    return list(raw_data.values())

def run_inference(model, processor, data, handle):
    model.eval()
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_latency = 0
    correct = 0
    evaluated = 0
    total_gpu_util = 0
    total_gpu_mem = 0
    total_cpu_util = 0
    total_cpu_mem = 0

    tracemalloc.start()
    gpu_before = get_gpu_memory_and_util(handle)
    # print("GPU Before: ", gpu_before)
    cpu_before = get_cpu_memory_and_util()
    start_time = time.time()

    is_moondream = "moondream" in model_name
    is_qwen = "qwen" in model_name.lower()
    is_molmo = "molmo" in model_name.lower()
    is_phi = "phi" in model_name.lower()
    is_smolvlm = "smolvlm" in model_name.lower()
    is_llama = "llama-3.2" in model_name.lower()

    for i, item in enumerate(data):
        print("\nRunning inference")
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

        start = time.time()
        
        with torch.no_grad():
            if is_moondream:
                image_embeds = model.encode_image(image)
                answer = model.answer_question(image_embeds=image_embeds, question=question+ " Please answer in one word.")
                input_tokens = len(question.split())
                generated_tokens = len(answer.split())
            elif is_qwen:
                from qwen_vl_utils import process_vision_info
                # Build messages for vision + text input
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question+" Please answer in one word."},
                        ],}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",).to("cuda")
                outputs = model.generate(**inputs, max_new_tokens=20)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, outputs)]
                response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
                answer = response.split()[0] if response else ""
                input_tokens = inputs.input_ids.shape[1]
                generated_tokens = outputs.shape[1] - input_tokens
            elif is_molmo:
                processed = processor.process(images=[image], text=question+ " Please answer in one word.")
                inputs = {k: v.to(model.device).unsqueeze(0) for k, v in processed.items()}
                outputs = model(**inputs)
                logits = outputs.logits
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
                generated_token_ids = torch.cat([inputs["input_ids"], next_token_id[:, None]], dim=1)
                answer = processor.tokenizer.decode(generated_token_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                input_tokens = inputs["input_ids"].shape[1]
                generated_tokens = 1
            elif is_phi:
                prompt = question + " Please answer in one word."
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_new_tokens=20)
                response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = response.strip().split()[0]
                input_tokens = inputs["input_ids"].shape[1]
                generated_tokens = outputs.shape[1] - input_tokens
            elif is_smolvlm:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question + " Please answer in one word."}]}]
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_new_tokens=20)
                response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                print(response)
                import re
                match = re.search(r"assistant:\s*(\w+)", response, flags=re.IGNORECASE)
                answer = match.group(1) if match else response.strip().split()[0]
                print(answer)
                input_tokens = inputs["input_ids"].shape[1]
                generated_tokens = outputs.shape[1] - input_tokens
            if is_llama:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question + " Please answer in one word."}]}]
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(image, prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
                generate_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                outputs = model.generate(**generate_inputs, max_new_tokens=20)
                response = processor.decode(outputs[0], skip_special_tokens=True).strip()
                import re
                match = re.search(r"assistant\s*:?[\s\n]*(.*?)(?:\n+user|$)", response, re.IGNORECASE | re.DOTALL)
                if match:
                    answer_chunk = match.group(1).strip()
                    answer = answer_chunk.split()[0] if answer_chunk else ""
                else:
                    answer = response.strip().split()[0]
                input_tokens = inputs["input_ids"].shape[1]
                generated_tokens = outputs.shape[1] - input_tokens
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

        gpu_after = get_gpu_memory_and_util(handle)
        cpu_after = get_cpu_memory_and_util()
        gpu_mem_delta = gpu_after["gpu_mem_used_mb"] - gpu_before["gpu_mem_used_mb"]
        gpu_util_delta = gpu_after["gpu_util_percent"] - gpu_before["gpu_util_percent"]
        cpu_mem_delta = cpu_after["cpu_mem_used_mb"] - cpu_before["cpu_mem_used_mb"]
        cpu_util_delta = cpu_after["cpu_util_percent"] - cpu_before["cpu_util_percent"]
        print(f"[{i}] GPU util: {gpu_util_delta}%, Mem used: Δ{gpu_mem_delta:.2f} MB")
        total_gpu_util += gpu_after["gpu_util_percent"]
        total_gpu_mem += gpu_mem_delta
        total_cpu_util += cpu_after["cpu_util_percent"]
        total_cpu_mem += cpu_mem_delta

        evaluated += 1
        if gt_answer.lower() in answer.lower():
            correct += 1
        print(f"[{i}] Q: {question}")
        print(f"    GT: {gt_answer.lower()}")
        print(f"    →  Predicted: {answer.lower()}")
        print(f"[{i}] Accuracy so far: {correct}/{evaluated} = {correct / evaluated:.2f}")
        # print("GPU Before: ", gpu_before)
        # print("GPU After: ", gpu_after)
        # print("GPU Delta: ", gpu_util_delta)
        # print("GPU Total: ", total_gpu_util)

    total_inference_time = time.time() - start_time
    # current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    accuracy = correct / len(data) if data else 0
    avg_gpu_util = total_gpu_util / len(data)
    avg_gpu_mem = total_gpu_mem / len(data)
    avg_cpu_util = total_cpu_util / len(data)
    avg_cpu_mem = total_cpu_mem / len(data)

    # print(total_gpu_util, avg_gpu_util)
    # print(total_cpu_util, avg_cpu_util)

    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_generated_tokens": total_generated_tokens,
        "ttft_ms": 0,
        "avg_latency_ms": (total_latency / len(data)) * 1000,
        "throughput_tps": total_generated_tokens / total_inference_time,
        "accuracy": accuracy,
        "total_time": total_inference_time,
        "cpu_mem": avg_cpu_mem,
        "cpu_util": avg_cpu_util,
        "gpu_util": avg_gpu_util,
        "gpu_mem": avg_gpu_mem,
        "num_samples": len(data),
    }

# ============ Main ============
def main():
    handle = init_nvml()
    model, processor, load_duration, memory_after_load_mb = load_model(model_name, handle)
    data = load_dataset(dataset_json)
    data = data[:100]
    results = run_inference(model, processor, data, handle)

    metrics = {
        "model_name": model_name,
        "dataset_name": dataset,
        "device": device,
        "model_load_duration_sec": load_duration,  # seconds
        "gpu_load_memory_mb": memory_after_load_mb,  # MB (GPU)
        "avg_cpu_memory_usage_mb": results["cpu_mem"],  # MB (CPU peak)
        "avg_cpu_usage_percent": results["cpu_util"],  # %
        "avg_gpu_usage_percent": results["gpu_util"],  # %
        "avg_gpu_memory_usage_mb": results["gpu_mem"],  # MB (GPU memory used)
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

    csv_path = log_file
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    print("Inference completed. Metrics saved to {csv_path}")

if __name__ == "__main__":
    main()