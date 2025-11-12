# gpu_run_model_inference_activity.py
import os
import re
import csv
import json
import time
import psutil
import tracemalloc
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from huggingface_hub import login
from transformers.image_utils import load_image

from config import model_name as cfg_model_name, dataset, dataset_json, models_directory, log_file as cfg_log_file

from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetName,
)

import argparse
import builtins
from typing import List as _List  # workaround like your original
builtins.List = _List

os.environ["HF_HOME"] = models_directory
os.environ["HUGGINGFACE_HUB_CACHE"] = models_directory
os.environ["TRANSFORMERS_CACHE"] = models_directory

# ----------------------------- CLI -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--log_file", type=str, default=None)
    return p.parse_args()

args = parse_args()
model_name = args.model_name if args.model_name else cfg_model_name
log_file = args.log_file if args.log_file else cfg_log_file

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device.startswith("cuda") else torch.float32
print(device)


# ----------------------- GPU/CPU Telemetry ----------------------
def init_nvml():
    nvmlInit()
    return nvmlDeviceGetHandleByIndex(0)

def get_gpu_memory_and_util(handle):
    mem = nvmlDeviceGetMemoryInfo(handle)
    util = nvmlDeviceGetUtilizationRates(handle)
    return {
        "gpu_mem_used_mb": mem.used / (1024**2),
        "gpu_util_percent": util.gpu,
    }

def get_cpu_memory_and_util():
    proc = psutil.Process()
    return {
        "cpu_mem_used_mb": proc.memory_info().rss / (1024**2),
        "cpu_util_percent": psutil.cpu_percent(),
    }


# ---------------------------- IO -------------------------------
def ensure_path(p: str) -> str:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Path not found: {p}")
    return p

def load_dataset(json_path):
    import os, json
    from config import dataset  # absolute .../dataset/activity_recognition

    with open(json_path, "r") as f:
        raw = json.load(f)
    data = list(raw.values()) if isinstance(raw, dict) else raw

    for rec in data:
        p = rec["image_path"]
        if not os.path.isabs(p):
            rec["image_path"] = os.path.join(dataset, p)  # joins .../activity_recognition + images/00000.jpg
    return data

def open_image_rgb(path: str) -> Image.Image:
    path = ensure_path(path)
    return Image.open(path).convert("RGB")


# ---------------------- Human Activity Recognition prompt/parse ----------------------
PROMPT_TEXT = (
    "What is the person doing in this image? "
    "Answer with the activity name only, without any explanation. "
    "Choose from: calling, clapping, cycling, dancing, drinking, eating, fighting, "
    "hugging, laughing, listening to music, running, sitting, sleeping, texting, or using laptop."
)

def parse_label(text: str) -> str:
    """Parse the model's answer to extract a human activity label."""
    if not text:
        return ""
    # Strip and lowercase
    text = text.strip().lower()
    # Remove common punctuation
    text = text.strip('.,!?;:"\'"')
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


# -------------------------- Model Load --------------------------
def load_model(model_id: str, handle):
    # reuse the same HF token flow your original uses
    if os.path.exists("../../hf-token.txt"):
        with open("../../hf-token.txt", "r") as f:
            token = f.read().strip()
        if token:
            login(token=token)

    load_mem_before = get_gpu_memory_and_util(handle)["gpu_mem_used_mb"]
    t0 = time.time()

    # Branches aligned with common VLM IDs you likely already use
    mname = str(model_name).lower()

    processor = None
    model = None

    try:
        if "llava-1.5" in mname:
            from transformers import LlavaForConditionalGeneration, LlavaProcessor
            processor = LlavaProcessor.from_pretrained(model_id, cache_dir=models_directory, use_fast=True)
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id, cache_dir=models_directory, torch_dtype=dtype, device_map={"": "cuda:0"} if device.startswith("cuda") else None
            ).to(device)

        elif "llava-v1.6" in mname or "llava-next" in mname:
            from transformers import LlavaNextForConditionalGeneration, AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, use_fast=True)
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id, cache_dir=models_directory, torch_dtype=dtype, device_map={"": "cuda:0"} if device.startswith("cuda") else None
            ).to(device)

        elif "qwen" in mname:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            processor = AutoProcessor.from_pretrained(
                model_id, cache_dir=models_directory, trust_remote_code=True, use_fast=True
            )
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                cache_dir=models_directory,
                torch_dtype=dtype,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                device_map={"": "cuda:0"} if device.startswith("cuda") else None,
            ).to(device)

        elif "phi" in mname:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, cache_dir=models_directory, torch_dtype=dtype, device_map={"": "cuda:0"} if device.startswith("cuda") else None, trust_remote_code=True
            ).to(device)

        elif "smolvlm" in mname:
            # both typically use vision2seq style
            from transformers import AutoModelForVision2Seq
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, use_fast=False)
            model = AutoModelForVision2Seq.from_pretrained(
                model_id, cache_dir=models_directory, torch_dtype=dtype, device_map={"": "cuda:0"} if device.startswith("cuda") else None, trust_remote_code=True
            ).to(device)

        elif "molmo" in mname.lower():
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager", device_map={"": "cuda:0"}).to("cuda")

        elif "llama" in mname:
            from transformers import MllamaForConditionalGeneration, AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, use_fast=True)
            model = MllamaForConditionalGeneration.from_pretrained(model_id, cache_dir=models_directory, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="eager", device_map={"": "cuda:0"}).to("cuda")

        elif "minicpm" in mname:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=models_directory)
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=models_directory, attn_implementation="flash_attention_2").eval().cuda()
            processor = tokenizer

        elif "moondream" in mname:
            # some Moondream builds ship their own helpers; we keep an HF path
            processor = None #AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, cache_dir=models_directory, torch_dtype=dtype, device_map={"": "cuda:0"} if device.startswith("cuda") else None, trust_remote_code=True
            ).to(device)

        else:
            # generic VLM fallback
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, cache_dir=models_directory, torch_dtype=dtype, device_map={"": "cuda:0"} if device.startswith("cuda") else None, trust_remote_code=True
            ).to(device)

    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_id}: {e}")

    t1 = time.time()
    load_mem_after = get_gpu_memory_and_util(handle)["gpu_mem_used_mb"]
    return model, processor, (t1 - t0), load_mem_after

# ------------------------- Runner Core --------------------------
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
    is_minicpm = "minicpm" in model_name.lower()

    for i, item in enumerate(data):
        # Skip image 73 for Qwen model if needed (same as img_cls)
        if is_qwen and i == 73:
            print(f"Skipping image [{i}] for Qwen model")
            continue

        print("\nRunning inference")
        question = PROMPT_TEXT
        gt_label = item["label"].lower()  # Ground truth label
        image_path = item["image_path"]
        image_name = os.path.basename(image_path)

        if not os.path.exists(image_path):
            print(f"Skipping [{i}] - Image not found: {image_path, image_name}")
            continue
        print(f"✅ Using [{i}] - Found image: {image_name}")

        if not is_smolvlm:
            image = Image.open(image_path).convert("RGB")
        else:
            image = load_image(image_path).resize((512, 512), Image.LANCZOS)

        start = time.time()

        try:
            with torch.no_grad():
                if is_moondream:
                    answer = model.query(image, question + " Please answer in a few words.")["answer"]
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
                                {"type": "text", "text": question+" Please answer in a few words."},
                            ],}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",).to("cuda")
                    outputs = model.generate(**inputs, max_new_tokens=20, min_new_tokens=1)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(inputs.input_ids, outputs)]
                    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
                    answer = response
                    input_tokens = inputs.input_ids.shape[1]
                    generated_tokens = outputs.shape[1] - input_tokens
                elif is_molmo:
                    from transformers import GenerationConfig
                    processed = processor.process(images=[image], text=question+ " Please answer in a few words.")
                    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in processed.items()}

                    # Molmo uses generate_from_batch instead of generate
                    gen_config = GenerationConfig(max_new_tokens=20, stop_strings="<|endoftext|>")
                    output = model.generate_from_batch(
                        inputs,
                        gen_config,
                        tokenizer=processor.tokenizer
                    )

                    # output is a tensor, decode it
                    generated_ids = output[0, inputs["input_ids"].shape[1]:]
                    answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    input_tokens = inputs["input_ids"].shape[1]
                    generated_tokens = generated_ids.shape[0]

                elif is_phi:
                    # Add image placeholder to match expected format
                    prompt = f"<|image_1|>\n{question} Please answer in a few words."
                    messages = [{"role": "user", "content": prompt}]

                    # Create chat-style prompt with special tokens
                    text_prompt = processor.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    # Tokenize + process image(s) together
                    inputs = processor(text_prompt, [image], return_tensors="pt").to(model.device)

                    # Run inference
                    outputs = model.generate(
                        **inputs,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        max_new_tokens=20,
                        do_sample=False,
                        use_cache=False,
                    )

                    # Strip off prompt tokens
                    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
                    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    answer = response

                    input_tokens = inputs["input_ids"].shape[1]
                    generated_tokens = generated_ids.shape[1]
                elif is_smolvlm:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": question + " Please answer in a few words."}
                            ]
                        }
                    ]
                    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)

                    outputs = model.generate(**inputs, max_new_tokens=20)
                    response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

                    import re
                    match = re.search(r"assistant\s*:?[\s\n]*(.*?)(?:\n|$)", response, flags=re.IGNORECASE | re.DOTALL)
                    answer = match.group(1).strip() if match else response.strip()

                    input_tokens = inputs["input_ids"].shape[1]
                    generated_tokens = outputs.shape[1] - input_tokens
                elif is_llama:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": question + " Please answer in a few words."}]}]
                    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(image, prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
                    generate_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                    outputs = model.generate(**generate_inputs, max_new_tokens=20)
                    response = processor.decode(outputs[0], skip_special_tokens=True).strip()
                    import re
                    match = re.search(r"assistant\s*:?[\s\n]*(.*?)(?:\n+user|$)", response, re.IGNORECASE | re.DOTALL)
                    if match:
                        answer = match.group(1).strip()
                    else:
                        answer = response.strip()
                    input_tokens = inputs["input_ids"].shape[1]
                    generated_tokens = outputs.shape[1] - input_tokens
                elif is_minicpm:
                    question_prompt = question + " Please answer in a few words."
                    msgs = [{"role": "user", "content": [image, question_prompt]}]

                    response = model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=processor,  # tokenizer = processor in this case
                    )

                    answer = response.strip()
                    input_tokens = len(question_prompt.split())
                    generated_tokens = len(answer.split())
                else:
                    prompt = f"USER: <image>\n{question}\nPlease answer in a few words.\nASSISTANT:"
                    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
                    input_tokens = inputs["input_ids"].shape[1]
                    outputs = model.generate(**inputs, max_new_tokens=20)
                    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                    answer = response.split("ASSISTANT:")[-1].strip()
                    generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        except Exception as e:
            print(f"⚠️  [{i}] Error processing {image_name}: {type(e).__name__}: {str(e)[:100]}")
            print(f"   Skipping image and continuing...")
            continue

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
        pred_label = parse_label(answer)
        # Check if prediction matches ground truth
        # Use substring matching for traffic signs
        is_correct = (gt_label in pred_label) or (pred_label in gt_label)
        if is_correct:
            correct += 1
        print(f"[{i}] GT label: {gt_label}  →  Pred: {pred_label}  |  Raw: {answer}")
        print(f"[{i}] Accuracy so far: {correct}/{evaluated} = {correct / evaluated:.2f}")

    total_inference_time = time.time() - start_time
    # current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    accuracy = (correct / evaluated) if evaluated else 0
    avg_gpu_util = total_gpu_util / len(data)
    avg_gpu_mem = total_gpu_mem / len(data)
    avg_cpu_util = total_cpu_util / len(data)
    avg_cpu_mem = total_cpu_mem / len(data)

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



# ----------------------------- Main -----------------------------
def main():
    handle = init_nvml()

    # You likely pass a HF model id via config.model_name or CLI
    model_id = str(model_name)

    model, processor, load_duration, memory_after_load_mb = load_model(model_id, handle)

    data = load_dataset(dataset_json)

    # your original script takes the first 100 samples; keep that here
    data = data[:100]

    results = run_inference(model, processor, data, handle)

    metrics = {
        "model_name": model_name,
        "dataset_name": dataset,
        "device": device,
        "model_load_duration_sec": load_duration,
        "gpu_load_memory_mb": memory_after_load_mb,
        "avg_cpu_memory_usage_mb": results["cpu_mem"],
        "avg_cpu_usage_percent": results["cpu_util"],
        "avg_gpu_usage_percent": results["gpu_util"],
        "avg_gpu_memory_usage_mb": results["gpu_mem"],
        "total_prompt_tokens": results["total_prompt_tokens"],
        "total_generated_tokens": results["total_generated_tokens"],
        "ttft_ms": 0,
        "avg_latency_ms": results["avg_latency_ms"],
        "throughput_tps": results["throughput_tps"],
        "accuracy": results["accuracy"],
        "total_time": results["total_time"],
        "num_samples": results["num_samples"],
    }

    csv_path = log_file
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    print(f"Inference completed. Metrics saved to {csv_path}")

if __name__ == "__main__":
    main()
