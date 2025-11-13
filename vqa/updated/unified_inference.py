"""
Unified inference script:
- Model loading (MODEL-first switch, task-specific within)
- Inference execution (MODEL-first switch, task-specific within)
- Dataset loading and evaluation
- Single unified CSV output
"""
import os
import time
import json
import csv
import argparse
import psutil
import torch
from PIL import Image
from pathlib import Path

# Imports for different models
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    LlavaProcessor,
    LlavaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoModel
)
from huggingface_hub import login

# Import task definitions
from task_definitions import TASK_REGISTRY, get_parser, get_evaluator, UNIFIED_LOG_FILE, CSV_COLUMNS

# GPU monitoring
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetName,
)

# ==================== CONFIGURATION ====================
models_directory = str(Path(__file__).parent / "models")
os.environ["HF_HOME"] = models_directory
os.environ["HUGGINGFACE_HUB_CACHE"] = models_directory
os.environ["TRANSFORMERS_CACHE"] = models_directory

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device.startswith("cuda") else torch.float32

# ==================== GPU/CPU UTILS ====================
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

# ==================== MODEL LOADING ====================
def load_model(model_name, task_name, handle):
    """
    MODEL-FIRST switch for loading.
    Within each model, handle task-specific variations if needed.

    Returns: (model, processor, load_duration, memory_used)
    """
    # Login to HuggingFace if token exists
    if os.path.exists("../../hf-token.txt"):
        with open("../../hf-token.txt", "r") as f:
            token = f.read().strip()
        if token:
            login(token=token)

    load_mem_before = get_gpu_memory_and_util(handle)["gpu_mem_used_mb"]
    t0 = time.time()

    mname = model_name.lower()

    try:
        # ============= MOONDREAM =============
        if "moondream" in mname:
            # Same for ALL tasks
            processor = None  # Moondream doesn't use a separate processor
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=models_directory,
                torch_dtype=dtype,
                device_map={"": "cuda:0"} if device.startswith("cuda") else None,
                trust_remote_code=True
            ).to(device)

        # ============= LLAVA =============
        elif "llava-1.5" in mname:
            # LLAVA 1.5 models
            from transformers import LlavaProcessor
            processor = LlavaProcessor.from_pretrained(model_name, cache_dir=models_directory, use_fast=True)
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=models_directory,
                torch_dtype=dtype,
                device_map={"": "cuda:0"} if device.startswith("cuda") else None
            ).to(device)

        elif "llava-v1.6" in mname or "llava-next" in mname:
            # LLAVA 1.6 (Next) models
            from transformers import LlavaNextForConditionalGeneration
            processor = AutoProcessor.from_pretrained(model_name, cache_dir=models_directory, trust_remote_code=True, use_fast=True)
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=models_directory,
                torch_dtype=dtype,
                device_map={"": "cuda:0"} if device.startswith("cuda") else None
            ).to(device)

        # ============= MOLMO =============
        elif "molmo" in mname:
            # Same for ALL tasks
            processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=models_directory,
                trust_remote_code=True,
                use_fast=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=models_directory,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                attn_implementation="eager",
                device_map={"": "cuda:0"}
            ).to("cuda")

        # ============= QWEN =============
        elif "qwen" in mname:
            # Same for ALL tasks
            from transformers import AutoModelForVision2Seq
            processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=models_directory,
                trust_remote_code=True,
                use_fast=True
            )
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                cache_dir=models_directory,
                torch_dtype=dtype,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                device_map={"": "cuda:0"} if device.startswith("cuda") else None
            ).to(device)

        # ============= PHI =============
        elif "phi" in mname:
            # Same for ALL tasks
            processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=models_directory,
                trust_remote_code=True,
                use_fast=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=models_directory,
                torch_dtype=dtype,
                device_map={"": "cuda:0"} if device.startswith("cuda") else None,
                trust_remote_code=True
            ).to(device)

        # ============= LLAMA =============
        elif "llama" in mname and "vision" in mname:
            # Same for ALL tasks
            from transformers import MllamaForConditionalGeneration
            processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=models_directory,
                trust_remote_code=True,
                use_fast=True
            )
            model = MllamaForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=models_directory,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                attn_implementation="eager",
                device_map={"": "cuda:0"}
            ).to("cuda")

        # ============= MINICPM =============
        elif "minicpm" in mname:
            # Same for ALL tasks
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=models_directory
            )
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                cache_dir=models_directory,
                attn_implementation="flash_attention_2"
            ).eval().cuda()
            processor = tokenizer

        else:
            raise ValueError(f"Unsupported model: {model_name}")

    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")

    # Calculate metrics
    load_duration = time.time() - t0
    load_mem_after = get_gpu_memory_and_util(handle)["gpu_mem_used_mb"]
    memory_used = load_mem_after - load_mem_before

    print(f"✅ Model loaded in {load_duration:.2f}s, GPU memory: {memory_used:.2f} MB")

    return model, processor, load_duration, memory_used


# ==================== INFERENCE ====================
def run_inference(model, processor, image, prompt, model_name, task_name):
    """
    MODEL-FIRST switch for inference.
    Within each model, handle task-specific variations if needed.

    Returns: (raw_output, input_tokens, generated_tokens)
    """
    import re
    mname = model_name.lower()

    # ============= MOONDREAM =============
    if "moondream" in mname:
        # Same for ALL tasks - Moondream has unique API
        answer = model.query(image, prompt)["answer"]
        input_tokens = len(prompt.split())
        generated_tokens = len(answer.split())
        return answer, input_tokens, generated_tokens

    # ============= LLAVA =============
    elif "llava" in mname:
        # Same for ALL tasks
        prompt_formatted = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = processor(text=prompt_formatted, images=image, return_tensors="pt").to(device)
        input_tokens = inputs["input_ids"].shape[1]
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Extract answer after ASSISTANT:
        if "ASSISTANT:" in output:
            output = output.split("ASSISTANT:")[-1].strip()
        generated_tokens = generated_ids.shape[1] - input_tokens
        return output, input_tokens, generated_tokens

    # ============= MOLMO =============
    elif "molmo" in mname:
        # Molmo uses generate_from_batch API (not standard generate)
        from transformers import GenerationConfig
        processed = processor.process(images=[image], text=prompt)
        inputs = {k: v.to(device).unsqueeze(0) for k, v in processed.items()}
        input_tokens = inputs["input_ids"].shape[1]

        gen_config = GenerationConfig(max_new_tokens=20, stop_strings="<|endoftext|>")
        output_ids = model.generate_from_batch(inputs, gen_config, tokenizer=processor.tokenizer)

        # Decode only new tokens
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        output = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        generated_tokens = generated_ids.shape[0]
        return output, input_tokens, generated_tokens

    # ============= QWEN =============
    elif "qwen" in mname:
        # Qwen requires special process_vision_info helper
        from qwen_vl_utils import process_vision_info

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
        input_tokens = inputs.input_ids.shape[1]

        outputs = model.generate(**inputs, max_new_tokens=20, min_new_tokens=1)
        generated_tokens = outputs.shape[1] - input_tokens

        # Trim to only generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
        return output, input_tokens, generated_tokens

    # ============= PHI =============
    elif "phi" in mname:
        # Phi uses chat template with messages
        prompt_with_placeholder = f"<|image_1|>\n{prompt}"
        messages = [{"role": "user", "content": prompt_with_placeholder}]

        text_prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text_prompt, [image], return_tensors="pt").to(device)
        input_tokens = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=20,
            do_sample=False,
            use_cache=False,
        )

        # Decode only generated tokens
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        generated_tokens = generated_ids.shape[1]
        return output, input_tokens, generated_tokens

    # ============= LLAMA =============
    elif "llama" in mname and "vision" in mname:
        # Llama uses messages format with special decoding
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image, prompt_text, add_special_tokens=False, return_tensors="pt").to(device)
        input_tokens = inputs["input_ids"].shape[1]

        generate_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        outputs = model.generate(**generate_inputs, max_new_tokens=20)
        generated_tokens = outputs.shape[1] - input_tokens
        response = processor.decode(outputs[0], skip_special_tokens=True).strip()

        # Extract assistant response using regex
        match = re.search(r"assistant\s*:?[\s\n]*(.*?)(?:\n+user|$)", response, re.IGNORECASE | re.DOTALL)
        if match:
            output = match.group(1).strip()
        else:
            output = response.strip()
        return output, input_tokens, generated_tokens

    # ============= MINICPM =============
    elif "minicpm" in mname:
        # MiniCPM passes image in content array, not as separate parameter
        msgs = [{"role": "user", "content": [image, prompt]}]
        output = model.chat(
            image=None,  # Image is in msgs content
            msgs=msgs,
            tokenizer=processor
        )
        # MiniCPM uses word count like Moondream
        input_tokens = len(prompt.split())
        generated_tokens = len(output.split())
        return output, input_tokens, generated_tokens

    else:
        raise ValueError(f"Unsupported model: {model_name}")


# ==================== MAIN EXECUTION ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    args = parser.parse_args()

    # Get task configuration
    if args.task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {args.task_name}")

    task_config = TASK_REGISTRY[args.task_name]

    # Load dataset
    dataset_json = task_config["dataset_root"] / task_config["dataset_json"]
    with open(dataset_json, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = list(data.values())

    # Make image paths absolute and handle special dataset structures
    for rec in data:
        # VQA: convert image_id to image_path
        if "image_id" in rec and "image_path" not in rec:
            img_id = int(rec["image_id"])
            rec["image_path"] = str(task_config["dataset_root"] / "val2014" / f"COCO_val2014_{img_id:012d}.jpg")

        # Normalize image_path to absolute
        if "image_path" in rec and not os.path.isabs(rec["image_path"]):
            rec["image_path"] = str(task_config["dataset_root"] / rec["image_path"])

        # Normalize label field names: VQA uses 'answer', object_detection uses 'categories'
        if "answer" in rec and "label" not in rec:
            rec["label"] = rec["answer"]
        elif "categories" in rec and "label" not in rec:
            rec["label"] = rec["categories"]

    # Limit to first 100 samples (consistent with individual scripts)
    data = data[:100]

    print(f"\n📊 Loaded {len(data)} samples for task '{args.task_name}'")

    # Initialize GPU/CPU monitoring
    handle = init_nvml()

    # Load model
    print(f"\n🔄 Loading model: {args.model_name}")
    model, processor, load_duration, gpu_load_memory = load_model(args.model_name, args.task_name, handle)

    # Run inference on all samples
    prompt_template = task_config["prompt"]
    parser_fn = get_parser(task_config["parser"])
    evaluator_fn = get_evaluator(task_config["evaluator"])

    correct = 0
    total = 0
    latencies = []
    cpu_mem_readings = []
    cpu_util_readings = []
    gpu_util_readings = []
    gpu_mem_readings = []
    total_prompt_tokens = 0
    total_generated_tokens = 0

    # Get baseline GPU/CPU before starting inference loop
    gpu_before = get_gpu_memory_and_util(handle)
    cpu_before = get_cpu_memory_and_util()
    total_start = time.time()

    for i, rec in enumerate(data):
        # Skip image 73 for Qwen model in img_cls task (matching individual files)
        is_qwen = "qwen" in args.model_name.lower()
        if is_qwen and i == 73 and args.task_name == "image_classification":
            print(f"Skipping image [{i}] for Qwen model in img_cls task")
            continue

        print(f"\nRunning inference")
        image_name = os.path.basename(rec['image_path'])

        # Check if image exists (matching individual files)
        if not os.path.exists(rec["image_path"]):
            print(f"Skipping [{i}] - Image not found: {rec['image_path']}")
            continue

        print(f"✅ Using [{i}] - Found image: {image_name}")

        # Load image with SmolVLM special handling (matching individual files)
        is_smolvlm = "smolvlm" in args.model_name.lower()
        if not is_smolvlm:
            image = Image.open(rec["image_path"]).convert("RGB")
        else:
            from transformers.image_utils import load_image
            image = load_image(rec["image_path"]).resize((512, 512), Image.LANCZOS)

        # Format prompt (VQA uses {question} placeholder which already includes suffix)
        # NOTE: Suffixes are ALREADY in task definitions - do NOT add again!
        if "{question}" in prompt_template and "question" in rec:
            prompt = prompt_template.format(question=rec["question"])
        else:
            prompt = prompt_template

        # Run inference with torch.no_grad() and error handling (matching individual files)
        try:
            inf_start = time.time()
            with torch.no_grad():
                raw_output, input_tokens, generated_tokens = run_inference(model, processor, image, prompt, args.model_name, args.task_name)
            latency_ms = (time.time() - inf_start) * 1000
            latencies.append(latency_ms)
        except Exception as e:
            print(f"⚠️  [{i}] Error processing {image_name}: {type(e).__name__}: {str(e)[:100]}")
            print(f"   Skipping image and continuing...")
            continue

        # Get GPU/CPU metrics AFTER inference
        gpu_after = get_gpu_memory_and_util(handle)
        cpu_after = get_cpu_memory_and_util()

        # Calculate deltas and accumulate (matching individual files)
        gpu_mem_delta = gpu_after["gpu_mem_used_mb"] - gpu_before["gpu_mem_used_mb"]
        cpu_mem_delta = cpu_after["cpu_mem_used_mb"] - cpu_before["cpu_mem_used_mb"]

        # Collect metrics (use AFTER values for utilization, DELTA for memory)
        cpu_mem_readings.append(cpu_mem_delta)
        cpu_util_readings.append(cpu_after["cpu_util_percent"])
        gpu_util_readings.append(gpu_after["gpu_util_percent"])
        gpu_mem_readings.append(gpu_mem_delta)

        # Accumulate ACTUAL token counts (matching individual files)
        total_prompt_tokens += input_tokens
        total_generated_tokens += generated_tokens

        # Parse output
        predicted = parser_fn(raw_output)
        ground_truth = rec["label"]

        # Evaluate using task-specific evaluator
        is_correct = evaluator_fn(predicted, ground_truth)
        if is_correct:
            correct += 1
        total += 1

        print(f"[{i}] GPU util: {gpu_after['gpu_util_percent']}%, Mem used: Δ{gpu_mem_delta:.2f} MB")
        print(f"[{i}] GT label: {ground_truth}  →  Pred: {predicted}  |  Raw: {raw_output}")
        print(f"[{i}] Accuracy so far: {correct}/{total} = {correct/total:.2f}")

    total_time = time.time() - total_start

    # Calculate aggregated metrics
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0
    avg_cpu_mem = sum(cpu_mem_readings) / len(cpu_mem_readings) if cpu_mem_readings else 0
    avg_cpu_util = sum(cpu_util_readings) / len(cpu_util_readings) if cpu_util_readings else 0
    avg_gpu_util = sum(gpu_util_readings) / len(gpu_util_readings) if gpu_util_readings else 0
    avg_gpu_mem = sum(gpu_mem_readings) / len(gpu_mem_readings) if gpu_mem_readings else 0

    # Throughput (tokens per second) - using actual generated tokens
    throughput_tps = total_generated_tokens / total_time if total_time > 0 else 0

    accuracy = correct / total if total > 0 else 0

    # Prepare CSV row
    dataset_name = str(task_config["dataset_root"])

    # Get GPU name
    gpu_name = nvmlDeviceGetName(handle) if device.startswith("cuda") else ""

    csv_row = {
        "model_name": args.model_name,
        "dataset_name": dataset_name,
        "device": device,
        "model_load_duration_sec": load_duration,
        "gpu_load_memory_mb": gpu_load_memory,
        "avg_cpu_memory_usage_mb": avg_cpu_mem,
        "avg_cpu_usage_percent": avg_cpu_util,
        "avg_gpu_usage_percent": avg_gpu_util,
        "avg_gpu_memory_usage_mb": avg_gpu_mem,
        "total_prompt_tokens": total_prompt_tokens,
        "total_generated_tokens": total_generated_tokens,
        "ttft_ms": 0,  # Not tracked in this implementation
        "avg_latency_ms": avg_latency_ms,
        "throughput_tps": throughput_tps,
        "accuracy": accuracy,
        "total_time": total_time,
        "num_samples": total,
        "gpu_name": gpu_name
    }

    # Save to unified CSV
    print(f"\n💾 Saving results to {UNIFIED_LOG_FILE}")

    file_exists = os.path.isfile(UNIFIED_LOG_FILE)

    with open(UNIFIED_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_row)

    print(f"Inference completed. Metrics saved to {UNIFIED_LOG_FILE}")
    print(f"\n✅ Final Accuracy: {correct}/{total} = {accuracy:.2%}")

if __name__ == "__main__":
    main()
