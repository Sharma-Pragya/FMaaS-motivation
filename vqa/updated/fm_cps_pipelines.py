import torch
from transformers import AutoModelForCausalLM

device = 'cuda'

def gpu_mem_mb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def measure_model_load(model_func, name):
    torch.cuda.reset_peak_memory_stats()
    model_func()
    mem = gpu_mem_mb()
    print(f"{name:<40}: {mem:.2f} MB")
    torch.cuda.empty_cache()

model_name = 'vikhyatk/moondream2'

measure_model_load(
    lambda: AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device),
    "Moondream GPU Memory"
)
