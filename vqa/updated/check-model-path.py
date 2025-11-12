from transformers import AutoModelForCausalLM, AutoProcessor
from huggingface_hub import hf_hub_download
import os

model_id = "llava-hf/llava-1.5-13b-hf"

# Check all cache-related env vars
print("HF_HOME:", os.environ.get("HF_HOME"))
print("HUGGINGFACE_HUB_CACHE:", os.environ.get("HUGGINGFACE_HUB_CACHE"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))

# Check default cache path
from huggingface_hub.constants import HF_HUB_CACHE
print("HF default cache path:", HF_HUB_CACHE)

# Find one file (config.json) in the local cache
config_path = hf_hub_download(repo_id=model_id, filename="config.json", local_files_only=True)
print("\nModel found at:", config_path)
print("Parent directory:", os.path.dirname(config_path))