can_serve = {
  ('Qwen/Qwen2.5-VL-3B-Instruct', 'dataset/val2014'): 1,
  ('Qwen/Qwen2.5-VL-7B-Instruct', 'dataset/val2014'): 1,
  ('allenai/Molmo-7B-D-0924', 'dataset/val2014'): 1,
  ('llava-hf/llava-1.5-13b-hf', 'dataset/val2014'): 1,
  ('llava-hf/llava-1.5-7b-hf', 'dataset/val2014'): 1,
  ('llava-hf/llava-v1.6-vicuna-13b-hf', 'dataset/val2014'): 1,
  ('meta-llama/Llama-3.2-11B-Vision-Instruct', 'dataset/val2014'): 1,
  ('microsoft/Phi-3.5-vision-instruct', 'dataset/val2014'): 1,
  ('openbmb/MiniCPM-V-2_6', 'dataset/val2014'): 1,
  ('vikhyatk/moondream2', 'dataset/val2014'): 1
}

# Latency per (task, model, device) in ms (best/min observed per CSV)
latency_tmd = {
  ('dataset/val2014', 'Qwen/Qwen2.5-VL-3B-Instruct', 'A100'): 245.462,
  ('dataset/val2014', 'Qwen/Qwen2.5-VL-3B-Instruct', 'A6000'): 136.9636082649231,
  ('dataset/val2014', 'Qwen/Qwen2.5-VL-7B-Instruct', 'A100'): 178.765,
  ('dataset/val2014', 'Qwen/Qwen2.5-VL-7B-Instruct', 'A6000'): 178.29195976257324,
  ('dataset/val2014', 'allenai/Molmo-7B-D-0924', 'A100'): 924.253,
  ('dataset/val2014', 'allenai/Molmo-7B-D-0924', 'A6000'): 927.6322436332704,
  ('dataset/val2014', 'llava-hf/llava-1.5-13b-hf', 'A100'): 245.462,
  ('dataset/val2014', 'llava-hf/llava-1.5-13b-hf', 'A6000'): 278.7865138053894,
  ('dataset/val2014', 'llava-hf/llava-1.5-7b-hf', 'A100'): 183.687,
  ('dataset/val2014', 'llava-hf/llava-1.5-7b-hf', 'A16'): 1257.132,
  ('dataset/val2014', 'llava-hf/llava-1.5-7b-hf', 'A6000'): 161.4958667755127,
  ('dataset/val2014', 'llava-hf/llava-v1.6-vicuna-13b-hf', 'A100'): 803.34,
  ('dataset/val2014', 'llava-hf/llava-v1.6-vicuna-13b-hf', 'A6000'): 831.543,
  ('dataset/val2014', 'meta-llama/Llama-3.2-11B-Vision-Instruct', 'A100'): 187.6,
  ('dataset/val2014', 'meta-llama/Llama-3.2-11B-Vision-Instruct', 'A6000'): 114.5,
  ('dataset/val2014', 'microsoft/Phi-3.5-vision-instruct', 'A6000'): 201.49,
  ('dataset/val2014', 'openbmb/MiniCPM-V-2_6', 'A100'): 271.368,
  ('dataset/val2014', 'openbmb/MiniCPM-V-2_6', 'A6000'): 163.091,
  ('dataset/val2014', 'vikhyatk/moondream2', 'A6000'): 146.532
}

# Accuracy per (task, model) — from A6000 CSV (device-agnostic)
accuracy_tm = {
  ('dataset/val2014', 'Qwen/Qwen2.5-VL-3B-Instruct'): 0.7,
  ('dataset/val2014', 'Qwen/Qwen2.5-VL-7B-Instruct'): 0.69,
  ('dataset/val2014', 'allenai/Molmo-7B-D-0924'): 0.52,
  ('dataset/val2014', 'llava-hf/llava-1.5-13b-hf'): 0.61,
  ('dataset/val2014', 'llava-hf/llava-1.5-7b-hf'): 0.66,
  ('dataset/val2014', 'llava-hf/llava-v1.6-vicuna-13b-hf'): 0.58,
  ('dataset/val2014', 'meta-llama/Llama-3.2-11B-Vision-Instruct'): 0.74,
  ('dataset/val2014', 'microsoft/Phi-3.5-vision-instruct'): 0.64,
  ('dataset/val2014', 'openbmb/MiniCPM-V-2_6'): 0.7,
  ('dataset/val2014', 'vikhyatk/moondream2'): 0.61
}

# Max observed VRAM (MB) per model — from A6000 CSV
model_memory = {
  'Qwen/Qwen2.5-VL-3B-Instruct': 8138.9375,
  'Qwen/Qwen2.5-VL-7B-Instruct': 18206.9375,
  'allenai/Molmo-7B-D-0924': 31010.9375,
  'llava-hf/llava-1.5-13b-hf': 25852.9375,
  'llava-hf/llava-1.5-7b-hf': 13752.9375,
  'llava-hf/llava-v1.6-vicuna-13b-hf': 4018.9375,
  'meta-llama/Llama-3.2-11B-Vision-Instruct': 25852.9375,
  'microsoft/Phi-3.5-vision-instruct': 7772.9375,
  'openbmb/MiniCPM-V-2_6': 12214.9375,
  'vikhyatk/moondream2': 4018.9375
}

# Device VRAM caps (MB). Adjust if your A100 is 40GB.
memory_device = {
  'A6000': 48000.0,
  'A100': 80000.0,
  'A16': 16000.0
}

# Endpoint capacity (req/s) per (model, device): 1000 / worst best-latency across tasks
throughput_capacity = {
  ('Qwen/Qwen2.5-VL-3B-Instruct', 'A100'): 4.074421,
  ('Qwen/Qwen2.5-VL-3B-Instruct', 'A6000'): 7.301448,
  ('Qwen/Qwen2.5-VL-7B-Instruct', 'A100'): 5.594779,
  ('Qwen/Qwen2.5-VL-7B-Instruct', 'A6000'): 5.607168,
  ('allenai/Molmo-7B-D-0924', 'A100'): 1.08198,
  ('allenai/Molmo-7B-D-0924', 'A6000'): 1.077951,
  ('llava-hf/llava-1.5-13b-hf', 'A100'): 4.074421,
  ('llava-hf/llava-1.5-13b-hf', 'A6000'): 3.587004,
  ('llava-hf/llava-1.5-7b-hf', 'A100'): 5.445629,
  ('llava-hf/llava-1.5-7b-hf', 'A16'): 0.795756,
  ('llava-hf/llava-1.5-7b-hf', 'A6000'): 6.192048,
  ('llava-hf/llava-v1.6-vicuna-13b-hf', 'A100'): 1.24535,
  ('llava-hf/llava-v1.6-vicuna-13b-hf', 'A6000'): 1.202225,
  ('meta-llama/Llama-3.2-11B-Vision-Instruct', 'A100'): 5.331122,
  ('meta-llama/Llama-3.2-11B-Vision-Instruct', 'A6000'): 8.735632,
  ('microsoft/Phi-3.5-vision-instruct', 'A6000'): 4.964558,
  ('openbmb/MiniCPM-V-2_6', 'A100'): 3.685162,
  ('openbmb/MiniCPM-V-2_6', 'A6000'): 6.132679,
  ('vikhyatk/moondream2', 'A6000'): 6.821835
}