devices = {
  'A100': {
    'type': 'A100'
  },
  'A16': {
    'type': 'A16'
  },
  'A6000': {
    'type': 'A6000'
  }
}

models = [
  'Qwen/Qwen2.5-VL-3B-Instruct',
  'Qwen/Qwen2.5-VL-7B-Instruct',
  'allenai/Molmo-7B-D-0924',
  'llava-hf/llava-1.5-13b-hf',
  'llava-hf/llava-1.5-7b-hf',
  'llava-hf/llava-v1.6-vicuna-13b-hf',
  'meta-llama/Llama-3.2-11B-Vision-Instruct',
  'microsoft/Phi-3.5-vision-instruct',
  'openbmb/MiniCPM-V-2_6',
  'vikhyatk/moondream2'
]

# Task SLOs (Lmax in ms), derived as ceil(p95 of best per-(m,d) latencies for that task)
tasks = {
  'dataset/val2014': 1120
}