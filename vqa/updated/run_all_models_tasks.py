"""
Orchestrator: Runs all model×task combinations
Calls unified_inference.py for each combination
"""
import subprocess
import sys
from pathlib import Path

# Define all models to evaluate
MODELS = [
    'vikhyatk/moondream2',
    'llava-hf/llava-1.5-7b-hf',
    'llava-hf/llava-1.5-13b-hf',
    'llava-hf/llava-v1.6-vicuna-13b-hf',
    'Qwen/Qwen2.5-VL-3B-Instruct',
    'Qwen/Qwen2.5-VL-7B-Instruct',
    'microsoft/Phi-3.5-vision-instruct',
    'allenai/Molmo-7B-D-0924',
    'meta-llama/Llama-3.2-11B-Vision-Instruct',
    'openbmb/MiniCPM-V-2_6'
]

# Define all tasks to evaluate
TASKS = [
    'crowd',
    'scene',
    'ocr',
    'vqa',
    'traffic',
    'gesture',
    'activity',
    'object_detection',
    'image_classification',
]

def main():
    print("=" * 80)
    print(f"Running {len(MODELS)} models × {len(TASKS)} tasks = {len(MODELS) * len(TASKS)} experiments")
    print("=" * 80)

    total_experiments = len(MODELS) * len(TASKS)
    completed = 0
    failed = 0

    for model in MODELS:
        for task in TASKS:
            completed += 1
            print(f"\n{'='*80}")
            print(f"Experiment {completed}/{total_experiments}")
            print(f"Model: {model}")
            print(f"Task: {task}")
            print(f"{'='*80}\n")

            # Call unified inference script
            cmd = [
                sys.executable,
                "unified_inference.py",
                "--model_name", model,
                "--task_name", task
            ]

            try:
                result = subprocess.run(cmd, check=True, capture_output=False, text=True)
                print(f"\n✅ Completed: {model} on {task}")
            except subprocess.CalledProcessError as e:
                failed += 1
                print(f"\n❌ Failed: {model} on {task}")
                print(f"Error: {e}")
                # Continue with next combination instead of stopping
                continue
            except Exception as e:
                failed += 1
                print(f"\n❌ Unexpected error: {model} on {task}")
                print(f"Error: {e}")
                continue

    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Total: {total_experiments} | Successful: {completed - failed} | Failed: {failed}")
    print(f"Results saved to: unified_metrics.csv")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
