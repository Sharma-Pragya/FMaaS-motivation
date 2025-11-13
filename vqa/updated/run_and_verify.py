"""
Orchestrator with VERIFICATION: Runs all model×task combinations
Calls unified_inference.py and COMPARES results against individual CSVs
"""
import subprocess
import sys
import csv
import os
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

# Mapping task names to individual CSV files
TASK_TO_CSV = {
    'crowd': 'combined_metrics_crowd.csv',
    'scene': 'combined_metrics_scene.csv',
    'ocr': 'combined_metrics_ocr.csv',
    'vqa': 'combined_metrics.csv',  # Original VQA CSV
    'traffic': 'combined_metrics_traffic.csv',
    'gesture': 'combined_metrics_gesture.csv',
    'activity': 'combined_metrics_activity.csv',
    'object_detection': 'combined_metrics_obj_det.csv',
    'image_classification': 'combined_metrics_img_cls.csv',
}

def get_latest_result_from_csv(csv_path, model_name):
    """Get the most recent result for a model from individual CSV"""
    if not os.path.exists(csv_path):
        return None

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row['model_name'] == model_name]

    if not rows:
        return None

    # Return the last matching row (most recent)
    return rows[-1]

def get_latest_unified_result(model_name, task_name):
    """Get the most recent unified result for model+task"""
    unified_csv = "unified_metrics.csv"
    if not os.path.exists(unified_csv):
        return None

    with open(unified_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Find matching row (most recent)
    for row in reversed(rows):
        if row.get('model_name') == model_name:
            # Check if dataset matches task
            if task_name in row.get('dataset_name', ''):
                return row

    return None

def compare_results(unified, individual, model, task):
    """Compare unified result against individual CSV result"""
    if not individual:
        print(f"  ⚠️  No individual result found for comparison")
        return True  # Can't compare, assume ok

    if not unified:
        print(f"  ❌ No unified result found!")
        return False

    # Key metrics to compare
    metrics_to_check = ['accuracy', 'num_samples']

    print(f"\n  📊 COMPARISON:")
    mismatches = []

    for metric in metrics_to_check:
        ind_val = individual.get(metric, '')
        uni_val = unified.get(metric, '')

        # Convert to float for comparison
        try:
            ind_float = float(ind_val) if ind_val else 0
            uni_float = float(uni_val) if uni_val else 0

            # Allow small floating point differences (0.01 tolerance)
            if abs(ind_float - uni_float) > 0.01:
                print(f"    ❌ {metric}: Individual={ind_val} vs Unified={uni_val}")
                mismatches.append(metric)
            else:
                print(f"    ✅ {metric}: {uni_val} (matches)")
        except ValueError:
            # For non-numeric, do exact match
            if ind_val != uni_val:
                print(f"    ❌ {metric}: Individual={ind_val} vs Unified={uni_val}")
                mismatches.append(metric)
            else:
                print(f"    ✅ {metric}: {uni_val} (matches)")

    if mismatches:
        print(f"\n  ❌ MISMATCH DETECTED in: {', '.join(mismatches)}")
        return False
    else:
        print(f"\n  ✅ ALL METRICS MATCH!")
        return True

def main():
    print("=" * 80)
    print(f"Running {len(MODELS)} models × {len(TASKS)} tasks = {len(MODELS) * len(TASKS)} experiments")
    print(f"WITH VERIFICATION against individual CSVs")
    print("=" * 80)

    total_experiments = len(MODELS) * len(TASKS)
    completed = 0
    failed = 0
    mismatched = 0

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
                print(f"\n✅ Inference Completed: {model} on {task}")

                # NOW VERIFY RESULTS
                print(f"\n🔍 VERIFYING RESULTS...")
                individual_csv = TASK_TO_CSV.get(task)
                if individual_csv:
                    individual_result = get_latest_result_from_csv(individual_csv, model)
                    unified_result = get_latest_unified_result(model, task)

                    if not compare_results(unified_result, individual_result, model, task):
                        mismatched += 1
                        print(f"\n⚠️  MISMATCH DETECTED - PAUSING FOR REVIEW")
                        response = input("Continue anyway? (y/n): ")
                        if response.lower() != 'y':
                            print("STOPPING EXECUTION")
                            break
                else:
                    print(f"  ⚠️  No individual CSV mapping for task: {task}")

            except subprocess.CalledProcessError as e:
                failed += 1
                print(f"\n❌ Failed: {model} on {task}")
                print(f"Error: {e}")
                continue
            except Exception as e:
                failed += 1
                print(f"\n❌ Unexpected error: {model} on {task}")
                print(f"Error: {e}")
                continue

    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Total: {total_experiments} | Successful: {completed - failed} | Failed: {failed} | Mismatched: {mismatched}")
    print(f"Results saved to: unified_metrics.csv")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
