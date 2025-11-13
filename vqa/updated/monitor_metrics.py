#!/usr/bin/env python3
"""
Monitor and compare metrics between unified and individual workflows
"""
import csv
import time
from pathlib import Path
from collections import defaultdict

# Threshold for "significant" difference (percentage)
THRESHOLDS = {
    "accuracy": 0.05,  # 5% difference
    "avg_latency_ms": 0.10,  # 10% difference
    "throughput_tps": 0.10,  # 10% difference
    "avg_gpu_memory_usage_mb": 0.15,  # 15% difference
    "gpu_load_memory_mb": 0.15,  # 15% difference
}

def read_csv(file_path):
    """Read CSV and return list of dicts"""
    if not file_path.exists():
        return []

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def extract_task_from_dataset(dataset_path):
    """Extract task name from dataset path"""
    # e.g., .../dataset/crowd_counting -> crowd
    # IMPORTANT: Check longer strings first to avoid false matches
    task_map = [
        ('crowd_counting', 'crowd'),
        ('scene_classification', 'scene'),
        ('traffic_classification', 'traffic'),
        ('gesture_recognition', 'gesture'),
        ('activity_recognition', 'activity'),
        ('object_detection', 'obj_det'),
        ('image_classification', 'img_cls'),
        ('vqa', 'vqa'),  # Check after image_classification
        ('ocr', 'ocr'),
    ]

    for key, value in task_map:
        if key in dataset_path:
            return value
    return None

def compare_metrics(unified_row, individual_row):
    """Compare two metric rows and return differences"""
    diffs = {}

    for metric in ["accuracy", "avg_latency_ms", "throughput_tps",
                   "avg_gpu_memory_usage_mb", "gpu_load_memory_mb"]:
        try:
            unified_val = float(unified_row.get(metric, 0))
            individual_val = float(individual_row.get(metric, 0))

            if individual_val == 0:
                continue

            pct_diff = abs(unified_val - individual_val) / individual_val

            if metric in THRESHOLDS and pct_diff > THRESHOLDS[metric]:
                diffs[metric] = {
                    'unified': unified_val,
                    'individual': individual_val,
                    'pct_diff': pct_diff * 100
                }
        except (ValueError, TypeError):
            continue

    return diffs

def monitor():
    """Main monitoring function"""
    root = Path(__file__).parent
    unified_csv = root / "unified_metrics.csv"

    # Task CSV mapping
    task_csvs = {
        'crowd': root / "combined_metrics_crowd.csv",
        'scene': root / "combined_metrics_scene.csv",
        'ocr': root / "combined_metrics_ocr.csv",
        'vqa': root / "combined_metrics.csv",  # Note: VQA uses combined_metrics.csv
        'traffic': root / "combined_metrics_traffic.csv",
        'gesture': root / "combined_metrics_gesture.csv",
        'activity': root / "combined_metrics_activity.csv",
        'obj_det': root / "combined_metrics_obj_det.csv",
        'img_cls': root / "combined_metrics_img_cls.csv",
    }

    # Read unified metrics
    unified_data = read_csv(unified_csv)

    print(f"📊 Monitoring Metrics Comparison")
    print(f"=" * 80)
    print(f"Unified CSV entries: {len(unified_data)}")

    # Group by model+task
    unified_by_key = {}
    for row in unified_data:
        model = row['model_name']
        task = extract_task_from_dataset(row['dataset_name'])
        if task:
            key = (model, task)
            unified_by_key[key] = row

    # Read individual CSVs and compare
    total_matches = 0
    total_discrepancies = 0
    discrepancy_details = []

    for task, csv_path in task_csvs.items():
        individual_data = read_csv(csv_path)

        # Get latest entry for each model in this task
        individual_by_model = {}
        for row in individual_data:
            model = row['model_name']
            individual_by_model[model] = row  # Latest entry wins

        print(f"\n📁 Task: {task} ({len(individual_by_model)} models)")

        for model, individual_row in individual_by_model.items():
            key = (model, task)

            if key in unified_by_key:
                unified_row = unified_by_key[key]
                diffs = compare_metrics(unified_row, individual_row)

                total_matches += 1

                if diffs:
                    total_discrepancies += 1
                    print(f"  ⚠️  {model} (task: {task})")
                    for metric, diff_info in diffs.items():
                        print(f"      {metric}: Unified={diff_info['unified']:.4f}, "
                              f"Individual={diff_info['individual']:.4f}, "
                              f"Diff={diff_info['pct_diff']:.1f}%")
                    discrepancy_details.append({
                        'model': model,
                        'task': task,
                        'diffs': diffs
                    })
                else:
                    print(f"  ✅ {model} - Metrics match")

    print(f"\n" + "=" * 80)
    print(f"📈 Summary:")
    print(f"  Total matches found: {total_matches}")
    print(f"  Entries with significant discrepancies: {total_discrepancies}")
    print(f"  Match rate: {(total_matches - total_discrepancies) / total_matches * 100:.1f}%")

    return total_matches, total_discrepancies, discrepancy_details

if __name__ == "__main__":
    monitor()
