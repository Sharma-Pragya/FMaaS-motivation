from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_ROOT = ROOT / "models"

models_directory = str(MODELS_ROOT.resolve())

# DATASET_ROOT = ROOT / "dataset/vqa"
# dataset_directory = str(DATASET_ROOT.resolve())
# dataset = str((DATASET_ROOT / "val2014").resolve())
# dataset_json = str((DATASET_ROOT / "val.json").resolve())
# log_file = "combined_metrics.csv"

# OCR Dataset
# DATASET_ROOT = ROOT / "dataset/ocr"
# dataset_directory = str(DATASET_ROOT.resolve())
# dataset = str(DATASET_ROOT.resolve())
# dataset_json = str((DATASET_ROOT / "labels.json").resolve())
# log_file = "combined_metrics_ocr.csv"

# Object Detection Dataset
# DATASET_ROOT = ROOT / "dataset/object_detection"
# dataset_directory = str(DATASET_ROOT.resolve())
# dataset = str(DATASET_ROOT.resolve())
# dataset_json = str((DATASET_ROOT / "annotations.json").resolve())
# log_file = "combined_metrics_obj_det.csv"

# Image Classification Dataset
# DATASET_ROOT = ROOT / "dataset/image_classification"
# dataset_directory = str(DATASET_ROOT.resolve())
# dataset = str(DATASET_ROOT.resolve())
# dataset_json = str((DATASET_ROOT / "labels.json").resolve())
# log_file = "combined_metrics_img_cls.csv"

# Traffic Sign Classification Dataset (GTSRB)
# DATASET_ROOT = ROOT / "dataset/traffic_classification"
# dataset_directory = str(DATASET_ROOT.resolve())
# dataset = str(DATASET_ROOT.resolve())
# dataset_json = str((DATASET_ROOT / "labels.json").resolve())
# log_file = "combined_metrics_traffic.csv"

# Gesture Recognition Dataset (HaGRID)
# DATASET_ROOT = ROOT / "dataset/gesture_recognition"
# dataset_directory = str(DATASET_ROOT.resolve())
# dataset = str(DATASET_ROOT.resolve())
# dataset_json = str((DATASET_ROOT / "labels.json").resolve())
# log_file = "combined_metrics_gesture.csv"

# Activity Recognition Dataset (Human Action Recognition)
DATASET_ROOT = ROOT / "dataset/activity_recognition"
dataset_directory = str(DATASET_ROOT.resolve())
dataset = str(DATASET_ROOT.resolve())
dataset_json = str((DATASET_ROOT / "labels.json").resolve())
log_file = "combined_metrics_activity.csv"

# Crowd Counting Dataset (VisDrone/DroneCrowd)
# DATASET_ROOT = ROOT / "dataset/crowd_counting"
# dataset_directory = str(DATASET_ROOT.resolve())
# dataset = str(DATASET_ROOT.resolve())
# dataset_json = str((DATASET_ROOT / "labels.json").resolve())
# log_file = "combined_metrics_crowd.csv"

# Scene Classification Dataset (SUN397)
# DATASET_ROOT = ROOT / "dataset/scene_classification"
# dataset_directory = str(DATASET_ROOT.resolve())
# dataset = str(DATASET_ROOT.resolve())
# dataset_json = str((DATASET_ROOT / "labels.json").resolve())
# log_file = "combined_metrics_scene.csv"

hf_token_path = "../../hf-token.txt"

# model_name="google/gemma-3-4b-it"
# model_name="vikhyatk/moondream2"
# model_name="llava-hf/llava-1.5-7b-hf"
# model_name="llava-hf/llava-1.5-13b-hf"
# model_name="llava-hf/llava-v1.6-vicuna-13b-hf"
# model_name="Qwen/Qwen2.5-VL-3B-Instruct" / 7B
# model_name="microsoft/Phi-3.5-vision-instruct" #Requires transofrmers 46.0
# HuggingFaceTB/SmolVLM-256M-Instruct #NOT WORKING
# allenai/Molmo-7B-D-0924
# meta-llama/Llama-3.2-11B-Vision-Instruct
# openbmb/MiniCPM-V-2_6
model_name = ['vikhyatk/moondream2', 
                "llava-hf/llava-1.5-7b-hf",
                "llava-hf/llava-1.5-13b-hf",
                "llava-hf/llava-v1.6-vicuna-13b-hf",
                "Qwen/Qwen2.5-VL-3B-Instruct",
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "microsoft/Phi-3.5-vision-instruct",
                "allenai/Molmo-7B-D-0924",
                "meta-llama/Llama-3.2-11B-Vision-Instruct",
                "openbmb/MiniCPM-V-2_6"]