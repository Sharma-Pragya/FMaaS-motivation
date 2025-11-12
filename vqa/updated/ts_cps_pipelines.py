import torch
from ultralytics import YOLO
from torchvision.models import mobilenet_v2, efficientnet_b0, resnet50, video
from transformers import AutoModelForCausalLM

device = 'cuda'

def gpu_mem_mb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)

def measure_model_load(model_func, name):
    torch.cuda.reset_peak_memory_stats()
    model_func()
    mem = gpu_mem_mb()
    print(f"{name:<40}: {mem:.2f} MB")
    torch.cuda.empty_cache()

pipelines = {
    "Pipeline A (Tracking)": [
        (lambda: YOLO('yolov8x.pt'), "YOLOv8x (Detector)"),
        (mobilenet_v2, "DeepSORT CNN (MobileNet)"),
        (efficientnet_b0, "Localization CNN (EfficientNet)")],

    "Pipeline B (Anomaly Detection)": [
        (lambda: YOLO('yolov8x.pt'), "YOLOv8x (Detector)"),
        (resnet50, "CNN Anomaly (ResNet50)"),
        (mobilenet_v2, "Classification (MobileNet)")],

    "Pipeline C (Event Analysis)": [
        (lambda: YOLO('yolov8x.pt'), "YOLOv8x (Detector)"),
        (video.r3d_18, "Action Recognition (R3D-18)"),
        (resnet50, "Event Summary (ResNet50)"),
        (lambda: AutoModelForCausalLM.from_pretrained('distilgpt2'), "Reporting (distilGPT2)")],

    "Pipeline D (Incident Detection)": [
        (lambda: YOLO('yolov8x.pt'), "YOLOv8x (Detector)"),
        (resnet50, "Incident Recognition (ResNet50)"),
        (lambda: AutoModelForCausalLM.from_pretrained('distilgpt2'), "Alert Generation (distilGPT2)")],

    "Pipeline E (Traffic Monitoring)": [
        (lambda: YOLO('yolov8x.pt'), "YOLOv8x (Detector)"),
        (mobilenet_v2, "Vehicle Counting CNN"),
        (resnet50, "Traffic Pattern CNN-LSTM")],

    "Pipeline F (Intrusion Detection)": [
        (lambda: YOLO('yolov8x.pt'), "YOLOv8x (Detector)"),
        (mobilenet_v2, "Re-ID CNN (MobileNet)"),
        (mobilenet_v2, "Intrusion Decision CNN"),
        (lambda: AutoModelForCausalLM.from_pretrained('distilgpt2'), "Security Alert (distilGPT2)")],

    "Pipeline G (Equipment Monitoring)": [
        (lambda: YOLO('yolov8x.pt'), "YOLOv8x (Detector)"),
        (resnet50, "Equipment State CNN"),
        (resnet50, "Anomaly CNN"),
        (lambda: AutoModelForCausalLM.from_pretrained('distilgpt2'), "Equipment Report (distilGPT2)")],

    "Pipeline H (Environmental Safety)": [
        (lambda: YOLO('yolov8x.pt'), "YOLOv8x (Detector)"),
        (mobilenet_v2, "Fire Detection CNN"),
        (resnet50, "Hazard Identification CNN"),
        (lambda: AutoModelForCausalLM.from_pretrained('distilgpt2'), "Safety Alert (distilGPT2)")],

    "Pipeline I (Crowd Behavior)": [
        (lambda: YOLO('yolov8x.pt'), "YOLOv8x (Detector)"),
        (resnet50, "Crowd Counting CNN"),
        (resnet50, "Behavior Classification CNN"),
        (lambda: AutoModelForCausalLM.from_pretrained('distilgpt2'), "Incident Summary (distilGPT2)")]
}

# Measure each pipeline separately
for pipeline_name, models in pipelines.items():
    print(f"\n--- {pipeline_name} ---")
    for model_func, model_name in models:
        measure_model_load(lambda m=model_func: m().to(device), model_name)