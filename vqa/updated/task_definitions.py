"""
Task definitions: prompts, parsers, and evaluation logic
All task-specific code in ONE place
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Single unified CSV file for ALL results (no per-task CSVs)
UNIFIED_LOG_FILE = "unified_metrics.csv"

# CSV columns (same as existing files)
CSV_COLUMNS = [
    "model_name", "dataset_name", "device", "model_load_duration_sec",
    "gpu_load_memory_mb", "avg_cpu_memory_usage_mb", "avg_cpu_usage_percent",
    "avg_gpu_usage_percent", "avg_gpu_memory_usage_mb", "total_prompt_tokens",
    "total_generated_tokens", "ttft_ms", "avg_latency_ms", "throughput_tps",
    "accuracy", "total_time", "num_samples", "gpu_name"
]

# ==================== TASK REGISTRY ====================
TASK_REGISTRY = {
    "crowd": {
        "dataset_root": ROOT / "dataset/crowd_counting",
        "dataset_json": "labels.json",
        "prompt": (
            "Look at this image and estimate the crowd density. "
            "Answer with ONLY ONE of these categories: very_sparse, sparse, moderate, dense, or very_dense. "
            "Do not provide any explanation or numbers, just the category name. "
            "Please answer in a few words."
        ),
        "parser": "parse_crowd_label",
        "evaluator": "evaluate_crowd",
    },

    "scene": {
        "dataset_root": ROOT / "dataset/scene_classification",
        "dataset_json": "labels.json",
        "prompt": (
            "What type of scene is shown in this image? "
            "Answer with a short scene description (for example: 'kitchen', 'beach', 'office', 'mountain', etc.). "
            "Provide only the scene type, no additional explanation. "
            "Please answer in a few words."
        ),
        "parser": "parse_scene_label",
        "evaluator": "evaluate_scene",
    },

    "ocr": {
        "dataset_root": ROOT / "dataset/ocr",
        "dataset_json": "labels.json",
        "prompt": (
            "Read all the text visible in this image. "
            "Provide only the text content, nothing else. "
            "Please answer in one word."
        ),
        "parser": "parse_ocr_label",
        "evaluator": "evaluate_ocr",
    },

    "vqa": {
        "dataset_root": ROOT / "dataset/vqa",
        "dataset_json": "val.json",
        "prompt": "{question} Please answer in one word.",  # Placeholder - actual question from dataset
        "parser": "parse_vqa_label",
        "evaluator": "evaluate_vqa",
    },

    "traffic": {
        "dataset_root": ROOT / "dataset/traffic_classification",
        "dataset_json": "labels.json",
        "prompt": (
            "What traffic sign is shown in this image? "
            "Answer with the sign type only (e.g., 'stop sign', 'speed limit', 'yield', etc.). "
            "Please answer in a few words."
        ),
        "parser": "parse_classification_label",
        "evaluator": "evaluate_substring_match",
    },

    "gesture": {
        "dataset_root": ROOT / "dataset/gesture_recognition",
        "dataset_json": "labels.json",
        "prompt": (
            "What hand gesture is being shown in this image? "
            "Answer with the gesture name only. "
            "Please answer in a few words."
        ),
        "parser": "parse_classification_label",
        "evaluator": "evaluate_substring_match",
    },

    "activity": {
        "dataset_root": ROOT / "dataset/activity_recognition",
        "dataset_json": "labels.json",
        "prompt": (
            "What activity is the person doing in this image? "
            "Answer with the activity name only. "
            "Please answer in a few words."
        ),
        "parser": "parse_classification_label",
        "evaluator": "evaluate_substring_match",
    },

    "object_detection": {
        "dataset_root": ROOT / "dataset/object_detection",
        "dataset_json": "annotations.json",
        "prompt": (
            "What is the main object in this image? "
            "Answer with one object name from the COCO dataset. "
            "Provide only the object name without any explanation. "
            "Please answer in one word."
        ),
        "parser": "parse_object_detection_label",
        "evaluator": "evaluate_object_detection",
    },

    "image_classification": {
        "dataset_root": ROOT / "dataset/image_classification",
        "dataset_json": "labels.json",
        "prompt": (
            "What is the main object or subject in this image? "
            "Answer with a single word or short phrase. "
            "Please answer in one word."
        ),
        "parser": "parse_classification_label",
        "evaluator": "evaluate_image_classification",
    },
}

# ==================== LABEL PARSERS ====================
def parse_crowd_label(text: str) -> str:
    """Parse crowd density category"""
    if not text:
        return ""
    text = text.strip().lower()
    text = text.strip('.,!?;:"\'"')
    text = ' '.join(text.split())
    text = text.replace(' ', '_').replace('-', '_')

    # Extract category if model provides extra text
    if 'very_dense' in text or 'verydense' in text:
        return 'very_dense'
    elif 'very_sparse' in text or 'verysparse' in text:
        return 'very_sparse'
    elif 'dense' in text:
        return 'dense'
    elif 'moderate' in text:
        return 'moderate'
    elif 'sparse' in text:
        return 'sparse'

    return text

def parse_scene_label(text: str) -> str:
    """Parse scene classification label"""
    if not text:
        return ""
    text = text.strip().lower()
    text = text.strip('.,!?;:"\'"')
    text = ' '.join(text.split())

    # Remove common prefixes that models might add
    prefixes = ["the scene is", "this is", "this looks like", "this appears to be",
                "the image shows", "it is", "it's", "scene:", "type:"]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Remove articles at the beginning
    if text.startswith("a "):
        text = text[2:]
    elif text.startswith("an "):
        text = text[3:]
    elif text.startswith("the "):
        text = text[4:]

    return text

def parse_ocr_label(text: str) -> str:
    """Parse OCR text"""
    if not text:
        return ""
    # Normalize whitespace but preserve text
    return ' '.join(text.strip().split())

def parse_vqa_label(text: str) -> str:
    """Parse VQA answer"""
    if not text:
        return ""
    # Simple cleanup
    text = text.strip().lower()
    text = text.strip('.,!?;:"\'"')
    return ' '.join(text.split())

def parse_classification_label(text: str) -> str:
    """Generic classification label parser"""
    if not text:
        return ""
    text = text.strip().lower()
    text = text.strip('.,!?;:"\'"')

    # Remove common prefixes
    prefixes = ["this is", "the image shows", "it is", "it's", "a ", "an ", "the "]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    return ' '.join(text.split())

def parse_object_detection_label(text: str) -> str:
    """Parse object detection output (single object name)"""
    if not text:
        return ""
    # Strip and lowercase
    text = text.strip().lower()
    # Take first word if multiple words
    category = text.split()[0] if text.split() else ""
    # Remove common punctuation
    category = category.strip('.,!?;:"\'"')
    return category

# ==================== EVALUATORS ====================
def evaluate_crowd(predicted: str, ground_truth: str) -> bool:
    """Evaluate crowd density prediction - bidirectional substring matching"""
    pred = predicted.lower().strip()
    gt = str(ground_truth).lower().strip()
    # Match individual files: (gt_label in pred_label) or (pred_label in gt_label)
    return (gt in pred) or (pred in gt)

def evaluate_scene(predicted: str, ground_truth: str) -> bool:
    """Evaluate scene classification - bidirectional substring matching"""
    pred = predicted.lower().strip()
    gt = str(ground_truth).lower().strip()
    # Match individual files: (gt_label in pred_label) or (pred_label in gt_label)
    return (gt in pred) or (pred in gt)

def evaluate_ocr(predicted: str, ground_truth: str) -> bool:
    """Evaluate OCR - normalized exact match"""
    # Normalize both strings: lowercase, remove extra whitespace
    pred_norm = ' '.join(predicted.lower().strip().split())
    gt_norm = ' '.join(str(ground_truth).lower().strip().split())
    return pred_norm == gt_norm

def evaluate_vqa(predicted: str, ground_truth) -> bool:
    """
    Evaluate VQA answer - substring matching.
    Ground truth can be a single answer or list of valid answers.
    Match individual files: if gt_answer.lower() in answer.lower()
    """
    pred_norm = predicted.lower().strip()

    # If ground truth is a list, check if any answer is substring of prediction
    if isinstance(ground_truth, list):
        return any(ans.lower().strip() in pred_norm for ans in ground_truth)
    else:
        # Match individual files: gt_answer in predicted answer
        return str(ground_truth).lower().strip() in pred_norm

def evaluate_exact_match(predicted: str, ground_truth: str) -> bool:
    """Generic exact match evaluation"""
    return predicted.lower().strip() == str(ground_truth).lower().strip()

def evaluate_substring_match(predicted: str, ground_truth: str) -> bool:
    """Bidirectional substring matching - used for traffic, gesture, activity"""
    pred = predicted.lower().strip()
    gt = str(ground_truth).lower().strip()
    # Match individual files: (gt_label in pred_label) or (pred_label in gt_label)
    return (gt in pred) or (pred in gt)

def evaluate_image_classification(predicted: str, ground_truth: str) -> bool:
    """One-way substring matching - used for image_classification"""
    pred = predicted.lower().strip()
    gt = str(ground_truth).lower().strip()
    # Match individual files: pred_label in gt_label
    return pred in gt

def evaluate_object_detection(predicted: str, ground_truth) -> bool:
    """
    Evaluate object detection.
    Check if predicted category is in ground truth categories list.
    """
    # Ground truth is a list of valid category names
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]

    # Normalize to lowercase
    pred_category = predicted.lower().strip()
    gt_categories = [cat.lower().strip() for cat in ground_truth]

    # Check if predicted category is in ground truth list
    return pred_category in gt_categories

# ==================== REGISTRY ACCESS FUNCTIONS ====================
def get_parser(parser_name: str):
    """Get parser function by name"""
    parsers = {
        "parse_crowd_label": parse_crowd_label,
        "parse_scene_label": parse_scene_label,
        "parse_ocr_label": parse_ocr_label,
        "parse_vqa_label": parse_vqa_label,
        "parse_classification_label": parse_classification_label,
        "parse_object_detection_label": parse_object_detection_label,
    }

    if parser_name not in parsers:
        raise ValueError(f"Unknown parser: {parser_name}")

    return parsers[parser_name]

def get_evaluator(evaluator_name: str):
    """Get evaluator function by name"""
    evaluators = {
        "evaluate_crowd": evaluate_crowd,
        "evaluate_scene": evaluate_scene,
        "evaluate_ocr": evaluate_ocr,
        "evaluate_vqa": evaluate_vqa,
        "evaluate_exact_match": evaluate_exact_match,
        "evaluate_object_detection": evaluate_object_detection,
        "evaluate_substring_match": evaluate_substring_match,
        "evaluate_image_classification": evaluate_image_classification,
    }

    if evaluator_name not in evaluators:
        raise ValueError(f"Unknown evaluator: {evaluator_name}")

    return evaluators[evaluator_name]
