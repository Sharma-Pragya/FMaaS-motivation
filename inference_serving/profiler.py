accuracy = {"m1": 0.9, "m2": 0.8}
can_serve = {
    ("llama3", "sentiment_analysis"): 1,
    ("llama3", "question_answering"): 1,
    ("llama3", "named_entity_recognition"): 1,
    ("llama3", "text_classification"): 1,
    ("llama3", "code_generation"): 1,
    ("llama3", "code_completion"): 1,
    ("llama3", "code_translation"): 1,
    ("llama3", "text_embedding"): 1,
    ("llama3", "dialogue_generation"): 1,
    ("llama3", "prompted_reasoning"): 1,

    ("mistral", "sentiment_analysis"): 1,
    ("mistral", "question_answering"): 1,
    ("mistral", "text_classification"): 1,
    ("mistral", "code_generation"): 1,
    ("mistral", "code_completion"): 1,
    ("mistral", "code_translation"): 1,
    ("mistral", "text_embedding"): 1,
    ("mistral", "dialogue_generation"): 1,
    ("mistral", "prompted_reasoning"): 1,

    ("t5", "sentiment_analysis"): 1,
    ("t5", "text_summarization"): 1,
    ("t5", "question_answering"): 1,
    ("t5", "text_classification"): 1,
    ("t5", "language_translation"): 1,

    ("bart", "sentiment_analysis"): 1,
    ("bart", "text_summarizations"): 1,
    ("bart", "question_answering"): 1,
    ("bart", "text_classification"): 1,
    ("bart", "dialogue_generation"): 1, 

    ("whisper", "speech_recognition"): 1,
    ("whisper", "speaker_identification"): 1,
    ("whisper", "audio_classification"): 1,

    ("wav2vec2", "speech_recognition"): 1,
    ("wav2vec2", "speaker_identification"): 1,

    ("vggish", "audio_classification"): 1,
    ("vggish", "sound_event_detection"): 1,

    ("sam", "image_segmentation"): 1,
    ("sam", "object_detection"): 1,

    ("clip", "image_captioning"): 1,
    ("clip", "image_classification"): 1,
    ("clip", "text_embedding"): 1,
    ("clip", "image_embedding"): 1,
    ("clip", "multimodal_embedding"): 1,
    ("clip", "semantic_search"): 1,

    ("blip2", "image_captioning"): 1,
    ("blip2", "visual_question_answering"): 1,
    ("blip2", "video_captioning"): 1,
    ("blip2", "multimodal_embedding"): 1,

    ("flamingo", "image_captioning"): 1,
    ("flamingo", "visual_question_answering"): 1,
    ("flamingo", "multimodal_embedding"): 1,
    ("flamingo", "dialogue_generation"): 1,
    ("flamingo", "video_captioning"): 1,

    ("imagebind", "text_embedding"): 1,
    ("imagebind", "image_embedding"): 1,
    ("imagebind", "multimodal_embedding"): 1,
    ("imagebind", "multimodal_segmentation"): 1,

    ("clipseg", "image_segmentation"): 1,
    ("clipseg", "multimodal_segmentation"): 1,

    ("yolov8", "object_detection"): 1,
    ("yolov8", "image_segmentation"): 1,

    ("dinov2", "image_classification"): 1,
    ("dinov2", "text_embedding"): 1,
    ("dinov2", "image_embedding"): 1,
    ("dinov2", "image_segmentation"): 1,

    ("stable_diffusion", "text_to_image_generation"): 1,

    ("dalle3", "text_to_image_generation"): 1,

    ("tesseract", "ocr"): 1,

    ("donut", "image_captioning"): 1,
    ("donut", "ocr"): 1,
    ("donut", "document_retrieval"): 1,

    ("bark", "text_to_speech"): 1,

    ("valle", "text_to_speech"): 1,

    ("codellama", "code_generation"): 1,
    ("codellama", "code_completion"): 1,
    ("codellama", "code_translation"): 1,

    ("starcoder", "code_generation"): 1,
    ("starcoder", "code_completion"): 1,
    ("starcoder", "code_translation"): 1,

    ("gemma", "sentiment_analysis"): 1,
    ("gemma", "question_answering"): 1,
    ("gemma", "text_classification"): 1,
    ("gemma", "dialogue_generation"): 1,

    ("phi3", "code_generation"): 1,

    ('palm2', 'sentiment_analysis'): 1,
    ('palm2', 'text_summarization'): 1,
    ('palm2', 'question_answering'): 1,
    ('palm2', 'named_entity_recognition'): 1,
    ('palm2', 'text_classification'): 1,
    ('palm2', 'code_generation'): 1,
    ('palm2', 'code_translation'): 1,
    ('palm2', 'text_embedding'): 1,
    ('palm2', 'language_translation'): 1,
    ('palm2', 'dialogue_generation'): 1,
    ('palm2', 'prompted_reasoning'): 1,
    ('palm2', 'semantic_search'): 1,

}
model_memory = {
    "llama3": 16,
    "mistral": 14,
    "t5": 3,
    "bart": 2,
    "whisper": 1.5,
    "wav2vec2": 1.5,
    "vggish": 1,
    "sam": 6,
    "clip": 2.5,
    "blip2": 8,
    "flamingo": 18,
    "imagebind": 5,
    "clipseg": 2,
    "yolov8": 2.5,
    "dinov2": 9,
    "stable_diffusion": 6.5,
    "tesseract": 0.5,
    "donut": 6,
    "bark": 7,
    "valle": 8,
    "codellama": 14,
    "starcoder": 30,
    "gemma": 14,
    "phi3": 8,
    "palm2": 16
}

# memory = {
#     "llama3": {
#         8: {"vram_fp16": 16, "vram_4bit": 5},
#         70: {"vram_fp16": 140, "vram_4bit": 20}
#     },
#     "mistral": {
#         7: {"vram_fp16": 14, "vram_4bit": 4.5}
#     },
#     "t5": {
#         0.77: {"vram_fp16": 3, "vram_4bit": None}
#     },
#     "bart": {
#         0.41: {"vram_fp16": 2, "vram_4bit": None}
#     },
#     "whisper": {
#         0.07: {"vram_fp16": 1.5, "vram_4bit": None}
#     },
#     "wav2vec2": {
#         0.3: {"vram_fp16": 1.5, "vram_4bit": None}
#     },
#     "vggish": {
#         0.06: {"vram_fp16": 1, "vram_4bit": None}
#     },
#     "sam": {
#         1: {"vram_fp16": 6, "vram_4bit": None}
#     },
#     "clip": {
#         0.15: {"vram_fp16": 2.5, "vram_4bit": 1}
#     },
#     "blip2": {
#         1: {"vram_fp16": 8, "vram_4bit": 4}
#     },
#     "flamingo": {
#         9: {"vram_fp16": 18, "vram_4bit": 6}
#     },
#     "imagebind": {
#         1.3: {"vram_fp16": 5, "vram_4bit": 2}
#     },
#     "clipseg": {
#         0.12: {"vram_fp16": 2, "vram_4bit": None}
#     },
#     "yolov8": {
#         0.3: {"vram_fp16": 2.5, "vram_4bit": None}
#     },
#     "dinov2": {
#         1: {"vram_fp16": 9, "vram_4bit": 3}
#     },
#     "stable_diffusion": {
#         0.89: {"vram_fp16": 6.5, "vram_4bit": 3.5}
#     },
#     "dalle3": {
#         None: {"vram_fp16": None, "vram_4bit": None, "note": "Cloud-only, proprietary"}
#     },
#     "tesseract": {
#         0.02: {"vram_fp16": 0.5, "vram_4bit": None, "note": "CPU-only"}
#     },
#     "donut": {
#         1: {"vram_fp16": 6, "vram_4bit": 2.5}
#     },
#     "bark": {
#         1: {"vram_fp16": 7, "vram_4bit": 3}
#     },
#     "valle": {
#         1.3: {"vram_fp16": 8, "vram_4bit": 4}
#     },
#     "codellama": {
#         7: {"vram_fp16": 14, "vram_4bit": 4.5}
#     },
#     "starcoder": {
#         15: {"vram_fp16": 30, "vram_4bit": 6.5}
#     },
#     "gemma": {
#         7: {"vram_fp16": 14, "vram_4bit": 4.5}
#     },
#     "phi3": {
#         3.8: {"vram_fp16": 8, "vram_4bit": 2.5}
#     },
# }

