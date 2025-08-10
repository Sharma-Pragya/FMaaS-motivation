# accuracy = {"m1": 0.9, "m2": 0.8}
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

    ('Qwen2.5',"image_captioning"):1,
    ('Qwen2.5',"visual_question_answering"):1,

    ('Molmo',"image_captioning"):1,
    ('Molmo',"visual_question_answering"):1,

    ('Llama-Vision',"image_captioning"):1,
    ('Llama-Vision',"visual_question_answering"):1,

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

    ('chronos','eeg_anamoly_detection'):1,
    ('chronos','ppg_anamoly_detection'):1,
    ('chronos','eeg_classification'):1,
    ('chronos','energy_forecasting'):1,

    ('moment','eeg_anamoly_detection'):1,
    ('moment','ppg_anamoly_detection'):1,
    ('moment','eeg_classification'):1,
    ('moment','energy_forecasting'):1,

    ('TimesFM','eeg_anamoly_detection'):1,
    ('TimesFM','ppg_anamoly_detection'):1,
    ('TimesFM','eeg_classification'):1,
    ('TimesFM','energy_forecasting'):1,

    ('Lag-Llama','eeg_anamoly_detection'):1,
    ('Lag-Llama','ppg_anamoly_detection'):1,
    ('Lag-Llama','eeg_classification'):1,
    ('Lag-Llama','energy_forecasting'):1,
}

latency = {
    # Text Models
    ("A16","llama3", "sentiment_analysis"): 300,
    ("A16","llama3", "question_answering"): 310,
    ("A16","llama3", "named_entity_recognition"): 290,
    ("A16","llama3", "text_classification"): 300,
    ("A16","llama3", "code_generation"): 320,
    ("A16","llama3", "code_completion"): 310,
    ("A16","llama3", "code_translation"): 315,
    ("A16","llama3", "text_embedding"): 280,
    ("A16","llama3", "dialogue_generation"): 320,
    ("A16","llama3", "prompted_reasoning"): 330,

    ("A16","mistral", "sentiment_analysis"): 270,
    ("A16","mistral", "question_answering"): 280,
    ("A16","mistral", "text_classification"): 275,
    ("A16","mistral", "code_generation"): 290,
    ("A16","mistral", "code_completion"): 285,
    ("A16","mistral", "code_translation"): 288,
    ("A16","mistral", "text_embedding"): 260,
    ("A16","mistral", "dialogue_generation"): 295,
    ("A16","mistral", "prompted_reasoning"): 300,

    ("A16","t5", "sentiment_analysis"): 200,
    ("A16","t5", "text_summarization"): 210,
    ("A16","t5", "question_answering"): 215,
    ("A16","t5", "text_classification"): 190,
    ("A16","t5", "language_translation"): 195,

    ("A16","bart", "sentiment_analysis"): 220,
    ("A16","bart", "text_summarizations"): 230,
    ("A16","bart", "question_answering"): 240,
    ("A16","bart", "text_classification"): 225,
    ("A16","bart", "dialogue_generation"): 260,

    # Audio
    ("A16","whisper", "speech_recognition"): 350,
    ("A16","whisper", "speaker_identification"): 320,
    ("A16","whisper", "audio_classification"): 300,

    ("A16","wav2vec2", "speech_recognition"): 310,
    ("A16","wav2vec2", "speaker_identification"): 295,

    ("A16","vggish", "audio_classification"): 80,
    ("A16","vggish", "sound_event_detection"): 100,

    # Vision & Multimodal
    ("A16","sam", "image_segmentation"): 950,
    ("A16","sam", "object_detection"): 820,

    ("A16","clip", "image_captioning"): 250,
    ("A16","clip", "image_classification"): 110,
    ("A16","clip", "text_embedding"): 100,
    ("A16","clip", "image_embedding"): 120,
    ("A16","clip", "multimodal_embedding"): 130,
    ("A16","clip", "semantic_search"): 140,

    ("A16","blip2", "image_captioning"): 550,
    ("A16","blip2", "visual_question_answering"): 600,
    ("A16","blip2", "video_captioning"): 700,
    ("A16","blip2", "multimodal_embedding"): 520,

    ("A16","flamingo", "image_captioning"): 850,
    ("A16","flamingo", "visual_question_answering"): 950,
    ("A16","flamingo", "multimodal_embedding"): 900,
    ("A16","flamingo", "dialogue_generation"): 880,
    ("A16","flamingo", "video_captioning"): 1000,

    ("A16","Qwen2.5", "image_captioning"): 400,
    ("A16","Qwen2.5", "visual_question_answering"): 450,

    ("A16","Molmo", "image_captioning"): 420,
    ("A16","Molmo", "visual_question_answering"): 470,

    ("A16","Llama-Vision", "image_captioning"): 430,
    ("A16","Llama-Vision", "visual_question_answering"): 480,

    ("A16","imagebind", "text_embedding"): 200,
    ("A16","imagebind", "text_embedding"): 200,
    ("A16","imagebind", "image_embedding"): 210,
    ("A16","imagebind", "multimodal_embedding"): 220,
    ("A16","imagebind", "multimodal_segmentation"): 250,

    ("A16","clipseg", "image_segmentation"): 420,
    ("A16","clipseg", "multimodal_segmentation"): 440,

    ("A16","yolov8", "object_detection"): 60,
    ("A16","yolov8", "image_segmentation"): 75,

    ("A16","dinov2", "image_classification"): 130,
    ("A16","dinov2", "text_embedding"): 160,
    ("A16","dinov2", "image_embedding"): 165,
    ("A16","dinov2", "image_segmentation"): 150,

    ("A16","stable_diffusion", "text_to_image_generation"): 950,
    ("A16","dalle3", "text_to_image_generation"): 1000,

    ("A16","tesseract", "ocr"): 180,

    ("A16","donut", "image_captioning"): 500,
    ("A16","donut", "ocr"): 400,
    ("A16","donut", "document_retrieval"): 450,

    ("A16","bark", "text_to_speech"): 480,
    ("A16","valle", "text_to_speech"): 520,

    # Code models
    ("A16","codellama", "code_generation"): 320,
    ("A16","codellama", "code_completion"): 310,
    ("A16","codellama", "code_translation"): 315,

    ("A16","starcoder", "code_generation"): 330,
    ("A16","starcoder", "code_completion"): 320,
    ("A16","starcoder", "code_translation"): 325,

    ("A16","gemma", "sentiment_analysis"): 290,
    ("A16","gemma", "question_answering"): 300,
    ("A16","gemma", "text_classification"): 295,
    ("A16","gemma", "dialogue_generation"): 310,

    ("A16","phi3", "code_generation"): 120,

    ("A16","palm2", "sentiment_analysis"): 310,
    ("A16","palm2", "text_summarization"): 320,
    ("A16","palm2", "question_answering"): 330,
    ("A16","palm2", "named_entity_recognition"): 305,
    ("A16","palm2", "text_classification"): 315,
    ("A16","palm2", "code_generation"): 340,
    ("A16","palm2", "code_translation"): 345,
    ("A16","palm2", "text_embedding"): 270,
    ("A16","palm2", "language_translation"): 290,
    ("A16","palm2", "dialogue_generation"): 350,
    ("A16","palm2", "prompted_reasoning"): 360,
    ("A16","palm2", "semantic_search"): 275,

    # Time series
    ("A16","chronos", "eeg_anamoly_detection"): 100,
    ("A16","chronos", "ppg_anamoly_detection"): 110,
    ("A16","chronos", "eeg_classification"): 120,
    ("A16","chronos", "energy_forecasting"): 130,

    ("A16","moment", "eeg_anamoly_detection"): 110,
    ("A16","moment", "ppg_anamoly_detection"): 115,
    ("A16","moment", "eeg_classification"): 125,
    ("A16","moment", "energy_forecasting"): 135,

    ("A16","TimesFM", "eeg_anamoly_detection"): 120,
    ("A16","TimesFM", "ppg_anamoly_detection"): 125,
    ("A16","TimesFM", "eeg_classification"): 130,
    ("A16","TimesFM", "energy_forecasting"): 140,

    ("A16","Lag-Llama", "eeg_anamoly_detection"): 150,
    ("A16","Lag-Llama", "ppg_anamoly_detection"): 155,
    ("A16","Lag-Llama", "eeg_classification"): 160,
    ("A16","Lag-Llama", "energy_forecasting"): 170,
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
    "Qwen2.5":5.2,
    "Llama-Vision":7.8,
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
    "palm2": 16,
    'chronos': 0.3, 
    'moment': 0.3, 
    'TimesFM': 0.3, 
    'TimeGPT‑1': 0.3, 
    'Lag-Llama': 0.3
}

# memory_device={"d1": 16,"d2": 16,"d3": 16,"d4": 16,"d5": 16,"d6":16,"d7": 16,"d8":16,"d9":16,'d10':16}
memory_device={"A16":16}