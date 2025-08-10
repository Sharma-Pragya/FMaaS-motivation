# devices = ['d1', 'd2', 'd3', 'd4', 'd5','d6','d7','d8','d9','d10']
devices={
    'd1':
    {
        'type':'A16'
    },
    'd2':
    {
        'type':'A16'
    },
    'd3':
    {
        'type':'A16'
    },
    'd4':
    {
        'type':'A16'
    },
    'd5':
    {
        'type':'A16'
    },
    'd6':
    {
        'type':'A16'
    },
    'd7':
    {
        'type':'A16'
    },
    'd8':
    {
        'type':'A16'
    },
    'd9':
    {
        'type':'A16'
    }, 
    'd10':
    {
        'type':'A16'
    },    
    'd11':
    {
        'type':'A16'
    },   

    'd12':
    {
        'type':'A16'
    },   
}
devices={}
for i in range(1,100):
    devices.update({f"d{i}": {'type':'A16'}})

models = ["llama3", "mistral","t5","bart","whisper","wav2vec2",
          "vggish","sam","clip","blip2","flamingo","imagebind",
          "clipseg","yolov8","dinov2","stable_diffusion",
          "tesseract","donut","bark","valle","codellama","starcoder",
          'gemma','phi3','Qwen2.5','Llama-Vision',
          'palm2', 'chronos', 'moment', 'TimesFM', 'Lag-Llama'
        ]

tasks = {
    "sentiment_analysis": 400,              # near-instant response expected
    "text_summarization": 500,              # longer task, tolerable delay
    "question_answering": 250,              # short queries expected to be fast
    "named_entity_recognition": 300,        # typically fast NLP task
    "text_classification": 300,             # fast response expected
    "text_to_image_generation": 1000,       # known to be slow (SD/DALL·E)
    "image_captioning": 500,                # acceptable within half a second
    "visual_question_answering": 600,       # depends on multimodal model
    "image_segmentation": 600,              # moderate latency
    "object_detection": 300,                # fast, esp. for robotic vision
    "image_classification": 200,            # fast real-time systems
    "ocr": 500,                             # typical OCR speed
    "speech_recognition": 400,              # ideally real-time or near
    "speaker_identification": 500,          # tolerable within 0.3s
    "text_to_speech": 500,                  # user can tolerate slight delay
    "audio_classification": 400,            # fast response expected
    "sound_event_detection": 250,           # real-time or streaming use
    "code_generation": 500,                 # interactive coding use
    "code_translation": 450,                # slightly faster expectation
    "text_embedding": 400,                  # used in search/vector indexing
    "image_embedding": 300,                 # depends on image size
    "multimodal_embedding": 400,            # typically slower than unimodal
    "language_translation": 300,            # interactive
    "dialogue_generation": 400,             # natural conversation pacing
    "prompted_reasoning": 500,              # slower is acceptable
    "semantic_search": 300,                 # quick retrieval needed
    "document_retrieval": 500,              # fast feedback expected
    "video_captioning": 800,                # longer task
    "eeg_anamoly_detection": 250,           # near-real-time for medical
    "ppg_anamoly_detection": 250,           # similar
    "eeg_classification": 300,              # real-time use cases
    "energy_forecasting": 500               # often batch or interval based
}

