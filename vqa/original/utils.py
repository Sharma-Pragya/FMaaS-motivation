import logging
import subprocess
import re
import time
import os
import string
from ollama import chat

def setup_logger(name, log_file, level=logging.INFO):
    """Sets up a logger."""
    handler = logging.FileHandler(log_file)
    logging.getLogger().addHandler(logging.StreamHandler())  # Log to console
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def compute_accuracy(predictions, truths):


    """Computes accuracy and logs detailed metrics."""
    def normalise(s):
        return s.lower().strip().translate(str.maketrans('', '', string.punctuation))

    correct_count = sum(normalise(p) == normalise(t) for p, t in zip(predictions, truths))
    total_count = len(truths)
    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy

def model_predict_ollama(prompt,model="llama3.2",image_path=".",cwd="."):
    # Set CUDA_VISIBLE_DEVICES in the environment
    """Uses Ollama to get a prediction for the given prompt."""
    try:
        start_time=time.time()
        response = chat(
        model=model,
        messages=[
            {
            'role': 'user',
            'content': prompt,
            'images': [image_path],
            }
        ],
        )
        print(response)
        return response  
    
    except Exception as e:
        print(f"Error while using Ollama: {e}")
        return None
    
def sum_value(value_list):
    return sum(x for x in value_list if x is not None)

def length_value(value_list):
    return sum(1 for x in value_list if x is not None)

