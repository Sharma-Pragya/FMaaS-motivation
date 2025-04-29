import tqdm
import os
import json
import subprocess
from pathlib import Path
from utils import *
from prompt_generation import generate_prompt_vqa
import log_system_stats

c=100

model="llava-llama3:8b"
device='macmini'
device_type='mac'
logging.basicConfig(level=logging.INFO, force=True)
log_system_stats.start_logging(f"./logs/{device}/vqa/{model}",device_type)
log_path = Path(f"./logs/{device}/vqa/{model}/accuracy_latency.log")
# # Logger setup
logger = setup_logger("Ollama Evaluation Log", log_path)
logger.info("Starting evaluation with Ollama...")
image_dir='../datasets/vqa/val/val2014'
orignal_json_path='../datasets/vqa/val/val.json'
QA = json.load(open(orignal_json_path,'r'))
def main():
    count=0
    predictions = []
    truths = []
    total_time=[]
    load_duration=[]
    prompt_eval_count=[]
    prompt_eval_duration=[]
    prompt_eval_rate=[]
    eval_count=[]
    eval_duration=[]
    eval_rate=[]
    for each in tqdm.tqdm(QA.values()):
        image_path=f'{image_dir}/COCO_val2014_'+ str(each['image_id']).zfill(12) + '.jpg'
        prompt = generate_prompt_vqa(each['question'],image_path)
        response=model_predict_ollama(prompt=prompt, model=model,image_path=image_path)
        print(response)
        if response is not None:
            predictions.append(response.message.content)
            truths.append(each['answer'])
            total_time.append(response.total_duration)
            load_duration.append(response.load_duration)
            prompt_eval_count.append(response.prompt_eval_count)
            prompt_eval_duration.append(response.prompt_eval_duration)
            eval_count.append(response.eval_count)
            eval_duration.append(response.eval_duration)
            logger.info(f"{response} truth:{each['answer']}")
        if count==c:
            break
        count+=1
    log_system_stats.stop_logging_thread()
    accuracy = compute_accuracy(predictions, truths)

    logger.info(f"predictions: {predictions}")
    logger.info(f"truths: {truths}")
    logger.info(f"Total Accuracy: {accuracy:.4f}")
    logger.info(f"Total Duration: {sum_value(total_time)/(10**9*length_value(total_time)):.4f}")
    logger.info(f"Load Duration: {sum_value(load_duration)/(10**9*length_value(load_duration)):.4f}")
    logger.info(f"Prompt Eval Count: {sum_value(prompt_eval_count)/(length_value(prompt_eval_count)):.4f}")
    logger.info(f"Prompt Eval Duration: {sum_value(prompt_eval_duration)/(10**9*length_value(prompt_eval_duration)):.4f}")
    logger.info(f"Prompt Eval Rate: {sum_value(prompt_eval_count)*(10**9*length_value(prompt_eval_duration))/(sum_value(prompt_eval_duration)*length_value(prompt_eval_count)):.4f}")
    logger.info(f"Eval Count: {sum_value(eval_count)/(length_value(eval_count)):.4f}")
    logger.info(f"Eval Duration: {sum_value(eval_duration)/(10**9*length_value(eval_duration)):.4f}")
    logger.info(f"Eval Rate: {sum_value(eval_count)*(10**9*length_value(eval_duration))/(sum_value(eval_duration)*length_value(eval_count)):.4f}")

if __name__ == '__main__':
    main()
