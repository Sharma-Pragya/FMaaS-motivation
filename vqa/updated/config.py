models_directory = "models/"
dataset_directory = "dataset/"
hf_token_path = "../../hf-token.txt"
log_file= "combined_metrics.csv"

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
model_name = ["vikhyatk/moondream2", 
                "llava-hf/llava-1.5-7b-hf", 
                "llava-hf/llava-1.5-13b-hf", 
                "llava-hf/llava-v1.6-vicuna-13b-hf", 
                "Qwen/Qwen2.5-VL-3B-Instruct", 
                "Qwen/Qwen2.5-VL-7B-Instruct", 
                # "microsoft/Phi-3.5-vision-instruct",
                "allenai/Molmo-7B-D-0924",
                "meta-llama/Llama-3.2-11B-Vision-Instruct",
                "openbmb/MiniCPM-V-2_6"]
dataset=dataset_directory+"val2014"
dataset_json=dataset_directory+"val.json"