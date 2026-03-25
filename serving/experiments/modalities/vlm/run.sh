#!/bin/bash
N_RUNS=${1:-5}
declare -A ADAPTERS=(
    ["phi"]="vlm_ocr_phi_lora"
    ["qwen-2B"]="vlm_ocr_qwen_lora"
)
for backbone in phi qwen-2B; do
    echo "=== Backbone: $backbone ==="
    for i in $(seq 0 $((N_RUNS - 1))); do
        echo "=== Run $i ==="
        python experiments/modalities/vlm/run.py \
            --run_idx $i \
            --backbone $backbone \
            --adapter_path "${ADAPTERS[$backbone]}" \
            "${@:2}"
    done
done
