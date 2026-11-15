#!/usr/bin/env bash
# lora_cost_attribution — sweep batch size x LoRA config on one GPU.
#
# Each point is a separate process (fresh vLLM engine). Results append to
# $EXP_DIR/raw.jsonl.
#
# Run from serving/:
#   QUICK=1 bash experiments/lora_cost_attribution/run.sh   # smoke: B=8, A/B/C/D
#   bash experiments/lora_cost_attribution/run.sh           # full sweep
#
# Overrides: MODEL, DTYPE, BATCH_SIZES, KS, REPS, EXP_DIR, CONDA_ENV
#   T4 / pre-Ampere (no bf16): DTYPE=half

set -euo pipefail

SERVING_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SERVING_DIR"
EXP="experiments/lora_cost_attribution"

CONDA_ENV="${CONDA_ENV:-}"
if [[ -n "$CONDA_ENV" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

MODEL="${MODEL:-Qwen/Qwen2.5-3B}"
DTYPE="${DTYPE:-auto}"
REPS="${REPS:-5}"
EXP_DIR="${EXP_DIR:-$EXP/results}"

if [[ "${QUICK:-0}" == "1" ]]; then
    BATCH_SIZES="${BATCH_SIZES:-8}"
    CONFIGS="${CONFIGS:-A B C D}"
    KS="${KS:-}"
else
    BATCH_SIZES="${BATCH_SIZES:-1 8 16 32 64}"
    CONFIGS="${CONFIGS:-A B C C64 D E}"
    KS="${KS:-1 2 4 8 16}"
fi

if [[ ! -f "$EXP/adapters/r64_15/adapter_model.safetensors" ]]; then
    python "$EXP/make_adapters.py" --model "$MODEL" --num-r64 16
fi

run() {
    python "$EXP/run.py" --model "$MODEL" --dtype "$DTYPE" --reps "$REPS" \
        --exp-dir "$EXP_DIR" "$@" || echo "!! failed: $*"
}

for B in $BATCH_SIZES; do
    for C in $CONFIGS; do
        [[ "$C" == "E" && "$B" -lt 2 ]] && continue
        run --batch-size "$B" --config "$C"
    done
    for K in $KS; do
        [[ "$K" -gt "$B" ]] && continue
        run --batch-size "$B" --config F --k "$K"
    done
done

echo "done -> $EXP_DIR/raw.jsonl"
