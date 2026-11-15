#!/usr/bin/env bash
# lora_fair_share — basic run: B=64, weights 1:1. BE and SP first (one
# engine per tenant, two processes), then the two standalone baselines and
# all five schedulers on one shared engine (~9 x (WARMUP+DURATION) s plus
# engine loads), once per adapter case.
#
# Cases (which tenant's requests use the rank-64 adapter, cost_model.CASES):
#   mixed          P plain, A rank-64   (the original experiment)
#   both_adapter   P and A both rank-64 (same adapter)
#   both_plain     P and A both plain
#
# Run from serving/:
#   bash experiments/lora_fair_share/run.sh                                  # mixed only
#   CASES="both_adapter both_plain" bash experiments/lora_fair_share/run.sh  # new cases
#   WARMUP=30 DURATION=90 bash experiments/lora_fair_share/run.sh            # quicker
#
# DEMAND=A:P (default 1:1) sets the arrival ratio. Only S-BE (fcfs) depends on
# it: it admits in arrival order, while VTC/BFQ choose between two queues that
# are both non-empty. Example, A sending twice as many requests:
#   DEMAND=2:1 CASES="mixed both_adapter both_plain" \
#     MODES=standalone_P,standalone_A,fcfs,vtc,bfq_stfq bash experiments/lora_fair_share/run.sh
#
# Every case appends to the same summary.jsonl; plot.py separates them by case
# and prints a cross-case table.
#
# BE/SP need two copies of the model in GPU memory; a mode whose workers
# fail to start is skipped and the rest still run. SP needs libsmctrl
# (TPC_LIB_DIR, default ../../TPC_controller/tpc_controller from serving/).
#
# Overrides: CASES, DEMAND, MODEL, BATCH_CAP, WEIGHTS (A:P), MODES, REPS, WARMUP,
#            DURATION, EXP_DIR, CONDA_ENV, ISO_CAP, ISO_GPU_MEM, ISO_MAX_LORAS

set -euo pipefail

SERVING_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SERVING_DIR"
EXP="experiments/lora_fair_share"

CONDA_ENV="${CONDA_ENV:-}"
if [[ -n "$CONDA_ENV" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

for CASE in ${CASES:-mixed}; do
    python "$EXP/run.py" \
        --case "$CASE" \
        --demand "${DEMAND:-1:1}" \
        --model "${MODEL:-Qwen/Qwen2.5-3B}" \
        --batch-cap "${BATCH_CAP:-64}" \
        --weights "${WEIGHTS:-1:1}" \
        --modes "${MODES:-be,sp,standalone_P,standalone_A,fcfs,vtc,tuned_vtc,bfq,bfq_stfq}" \
        --reps "${REPS:-1}" \
        --warmup "${WARMUP:-60}" \
        --duration "${DURATION:-180}" \
        ${ISO_CAP:+--iso-cap "$ISO_CAP"} \
        ${ISO_GPU_MEM:+--iso-gpu-mem "$ISO_GPU_MEM"} \
        ${ISO_MAX_LORAS:+--iso-max-loras "$ISO_MAX_LORAS"} \
        --exp-dir "${EXP_DIR:-$EXP/results_t4}"
done

python "$EXP/plot.py" --exp-dir "${EXP_DIR:-$EXP/results_t4}"
