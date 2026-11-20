#!/usr/bin/env bash
# staging_ablation: host vs GPU staging of waiting requests, shared backbone.
#
# Run from serving/:
#   bash experiments/staging_ablation/run.sh
# One backbone, quick check:
#   BACKBONES="momentlarge" RPS="10" DURATION=10 bash experiments/staging_ablation/run.sh
set -euo pipefail

SERVING_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SERVING_DIR"

CONDA_ENV="${CONDA_ENV:-fmtk}"
FMTK_DIR="${FMTK_DIR:-$(cd "$SERVING_DIR/../.." && pwd)/FMTK}"
FMAAS_DIR="${FMAAS_DIR:-$(cd "$SERVING_DIR/.." && pwd)}"
export PYTHONPATH="${FMTK_DIR}/src:${FMAAS_DIR}:${SERVING_DIR}:${PYTHONPATH:-}"

if command -v conda &> /dev/null; then
    PYTHON="${PYTHON:-conda run --no-capture-output -n ${CONDA_ENV} python}"
else
    PYTHON="${PYTHON:-python}"
fi

# Same backbones as the T4 sharing_benefit figure.
BACKBONES="${BACKBONES:-momentlarge dinobase-patch swinlarge}"
RPS="${RPS:-1,5,10,15,20}"
MODES="${MODES:-host,gpu}"
N_TASKS="${N_TASKS:-2}"
DURATION="${DURATION:-30}"
REPEATS="${REPEATS:-2}"
MAX_BATCH="${MAX_BATCH:-5}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
OUT_DIR="${OUT_DIR:-experiments/staging_ablation/results_t4}"

for bb in $BACKBONES; do
    echo "=== $bb ==="
    $PYTHON experiments/staging_ablation/run.py \
        --backbone "$bb" --rps "$RPS" --modes "$MODES" --n-tasks "$N_TASKS" \
        --duration "$DURATION" --repeats "$REPEATS" --max-batch "$MAX_BATCH" \
        --cuda "$CUDA_DEVICE" --out-dir "$OUT_DIR"
done
echo "Summary: $OUT_DIR/summary.csv"
