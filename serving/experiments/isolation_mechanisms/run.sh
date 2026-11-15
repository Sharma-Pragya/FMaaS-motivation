#!/usr/bin/env bash
# isolation_mechanisms — overhead of model_only / thread_stream / process_context.
#
# Each (mode, ntasks) point runs in a fresh Python process so every point starts
# from a clean GPU. Results are appended to $EXP_DIR/summary.csv.
#
# Run from serving/:
#   bash experiments/isolation_mechanisms/run.sh
#
# Overrides:
#   BACKBONES="momentlarge swinlarge" MODES="thread_stream process_context_ipc" \
#     NTASKS="1 2 4 8" DURATION=30 CUDA_DEVICE=cuda:0 bash experiments/isolation_mechanisms/run.sh
#
# BACKBONES defaults to the paper's PyTorch backbones (built through FMTK, so
# FMTK_DIR must point at the FMTK checkout); resnet50 also works without FMTK.
#
# The *_mps modes start and stop their own private MPS daemon (separate
# pipe/log dirs), so no manual MPS setup is needed and other GPU jobs are not
# attached to it. Do not run it while a global MPS daemon owns the same GPU.

set -euo pipefail

SERVING_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SERVING_DIR"

CONDA_ENV="${CONDA_ENV:-fmtk}"
MODES="${MODES:-model_only thread_stream process_context process_context_ipc process_context_ipc_mps instance_per_task instance_per_task_mps}"
NTASKS="${NTASKS:-1 2 4 8}"
BACKBONES="${BACKBONES:-${BACKBONE:-momentlarge papageip dinobase swinlarge}}"
DURATION="${DURATION:-30}"
WARMUP="${WARMUP:-5}"
BATCH_WINDOW_MS="${BATCH_WINDOW_MS:-1.0}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
EXP_DIR="${EXP_DIR:-experiments/isolation_mechanisms/results}"

# FMTK provides the paper backbones and decoders. Default: sibling of the repo.
FMTK_DIR="${FMTK_DIR:-$(cd "$SERVING_DIR/../.." && pwd)/FMTK}"
if [[ ! -d "$FMTK_DIR/src/fmtk" ]]; then
    echo "WARNING: FMTK not found at $FMTK_DIR (set FMTK_DIR); only resnet* backbones will work"
fi
export PYTHONPATH="${FMTK_DIR}/src:${PYTHONPATH:-}"

if command -v conda &> /dev/null; then
    PYTHON="${PYTHON:-conda run --no-capture-output -n ${CONDA_ENV} python}"
else
    PYTHON="${PYTHON:-python}"
fi

LOG_DIR="$EXP_DIR/logs"
mkdir -p "$LOG_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  isolation_mechanisms"
echo "  Modes    : $MODES"
echo "  N tasks  : $NTASKS"
echo "  Backbones: $BACKBONES   device=$CUDA_DEVICE"
echo "  Duration : ${DURATION}s (+${WARMUP}s warmup)"
echo "  Summary  : $EXP_DIR/summary.csv"
echo "════════════════════════════════════════════════════════════════"

for BACKBONE in $BACKBONES; do
    for N in $NTASKS; do
        for MODE in $MODES; do
            echo ""
            echo "[RUN] backbone=$BACKBONE mode=$MODE ntasks=$N"
            $PYTHON -u experiments/isolation_mechanisms/run.py \
                --mode            "$MODE"            \
                --ntasks          "$N"               \
                --backbone        "$BACKBONE"        \
                --duration        "$DURATION"        \
                --warmup          "$WARMUP"          \
                --batch-window-ms "$BATCH_WINDOW_MS" \
                --device          "$CUDA_DEVICE"     \
                --out-csv         "$EXP_DIR/summary.csv" \
                2>&1 | tee "$LOG_DIR/${BACKBONE}_${MODE}_n${N}.log"
            sleep 2
        done
    done
done

echo ""
echo "[INFO] Done. Results: $EXP_DIR/summary.csv"
