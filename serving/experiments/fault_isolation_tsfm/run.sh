#!/usr/bin/env bash
# fault_isolation_tsfm — a fault in one task's decoder vs. the other tasks'
# service, on MOMENT-large: S-BE, BE, SP, FMVisor (see run.py).
#
# Every mode runs its serving processes fresh; the parent never touches CUDA,
# so modes do not share GPU state. One run per fault kind, then the plots.
#
# Run from serving/:
#   bash experiments/fault_isolation_tsfm/run.sh
#
# Overrides:
#   KINDS="gpu segv" MODES="s_be,be,sp,fmvisor" NTASKS=8 CLIENTS=1 \
#   WARMUP=10 DURATION=40 FAULT_AT=15 CUDA_DEVICE=cuda:0 \
#   EXP_DIR=experiments/fault_isolation_tsfm/results_t4 bash experiments/fault_isolation_tsfm/run.sh
#
# Memory leak in the faulty task's decoder (plotted with mem_plot.py):
#   KINDS=leak MODES=s_be,be,sp DURATION=60 LEAK_MB_S=400 \
#   EXP_DIR=experiments/fault_isolation_tsfm/results_t4 bash experiments/fault_isolation_tsfm/run.sh
#
# SP needs libsmctrl (TPC_LIB_DIR, default <repo parent>/TPC_controller/tpc_controller).

set -euo pipefail

SERVING_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SERVING_DIR"

CONDA_ENV="${CONDA_ENV:-fmtk}"
KINDS="${KINDS:-gpu segv}"
MODES="${MODES:-s_be,be,sp,fmvisor}"
NTASKS="${NTASKS:-8}"
CLIENTS="${CLIENTS:-1}"
BACKBONE="${BACKBONE:-momentlarge}"
WARMUP="${WARMUP:-10}"
DURATION="${DURATION:-40}"
FAULT_AT="${FAULT_AT:-15}"
LEAK_MB_S="${LEAK_MB_S:-400}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
EXP_DIR="${EXP_DIR:-experiments/fault_isolation_tsfm/results}"

# FMTK provides the backbone and decoders (same default as isolation_mechanisms).
FMTK_DIR="${FMTK_DIR:-$(cd "$SERVING_DIR/../.." && pwd)/FMTK}"
if [[ ! -d "$FMTK_DIR/src/fmtk" ]]; then
    echo "ERROR: FMTK not found at $FMTK_DIR (set FMTK_DIR)"
    exit 1
fi
export PYTHONPATH="${FMTK_DIR}/src:${PYTHONPATH:-}"

if command -v conda &> /dev/null; then
    PYTHON="${PYTHON:-conda run --no-capture-output -n ${CONDA_ENV} python}"
else
    PYTHON="${PYTHON:-python}"
fi

LOG_DIR="$EXP_DIR/logs"
mkdir -p "$LOG_DIR"

for KIND in $KINDS; do
    echo ""
    echo "[RUN] fault=$KIND modes=$MODES ntasks=$NTASKS clients=$CLIENTS backbone=$BACKBONE"
    $PYTHON -u experiments/fault_isolation_tsfm/run.py \
        --modes        "$MODES"      \
        --n-tasks      "$NTASKS"     \
        --clients      "$CLIENTS"    \
        --backbone     "$BACKBONE"   \
        --warmup       "$WARMUP"     \
        --duration     "$DURATION"   \
        --fault-at     "$FAULT_AT"   \
        --fault-kind   "$KIND"       \
        --leak-mb-per-s "$LEAK_MB_S" \
        --device       "$CUDA_DEVICE" \
        --exp-dir      "$EXP_DIR"    \
        2>&1 | tee "$LOG_DIR/fault_${KIND}.log"
    if [[ "$KIND" == "leak" ]]; then
        $PYTHON experiments/fault_isolation_tsfm/mem_plot.py --exp-dir "$EXP_DIR"
    else
        $PYTHON experiments/lora_fair_share/fault_plot.py \
            --exp-dir "$EXP_DIR" --fault-kind "$KIND"
    fi
done

echo ""
echo "[INFO] Done. Results under $EXP_DIR/fault*/"
