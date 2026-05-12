#!/usr/bin/env bash
# Direct-runtime interference calibration:
# Fix T1 at T1_TPCS (default 5). Sweep T2 TPC count over T2_TPC_COUNTS
# (default {0, 5, 10, 15}). Each task runs in its own subprocess pinned to
# its TPC partition and calls runtime.run_batch in a tight loop.
#
# T2_TPCS=0 means "T1 alone".
#
# Run from serving/:
#   bash experiments/fair_share/tsfm_victim_sweep/calibrate2.sh
set -euo pipefail

SERVING_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$SERVING_DIR"

FMTK_DIR="${FMTK_DIR:-../../FMTK}"
FMAAS_DIR="${FMAAS_DIR:-..}"
[[ "$FMTK_DIR" = /* ]]  || FMTK_DIR="$SERVING_DIR/$FMTK_DIR"
[[ "$FMAAS_DIR" = /* ]] || FMAAS_DIR="$SERVING_DIR/$FMAAS_DIR"
export PYTHONPATH="${FMTK_DIR}/src:${FMAAS_DIR}:${PYTHONPATH:-}"

CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:-momentlarge}"
TPC_MODE="${TPC_MODE:-libsmctrl}"

T1_TASK="${T1_TASK:-ecgclass}"
T2_TASK="${T2_TASK:-gestureclass}"
T1_TPCS="${T1_TPCS:-5}"
T2_TPC_COUNTS=(${T2_TPC_COUNTS:-0 5 10 15})

BATCH_SIZE="${BATCH_SIZE:-8}"
DURATION="${DURATION:-15}"
WARMUP_SECS="${WARMUP_SECS:-3}"

OUT_DIR="${OUT_DIR:-experiments/fair_share/tsfm_victim_sweep/results/calibration2}"
OUT_CSV="$OUT_DIR/calibration2.csv"
mkdir -p "$OUT_DIR"
rm -f "$OUT_CSV"

if command -v conda &> /dev/null; then
    CONDA_ENV="${CONDA_ENV:-fmtk}"
    PYTHON="${PYTHON:-conda run --no-capture-output -n ${CONDA_ENV} python}"
else
    PYTHON="${PYTHON:-python}"
fi

echo "================================================================"
echo "  Direct-runtime interference calibration"
echo "  T1: $T1_TASK @ $T1_TPCS TPCs (fixed, partition [0..$((T1_TPCS-1))])"
echo "  T2: $T2_TASK, TPC sweep = {${T2_TPC_COUNTS[*]}}"
echo "  batch=$BATCH_SIZE  dur=${DURATION}s  warmup=${WARMUP_SECS}s"
echo "  output: $OUT_CSV"
echo "================================================================"

for N in "${T2_TPC_COUNTS[@]}"; do
    echo ""
    echo "---- T2 tpcs=$N ----"
    T2_ARGS=()
    if [[ "$N" -gt 0 ]]; then
        T2_ARGS=(--t2-task "$T2_TASK" --t2-tpcs "$N")
    else
        T2_ARGS=(--t2-tpcs 0)
    fi

    $PYTHON -u "$SERVING_DIR/experiments/fair_share/tsfm_victim_sweep/calibrate2.py" \
        --backbone     "$BACKBONE"     \
        --cuda-device  "$CUDA_DEVICE"  \
        --tpc-mode     "$TPC_MODE"     \
        --t1-task      "$T1_TASK"      \
        --t1-tpcs      "$T1_TPCS"      \
        "${T2_ARGS[@]}"                \
        --batch-size   "$BATCH_SIZE"   \
        --duration     "$DURATION"     \
        --warmup-secs  "$WARMUP_SECS"  \
        --out-csv      "$OUT_CSV"      \
    || echo "[calib2] WARNING: failed for T2_TPCS=$N"
done

echo ""
echo "[calib2] done. Results: $OUT_CSV"
