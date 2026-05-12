#!/usr/bin/env bash
# Calibration: measure single-task max throughput vs TPC count.
#
# For each TPC count in TPC_COUNTS, starts one TPC-partitioned device,
# pushes saturating offered load via calibrate.py, and records delivered
# throughput. Output CSV → results/calibration/calibration.csv.
#
# Run from serving/:
#   bash experiments/fair_share/tsfm_victim_sweep/calibrate.sh
set -euo pipefail

SERVING_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$SERVING_DIR"

FMTK_DIR="${FMTK_DIR:-../../FMTK}"
FMAAS_DIR="${FMAAS_DIR:-..}"
[[ "$FMTK_DIR" = /* ]]  || FMTK_DIR="$SERVING_DIR/$FMTK_DIR"
[[ "$FMAAS_DIR" = /* ]] || FMAAS_DIR="$SERVING_DIR/$FMAAS_DIR"
export PYTHONPATH="${FMTK_DIR}/src:${FMAAS_DIR}:${PYTHONPATH:-}"

DEVICE_PORT="${DEVICE_PORT:-8000}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:-momentlarge}"
TPC_MODE="${TPC_MODE:-libsmctrl}"

# Tasks to calibrate (run separately on the partitioned device).
TASKS=(${TASKS:-ecgclass gestureclass})

# TPC counts to sweep. Total TPCs on the GPU = 20 (40 SMs / 2).
TPC_COUNTS=(${TPC_COUNTS:-2 5 10 14 20})

OFFERED_RPS="${OFFERED_RPS:-200}"
DURATION="${DURATION:-15}"
WARMUP_SECS="${WARMUP_SECS:-3}"
BATCH_SIZE="${BATCH_SIZE:-3}"
BATCH_WAIT="${BATCH_WAIT:-0}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-5}"

OUT_DIR="${OUT_DIR:-experiments/fair_share/tsfm_victim_sweep/results/calibration}"
LOG_DIR="$OUT_DIR/logs"
OUT_CSV="$OUT_DIR/calibration.csv"
mkdir -p "$LOG_DIR"
rm -f "$OUT_CSV"

if command -v conda &> /dev/null; then
    CONDA_ENV="${CONDA_ENV:-fmtk}"
    PYTHON="${PYTHON:-conda run --no-capture-output -n ${CONDA_ENV} python}"
else
    PYTHON="${PYTHON:-python}"
fi

DEVICE_PID=""
stop_device() {
    if [[ -n "$DEVICE_PID" ]] && kill -0 "$DEVICE_PID" 2>/dev/null; then
        kill "$DEVICE_PID" 2>/dev/null || true
        local deadline=$((SECONDS + 12))
        while kill -0 "$DEVICE_PID" 2>/dev/null; do
            if [[ $SECONDS -ge $deadline ]]; then
                kill -9 "$DEVICE_PID" 2>/dev/null || true
                break
            fi
            sleep 0.5
        done
        wait "$DEVICE_PID" 2>/dev/null || true
    fi
    pkill -f "device/main.py.*--port ${DEVICE_PORT}" 2>/dev/null || true
    DEVICE_PID=""
    sleep 2
}
trap 'stop_device' EXIT

echo "================================================================"
echo "  Calibration: tpc_counts={${TPC_COUNTS[*]}}  tasks={${TASKS[*]}}"
echo "  offered_rps=$OFFERED_RPS  duration=${DURATION}s  warmup=${WARMUP_SECS}s"
echo "  output: $OUT_CSV"
echo "================================================================"

for TASK in "${TASKS[@]}"; do
    for N in "${TPC_COUNTS[@]}"; do
        echo ""
        echo "---- task=$TASK  tpcs=$N ----"
        stop_device

        TPC_PARTITION=$(seq -s ' ' 0 $((N - 1)))
        DEVICE_LOG="$LOG_DIR/device_${TASK}_tpc${N}.log"

        echo "[calib] starting device tpcs=[$TPC_PARTITION]"
        $PYTHON -u "$SERVING_DIR/device/main.py" \
            --port              "$DEVICE_PORT"  \
            --runtime-type      pytorch         \
            --cuda              "$CUDA_DEVICE"  \
            --scheduler-policy  fifo            \
            --max-batch-wait-ms "$BATCH_WAIT"   \
            --task-rates        "${TASK}:${OFFERED_RPS}" \
            --max-batch-size    "$BATCH_SIZE"   \
            --tpc-mode          "$TPC_MODE"     \
            --tpc-partition     $TPC_PARTITION  \
            --worker-mode       inline          \
            > "$DEVICE_LOG" 2>&1 &
        DEVICE_PID=$!
        sleep "$DEVICE_STARTUP_WAIT"

        $PYTHON -u "$SERVING_DIR/experiments/fair_share/tsfm_victim_sweep/calibrate.py" \
            --url         "localhost:${DEVICE_PORT}" \
            --backbone    "$BACKBONE"                \
            --task        "$TASK"                    \
            --tpc-count   "$N"                       \
            --offered-rps "$OFFERED_RPS"             \
            --duration    "$DURATION"                \
            --warmup-secs "$WARMUP_SECS"             \
            --out-csv     "$OUT_CSV"                 \
        || echo "[calib] WARNING: client failed for task=$TASK tpcs=$N"

        stop_device
    done
done

echo ""
echo "[calib] done. Results: $OUT_CSV"
