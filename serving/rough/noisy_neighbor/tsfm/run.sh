#!/usr/bin/env bash
# noisy_neighbor/tsfm — Cross-task interference experiment
#
# Starts a fresh device server for each aggressor_rps point.
# Run from serving/:
#   bash experiments/noisy_neighbor/tsfm/run.sh

set -euo pipefail
SERVING_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$SERVING_DIR"

PYTHONPATH_EXTRA="/project/pi_shenoy_umass_edu/hshastri/FMTK/src:/project/pi_shenoy_umass_edu/hshastri/FMaaS-motivation"
export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"

DEVICE_PORT="${DEVICE_PORT:-8000}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:-momentbase}"
VICTIM_TASK="${VICTIM_TASK:-ecgclass}"
VICTIM_RPS="${VICTIM_RPS:-2}"
AGGRESSOR_TASK="${AGGRESSOR_TASK:-gestureclass}"
AGGRESSOR_RPS_SWEEP="${AGGRESSOR_RPS_SWEEP:-0,10,50,100,200,500}"
DURATION="${DURATION:-30}"
MAX_BATCH_WAIT_MS="${MAX_BATCH_WAIT_MS:-1}"
SCHEDULER_POLICY="${SCHEDULER_POLICY:-fifo}"
EXP_DIR="${EXP_DIR:-experiments/noisy_neighbor/tsfm/results/${SCHEDULER_POLICY}_bwait_${MAX_BATCH_WAIT_MS}ms}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-5}"
PYTHON="${PYTHON:-/home/hshastri_umass_edu/.conda/envs/fmtk/bin/python}"

LOG_DIR="experiments/noisy_neighbor/tsfm/logs"
mkdir -p "$LOG_DIR"

IFS=',' read -ra AGG_RPS_LIST <<< "$AGGRESSOR_RPS_SWEEP"

echo "================================================================"
echo "  noisy_neighbor/tsfm"
echo "  Backbone       : $BACKBONE"
echo "  Victim         : $VICTIM_TASK @ ${VICTIM_RPS} rps (fixed)"
echo "  Aggressor      : $AGGRESSOR_TASK @ [${AGG_RPS_LIST[*]}] rps (sweep)"
echo "  Duration       : ${DURATION}s per point"
echo "  BatchWait      : ${MAX_BATCH_WAIT_MS}ms"
echo "  Scheduler      : $SCHEDULER_POLICY"
echo "  Total points   : ${#AGG_RPS_LIST[@]}"
echo "================================================================"

start_device() {
    local log="$1"
    pkill -f "noisy_neighbor/tsfm/run.py" 2>/dev/null || true
    echo "[run.sh] Starting device server on port $DEVICE_PORT ..."
    "$PYTHON" -u "$SERVING_DIR/device/main.py" \
        --port "$DEVICE_PORT" \
        --runtime-type pytorch \
        --cuda "$CUDA_DEVICE" \
        --max-batch-wait-ms "$MAX_BATCH_WAIT_MS" \
        --scheduler-policy  "$SCHEDULER_POLICY" \
        > "$log" 2>&1 &
    DEVICE_PID=$!
    echo "[run.sh] Device PID=$DEVICE_PID  log=$log"
    echo "[run.sh] Waiting ${DEVICE_STARTUP_WAIT}s for server to be ready..."
    sleep "$DEVICE_STARTUP_WAIT"
}

stop_device() {
    if [[ -n "${DEVICE_PID:-}" ]]; then
        echo "[run.sh] Stopping device server (PID=$DEVICE_PID)"
        kill "$DEVICE_PID" 2>/dev/null || true
        wait "$DEVICE_PID" 2>/dev/null || true
        DEVICE_PID=""
    fi
    pkill -f "device/main.py.*--port ${DEVICE_PORT}" 2>/dev/null || true
    sleep 1
}

trap 'stop_device' EXIT

POINT=0
for agg_rps in "${AGG_RPS_LIST[@]}"; do
    POINT=$(( POINT + 1 ))
    echo ""
    echo "================================================================"
    echo "  Point $POINT / ${#AGG_RPS_LIST[@]}  aggressor_rps=$agg_rps"
    echo "================================================================"

    DEVICE_LOG="$LOG_DIR/device_agg_rps${agg_rps}.log"
    start_device "$DEVICE_LOG"

    "$PYTHON" -u experiments/noisy_neighbor/tsfm/run.py \
        --device-url          "localhost:${DEVICE_PORT}" \
        --backbone            "$BACKBONE"                \
        --victim-task         "$VICTIM_TASK"             \
        --victim-rps          "$VICTIM_RPS"              \
        --aggressor-task      "$AGGRESSOR_TASK"          \
        --aggressor-rps-sweep "$agg_rps"                 \
        --duration            "$DURATION"                \
        --exp-dir             "$EXP_DIR"

    stop_device
    sleep 2
done

echo ""
echo "[run.sh] All points done. Results in $EXP_DIR"
