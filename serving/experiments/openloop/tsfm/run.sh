#!/usr/bin/env bash
# motivation2/tsfm — Open-loop TSFM overhead experiment
#
# Loops over every (n_tasks, rps) pair. Each iteration starts a fresh device
# server, runs one RPS point, then kills the server before moving on.
#
# Run from serving/:
#   bash experiments/motivation2/tsfm/run.sh

set -euo pipefail
SERVING_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$SERVING_DIR"

PYTHONPATH_EXTRA="/project/pi_shenoy_umass_edu/hshastri/FMTK/src:/project/pi_shenoy_umass_edu/hshastri/FMaaS-motivation"
export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"

DEVICE_PORT="${DEVICE_PORT:-8000}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:-momentbase}"
N_TASKS="${N_TASKS:-1,2,4}"
RPS_SWEEP="${RPS_SWEEP:-2,5,10,20,30}"
DURATION="${DURATION:-30}"
EXP_DIR="${EXP_DIR:-experiments/motivation2/tsfm/results}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-5}"
PYTHON="${PYTHON:-/home/hshastri_umass_edu/.conda/envs/fmtk/bin/python}"

LOG_DIR="experiments/motivation2/tsfm/logs"
mkdir -p "$LOG_DIR"

# Convert comma-separated strings to arrays
IFS=',' read -ra TASK_LIST <<< "$N_TASKS"
IFS=',' read -ra RPS_LIST  <<< "$RPS_SWEEP"

echo "================================================================"
echo "  motivation2/tsfm"
echo "  Backbone : $BACKBONE"
echo "  N-tasks  : ${TASK_LIST[*]}"
echo "  RPS sweep: ${RPS_LIST[*]}"
echo "  Duration : ${DURATION}s per point"
echo "  Total    : $(( ${#TASK_LIST[@]} * ${#RPS_LIST[@]} )) points"
echo "================================================================"

# ---------------------------------------------------------------------------
# Helper: start device server, sets DEVICE_PID
# ---------------------------------------------------------------------------
start_device() {
    local log="$1"
    # Kill any leftover run.py processes from a previous interrupted run
    pkill -f "motivation2/tsfm/run.py" 2>/dev/null || true
    echo "[run.sh] Starting device server on port $DEVICE_PORT ..."
    "$PYTHON" -u "$SERVING_DIR/device/main.py" \
        --port "$DEVICE_PORT" \
        --runtime-type pytorch \
        --cuda "$CUDA_DEVICE" \
        > "$log" 2>&1 &
    DEVICE_PID=$!
    echo "[run.sh] Device PID=$DEVICE_PID  log=$log"
    echo "[run.sh] Waiting ${DEVICE_STARTUP_WAIT}s for server to be ready..."
    sleep "$DEVICE_STARTUP_WAIT"
}

# Helper: kill device server
stop_device() {
    if [[ -n "${DEVICE_PID:-}" ]]; then
        echo "[run.sh] Stopping device server (PID=$DEVICE_PID)"
        kill "$DEVICE_PID" 2>/dev/null || true
        wait "$DEVICE_PID" 2>/dev/null || true
        DEVICE_PID=""
    fi
    # Kill any other device/main.py processes still holding the port
    pkill -f "device/main.py.*--port ${DEVICE_PORT}" 2>/dev/null || true
    sleep 1
}

# Always kill on unexpected exit
trap 'stop_device' EXIT

# ---------------------------------------------------------------------------
# Main loop: one device server per (n_tasks, rps) point
# ---------------------------------------------------------------------------
POINT=0
for n_tasks in "${TASK_LIST[@]}"; do
    for rps in "${RPS_LIST[@]}"; do
        POINT=$(( POINT + 1 ))
        echo ""
        echo "================================================================"
        echo "  Point $POINT / $(( ${#TASK_LIST[@]} * ${#RPS_LIST[@]} ))"
        echo "  n_tasks=$n_tasks  rps=$rps"
        echo "================================================================"

        DEVICE_LOG="$LOG_DIR/device_n${n_tasks}_rps${rps}.log"
        start_device "$DEVICE_LOG"

        "$PYTHON" -u experiments/motivation2/tsfm/run.py \
            --device-url "localhost:${DEVICE_PORT}" \
            --backbone   "$BACKBONE"                \
            --n-tasks    "$n_tasks"                 \
            --rps-sweep  "$rps"                     \
            --duration   "$DURATION"                \
            --exp-dir    "$EXP_DIR"

        stop_device
        sleep 2   # brief pause between points
    done
done

echo ""
echo "[run.sh] All points done. Results in $EXP_DIR"
