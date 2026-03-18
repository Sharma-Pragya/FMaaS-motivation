#!/usr/bin/env bash
# isolation_overhead — Compare none / shared / process isolation modes.
#
# Starts a fresh device server per mode, runs closed-loop requests,
# appends results to a single summary.csv.
#
# Run from serving/:
#   bash experiments/isolation_overhead/run.sh
#
# Run a single mode:
#   ISOLATION_MODES="shared" bash experiments/isolation_overhead/run.sh

set -euo pipefail

SERVING_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SERVING_DIR"

export PYTHONPATH="/project/pi_shenoy_umass_edu/hshastri/FMTK/src:/project/pi_shenoy_umass_edu/hshastri/FMaaS-motivation:${PYTHONPATH:-}"

# ── Config ────────────────────────────────────────────────────────────────
ISOLATION_MODES="${ISOLATION_MODES:-none shared process}"
DEVICE_PORT="${DEVICE_PORT:-8000}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:-momentlarge}"
TASK="${TASK:-ecgclass}"
DURATION="${DURATION:-60}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
MAX_BATCH_WAIT_MS="${MAX_BATCH_WAIT_MS:-1}"
SCHEDULER_POLICY="${SCHEDULER_POLICY:-round_robin}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-10}"
EXP_DIR="${EXP_DIR:-experiments/isolation_overhead/results}"
PYTHON="${PYTHON:-/home/hshastri_umass_edu/.conda/envs/fmtk/bin/python}"

LOG_DIR="experiments/isolation_overhead/logs"
mkdir -p "$LOG_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  isolation_overhead experiment"
echo "════════════════════════════════════════════════════════════════"
echo "  Modes      : $ISOLATION_MODES"
echo "  Backbone   : $BACKBONE"
echo "  Task       : $TASK"
echo "  Duration   : ${DURATION}s"
echo "  Summary    : $EXP_DIR/summary.csv"
echo "════════════════════════════════════════════════════════════════"

DEVICE_PID=""

start_device() {
    local mode="$1"
    local log="$LOG_DIR/device_${mode}.log"
    echo "[INFO] Starting device (mode=$mode) on port $DEVICE_PORT ..."
    "$PYTHON" -u "$SERVING_DIR/device/main.py" \
        --port              "$DEVICE_PORT"       \
        --cuda              "$CUDA_DEVICE"       \
        --runtime-type      pytorch              \
        --max-batch-size    "$MAX_BATCH_SIZE"    \
        --max-batch-wait-ms "$MAX_BATCH_WAIT_MS" \
        --scheduler-policy  "$SCHEDULER_POLICY"  \
        --isolation-mode    "$mode"              \
        > "$log" 2>&1 &
    DEVICE_PID=$!
    echo "[INFO] Device PID=$DEVICE_PID  log=$log"
    sleep "$DEVICE_STARTUP_WAIT"
}

stop_device() {
    if [[ -n "${DEVICE_PID:-}" ]]; then
        echo "[INFO] Stopping device (PID=$DEVICE_PID)"
        kill "$DEVICE_PID" 2>/dev/null || true
        wait "$DEVICE_PID" 2>/dev/null || true
        DEVICE_PID=""
    fi
    pkill -f "device/main.py.*--port ${DEVICE_PORT}" 2>/dev/null || true
    sleep 2
}

trap 'stop_device' EXIT

for MODE in $ISOLATION_MODES; do
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "[RUN] isolation_mode=$MODE"
    echo "════════════════════════════════════════════════════════════════"

    start_device "$MODE"

    "$PYTHON" -u experiments/isolation_overhead/run.py \
        --device-url     "localhost:${DEVICE_PORT}" \
        --backbone       "$BACKBONE"                \
        --task           "$TASK"                    \
        --duration       "$DURATION"                \
        --isolation-mode "$MODE"                    \
        --exp-dir        "$EXP_DIR"

    stop_device
    echo "[INFO] [$MODE] done. Pausing 3s ..."
    sleep 3
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "[INFO] All modes complete."
echo "[INFO] Results: $EXP_DIR/summary.csv"
echo "════════════════════════════════════════════════════════════════"
