#!/bin/bash
# Motivation Experiment #2 — Sharing Benefit
# Starts/stops device servers per condition and calls run.py.
#
# Conditions:
#   single_ecgclass     — 1 server (port A), ecgclass only, FIFO
#   single_gestureclass — 1 server (port A), gestureclass only, FIFO
#   no_sharing          — 2 servers (port A + B), one backbone each, FIFO
#   sharing             — 1 server (port A), both tasks, STFQ
#
# Environment variables (all optional):
#   CUDA_DEVICE       cuda:0
#   BACKBONE          momentbase
#   RPS               20
#   PHASE_DURATION    180
#   DEVICE_PORT       8000
#   DEVICE_PORT_2     8001
#   MAX_BATCH_SIZE    5
#   RESULTS_BASE      experiments/motivation2/results
#   DECODER_DIR       /project/pi_shenoy_umass_edu/hshastri/FMTK/models/tsfm/finetuned

set -euo pipefail
SERVING_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SERVING_DIR"

PYTHONPATH_EXTRA="/project/pi_shenoy_umass_edu/hshastri/FMTK/src:/project/pi_shenoy_umass_edu/hshastri/FMaaS-motivation"
export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"

CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:-momentbase}"
RPS_SWEEP="${RPS_SWEEP:-10,20,30,40,60,80}"
PHASE_DURATION="${PHASE_DURATION:-180}"
DEVICE_PORT="${DEVICE_PORT:-8000}"
DEVICE_PORT_2="${DEVICE_PORT_2:-8001}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-5}"
RESULTS_BASE="${RESULTS_BASE:-experiments/motivation2/results}"
PYTHON="${PYTHON:-/home/hshastri_umass_edu/.conda/envs/fmtk/bin/python}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-5}"

LOG_DIR="${RESULTS_BASE}/logs"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  Motivation Experiment #2 — Sharing Benefit"
echo "  Backbone      : $BACKBONE"
echo "  RPS sweep     : $RPS_SWEEP"
echo "  Duration/run  : ${PHASE_DURATION}s"
echo "  Results       : $RESULTS_BASE"
echo "================================================================"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEVICE_PID=""
DEVICE_PID_2=""

stop_devices() {
    for pid_var in DEVICE_PID DEVICE_PID_2; do
        local pid="${!pid_var:-}"
        if [[ -n "$pid" ]]; then
            echo "[run.sh] Stopping device server PID=$pid"
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    DEVICE_PID=""
    DEVICE_PID_2=""
    pkill -f "device/main.py.*--port ${DEVICE_PORT}"  2>/dev/null || true
    pkill -f "device/main.py.*--port ${DEVICE_PORT_2}" 2>/dev/null || true
    sleep 2
}
trap 'stop_devices' EXIT

start_device() {
    local port="$1" scheduler="$2" log="$3" rps="$4"
    local task_rates="ecgclass:${rps},gestureclass:${rps}"
    pkill -f "device/main.py.*--port ${port}" 2>/dev/null || true
    sleep 1
    echo "[run.sh] Starting device server port=$port scheduler=$scheduler ..."
    "$PYTHON" -u "$SERVING_DIR/device/main.py" \
        --port              "$port"          \
        --runtime-type      pytorch          \
        --cuda              "$CUDA_DEVICE"   \
        --scheduler-policy  "$scheduler"     \
        --max-batch-size    "$MAX_BATCH_SIZE" \
        --max-batch-wait-ms 0                \
        --task-rates        "$task_rates"    \
        > "$log" 2>&1 &
    local pid=$!
    echo "[run.sh] PID=$pid  log=$log"
    sleep "$DEVICE_STARTUP_WAIT"
    echo "$pid"
}

run_condition() {
    local condition="$1" rps="$2"
    local out_dir="${RESULTS_BASE}/rps_${rps}/${condition}"

    echo ""
    echo "================================================================"
    echo "  condition=$condition"
    echo "  Results: $out_dir"
    echo "================================================================"

    stop_devices

    case "$condition" in
        single_ecgclass)
            DEVICE_PID=$(start_device "$DEVICE_PORT" "fifo" "$LOG_DIR/device_${condition}_rps${rps}.log" "$rps")
            "$PYTHON" -u experiments/motivation2/run.py \
                --condition    single_ecgclass \
                --device-url   "localhost:${DEVICE_PORT}" \
                --backbone     "$BACKBONE" \
                --rps          "$rps" \
                --duration     "$PHASE_DURATION" \
                --exp-dir      "$out_dir"
            ;;
        single_gestureclass)
            DEVICE_PID=$(start_device "$DEVICE_PORT" "fifo" "$LOG_DIR/device_${condition}_rps${rps}.log" "$rps")
            "$PYTHON" -u experiments/motivation2/run.py \
                --condition    single_gestureclass \
                --device-url   "localhost:${DEVICE_PORT}" \
                --backbone     "$BACKBONE" \
                --rps          "$rps" \
                --duration     "$PHASE_DURATION" \
                --exp-dir      "$out_dir"
            ;;
        no_sharing)
            DEVICE_PID=$(start_device   "$DEVICE_PORT"   "fifo" "$LOG_DIR/device_${condition}_1_rps${rps}.log" "$rps")
            DEVICE_PID_2=$(start_device "$DEVICE_PORT_2" "fifo" "$LOG_DIR/device_${condition}_2_rps${rps}.log" "$rps")
            "$PYTHON" -u experiments/sharing_benefit/run.py \
                --condition    no_sharing \
                --device-url   "localhost:${DEVICE_PORT}" \
                --device-url-2 "localhost:${DEVICE_PORT_2}" \
                --backbone     "$BACKBONE" \
                --rps          "$rps" \
                --duration     "$PHASE_DURATION" \
                --exp-dir      "$out_dir"
            ;;
        sharing)
            DEVICE_PID=$(start_device "$DEVICE_PORT" "stfq" "$LOG_DIR/device_${condition}_rps${rps}.log" "$rps")
            "$PYTHON" -u experiments/motivation2/run.py \
                --condition    sharing \
                --device-url   "localhost:${DEVICE_PORT}" \
                --backbone     "$BACKBONE" \
                --rps          "$rps" \
                --duration     "$PHASE_DURATION" \
                --exp-dir      "$out_dir"
            ;;
    esac

    stop_devices
}

# Sweep RPS values, run all four conditions per RPS
IFS=',' read -ra RPS_LIST <<< "$RPS_SWEEP"
for rps in "${RPS_LIST[@]}"; do
    echo ""
    echo "################################################################"
    echo "  RPS = $rps"
    echo "################################################################"
    for condition in single_ecgclass single_gestureclass no_sharing sharing; do
        run_condition "$condition" "$rps" \
            || echo "[run.sh] WARNING: $condition rps=$rps failed — continuing"
    done
done

echo ""
echo "[run.sh] All done. Results in $RESULTS_BASE"
