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
#   CONDA_ENV          fmtk (conda environment name)
#   FMTK_DIR           ../../../FMTK (relative path or absolute)
#   FMAAS_DIR          ../.. (relative path or absolute)
#   DECODER_DIR        ${FMTK_DIR}/models/tsfm/finetuned
#   CUDA_DEVICE        cuda:0
#   BACKBONE           momentbase
#   RPS_SWEEP          20,40,60
#   PHASE_DURATION     180
#   DEVICE_PORT        8000
#   DEVICE_PORT_2      8001
#   MAX_BATCH_SIZE     5
#   RESULTS_BASE       experiments/sharing_benefit/results
#   DEVICE_STARTUP_WAIT 5

set -euo pipefail

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SERVING_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SERVING_DIR"

# Conda environment for Python
CONDA_ENV="${CONDA_ENV:-fmtk}"

# Project directories (can be relative or absolute)
FMTK_DIR="${FMTK_DIR:-../../../FMTK}"
FMAAS_DIR="${FMAAS_DIR:-../..}"

# Convert to absolute paths if relative
if [[ ! "$FMTK_DIR" = /* ]]; then
    FMTK_DIR="$(cd "$SERVING_DIR" && cd "$FMTK_DIR" && pwd)"
fi
if [[ ! "$FMAAS_DIR" = /* ]]; then
    FMAAS_DIR="$(cd "$SERVING_DIR" && cd "$FMAAS_DIR" && pwd)"
fi

# Validate paths exist
if [[ ! -d "$FMTK_DIR" ]]; then
    echo "ERROR: FMTK_DIR not found at: $FMTK_DIR"
    echo "Set FMTK_DIR environment variable to correct path"
    exit 1
fi
if [[ ! -d "$FMAAS_DIR" ]]; then
    echo "ERROR: FMAAS_DIR not found at: $FMAAS_DIR"
    echo "Set FMAAS_DIR environment variable to correct path"
    exit 1
fi

# Set up PYTHONPATH
export PYTHONPATH="${FMTK_DIR}/src:${FMAAS_DIR}:${PYTHONPATH:-}"

# Python executable from conda environment
# Try conda run first, fall back to explicit PYTHON variable
if command -v conda &> /dev/null; then
    PYTHON="${PYTHON:-conda run -n ${CONDA_ENV} python}"
else
    # If conda not in PATH, try to find Python from environment
    PYTHON="${PYTHON:-python}"
fi

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:-momentbase}"
RPS_SWEEP="${RPS_SWEEP:-20,40,60}"
PHASE_DURATION="${PHASE_DURATION:-180}"
DEVICE_PORT="${DEVICE_PORT:-8000}"
DEVICE_PORT_2="${DEVICE_PORT_2:-8001}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-5}"
RESULTS_BASE="${RESULTS_BASE:-experiments/sharing_benefit/results}"
DECODER_DIR="${DECODER_DIR:-${FMTK_DIR}/models/tsfm/finetuned}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-5}"

LOG_DIR="${RESULTS_BASE}/logs"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  Motivation Experiment #2 — Sharing Benefit"
echo "  Conda env      : $CONDA_ENV"
echo "  FMTK_DIR       : $FMTK_DIR"
echo "  FMAAS_DIR      : $FMAAS_DIR"
echo "  Backbone       : $BACKBONE"
echo "  RPS sweep      : $RPS_SWEEP"
echo "  Duration/run   : ${PHASE_DURATION}s"
echo "  Results        : $RESULTS_BASE"
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
    $PYTHON -u "$SERVING_DIR/device/main.py" \
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
            $PYTHON -u experiments/sharing_benefit/run.py \
                --condition    single_ecgclass \
                --device-url   "localhost:${DEVICE_PORT}" \
                --backbone     "$BACKBONE" \
                --rps          "$rps" \
                --duration     "$PHASE_DURATION" \
                --exp-dir      "$out_dir"
            ;;
        single_gestureclass)
            DEVICE_PID=$(start_device "$DEVICE_PORT" "fifo" "$LOG_DIR/device_${condition}_rps${rps}.log" "$rps")
            $PYTHON -u experiments/sharing_benefit/run.py \
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
            $PYTHON -u experiments/sharing_benefit/run.py \
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
            $PYTHON -u experiments/sharing_benefit/run.py \
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
