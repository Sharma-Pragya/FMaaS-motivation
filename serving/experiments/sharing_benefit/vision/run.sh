#!/bin/bash
# Vision Sharing Benefit Experiment
# Starts/stops device servers per condition and calls run.py.
#
# Conditions:
#   single_nyudepth  — 1 server (port A), nyudepth only, FIFO
#   single_vocseg    — 1 server (port A), vocseg only, FIFO
#   no_sharing           — 2 servers (port A + B), one backbone each, FIFO
#   sharing              — 1 server (port A), both tasks, STFQ
#
# Environment variables (all optional):
#   CONDA_ENV          fmtk (conda environment name)
#   FMTK_DIR           ../../../FMTK (relative path or absolute)
#   FMAAS_DIR          ../../.. (relative path or absolute)
#   DECODER_DIR        ${FMTK_DIR}/models/vision/finetuned
#   CUDA_DEVICE        cuda:0
#   BACKBONE           dinobase-patch
#   RPS_SWEEP          20,40,60
#   PHASE_DURATION     180
#   DEVICE_PORT        8000
#   DEVICE_PORT_2      8001
#   MAX_BATCH_SIZE     5
#   RESULTS_BASE       experiments/sharing_benefit/vision/results
#   DEVICE_STARTUP_WAIT 5

set -euo pipefail

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SERVING_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$SERVING_DIR"

# Conda environment for Python
CONDA_ENV="${CONDA_ENV:-fmtk}"

# Project directories (can be relative or absolute)
FMTK_DIR="${FMTK_DIR:-../../FMTK}"
FMAAS_DIR="${FMAAS_DIR:-..}"

# Convert to absolute paths if relative
if [[ ! "$FMTK_DIR" = /* ]]; then
    if [[ -d "$SERVING_DIR/$FMTK_DIR" ]]; then
        FMTK_DIR="$SERVING_DIR/$FMTK_DIR"
    elif [[ -d "$(dirname "$SERVING_DIR")/../FMTK" ]]; then
        FMTK_DIR="$(cd "$(dirname "$SERVING_DIR")/../FMTK" && pwd)"
    fi
fi

if [[ ! "$FMAAS_DIR" = /* ]]; then
    if [[ -d "$SERVING_DIR/$FMAAS_DIR" ]]; then
        FMAAS_DIR="$SERVING_DIR/$FMAAS_DIR"
    elif [[ -d "$(dirname "$SERVING_DIR")" ]]; then
        FMAAS_DIR="$(dirname "$SERVING_DIR")"
    fi
fi

# Validate paths exist
if [[ ! -d "$FMTK_DIR" ]]; then
    echo "ERROR: FMTK_DIR not found at: $FMTK_DIR"
    echo "Please set FMTK_DIR environment variable:"
    echo "  export FMTK_DIR=/path/to/FMTK"
    echo "  bash experiments/sharing_benefit/vision/run.sh"
    exit 1
fi
if [[ ! -d "$FMAAS_DIR" ]]; then
    echo "ERROR: FMAAS_DIR not found at: $FMAAS_DIR"
    echo "Please set FMAAS_DIR environment variable:"
    echo "  export FMAAS_DIR=/path/to/FMaaS-motivation"
    echo "  bash experiments/sharing_benefit/vision/run.sh"
    exit 1
fi

# Set up PYTHONPATH
export PYTHONPATH="${FMTK_DIR}/src:${FMAAS_DIR}:${PYTHONPATH:-}"

# Export dataset directory for FMTK to find datasets
export DATASET_DIR

# Python executable from conda environment
if command -v conda &> /dev/null; then
    PYTHON="${PYTHON:-conda run --no-capture-output -n ${CONDA_ENV} python}"
else
    PYTHON="${PYTHON:-python}"
fi

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:-dinobase-patch}"
RPS_SWEEP="${RPS_SWEEP:-5,10,15,20}"
PHASE_DURATION="${PHASE_DURATION:-20}"
DEVICE_PORT="${DEVICE_PORT:-8000}"
DEVICE_PORT_2="${DEVICE_PORT_2:-8001}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-5}"
RESULTS_BASE="${RESULTS_BASE:-experiments/sharing_benefit/vision/results}"
DECODER_DIR="${DECODER_DIR:-${FMTK_DIR}/models/vision/finetuned}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-5}"
MAX_BATCH_WAIT_MS="${MAX_BATCH_WAIT_MS:-0}"

LOG_DIR="${RESULTS_BASE}/logs"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  Vision Sharing Benefit Experiment"
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
    local task_rates="nyudepth:${rps},vocseg:${rps}"
    pkill -f "device/main.py.*--port ${port}" 2>/dev/null || true
    sleep 1
    echo "[run.sh] Starting device server port=$port scheduler=$scheduler ..."
    $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port              "$port"          \
        --runtime-type      pytorch          \
        --cuda              "$CUDA_DEVICE"   \
        --scheduler-policy  "$scheduler"     \
        --max-batch-size    "$MAX_BATCH_SIZE" \
        --max-batch-wait-ms "$MAX_BATCH_WAIT_MS"                 \
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
    local trace_file="${RESULTS_BASE}/rps_${rps}/trace.json"
    local scheduler=""

    echo ""
    echo "================================================================"
    echo "  condition=$condition"
    echo "  Results: $out_dir"
    echo "================================================================"

    stop_devices

    mkdir -p "$out_dir"

    case "$condition" in
        single_nyudepth)
            scheduler="stfq"
            DEVICE_PID=$(start_device "$DEVICE_PORT" "fifo" "$LOG_DIR/device_${condition}_rps${rps}.log" "$rps")
            $PYTHON -u experiments/sharing_benefit/vision/run.py \
                --condition    single_nyudepth \
                --device-url   "localhost:${DEVICE_PORT}" \
                --backbone     "$BACKBONE" \
                --rps          "$rps" \
                --duration     "$PHASE_DURATION" \
                --exp-dir      "$out_dir" \
                --trace-file   "$trace_file"
            ;;
        single_vocseg)
            scheduler="stfq"
            DEVICE_PID=$(start_device "$DEVICE_PORT" "fifo" "$LOG_DIR/device_${condition}_rps${rps}.log" "$rps")
            $PYTHON -u experiments/sharing_benefit/vision/run.py \
                --condition    single_vocseg \
                --device-url   "localhost:${DEVICE_PORT}" \
                --backbone     "$BACKBONE" \
                --rps          "$rps" \
                --duration     "$PHASE_DURATION" \
                --exp-dir      "$out_dir" \
                --trace-file   "$trace_file"
            ;;
        no_sharing)
            scheduler="stfq"
            DEVICE_PID=$(start_device   "$DEVICE_PORT"   "fifo" "$LOG_DIR/device_${condition}_1_rps${rps}.log" "$rps")
            DEVICE_PID_2=$(start_device "$DEVICE_PORT_2" "fifo" "$LOG_DIR/device_${condition}_2_rps${rps}.log" "$rps")
            $PYTHON -u experiments/sharing_benefit/vision/run.py \
                --condition    no_sharing \
                --device-url   "localhost:${DEVICE_PORT}" \
                --device-url-2 "localhost:${DEVICE_PORT_2}" \
                --backbone     "$BACKBONE" \
                --rps          "$rps" \
                --duration     "$PHASE_DURATION" \
                --exp-dir      "$out_dir" \
                --trace-file   "$trace_file"
            ;;
        sharing)
            scheduler="stfq"
            DEVICE_PID=$(start_device "$DEVICE_PORT" "stfq" "$LOG_DIR/device_${condition}_rps${rps}.log" "$rps")
            $PYTHON -u experiments/sharing_benefit/vision/run.py \
                --condition    sharing \
                --device-url   "localhost:${DEVICE_PORT}" \
                --backbone     "$BACKBONE" \
                --rps          "$rps" \
                --duration     "$PHASE_DURATION" \
                --exp-dir      "$out_dir" \
                --trace-file   "$trace_file"
            ;;
    esac

    cat > "${out_dir}/run_config.json" <<EOF
{
  "condition": "${condition}",
  "backbone": "${BACKBONE}",
  "cuda_device": "${CUDA_DEVICE}",
  "scheduler_policy": "${scheduler}",
  "max_batch_size": ${MAX_BATCH_SIZE},
  "max_batch_wait_ms": ${MAX_BATCH_WAIT_MS},
  "phase_duration_s": ${PHASE_DURATION},
  "rps_per_task": ${rps},
  "device_port": ${DEVICE_PORT},
  "device_port_2": ${DEVICE_PORT_2},
  "device_startup_wait_s": ${DEVICE_STARTUP_WAIT}
}
EOF

    stop_devices
}

# Sweep RPS values, run all four conditions per RPS
IFS=',' read -ra RPS_LIST <<< "$RPS_SWEEP"
for rps in "${RPS_LIST[@]}"; do
    echo ""
    echo "################################################################"
    echo "  RPS = $rps"
    echo "################################################################"
    for condition in single_nyudepth single_vocseg no_sharing sharing; do
    # for condition in single_vocseg no_sharing sharing; do
        run_condition "$condition" "$rps" \
            || echo "[run.sh] WARNING: $condition rps=$rps failed — continuing"
    done
done

echo ""
echo "[run.sh] All done. Results in $RESULTS_BASE"
