#!/usr/bin/env bash
# noisy_neighbor/llm — Cross-task interference experiment
#
# Starts a fresh device server for each aggressor_rps point.
# Run from serving/:
#   bash experiments/noisy_neighbor/llm/run.sh

set -euo pipefail
SERVING_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$SERVING_DIR"

PYTHONPATH_EXTRA="/project/pi_shenoy_umass_edu/hshastri/FMTK/src:/project/pi_shenoy_umass_edu/hshastri/FMaaS-motivation"
export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"

DEVICE_PORT="${DEVICE_PORT:-8000}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:-qwen2.5-0.5b}"
VICTIM_TASK="${VICTIM_TASK:-sst2}"
VICTIM_RPS="${VICTIM_RPS:-2}"
AGGRESSOR_TASK="${AGGRESSOR_TASK:-conll2003}"
AGGRESSOR_RPS_SWEEP="${AGGRESSOR_RPS_SWEEP:-0,2,5,10,20,30}"
DURATION="${DURATION:-30}"
MAX_BATCH_WAIT_MS="${MAX_BATCH_WAIT_MS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-256}"
GPU_UTIL="${GPU_UTIL:-0.85}"
EXP_DIR="${EXP_DIR:-experiments/noisy_neighbor/llm/results/bwait_${MAX_BATCH_WAIT_MS}ms}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-15}"
PYTHON="${PYTHON:-/home/hshastri_umass_edu/.conda/envs/fmtk_vllm/bin/python}"

LOG_DIR="experiments/noisy_neighbor/llm/logs"
mkdir -p "$LOG_DIR"

IFS=',' read -ra AGG_RPS_LIST <<< "$AGGRESSOR_RPS_SWEEP"

echo "================================================================"
echo "  noisy_neighbor/llm"
echo "  Backbone       : $BACKBONE"
echo "  Victim         : $VICTIM_TASK @ ${VICTIM_RPS} rps (fixed)"
echo "  Aggressor      : $AGGRESSOR_TASK @ [${AGG_RPS_LIST[*]}] rps (sweep)"
echo "  Duration       : ${DURATION}s per point"
echo "  BatchWait      : ${MAX_BATCH_WAIT_MS}ms"
echo "  Total points   : ${#AGG_RPS_LIST[@]}"
echo "================================================================"

start_device() {
    local log="$1"
    pkill -f "noisy_neighbor/llm/run.py" 2>/dev/null || true
    echo "[run.sh] Starting device server on port $DEVICE_PORT ..."
    "$PYTHON" -u "$SERVING_DIR/device/main.py" \
        --port "$DEVICE_PORT" \
        --runtime-type vllm \
        --cuda "$CUDA_DEVICE" \
        --max-batch-wait-ms "$MAX_BATCH_WAIT_MS" \
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

    "$PYTHON" -u experiments/noisy_neighbor/llm/run.py \
        --device-url          "localhost:${DEVICE_PORT}" \
        --backbone            "$BACKBONE"                \
        --victim-task         "$VICTIM_TASK"             \
        --victim-rps          "$VICTIM_RPS"              \
        --aggressor-task      "$AGGRESSOR_TASK"          \
        --aggressor-rps-sweep "$agg_rps"                 \
        --duration            "$DURATION"                \
        --max-new-tokens      "$MAX_NEW_TOKENS"          \
        --max-model-len       "$MAX_MODEL_LEN"           \
        --gpu-util            "$GPU_UTIL"                \
        --exp-dir             "$EXP_DIR"

    stop_device
    sleep 2
done

echo ""
echo "[run.sh] All points done. Results in $EXP_DIR"
