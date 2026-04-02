#!/usr/bin/env bash
# noisy_neighbor/llm — LLM noisy-neighbor experiment on vLLM runtime
#
# Runs a shared vLLM device server and replays a multi-phase open-loop trace:
#   victim:     constant low-rate task
#   aggressor:  ramps across phases
#
# This is intentionally a continuous-batching baseline. The current vLLM
# runtime path does not apply the PyTorch DeviceBatcher fairness policies.
#
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
VICTIM_TASK="${VICTIM_TASK:-llm_sst2}"
AGGRESSOR_TASK="${AGGRESSOR_TASK:-llm_ag_news}"
VICTIM_RPS="${VICTIM_RPS:-10}"
AGGRESSOR_RPS_PHASES="${AGGRESSOR_RPS_PHASES:-10,70,15}"
PHASE_DURATIONS="${PHASE_DURATIONS:-5,10,5}"
RESULTS_BASE="${RESULTS_BASE:-experiments/noisy_neighbor/llm/results}"
RUN_NAME="${RUN_NAME:-continuous_batching}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-8}"
PYTHON="${PYTHON:-/home/hshastri_umass_edu/.conda/envs/fmtk_vllm/bin/python}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.42}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-256}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"

IFS=',' read -ra AGGRESSOR_RPS_LIST <<< "$AGGRESSOR_RPS_PHASES"
NUM_PHASES="${#AGGRESSOR_RPS_LIST[@]}"

IFS=',' read -ra RAW_DURATIONS <<< "$PHASE_DURATIONS"
if [[ "${#RAW_DURATIONS[@]}" -eq 1 ]]; then
    DURATION_LIST=()
    for (( i=0; i<NUM_PHASES; i++ )); do
        DURATION_LIST+=("${RAW_DURATIONS[0]}")
    done
else
    DURATION_LIST=("${RAW_DURATIONS[@]}")
    if [[ "${#DURATION_LIST[@]}" -ne "$NUM_PHASES" ]]; then
        echo "ERROR: PHASE_DURATIONS has ${#DURATION_LIST[@]} entries but AGGRESSOR_RPS_PHASES has ${NUM_PHASES}." >&2
        exit 1
    fi
fi

TOTAL_DURATION=0
for d in "${DURATION_LIST[@]}"; do
    TOTAL_DURATION=$(( TOTAL_DURATION + d ))
done

PHASE_DURATIONS_CSV=$(IFS=','; echo "${DURATION_LIST[*]}")
LOG_DIR="${RESULTS_BASE}/logs"
mkdir -p "$LOG_DIR"

CONFIG_FILE="$RESULTS_BASE/config.txt"
mkdir -p "$(dirname "$CONFIG_FILE")"
{
    echo "Experiment config - $(date)"
    echo "Runtime: vllm"
    echo "Backbone: $BACKBONE"
    echo "Victim: $VICTIM_TASK @ ${VICTIM_RPS} rps (constant)"
    echo "Aggressor: $AGGRESSOR_TASK"
    echo "Policy label: $RUN_NAME"
    echo "gpu_memory_utilization: $GPU_MEMORY_UTILIZATION"
    echo "max_model_len: $MAX_MODEL_LEN"
    echo "Number of phases: $NUM_PHASES"
    for (( i=0; i<NUM_PHASES; i++ )); do
        echo "  Phase $(( i+1 )) (${DURATION_LIST[$i]}s): aggressor @ ${AGGRESSOR_RPS_LIST[$i]} rps"
    done
    echo "Total duration: ${TOTAL_DURATION}s"
    echo "Results base: $RESULTS_BASE"
} > "$CONFIG_FILE"

echo "================================================================"
echo "  noisy_neighbor/llm — vLLM continuous-batching baseline"
echo "  Backbone     : $BACKBONE"
echo "  Victim       : $VICTIM_TASK @ ${VICTIM_RPS} rps (constant)"
echo "  Aggressor    : $AGGRESSOR_TASK"
for (( i=0; i<NUM_PHASES; i++ )); do
    echo "  Phase $(( i+1 )) (${DURATION_LIST[$i]}s): aggressor @ ${AGGRESSOR_RPS_LIST[$i]} rps"
done
echo "  Total        : ${TOTAL_DURATION}s"
echo "  Run name     : $RUN_NAME"
echo "  Results base : $RESULTS_BASE"
echo "================================================================"

DEVICE_PID=""

stop_device() {
    if [[ -n "${DEVICE_PID:-}" ]]; then
        echo "[run.sh] Stopping device server (PID=$DEVICE_PID)"
        kill "$DEVICE_PID" 2>/dev/null || true
        wait "$DEVICE_PID" 2>/dev/null || true
        DEVICE_PID=""
    fi
    pkill -f "device/main.py.*--port ${DEVICE_PORT}" 2>/dev/null || true
    sleep 2
}
trap 'stop_device' EXIT

DEVICE_LOG="$LOG_DIR/device_${RUN_NAME}.log"
EXP_DIR="${RESULTS_BASE}/${RUN_NAME}"

pkill -f "device/main.py.*--port ${DEVICE_PORT}" 2>/dev/null || true
pkill -f "noisy_neighbor/llm/run.py" 2>/dev/null || true
sleep 1

echo "[run.sh] Starting vLLM device server..."
"$PYTHON" -u "$SERVING_DIR/device/main.py" \
    --port "$DEVICE_PORT" \
    --runtime-type vllm \
    --cuda "$CUDA_DEVICE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    > "$DEVICE_LOG" 2>&1 &
DEVICE_PID=$!
echo "[run.sh] Device PID=$DEVICE_PID  log=$DEVICE_LOG"
echo "[run.sh] Waiting ${DEVICE_STARTUP_WAIT}s for server to be ready..."
sleep "$DEVICE_STARTUP_WAIT"

"$PYTHON" -u experiments/noisy_neighbor/llm/run.py \
    --device-url "localhost:${DEVICE_PORT}" \
    --backbone "$BACKBONE" \
    --victim-task "$VICTIM_TASK" \
    --aggressor-task "$AGGRESSOR_TASK" \
    --victim-rps "$VICTIM_RPS" \
    --aggressor-rps-phases "$AGGRESSOR_RPS_PHASES" \
    --phase-durations "$PHASE_DURATIONS_CSV" \
    --policy-label "$RUN_NAME" \
    --max-samples "$MAX_SAMPLES" \
    --exp-dir "$EXP_DIR"

stop_device
echo ""
echo "[run.sh] Run complete. Results in $EXP_DIR"
