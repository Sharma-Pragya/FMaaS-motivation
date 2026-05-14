#!/usr/bin/env bash
# lora_adapter_fraction — sweep over # of LoRA-adapted tasks at fixed total N.
#
# Always deploys NUM_TASKS replicas on a single shared backbone. For each
# value K in NUM_ADAPTED_LIST, the first K replicas get a LoRA adapter and
# the remaining (NUM_TASKS - K) use only the MLP decoder. Measures aggregate
# closed-loop throughput so we can see how adapter-swap cost grows with K.
#
# Run from serving/:
#   bash experiments/lora_adapter_fraction/run.sh
#
# Common overrides:
#   NUM_TASKS=10 NUM_ADAPTED_LIST="0,1,2,4,6,8,10" DURATION=60 \
#       bash experiments/lora_adapter_fraction/run.sh

set -euo pipefail

SERVING_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SERVING_DIR"

# ---------------------------------------------------------------------------
# Path / python setup (matches lora_ntasks_throughput/run.sh)
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-fmtk}"
FMTK_DIR="${FMTK_DIR:-../FMTK}"
FMAAS_DIR="${FMAAS_DIR:-..}"

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
[[ -d "$FMTK_DIR"  ]] || { echo "ERROR: FMTK_DIR not found: $FMTK_DIR";   exit 1; }
[[ -d "$FMAAS_DIR" ]] || { echo "ERROR: FMAAS_DIR not found: $FMAAS_DIR"; exit 1; }

export PYTHONPATH="${FMTK_DIR}/src:${FMAAS_DIR}:${PYTHONPATH:-}"

if [[ -n "${PYTHON:-}" ]]; then
    read -ra PYTHON <<< "$PYTHON"
elif command -v conda &> /dev/null; then
    PYTHON=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
    PYTHON=(python)
fi

# ---------------------------------------------------------------------------
# Experiment knobs
# ---------------------------------------------------------------------------
DEVICE_PORT="${DEVICE_PORT:-8000}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:-momentlarge}"
BASE_TASK="${BASE_TASK:-ecgclass}"

NUM_TASKS="${NUM_TASKS:-10}"
NUM_ADAPTED_LIST="${NUM_ADAPTED_LIST:-0,2,4,6,8,10}"  # Comma-separated values of K (num adapted) to sweep.

CONCURRENCY_PER_TASK="${CONCURRENCY_PER_TASK:-1}"
DURATION="${DURATION:-60}"
WARMUP_SECS="${WARMUP_SECS:-5}"

SCHEDULER="${SCHEDULER:-stfq}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BATCH_WAIT_MS="${BATCH_WAIT_MS:-0}"

RESULTS_BASE="${RESULTS_BASE:-experiments/lora_adapter_fraction/results}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-5}"

LOG_DIR="${RESULTS_BASE}/logs"
mkdir -p "$LOG_DIR"

IFS=',' read -ra K_LIST <<< "$NUM_ADAPTED_LIST"

CONFIG_FILE="$RESULTS_BASE/config.txt"
{
    echo "Experiment config - $(date)"
    echo "Backbone           : $BACKBONE"
    echo "Base task          : $BASE_TASK"
    echo "Total tasks (N)    : $NUM_TASKS"
    echo "Adapted (K) sweep  : ${K_LIST[*]}"
    echo "Concurrency / task : $CONCURRENCY_PER_TASK"
    echo "Duration           : ${DURATION}s (warmup=${WARMUP_SECS}s)"
    echo "Scheduler          : $SCHEDULER (bsize=$BATCH_SIZE, bwait=${BATCH_WAIT_MS}ms)"
    echo "Results base       : $RESULTS_BASE"
} > "$CONFIG_FILE"

echo "================================================================"
echo "  lora_adapter_fraction — closed-loop sweep"
echo "  Backbone           : $BACKBONE"
echo "  Base task          : $BASE_TASK"
echo "  Total tasks (N)    : $NUM_TASKS"
echo "  Adapted (K) sweep  : ${K_LIST[*]}"
echo "  Concurrency / task : $CONCURRENCY_PER_TASK"
echo "  Duration           : ${DURATION}s (warmup=${WARMUP_SECS}s)"
echo "  Scheduler          : $SCHEDULER"
echo "  Results base       : $RESULTS_BASE"
echo "================================================================"

# ---------------------------------------------------------------------------
# Device lifecycle (one server per K, since the deployed task set changes)
# ---------------------------------------------------------------------------
DEVICE_PID=""

stop_device() {
    if [[ -n "${DEVICE_PID:-}" ]]; then
        echo "[run.sh] Stopping device server (PID=$DEVICE_PID)"
        kill "$DEVICE_PID" 2>/dev/null || true
        wait "$DEVICE_PID" 2>/dev/null || true
        DEVICE_PID=""
    fi
    pkill -f "device/main.py.*--port ${DEVICE_PORT}\b" 2>/dev/null || true
    sleep 2
}
trap 'stop_device' EXIT

start_device() {
    local log="$1"
    pkill -f "device/main.py.*--port ${DEVICE_PORT}\b" 2>/dev/null || true
    pkill -f "lora_adapter_fraction/run.py" 2>/dev/null || true
    sleep 1
    echo "[run.sh] Starting device server (scheduler=$SCHEDULER bsize=$BATCH_SIZE bwait=${BATCH_WAIT_MS}ms)"
    "${PYTHON[@]}" -u "$SERVING_DIR/device/main.py" \
        --port              "$DEVICE_PORT"   \
        --runtime-type      pytorch          \
        --cuda              "$CUDA_DEVICE"   \
        --scheduler-policy  "$SCHEDULER"     \
        --max-batch-wait-ms "$BATCH_WAIT_MS" \
        --max-batch-size    "$BATCH_SIZE"    \
        > "$log" 2>&1 &
    DEVICE_PID=$!
    echo "[run.sh] Device PID=$DEVICE_PID  log=$log"
    sleep "$DEVICE_STARTUP_WAIT"
}

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
TOTAL="${#K_LIST[@]}"
IDX=0
for K in "${K_LIST[@]}"; do
    IDX=$(( IDX + 1 ))
    EXP_DIR="${RESULTS_BASE}/K${K}"
    DEVICE_LOG="${LOG_DIR}/device_K${K}.log"

    echo ""
    echo "================================================================"
    echo "  [$IDX/$TOTAL] K=$K adapted / $NUM_TASKS total tasks"
    echo "  Results: $EXP_DIR"
    echo "================================================================"

    start_device "$DEVICE_LOG"

    "${PYTHON[@]}" -u experiments/lora_adapter_fraction/run.py \
        --device-url           "localhost:${DEVICE_PORT}" \
        --backbone             "$BACKBONE"               \
        --base-task            "$BASE_TASK"              \
        --num-tasks            "$NUM_TASKS"              \
        --num-adapted          "$K"                      \
        --concurrency-per-task "$CONCURRENCY_PER_TASK"   \
        --duration             "$DURATION"               \
        --warmup-secs          "$WARMUP_SECS"            \
        --exp-dir              "$EXP_DIR"                \
    || echo "[run.sh] WARNING: run.py failed for K=$K — continuing"

    stop_device
done

# ---------------------------------------------------------------------------
# Aggregate sweep summary
# ---------------------------------------------------------------------------
SWEEP_CSV="${RESULTS_BASE}/sweep_summary.csv"
echo "num_adapted,num_tasks,aggregate_throughput_rps,avg_latency_ms_all,p95_latency_ms_all,p99_latency_ms_all,avg_latency_ms_adapted,avg_latency_ms_plain" > "$SWEEP_CSV"
for K in "${K_LIST[@]}"; do
    S="${RESULTS_BASE}/K${K}/summary.json"
    if [[ -f "$S" ]]; then
        "${PYTHON[@]}" - "$S" "$K" >> "$SWEEP_CSV" <<'PY'
import json, sys
p, k = sys.argv[1], sys.argv[2]
with open(p) as f:
    d = json.load(f)
print(",".join(str(x) for x in [
    k,
    d.get("num_tasks"),
    d.get("aggregate_throughput_rps"),
    d.get("avg_latency_ms_all"),
    d.get("p95_latency_ms_all"),
    d.get("p99_latency_ms_all"),
    d.get("avg_latency_ms_adapted"),
    d.get("avg_latency_ms_plain"),
]))
PY
    fi
done
echo ""
echo "[run.sh] Sweep summary → $SWEEP_CSV"
cat "$SWEEP_CSV"
