#!/usr/bin/env bash
# noisy_neighbor/tsfm_inaction — Time-series interference experiment
#
# Loops over all scheduler policies, running each with a fresh device server.
# Run from serving/:
#   bash experiments/noisy_neighbor/tsfm_inaction/run.sh
#
# Override policies:
#   SCHEDULERS=fifo,wfq bash experiments/noisy_neighbor/tsfm_inaction/run.sh
#
# Configure phases (comma-separated lists, must be same length):
#   AGGRESSOR_RPS_PHASES=20,30,50,150       # aggressor RPS per phase
#   PHASE_DURATIONS=30,30,30,30             # duration (s) per phase
#   PHASE_DURATIONS=30                      # single value → same duration for all phases

set -euo pipefail
SERVING_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$SERVING_DIR"

PYTHONPATH_EXTRA="/project/pi_shenoy_umass_edu/hshastri/FMTK/src:/project/pi_shenoy_umass_edu/hshastri/FMaaS-motivation"
export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"

DEVICE_PORT="${DEVICE_PORT:-8000}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:-momentbase}"
VICTIM_TASK="${VICTIM_TASK:-ecgclass}"
AGGRESSOR_TASK="${AGGRESSOR_TASK:-gestureclass}"

# Victim is constant across all phases
VICTIM_RPS="${VICTIM_RPS:-20}"
MAX_BATCH_WAIT_MS="${MAX_BATCH_WAIT_MS:-20}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-5}"

# Aggressor RPS per phase (comma-separated); number of entries = number of phases
AGGRESSOR_RPS_PHASES="${AGGRESSOR_RPS_PHASES:-20,30,50,150}"

# Duration per phase in seconds.
# Either a single value (same for all phases) or one value per phase.
PHASE_DURATIONS="${PHASE_DURATIONS:-30}"

SCHEDULERS="${SCHEDULERS:-fifo,stfq,round_robin}"
RESULTS_BASE="${RESULTS_BASE:-experiments/noisy_neighbor/tsfm_inaction/results}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-5}"
PYTHON="${PYTHON:-/home/hshastri_umass_edu/.conda/envs/fmtk/bin/python}"

# ---------------------------------------------------------------------------
# Resolve phase count and per-phase duration list
# ---------------------------------------------------------------------------
IFS=',' read -ra AGGRESSOR_RPS_LIST <<< "$AGGRESSOR_RPS_PHASES"
NUM_PHASES="${#AGGRESSOR_RPS_LIST[@]}"

IFS=',' read -ra RAW_DURATIONS <<< "$PHASE_DURATIONS"
if [[ "${#RAW_DURATIONS[@]}" -eq 1 ]]; then
    # Expand single value to all phases
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

# Peak aggressor RPS is the last phase value (used for WFQ weight calculation)
AGGRESSOR_RPS_PEAK="${AGGRESSOR_RPS_LIST[$(( NUM_PHASES - 1 ))]}"

# Rebuild canonical comma-separated strings for passing to run.py
PHASE_DURATIONS_CSV=$(IFS=','; echo "${DURATION_LIST[*]}")

LOG_DIR="${RESULTS_BASE}/logs"
mkdir -p "$LOG_DIR"

IFS=',' read -ra SCHEDULER_LIST <<< "$SCHEDULERS"

# write the config to a file for record-keeping
CONFIG_FILE="$RESULTS_BASE/config.txt"
mkdir -p "$(dirname "$CONFIG_FILE")"
{
    echo "Experiment config - $(date)"
    echo "Backbone: $BACKBONE"
    echo "Victim: $VICTIM_TASK @ ${VICTIM_RPS} rps (constant)"
    echo "Aggressor: $AGGRESSOR_TASK"
    echo "Number of phases: $NUM_PHASES"
    for (( i=0; i<NUM_PHASES; i++ )); do
        echo "  Phase $(( i+1 )) (${DURATION_LIST[$i]}s): aggressor @ ${AGGRESSOR_RPS_LIST[$i]} rps"
    done
    echo "Total duration: ${TOTAL_DURATION}s"
    echo "Schedulers: ${SCHEDULER_LIST[*]}"
    echo "BatchWait: ${MAX_BATCH_WAIT_MS}ms"
    echo "Max batch size: ${MAX_BATCH_SIZE}"
    echo "Results base: $RESULTS_BASE"
} > "$CONFIG_FILE"

echo "================================================================"
echo "  noisy_neighbor/tsfm_inaction — ${NUM_PHASES}-phase experiment"
echo "  Backbone     : $BACKBONE"
echo "  Victim       : $VICTIM_TASK @ ${VICTIM_RPS} rps (constant)"
echo "  Aggressor    : $AGGRESSOR_TASK"
for (( i=0; i<NUM_PHASES; i++ )); do
    echo "  Phase $(( i+1 )) (${DURATION_LIST[$i]}s): aggressor @ ${AGGRESSOR_RPS_LIST[$i]} rps"
done
echo "  Total        : ${TOTAL_DURATION}s"
echo "  Schedulers   : ${SCHEDULER_LIST[*]}"
echo "  BatchWait    : ${MAX_BATCH_WAIT_MS}ms"
echo "  Results base : $RESULTS_BASE"
echo "================================================================"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEVICE_PID=""

stop_device() {
    if [[ -n "${DEVICE_PID:-}" ]]; then
        echo "[run.sh] Stopping device server (PID=$DEVICE_PID)"
        kill "$DEVICE_PID" 2>/dev/null || true
        wait "$DEVICE_PID" 2>/dev/null || true
        DEVICE_PID=""
    fi
    # Kill any other device/main.py still holding the port
    pkill -f "device/main.py.*--port ${DEVICE_PORT}" 2>/dev/null || true
    sleep 2   # give OS time to release port
}
trap 'stop_device' EXIT

start_device() {
    local scheduler="$1" log="$2"
    # local task_rates="${VICTIM_TASK}:${VICTIM_RPS},${AGGRESSOR_TASK}:${AGGRESSOR_RPS_PEAK}"
    #set equal rates for victim and aggressor to stress test scheduler behavior
    local task_rates="${VICTIM_TASK}:${VICTIM_RPS},${AGGRESSOR_TASK}:${VICTIM_RPS}"
    # Clean up any leftover processes before starting fresh
    pkill -f "device/main.py.*--port ${DEVICE_PORT}" 2>/dev/null || true
    pkill -f "tsfm_inaction/run.py" 2>/dev/null || true
    sleep 1
    echo "[run.sh] Starting device server (scheduler=$scheduler, bwait=${MAX_BATCH_WAIT_MS}ms, rates=$task_rates)..."
    "$PYTHON" -u "$SERVING_DIR/device/main.py" \
        --port              "$DEVICE_PORT"       \
        --runtime-type      pytorch              \
        --cuda              "$CUDA_DEVICE"       \
        --scheduler-policy  "$scheduler"         \
        --max-batch-wait-ms "$MAX_BATCH_WAIT_MS" \
        --task-rates        "$task_rates"        \
        --max-batch-size      "$MAX_BATCH_SIZE"    \
        > "$log" 2>&1 &
    DEVICE_PID=$!
    echo "[run.sh] Device PID=$DEVICE_PID  log=$log"
    echo "[run.sh] Waiting ${DEVICE_STARTUP_WAIT}s for server to be ready..."
    sleep "$DEVICE_STARTUP_WAIT"
}

# ---------------------------------------------------------------------------
# Main loop — one fresh device server per scheduler policy
# ---------------------------------------------------------------------------
TOTAL=${#SCHEDULER_LIST[@]}
IDX=0

for SCHEDULER in "${SCHEDULER_LIST[@]}"; do
    IDX=$(( IDX + 1 ))
    EXP_DIR="${RESULTS_BASE}/${SCHEDULER}"
    DEVICE_LOG="$LOG_DIR/device_${SCHEDULER}.log"

    echo ""
    echo "================================================================"
    echo "  [$IDX/$TOTAL] scheduler=$SCHEDULER"
    echo "  Results: $EXP_DIR"
    echo "================================================================"

    start_device "$SCHEDULER" "$DEVICE_LOG"

    "$PYTHON" -u experiments/noisy_neighbor/tsfm_inaction/run.py \
        --device-url            "localhost:${DEVICE_PORT}"   \
        --backbone              "$BACKBONE"                  \
        --victim-task           "$VICTIM_TASK"               \
        --aggressor-task        "$AGGRESSOR_TASK"            \
        --victim-rps            "$VICTIM_RPS"                \
        --aggressor-rps-phases  "$AGGRESSOR_RPS_PHASES"      \
        --phase-durations       "$PHASE_DURATIONS_CSV"       \
        --scheduler-policy      "$SCHEDULER"                 \
        --exp-dir               "$EXP_DIR"                   \
    || echo "[run.sh] WARNING: run.py failed for scheduler=$SCHEDULER — continuing"

    stop_device
done

echo ""
echo "[run.sh] All $TOTAL schedulers done. Results in $RESULTS_BASE"
