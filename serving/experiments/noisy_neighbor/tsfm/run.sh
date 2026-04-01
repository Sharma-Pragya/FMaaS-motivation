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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SERVING_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$SERVING_DIR"

# Project directories (can be relative or absolute)
# Default: FMTK is sibling directory of FMaaS-motivation, FMAAS_DIR is parent of serving
FMTK_DIR="${FMTK_DIR:-../../FMTK}"
FMAAS_DIR="${FMAAS_DIR:-..}"

# Convert to absolute paths if relative
if [[ ! "$FMTK_DIR" = /* ]]; then
    # Try relative to SERVING_DIR first
    if [[ -d "$SERVING_DIR/$FMTK_DIR" ]]; then
        FMTK_DIR="$SERVING_DIR/$FMTK_DIR"
    # Try from workspace root (one level up from FMaaS-motivation)
    elif [[ -d "$(dirname "$SERVING_DIR")/../FMTK" ]]; then
        FMTK_DIR="$(cd "$(dirname "$SERVING_DIR")/../FMTK" && pwd)"
    fi
fi

if [[ ! "$FMAAS_DIR" = /* ]]; then
    # Try relative to SERVING_DIR first
    if [[ -d "$SERVING_DIR/$FMAAS_DIR" ]]; then
        FMAAS_DIR="$SERVING_DIR/$FMAAS_DIR"
    # Try parent directory
    elif [[ -d "$(dirname "$SERVING_DIR")" ]]; then
        FMAAS_DIR="$(dirname "$SERVING_DIR")"
    fi
fi

# Validate paths exist
if [[ ! -d "$FMTK_DIR" ]]; then
    echo "ERROR: FMTK_DIR not found at: $FMTK_DIR"
    echo "Please set FMTK_DIR environment variable:"
    echo "  export FMTK_DIR=/path/to/FMTK"
    echo "  bash experiments/noisy_neighbor/tsfm/run.sh"
    exit 1
fi
if [[ ! -d "$FMAAS_DIR" ]]; then
    echo "ERROR: FMAAS_DIR not found at: $FMAAS_DIR"
    echo "Please set FMAAS_DIR environment variable:"
    echo "  export FMAAS_DIR=/path/to/FMaaS-motivation"
    echo "  bash experiments/noisy_neighbor/tsfm/run.sh"
    exit 1
fi

# Set up PYTHONPATH
export PYTHONPATH="${FMTK_DIR}/src:${FMAAS_DIR}:${PYTHONPATH:-}"

DEVICE_PORT="${DEVICE_PORT:-8000}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:-momentbase}"
VICTIM_TASK="${VICTIM_TASK:-ecgclass}"
AGGRESSOR_TASK="${AGGRESSOR_TASK:-gestureclass}"

# Victim is constant across all phases
VICTIM_RPS="${VICTIM_RPS:-20}"

# Aggressor RPS per phase (comma-separated); number of entries = number of phases
AGGRESSOR_RPS_PHASES="${AGGRESSOR_RPS_PHASES:-20,150,30}"

# Duration per phase in seconds.
# Either a single value (same for all phases) or one value per phase.
PHASE_DURATIONS="${PHASE_DURATIONS:-30,10,30}"

# Runs: "scheduler  batch_size  batch_wait_ms  run_name"
RUNS=(
    "fifo  3  0  fcfs"
    "stfq  1  0  stfq"
    "stfq  3  0  bfq"
)

RESULTS_BASE="${RESULTS_BASE:-experiments/noisy_neighbor/tsfm/results}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-5}"

# Python executable from conda environment
# Try conda run first, fall back to explicit PYTHON variable
if command -v conda &> /dev/null; then
    CONDA_ENV="${CONDA_ENV:-fmtk}"
    PYTHON="${PYTHON:-conda run -n ${CONDA_ENV} python}"
else
    # If conda not in PATH, try to find Python from environment
    PYTHON="${PYTHON:-python}"
fi

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

PHASE_DURATIONS_CSV=$(IFS=','; echo "${DURATION_LIST[*]}")

LOG_DIR="${RESULTS_BASE}/logs"
mkdir -p "$LOG_DIR"

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
    echo "Runs:"
    for run in "${RUNS[@]}"; do
        read -r sched bsize bwait rname <<< "$run"
        echo "  $rname: scheduler=$sched batch_size=$bsize batch_wait_ms=$bwait"
    done
    echo "Results base: $RESULTS_BASE"
} > "$CONFIG_FILE"

echo "================================================================"
echo "  noisy_neighbor/tsfm — ${NUM_PHASES}-phase experiment"
echo "  Backbone     : $BACKBONE"
echo "  Victim       : $VICTIM_TASK @ ${VICTIM_RPS} rps (constant)"
echo "  Aggressor    : $AGGRESSOR_TASK"
for (( i=0; i<NUM_PHASES; i++ )); do
    echo "  Phase $(( i+1 )) (${DURATION_LIST[$i]}s): aggressor @ ${AGGRESSOR_RPS_LIST[$i]} rps"
done
echo "  Total        : ${TOTAL_DURATION}s"
echo "  Runs         :"
for run in "${RUNS[@]}"; do
    read -r sched bsize bwait rname <<< "$run"
    echo "    $rname: scheduler=$sched batch_size=$bsize batch_wait_ms=$bwait"
done
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
    local scheduler="$1" batch_size="$2" batch_wait="$3" log="$4"
    local task_rates="${VICTIM_TASK}:${VICTIM_RPS},${AGGRESSOR_TASK}:${VICTIM_RPS}"
    pkill -f "device/main.py.*--port ${DEVICE_PORT}" 2>/dev/null || true
    pkill -f "tsfm/run.py" 2>/dev/null || true
    sleep 1
    echo "[run.sh] Starting device server (scheduler=$scheduler, bsize=$batch_size, bwait=${batch_wait}ms, rates=$task_rates)..."
    "$PYTHON" -u "$SERVING_DIR/device/main.py" \
        --port              "$DEVICE_PORT"  \
        --runtime-type      pytorch         \
        --cuda              "$CUDA_DEVICE"  \
        --scheduler-policy  "$scheduler"    \
        --max-batch-wait-ms "$batch_wait"   \
        --task-rates        "$task_rates"   \
        --max-batch-size    "$batch_size"   \
        > "$log" 2>&1 &
    DEVICE_PID=$!
    echo "[run.sh] Device PID=$DEVICE_PID  log=$log"
    echo "[run.sh] Waiting ${DEVICE_STARTUP_WAIT}s for server to be ready..."
    sleep "$DEVICE_STARTUP_WAIT"
}

# ---------------------------------------------------------------------------
# Main loop — one fresh device server per run configuration
# ---------------------------------------------------------------------------
TOTAL=${#RUNS[@]}
IDX=0

for run in "${RUNS[@]}"; do
    read -r SCHEDULER BATCH_SIZE BATCH_WAIT RUN_NAME <<< "$run"
    IDX=$(( IDX + 1 ))
    EXP_DIR="${RESULTS_BASE}/${RUN_NAME}"
    DEVICE_LOG="$LOG_DIR/device_${RUN_NAME}.log"

    echo ""
    echo "================================================================"
    echo "  [$IDX/$TOTAL] $RUN_NAME  (scheduler=$SCHEDULER, bsize=$BATCH_SIZE, bwait=${BATCH_WAIT}ms)"
    echo "  Results: $EXP_DIR"
    echo "================================================================"

    start_device "$SCHEDULER" "$BATCH_SIZE" "$BATCH_WAIT" "$DEVICE_LOG"

    "$PYTHON" -u experiments/noisy_neighbor/tsfm/run.py \
        --device-url            "localhost:${DEVICE_PORT}"   \
        --backbone              "$BACKBONE"                  \
        --victim-task           "$VICTIM_TASK"               \
        --aggressor-task        "$AGGRESSOR_TASK"            \
        --victim-rps            "$VICTIM_RPS"                \
        --aggressor-rps-phases  "$AGGRESSOR_RPS_PHASES"      \
        --phase-durations       "$PHASE_DURATIONS_CSV"       \
        --scheduler-policy      "$SCHEDULER"                 \
        --exp-dir               "$EXP_DIR"                   \
    || echo "[run.sh] WARNING: run.py failed for $RUN_NAME — continuing"

    stop_device
done

echo ""
echo "[run.sh] All $TOTAL runs done. Results in $RESULTS_BASE"
