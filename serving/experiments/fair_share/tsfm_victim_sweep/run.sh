#!/usr/bin/env bash
# fair_share/tsfm_victim_sweep — Single-phase sweep over victim RPS.
#
# Aggressor RPS fixed at 50. Victim RPS swept over {10, 30, 50, 70} as
# independent experiments. Each (victim_rps, method) pair runs for
# DURATION seconds (default 10s). Methods with a priority knob use
# weights 2:1 (victim:aggressor).
#
# Reuses run.py / plot.py from ../tsfm.
#
# Run from serving/:
#   bash experiments/fair_share/tsfm_victim_sweep/run.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Path setup (mirrors ../tsfm/run.sh)
# ---------------------------------------------------------------------------
SERVING_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$SERVING_DIR"

FMTK_DIR="${FMTK_DIR:-../../FMTK}"
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

if [[ ! -d "$FMTK_DIR" ]]; then
    echo "ERROR: FMTK_DIR not found at: $FMTK_DIR"
    exit 1
fi
if [[ ! -d "$FMAAS_DIR" ]]; then
    echo "ERROR: FMAAS_DIR not found at: $FMAAS_DIR"
    exit 1
fi

export PYTHONPATH="${FMTK_DIR}/src:${FMAAS_DIR}:${PYTHONPATH:-}"

DEVICE_PORT="${DEVICE_PORT:-8000}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:-momentlarge}"
VICTIM_TASK="${VICTIM_TASK:-ecgclass}"
AGGRESSOR_TASK="${AGGRESSOR_TASK:-gestureclass}"

# Sweep config: single phase, fixed aggressor, varying victim.
VICTIM_RPS_VALUES=(${VICTIM_RPS_VALUES:-10 30 50 70})
AGGRESSOR_RPS="${AGGRESSOR_RPS:-50}"
DURATION="${DURATION:-10}"

# Sharing runs that share one device server.
# Format: "scheduler  batch_size  batch_wait_ms  run_name  [task_rates_override]"
# bfq_2_1 uses STFQ with rates such that weight_victim:weight_aggressor = 2:1
# (scheduler weight = 1/rps internally, so victim:1, aggressor:2).
SHARING_RUNS=(
    "fifo  3  0  fcfs            "
    "stfq  3  0  bfq_2_1         ${VICTIM_TASK}:1,${AGGRESSOR_TASK}:2"
)

# No-sharing TPC runs: each task on its own TPC-partitioned server.
# Format: "batch_size  batch_wait_ms  run_name  weight_a  weight_b"
NO_SHARING_TPC_RUNS=(
    "3  0  no_sharing_tpc  2  1"
)
TPC_MODE="${TPC_MODE:-libsmctrl}"

# No-sharing runs: each task on its own device server (no TPC partition).
# Format: "batch_size  batch_wait_ms  run_name"
NO_SHARING_RUNS=(
    "3  0  no_sharing"
)

RESULTS_BASE="${RESULTS_BASE:-experiments/fair_share/tsfm_victim_sweep/results}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-5}"

if command -v conda &> /dev/null; then
    CONDA_ENV="${CONDA_ENV:-fmtk}"
    PYTHON="${PYTHON:-conda run --no-capture-output -n ${CONDA_ENV} python}"
else
    PYTHON="${PYTHON:-python}"
fi

mkdir -p "$RESULTS_BASE"

CONFIG_FILE="$RESULTS_BASE/config.txt"
{
    echo "Experiment config - $(date)"
    echo "Backbone: $BACKBONE"
    echo "Victim task : $VICTIM_TASK"
    echo "Aggressor task: $AGGRESSOR_TASK"
    echo "Aggressor RPS (fixed): $AGGRESSOR_RPS"
    echo "Victim RPS sweep    : ${VICTIM_RPS_VALUES[*]}"
    echo "Duration per run    : ${DURATION}s"
    echo "Sharing runs       : ${SHARING_RUNS[*]}"
    echo "No-sharing TPC runs: ${NO_SHARING_TPC_RUNS[*]} (tpc_mode=$TPC_MODE)"
    echo "No-sharing runs    : ${NO_SHARING_RUNS[*]}"
    echo "Results base: $RESULTS_BASE"
} > "$CONFIG_FILE"

echo "================================================================"
echo "  fair_share/tsfm_victim_sweep"
echo "  Backbone        : $BACKBONE"
echo "  Victim task     : $VICTIM_TASK"
echo "  Aggressor task  : $AGGRESSOR_TASK   (RPS=$AGGRESSOR_RPS)"
echo "  Victim RPS sweep: ${VICTIM_RPS_VALUES[*]}"
echo "  Duration        : ${DURATION}s per run"
echo "  Results base    : $RESULTS_BASE"
echo "================================================================"

# ---------------------------------------------------------------------------
# Device-server lifecycle helpers (copied from ../tsfm/run.sh)
# ---------------------------------------------------------------------------
DEVICE_PIDS=()
ACTIVE_PORTS=()

stop_devices() {
    local pids_to_wait=()
    for pid in "${DEVICE_PIDS[@]:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "[run.sh] Stopping device PID=$pid (SIGTERM)" >&2
            kill "$pid" 2>/dev/null || true
            pids_to_wait+=("$pid")
        fi
    done
    local deadline=$((SECONDS + 12))
    for pid in "${pids_to_wait[@]:-}"; do
        while kill -0 "$pid" 2>/dev/null; do
            if [[ $SECONDS -ge $deadline ]]; then
                echo "[run.sh] WARNING: PID=$pid did not exit in 12s — sending SIGKILL" >&2
                kill -9 "$pid" 2>/dev/null || true
                break
            fi
            sleep 0.5
        done
        wait "$pid" 2>/dev/null || true
    done
    for port in "${ACTIVE_PORTS[@]:-}"; do
        pkill -f "device/main.py.*--port ${port}" 2>/dev/null || true
    done
    DEVICE_PIDS=()
    ACTIVE_PORTS=()
    sleep 2
}
trap 'stop_devices' EXIT

# start_shared_device SCHEDULER BATCH_SIZE BATCH_WAIT LOG VICTIM_RPS [RATES_OVERRIDE]
start_shared_device() {
    local scheduler="$1" batch_size="$2" batch_wait="$3" log="$4" victim_rps="$5"
    local rates_override="${6:-}"
    local task_rates
    if [[ -n "$rates_override" ]]; then
        task_rates="$rates_override"
    else
        task_rates="${VICTIM_TASK}:${victim_rps},${AGGRESSOR_TASK}:${victim_rps}"
    fi
    pkill -f "device/main.py.*--port ${DEVICE_PORT}" 2>/dev/null || true
    pkill -f "fair_share/tsfm/run.py" 2>/dev/null || true
    sleep 1
    echo "[run.sh] Starting device server (scheduler=$scheduler, bsize=$batch_size, bwait=${batch_wait}ms, rates=$task_rates)..."
    $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port              "$DEVICE_PORT"  \
        --runtime-type      pytorch         \
        --cuda              "$CUDA_DEVICE"  \
        --scheduler-policy  "$scheduler"    \
        --max-batch-wait-ms "$batch_wait"   \
        --task-rates        "$task_rates"   \
        --max-batch-size    "$batch_size"   \
        > "$log" 2>&1 &
    DEVICE_PIDS+=("$!")
    ACTIVE_PORTS+=("$DEVICE_PORT")
    echo "[run.sh] Device PID=$!  log=$log"
    sleep "$DEVICE_STARTUP_WAIT"
}

# start_tpc_device PORT TASK TPC_PARTITION BATCH_SIZE BATCH_WAIT LOG VICTIM_RPS
start_tpc_device() {
    local port="$1" task="$2" tpc_partition="$3" batch_size="$4" batch_wait="$5" log="$6" victim_rps="$7"
    local task_rates="${task}:${victim_rps}"
    pkill -f "device/main.py.*--port ${port}" 2>/dev/null || true
    sleep 1
    echo "[run.sh] Starting TPC device: port=$port task=$task tpcs=[$tpc_partition] bsize=$batch_size bwait=${batch_wait}ms" >&2
    $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port              "$port"          \
        --runtime-type      pytorch          \
        --cuda              "$CUDA_DEVICE"   \
        --scheduler-policy  fifo             \
        --max-batch-wait-ms "$batch_wait"    \
        --task-rates        "$task_rates"    \
        --max-batch-size    "$batch_size"    \
        --tpc-mode          "$TPC_MODE"      \
        --tpc-partition     $tpc_partition    \
        --worker-mode       inline            \
        > "$log" 2>&1 &
    local pid=$!
    DEVICE_PIDS+=("$pid")
    ACTIVE_PORTS+=("$port")
    echo "[run.sh] PID=$pid  log=$log" >&2
    sleep "$DEVICE_STARTUP_WAIT"
}

# start_isolated_device PORT TASK BATCH_SIZE BATCH_WAIT LOG VICTIM_RPS
start_isolated_device() {
    local port="$1" task="$2" batch_size="$3" batch_wait="$4" log="$5" victim_rps="$6"
    local task_rates="${task}:${victim_rps}"
    pkill -f "device/main.py.*--port ${port}" 2>/dev/null || true
    sleep 1
    echo "[run.sh] Starting isolated device: port=$port task=$task bsize=$batch_size bwait=${batch_wait}ms" >&2
    $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port              "$port"          \
        --runtime-type      pytorch          \
        --cuda              "$CUDA_DEVICE"   \
        --scheduler-policy  fifo             \
        --max-batch-wait-ms "$batch_wait"    \
        --task-rates        "$task_rates"    \
        --max-batch-size    "$batch_size"    \
        --worker-mode       inline            \
        > "$log" 2>&1 &
    local pid=$!
    DEVICE_PIDS+=("$pid")
    ACTIVE_PORTS+=("$port")
    echo "[run.sh] PID=$pid  log=$log" >&2
    sleep "$DEVICE_STARTUP_WAIT"
}

RUN_PY="$SERVING_DIR/experiments/fair_share/tsfm_victim_sweep/run.py"

# ---------------------------------------------------------------------------
# Outer loop: sweep victim RPS
# ---------------------------------------------------------------------------
for VRPS in "${VICTIM_RPS_VALUES[@]}"; do
    SWEEP_DIR="${RESULTS_BASE}/victim_${VRPS}"
    SWEEP_LOG_DIR="${SWEEP_DIR}/logs"
    mkdir -p "$SWEEP_LOG_DIR"

    echo ""
    echo "################################################################"
    echo "#  VICTIM RPS = $VRPS    (aggressor=${AGGRESSOR_RPS}, ${DURATION}s)"
    echo "#  → $SWEEP_DIR"
    echo "################################################################"

    # ---- Sharing runs ----
    for run in "${SHARING_RUNS[@]}"; do
        read -r SCHEDULER BATCH_SIZE BATCH_WAIT RUN_NAME RATES_OVERRIDE <<< "$run"
        EXP_DIR="${SWEEP_DIR}/${RUN_NAME}"
        DEVICE_LOG="$SWEEP_LOG_DIR/device_${RUN_NAME}.log"

        echo ""
        echo "----------------------------------------------------------------"
        echo "  [victim=${VRPS}] $RUN_NAME (scheduler=$SCHEDULER, bsize=$BATCH_SIZE${RATES_OVERRIDE:+, rates=$RATES_OVERRIDE})"
        echo "----------------------------------------------------------------"

        start_shared_device "$SCHEDULER" "$BATCH_SIZE" "$BATCH_WAIT" "$DEVICE_LOG" "$VRPS" "$RATES_OVERRIDE"

        $PYTHON -u "$RUN_PY" \
            --device-url            "localhost:${DEVICE_PORT}"   \
            --backbone              "$BACKBONE"                  \
            --victim-task           "$VICTIM_TASK"               \
            --aggressor-task        "$AGGRESSOR_TASK"            \
            --victim-rps            "$VRPS"                      \
            --aggressor-rps         "$AGGRESSOR_RPS"             \
            --duration              "$DURATION"                  \
            --scheduler-policy      "$SCHEDULER"                 \
            --exp-dir               "$EXP_DIR"                   \
        || echo "[run.sh] WARNING: run.py failed for $RUN_NAME (victim=$VRPS) — continuing"

        stop_devices
    done

    # ---- No-sharing TPC runs ----
    for run in "${NO_SHARING_TPC_RUNS[@]}"; do
        read -r BATCH_SIZE BATCH_WAIT RUN_NAME WEIGHT_A WEIGHT_B <<< "$run"
        EXP_DIR="${SWEEP_DIR}/${RUN_NAME}"

        echo ""
        echo "----------------------------------------------------------------"
        echo "  [victim=${VRPS}] $RUN_NAME (TPC, weights=${WEIGHT_A}:${WEIGHT_B}, tpc_mode=$TPC_MODE)"
        echo "----------------------------------------------------------------"

        stop_devices
        DEVICE_PIDS=()
        ACTIVE_PORTS=()

        TOTAL_TPCS=$($PYTHON -c "
import torch
sm = torch.cuda.get_device_properties('${CUDA_DEVICE}').multi_processor_count
print(sm // 2)
")
        echo "[run.sh] Total TPCs: $TOTAL_TPCS"

        WEIGHT_TOTAL=$((WEIGHT_A + WEIGHT_B))
        VICTIM_TPC_COUNT=$(( TOTAL_TPCS * WEIGHT_A / WEIGHT_TOTAL ))
        if [[ $VICTIM_TPC_COUNT -le 0 ]]; then VICTIM_TPC_COUNT=1; fi
        if [[ $VICTIM_TPC_COUNT -ge $TOTAL_TPCS ]]; then VICTIM_TPC_COUNT=$(( TOTAL_TPCS - 1 )); fi
        AGGRESSOR_TPC_COUNT=$(( TOTAL_TPCS - VICTIM_TPC_COUNT ))

        VICTIM_TPCS=$(seq -s ' ' 0 $((VICTIM_TPC_COUNT - 1)))
        AGGRESSOR_TPCS=$(seq -s ' ' "$VICTIM_TPC_COUNT" $((TOTAL_TPCS - 1)))

        VICTIM_PORT="$DEVICE_PORT"
        AGGRESSOR_PORT=$((DEVICE_PORT + 1))

        echo "[run.sh] Victim:    port=$VICTIM_PORT    tpcs=[$VICTIM_TPCS]"
        echo "[run.sh] Aggressor: port=$AGGRESSOR_PORT tpcs=[$AGGRESSOR_TPCS]"

        start_tpc_device "$VICTIM_PORT"    "$VICTIM_TASK"    "$VICTIM_TPCS"    "$BATCH_SIZE" "$BATCH_WAIT" "$SWEEP_LOG_DIR/device_${RUN_NAME}_victim.log"    "$VRPS"
        start_tpc_device "$AGGRESSOR_PORT" "$AGGRESSOR_TASK" "$AGGRESSOR_TPCS" "$BATCH_SIZE" "$BATCH_WAIT" "$SWEEP_LOG_DIR/device_${RUN_NAME}_aggressor.log" "$VRPS"

        $PYTHON -u "$RUN_PY" \
            --victim-url            "localhost:${VICTIM_PORT}"      \
            --aggressor-url         "localhost:${AGGRESSOR_PORT}"   \
            --backbone              "$BACKBONE"                     \
            --victim-task           "$VICTIM_TASK"                  \
            --aggressor-task        "$AGGRESSOR_TASK"               \
            --victim-rps            "$VRPS"                         \
            --aggressor-rps         "$AGGRESSOR_RPS"                \
            --duration              "$DURATION"                     \
            --scheduler-policy      "fifo"                          \
            --exp-dir               "$EXP_DIR"                      \
        || echo "[run.sh] WARNING: run.py failed for $RUN_NAME (victim=$VRPS) — continuing"

        stop_devices
    done

    # ---- No-sharing runs (process-isolated, no TPC partition) ----
    for run in "${NO_SHARING_RUNS[@]}"; do
        read -r BATCH_SIZE BATCH_WAIT RUN_NAME <<< "$run"
        EXP_DIR="${SWEEP_DIR}/${RUN_NAME}"

        echo ""
        echo "----------------------------------------------------------------"
        echo "  [victim=${VRPS}] $RUN_NAME (process-isolated, 2 servers)"
        echo "----------------------------------------------------------------"

        stop_devices
        DEVICE_PIDS=()
        ACTIVE_PORTS=()

        VICTIM_PORT="$DEVICE_PORT"
        AGGRESSOR_PORT=$((DEVICE_PORT + 1))

        start_isolated_device "$VICTIM_PORT"    "$VICTIM_TASK"    "$BATCH_SIZE" "$BATCH_WAIT" "$SWEEP_LOG_DIR/device_${RUN_NAME}_victim.log"    "$VRPS"
        start_isolated_device "$AGGRESSOR_PORT" "$AGGRESSOR_TASK" "$BATCH_SIZE" "$BATCH_WAIT" "$SWEEP_LOG_DIR/device_${RUN_NAME}_aggressor.log" "$VRPS"

        $PYTHON -u "$RUN_PY" \
            --victim-url            "localhost:${VICTIM_PORT}"      \
            --aggressor-url         "localhost:${AGGRESSOR_PORT}"   \
            --backbone              "$BACKBONE"                     \
            --victim-task           "$VICTIM_TASK"                  \
            --aggressor-task        "$AGGRESSOR_TASK"               \
            --victim-rps            "$VRPS"                         \
            --aggressor-rps         "$AGGRESSOR_RPS"                \
            --duration              "$DURATION"                     \
            --scheduler-policy      "fifo"                          \
            --exp-dir               "$EXP_DIR"                      \
        || echo "[run.sh] WARNING: run.py failed for $RUN_NAME (victim=$VRPS) — continuing"

        stop_devices
    done

done

echo ""
echo "[run.sh] Sweep complete. Results in $RESULTS_BASE"
