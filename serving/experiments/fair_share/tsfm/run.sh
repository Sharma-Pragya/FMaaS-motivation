#!/usr/bin/env bash
# fair_share/tsfm — Three-phase priority experiment with phased victim load
#
# Both clients send Poisson open-loop. The aggressor stays near GPU capacity
# the whole run; the victim's load varies (low → high → low). Victim is the
# high-priority client (BFQ uses biased weights to favor it).
#
# Phase 1: aggressor near limit, victim quiet. Aggressor uses spare capacity.
# Phase 2: victim ramps to near limit. Over-saturated; priority decides who wins.
# Phase 3: victim drops back. Aggressor reclaims capacity.
#
# Only BFQ exposes a priority knob; fcfs/stfq/no_sharing/no_sharing_tpc treat
# both clients equally (no operator-controlled differentiation).
#
# Run from serving/:
#   bash experiments/fair_share/tsfm/run.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Path setup
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

# Three-phase priority experiment (10s each, 30s total):
#   Phase 1: aggressor near system limit, victim very low load.
#   Phase 2: victim ramps near system limit (aggressor stays high → over-saturation).
#   Phase 3: victim drops back to low load.
# Victim is the high-priority client throughout. Under BFQ, weights bias
# scheduling toward victim; aggressor gets reclaimed capacity in phases 1 & 3.
VICTIM_RPS_PHASES="${VICTIM_RPS_PHASES:-2,32,2}"
AGGRESSOR_RPS_PHASES="${AGGRESSOR_RPS_PHASES:-32,32,32}"
PHASE_DURATIONS="${PHASE_DURATIONS:-10,10,10}"
# VICTIM_RPS kept as a fallback constant for backward compat; ignored when
# VICTIM_RPS_PHASES is set.
VICTIM_RPS="${VICTIM_RPS:-5}"

# Sharing runs: "scheduler  batch_size  batch_wait_ms  run_name  [task_rates_override]"
# task_rates_override is fed verbatim to --task-rates. The scheduler computes
# weight = 1/rps internally; STFQ then advances virtual finish time by 1/weight,
# so a LOW rps value yields HIGH weight (more slots). For each BFQ variant
# we set A's rps=1 and B's rps={1,2,3}, giving W_A:W_B = 1:1, 2:1, 3:1.
#
# fcfs_nobatch (FIFO scheduler, batch=1) is included to isolate the cost
# of batching from the cost of scheduling in the throughput/latency plots.
RUNS=(
    "fifo  3  0  fcfs            "
    "fifo  1  0  fcfs_nobatch    "
    "stfq  1  0  stfq            "
    "stfq  3  0  bfq_1_1         ${VICTIM_TASK}:1,${AGGRESSOR_TASK}:1"
    "stfq  3  0  bfq_2_1         ${VICTIM_TASK}:1,${AGGRESSOR_TASK}:2"
    "stfq  3  0  bfq_3_1         ${VICTIM_TASK}:1,${AGGRESSOR_TASK}:3"
)

# No-sharing TPC runs: victim and aggressor each get their own TPC-partitioned server.
# Format: "batch_size  batch_wait_ms  run_name"
NO_SHARING_TPC_RUNS=(
    "3  0  no_sharing_tpc"
)
TPC_MODE="${TPC_MODE:-libsmctrl}"

# No-sharing runs: victim and aggressor each get their own device server on
# the same GPU (no TPC partitioning — process-level isolation only).
# Format: "batch_size  batch_wait_ms  run_name"
NO_SHARING_RUNS=(
    "3  0  no_sharing"
)

RESULTS_BASE="${RESULTS_BASE:-experiments/fair_share/tsfm/results}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-5}"

if command -v conda &> /dev/null; then
    CONDA_ENV="${CONDA_ENV:-fmtk}"
    # --no-capture-output is required so the device server's stdout/stderr
    # streams live to the log file. Without it, `conda run` buffers everything
    # internally and only flushes on clean child exit; since we SIGTERM the
    # device between methods, the buffer is discarded → empty log files.
    PYTHON="${PYTHON:-conda run --no-capture-output -n ${CONDA_ENV} python}"
else
    PYTHON="${PYTHON:-python}"
fi

# ---------------------------------------------------------------------------
# Resolve phase count and per-phase duration list
# ---------------------------------------------------------------------------
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
    echo "Backbone: $BACKBONE"
    echo "Client A (priority/victim): $VICTIM_TASK"
    echo "Client B (other/aggressor): $AGGRESSOR_TASK"
    echo "Number of phases: $NUM_PHASES"
    IFS=',' read -ra _VRPS <<< "$VICTIM_RPS_PHASES"
    for (( i=0; i<NUM_PHASES; i++ )); do
        echo "  Phase $(( i+1 )) (${DURATION_LIST[$i]}s): A @ ${_VRPS[$i]} rps, B @ ${AGGRESSOR_RPS_LIST[$i]} rps"
    done
    echo "Total duration: ${TOTAL_DURATION}s"
    echo "Sharing runs:"
    for run in "${RUNS[@]}"; do
        read -r sched bsize bwait rname rates <<< "$run"
        echo "  $rname: scheduler=$sched batch_size=$bsize batch_wait_ms=$bwait${rates:+ rates=$rates}"
    done
    echo "No-sharing TPC runs: ${NO_SHARING_TPC_RUNS[*]} (tpc_mode=$TPC_MODE)"
    echo "No-sharing runs: ${NO_SHARING_RUNS[*]}"
    echo "Results base: $RESULTS_BASE"
} > "$CONFIG_FILE"

echo "================================================================"
echo "  fair_share/tsfm — saturation experiment"
echo "  Backbone     : $BACKBONE"
echo "  Client A (priority/victim): $VICTIM_TASK"
echo "  Client B (other/aggressor): $AGGRESSOR_TASK"
IFS=',' read -ra _VRPS <<< "$VICTIM_RPS_PHASES"
for (( i=0; i<NUM_PHASES; i++ )); do
    echo "  Phase $(( i+1 )) (${DURATION_LIST[$i]}s): A @ ${_VRPS[$i]} rps, B @ ${AGGRESSOR_RPS_LIST[$i]} rps"
done
echo "  Total        : ${TOTAL_DURATION}s"
echo "  Sharing runs :"
for run in "${RUNS[@]}"; do
    read -r sched bsize bwait rname rates <<< "$run"
    echo "    $rname: scheduler=$sched batch_size=$bsize batch_wait_ms=$bwait${rates:+ rates=$rates}"
done
echo "  No-sharing TPC runs: ${NO_SHARING_TPC_RUNS[*]} (tpc_mode=$TPC_MODE)"
echo "  No-sharing runs    : ${NO_SHARING_RUNS[*]}"
echo "  Results base : $RESULTS_BASE"
echo "================================================================"

# ---------------------------------------------------------------------------
# Helpers
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

# start_shared_device SCHEDULER BATCH_SIZE BATCH_WAIT LOG [RATES_OVERRIDE]
# RATES_OVERRIDE (optional): full --task-rates string. If empty, defaults to
# equal weights (both tasks at VICTIM_RPS). The scheduler uses 1/rps as the
# task weight, so passing skewed rates lets us bias scheduling priority
# (e.g. for bfq_priority).
start_shared_device() {
    local scheduler="$1" batch_size="$2" batch_wait="$3" log="$4"
    local rates_override="${5:-}"
    local task_rates
    if [[ -n "$rates_override" ]]; then
        task_rates="$rates_override"
    else
        task_rates="${VICTIM_TASK}:${VICTIM_RPS},${AGGRESSOR_TASK}:${VICTIM_RPS}"
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
    echo "[run.sh] Waiting ${DEVICE_STARTUP_WAIT}s for server to be ready..."
    sleep "$DEVICE_STARTUP_WAIT"
}

# start_tpc_device PORT TASK TPC_PARTITION BATCH_SIZE BATCH_WAIT LOG
start_tpc_device() {
    local port="$1" task="$2" tpc_partition="$3" batch_size="$4" batch_wait="$5" log="$6"
    local task_rates="${task}:${VICTIM_RPS}"
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

# start_isolated_device PORT TASK BATCH_SIZE BATCH_WAIT LOG
#   Starts a device server for a single task (no TPC partitioning).
start_isolated_device() {
    local port="$1" task="$2" batch_size="$3" batch_wait="$4" log="$5"
    local task_rates="${task}:${VICTIM_RPS}"
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

# ---------------------------------------------------------------------------
# Sharing runs — single device server for both tasks
# ---------------------------------------------------------------------------
TOTAL=${#RUNS[@]}
IDX=0

for run in "${RUNS[@]}"; do
    read -r SCHEDULER BATCH_SIZE BATCH_WAIT RUN_NAME RATES_OVERRIDE <<< "$run"
    IDX=$(( IDX + 1 ))
    EXP_DIR="${RESULTS_BASE}/${RUN_NAME}"
    DEVICE_LOG="$LOG_DIR/device_${RUN_NAME}.log"

    echo ""
    echo "================================================================"
    echo "  [$IDX/$TOTAL] $RUN_NAME  (scheduler=$SCHEDULER, bsize=$BATCH_SIZE, bwait=${BATCH_WAIT}ms${RATES_OVERRIDE:+, rates=$RATES_OVERRIDE})"
    echo "  Results: $EXP_DIR"
    echo "================================================================"

    start_shared_device "$SCHEDULER" "$BATCH_SIZE" "$BATCH_WAIT" "$DEVICE_LOG" "$RATES_OVERRIDE"

    $PYTHON -u experiments/fair_share/tsfm/run.py \
        --device-url            "localhost:${DEVICE_PORT}"   \
        --backbone              "$BACKBONE"                  \
        --victim-task           "$VICTIM_TASK"               \
        --aggressor-task        "$AGGRESSOR_TASK"            \
        --victim-rps            "$VICTIM_RPS"                \
        --victim-rps-phases     "$VICTIM_RPS_PHASES"         \
        --aggressor-rps-phases  "$AGGRESSOR_RPS_PHASES"      \
        --phase-durations       "$PHASE_DURATIONS_CSV"       \
        --scheduler-policy      "$SCHEDULER"                 \
        --exp-dir               "$EXP_DIR"                   \
    || echo "[run.sh] WARNING: run.py failed for $RUN_NAME — continuing"

    stop_devices
done

# ---------------------------------------------------------------------------
# No-sharing TPC runs — each task gets its own TPC-partitioned device server
# ---------------------------------------------------------------------------
for run in "${NO_SHARING_TPC_RUNS[@]}"; do
    read -r BATCH_SIZE BATCH_WAIT RUN_NAME <<< "$run"
    EXP_DIR="${RESULTS_BASE}/${RUN_NAME}"

    echo ""
    echo "================================================================"
    echo "  $RUN_NAME  (TPC-isolated, 2 servers, tpc_mode=$TPC_MODE, bsize=$BATCH_SIZE, bwait=${BATCH_WAIT}ms)"
    echo "  Results: $EXP_DIR"
    echo "================================================================"

    stop_devices
    DEVICE_PIDS=()
    ACTIVE_PORTS=()

    TOTAL_TPCS=$($PYTHON -c "
import torch
sm = torch.cuda.get_device_properties('${CUDA_DEVICE}').multi_processor_count
print(sm // 2)
")
    echo "[run.sh] Total TPCs on GPU: $TOTAL_TPCS"

    HALF=$((TOTAL_TPCS / 2))
    VICTIM_TPCS=$(seq -s ' ' 0 $((HALF - 1)))
    AGGRESSOR_TPCS=$(seq -s ' ' "$HALF" $((TOTAL_TPCS - 1)))

    VICTIM_PORT="$DEVICE_PORT"
    AGGRESSOR_PORT=$((DEVICE_PORT + 1))

    echo "[run.sh] Victim:     port=$VICTIM_PORT     task=$VICTIM_TASK     tpcs=[$VICTIM_TPCS]"
    echo "[run.sh] Aggressor:  port=$AGGRESSOR_PORT  task=$AGGRESSOR_TASK  tpcs=[$AGGRESSOR_TPCS]"

    start_tpc_device "$VICTIM_PORT"    "$VICTIM_TASK"    "$VICTIM_TPCS"    "$BATCH_SIZE" "$BATCH_WAIT" "$LOG_DIR/device_${RUN_NAME}_victim.log"
    start_tpc_device "$AGGRESSOR_PORT" "$AGGRESSOR_TASK" "$AGGRESSOR_TPCS" "$BATCH_SIZE" "$BATCH_WAIT" "$LOG_DIR/device_${RUN_NAME}_aggressor.log"

    $PYTHON -u experiments/fair_share/tsfm/run.py \
        --victim-url            "localhost:${VICTIM_PORT}"      \
        --aggressor-url         "localhost:${AGGRESSOR_PORT}"   \
        --backbone              "$BACKBONE"                     \
        --victim-task           "$VICTIM_TASK"                  \
        --aggressor-task        "$AGGRESSOR_TASK"               \
        --victim-rps            "$VICTIM_RPS"                   \
        --victim-rps-phases     "$VICTIM_RPS_PHASES"            \
        --aggressor-rps-phases  "$AGGRESSOR_RPS_PHASES"         \
        --phase-durations       "$PHASE_DURATIONS_CSV"          \
        --scheduler-policy      "fifo"                          \
        --exp-dir               "$EXP_DIR"                      \
    || echo "[run.sh] WARNING: run.py failed for $RUN_NAME — continuing"

    stop_devices
done

# ---------------------------------------------------------------------------
# No-sharing runs — each task gets its own device server (no TPC partitioning)
# ---------------------------------------------------------------------------
for run in "${NO_SHARING_RUNS[@]}"; do
    read -r BATCH_SIZE BATCH_WAIT RUN_NAME <<< "$run"
    EXP_DIR="${RESULTS_BASE}/${RUN_NAME}"

    echo ""
    echo "================================================================"
    echo "  $RUN_NAME  (process-isolated, 2 servers, no TPC partition, bsize=$BATCH_SIZE, bwait=${BATCH_WAIT}ms)"
    echo "  Results: $EXP_DIR"
    echo "================================================================"

    stop_devices
    DEVICE_PIDS=()
    ACTIVE_PORTS=()

    VICTIM_PORT="$DEVICE_PORT"
    AGGRESSOR_PORT=$((DEVICE_PORT + 1))

    echo "[run.sh] Victim:     port=$VICTIM_PORT     task=$VICTIM_TASK"
    echo "[run.sh] Aggressor:  port=$AGGRESSOR_PORT  task=$AGGRESSOR_TASK"

    start_isolated_device "$VICTIM_PORT"    "$VICTIM_TASK"    "$BATCH_SIZE" "$BATCH_WAIT" "$LOG_DIR/device_${RUN_NAME}_victim.log"
    start_isolated_device "$AGGRESSOR_PORT" "$AGGRESSOR_TASK" "$BATCH_SIZE" "$BATCH_WAIT" "$LOG_DIR/device_${RUN_NAME}_aggressor.log"

    $PYTHON -u experiments/fair_share/tsfm/run.py \
        --victim-url            "localhost:${VICTIM_PORT}"      \
        --aggressor-url         "localhost:${AGGRESSOR_PORT}"   \
        --backbone              "$BACKBONE"                     \
        --victim-task           "$VICTIM_TASK"                  \
        --aggressor-task        "$AGGRESSOR_TASK"               \
        --victim-rps            "$VICTIM_RPS"                   \
        --victim-rps-phases     "$VICTIM_RPS_PHASES"            \
        --aggressor-rps-phases  "$AGGRESSOR_RPS_PHASES"         \
        --phase-durations       "$PHASE_DURATIONS_CSV"          \
        --scheduler-policy      "fifo"                          \
        --exp-dir               "$EXP_DIR"                      \
    || echo "[run.sh] WARNING: run.py failed for $RUN_NAME — continuing"

    stop_devices
done

echo ""
echo "[run.sh] All runs done. Results in $RESULTS_BASE"
