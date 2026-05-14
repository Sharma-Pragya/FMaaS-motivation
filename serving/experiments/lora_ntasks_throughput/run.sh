#!/usr/bin/env bash
# lora_ntasks_throughput — closed-loop throughput sweep over N LoRA-adapted tasks.
#
# For each value of N in NUM_TASKS_LIST, this script:
#   1. starts a fresh device server,
#   2. deploys N replicas of <BASE_TASK> on <BACKBONE> with MLP+LoRA decoder,
#   3. runs closed-loop traffic with K=CONCURRENCY_PER_TASK workers per replica
#      for DURATION seconds, and
#   4. records per-task and aggregate throughput / latency.
#
# Run from serving/:
#   bash experiments/lora_ntasks_throughput/run.sh
#
# Common overrides:
#   NUM_TASKS_LIST="1,2,4,8,16" CONCURRENCY_PER_TASK=2 DURATION=90 \
#       bash experiments/lora_ntasks_throughput/run.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SERVING_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SERVING_DIR"

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

if [[ ! -d "$FMTK_DIR" ]]; then
    echo "ERROR: FMTK_DIR not found at: $FMTK_DIR"
    echo "Please set FMTK_DIR environment variable"
    exit 1
fi
if [[ ! -d "$FMAAS_DIR" ]]; then
    echo "ERROR: FMAAS_DIR not found at: $FMAAS_DIR"
    echo "Please set FMAAS_DIR environment variable"
    exit 1
fi

export PYTHONPATH="${FMTK_DIR}/src:${FMAAS_DIR}:${PYTHONPATH:-}"
export DATASET_DIR

if [[ -n "${PYTHON:-}" ]]; then
    # User-provided PYTHON: split on whitespace into array (allows e.g.
    # PYTHON="conda run -n env python" or PYTHON=/path/to/python).
    read -ra PYTHON <<< "$PYTHON"
elif command -v conda &> /dev/null; then
    PYTHON=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
    PYTHON=(python)
fi


DEVICE_PORT="${DEVICE_PORT:-8000}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:-momentlarge}"
BASE_TASK="${BASE_TASK:-ecgclass}"

CONCURRENCY_PER_TASK="${CONCURRENCY_PER_TASK:-1}"
DURATION="${DURATION:-60}"
WARMUP_SECS="${WARMUP_SECS:-5}"

# Comma-separated values of N (number of replicated tasks) to sweep.
NUM_TASKS_LIST="${NUM_TASKS_LIST:-2,4,6,8,10}"

SCHEDULER="${SCHEDULER:-stfq}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BATCH_WAIT_MS="${BATCH_WAIT_MS:-0}"

RESULTS_BASE="${RESULTS_BASE:-experiments/lora_ntasks_throughput/results}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-5}"

# Modes to run for each N. A subdirectory per mode is produced under RESULTS_BASE.
#   sharing            — 1 device, N LoRA adapters on shared backbone (the "method")
#   no_sharing         — N devices on consecutive ports, 1 LoRA task per backbone
#   sharing_no_adapter — 1 device, N decoders on shared backbone, no LoRA
MODES="${MODES:-sharing,no_sharing,sharing_no_adapter}"

LOG_DIR="${RESULTS_BASE}/logs"
mkdir -p "$LOG_DIR"

IFS=',' read -ra N_LIST <<< "$NUM_TASKS_LIST"

CONFIG_FILE="$RESULTS_BASE/config.txt"
{
    echo "Experiment config - $(date)"
    echo "Backbone           : $BACKBONE + LoRA"
    echo "Base task          : $BASE_TASK"
    echo "Sweep N            : ${N_LIST[*]}"
    echo "Concurrency / task : $CONCURRENCY_PER_TASK"
    echo "Duration           : ${DURATION}s (warmup=${WARMUP_SECS}s)"
    echo "Scheduler          : $SCHEDULER (bsize=$BATCH_SIZE, bwait=${BATCH_WAIT_MS}ms)"
    echo "Results base       : $RESULTS_BASE"
} > "$CONFIG_FILE"

echo "================================================================"
echo "  lora_ntasks_throughput — closed-loop sweep"
echo "  Backbone           : $BACKBONE + LoRA"
echo "  Base task          : $BASE_TASK"
echo "  Sweep N            : ${N_LIST[*]}"
echo "  Concurrency / task : $CONCURRENCY_PER_TASK"
echo "  Duration           : ${DURATION}s (warmup=${WARMUP_SECS}s)"
echo "  Scheduler          : $SCHEDULER"
echo "  Results base       : $RESULTS_BASE"
echo "================================================================"

DEVICE_PIDS=()

stop_devices() {
    for pid in "${DEVICE_PIDS[@]:-}"; do
        [[ -z "$pid" ]] && continue
        echo "[run.sh] Stopping device server (PID=$pid)"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    DEVICE_PIDS=()
    pkill -f "device/main.py.*--port" 2>/dev/null || true
    sleep 2
}
trap 'stop_devices' EXIT

# Launch one device server on the given port. Writes its PID into DEVICE_PIDS.
start_device_on_port() {
    local port="$1" log="$2"
    pkill -f "device/main.py.*--port ${port}\b" 2>/dev/null || true
    sleep 1
    echo "[run.sh] Starting device server port=$port (scheduler=$SCHEDULER bsize=$BATCH_SIZE bwait=${BATCH_WAIT_MS}ms)"
    "${PYTHON[@]}" -u "$SERVING_DIR/device/main.py" \
        --port              "$port"          \
        --runtime-type      pytorch          \
        --cuda              "$CUDA_DEVICE"   \
        --scheduler-policy  "$SCHEDULER"     \
        --max-batch-wait-ms "$BATCH_WAIT_MS" \
        --max-batch-size    "$BATCH_SIZE"    \
        > "$log" 2>&1 &
    local pid=$!
    DEVICE_PIDS+=("$pid")
    echo "[run.sh] Device PID=$pid port=$port  log=$log"
}

IFS=',' read -ra MODE_LIST <<< "$MODES"
TOTAL=$(( ${#N_LIST[@]} * ${#MODE_LIST[@]} ))
IDX=0
for MODE in "${MODE_LIST[@]}"; do
    for N in "${N_LIST[@]}"; do
        IDX=$(( IDX + 1 ))
        EXP_DIR="${RESULTS_BASE}/${MODE}/N${N}"

        echo ""
        echo "================================================================"
        echo "  [$IDX/$TOTAL] mode=$MODE  N=$N replicas of $BASE_TASK"
        echo "  Results: $EXP_DIR"
        echo "================================================================"

        case "$MODE" in
            sharing|sharing_no_adapter)
                DEVICE_LOG="${LOG_DIR}/device_${MODE}_N${N}.log"
                start_device_on_port "$DEVICE_PORT" "$DEVICE_LOG"
                sleep "$DEVICE_STARTUP_WAIT"

                "${PYTHON[@]}" -u experiments/lora_ntasks_throughput/run.py \
                    --mode                 "$MODE"                     \
                    --device-url           "localhost:${DEVICE_PORT}"  \
                    --backbone             "$BACKBONE"                 \
                    --base-task            "$BASE_TASK"                \
                    --num-tasks            "$N"                        \
                    --concurrency-per-task "$CONCURRENCY_PER_TASK"     \
                    --duration             "$DURATION"                 \
                    --warmup-secs          "$WARMUP_SECS"              \
                    --exp-dir              "$EXP_DIR"                  \
                || echo "[run.sh] WARNING: run.py failed for $MODE N=$N — continuing"
                ;;

            no_sharing)
                # N device servers on consecutive ports, same CUDA device.
                URLS=()
                for (( i=0; i<N; i++ )); do
                    port=$(( DEVICE_PORT + i ))
                    log="${LOG_DIR}/device_${MODE}_N${N}_p${port}.log"
                    start_device_on_port "$port" "$log"
                    URLS+=("localhost:${port}")
                done
                # One combined startup wait — backbones load in parallel.
                sleep "$DEVICE_STARTUP_WAIT"
                URLS_CSV=$(IFS=','; echo "${URLS[*]}")

                "${PYTHON[@]}" -u experiments/lora_ntasks_throughput/run.py \
                    --mode                 no_sharing                  \
                    --device-urls          "$URLS_CSV"                 \
                    --backbone             "$BACKBONE"                 \
                    --base-task            "$BASE_TASK"                \
                    --num-tasks            "$N"                        \
                    --concurrency-per-task "$CONCURRENCY_PER_TASK"     \
                    --duration             "$DURATION"                 \
                    --warmup-secs          "$WARMUP_SECS"              \
                    --exp-dir              "$EXP_DIR"                  \
                || echo "[run.sh] WARNING: run.py failed for $MODE N=$N — continuing"
                ;;

            *)
                echo "[run.sh] Unknown MODE='$MODE' — skipping"
                ;;
        esac

        stop_devices
    done
done

# ---------------------------------------------------------------------------
# Aggregate sweep summary
# ---------------------------------------------------------------------------
SWEEP_CSV="${RESULTS_BASE}/sweep_summary.csv"
echo "mode,num_tasks,aggregate_throughput_rps,avg_latency_ms_all,p95_latency_ms_all,p99_latency_ms_all" > "$SWEEP_CSV"
for MODE in "${MODE_LIST[@]}"; do
    for N in "${N_LIST[@]}"; do
        S="${RESULTS_BASE}/${MODE}/N${N}/summary.json"
        if [[ -f "$S" ]]; then
            "${PYTHON[@]}" - "$S" "$MODE" "$N" >> "$SWEEP_CSV" <<'PY'
import json, sys
p, mode, n = sys.argv[1], sys.argv[2], sys.argv[3]
with open(p) as f:
    d = json.load(f)
print(",".join(str(x) for x in [
    mode,
    n,
    d.get("aggregate_throughput_rps"),
    d.get("avg_latency_ms_all"),
    d.get("p95_latency_ms_all"),
    d.get("p99_latency_ms_all"),
]))
PY
        fi
    done
done
echo ""
echo "[run.sh] Sweep summary → $SWEEP_CSV"
cat "$SWEEP_CSV"
