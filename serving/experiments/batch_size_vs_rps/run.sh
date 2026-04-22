#!/bin/bash
# Experiment: batch_size_vs_rps
# Sweeps request rate (RPS) and batch waiting time (BATCH_WAIT_MS).
#
# Modes:
#   EXPERIMENT_MODE=single_task     → original ecgclass-only experiment
#   EXPERIMENT_MODE=system_tasks    → 10 SystemInAction tasks with poisson_per_task
#
# Outer loop : WAIT_SWEEP  — batch waiting times in ms  (e.g. 0,10,50,100,200)
# Inner loop : RPS_SWEEP   — request rates              (e.g. 1,2,5,10,20)
#
# For each (wait, rps) pair:
#   1. Start a fresh device server with --max-batch-wait-ms <wait>
#   2. Run run.py to send Poisson arrivals and record observed batch sizes
#   3. Stop the device server
#
# Results land in  <RESULTS_BASE>/wait_<wait_ms>/rps_<rps>/
#
# Environment variables (all optional):
#   CONDA_ENV          fmtk
#   FMTK_DIR           ../../../FMTK
#   FMAAS_DIR          ../..
#   DECODER_DIR        ${FMTK_DIR}/models/tsfm/finetuned
#   CUDA_DEVICE        cuda:0
#   BACKBONE           momentbase
#   RPS_SWEEP          1,2,5,10,20
#   WAIT_SWEEP         0,10,50,100,200      (ms)
#   NUM_CLIENTS        1                    (aggregate RPS split evenly across clients)
#   EXPERIMENT_MODE    single_task          (or system_tasks)
#   PHASE_DURATION     60
#   MAX_BATCH_SIZE     64
#   DEVICE_PORT        8000
#   RESULTS_BASE       experiments/batch_size_vs_rps/results
#   DEVICE_STARTUP_WAIT 5

set -euo pipefail

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SERVING_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SERVING_DIR"

CONDA_ENV="${CONDA_ENV:-fmtk}"

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
    echo "  export FMTK_DIR=/path/to/FMTK"
    exit 1
fi
if [[ ! -d "$FMAAS_DIR" ]]; then
    echo "ERROR: FMAAS_DIR not found at: $FMAAS_DIR"
    echo "  export FMAAS_DIR=/path/to/FMaaS-motivation"
    exit 1
fi

export PYTHONPATH="${FMTK_DIR}/src:${FMAAS_DIR}:${PYTHONPATH:-}"
export DATASET_DIR

if command -v conda &> /dev/null; then
    PYTHON="${PYTHON:-conda run --no-capture-output -n ${CONDA_ENV} python}"
else
    PYTHON="${PYTHON:-python}"
fi

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:-momentbase}"
RPS_SWEEP="${RPS_SWEEP:-100,200,300,400,500}"
WAIT_SWEEP="${WAIT_SWEEP:-5,10}"
PHASE_DURATION="${PHASE_DURATION:-60}"
NUM_CLIENTS="${NUM_CLIENTS:-1}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-single_task}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
DEVICE_PORT="${DEVICE_PORT:-8000}"
RESULTS_BASE="${RESULTS_BASE:-experiments/batch_size_vs_rps/results}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-5}"

LOG_DIR="${RESULTS_BASE}/logs"
mkdir -p "$LOG_DIR"

# Top-level experiment config — written once before the sweep starts
cat > "${RESULTS_BASE}/experiment_config.json" <<EOF
{
  "experiment":          "batch_size_vs_rps",
  "backbone":            "${BACKBONE}",
  "cuda_device":         "${CUDA_DEVICE}",
  "rps_sweep":           "${RPS_SWEEP}",
  "wait_sweep_ms":       "${WAIT_SWEEP}",
  "num_clients":         ${NUM_CLIENTS},
  "experiment_mode":     "${EXPERIMENT_MODE}",
  "phase_duration_s":    ${PHASE_DURATION},
  "max_batch_size":      ${MAX_BATCH_SIZE},
  "scheduler_policy":    "fifo",
  "device_port":         ${DEVICE_PORT},
  "device_startup_wait": ${DEVICE_STARTUP_WAIT},
  "t_start":             "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hostname":            "$(hostname)"
}
EOF

echo "================================================================"
echo "  Experiment: batch_size_vs_rps"
echo "  Conda env      : $CONDA_ENV"
echo "  FMTK_DIR       : $FMTK_DIR"
echo "  FMAAS_DIR      : $FMAAS_DIR"
echo "  Backbone       : $BACKBONE"
echo "  RPS sweep      : $RPS_SWEEP"
echo "  Wait sweep (ms): $WAIT_SWEEP"
echo "  Num clients    : $NUM_CLIENTS"
echo "  Mode           : $EXPERIMENT_MODE"
echo "  Duration/run   : ${PHASE_DURATION}s"
echo "  Max batch size : $MAX_BATCH_SIZE"
echo "  Results        : $RESULTS_BASE"
echo "================================================================"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEVICE_PID=""

stop_device() {
    if [[ -n "$DEVICE_PID" ]]; then
        echo "[run.sh] Stopping device server PID=$DEVICE_PID"
        kill "$DEVICE_PID" 2>/dev/null || true
        wait "$DEVICE_PID" 2>/dev/null || true
        DEVICE_PID=""
    fi
    pkill -f "device/main.py.*--port ${DEVICE_PORT}" 2>/dev/null || true
    sleep 2
}
trap 'stop_device' EXIT

start_device() {
    local wait_ms="$1" log="$2"
    pkill -f "device/main.py.*--port ${DEVICE_PORT}" 2>/dev/null || true
    sleep 1
    echo "[run.sh] Starting device server  wait_ms=${wait_ms}  port=${DEVICE_PORT} ..."
    $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port              "$DEVICE_PORT"   \
        --runtime-type      pytorch          \
        --cuda              "$CUDA_DEVICE"   \
        --scheduler-policy  fifo             \
        --max-batch-size    "$MAX_BATCH_SIZE" \
        --max-batch-wait-ms "$wait_ms"       \
        > "$log" 2>&1 &
    DEVICE_PID=$!
    echo "[run.sh] PID=$DEVICE_PID  log=$log"
    sleep "$DEVICE_STARTUP_WAIT"
}

run_one() {
    local wait_ms="$1" rps="$2"
    local out_dir
    if [[ "$EXPERIMENT_MODE" == "system_tasks" ]]; then
        out_dir="${RESULTS_BASE}/system_tasks/wait_${wait_ms}/rps_${rps}"
    else
        out_dir="${RESULTS_BASE}/clients_${NUM_CLIENTS}/wait_${wait_ms}/rps_${rps}"
    fi
    mkdir -p "$out_dir"

    echo ""
    echo "----------------------------------------------------------------"
    echo "  wait_ms=${wait_ms}  rps=${rps}"
    echo "  Results: $out_dir"
    echo "----------------------------------------------------------------"

    stop_device
    start_device "$wait_ms" "$LOG_DIR/device_wait${wait_ms}_rps${rps}.log"

    if [[ "$EXPERIMENT_MODE" == "system_tasks" ]]; then
        $PYTHON -u experiments/batch_size_vs_rps/run_systeminspired.py \
            --device-url    "localhost:${DEVICE_PORT}" \
            --backbone      "$BACKBONE"               \
            --req-rate      "$rps"                    \
            --duration      "$PHASE_DURATION"         \
            --batch-wait-ms "$wait_ms"                \
            --exp-dir       "$out_dir"
    else
        $PYTHON -u experiments/batch_size_vs_rps/run.py \
            --device-url    "localhost:${DEVICE_PORT}" \
            --backbone      "$BACKBONE"               \
            --rps           "$rps"                    \
            --num-clients   "$NUM_CLIENTS"            \
            --duration      "$PHASE_DURATION"         \
            --batch-wait-ms "$wait_ms"                \
            --exp-dir       "$out_dir"
    fi

    stop_device
}

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
IFS=',' read -ra WAIT_LIST <<< "$WAIT_SWEEP"
IFS=',' read -ra RPS_LIST  <<< "$RPS_SWEEP"

for wait_ms in "${WAIT_LIST[@]}"; do
    echo ""
    echo "################################################################"
    echo "  BATCH_WAIT_MS = $wait_ms"
    echo "################################################################"
    for rps in "${RPS_LIST[@]}"; do
        run_one "$wait_ms" "$rps" \
            || echo "[run.sh] WARNING: wait=${wait_ms} rps=${rps} failed — continuing"
    done
done

echo ""
echo "[run.sh] All done. Results in $RESULTS_BASE"
echo "[run.sh] Plot with:"
if [[ "$EXPERIMENT_MODE" == "system_tasks" ]]; then
    echo "  $PYTHON experiments/batch_size_vs_rps/plot.py --exp-dir ${RESULTS_BASE}/system_tasks --rps-sweep ${RPS_SWEEP} --wait-sweep ${WAIT_SWEEP}"
else
    echo "  $PYTHON experiments/batch_size_vs_rps/plot.py --exp-dir ${RESULTS_BASE}/clients_${NUM_CLIENTS} --rps-sweep ${RPS_SWEEP} --wait-sweep ${WAIT_SWEEP}"
fi
