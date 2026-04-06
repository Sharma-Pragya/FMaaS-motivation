#!/bin/bash
# Closed-loop ecgclass TPC sweep.
#
# Sweeps TPC counts:
#   total_num_tpcs
#   total_num_tpcs / 2
#   total_num_tpcs / 4
#
# Runs a single clmean_service_time_msosed-loop client against a single ecgclass server for each
# TPC budget and records latency/response-time breakdowns.

set -euo pipefail

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
    exit 1
fi
if [[ ! -d "$FMAAS_DIR" ]]; then
    echo "ERROR: FMAAS_DIR not found at: $FMAAS_DIR"
    exit 1
fi

export PYTHONPATH="${FMTK_DIR}/src:${FMAAS_DIR}:${PYTHONPATH:-}"
export DATASET_DIR

if command -v conda &> /dev/null; then
    PYTHON="${PYTHON:-conda run --no-capture-output -n ${CONDA_ENV} python}"
else
    PYTHON="${PYTHON:-python}"
fi

CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:-chronoslarge}"
PHASE_DURATION="${PHASE_DURATION:-60}"
WARMUP_SECS="${WARMUP_SECS:-5}"
CONCURRENCY="${CONCURRENCY:-1}"
DEVICE_PORT="${DEVICE_PORT:-8000}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-100}"
MAX_BATCH_WAIT_MS="${MAX_BATCH_WAIT_MS:-0}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-5}"
RESULTS_BASE="${RESULTS_BASE:-experiments/tpc_closed_loop_ecg/results}"
TPC_MODE="${TPC_MODE:-libsmctrl}"

LOG_DIR="${RESULTS_BASE}/logs"
mkdir -p "$LOG_DIR"

DEVICE_PID=""

stop_device() {
    if [[ -n "${DEVICE_PID:-}" ]]; then
        echo "[run.sh] Stopping device server PID=$DEVICE_PID"
        kill "$DEVICE_PID" 2>/dev/null || true
        wait "$DEVICE_PID" 2>/dev/null || true
    fi
    DEVICE_PID=""
    pkill -f "device/main.py.*--port ${DEVICE_PORT}" 2>/dev/null || true
    sleep 2
}
trap 'stop_device' EXIT

TOTAL_TPCS=$($PYTHON -c "import torch; sm=torch.cuda.get_device_properties('${CUDA_DEVICE}').multi_processor_count; print(max(1, sm // 2))")
HALF_TPCS=$(( TOTAL_TPCS / 2 ))
QUARTER_TPCS=$(( TOTAL_TPCS / 4 ))
EITHER_TPCS=$(( TOTAL_TPCS / 8 ))
if [[ "$HALF_TPCS" -lt 1 ]]; then HALF_TPCS=1; fi
if [[ "$QUARTER_TPCS" -lt 1 ]]; then QUARTER_TPCS=1; fi
if [[ "$EITHER_TPCS" -lt 1 ]]; then EITHER_TPCS=1; fi

TPC_COUNTS=()
for count in "1" "$TOTAL_TPCS" "$HALF_TPCS" "$QUARTER_TPCS" "$EITHER_TPCS"; do
    skip=0
    for seen in "${TPC_COUNTS[@]:-}"; do
        if [[ "$seen" == "$count" ]]; then
            skip=1
            break
        fi
    done
    if [[ "$skip" -eq 0 ]]; then
        TPC_COUNTS+=("$count")
    fi
done

echo "================================================================"
echo "  Closed-Loop ECG TPC Sweep"
echo "  Conda env      : $CONDA_ENV"
echo "  FMTK_DIR       : $FMTK_DIR"
echo "  FMAAS_DIR      : $FMAAS_DIR"
echo "  Backbone       : $BACKBONE"
echo "  CUDA device    : $CUDA_DEVICE"
echo "  Duration/run   : ${PHASE_DURATION}s"
echo "  Warmup         : ${WARMUP_SECS}s"
echo "  Concurrency    : $CONCURRENCY"
echo "  TPC mode       : $TPC_MODE"
echo "  Total TPCs     : $TOTAL_TPCS"
echo "  Sweep TPCs     : ${TPC_COUNTS[*]}"
echo "  Results        : $RESULTS_BASE"
echo "================================================================"

start_device() {
    local tpc_count="$1"
    local log_file="$2"
    local partition
    partition=$(seq -s ' ' 0 $((tpc_count - 1)))

    pkill -f "device/main.py.*--port ${DEVICE_PORT}" 2>/dev/null || true
    sleep 1
    echo "[run.sh] Starting device server port=$DEVICE_PORT tpcs=$tpc_count partition=[$partition]"
    $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port "$DEVICE_PORT" \
        --runtime-type pytorch \
        --cuda "$CUDA_DEVICE" \
        --scheduler-policy fifo \
        --max-batch-size "$MAX_BATCH_SIZE" \
        --max-batch-wait-ms "$MAX_BATCH_WAIT_MS" \
        --tpc-mode "$TPC_MODE" \
        --tpc-partition $partition \
        > "$log_file" 2>&1 &
    DEVICE_PID=$!
    echo "[run.sh] PID=$DEVICE_PID log=$log_file"
    sleep "$DEVICE_STARTUP_WAIT"
}

run_case() {
    local tpc_count="$1"
    local out_dir="${RESULTS_BASE}/tpc_${tpc_count}"
    local log_file="${LOG_DIR}/device_tpc${tpc_count}.log"

    echo ""
    echo "================================================================"
    echo "  tpc_count=$tpc_count  concurrency=$CONCURRENCY"
    echo "  Results: $out_dir"
    echo "================================================================"

    stop_device
    mkdir -p "$out_dir"
    start_device "$tpc_count" "$log_file"

    $PYTHON -u experiments/tpc_closed_loop_ecg/run.py \
        --device-url "localhost:${DEVICE_PORT}" \
        --backbone "$BACKBONE" \
        --concurrency "$CONCURRENCY" \
        --duration "$PHASE_DURATION" \
        --warmup-secs "$WARMUP_SECS" \
        --tpc-count "$tpc_count" \
        --exp-dir "$out_dir"

    cat > "${out_dir}/run_config.json" <<EOF
{
  "task": "ecgclass",
  "backbone": "${BACKBONE}",
  "cuda_device": "${CUDA_DEVICE}",
  "tpc_mode": "${TPC_MODE}",
  "tpc_count": ${tpc_count},
  "concurrency": ${CONCURRENCY},
  "phase_duration_s": ${PHASE_DURATION},
  "warmup_s": ${WARMUP_SECS},
  "device_port": ${DEVICE_PORT},
  "max_batch_size": ${MAX_BATCH_SIZE},
  "max_batch_wait_ms": ${MAX_BATCH_WAIT_MS},
  "device_startup_wait_s": ${DEVICE_STARTUP_WAIT}
}
EOF

    stop_device
}

for tpc_count in "${TPC_COUNTS[@]}"; do
    run_case "$tpc_count" \
        || echo "[run.sh] WARNING: tpc_count=$tpc_count failed"
done

echo ""
echo "[run.sh] All done. Results in $RESULTS_BASE"
