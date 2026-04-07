#!/bin/bash
# Sharing Benefit + TPC Isolation experiment
# Runs five conditions per task set.
#
# Task sets:
#   tsfm   — single_ecgclass, single_gestureclass, no_sharing_tpc, no_sharing, sharing
#   vision — single_nyudepth, single_vocseg,       no_sharing_tpc, no_sharing, sharing
#
# Environment variables (all optional):
#   CONDA_ENV          fmtk
#   FMTK_DIR           ../../../FMTK
#   FMAAS_DIR          ../..
#   CUDA_DEVICE        cuda:0
#   TASK_SET           tsfm  (or vision)
#   BACKBONE           momentbase  (tsfm) / dinobase-patch  (vision)
#   RPS_SWEEP          20,40,60
#   PHASE_DURATION     180
#   DEVICE_PORT        8000
#   DEVICE_PORT_2      8001
#   MAX_BATCH_SIZE     5
#   TPC_MODE           libsmctrl (or green)
#   RESULTS_BASE       experiments/sharing_benefit/tpc/results
#   NYUDEPTH_PATH      ../../FMTK/dataset/nyu-depth-v2   (vision only)
#   PASCALVOC_PATH     ../../FMTK/dataset/PASCAL-VOC      (vision only)

set -euo pipefail

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SERVING_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
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

if command -v conda &> /dev/null; then
    PYTHON="${PYTHON:-conda run --no-capture-output -n ${CONDA_ENV} python}"
else
    PYTHON="${PYTHON:-python}"
fi

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
TASK_SET="${TASK_SET:-tsfm}"
RPS_SWEEP="${RPS_SWEEP:-25}"
PHASE_DURATION="${PHASE_DURATION:-300}"
DEVICE_PORT="${DEVICE_PORT:-8000}"
DEVICE_PORT_2="${DEVICE_PORT_2:-8001}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-100}"
RESULTS_BASE="${RESULTS_BASE:-experiments/sharing_benefit/tpc/results_momentbase}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-5}"
MAX_BATCH_WAIT_MS="${MAX_BATCH_WAIT_MS:-0}"
TPC_MODE="${TPC_MODE:-libsmctrl}"

# Task-set-specific defaults
if [[ "$TASK_SET" == "vision" ]]; then
    BACKBONE="${BACKBONE:-dinobase-patch}"
    DECODER_DIR="${DECODER_DIR:-${FMTK_DIR}/models/vision/finetuned}"
    TASK_RATES_TEMPLATE="nyudepth:{rps},vocseg:{rps}"
    CONDITIONS=("single_nyudepth" "single_vocseg" "no_sharing_tpc" "no_sharing" "sharing")
    export NYUDEPTH_PATH="${NYUDEPTH_PATH:-../../FMTK/dataset/nyu-depth-v2}"
    export PASCALVOC_PATH="${PASCALVOC_PATH:-../../FMTK/dataset/PASCAL-VOC}"
else
    BACKBONE="${BACKBONE:-momentlarge}"
    DECODER_DIR="${DECODER_DIR:-${FMTK_DIR}/models/tsfm/finetuned}"
    TASK_RATES_TEMPLATE="ecgclass:{rps},gestureclass:{rps}"
    CONDITIONS=("single_ecgclass" "single_gestureclass" "no_sharing_tpc" "no_sharing" "sharing")
fi

LOG_DIR="${RESULTS_BASE}/logs"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  Sharing Benefit + TPC Isolation"
echo "  Conda env      : $CONDA_ENV"
echo "  FMTK_DIR       : $FMTK_DIR"
echo "  FMAAS_DIR      : $FMAAS_DIR"
echo "  Task set       : $TASK_SET"
echo "  Backbone       : $BACKBONE"
echo "  RPS sweep      : $RPS_SWEEP"
echo "  Duration/run   : ${PHASE_DURATION}s"
echo "  TPC mode       : $TPC_MODE"
echo "  Results        : $RESULTS_BASE"
echo "================================================================"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEVICE_PID=""
DEVICE_PID_2=""

stop_devices() {
    for pid_var in DEVICE_PID DEVICE_PID_2; do
        local pid="${!pid_var:-}"
        if [[ -n "$pid" ]]; then
            echo "[run.sh] Stopping device server PID=$pid"
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    DEVICE_PID=""
    DEVICE_PID_2=""
    pkill -f "device/main.py.*--port ${DEVICE_PORT}"  2>/dev/null || true
    pkill -f "device/main.py.*--port ${DEVICE_PORT_2}" 2>/dev/null || true
    sleep 2
}
trap 'stop_devices' EXIT

start_device() {
    local port="$1" scheduler="$2" log="$3" rps="$4"
    local tpc_mode="${5:-none}" tpc_partition="${6:-}"
    local task_rates="${TASK_RATES_TEMPLATE//\{rps\}/${rps}}"
    pkill -f "device/main.py.*--port ${port}" 2>/dev/null || true
    sleep 1
    local tpc_args=""
    if [[ "$tpc_mode" != "none" && -n "$tpc_partition" ]]; then
        tpc_args="--tpc-mode $tpc_mode --tpc-partition $tpc_partition"
    fi
    echo "[run.sh] Starting device server port=$port scheduler=$scheduler tpc=$tpc_mode ..."
    $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port              "$port"          \
        --runtime-type      pytorch          \
        --cuda              "$CUDA_DEVICE"   \
        --scheduler-policy  "$scheduler"     \
        --max-batch-size    "$MAX_BATCH_SIZE" \
        --max-batch-wait-ms "$MAX_BATCH_WAIT_MS" \
        --task-rates        "$task_rates"    \
        $tpc_args \
        > "$log" 2>&1 &
    local pid=$!
    echo "[run.sh] PID=$pid  log=$log"
    sleep "$DEVICE_STARTUP_WAIT"
    echo "$pid"
}

run_condition() {
    local condition="$1" rps="$2"
    local out_dir="${RESULTS_BASE}/rps_${rps}/${condition}"
    local trace_file="${RESULTS_BASE}/rps_${rps}/trace.json"

    echo ""
    echo "================================================================"
    echo "  condition=$condition  rps=$rps"
    echo "  Results: $out_dir"
    echo "================================================================"

    stop_devices
    mkdir -p "$out_dir"

    # Helper to invoke run.py with common args + task-set
    run_py() {
        $PYTHON -u experiments/sharing_benefit/tpc/run.py \
            --task-set     "$TASK_SET" \
            --backbone     "$BACKBONE" \
            --rps          "$rps" \
            --duration     "$PHASE_DURATION" \
            --exp-dir      "$out_dir" \
            --trace-file   "$trace_file" \
            "$@"
    }

    case "$condition" in
        single_ecgclass|single_nyudepth)
            DEVICE_PID=$(start_device "$DEVICE_PORT" "fifo" "$LOG_DIR/device_${condition}_rps${rps}.log" "$rps")
            run_py --condition "$condition" --device-url "localhost:${DEVICE_PORT}"
            ;;
        single_gestureclass|single_vocseg)
            DEVICE_PID=$(start_device "$DEVICE_PORT" "fifo" "$LOG_DIR/device_${condition}_rps${rps}.log" "$rps")
            run_py --condition "$condition" --device-url "localhost:${DEVICE_PORT}"
            ;;
        no_sharing_tpc)
            # Query total TPCs, split in half for two servers
            TOTAL_TPCS=$($PYTHON -c "
import torch
sm = torch.cuda.get_device_properties('${CUDA_DEVICE}').multi_processor_count
print(sm // 2)  # TPCs ~ SMs/2
")
            HALF=$((TOTAL_TPCS / 2))
            # Build partition lists: server1 gets TPCs 0..HALF-1, server2 gets HALF..TOTAL-1
            PART1=$(seq -s ' ' 0 $((HALF - 1)))
            PART2=$(seq -s ' ' $HALF $((TOTAL_TPCS - 1)))
            echo "[run.sh] TPC split: server1=[${PART1}]  server2=[${PART2}]"

            DEVICE_PID=$(start_device   "$DEVICE_PORT"   "fifo" "$LOG_DIR/device_${condition}_1_rps${rps}.log" "$rps" "$TPC_MODE" "$PART1")
            DEVICE_PID_2=$(start_device "$DEVICE_PORT_2" "fifo" "$LOG_DIR/device_${condition}_2_rps${rps}.log" "$rps" "$TPC_MODE" "$PART2")
            run_py --condition no_sharing_tpc \
                --device-url   "localhost:${DEVICE_PORT}" \
                --device-url-2 "localhost:${DEVICE_PORT_2}"
            ;;
        no_sharing)
            DEVICE_PID=$(start_device   "$DEVICE_PORT"   "fifo" "$LOG_DIR/device_${condition}_1_rps${rps}.log" "$rps")
            DEVICE_PID_2=$(start_device "$DEVICE_PORT_2" "fifo" "$LOG_DIR/device_${condition}_2_rps${rps}.log" "$rps")
            run_py --condition no_sharing \
                --device-url   "localhost:${DEVICE_PORT}" \
                --device-url-2 "localhost:${DEVICE_PORT_2}"
            ;;
        sharing)
            DEVICE_PID=$(start_device "$DEVICE_PORT" "stfq" "$LOG_DIR/device_${condition}_rps${rps}.log" "$rps")
            run_py --condition sharing --device-url "localhost:${DEVICE_PORT}"
            ;;
    esac

    cat > "${out_dir}/run_config.json" <<EOF
{
  "condition": "${condition}",
  "task_set": "${TASK_SET}",
  "backbone": "${BACKBONE}",
  "cuda_device": "${CUDA_DEVICE}",
  "max_batch_size": ${MAX_BATCH_SIZE},
  "max_batch_wait_ms": ${MAX_BATCH_WAIT_MS},
  "phase_duration_s": ${PHASE_DURATION},
  "rps_per_task": ${rps},
  "tpc_mode": "${TPC_MODE}",
  "device_port": ${DEVICE_PORT},
  "device_port_2": ${DEVICE_PORT_2},
  "device_startup_wait_s": ${DEVICE_STARTUP_WAIT}
}
EOF

    stop_devices
}

# Sweep RPS values, run all five conditions per RPS
IFS=',' read -ra RPS_LIST <<< "$RPS_SWEEP"
for rps in "${RPS_LIST[@]}"; do
    echo ""
    echo "################################################################"
    echo "  RPS = $rps  (task_set=$TASK_SET)"
    echo "################################################################"
    for condition in "${CONDITIONS[@]}"; do
        run_condition "$condition" "$rps" \
            || echo "[run.sh] WARNING: $condition rps=$rps failed — continuing"
    done
done

echo ""
echo "[run.sh] All done. Results in $RESULTS_BASE"
