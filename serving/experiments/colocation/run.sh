#!/bin/bash
# Colocation experiment: ecgclass (momentlarge) + nyudepth (dinobase)
#
# Conditions per RPS:
#   single_ecgclass   — 1 device server, momentlarge + ecgclass
#   single_nyudepth   — 1 device server, dinobase + nyudepth
#   no_sharing        — 2 device servers (both backbones), running concurrently
#
# Environment variables (all optional):
#   CONDA_ENV          fmtk
#   FMTK_DIR           ../../FMTK
#   FMAAS_DIR          ..
#   CUDA_DEVICE        cuda:0
#   RPS_SWEEP          1,5,10
#   PHASE_DURATION     600
#   DEVICE_PORT        8000 (base; second server uses 8001)
#   MAX_BATCH_SIZE     1
#   TSFM_BACKBONE      momentlarge
#   VISION_BACKBONE    dinobase-patch
#   RESULTS_BASE       experiments/colocation/results
#   NYUDEPTH_PATH      ../../FMTK/dataset/nyu-depth-v2

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

[[ -d "$FMTK_DIR"  ]] || { echo "ERROR: FMTK_DIR not found: $FMTK_DIR"; exit 1; }
[[ -d "$FMAAS_DIR" ]] || { echo "ERROR: FMAAS_DIR not found: $FMAAS_DIR"; exit 1; }

export PYTHONPATH="${FMTK_DIR}/src:${FMAAS_DIR}:${PYTHONPATH:-}"
export DATASET_DIR

if command -v conda &> /dev/null; then
    PYTHON="${PYTHON:-conda run --no-capture-output -n ${CONDA_ENV} python}"
else
    PYTHON="${PYTHON:-python}"
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
RPS_SWEEP="${RPS_SWEEP:-1,5,10}"
PHASE_DURATION="${PHASE_DURATION:-190}"
DEVICE_PORT="${DEVICE_PORT:-8000}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
MAX_BATCH_WAIT_MS="${MAX_BATCH_WAIT_MS:-0}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-5}"
WARMUP_BURST_SECS="${WARMUP_BURST_SECS:-15}"

TSFM_BACKBONE="${TSFM_BACKBONE:-momentlarge}"
VISION_BACKBONE="${VISION_BACKBONE:-dinobase-patch}"

RESULTS_BASE="${RESULTS_BASE:-experiments/colocation/results}"

export NYUDEPTH_PATH="${NYUDEPTH_PATH:-../../FMTK/dataset/nyu-depth-v2}"

TASKS=(ecgclass nyudepth)
TASK_BACKBONES="ecgclass:${TSFM_BACKBONE},nyudepth:${VISION_BACKBONE}"
TASKS_CSV="ecgclass,nyudepth"

LOG_DIR="${RESULTS_BASE}/logs"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  Colocation"
echo "  Conda env      : $CONDA_ENV"
echo "  FMTK_DIR       : $FMTK_DIR"
echo "  FMAAS_DIR      : $FMAAS_DIR"
echo "  Tasks          : ${TASKS[*]}"
echo "  Backbones      : $TASK_BACKBONES"
echo "  RPS sweep      : $RPS_SWEEP"
echo "  Duration/run   : ${PHASE_DURATION}s"
echo "  Max batch size : $MAX_BATCH_SIZE"
echo "  Warmup burst   : ${WARMUP_BURST_SECS}s"
echo "  Base port      : $DEVICE_PORT"
echo "  Results        : $RESULTS_BASE"
echo "================================================================"

# ---------------------------------------------------------------------------
# Device server lifecycle
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
                echo "[run.sh] WARNING: PID=$pid did not exit in 12s — SIGKILL" >&2
                kill -9 "$pid" 2>/dev/null || true
                break
            fi
            sleep 0.5
        done
        wait "$pid" 2>/dev/null || true
    done
    for port in "${ACTIVE_PORTS[@]:-}"; do
        if pkill -f "device/main.py.*--port ${port}" 2>/dev/null; then
            echo "[run.sh] pkill fallback killed orphan on port $port" >&2
            sleep 1
        fi
    done
    DEVICE_PIDS=()
    ACTIVE_PORTS=()
    sleep 2
}
trap 'stop_devices' EXIT

# start_device PORT LOG TASK_RATES
start_device() {
    local port="$1" log="$2" task_rates="$3"
    pkill -f "device/main.py.*--port ${port}" 2>/dev/null || true
    sleep 1

    echo "[run.sh] Starting device server port=$port task_rates=$task_rates ..." >&2
    $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port              "$port"              \
        --runtime-type      pytorch              \
        --cuda              "$CUDA_DEVICE"       \
        --scheduler-policy  fifo                 \
        --max-batch-size    "$MAX_BATCH_SIZE"    \
        --max-batch-wait-ms "$MAX_BATCH_WAIT_MS" \
        --task-rates        "$task_rates"        \
        --worker-mode       inline               \
        > "$log" 2>&1 &
    local pid=$!
    echo "[run.sh] PID=$pid  log=$log" >&2
    ACTIVE_PORTS+=("$port")
    sleep "$DEVICE_STARTUP_WAIT"
    echo "$pid"
}

# ---------------------------------------------------------------------------
# run_condition CONDITION RPS
# ---------------------------------------------------------------------------
abs_results="$(cd "$SERVING_DIR" && realpath -m "$RESULTS_BASE")"
mkdir -p "$abs_results"

run_condition() {
    local condition="$1" rps="$2"

    local out_dir trace_file
    local rps_dir="${abs_results}/rps_${rps}"
    trace_file="${rps_dir}/trace.json"
    out_dir="${rps_dir}/${condition}"

    echo ""
    echo "================================================================"
    echo "  condition=$condition  rps=$rps"
    echo "  Results : $out_dir"
    echo "================================================================"

    if [[ -f "${out_dir}/latencies.csv" ]]; then
        echo "[run.sh] Skipping $condition — results already exist"
        return 0
    fi

    stop_devices
    DEVICE_PIDS=()
    ACTIVE_PORTS=()
    mkdir -p "$out_dir"

    local run_py_args=(
        --tasks           "$TASKS_CSV"
        --task-backbones  "$TASK_BACKBONES"
        --rps             "$rps"
        --duration        "$PHASE_DURATION"
        --exp-dir         "$out_dir"
        --trace-file      "$trace_file"
        --warmup-burst-secs "$WARMUP_BURST_SECS"
    )

    case "$condition" in
        single_ecgclass)
            local port="$DEVICE_PORT"
            local pid
            pid=$(start_device "$port" \
                  "$LOG_DIR/device_single_ecgclass_rps${rps}.log" \
                  "ecgclass:${rps}")
            DEVICE_PIDS+=("$pid")
            $PYTHON -u experiments/colocation/run.py \
                --condition   single_ecgclass \
                --device-urls "localhost:${port}" \
                "${run_py_args[@]}"
            ;;
        single_nyudepth)
            local port="$DEVICE_PORT"
            local pid
            pid=$(start_device "$port" \
                  "$LOG_DIR/device_single_nyudepth_rps${rps}.log" \
                  "nyudepth:${rps}")
            DEVICE_PIDS+=("$pid")
            $PYTHON -u experiments/colocation/run.py \
                --condition   single_nyudepth \
                --device-urls "localhost:${port}" \
                "${run_py_args[@]}"
            ;;
        no_sharing)
            local port0=$DEVICE_PORT
            local port1=$((DEVICE_PORT + 1))
            local pid0 pid1
            pid0=$(start_device "$port0" \
                   "$LOG_DIR/device_ns_ecgclass_rps${rps}.log" \
                   "ecgclass:${rps}")
            DEVICE_PIDS+=("$pid0")
            pid1=$(start_device "$port1" \
                   "$LOG_DIR/device_ns_nyudepth_rps${rps}.log" \
                   "nyudepth:${rps}")
            DEVICE_PIDS+=("$pid1")
            $PYTHON -u experiments/colocation/run.py \
                --condition   no_sharing \
                --device-urls "localhost:${port0},localhost:${port1}" \
                "${run_py_args[@]}"
            ;;
        *)
            echo "[run.sh] ERROR: unknown condition '$condition'"
            return 1
            ;;
    esac

    cat > "${out_dir}/run_config.json" <<EOF
{
  "condition": "${condition}",
  "tasks": ["ecgclass", "nyudepth"],
  "task_backbones": "${TASK_BACKBONES}",
  "cuda_device": "${CUDA_DEVICE}",
  "max_batch_size": ${MAX_BATCH_SIZE},
  "max_batch_wait_ms": ${MAX_BATCH_WAIT_MS},
  "phase_duration_s": ${PHASE_DURATION},
  "rps_per_task": ${rps},
  "device_port_base": ${DEVICE_PORT},
  "device_startup_wait_s": ${DEVICE_STARTUP_WAIT}
}
EOF

    stop_devices
}

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
IFS=',' read -ra RPS_LIST <<< "$RPS_SWEEP"

CONDITIONS=(single_ecgclass single_nyudepth no_sharing)

for rps in "${RPS_LIST[@]}"; do
    echo ""
    echo "################################################################"
    echo "  RPS=$rps"
    echo "################################################################"
    for condition in "${CONDITIONS[@]}"; do
        run_condition "$condition" "$rps" \
            || echo "[run.sh] WARNING: $condition rps=$rps failed — continuing"
    done
done

echo ""
echo "[run.sh] All done. Results in $RESULTS_BASE"