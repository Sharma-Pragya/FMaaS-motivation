#!/usr/bin/env bash
# isolation_overhead — Compare none / shared / process isolation modes.
#
# Starts a fresh device server per mode, runs closed-loop requests,
# appends results to a single summary.csv.
#
# Run from serving/:
#   bash experiments/isolation_overhead/run.sh
#
# Run a single mode:
#   ISOLATION_MODES="shared" bash experiments/isolation_overhead/run.sh
#
# Sweep multiple TSFM backbones:
#   TSFM_BACKBONES="momentbase momentlarge" bash experiments/isolation_overhead/run.sh
#
# Vision only:
#   TSFM_BACKBONES="" VISION_BACKBONES="dinov2small" VISION_TASKS="nyudepth vocseg" \
#     bash experiments/isolation_overhead/run.sh
#
# Single mode, single (backbone, task):
#   TSFM_BACKBONES="momentbase" TSFM_TASKS="ecgclass" VISION_BACKBONES="" \
#     ISOLATION_MODES="none" bash experiments/isolation_overhead/run.sh

set -euo pipefail

SERVING_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SERVING_DIR"

# Conda environment for Python
CONDA_ENV="${CONDA_ENV:-fmtk}"

# Project directories (can be relative or absolute)
# Default: FMTK is sibling directory of FMaaS-motivation, FMAAS_DIR is parent of serving
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
    echo "Please set: export FMTK_DIR=/path/to/FMTK"
    exit 1
fi
if [[ ! -d "$FMAAS_DIR" ]]; then
    echo "ERROR: FMAAS_DIR not found at: $FMAAS_DIR"
    echo "Please set: export FMAAS_DIR=/path/to/FMaaS-motivation"
    exit 1
fi

export PYTHONPATH="${FMTK_DIR}/src:${FMAAS_DIR}:${PYTHONPATH:-}"

# ── Config ────────────────────────────────────────────────────────────────
# Mode → (scheduler_policy, worker_mode) wiring:
#   shared  (fmvisor) → stfq         + threaded
#   none              → round_robin  + inline   (batcher bypassed; cosmetic)
#   process           → round_robin  + inline
ISOLATION_MODES="${ISOLATION_MODES:-none shared}"
# Sweep groups: each group runs every (backbone × task) combination.
# Override TSFM_BACKBONES/TSFM_TASKS or VISION_BACKBONES/VISION_TASKS to add/skip.
# To skip a whole modality, set its BACKBONES to "" (e.g. VISION_BACKBONES="").
TSFM_BACKBONES="${TSFM_BACKBONES:-"momentlarge papageip"}"
TSFM_TASKS="${TSFM_TASKS:-ecgclass}"
VISION_BACKBONES="${VISION_BACKBONES:-"dinobase swinlarge"}"
VISION_TASKS="${VISION_TASKS:-nyudepth}"
DEVICE_PORT="${DEVICE_PORT:-8000}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
DURATION="${DURATION:-60}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
MAX_BATCH_WAIT_MS="${MAX_BATCH_WAIT_MS:-0}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-10}"
EXP_DIR="${EXP_DIR:-experiments/isolation_overhead/results}"

# Python executable: prefer conda run, fall back to plain python
if command -v conda &> /dev/null; then
    PYTHON="${PYTHON:-conda run --no-capture-output -n ${CONDA_ENV} python}"
else
    PYTHON="${PYTHON:-python}"
fi

LOG_DIR="experiments/isolation_overhead/logs"
mkdir -p "$LOG_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  isolation_overhead experiment"
echo "════════════════════════════════════════════════════════════════"
echo "  Modes      : $ISOLATION_MODES"
echo "  TSFM       : backbones=[$TSFM_BACKBONES]  tasks=[$TSFM_TASKS]"
echo "  Vision     : backbones=[$VISION_BACKBONES]  tasks=[$VISION_TASKS]"
echo "  Duration   : ${DURATION}s"
echo "  Summary    : $EXP_DIR/summary.csv"
echo "════════════════════════════════════════════════════════════════"

DEVICE_PID=""

mode_settings() {
    # Echoes "<scheduler_policy> <worker_mode>" for the given isolation mode.
    case "$1" in
        shared)  echo "stfq        threaded" ;;
        none)    echo "round_robin inline"   ;;
        process) echo "round_robin inline"   ;;
        *) echo "[ERR] unknown isolation mode: $1" >&2; exit 2 ;;
    esac
}

start_device() {
    local mode="$1"
    local sched worker
    read -r sched worker <<< "$(mode_settings "$mode")"
    local log="$LOG_DIR/device_${mode}_${BACKBONE}_${TASK}.log"
    echo "[INFO] Starting device (mode=$mode sched=$sched worker=$worker) on port $DEVICE_PORT ..."
    $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port              "$DEVICE_PORT"       \
        --cuda              "$CUDA_DEVICE"       \
        --runtime-type      pytorch              \
        --max-batch-size    "$MAX_BATCH_SIZE"    \
        --max-batch-wait-ms "$MAX_BATCH_WAIT_MS" \
        --scheduler-policy  "$sched"             \
        --worker-mode       "$worker"            \
        --isolation-mode    "$mode"              \
        > "$log" 2>&1 &
    DEVICE_PID=$!
    echo "[INFO] Device PID=$DEVICE_PID  log=$log"
    sleep "$DEVICE_STARTUP_WAIT"
}

stop_device() {
    if [[ -n "${DEVICE_PID:-}" ]]; then
        echo "[INFO] Stopping device (PID=$DEVICE_PID)"
        kill "$DEVICE_PID" 2>/dev/null || true
        wait "$DEVICE_PID" 2>/dev/null || true
        DEVICE_PID=""
    fi
    pkill -f "device/main.py.*--port ${DEVICE_PORT}" 2>/dev/null || true
    sleep 2
}

trap 'stop_device' EXIT

run_group() {
    # $1 = group label (tsfm/vision); $2 = backbones; $3 = tasks
    local label="$1" backbones="$2" tasks="$3"
    [[ -z "$backbones" || -z "$tasks" ]] && { echo "[INFO] Skipping $label (empty backbones or tasks)"; return; }

    for BACKBONE in $backbones; do
        for TASK in $tasks; do
            for MODE in $ISOLATION_MODES; do
                echo ""
                echo "════════════════════════════════════════════════════════════════"
                echo "[RUN] group=$label backbone=$BACKBONE task=$TASK isolation_mode=$MODE"
                echo "════════════════════════════════════════════════════════════════"

                start_device "$MODE"

                $PYTHON -u experiments/isolation_overhead/run.py \
                    --device-url     "localhost:${DEVICE_PORT}" \
                    --backbone       "$BACKBONE"                \
                    --task           "$TASK"                    \
                    --duration       "$DURATION"                \
                    --isolation-mode "$MODE"                    \
                    --exp-dir        "$EXP_DIR"

                stop_device
                echo "[INFO] [$label/$BACKBONE/$TASK/$MODE] done. Pausing 3s ..."
                sleep 3
            done
        done
    done
}

run_group "tsfm"   "$TSFM_BACKBONES"   "$TSFM_TASKS"
run_group "vision" "$VISION_BACKBONES" "$VISION_TASKS"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "[INFO] All modes complete."
echo "[INFO] Results: $EXP_DIR/summary.csv"
echo "════════════════════════════════════════════════════════════════"
