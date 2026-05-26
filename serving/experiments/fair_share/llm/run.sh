#!/usr/bin/env bash
# fair_share/llm — vLLM multi-LoRA noisy-neighbor experiment (LLM analog of fair_share/tsfm).
#
# Each RUN spawns its own device server (vLLM runtime, two LoRA adapters
# bound to qwenA/qwenB), drives the open-loop phased workload from run.py,
# then tears the server down before the next run.
#
# Prerequisite (one-time): generate two random LoRA adapters —
#     python experiments/fair_share/llm/make_loras.py
#
# Run (from serving/):
#     bash experiments/fair_share/llm/run.sh
#
# BFQ-with-weights against vLLM is intentionally not in RUNS yet:
# device/server.py:69 sets `self.batcher = None` for runtime_type=vllm, so
# --scheduler-policy / --task-rates are ignored on the vLLM path. To run BFQ
# vs vllm-baseline we need an admission scheduler in front of AsyncLLMEngine
# (not yet implemented). The commented-out BFQ entries below are placeholders.

set -euo pipefail

# ---------------------------------------------------------------------------
# Path setup (mirrors fair_share/tsfm/run.sh)
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
BACKBONE="${BACKBONE:-qwen2.5-1.5b}"
VICTIM_TASK="${VICTIM_TASK:-qwenA}"
AGGRESSOR_TASK="${AGGRESSOR_TASK:-qwenB}"

# Three-phase priority pattern, same shape as fair_share/tsfm/run.sh.
# LLM RPS budget is much lower than TSFM — Qwen2.5-3B on T4 is generation-
# bound. Defaults assume ~2–12 RPS aggregate; tune via env vars.
AGGRESSOR_RPS_PHASES="${AGGRESSOR_RPS_PHASES:-16,16,16}"   # was 8,8,8
VICTIM_RPS_PHASES="${VICTIM_RPS_PHASES:-2,2,2}"            # was 1,8,1 — keep constant
PHASE_DURATIONS="${PHASE_DURATIONS:-60,60,60}"
VICTIM_RPS="${VICTIM_RPS:-1}"

# vLLM engine knobs
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Sharing runs: "scheduler  batch_size  batch_wait_ms  run_name  [task_rates_override]"
#   scheduler="vllm"  → no admission scheduler (vLLM owns ordering + batching).
#   scheduler="stfq"  → device/vllm_admission.VLLMAdmissionScheduler in front
#                       of AsyncLLMEngine, reusing STFQPolicy from the tsfm path.
#                       batch_size = admission window (# concurrent in-flight in vLLM).
# Weight convention (same as tsfm): rps in --task-rates → weight = 1/rps, so
# A:1,B:1 = 1:1, A:1,B:2 = 2:1 (victim 2x), A:1,B:3 = 3:1.
RUNS=(
    "vllm   1  0  vllm_baseline      ${VICTIM_TASK}:1,${AGGRESSOR_TASK}:1"
    # "stfq  32  0  bfq_1_1            ${VICTIM_TASK}:1,${AGGRESSOR_TASK}:1"
    # "stfq  32  0  bfq_2_1            ${VICTIM_TASK}:1,${AGGRESSOR_TASK}:2"
    "stfq  32  0  bfq_3_1            ${VICTIM_TASK}:1,${AGGRESSOR_TASK}:3"
)

RESULTS_BASE="${RESULTS_BASE:-experiments/fair_share/llm/results_t4}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-60}"  # vLLM model load is slow

if command -v conda &> /dev/null; then
    CONDA_ENV="${CONDA_ENV:-fmtk}"
    PYTHON="${PYTHON:-conda run --no-capture-output -n ${CONDA_ENV} python}"
else
    PYTHON="${PYTHON:-python}"
fi

# ---------------------------------------------------------------------------
# Resolve phase durations
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
        echo "  $rname: scheduler=$sched${rates:+ rates=$rates}"
    done
    echo "Results base: $RESULTS_BASE"
} > "$CONFIG_FILE"

echo "================================================================"
echo "  fair_share/llm — vLLM multi-LoRA saturation experiment"
echo "  Backbone     : $BACKBONE  (max_model_len=$MAX_MODEL_LEN, gpu_mem_util=$GPU_MEM_UTIL)"
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
    echo "    $rname: scheduler=$sched${rates:+ rates=$rates}"
done
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
    local deadline=$((SECONDS + 20))  # vLLM shutdown can be slower
    for pid in "${pids_to_wait[@]:-}"; do
        while kill -0 "$pid" 2>/dev/null; do
            if [[ $SECONDS -ge $deadline ]]; then
                echo "[run.sh] WARNING: PID=$pid did not exit in 20s — SIGKILL" >&2
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
#   scheduler="vllm"  → --runtime-type vllm, no admission scheduler (server
#                       falls through to runtime.infer; vLLM owns ordering).
#   scheduler="stfq"  → --runtime-type vllm + --scheduler-policy stfq +
#                       --task-rates → VLLMAdmissionScheduler activates with
#                       batch_size as the admission window.
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
    pkill -f "fair_share/llm/run.py" 2>/dev/null || true
    sleep 1

    # Always vLLM runtime for the LLM experiment. Admission activates iff
    # scheduler ∈ {stfq,wfq} AND task_rates are provided (see server.py).
    local sched_policy
    if [[ "$scheduler" == "vllm" ]]; then
        sched_policy="fifo"
    else
        sched_policy="$scheduler"
    fi

    echo "[run.sh] Starting device server (runtime=vllm, scheduler=$sched_policy, "\
"bsize=$batch_size, bwait=${batch_wait}ms, rates=$task_rates)..."

    $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port                   "$DEVICE_PORT"  \
        --runtime-type           vllm            \
        --cuda                   "$CUDA_DEVICE"  \
        --scheduler-policy       "$sched_policy" \
        --task-rates             "$task_rates"   \
        --max-batch-size         "$batch_size"   \
        --max-batch-wait-ms      "$batch_wait"   \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len          "$MAX_MODEL_LEN" \
        > "$log" 2>&1 &
    DEVICE_PIDS+=("$!")
    ACTIVE_PORTS+=("$DEVICE_PORT")
    echo "[run.sh] Device PID=$!  log=$log"
    echo "[run.sh] Waiting up to ${DEVICE_STARTUP_WAIT}s for vLLM model load..."

    # Poll the log for an engine-ready signal instead of a fixed sleep.
    local waited=0
    while (( waited < DEVICE_STARTUP_WAIT )); do
        if grep -q -E "Runtime application started|gRPC server.*listen|application started" "$log" 2>/dev/null; then
            echo "[run.sh] Device server reports ready (after ${waited}s)"
            sleep 2
            return 0
        fi
        if ! kill -0 "${DEVICE_PIDS[-1]}" 2>/dev/null; then
            echo "[run.sh] ERROR: device server exited early — check $log" >&2
            tail -n 40 "$log" >&2 || true
            return 1
        fi
        sleep 2
        waited=$(( waited + 2 ))
    done
    echo "[run.sh] WARNING: ready signal not seen in ${DEVICE_STARTUP_WAIT}s, proceeding anyway"
}

# ---------------------------------------------------------------------------
# Main loop
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

    $PYTHON -u experiments/fair_share/llm/run.py \
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

echo ""
echo "[run.sh] All runs done. Results in $RESULTS_BASE"
