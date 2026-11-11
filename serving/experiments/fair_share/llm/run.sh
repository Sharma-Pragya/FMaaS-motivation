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
BACKBONE="${BACKBONE:-qwen2.5-3b}"
VICTIM_TASK="${VICTIM_TASK:-qwenA}"
AGGRESSOR_TASK="${AGGRESSOR_TASK:-qwenB}"

# On-disk LoRA directory for each client; defaults to the task name. Task IDs
# stay distinct (they key the CSVs, STFQ weights and --task-rates) while the
# adapter can be shared. Point both at one adapter to isolate vLLM's
# multi-LoRA batching cost from everything else:
#     VICTIM_ADAPTER=qwenA AGGRESSOR_ADAPTER=qwenA bash .../run.sh
VICTIM_ADAPTER="${VICTIM_ADAPTER:-$VICTIM_TASK}"
AGGRESSOR_ADAPTER="${AGGRESSOR_ADAPTER:-$AGGRESSOR_TASK}"

# Three-phase priority pattern, same shape as fair_share/tsfm/run.sh.
# LLM RPS budget is much lower than TSFM — Qwen2.5-3B on T4 is generation-
# bound. Defaults assume ~2–12 RPS aggregate; tune via env vars.
# Victim-burst shape, mirroring fair_share/tsfm/results_t4 (A: 5,60,5 @ B=60).
# Phase 2 is the binding window: A's demand (24) exceeds its weighted share, so
# the 3:1 weight actually determines the split. plot.py scores fairness over
# exactly this phase (_phase2_window). Phases 1/3 are the reference state where
# A under-demands and B soaks up the remaining capacity.
# Scale (L40S, Qwen2.5-3B, 32 output tokens): baseline capacity ~30 rps
# (extrapolated from RTVSntask L40S logs: 919 tok/s at Running=147, still
# climbing; ~1300 tok/s saturated / ~40 tok per request). So phases 1/3 sit
# at 26 rps (0.87x capacity) and phase 2 at 46 rps (1.5x). A's 3:1
# entitlement is ~22 rps, and it bursts to 24 so the weight binds.
# T4 equivalents were B=11, A=2,12,2 at ~15 rps capacity.
AGGRESSOR_RPS_PHASES="${AGGRESSOR_RPS_PHASES:-33,33,33}"   # B constant, as in tsfm
VICTIM_RPS_PHASES="${VICTIM_RPS_PHASES:-4,34,4}"           # A bursts in phase 2
# 180s phases, not 60s. The phase-2/3 metrics count completions, so the first
# ~one-latency-period of each phase is still draining the previous phase's
# pipeline. At p50 ~26s that is 43% of a 60s phase (which made the baseline
# report A:B=0.48 when its true steady-state split is 1.01) and 14% of a 180s
# one. Phase 3 also needs >=150s: phase 2 leaves ~5200 requests queued, which
# takes ~135s to drain before any recovery is visible.
PHASE_DURATIONS="${PHASE_DURATIONS:-180,180,180}"
VICTIM_RPS="${VICTIM_RPS:-1}"

# vLLM engine knobs
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"      # match RTVSntask/llm
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"     # match RTVSntask/llm (was unset -> FMTK default 64)
# vLLM caps the running batch at max_num_seqs (default 256). For short
# generations that binds long before KV memory does, and because it is
# per-engine it hands the 2-server no-sharing arms a 2x concurrency budget
# over the single shared engine. Set it high on EVERY arm so the real
# hardware limits (KV capacity, compute) decide instead.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1000}"   # shared arms. 2048 OOMs at load: vLLM sizes
                                       # profiling/graph-capture buffers by
                                       # max_num_seqs x vocab (152k), so the real
                                       # ceiling is far below steady-state KV maths.

# Sharing runs: "scheduler  batch_size  batch_wait_ms  run_name  [task_rates_override]"
#   scheduler="vllm"  → no admission scheduler (vLLM owns ordering + batching).
#   scheduler="stfq"  → device/vllm_admission.VLLMAdmissionScheduler in front
#                       of AsyncLLMEngine, reusing STFQPolicy from the tsfm path.
#                       batch_size = admission window (# concurrent in-flight in vLLM).
# Weight convention (same as tsfm): rps in --task-rates → weight = 1/rps, so
# A:1,B:1 = 1:1, A:1,B:2 = 2:1 (victim 2x), A:1,B:3 = 3:1.
RUNS=(
    "vllm   1  0  vllm_baseline      ${VICTIM_TASK}:1,${AGGRESSOR_TASK}:1"   # already in results_t4_burst
    # "stfq   1  0  stfq_1_3          ${VICTIM_TASK}:1,${AGGRESSOR_TASK}:3"
    # "stfq  32  0  bfq_1_1            ${VICTIM_TASK}:1,${AGGRESSOR_TASK}:1"
    # "stfq  32  0  bfq_2_1            ${VICTIM_TASK}:1,${AGGRESSOR_TASK}:2"
    # "stfq  32  0  bfq_3_1            ${VICTIM_TASK}:1,${AGGRESSOR_TASK}:3"   # already in results_t4_burst
    # Admission-window sweep at fixed 3:1 weights. batch_size == max_in_flight
    # (server.py:92), i.e. the cap on concurrent sequences inside vLLM. W=32 is
    # bfq_3_1 above; these fill in the throughput/victim-latency frontier.
    # "stfq  64  0  bfq_3_1_w64        ${VICTIM_TASK}:1,${AGGRESSOR_TASK}:3"
    # "stfq 128  0  bfq_3_1_w128       ${VICTIM_TASK}:1,${AGGRESSOR_TASK}:3"
    "stfq 256  0  bfq_3_1_w256       ${VICTIM_TASK}:1,${AGGRESSOR_TASK}:3"
    # S-STFQ — same 3:1 weights as FMVisor, no batching. The TSFM analog is
    # batch_size=1 (tsfm/run.sh:91-93 "stfq 1 0 stfq_1_3"); on the vLLM path
    # batch_size == max_in_flight, so W=1 means one sequence in the engine at
    # a time. That is the faithful port, and it collapses: ~0.7 rps against
    # 14.5 at W=128, so most requests die at the client's 320s timeout and are
    # silently dropped by run.py's bare `except`. Run bounded by the timeout,
    # so it finishes in ~8 min with ~13% of requests completing.
    # Softer stand-in if the W=1 collapse is too degenerate to plot: keeps the
    # "STFQ without effective batching" message at a penalty ratio closer to
    # the ~2.3x TSFM shows, and completes normally.
    # "stfq   8  0  sstfq_3_1_w8       ${VICTIM_TASK}:1,${AGGRESSOR_TASK}:3"
)

# No-sharing TPC runs (SP) — victim and aggressor each get their own vLLM
# server, each pinned to a TPC partition. TPCs split proportionally to
# weight_a:weight_b, same convention as fair_share/tsfm/run.sh.
# Format: "run_name  weight_a  weight_b"
NO_SHARING_TPC_RUNS=(
#     # "no_sharing_tpc_1_1  1  1"
    # "no_sharing_tpc_3_1  3  1"
)
TPC_MODE="${TPC_MODE:-libsmctrl}"

# # No-sharing runs (BE) — victim and aggressor each get their own vLLM server
# # on the same GPU, no TPC partitioning (process-level isolation only).
# # Format: "run_name"
NO_SHARING_RUNS=(
    "no_sharing"
)

# Per-server GPU memory when two full backbones share one GPU. Qwen2.5-3B is
# ~5.8 GB of weights + ~0.3 GB of CUDA graphs, so on a 16 GB T4 this leaves
# little room for KV — lower MAX_MODEL_LEN if a no-sharing server fails to
# allocate its KV cache.
NO_SHARING_GPU_MEM="${NO_SHARING_GPU_MEM:-$(awk -v g="$GPU_MEM_UTIL" 'BEGIN{v=g/2; if(v<0.10) v=0.10; printf "%.2f", v}')}"

# Per-isolated-server max_num_seqs, derived as MAX_NUM_SEQS/2 so that the two
# no-sharing engines together get the SAME aggregate batch budget as the single
# shared engine. Without this the 2-server arms run 2 x MAX_NUM_SEQS concurrent
# sequences and win on throughput purely on budget: at matched batch (~1000) the
# shared engine is already faster per engine (1989 vs 1803 tok/s on L40S), so an
# unmatched comparison measures slot count rather than architecture.
#
# This is a *control*, not BE's best possible configuration — each isolated
# engine could take MAX_NUM_SEQS on its own. To run that variant instead:
#     NO_SHARING_MAX_NUM_SEQS=$MAX_NUM_SEQS bash .../run.sh
NO_SHARING_MAX_NUM_SEQS="${NO_SHARING_MAX_NUM_SEQS:-$(( MAX_NUM_SEQS / 2 > 1 ? MAX_NUM_SEQS / 2 : 1 ))}"

RESULTS_BASE="${RESULTS_BASE:-experiments/fair_share/llm/results_l40s_qwen3b}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-200}"  # vLLM model load is slow

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
    echo "Backbone: $BACKBONE (max_model_len=$MAX_MODEL_LEN, max_new_tokens=$MAX_NEW_TOKENS, max_num_seqs=$MAX_NUM_SEQS, gpu_mem_util=$GPU_MEM_UTIL)"
    echo "Client A (priority/victim): $VICTIM_TASK"
    echo "Client B (other/aggressor): $AGGRESSOR_TASK"
    echo "Adapters: $VICTIM_TASK->$VICTIM_ADAPTER, $AGGRESSOR_TASK->$AGGRESSOR_ADAPTER"
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
    if [[ ${#NO_SHARING_TPC_RUNS[@]} -gt 0 || ${#NO_SHARING_RUNS[@]} -gt 0 ]]; then
        echo "No-sharing runs (gpu_mem=$NO_SHARING_GPU_MEM, max_num_seqs=$NO_SHARING_MAX_NUM_SEQS each, tpc_mode=$TPC_MODE):"
        for run in "${NO_SHARING_TPC_RUNS[@]:-}"; do [[ -n "$run" ]] && echo "  SP: $run"; done
        for run in "${NO_SHARING_RUNS[@]:-}";     do [[ -n "$run" ]] && echo "  BE: $run"; done
    fi
    echo "Results base: $RESULTS_BASE"
} > "$CONFIG_FILE"

echo "================================================================"
echo "  fair_share/llm — vLLM multi-LoRA saturation experiment"
echo "  Backbone     : $BACKBONE  (max_model_len=$MAX_MODEL_LEN, max_new_tokens=$MAX_NEW_TOKENS, max_num_seqs=$MAX_NUM_SEQS, gpu_mem_util=$GPU_MEM_UTIL)"
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
        --max-new-tokens         "$MAX_NEW_TOKENS" \
        --max-num-seqs           "$MAX_NUM_SEQS"   \
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
# start_vllm_isolated_device PORT GPU_MEM LOG [TPC_PARTITION]
#   One vLLM server holding a full backbone copy for a single task. Adapted
#   from RTVSntask/llm/run.sh:224 — no admission scheduler (scheduler-policy
#   fifo, no --task-rates), so vLLM owns ordering exactly as in the shared
#   baseline. TPC_PARTITION, when given, pins the server to those TPCs.
start_vllm_isolated_device() {
    local port="$1" gpu_mem="$2" log="$3"
    local tpc_part="${4:-}"

    pkill -f "device/main.py.*--port ${port}" 2>/dev/null || true
    sleep 1

    local tpc_args=""
    [[ -n "$tpc_part" ]] && tpc_args="--tpc-mode $TPC_MODE --tpc-partition $tpc_part"

    echo "[run.sh] Starting isolated vLLM server port=$port gpu_mem=$gpu_mem ${tpc_part:+tpc=[$tpc_part]}"

    # shellcheck disable=SC2086
    $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port                   "$port"           \
        --runtime-type           vllm              \
        --cuda                   "$CUDA_DEVICE"    \
        --scheduler-policy       fifo              \
        --gpu-memory-utilization "$gpu_mem"        \
        --max-model-len          "$MAX_MODEL_LEN"  \
        --max-new-tokens         "$MAX_NEW_TOKENS" \
        --max-num-seqs           "$NO_SHARING_MAX_NUM_SEQS" \
        $tpc_args \
        > "$log" 2>&1 &
    local pid=$!
    DEVICE_PIDS+=("$pid")
    ACTIVE_PORTS+=("$port")
    echo "[run.sh] PID=$pid  log=$log"

    local waited=0
    while (( waited < DEVICE_STARTUP_WAIT )); do
        if grep -q -E "Runtime application started|application started|gRPC server.*listen" "$log" 2>/dev/null; then
            echo "[run.sh] Server ready (after ${waited}s)"
            sleep 2
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[run.sh] ERROR: server exited early — see $log" >&2
            tail -n 40 "$log" >&2 || true
            return 1
        fi
        sleep 2; waited=$(( waited + 2 ))
    done
    echo "[run.sh] WARNING: ready signal not seen in ${DEVICE_STARTUP_WAIT}s, proceeding"
}

# run_two_server_experiment RUN_NAME LABEL [VICTIM_TPCS] [AGGRESSOR_TPCS]
#   Shared body of the BE and SP loops: bring up one server per task, drive the
#   same phased workload through run.py with separate URLs (run.py:388 deploys
#   each task's adapter to its own server), then tear both down.
run_two_server_experiment() {
    local run_name="$1" label="$2" victim_tpcs="${3:-}" aggressor_tpcs="${4:-}"
    local exp_dir="${RESULTS_BASE}/${run_name}"
    local victim_port="$DEVICE_PORT"
    local aggressor_port=$(( DEVICE_PORT + 1 ))

    stop_devices
    DEVICE_PIDS=()
    ACTIVE_PORTS=()

    echo "[run.sh] Victim:    port=$victim_port    task=$VICTIM_TASK    ${victim_tpcs:+tpcs=[$victim_tpcs]}"
    echo "[run.sh] Aggressor: port=$aggressor_port task=$AGGRESSOR_TASK ${aggressor_tpcs:+tpcs=[$aggressor_tpcs]}"

    if ! start_vllm_isolated_device "$victim_port" "$NO_SHARING_GPU_MEM" \
            "$LOG_DIR/device_${run_name}_victim.log" "$victim_tpcs"; then
        echo "[run.sh] WARNING: victim server failed for $run_name — skipping"
        stop_devices
        return 1
    fi
    if ! start_vllm_isolated_device "$aggressor_port" "$NO_SHARING_GPU_MEM" \
            "$LOG_DIR/device_${run_name}_aggressor.log" "$aggressor_tpcs"; then
        echo "[run.sh] WARNING: aggressor server failed for $run_name — skipping"
        stop_devices
        return 1
    fi

    $PYTHON -u experiments/fair_share/llm/run.py \
        --victim-url            "localhost:${victim_port}"      \
        --aggressor-url         "localhost:${aggressor_port}"   \
        --backbone              "$BACKBONE"                     \
        --victim-task           "$VICTIM_TASK"                  \
        --aggressor-task        "$AGGRESSOR_TASK"               \
        --victim-rps            "$VICTIM_RPS"                   \
        --victim-rps-phases     "$VICTIM_RPS_PHASES"            \
        --aggressor-rps-phases  "$AGGRESSOR_RPS_PHASES"         \
        --phase-durations       "$PHASE_DURATIONS_CSV"          \
        --victim-adapter-path   "$VICTIM_ADAPTER"               \
        --aggressor-adapter-path "$AGGRESSOR_ADAPTER"           \
        --scheduler-policy      "$label"                        \
        --exp-dir               "$exp_dir"                      \
    || echo "[run.sh] WARNING: run.py failed for $run_name — continuing"

    stop_devices
}

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
        --victim-adapter-path   "$VICTIM_ADAPTER"            \
        --aggressor-adapter-path "$AGGRESSOR_ADAPTER"        \
        --scheduler-policy      "$SCHEDULER"                 \
        --exp-dir               "$EXP_DIR"                   \
    || echo "[run.sh] WARNING: run.py failed for $RUN_NAME — continuing"

    stop_devices
done

# ---------------------------------------------------------------------------
# No-sharing TPC runs (SP) — one TPC-partitioned vLLM server per task
# ---------------------------------------------------------------------------
for run in "${NO_SHARING_TPC_RUNS[@]:-}"; do
    [[ -z "$run" ]] && continue
    read -r RUN_NAME WEIGHT_A WEIGHT_B <<< "$run"

    echo ""
    echo "================================================================"
    echo "  $RUN_NAME  (SP: 2 TPC-partitioned vLLM servers, tpc_mode=$TPC_MODE,"
    echo "              weights=${WEIGHT_A}:${WEIGHT_B}, gpu_mem=$NO_SHARING_GPU_MEM each)"
    echo "  Results: ${RESULTS_BASE}/${RUN_NAME}"
    echo "================================================================"

    TOTAL_TPCS=$($PYTHON -c "
import torch
sm = torch.cuda.get_device_properties('${CUDA_DEVICE}').multi_processor_count
print(sm // 2)
")
    echo "[run.sh] Total TPCs on GPU: $TOTAL_TPCS"

    WEIGHT_TOTAL=$(( WEIGHT_A + WEIGHT_B ))
    VICTIM_TPC_COUNT=$(( TOTAL_TPCS * WEIGHT_A / WEIGHT_TOTAL ))
    if [[ $VICTIM_TPC_COUNT -le 0 ]]; then VICTIM_TPC_COUNT=1; fi
    if [[ $VICTIM_TPC_COUNT -ge $TOTAL_TPCS ]]; then VICTIM_TPC_COUNT=$(( TOTAL_TPCS - 1 )); fi

    VICTIM_TPCS=$(seq -s ' ' 0 $(( VICTIM_TPC_COUNT - 1 )))
    AGGRESSOR_TPCS=$(seq -s ' ' "$VICTIM_TPC_COUNT" $(( TOTAL_TPCS - 1 )))

    run_two_server_experiment "$RUN_NAME" "no_sharing_tpc" "$VICTIM_TPCS" "$AGGRESSOR_TPCS"
done

# ---------------------------------------------------------------------------
# No-sharing runs (BE) — one full-backbone vLLM server per task, no TPC split
# ---------------------------------------------------------------------------
for run in "${NO_SHARING_RUNS[@]:-}"; do
    [[ -z "$run" ]] && continue
    read -r RUN_NAME <<< "$run"

    echo ""
    echo "================================================================"
    echo "  $RUN_NAME  (BE: 2 process-isolated vLLM servers, no TPC partition,"
    echo "              gpu_mem=$NO_SHARING_GPU_MEM each)"
    echo "  Results: ${RESULTS_BASE}/${RUN_NAME}"
    echo "================================================================"

    run_two_server_experiment "$RUN_NAME" "no_sharing"
done

echo ""
echo "[run.sh] All runs done. Results in $RESULTS_BASE"
