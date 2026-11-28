#!/usr/bin/env bash
# RTVSntask/llm — response time vs. number of tasks, on vLLM multi-LoRA.
#
# LLM analog of RTVSntask/vision/run.sh. Tasks are LoRA adapters over a shared
# Qwen2.5 backbone. NUM_TASKS_SWEEP is unbounded: when it exceeds the adapter
# pool, adapters are cycled with a __appN suffix so each slot is a distinct
# task ID backed by its own LoRA slot in the engine.
#
# Usage:  bash experiments/RTVSntask/llm/run.sh [BACKBONE]     (default qwen2.5-3b)
#
# Conditions per (ntasks, rps). Every arm: the same LLMEngine stepped by
# device/llm_loop.py (--vllm-driver step), the same engine flags, and request
# tokens staged on the GPU (--staging gpu: shared-mode servers copy ids to the
# GPU and back exactly as fmvisor's slots do, so staging is not a difference
# between arms). The flags are fixed per arm, the same as RTVSntask/vision:
#
#   arm             servers  --isolation-mode  policy  staging    worker mode
#   no_sharing_tpc  N        shared            fifo    gpu        inline   (SP)
#   no_sharing      N        shared            fifo    gpu        inline   (BE)
#   sharing         1        shared            fifo    gpu        threaded (Shared)
#   sharing_inline  1        shared            fifo    gpu        inline   (reference, not plotted)
#   fmvisor         1        process           stfq    GPU slots  task processes
#
# Worker mode: where a finished request's per-task output work (ids back from
# the GPU, detokenize) runs. threaded = each task's own thread; inline = on the
# step-loop thread, one task after another. BE/SP have one task per server, so
# the two are the same there. Shared is threaded, as for vision, so per-task
# work is not serialised behind the step loop; sharing_inline is kept only as a
# reference (CONDITIONS=...,sharing_inline).
#
#   no_sharing_tpc  — SP: N vLLM servers, one per task, spatially partitioned.
#                     SP_MODE=mig (default where the GPU supports MIG):
#                       N=1   whole GPU, MIG off, no pinning
#                       N=2   2 MIG slices (MIG_PROFILE_HALF, 2g.48gb), one task each
#                       N>=3  4 MIG slices (MIG_PROFILE_QUARTER, 1g.24gb); task i on
#                             slice i mod 4, and the k tasks on a slice each pinned
#                             to 1/k of its TPCs (libsmctrl, --tpc-share j/k)
#                     Each server gets GPU_MEM_UTIL/k of its slice's memory. MIG is
#                     turned on for this arm only and off again after it (sudo).
#                     SP_MODE=tpc: no MIG, the whole GPU's TPCs split N ways.
#   sharing         — 1 vLLM server, all N LoRAs (Punica / S-LoRA style), FIFO, threaded
#   fmvisor         — 1 server in process-isolation mode: the vLLM engine in a
#                     backbone process (STFQ), one process
#                     per task (tokenize / detokenize), FM_SLOTS slots per task
#   no_sharing      — N vLLM servers, each a full backbone copy + 1 LoRA;
#                     gpu_memory_utilization split N ways
#
# These are the same three arms, in the same order, that RTVSntask/vision runs.
# A single_<task> baseline (1 server, that LoRA only, driven at the AGGREGATE
# rate rps*ntasks) is implemented but off by default, because vision never ran
# it. Opt in with CONDITIONS=single,sharing,no_sharing.
#
# no_sharing is EXPECTED to fail beyond a handful of tasks: N full 1.5B backbone
# copies do not fit. That failure is a result, not a bug — when a condition
# cannot be placed, a failure.json recording the classified reason is written to
# its results dir and the sweep continues.
#
# Prerequisite: adapters under <FMTK>/models/llm/finetuned/.
#     python experiments/fair_share/llm/make_loras.py \
#         --backbone qwen2.5-1.5b --names qwenA,qwenB
#
# Quick start (smoke test):
#     NUM_TASKS_SWEEP=3 PHASE_DURATION=90 RESULTS_BASE=experiments/RTVSntask/llm/smoke \
#         bash experiments/RTVSntask/llm/run.sh qwen2.5-3b
#
# Full sweep (defaults: ntasks 1,3,5,7,9, 2 rps/task, 600 s):
#     bash experiments/RTVSntask/llm/run.sh qwen2.5-3b
#
# Optional knobs (env vars):
#   CONDA_ENV             fmtk
#   BACKBONE              qwen2.5-3b     (or the first argument)
#   TASKS                 qwenA,qwenB    (adapter pool; cycled with __appN past its size)
#   NUM_TASKS_SWEEP       1,3,5,7,9      (comma-separated; no upper bound)
#   RPS_SWEEP             2              (comma-separated, PER-TASK rps)
#   PHASE_DURATION        600
#   GPU_MEM_UTIL          0.85   (single/sharing use it whole; no_sharing* gets /N each)
#   MAX_MODEL_LEN         512
#   MAX_NEW_TOKENS        32
#   DEVICE_PORT           8000   (base port; servers use 8000, 8001, ...)
#   DEVICE_STARTUP_WAIT   120    (vLLM cold load is slow, and slower with N servers)
#   DEPLOY_TIMEOUT        1800   (gRPC deadline for the load control call)
#   INCLUDE_TPC           1      (0 to skip no_sharing_tpc — it needs libsmctrl)
#   TPC_MODE              libsmctrl
#   SP_MODE               auto   mig if the GPU supports MIG, else tpc (see SP above)
#   MIG_PROFILE_HALF      2g.48gb   (RTX PRO 6000 Blackwell; `sudo nvidia-smi mig -lgip`)
#   MIG_PROFILE_QUARTER   1g.24gb
#   MIG_WAIT              120    s to wait for the GPU to be free before MIG changes
#   MAX_NUM_SEQS          (unset = vLLM's 256) engine batch cap, every arm
#   LLM_QUEUE_K           1      requests waiting inside vLLM beyond MAX_NUM_SEQS
#   FM_SLOTS              64     fmvisor: slots per task (bounds a task's in-flight requests)
#   FM_BACKLOG_CAP        10000  fmvisor: requests per task waiting on the CPU
#   FM_MAX_PROMPT_LEN     256    fmvisor: prompt tokens per slot
#   CONDITIONS            no_sharing_tpc,sharing,fmvisor,no_sharing (default order;
#                          also sharing_inline, single — "single" expands to the first task slot)
#   RESULTS_BASE          experiments/RTVSntask/llm/results_<backbone>_<gpu>
#                          (gpu tag from nvidia-smi: l40s, a10, t4, ...)

set -euo pipefail

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
    echo "ERROR: FMTK_DIR not found at: $FMTK_DIR"; exit 1
fi
if [[ ! -d "$FMAAS_DIR" ]]; then
    echo "ERROR: FMAAS_DIR not found at: $FMAAS_DIR"; exit 1
fi

export PYTHONPATH="${FMTK_DIR}/src:${FMAAS_DIR}:${PYTHONPATH:-}"

if command -v conda &> /dev/null; then
    PYTHON="${PYTHON:-conda run --no-capture-output -n ${CONDA_ENV} python}"
else
    PYTHON="${PYTHON:-python}"
fi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${1:-${BACKBONE:-qwen2.5-3b}}"
TASKS_CSV="${TASKS:-qwenA,qwenB}"
RPS_SWEEP="${RPS_SWEEP:-2}"
NUM_TASKS_SWEEP="${NUM_TASKS_SWEEP:-1,3,5,7,9}"
PHASE_DURATION="${PHASE_DURATION:-600}"
DEVICE_PORT="${DEVICE_PORT:-8000}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-180}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
INCLUDE_TPC="${INCLUDE_TPC:-1}"
TPC_MODE="${TPC_MODE:-libsmctrl}"
GPU_INDEX="${CUDA_DEVICE##*:}"
SP_MODE="${SP_MODE:-auto}"
if [[ "$SP_MODE" == auto ]]; then
    # mig.mode.current is "Enabled"/"Disabled" on a MIG-capable GPU, "[N/A]" otherwise.
    if nvidia-smi -i "$GPU_INDEX" --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null \
            | grep -qiE 'enabled|disabled'; then
        SP_MODE=mig
    else
        SP_MODE=tpc
    fi
fi
MIG_PROFILE_HALF="${MIG_PROFILE_HALF:-2g.48gb}"
MIG_PROFILE_QUARTER="${MIG_PROFILE_QUARTER:-1g.24gb}"
MIG_WAIT="${MIG_WAIT:-120}"
WARMUP_SECS="${WARMUP_SECS:-10}"
# gRPC deadline for the "load" control call. The client default is 120s,
# which a vLLM engine build over a networked filesystem blows straight past.
DEPLOY_TIMEOUT="${DEPLOY_TIMEOUT:-1800}"
if [[ -z "${RESULTS_BASE:-}" ]]; then
    # GPU tag from the device name: "NVIDIA L40S" -> l40s, "NVIDIA A10G" -> a10, "Tesla T4" -> t4
    _gpu_idx="${CUDA_DEVICE##*:}"
    _gpu_name=$(nvidia-smi -i "$_gpu_idx" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    case "$_gpu_name" in
        *T4*)   GPU_TAG=t4 ;;
        *A10*)  GPU_TAG=a10 ;;
        *L40S*) GPU_TAG=l40s ;;
        *L4*)   GPU_TAG=l4 ;;
        *A2*)   GPU_TAG=a2 ;;
        *"RTX PRO 6000"*) GPU_TAG=rtxpro6000 ;;
        *)      GPU_TAG=$(echo "${_gpu_name:-gpu}" | awk '{print tolower($NF)}') ;;
    esac
    RESULTS_BASE="experiments/RTVSntask/llm/results_${BACKBONE}_${GPU_TAG}"
fi
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
LLM_QUEUE_K="${LLM_QUEUE_K:-1}"
FM_SLOTS="${FM_SLOTS:-64}"
FM_BACKLOG_CAP="${FM_BACKLOG_CAP:-10000}"
FM_MAX_PROMPT_LEN="${FM_MAX_PROMPT_LEN:-256}"

IFS=',' read -ra ALL_TASKS <<< "$TASKS_CSV"
POOL_SIZE=${#ALL_TASKS[@]}

LOG_DIR="${RESULTS_BASE}/logs"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  RT vs NumTasks (LLM / multi-LoRA)"
echo "  Conda env      : $CONDA_ENV"
echo "  Backbone       : $BACKBONE  (max_model_len=$MAX_MODEL_LEN)"
echo "  Adapter pool   : ${ALL_TASKS[*]}  (cycled with __appN past ${POOL_SIZE})"
echo "  NUM_TASKS sweep: $NUM_TASKS_SWEEP"
echo "  RPS sweep      : $RPS_SWEEP  (per task; singles run at rps*ntasks)"
echo "  Duration/run   : ${PHASE_DURATION}s"
echo "  GPU mem (full) : $GPU_MEM_UTIL  (no_sharing* divides by N)"
echo "  max_new_tokens : $MAX_NEW_TOKENS"
echo "  Deploy timeout : ${DEPLOY_TIMEOUT}s"
if [[ "$INCLUDE_TPC" != 1 ]]; then
    echo "  SP condition   : skipped"
elif [[ "$SP_MODE" == mig ]]; then
    echo "  SP condition   : MIG (N=1 whole GPU, N=2 2x$MIG_PROFILE_HALF, N>=3 4x$MIG_PROFILE_QUARTER) + $TPC_MODE inside a shared slice"
else
    echo "  SP condition   : $TPC_MODE over the whole GPU"
fi
echo "  vLLM driver    : step  (max_num_seqs=${MAX_NUM_SEQS:-256 default}, queue_k=$LLM_QUEUE_K)"
echo "  Arms           : BE/SP fifo inline | Shared fifo threaded | FMVisor stfq process (all gpu staging)"
echo "  fmvisor        : slots/task=$FM_SLOTS backlog=$FM_BACKLOG_CAP max_prompt_len=$FM_MAX_PROMPT_LEN"
echo "  Base port      : $DEVICE_PORT"
echo "  Results        : $RESULTS_BASE"
echo "================================================================"

# ---------------------------------------------------------------------------
# MIG (SP_MODE=mig). As experiments/motivation2/run.sh: nvidia-smi with sudo
# (or directly as root); slices only while no process has the GPU open.
# ---------------------------------------------------------------------------
SUDO=""; [[ "$(id -u)" == 0 ]] || SUDO=sudo
MIG_TURNED_ON=0

mig_enabled() {
    nvidia-smi -i "$GPU_INDEX" --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null \
        | grep -qi enabled
}

# wait_gpu_idle: until no process (of this user) holds the GPU, or MIG_WAIT s.
wait_gpu_idle() {
    local deadline=$((SECONDS + MIG_WAIT)) apps pids
    while :; do
        apps=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name \
               --format=csv,noheader 2>/dev/null || true)
        # With MIG on, --query-compute-apps misses processes that still hold
        # the device open, so also look for this user's on /dev/nvidiaN.
        if [[ -z "$apps" ]]; then
            pids=$(fuser "/dev/nvidia$GPU_INDEX" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' \
                   | while read -r p; do [[ "$(stat -c %U /proc/$p 2>/dev/null)" == "$USER" ]] && echo "$p"; done || true)
            [[ -n "$pids" ]] && apps=$(ps -o pid=,cmd= -p "$(paste -sd, <<< "$pids")" 2>/dev/null || true)
        fi
        [[ -z "$apps" ]] && return 0
        if (( SECONDS > deadline )); then
            echo "[run.sh] GPU still in use after ${MIG_WAIT}s by:" >&2; echo "$apps" >&2
            return 1
        fi
        sleep 2
    done
}

# mig_on PROFILES_CSV — MIG mode on, and exactly these slices.
mig_on() {
    local profiles="$1" want n try
    want=$(tr ',' '\n' <<< "$profiles" | grep -c .)
    echo "[run.sh] MIG on: GPU $GPU_INDEX, slices $profiles" >&2
    wait_gpu_idle || true
    if ! mig_enabled; then
        $SUDO nvidia-smi -i "$GPU_INDEX" -mig 1 >&2
        if ! mig_enabled; then
            echo "[run.sh] MIG mode is pending (a GPU reset may be needed)" >&2
            return 1
        fi
    fi
    MIG_TURNED_ON=1
    # Drop any slices left from an earlier layout.
    $SUDO nvidia-smi mig -i "$GPU_INDEX" -dci >/dev/null 2>&1 || true
    $SUDO nvidia-smi mig -i "$GPU_INDEX" -dgi >/dev/null 2>&1 || true
    for try in 1 2 3 4 5; do
        $SUDO nvidia-smi mig -i "$GPU_INDEX" -cgi "$profiles" -C >&2 && break
        echo "[run.sh] creating MIG slices failed (try $try/5); retrying in 10 s" >&2
        sleep 10
    done
    n=$(nvidia-smi -L | grep -c 'MIG-' || true)
    nvidia-smi -L | grep 'MIG-' >&2 || true
    (( n == want )) || { echo "[run.sh] expected $want MIG slices, found $n" >&2; return 1; }
}

mig_off() {
    mig_enabled || { MIG_TURNED_ON=0; return 0; }
    echo "[run.sh] MIG off: GPU $GPU_INDEX" >&2
    wait_gpu_idle || true
    $SUDO nvidia-smi mig -i "$GPU_INDEX" -dci >/dev/null 2>&1 || true
    $SUDO nvidia-smi mig -i "$GPU_INDEX" -dgi >/dev/null 2>&1 || true
    $SUDO nvidia-smi -i "$GPU_INDEX" -mig 0 >&2
    MIG_TURNED_ON=0
    if mig_enabled; then
        echo "[run.sh] ERROR: MIG still enabled — the other arms need the whole GPU" >&2
        return 1
    fi
}

mig_uuids() {
    nvidia-smi -L 2>/dev/null | grep -o 'MIG-[0-9a-fA-F-]*' | tr '\n' ' '
}

# sp_layout N — MIG slices for SP with N tasks (empty: whole GPU, no MIG).
sp_layout() {
    local n="$1" i out=""
    if (( n <= 1 )); then
        echo ""
    elif (( n == 2 )); then
        echo "${MIG_PROFILE_HALF},${MIG_PROFILE_HALF}"
    else
        for ((i = 0; i < 4; i++)); do out+="${out:+,}$MIG_PROFILE_QUARTER"; done
        echo "$out"
    fi
}

# ---------------------------------------------------------------------------
# Device server lifecycle
# ---------------------------------------------------------------------------
DEVICE_PIDS=()
ACTIVE_PORTS=()
LAST_FAIL_LOG=""      # device log of the most recent startup failure

stop_devices() {
    local pids_to_wait=()
    for pid in "${DEVICE_PIDS[@]:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "[run.sh] Stopping device PID=$pid (SIGTERM)" >&2
            kill "$pid" 2>/dev/null || true
            pids_to_wait+=("$pid")
        fi
    done
    local deadline=$((SECONDS + 20))
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
    sleep 3   # vLLM CUDA cleanup is slow
}
trap 'stop_devices; [[ "$MIG_TURNED_ON" == 1 ]] && mig_off || true' EXIT

# classify_failure LOG -> prints a reason slug
#   Reads a device log and names why the server could not be placed. This is
#   the payload of failure.json: at high ntasks, no_sharing loses on memory and
#   we want the plot to say *which* memory.
classify_failure() {
    local log="$1"
    [[ -f "$log" ]] || { echo "no_log"; return; }
    if   grep -qi "AsyncEngineDeadError\|Background loop has errored" "$log" && \
         grep -qi "must match the size of tensor" "$log"; then
        echo "adapter_shape_mismatch"
    elif grep -qi "No available memory for the cache blocks\|Cannot allocate KV cache\|decrease gpu_memory_utilization" "$log"; then
        echo "kv_cache_insufficient"
    elif grep -qi "CUDA out of memory\|OutOfMemoryError\|CUDA error: out of memory" "$log"; then
        echo "cuda_oom"
    elif grep -qi "free memory.*is less than\|not enough memory\|memory profiling" "$log"; then
        echo "insufficient_gpu_memory"
    elif grep -qi "Address already in use\|address in use" "$log"; then
        echo "port_conflict"
    elif grep -qi "No such file or directory.*finetuned\|adapter.*not found" "$log"; then
        echo "adapter_missing"
    else
        echo "unknown"
    fi
}

# record_failure OUT_DIR CONDITION NTASKS RPS STAGE LOG
record_failure() {
    local out_dir="$1" condition="$2" ntasks="$3" rps="$4" stage="$5" log="$6"
    local reason; reason=$(classify_failure "$log")
    local tail_txt=""
    [[ -f "$log" ]] && tail_txt=$(tail -n 25 "$log" | sed 's/\\/\\\\/g; s/"/\\"/g' | awk '{printf "%s\\n", $0}')
    mkdir -p "$out_dir"
    cat > "${out_dir}/failure.json" <<EOF
{
  "condition": "${condition}",
  "num_tasks": ${ntasks},
  "rps_per_task": ${rps},
  "stage": "${stage}",
  "reason": "${reason}",
  "gpu_memory_utilization_full": ${GPU_MEM_UTIL},
  "gpu_memory_utilization_per_server": ${LAST_PER_MEM:-null},
  "device_log": "${log}",
  "log_tail": "${tail_txt}"
}
EOF
    echo "[run.sh] FAILURE recorded: $condition ntasks=$ntasks reason=$reason -> ${out_dir}/failure.json" >&2
}

# start_vllm_device PORT GPU_MEM LOG SCHED WORKER_MODE ISOLATION [TPC] [EXTRA_ARGS...]
#   Every arm gets the same engine settings, driver (step) and staging (gpu);
#   the arm's own flags (policy, worker mode, isolation) are passed explicitly.
#   TPC: "" (no pinning), a TPC list "0 1 2" (--tpc-partition), or "share:j/k"
#   (--tpc-share: the j-th of k TPC ranges of the device the server sees).
#   SV_VISIBLE (caller's local, optional): a MIG slice UUID the server runs on.
start_vllm_device() {
    local port="$1" gpu_mem="$2" log="$3"
    local sched="$4" worker_mode="$5" isolation="$6"
    local tpc_part="${7:-}"
    local extra=("${@:8}")

    pkill -f "device/main.py.*--port ${port}" 2>/dev/null || true
    sleep 1

    local tpc_args=""
    if [[ "$tpc_part" == share:* ]]; then
        tpc_args="--tpc-mode $TPC_MODE --tpc-share ${tpc_part#share:}"
    elif [[ -n "$tpc_part" ]]; then
        tpc_args="--tpc-mode $TPC_MODE --tpc-partition $tpc_part"
    fi
    local visible="${SV_VISIBLE:-}" cuda="$CUDA_DEVICE" envs=()
    if [[ -n "$visible" ]]; then envs=(env "CUDA_VISIBLE_DEVICES=$visible"); cuda="cuda:0"; fi

    echo "[run.sh] Starting vLLM server port=$port gpu_mem=$gpu_mem sched=$sched worker=$worker_mode isolation=$isolation ${tpc_part:+tpc=[$tpc_part] }${visible:+on $visible}" >&2

    # shellcheck disable=SC2086
    ${envs[@]+"${envs[@]}"} $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port                   "$port"           \
        --runtime-type           vllm              \
        --cuda                   "$cuda"           \
        --scheduler-policy       "$sched"          \
        --gpu-memory-utilization "$gpu_mem"        \
        --max-model-len          "$MAX_MODEL_LEN"  \
        --max-new-tokens         "$MAX_NEW_TOKENS" \
        --vllm-driver            step              \
        --llm-queue-k            "$LLM_QUEUE_K"    \
        --staging                gpu               \
        --worker-mode            "$worker_mode"    \
        --isolation-mode         "$isolation"      \
        ${MAX_NUM_SEQS:+--max-num-seqs "$MAX_NUM_SEQS"} \
        $tpc_args \
        ${extra[@]+"${extra[@]}"} \
        > "$log" 2>&1 &
    local pid=$!
    DEVICE_PIDS+=("$pid")
    ACTIVE_PORTS+=("$port")
    echo "[run.sh] PID=$pid  log=$log" >&2

    local waited=0
    while (( waited < DEVICE_STARTUP_WAIT )); do
        if grep -q -E "Runtime application started|application started|IsolatedApp\] Started|gRPC server.*listen" "$log" 2>/dev/null; then
            echo "[run.sh] Server ready (after ${waited}s)" >&2
            sleep 2
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[run.sh] ERROR: server exited early — see $log" >&2
            tail -n 40 "$log" >&2 || true
            LAST_FAIL_LOG="$log"
            return 1
        fi
        sleep 2; waited=$(( waited + 2 ))
    done
    echo "[run.sh] WARNING: ready signal not seen in ${DEVICE_STARTUP_WAIT}s, proceeding" >&2
}

# Per-server GPU memory util = GPU_MEM_UTIL/n, clamped >= 0.10.
# Past the point where 0.10*n exceeds GPU_MEM_UTIL the split is over-subscribed;
# we launch anyway and let the failure be the measurement.
ns_gpu_mem() {
    local n="$1"
    awk -v base="$GPU_MEM_UTIL" -v n="$n" 'BEGIN{
        v = base/n;
        if (v < 0.10) v = 0.10;
        printf "%.3f", v;
    }'
}

# agg_rps RPS NTASKS — float-safe rps*ntasks
agg_rps() {
    awk -v r="$1" -v n="$2" 'BEGIN{ v = r*n; if (v == int(v)) printf "%d", v; else printf "%g", v; }'
}

# ---------------------------------------------------------------------------
# Single condition runner
# ---------------------------------------------------------------------------
run_condition() {
    local condition="$1" rps="$2" ntasks="$3" tasks_csv="$4"

    local ntasks_dir="${abs_results}/ntasks_${ntasks}/rps_${rps}"
    local out_dir="${ntasks_dir}/${condition}"
    local trace_file run_rps
    if [[ "$condition" == single_* ]]; then
        # Singles absorb the whole aggregate arrival rate so the GPU sees the
        # same total load as the multi-task arms. Trace is keyed by that actual
        # rate, so (2 tasks @ 5rps) and (5 tasks @ 2rps) reuse one trace.
        run_rps=$(agg_rps "$rps" "$ntasks")
        trace_file="${abs_results}/singles/rps_${run_rps}/trace.json"
    else
        run_rps="$rps"
        trace_file="${ntasks_dir}/trace.json"
    fi

    echo ""
    echo "================================================================"
    echo "  condition=$condition  rps=$rps  ntasks=$ntasks  (run_rps=$run_rps)"
    echo "  tasks    = ${tasks_csv}"
    echo "  Results  : $out_dir"
    echo "================================================================"

    if [[ -f "${out_dir}/latencies.csv" ]]; then
        echo "[run.sh] Skipping $condition — results already exist"
        return 0
    fi

    stop_devices
    mkdir -p "$out_dir"
    rm -f "${out_dir}/failure.json"

    LAST_PER_MEM="null"
    LAST_FAIL_LOG=""
    LAST_SP_LAYOUT=""
    LAST_SP_ASSIGN=""

    local dev_log
    case "$condition" in
        single_*)
            mig_off || { record_failure "$out_dir" "$condition" "$ntasks" "$rps" "mig_off" ""; return 1; }
            local single_task="${condition#single_}"
            dev_log="$LOG_DIR/device_${condition}_rps${rps}_n${ntasks}.log"
            if ! start_vllm_device "$DEVICE_PORT" "$GPU_MEM_UTIL" "$dev_log" fifo inline shared; then
                record_failure "$out_dir" "$condition" "$ntasks" "$rps" "device_start" "$dev_log"
                return 1
            fi
            $PYTHON -u experiments/RTVSntask/llm/run.py \
                --tasks       "$single_task" \
                --condition   "$condition" \
                --device-urls "localhost:${DEVICE_PORT}" \
                --backbone    "$BACKBONE" \
                --rps         "$run_rps" \
                --duration    "$PHASE_DURATION" \
                --warmup-secs "$WARMUP_SECS" \
                --exp-dir     "$out_dir" \
                --trace-file  "$trace_file" \
                --deploy-timeout "$DEPLOY_TIMEOUT" \
                || { record_failure "$out_dir" "$condition" "$ntasks" "$rps" "workload" "$dev_log"; return 1; }
            ;;

        no_sharing|no_sharing_tpc)
            local urls="" per_mem
            local SV_VISIBLE=""
            local -a uuids=() slice_k=()
            local nslices=0
            per_mem=$(ns_gpu_mem "$ntasks")
            LAST_PER_MEM="$per_mem"

            if [[ "$condition" == no_sharing_tpc && "$SP_MODE" == mig ]]; then
                # SP on MIG: slices for this N, then TPC shares inside a slice.
                local layout; layout=$(sp_layout "$ntasks")
                LAST_SP_LAYOUT="${layout:-whole_gpu}"
                if [[ -z "$layout" ]]; then
                    mig_off || { record_failure "$out_dir" "$condition" "$ntasks" "$rps" "mig_off" ""; return 1; }
                    per_mem="$GPU_MEM_UTIL"; LAST_PER_MEM="$per_mem"
                else
                    mig_on "$layout" || { record_failure "$out_dir" "$condition" "$ntasks" "$rps" "mig_setup" ""; return 1; }
                    read -ra uuids <<< "$(mig_uuids)"
                    nslices=${#uuids[@]}
                    for ((s = 0; s < nslices; s++)); do
                        # tasks on slice s: i = s, s+nslices, s+2*nslices, ...
                        slice_k[s]=$(( (ntasks - s + nslices - 1) / nslices ))
                    done
                    LAST_PER_MEM="null"
                fi
            else
                [[ "$condition" == no_sharing_tpc ]] && LAST_SP_LAYOUT="whole_gpu_tpc"
                mig_off || { record_failure "$out_dir" "$condition" "$ntasks" "$rps" "mig_off" ""; return 1; }
                awk -v p="$per_mem" -v n="$ntasks" -v b="$GPU_MEM_UTIL" 'BEGIN{
                    if (p*n > b + 1e-9)
                        printf "[run.sh] NOTE: %d servers x %.3f = %.3f exceeds GPU_MEM_UTIL=%.2f — over-subscribed, launching anyway\n", n, p, p*n, b;
                }' >&2
            fi

            local tpcs_per=0
            if [[ "$condition" == "no_sharing_tpc" && "$SP_MODE" != mig ]]; then
                local total_tpcs
                total_tpcs=$($PYTHON -c "
import torch
sm = torch.cuda.get_device_properties('${CUDA_DEVICE}').multi_processor_count
print(sm // 2)
")
                tpcs_per=$((total_tpcs / ntasks))
                if (( tpcs_per < 1 )); then
                    echo "[run.sh] ERROR: ntasks=$ntasks exceeds available TPCs ($total_tpcs)" >&2
                    record_failure "$out_dir" "$condition" "$ntasks" "$rps" "tpc_partition" ""
                    return 1
                fi
            fi

            local cur=0 assign=""
            for ((i=0; i<ntasks; i++)); do
                local port=$((DEVICE_PORT + i))
                local part="" mem="$per_mem"
                SV_VISIBLE=""
                if [[ "$condition" == "no_sharing_tpc" ]]; then
                    if (( nslices > 0 )); then
                        local s=$(( i % nslices )) j=$(( i / nslices ))
                        local k=${slice_k[s]}
                        SV_VISIBLE="${uuids[s]}"
                        mem=$(ns_gpu_mem "$k")
                        (( k > 1 )) && part="share:${j}/${k}"
                        assign+="${assign:+;}${i}:slice${s}${part:+:${part#share:}}"
                    elif [[ "$SP_MODE" != mig ]]; then
                        part=$(seq -s ' ' "$cur" "$((cur + tpcs_per - 1))")
                        cur=$((cur + tpcs_per))
                    fi
                fi
                dev_log="$LOG_DIR/device_${condition}_${i}_rps${rps}_n${ntasks}.log"
                # BE / SP: one task per server, fifo, inline.
                if ! start_vllm_device "$port" "$mem" "$dev_log" fifo inline shared "$part"; then
                    echo "[run.sh] $condition could not place server $((i+1))/$ntasks" >&2
                    record_failure "$out_dir" "$condition" "$ntasks" "$rps" \
                        "device_start_${i}_of_${ntasks}" "$dev_log"
                    return 1
                fi
                # A server meant to be pinned that did not pin is not SP: stop
                # rather than measure an unpartitioned run under SP's name.
                if [[ -n "$part" ]] && ! grep -q "\[TPC\] libsmctrl: pinned" "$dev_log"; then
                    echo "[run.sh] ERROR: server $i did not report TPC pinning — see $dev_log" >&2
                    record_failure "$out_dir" "$condition" "$ntasks" "$rps" "tpc_pin_missing_${i}" "$dev_log"
                    return 1
                fi
                urls+="${urls:+,}localhost:${port}"
            done
            [[ -n "$assign" ]] && LAST_SP_ASSIGN="$assign"
            $PYTHON -u experiments/RTVSntask/llm/run.py \
                --tasks       "$tasks_csv" \
                --condition   "$condition" \
                --device-urls "$urls" \
                --backbone    "$BACKBONE" \
                --rps         "$run_rps" \
                --duration    "$PHASE_DURATION" \
                --warmup-secs "$WARMUP_SECS" \
                --exp-dir     "$out_dir" \
                --trace-file  "$trace_file" \
                --deploy-timeout "$DEPLOY_TIMEOUT" \
                || { record_failure "$out_dir" "$condition" "$ntasks" "$rps" "workload" "$dev_log"; return 1; }
            ;;

        sharing|sharing_inline)
            mig_off || { record_failure "$out_dir" "$condition" "$ntasks" "$rps" "mig_off" ""; return 1; }
            # Shared: one engine, all N LoRAs, fifo; per-task output work on
            # each task's own thread (sharing) or on the step loop (sharing_inline).
            local heads=threaded
            [[ "$condition" == sharing_inline ]] && heads=inline
            dev_log="$LOG_DIR/device_${condition}_rps${rps}_n${ntasks}.log"
            if ! start_vllm_device "$DEVICE_PORT" "$GPU_MEM_UTIL" "$dev_log" fifo "$heads" shared; then
                record_failure "$out_dir" "$condition" "$ntasks" "$rps" "device_start" "$dev_log"
                return 1
            fi
            $PYTHON -u experiments/RTVSntask/llm/run.py \
                --tasks       "$tasks_csv" \
                --condition   "$condition" \
                --device-urls "localhost:${DEVICE_PORT}" \
                --backbone    "$BACKBONE" \
                --rps         "$run_rps" \
                --duration    "$PHASE_DURATION" \
                --warmup-secs "$WARMUP_SECS" \
                --exp-dir     "$out_dir" \
                --trace-file  "$trace_file" \
                --deploy-timeout "$DEPLOY_TIMEOUT" \
                || { record_failure "$out_dir" "$condition" "$ntasks" "$rps" "workload" "$dev_log"; return 1; }
            ;;

        fmvisor)
            mig_off || { record_failure "$out_dir" "$condition" "$ntasks" "$rps" "mig_off" ""; return 1; }
            # One server, process isolation: the engine in the backbone
            # process (STFQ), one process per task. --worker-mode is ignored
            # in process mode (each task has its own process).
            dev_log="$LOG_DIR/device_fmvisor_rps${rps}_n${ntasks}.log"
            if ! start_vllm_device "$DEVICE_PORT" "$GPU_MEM_UTIL" "$dev_log" stfq inline process "" \
                    --max-tasks "$((ntasks * 2 + 4))" --slots-per-task "$FM_SLOTS" \
                    --task-backlog-cap "$FM_BACKLOG_CAP" --llm-max-prompt-len "$FM_MAX_PROMPT_LEN"; then
                record_failure "$out_dir" "$condition" "$ntasks" "$rps" "device_start" "$dev_log"
                return 1
            fi
            $PYTHON -u experiments/RTVSntask/llm/run.py \
                --tasks       "$tasks_csv" \
                --condition   fmvisor \
                --device-urls "localhost:${DEVICE_PORT}" \
                --backbone    "$BACKBONE" \
                --rps         "$run_rps" \
                --duration    "$PHASE_DURATION" \
                --warmup-secs "$WARMUP_SECS" \
                --exp-dir     "$out_dir" \
                --trace-file  "$trace_file" \
                --deploy-timeout "$DEPLOY_TIMEOUT" \
                || { record_failure "$out_dir" "$condition" "$ntasks" "$rps" "workload" "$dev_log"; return 1; }
            ;;

        *)
            echo "[run.sh] ERROR: unknown condition '$condition'" >&2
            return 1
            ;;
    esac

    cat > "${out_dir}/run_config.json" <<EOF
{
  "condition": "${condition}",
  "tasks": "${tasks_csv}",
  "num_tasks": ${ntasks},
  "backbone": "${BACKBONE}",
  "cuda_device": "${CUDA_DEVICE}",
  "rps_per_task": ${rps},
  "rps_applied": ${run_rps},
  "rps_single_aggregate": $([[ "$condition" == single_* ]] && echo "${run_rps}" || echo "null"),
  "phase_duration_s": ${PHASE_DURATION},
  "gpu_memory_utilization_full": ${GPU_MEM_UTIL},
  "gpu_memory_utilization_per_server": ${LAST_PER_MEM},
  "max_model_len": ${MAX_MODEL_LEN},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "vllm_driver": "step",
  "max_num_seqs": "${MAX_NUM_SEQS:-default}",
  "llm_queue_k": ${LLM_QUEUE_K},
  "staging": "$([[ "$condition" == fmvisor ]] && echo gpu_slots || echo gpu)",
  "worker_mode": "$(case "$condition" in fmvisor) echo task_processes ;; sharing) echo threaded ;; *) echo inline ;; esac)",
  "isolation_mode": "$([[ "$condition" == fmvisor ]] && echo process || echo shared)",
  "scheduler": "$([[ "$condition" == fmvisor ]] && echo stfq || echo fifo)",
  "fm_slots": ${FM_SLOTS},
  "fm_backlog_cap": ${FM_BACKLOG_CAP},
  "sp_mode": "$([[ "$condition" == no_sharing_tpc ]] && echo "${SP_MODE}" || echo null)",
  "sp_layout": "${LAST_SP_LAYOUT:-null}",
  "sp_assignment": "${LAST_SP_ASSIGN:-null}",
  "device_port_base": ${DEVICE_PORT}
}
EOF

    stop_devices
    if [[ "$condition" == no_sharing_tpc && "$MIG_TURNED_ON" == 1 ]]; then mig_off || true; fi
}

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
abs_results="$(cd "$SERVING_DIR" && mkdir -p "$RESULTS_BASE" && realpath "$RESULTS_BASE")"

IFS=',' read -ra RPS_LIST <<< "$RPS_SWEEP"
IFS=',' read -ra NTASKS_LIST <<< "$NUM_TASKS_SWEEP"

for ntasks in "${NTASKS_LIST[@]:-}"; do
    [[ -z "$ntasks" ]] && continue
    if (( ntasks < 1 )); then
        echo "[run.sh] ERROR: ntasks=$ntasks must be >= 1"; exit 1
    fi

    # Build the task list: cycle the adapter pool with modulo. Past the pool
    # size every slot gets a __appN suffix so task IDs stay unique; run.py
    # strips the suffix to find the adapter directory on disk.
    tasks_arr=()
    if (( ntasks <= POOL_SIZE )); then
        for ((i=0; i<ntasks; i++)); do tasks_arr+=("${ALL_TASKS[$i]}"); done
    else
        for ((i=0; i<ntasks; i++)); do
            base="${ALL_TASKS[$((i % POOL_SIZE))]}"
            tasks_arr+=("${base}__app$((i / POOL_SIZE + 1))")
        done
    fi
    tasks_csv=$(IFS=','; echo "${tasks_arr[*]}")

    conditions=()
    if [[ -n "${CONDITIONS:-}" ]]; then
        IFS=',' read -ra CUSTOM <<< "$CONDITIONS"
        for c in "${CUSTOM[@]}"; do
            if [[ "$c" == "single" ]]; then
                conditions+=("single_${tasks_arr[0]}")
            elif [[ ! "$c" =~ ^(no_sharing_tpc|no_sharing|sharing|sharing_inline|fmvisor)$ ]]; then
                echo "[run.sh] ERROR: unknown condition '$c'" >&2; exit 1
            else
                conditions+=("$c")
            fi
        done
    else
        # Same three arms, in the same order, as RTVSntask/vision/run.sh.
        # The single_* baseline exists (see run_condition) but is NOT run by
        # default — vision never ran it either, so leaving it out keeps the
        # figures directly comparable. Opt in with CONDITIONS=single,...
        [[ "$INCLUDE_TPC" == "1" ]] && conditions+=("no_sharing_tpc")
        conditions+=("sharing")
        conditions+=("fmvisor")
        conditions+=("no_sharing")
    fi

    for rps in "${RPS_LIST[@]}"; do
        echo ""
        echo "################################################################"
        echo "  ntasks=$ntasks  rps=$rps"
        echo "  tasks: $tasks_csv"
        echo "################################################################"
        for condition in "${conditions[@]}"; do
            run_condition "$condition" "$rps" "$ntasks" "$tasks_csv" \
                || echo "[run.sh] WARNING: $condition ntasks=$ntasks rps=$rps failed — continuing"
        done
    done
done

echo ""
echo "[run.sh] All done. Results in $RESULTS_BASE"
echo "[run.sh] Plot with: $PYTHON experiments/RTVSntask/llm/plot.py --results $RESULTS_BASE"
