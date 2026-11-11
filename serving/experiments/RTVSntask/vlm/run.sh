#!/usr/bin/env bash
# RTVSntask/vlm — response time vs. number of tasks, on vLLM multi-LoRA.
#
# VLM analog of RTVSntask/llm/run.sh. Tasks are LoRA adapters over a shared
# Qwen2-VL-2B backbone, and every request carries an MNIST OCR image.
# NUM_TASKS_SWEEP is unbounded: when it exceeds the adapter pool, adapters are
# cycled with a __appN suffix so each slot is a distinct task ID backed by its
# own LoRA slot in the engine.
#
# The adapters are language-side only (q_proj/v_proj). vLLM 0.6.6 cannot serve
# a LoRA over Qwen2-VL's vision encoder — it is an explicit upstream TODO.
#
# Conditions per (ntasks, rps):
#   no_sharing_tpc  — N vLLM servers, each pinned to a TPC set (needs libsmctrl)
#   sharing         — 1 vLLM server, all N LoRAs (Punica multi-LoRA)
#   no_sharing      — N vLLM servers, each a full backbone copy + 1 LoRA;
#                     gpu_memory_utilization split N ways
#
# These are the same three arms, in the same order, that RTVSntask/vision runs.
# A single_<task> baseline (1 server, that LoRA only, driven at the AGGREGATE
# rate rps*ntasks) is implemented but off by default, because vision never ran
# it. Opt in with CONDITIONS=single,sharing,no_sharing.
#
# no_sharing is EXPECTED to fail beyond a handful of tasks: N full 2B backbone
# copies (~4.4 GB of fp16 weights each) do not fit. That failure is a result,
# not a bug — when a condition cannot be placed, a failure.json recording the
# classified reason is written to its results dir and the sweep continues.
#
# Prerequisite: adapters under <FMTK>/models/vlm/finetuned/.
#     python experiments/RTVSntask/vlm/make_loras.py --names qwenvlA,qwenvlB
#
# Quick start (smoke test):
#     NUM_TASKS_SWEEP=2 RPS_SWEEP=2 PHASE_DURATION=90 \
#         bash experiments/RTVSntask/vlm/run.sh
#
# Full sweep:
#     NUM_TASKS_SWEEP=1,2,4,6,8,10 RPS_SWEEP=2 \
#         bash experiments/RTVSntask/vlm/run.sh
#
# Optional knobs (env vars):
#   CONDA_ENV             fmtk
#   BACKBONE              qwen2-vl-2b
#   TASKS                 qwenvlA,qwenvlB (adapter pool; cycled with __appN past its size)
#   NUM_TASKS_SWEEP       1,2,4,6,8      (comma-separated; no upper bound)
#   RPS_SWEEP             2              (comma-separated, PER-TASK rps)
#   PHASE_DURATION        600
#   GPU_MEM_UTIL          0.85   (single/sharing use it whole; no_sharing* gets /N each)
#   MAX_MODEL_LEN         512    (an OCR request is ~64 tokens: ~4 of them vision)
#   MAX_NEW_TOKENS        256
#   DEVICE_PORT           8000   (base port; servers use 8000, 8001, ...)
#   DEVICE_STARTUP_WAIT   120    (vLLM cold load is slow, and slower with N servers)
#   DEPLOY_TIMEOUT        1800   (gRPC deadline for the load control call)
#   INCLUDE_TPC           1      (0 to skip no_sharing_tpc — it needs libsmctrl)
#   TPC_MODE              libsmctrl
#   CONDITIONS            (explicit override, e.g. "sharing,no_sharing" or
#                          "single,sharing"; "single" expands to the first task slot)
#   RESULTS_BASE          experiments/RTVSntask/vlm/results

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
BACKBONE="${BACKBONE:-qwen2-vl-2b}"
TASKS_CSV="${TASKS:-qwenvlA,qwenvlB}"
RPS_SWEEP="${RPS_SWEEP:-2}"
NUM_TASKS_SWEEP="${NUM_TASKS_SWEEP:-1,2,3,4,5}"
PHASE_DURATION="${PHASE_DURATION:-600}"
DEVICE_PORT="${DEVICE_PORT:-8000}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-180}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
# IGNORE_EOS=1 forces every request to emit exactly MAX_NEW_TOKENS.
# Off by default: generation stops naturally at EOS.
# Be aware of what that means for this dataset — the OCR prompt asks for one
# character, so requests stop at ~2 tokens and are ~16x cheaper in DECODE than
# the LLM arm's 32-token requests. Measured: 104 ms/request natural vs 335 ms
# forced. At 6.1 rps aggregate that is 0.63 vs 2.05 requests resident on the
# GPU — i.e. natural length leaves the GPU idle, and backbone sharing has
# nothing to consolidate (sharing lost by 58%; with 32 tokens it won by 6%).
# Set IGNORE_EOS=1 for a decode-dominated, LLM-comparable operating point.
IGNORE_EOS="${IGNORE_EOS:-0}"
INCLUDE_TPC="${INCLUDE_TPC:-1}"
TPC_MODE="${TPC_MODE:-libsmctrl}"
WARMUP_SECS="${WARMUP_SECS:-10}"
# gRPC deadline for the "load" control call. The client default is 120s,
# which a vLLM engine build over a networked filesystem blows straight past.
DEPLOY_TIMEOUT="${DEPLOY_TIMEOUT:-1800}"
RESULTS_BASE="${RESULTS_BASE:-experiments/RTVSntask/vlm/results}"

IFS=',' read -ra ALL_TASKS <<< "$TASKS_CSV"
POOL_SIZE=${#ALL_TASKS[@]}

LOG_DIR="${RESULTS_BASE}/logs"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  RT vs NumTasks (VLM / multi-LoRA, OCR images)"
echo "  Conda env      : $CONDA_ENV"
echo "  Backbone       : $BACKBONE  (max_model_len=$MAX_MODEL_LEN)"
echo "  Adapter pool   : ${ALL_TASKS[*]}  (cycled with __appN past ${POOL_SIZE})"
echo "  NUM_TASKS sweep: $NUM_TASKS_SWEEP"
echo "  RPS sweep      : $RPS_SWEEP  (per task; singles run at rps*ntasks)"
echo "  Duration/run   : ${PHASE_DURATION}s"
echo "  GPU mem (full) : $GPU_MEM_UTIL  (no_sharing* divides by N)"
echo "  max_new_tokens : $MAX_NEW_TOKENS"
echo "  ignore_eos     : $([[ $IGNORE_EOS == 1 ]] && echo "on (every request emits exactly $MAX_NEW_TOKENS tokens)" || echo off)"
echo "  Deploy timeout : ${DEPLOY_TIMEOUT}s"
echo "  TPC condition  : $([[ $INCLUDE_TPC == 1 ]] && echo "enabled (mode=$TPC_MODE)" || echo skipped)"
echo "  Base port      : $DEVICE_PORT"
echo "  Results        : $RESULTS_BASE"
echo "================================================================"

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
trap 'stop_devices' EXIT

# classify_failure LOG -> prints a reason slug
#   Reads a device log and names why the server could not be placed. This is
#   the payload of failure.json: at high ntasks, no_sharing loses on memory and
#   we want the plot to say *which* memory.
classify_failure() {
    local log="$1"
    [[ -f "$log" ]] || { echo "no_log"; return; }
    # Order matters: the multimodal profiling mismatch below produces the same
    # "must match the size of tensor" text as a genuine adapter shape problem,
    # so it must be tested first or it gets misfiled as adapter_shape_mismatch.
    #
    # It fires at ENGINE CONSTRUCTION (profile_run), not on a request: vLLM
    # builds a dummy image at Qwen2-VL's default max_pixels, reserving 16384
    # vision tokens against a ~5120-token batch budget, and only does so when
    # enable_lora is set. The fix is the min_pixels/max_pixels cap in
    # fmtk/components/backbones/qwen2_vl_vllm.py — NOT a change to the adapters.
    if   grep -qi "must match the size of tensor" "$log" && \
         grep -qi "profile_run\|determine_num_available_blocks" "$log"; then
        echo "mm_profile_token_mismatch"
    elif grep -qi "AsyncEngineDeadError\|Background loop has errored" "$log" && \
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

# start_vllm_device PORT GPU_MEM LOG [TPC_PART]
start_vllm_device() {
    local port="$1" gpu_mem="$2" log="$3"
    local tpc_part="${4:-}"

    pkill -f "device/main.py.*--port ${port}" 2>/dev/null || true
    sleep 1

    local tpc_args=""
    [[ -n "$tpc_part" ]] && tpc_args="--tpc-mode $TPC_MODE --tpc-partition $tpc_part"
    local eos_args=""
    [[ "$IGNORE_EOS" == "1" ]] && eos_args="--ignore-eos"

    echo "[run.sh] Starting vLLM server port=$port gpu_mem=$gpu_mem ${tpc_part:+tpc=[$tpc_part]}" >&2

    # shellcheck disable=SC2086
    $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port                   "$port"           \
        --runtime-type           vllm              \
        --cuda                   "$CUDA_DEVICE"    \
        --scheduler-policy       fifo              \
        --gpu-memory-utilization "$gpu_mem"        \
        --max-model-len          "$MAX_MODEL_LEN"  \
        --max-new-tokens         "$MAX_NEW_TOKENS" \
        $tpc_args $eos_args \
        > "$log" 2>&1 &
    local pid=$!
    DEVICE_PIDS+=("$pid")
    ACTIVE_PORTS+=("$port")
    echo "[run.sh] PID=$pid  log=$log" >&2

    local waited=0
    while (( waited < DEVICE_STARTUP_WAIT )); do
        if grep -q -E "Runtime application started|application started|gRPC server.*listen" "$log" 2>/dev/null; then
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

    local dev_log
    case "$condition" in
        single_*)
            local single_task="${condition#single_}"
            dev_log="$LOG_DIR/device_${condition}_rps${rps}_n${ntasks}.log"
            if ! start_vllm_device "$DEVICE_PORT" "$GPU_MEM_UTIL" "$dev_log"; then
                record_failure "$out_dir" "$condition" "$ntasks" "$rps" "device_start" "$dev_log"
                return 1
            fi
            $PYTHON -u experiments/RTVSntask/vlm/run.py \
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
            per_mem=$(ns_gpu_mem "$ntasks")
            LAST_PER_MEM="$per_mem"
            awk -v p="$per_mem" -v n="$ntasks" -v b="$GPU_MEM_UTIL" 'BEGIN{
                if (p*n > b + 1e-9)
                    printf "[run.sh] NOTE: %d servers x %.3f = %.3f exceeds GPU_MEM_UTIL=%.2f — over-subscribed, launching anyway\n", n, p, p*n, b;
            }' >&2

            local tpcs_per=0
            if [[ "$condition" == "no_sharing_tpc" ]]; then
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

            local cur=0
            for ((i=0; i<ntasks; i++)); do
                local port=$((DEVICE_PORT + i))
                local part=""
                if [[ "$condition" == "no_sharing_tpc" ]]; then
                    part=$(seq -s ' ' "$cur" "$((cur + tpcs_per - 1))")
                    cur=$((cur + tpcs_per))
                fi
                dev_log="$LOG_DIR/device_${condition}_${i}_rps${rps}_n${ntasks}.log"
                if ! start_vllm_device "$port" "$per_mem" "$dev_log" "$part"; then
                    echo "[run.sh] $condition could not place server $((i+1))/$ntasks" >&2
                    record_failure "$out_dir" "$condition" "$ntasks" "$rps" \
                        "device_start_${i}_of_${ntasks}" "$dev_log"
                    return 1
                fi
                urls+="${urls:+,}localhost:${port}"
            done
            $PYTHON -u experiments/RTVSntask/vlm/run.py \
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

        sharing)
            dev_log="$LOG_DIR/device_sharing_rps${rps}_n${ntasks}.log"
            if ! start_vllm_device "$DEVICE_PORT" "$GPU_MEM_UTIL" "$dev_log"; then
                record_failure "$out_dir" "$condition" "$ntasks" "$rps" "device_start" "$dev_log"
                return 1
            fi
            $PYTHON -u experiments/RTVSntask/vlm/run.py \
                --tasks       "$tasks_csv" \
                --condition   sharing \
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
  "ignore_eos": ${IGNORE_EOS},
  "device_port_base": ${DEVICE_PORT}
}
EOF

    stop_devices
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
# vlm/plot.py mirrors llm/plot.py (identical latencies.csv / task_results.csv
# schema and identical styling); it only differs in its defaults.
echo "[run.sh] Plot with: $PYTHON experiments/RTVSntask/vlm/plot.py --results $RESULTS_BASE"
