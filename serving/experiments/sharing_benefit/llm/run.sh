#!/usr/bin/env bash
# sharing_benefit/llm — multi-LoRA on vLLM, "co-locate vs partition" comparison.
#
# LLM analog of sharing_benefit/tpc/run.sh. Tasks here are distinct LoRA
# adapters over the same Qwen2.5 backbone (default qwenA, qwenB, ...).
#
# Conditions per (ntasks, rps):
#   single_<task>     — 1 vLLM server, that LoRA only            (uses full GPU mem)
#   no_sharing        — N vLLM servers, each backbone+1 LoRA      (gpu mem split N ways)
#   no_sharing_tpc    — N vLLM servers each pinned to a TPC set   (experimental, off by default)
#   sharing           — 1 vLLM server, all N LoRAs (Punica multi-LoRA, full GPU mem)
#
# Prerequisite: generate adapters under <FMTK>/models/llm/finetuned/.
#     python experiments/fair_share/llm/make_loras.py \
#         --backbone qwen2.5-1.5b --names qwenA,qwenB,qwenC,qwenD,qwenE
#
# Quick start (smoke test, 2 tasks):
#     NUM_TASKS_SWEEP=2 RPS_SWEEP=2 PHASE_DURATION=90 \
#         bash experiments/sharing_benefit/llm/run.sh
#
# Optional knobs (env vars):
#   CONDA_ENV             fmtk
#   BACKBONE              qwen2.5-1.5b
#   TASKS                 qwenA,qwenB,qwenC,qwenD,qwenE   (canonical task list; ntasks selects first N)
#   NUM_TASKS_SWEEP       2                                (comma-separated)
#   RPS_SWEEP             2                                (comma-separated, per-task RPS)
#   PHASE_DURATION        120
#   GPU_MEM_UTIL          0.85   (used by single/sharing; no_sharing* gets GPU_MEM_UTIL/N each)
#   MAX_MODEL_LEN         2048
#   DEVICE_PORT           8000
#   DEVICE_STARTUP_WAIT   90     (vLLM cold load is slow)
#   INCLUDE_TPC           0      (set to 1 to also run no_sharing_tpc — needs libsmctrl)
#   TPC_MODE              libsmctrl
#   CONDITIONS            (optional explicit override, e.g. "single,no_sharing,sharing")
#   RESULTS_BASE          experiments/sharing_benefit/llm/results

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
BACKBONE="${BACKBONE:-qwen2.5-3b}"
TASKS_CSV="${TASKS:-qwenA,qwenB,qwenC,qwenD,qwenE}"
RPS_SWEEP="${RPS_SWEEP:-2}"
NUM_TASKS_SWEEP="${NUM_TASKS_SWEEP:-2}"
PHASE_DURATION="${PHASE_DURATION:-600}"
DEVICE_PORT="${DEVICE_PORT:-8000}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-90}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"   # per-request output cap; higher = more KV pressure on BE
INCLUDE_TPC="${INCLUDE_TPC:-0}"
TPC_MODE="${TPC_MODE:-libsmctrl}"
WARMUP_SECS="${WARMUP_SECS:-10}"
RESULTS_BASE="${RESULTS_BASE:-experiments/sharing_benefit/llm/results}"

IFS=',' read -ra ALL_TASKS <<< "$TASKS_CSV"

LOG_DIR="${RESULTS_BASE}/logs"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  Sharing Benefit (LLM)"
echo "  Conda env      : $CONDA_ENV"
echo "  Backbone       : $BACKBONE  (max_model_len=$MAX_MODEL_LEN)"
echo "  Task pool      : ${ALL_TASKS[*]}"
echo "  NUM_TASKS sweep: $NUM_TASKS_SWEEP"
echo "  RPS sweep      : $RPS_SWEEP"
echo "  Duration/run   : ${PHASE_DURATION}s"
echo "  GPU mem (full) : $GPU_MEM_UTIL  (no_sharing* divides by N)"
echo "  max_new_tokens : $MAX_NEW_TOKENS"
echo "  TPC condition  : $([[ $INCLUDE_TPC == 1 ]] && echo "enabled (mode=$TPC_MODE)" || echo skipped)"
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

# start_vllm_device PORT GPU_MEM LOG [TPC_PART]
#   Boots one vLLM device server. Adapter load happens later via run.py's deploy().
start_vllm_device() {
    local port="$1" gpu_mem="$2" log="$3"
    local tpc_part="${4:-}"

    pkill -f "device/main.py.*--port ${port}" 2>/dev/null || true
    sleep 1

    local tpc_args=""
    [[ -n "$tpc_part" ]] && tpc_args="--tpc-mode $TPC_MODE --tpc-partition $tpc_part"

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
        $tpc_args \
        > "$log" 2>&1 &
    local pid=$!
    DEVICE_PIDS+=("$pid")
    ACTIVE_PORTS+=("$port")
    echo "[run.sh] PID=$pid  log=$log" >&2

    # Poll for vLLM ready signal (model load can take 30–60s).
    local waited=0
    while (( waited < DEVICE_STARTUP_WAIT )); do
        if grep -q -E "Runtime application started|application started|gRPC server.*listen" "$log" 2>/dev/null; then
            echo "[run.sh] Server ready (after ${waited}s)" >&2
            sleep 2
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[run.sh] ERROR: server exited early — see $log" >&2
            tail -n 60 "$log" >&2 || true
            return 1
        fi
        sleep 2; waited=$(( waited + 2 ))
    done
    echo "[run.sh] WARNING: ready signal not seen in ${DEVICE_STARTUP_WAIT}s, proceeding" >&2
}

# Compute per-server GPU memory util as $GPU_MEM_UTIL/$n, clamped >= 0.10
ns_gpu_mem() {
    local n="$1"
    awk -v base="$GPU_MEM_UTIL" -v n="$n" 'BEGIN{
        v = base/n;
        if (v < 0.10) v = 0.10;
        printf "%.3f", v;
    }'
}

# ---------------------------------------------------------------------------
# Single condition runner
# ---------------------------------------------------------------------------
run_condition() {
    local condition="$1" rps="$2" ntasks="$3" tasks_csv="$4"

    local ntasks_dir="${abs_results}/ntasks_${ntasks}/rps_${rps}"
    local trace_file="${abs_results}/singles/rps_${rps}/trace.json"
    local out_dir
    if [[ "$condition" == single_* ]]; then
        out_dir="${abs_results}/singles/rps_${rps}/${condition}"
    else
        out_dir="${ntasks_dir}/${condition}"
    fi

    echo ""
    echo "================================================================"
    echo "  condition=$condition  rps=$rps  ntasks=$ntasks"
    echo "  tasks    = ${tasks_csv}"
    echo "  Results  : $out_dir"
    echo "================================================================"

    if [[ -f "${out_dir}/latencies.csv" ]]; then
        echo "[run.sh] Skipping $condition — results already exist"
        if [[ "$condition" == single_* ]]; then
            mkdir -p "$ntasks_dir"
            ln -sfn "$out_dir" "${ntasks_dir}/${condition}" 2>/dev/null || true
        fi
        return 0
    fi

    stop_devices
    DEVICE_PIDS=()
    ACTIVE_PORTS=()
    mkdir -p "$out_dir"

    IFS=',' read -ra TASKS_ARR <<< "$tasks_csv"

    case "$condition" in
        single_*)
            local single_task="${condition#single_}"
            start_vllm_device "$DEVICE_PORT" "$GPU_MEM_UTIL" \
                "$LOG_DIR/device_${condition}_rps${rps}_n${ntasks}.log"
            $PYTHON -u experiments/sharing_benefit/llm/run.py \
                --tasks       "$single_task" \
                --condition   "$condition" \
                --device-urls "localhost:${DEVICE_PORT}" \
                --backbone    "$BACKBONE" \
                --rps         "$rps" \
                --duration    "$PHASE_DURATION" \
                --warmup-secs "$WARMUP_SECS" \
                --exp-dir     "$out_dir" \
                --trace-file  "$trace_file"
            ;;

        no_sharing)
            local urls="" per_mem
            per_mem=$(ns_gpu_mem "$ntasks")
            for ((i=0; i<ntasks; i++)); do
                local port=$((DEVICE_PORT + i))
                start_vllm_device "$port" "$per_mem" \
                    "$LOG_DIR/device_ns_${i}_rps${rps}_n${ntasks}.log"
                urls+="${urls:+,}localhost:${port}"
            done
            $PYTHON -u experiments/sharing_benefit/llm/run.py \
                --tasks       "$tasks_csv" \
                --condition   no_sharing \
                --device-urls "$urls" \
                --backbone    "$BACKBONE" \
                --rps         "$rps" \
                --duration    "$PHASE_DURATION" \
                --warmup-secs "$WARMUP_SECS" \
                --exp-dir     "$out_dir" \
                --trace-file  "$trace_file"
            ;;

        no_sharing_tpc)
            # Split TPCs evenly. TPC ~= SMs/2 on most NVIDIA arch.
            local total_tpcs
            total_tpcs=$($PYTHON -c "
import torch
sm = torch.cuda.get_device_properties('${CUDA_DEVICE}').multi_processor_count
print(sm // 2)
")
            local base_per=$((total_tpcs / ntasks))
            local urls="" per_mem cur=0
            per_mem=$(ns_gpu_mem "$ntasks")
            for ((i=0; i<ntasks; i++)); do
                local port=$((DEVICE_PORT + i))
                local start_tpc=$cur
                local end_tpc=$((cur + base_per - 1))
                cur=$((cur + base_per))
                local part
                part=$(seq -s ' ' "$start_tpc" "$end_tpc")
                echo "[run.sh] TPC server $i: task=${TASKS_ARR[$i]} tpcs=[${part}]"
                start_vllm_device "$port" "$per_mem" \
                    "$LOG_DIR/device_ns_tpc_${i}_rps${rps}_n${ntasks}.log" \
                    "$part"
                urls+="${urls:+,}localhost:${port}"
            done
            $PYTHON -u experiments/sharing_benefit/llm/run.py \
                --tasks       "$tasks_csv" \
                --condition   no_sharing_tpc \
                --device-urls "$urls" \
                --backbone    "$BACKBONE" \
                --rps         "$rps" \
                --duration    "$PHASE_DURATION" \
                --warmup-secs "$WARMUP_SECS" \
                --exp-dir     "$out_dir" \
                --trace-file  "$trace_file"
            ;;

        sharing)
            start_vllm_device "$DEVICE_PORT" "$GPU_MEM_UTIL" \
                "$LOG_DIR/device_sharing_rps${rps}_n${ntasks}.log"
            $PYTHON -u experiments/sharing_benefit/llm/run.py \
                --tasks       "$tasks_csv" \
                --condition   sharing \
                --device-urls "localhost:${DEVICE_PORT}" \
                --backbone    "$BACKBONE" \
                --rps         "$rps" \
                --duration    "$PHASE_DURATION" \
                --warmup-secs "$WARMUP_SECS" \
                --exp-dir     "$out_dir" \
                --trace-file  "$trace_file"
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
  "phase_duration_s": ${PHASE_DURATION},
  "gpu_memory_utilization_full": ${GPU_MEM_UTIL},
  "max_model_len": ${MAX_MODEL_LEN},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "device_port_base": ${DEVICE_PORT}
}
EOF

    if [[ "$condition" == single_* ]]; then
        mkdir -p "$ntasks_dir"
        ln -sfn "$out_dir" "${ntasks_dir}/${condition}" 2>/dev/null || true
        echo "[run.sh] Linked: ${ntasks_dir}/${condition} -> $out_dir"
    fi

    stop_devices
}

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
abs_results="$(cd "$SERVING_DIR" && realpath "$RESULTS_BASE")"

IFS=',' read -ra RPS_LIST <<< "$RPS_SWEEP"
IFS=',' read -ra NTASKS_LIST <<< "$NUM_TASKS_SWEEP"

for ntasks in "${NTASKS_LIST[@]:-}"; do
    [[ -z "$ntasks" ]] && continue
    if (( ntasks < 1 || ntasks > ${#ALL_TASKS[@]} )); then
        echo "[run.sh] ERROR: ntasks=$ntasks out of range (1..${#ALL_TASKS[@]} from TASKS=$TASKS_CSV)"
        exit 1
    fi

    tasks_arr=("${ALL_TASKS[@]:0:$ntasks}")
    tasks_csv=$(IFS=','; echo "${tasks_arr[*]}")

    # Build the condition list
    conditions=()
    if [[ -n "${CONDITIONS:-}" ]]; then
        IFS=',' read -ra CUSTOM <<< "$CONDITIONS"
        for c in "${CUSTOM[@]}"; do
            if [[ "$c" == "single" ]]; then
                for t in "${tasks_arr[@]}"; do conditions+=("single_${t}"); done
            else
                conditions+=("$c")
            fi
        done
    else
        for t in "${tasks_arr[@]}"; do conditions+=("single_${t}"); done
        conditions+=("no_sharing")
        [[ "$INCLUDE_TPC" == "1" ]] && conditions+=("no_sharing_tpc")
        conditions+=("sharing")
    fi

    for rps in "${RPS_LIST[@]}"; do
        echo ""
        echo "################################################################"
        echo "  ntasks=$ntasks  rps=$rps  tasks=$tasks_csv"
        echo "################################################################"
        for condition in "${conditions[@]}"; do
            run_condition "$condition" "$rps" "$ntasks" "$tasks_csv" \
                || echo "[run.sh] WARNING: $condition ntasks=$ntasks rps=$rps failed — continuing"
        done
    done
done

echo ""
echo "[run.sh] All done. Results in $RESULTS_BASE"
