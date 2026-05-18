#!/bin/bash
# Sharing Benefit + TPC Isolation experiment
# Sweeps over number of tasks (NUM_TASKS_SWEEP) and/or request rates (RPS_SWEEP).
#
# Task sets:
#   tsfm   — selects from: ecgclass heartrate diasbp sysbp gestureclass
#                           etth1fore weatherfore trafficfore eclfore exchangefore
#   vision — single_nyudepth, single_vocseg, no_sharing_tpc, no_sharing, sharing
#
# For tsfm, NUM_TASKS_SWEEP controls how many tasks (picked in canonical order above).
# Conditions run per (ntasks, rps) pair:
#   single_{task_i} for each i, then no_sharing_tpc, no_sharing, sharing
#
# Device servers are started on consecutive ports: DEVICE_PORT, DEVICE_PORT+1, ...
#
# Environment variables (all optional):
#   CONDA_ENV          fmtk
#   FMTK_DIR           ../../../FMTK
#   FMAAS_DIR          ../..
#   CUDA_DEVICE        cuda:0
#   TASK_SET           tsfm  (or vision)
#   BACKBONE           momentlarge  (tsfm) / dinobase-patch  (vision)
#   NUM_TASKS_SWEEP    2          (comma-separated counts, e.g. "2,4,6,8,10")
#   RPS_SWEEP          25         (comma-separated RPS values, e.g. "25" or "10,25,50")
#   PHASE_DURATION     300
#   DEVICE_PORT        8000       (base port; additional servers use 8001, 8002, ...)
#   MAX_BATCH_SIZE     100
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
TASK_SET="${TASK_SET:-vision}"  # vision or tsfm
RPS_SWEEP="${RPS_SWEEP:-3,4,6,8,10}"
NUM_TASKS_SWEEP="${NUM_TASKS_SWEEP:-2}"   
PHASE_DURATION="${PHASE_DURATION:-600}"
DEVICE_PORT="${DEVICE_PORT:-8000}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
RESULTS_BASE="${RESULTS_BASE:-experiments/sharing_benefit/tpc/results_vision}"
DEVICE_STARTUP_WAIT="${DEVICE_STARTUP_WAIT:-5}"
MAX_BATCH_WAIT_MS="${MAX_BATCH_WAIT_MS:-0}"
TPC_MODE="${TPC_MODE:-libsmctrl}"
WARMUP_BURST_SECS="${WARMUP_BURST_SECS:-15}"   # closed-loop GPU warmup before each open-loop run

# Canonical ordered list of all tsfm tasks (NUM_TASKS_SWEEP selects the first N)
ALL_TSFM_TASKS=(ecgclass gestureclass heartrate diasbp sysbp etth1fore weatherfore trafficfore eclfore exchangefore)

# Canonical ordered list of all vision tasks
ALL_VISION_TASKS=(nyudepth vocseg)

# Task-set-specific defaults
if [[ "$TASK_SET" == "vision" ]]; then
    BACKBONE="${BACKBONE:-swinlarge}"
    DECODER_DIR="${DECODER_DIR:-${FMTK_DIR}/models/vision/finetuned}"
    export NYUDEPTH_PATH="${NYUDEPTH_PATH:-../../FMTK/dataset/nyu-depth-v2}"
    export PASCALVOC_PATH="${PASCALVOC_PATH:-../../FMTK/dataset/PASCAL-VOC}"
else
    BACKBONE="${BACKBONE:-momentlarge}"
    DECODER_DIR="${DECODER_DIR:-${FMTK_DIR}/models/tsfm/finetuned}"
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
echo "  NUM_TASKS sweep: $NUM_TASKS_SWEEP"
echo "  RPS sweep      : $RPS_SWEEP"
echo "  Duration/run   : ${PHASE_DURATION}s"
echo "  TPC mode       : $TPC_MODE"
echo "  Warmup burst   : ${WARMUP_BURST_SECS}s"
echo "  Base port      : $DEVICE_PORT"
echo "  Results        : $RESULTS_BASE"
echo "================================================================"

# ---------------------------------------------------------------------------
# Device server lifecycle helpers
# ---------------------------------------------------------------------------
DEVICE_PIDS=()   # PIDs of currently running device servers
ACTIVE_PORTS=()  # ports in use (for pkill fallback)

stop_devices() {
    # Send SIGTERM to all tracked PIDs
    local pids_to_wait=()
    for pid in "${DEVICE_PIDS[@]:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "[run.sh] Stopping device PID=$pid (SIGTERM)" >&2
            kill "$pid" 2>/dev/null || true
            pids_to_wait+=("$pid")
        fi
    done

    # Wait up to 12s for SIGTERM, then escalate to SIGKILL
    local deadline=$((SECONDS + 12))
    for pid in "${pids_to_wait[@]:-}"; do
        while kill -0 "$pid" 2>/dev/null; do
            if [[ $SECONDS -ge $deadline ]]; then
                echo "[run.sh] WARNING: PID=$pid did not exit in 12s — sending SIGKILL" >&2
                kill -9 "$pid" 2>/dev/null || true
                break
            fi
            sleep 0.5
        done
        wait "$pid" 2>/dev/null || true
        echo "[run.sh] PID=$pid confirmed dead" >&2
    done

    # pkill fallback: catch any orphans that don't match tracked PIDs
    for port in "${ACTIVE_PORTS[@]:-}"; do
        if pkill -f "device/main.py.*--port ${port}" 2>/dev/null; then
            echo "[run.sh] pkill fallback killed orphan on port $port" >&2
            sleep 1
        fi
    done

    # Stop MPS daemon if running — ensures non-MPS conditions get exclusive GPU access.
    # ensure_mps_running() will restart it when needed.
    if [[ -S /tmp/nvidia-mps/control ]]; then
        echo "[run.sh] Stopping MPS daemon..." >&2
        echo quit | nvidia-cuda-mps-control 2>/dev/null || true
        sleep 1
        echo "[run.sh] MPS daemon stopped" >&2
    fi

    DEVICE_PIDS=()
    ACTIVE_PORTS=()
    echo "[run.sh] All device servers stopped. Sleeping 2s for GPU memory release..." >&2
    sleep 2
}
trap 'stop_devices' EXIT

# Ensure NVIDIA MPS daemon is running (idempotent).
ensure_mps_running() {
    echo "[run.sh] Starting NVIDIA MPS daemon..." >&2
    nvidia-cuda-mps-control -d
    # Wait for the control socket to appear (up to 10s)
    local i=0
    while [[ ! -S /tmp/nvidia-mps/control ]] && [[ $i -lt 20 ]]; do
        sleep 0.5; i=$((i+1))
    done
    if [[ ! -S /tmp/nvidia-mps/control ]]; then
        echo "[run.sh] ERROR: MPS control socket did not appear after 10s" >&2
        exit 1
    fi
    echo "[run.sh] MPS daemon started (socket ready)" >&2

    local ready=0
    i=0
    while [[ $i -lt 20 ]]; do
        if echo "get_server_list" | nvidia-cuda-mps-control >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 0.5
        i=$((i + 1))
    done
    if [[ $ready -ne 1 ]]; then
        echo "[run.sh] ERROR: MPS control daemon did not become responsive after startup" >&2
        exit 1
    fi
    echo "[run.sh] MPS control daemon is responsive" >&2
}

# start_device PORT SCHEDULER LOG RPS TASK_RATES [TPC_MODE [TPC_PARTITION [MPS_PCT [WORKER_MODE]]]]
# MPS_PCT: if non-empty, passes --mps-thread-pct to the device server (set before CUDA init).
# WORKER_MODE: "threaded" (default) | "inline" — per-task pipeline worker mode.
# Prints the PID of the started server on stdout; all other output goes to stderr.
start_device() {
    local port="$1" scheduler="$2" log="$3" rps="$4" task_rates="$5"
    local tpc_mode="${6:-none}" tpc_partition="${7:-}" mps_pct="${8:-}" worker_mode="${9:-threaded}"

    pkill -f "device/main.py.*--port ${port}" 2>/dev/null || true
    sleep 1

    local tpc_args=""
    if [[ "$tpc_mode" != "none" && -n "$tpc_partition" ]]; then
        tpc_args="--tpc-mode $tpc_mode --tpc-partition $tpc_partition"
    fi
    local mps_args=""
    if [[ -n "$mps_pct" ]]; then
        mps_args="--mps-thread-pct $mps_pct"
    fi
    local worker_args="--worker-mode $worker_mode"

    # Sanity check: warn if GPU memory is still mostly occupied from a previous run
    local gpu_free gpu_total
    if gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1) && \
       gpu_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1); then
        local pct_free=$(( gpu_free * 100 / gpu_total ))
        echo "[run.sh] GPU memory: ${gpu_free}/${gpu_total} MiB free (${pct_free}%)" >&2
        if [[ $pct_free -lt 20 ]]; then
            echo "[run.sh] WARNING: GPU memory is low (<20% free). Previous cleanup may be incomplete." >&2
        fi
    fi

    if [[ -n "$mps_pct" ]]; then
        echo "[run.sh] Starting device server port=$port scheduler=$scheduler mps_pct=${mps_pct}% ..." >&2
    else
        echo "[run.sh] Starting device server port=$port scheduler=$scheduler tpc=$tpc_mode ..." >&2
    fi

    # shellcheck disable=SC2086
    $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port              "$port"          \
        --runtime-type      pytorch          \
        --cuda              "$CUDA_DEVICE"   \
        --scheduler-policy  "$scheduler"     \
        --max-batch-size    "$MAX_BATCH_SIZE" \
        --max-batch-wait-ms "$MAX_BATCH_WAIT_MS" \
        --task-rates        "$task_rates"    \
        $tpc_args \
        $mps_args \
        $worker_args \
        > "$log" 2>&1 &
    local pid=$!
    echo "[run.sh] PID=$pid  log=$log" >&2
    ACTIVE_PORTS+=("$port")
    sleep "$DEVICE_STARTUP_WAIT"
    echo "$pid"
}

# ---------------------------------------------------------------------------
# run_condition CONDITION RPS NTASKS TASKS_CSV
#   TASKS_CSV — comma-separated active task names
# ---------------------------------------------------------------------------
run_condition() {
    local condition="$1" rps="$2" ntasks="$3" tasks_csv="$4"

    # Single-task conditions are stored once in a shared directory so they are
    # never re-run across ntasks sweeps.  A symlink in the ntasks_N tree points
    # back to that shared directory so plot.py finds results transparently.
    local out_dir trace_file run_tasks_csv
    local ntasks_dir="${abs_results}/ntasks_${ntasks}/rps_${rps}"
    # All conditions share the same trace so arrival times are identical across
    # single, no_sharing, no_sharing_tpc, no_sharing_mps, and sharing.
    trace_file="${abs_results}/singles/rps_${rps}/trace.json"
    if [[ "$condition" == single_* ]]; then
        local single_task="${condition#single_}"
        out_dir="${abs_results}/singles/rps_${rps}/${condition}"
        run_tasks_csv="$single_task"
    else
        out_dir="${ntasks_dir}/${condition}"
        run_tasks_csv="$tasks_csv"
    fi

    # Parse tasks into an array
    IFS=',' read -ra TASKS <<< "$tasks_csv"

    echo ""
    echo "================================================================"
    echo "  condition=$condition  rps=$rps  ntasks=$ntasks"
    echo "  tasks    =${tasks_csv}"
    echo "  Results  : $out_dir"
    echo "================================================================"

    # Skip if results already exist; for singles, also ensure symlink is in place
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

    # Helper: call run.py with common args
    run_py() {
        $PYTHON -u experiments/sharing_benefit/tpc/run.py \
            --task-set          "$TASK_SET" \
            --tasks             "$run_tasks_csv" \
            --backbone          "$BACKBONE" \
            --rps               "$rps" \
            --duration          "$PHASE_DURATION" \
            --exp-dir           "$out_dir" \
            --trace-file        "$trace_file" \
            --warmup-burst-secs "$WARMUP_BURST_SECS" \
            "$@"
    }

    case "$condition" in
        single_*)
            local task="${condition#single_}"
            local task_rates="${task}:${rps}"
            local port="$DEVICE_PORT"
            local pid
            # Single task per server → no per-task overlap to gain; use inline worker.
            pid=$(start_device "$port" "fifo" "$LOG_DIR/device_${condition}_rps${rps}_n${ntasks}.log" \
                  "$rps" "$task_rates" "none" "" "" "inline")
            DEVICE_PIDS+=("$pid")
            run_py --condition "$condition" --device-urls "localhost:${port}"
            ;;

        no_sharing_tpc)
            # Split TPCs evenly across N servers
            local total_tpcs
            total_tpcs=$($PYTHON -c "
import torch
sm = torch.cuda.get_device_properties('${CUDA_DEVICE}').multi_processor_count
print(sm // 2)  # TPCs ~ SMs/2
")
            # Distribute TPCs as evenly as possible: first (total%ntasks) servers
            # get one extra TPC so the remainder never piles up on the last server.
            # e.g. 5 TPCs / 3 tasks → [0,1]  [2,3]  [4]  (2,2,1)
            local base_per=$((total_tpcs / ntasks))
            local remainder=$((total_tpcs % ntasks))
            local device_urls=""
            local cur_tpc=0
            for ((i=0; i<ntasks; i++)); do
                local task="${TASKS[$i]}"
                local port=$((DEVICE_PORT + i))
                local alloc=$base_per
                # if [[ $i -lt $remainder ]]; then
                #     alloc=$((base_per + 1))
                # fi
                local start_tpc=$cur_tpc
                local end_tpc=$((cur_tpc + alloc - 1))
                cur_tpc=$((cur_tpc + alloc))
                local part
                part=$(seq -s ' ' "$start_tpc" "$end_tpc")
                echo "[run.sh] TPC server $i: task=${task} port=${port} tpcs=[${part}]"
                local pid
                # One task per server → inline worker (no per-task overlap to gain).
                pid=$(start_device "$port" "fifo" \
                      "$LOG_DIR/device_ns_tpc_${i}_rps${rps}_n${ntasks}.log" \
                      "$rps" "${task}:${rps}" "$TPC_MODE" "$part" "" "inline")
                DEVICE_PIDS+=("$pid")
                device_urls+="${device_urls:+,}localhost:${port}"
            done
            run_py --condition no_sharing_tpc --device-urls "$device_urls"
            ;;

        no_sharing)
            local device_urls=""
            for ((i=0; i<ntasks; i++)); do
                local task="${TASKS[$i]}"
                local port=$((DEVICE_PORT + i))
                local pid
                # One task per server → inline worker (no per-task overlap to gain).
                pid=$(start_device "$port" "fifo" \
                      "$LOG_DIR/device_ns_${i}_rps${rps}_n${ntasks}.log" \
                      "$rps" "${task}:${rps}" "none" "" "" "inline")
                DEVICE_PIDS+=("$pid")
                device_urls+="${device_urls:+,}localhost:${port}"
            done
            run_py --condition no_sharing --device-urls "$device_urls"
            ;;

        no_sharing_mps)
            # Each server gets an equal share of SMs via CUDA MPS thread percentage.
            # e.g. 3 tasks → each server gets CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=33
            ensure_mps_running
            local mps_pct=$(( 100 / ntasks ))
            echo "[run.sh] MPS: $ntasks servers, each gets ${mps_pct}% of SMs" >&2
            local device_urls=""
            for ((i=0; i<ntasks; i++)); do
                local task="${TASKS[$i]}"
                local port=$((DEVICE_PORT + i))
                local pid
                # One task per server → inline worker (no per-task overlap to gain).
                pid=$(start_device "$port" "fifo" \
                      "$LOG_DIR/device_ns_mps_${i}_rps${rps}_n${ntasks}.log" \
                      "$rps" "${task}:${rps}" "none" "" "$mps_pct" "inline")
                DEVICE_PIDS+=("$pid")
                device_urls+="${device_urls:+,}localhost:${port}"
            done
            run_py --condition no_sharing_mps --device-urls "$device_urls"
            ;;

        sharing)
            # All tasks on one server
            local task_rates=""
            for t in "${TASKS[@]}"; do
                task_rates+="${task_rates:+,}${t}:${rps}"
            done
            local port="$DEVICE_PORT"
            local pid
            pid=$(start_device "$port" "stfq" \
                  "$LOG_DIR/device_sharing_rps${rps}_n${ntasks}.log" \
                  "$rps" "$task_rates")
            DEVICE_PIDS+=("$pid")
            run_py --condition sharing --device-urls "localhost:${port}"
            ;;
    esac

    local run_tasks_json
    run_tasks_json=$(printf '%s\n' "$run_tasks_csv" | awk -F, '{
        for (i = 1; i <= NF; i++) {
            printf "%s\"%s\"", (i > 1 ? "," : ""), $i
        }
    }')

    cat > "${out_dir}/run_config.json" <<EOF
{
  "condition": "${condition}",
  "task_set": "${TASK_SET}",
  "tasks": [${run_tasks_json}],
  "num_tasks": ${ntasks},
  "backbone": "${BACKBONE}",
  "cuda_device": "${CUDA_DEVICE}",
  "max_batch_size": ${MAX_BATCH_SIZE},
  "max_batch_wait_ms": ${MAX_BATCH_WAIT_MS},
  "phase_duration_s": ${PHASE_DURATION},
  "rps_per_task": ${rps},
  "tpc_mode": "${TPC_MODE}",
  "device_port_base": ${DEVICE_PORT},
  "device_startup_wait_s": ${DEVICE_STARTUP_WAIT}
}
EOF

    # For single conditions: symlink from the ntasks dir so plot.py finds results
    if [[ "$condition" == single_* ]]; then
        mkdir -p "$ntasks_dir"
        ln -sfn "$out_dir" "${ntasks_dir}/${condition}" 2>/dev/null || true
        echo "[run.sh] Linked: ${ntasks_dir}/${condition} -> $out_dir"
    fi

    stop_devices
}

# ---------------------------------------------------------------------------
# TPC count sweep: run a single task with a fixed number of TPCs
#   Condition name: tpc{N}_{task}  (e.g. tpc1_ecgclass, tpc2_ecgclass)
#   Results go to: RESULTS_BASE/tpc_sweep/rps_{rps}/tpc{N}_{task}/
# ---------------------------------------------------------------------------

run_tpc_count() {
    local task="$1" rps="$2" n_tpcs="$3"
    local condition="tpc${n_tpcs}_${task}"
    local out_dir="${abs_results}/tpc_sweep/rps_${rps}/${condition}"
    local trace_file="${abs_results}/tpc_sweep/rps_${rps}/trace.json"

    echo ""
    echo "================================================================"
    echo "  TPC count sweep: task=$task  rps=$rps  n_tpcs=$n_tpcs"
    echo "  Results: $out_dir"
    echo "================================================================"

    if [[ -f "${out_dir}/latencies.csv" ]]; then
        echo "[run.sh] Skipping $condition — results already exist"
        return 0
    fi

    # Query total TPCs and pick first n_tpcs
    local total_tpcs
    total_tpcs=$($PYTHON -c "
import torch
sm = torch.cuda.get_device_properties('${CUDA_DEVICE}').multi_processor_count
print(sm // 2)
")
    if [[ $n_tpcs -gt $total_tpcs ]]; then
        echo "[run.sh] WARNING: n_tpcs=$n_tpcs > total_tpcs=$total_tpcs, skipping"
        return 0
    fi
    local part
    part=$(seq -s ' ' 0 $((n_tpcs - 1)))
    echo "[run.sh] Pinning $task to TPCs [${part}] (of $total_tpcs)"

    stop_devices
    DEVICE_PIDS=()
    ACTIVE_PORTS=()
    mkdir -p "$out_dir"

    local pid
    # Single task per server → inline worker (no per-task overlap to gain).
    pid=$(start_device "$DEVICE_PORT" "fifo" \
          "$LOG_DIR/device_${condition}_rps${rps}.log" \
          "$rps" "${task}:${rps}" "$TPC_MODE" "$part" "" "inline")
    DEVICE_PIDS+=("$pid")

    $PYTHON -u experiments/sharing_benefit/tpc/run.py \
        --task-set          "$TASK_SET" \
        --tasks             "$task" \
        --backbone          "$BACKBONE" \
        --rps               "$rps" \
        --duration          "$PHASE_DURATION" \
        --exp-dir           "$out_dir" \
        --trace-file        "$trace_file" \
        --warmup-burst-secs "$WARMUP_BURST_SECS" \
        --condition         "single_${task}" \
        --device-urls       "localhost:${DEVICE_PORT}"

    cat > "${out_dir}/run_config.json" <<EOF
{
  "condition": "${condition}",
  "task": "${task}",
  "n_tpcs": ${n_tpcs},
  "tpc_partition": "${part}",
  "backbone": "${BACKBONE}",
  "cuda_device": "${CUDA_DEVICE}",
  "rps_per_task": ${rps},
  "phase_duration_s": ${PHASE_DURATION},
  "tpc_mode": "${TPC_MODE}"
}
EOF
    stop_devices
}

# ---------------------------------------------------------------------------
# Main sweep: NUM_TASKS_SWEEP x RPS_SWEEP
# For vision, NUM_TASKS_SWEEP selects first N from ALL_VISION_TASKS (default: all 2).
# ---------------------------------------------------------------------------

# Resolve absolute results path once (used by run_condition and run_tpc_count)
abs_results="$(cd "$SERVING_DIR" && realpath "$RESULTS_BASE")"

# Kill any leftover MPS daemon from a previous run before the sweep begins.
# stop_devices() handles this between conditions; this handles the initial state.
if [[ -S /tmp/nvidia-mps/control ]]; then
    echo "[run.sh] Shutting down pre-existing MPS daemon before sweep..." >&2
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    sleep 1
fi

IFS=',' read -ra RPS_LIST <<< "$RPS_SWEEP"

if [[ "$TASK_SET" == "vision" ]]; then
    # Vision: NUM_TASKS_SWEEP selects first N from ALL_VISION_TASKS (default: all 2)
    if [[ -z "$NUM_TASKS_SWEEP" ]]; then
        NUM_TASKS_SWEEP="${#ALL_VISION_TASKS[@]}"
    fi
    IFS=',' read -ra NTASKS_LIST <<< "$NUM_TASKS_SWEEP"
    for ntasks in "${NTASKS_LIST[@]:-}"; do
        [[ -z "$ntasks" ]] && continue
        if [[ $ntasks -lt 1 || $ntasks -gt ${#ALL_VISION_TASKS[@]} ]]; then
            echo "[run.sh] ERROR: NUM_TASKS=$ntasks out of range (1..${#ALL_VISION_TASKS[@]})"
            exit 1
        fi

        tasks_arr=("${ALL_VISION_TASKS[@]:0:$ntasks}")
        tasks_csv=$(IFS=','; echo "${tasks_arr[*]}")

        conditions=()
        for t in "${tasks_arr[@]}"; do
            conditions+=("single_${t}")
        done
        # conditions+=("no_sharing" "no_sharing_tpc" "no_sharing_mps" "sharing")
        conditions+=("no_sharing" "no_sharing_tpc" "sharing")
        for rps in "${RPS_LIST[@]}"; do
            echo ""
            echo "################################################################"
            echo "  TASK_SET=vision  ntasks=$ntasks  RPS=$rps"
            echo "  tasks: $tasks_csv"
            echo "################################################################"
            for condition in "${conditions[@]}"; do
                run_condition "$condition" "$rps" "$ntasks" "$tasks_csv" \
                    || echo "[run.sh] WARNING: $condition ntasks=$ntasks rps=$rps failed — continuing"
            done
        done
    done
else
    # tsfm: sweep both NUM_TASKS and RPS (only if NUM_TASKS_SWEEP is explicitly set)
    if [[ -z "$NUM_TASKS_SWEEP" ]]; then
        echo "[run.sh] NUM_TASKS_SWEEP is empty — skipping ntasks sweep"
        echo "[run.sh] (tip: set NUM_TASKS_SWEEP=2,3 or similar to enable)"
    fi
    IFS=',' read -ra NTASKS_LIST <<< "$NUM_TASKS_SWEEP"
    for ntasks in "${NTASKS_LIST[@]:-}"; do
        [[ -z "$ntasks" ]] && continue
        # Validate (only lower bound — ntasks can exceed pool size, tasks cycle)
        if [[ $ntasks -lt 1 ]]; then
            echo "[run.sh] ERROR: NUM_TASKS=$ntasks must be >= 1"
            exit 1
        fi

        # Build task list: cycle through ALL_TSFM_TASKS using modulo
        pool_size=${#ALL_TSFM_TASKS[@]}
        tasks_arr=()
        for ((i=0; i<ntasks; i++)); do
            tasks_arr+=("${ALL_TSFM_TASKS[$((i % pool_size))]}")
        done
        tasks_csv=$(IFS=','; echo "${tasks_arr[*]}")

        # Build single conditions for unique tasks only (no point re-running the same task)
        conditions=()
        declare -A _seen_tasks=()
        for t in "${tasks_arr[@]}"; do
            if [[ -z "${_seen_tasks[$t]+x}" ]]; then
                conditions+=("single_${t}")
                _seen_tasks[$t]=1
            fi
        done
        unset _seen_tasks
        # conditions+=("no_sharing" "no_sharing_tpc" "no_sharing_mps" "sharing")
        conditions+=("no_sharing" "no_sharing_tpc" "sharing")
        for rps in "${RPS_LIST[@]}"; do
            echo ""
            echo "################################################################"
            echo "  TASK_SET=tsfm  ntasks=$ntasks  RPS=$rps"
            echo "  tasks: $tasks_csv"
            echo "################################################################"
            for condition in "${conditions[@]}"; do
                run_condition "$condition" "$rps" "$ntasks" "$tasks_csv" \
                    || echo "[run.sh] WARNING: $condition ntasks=$ntasks rps=$rps failed — continuing"
            done
        done
    done
fi

# ---------------------------------------------------------------------------
# TPC count sweep (optional): TPC_COUNT_SWEEP x TPC_TASK x RPS_SWEEP
# e.g. TPC_COUNT_SWEEP=1,2,3,4,5 TPC_TASK=ecgclass RPS_SWEEP=5
# ---------------------------------------------------------------------------
if [[ -n "${TPC_COUNT_SWEEP:-}" ]]; then
    TPC_TASK="${TPC_TASK:-${ALL_TSFM_TASKS[0]}}"
    IFS=',' read -ra TPC_COUNT_LIST <<< "$TPC_COUNT_SWEEP"
    echo ""
    echo "################################################################"
    echo "  TPC count sweep: task=$TPC_TASK  counts=$TPC_COUNT_SWEEP  rps=$RPS_SWEEP"
    echo "################################################################"
    for rps in "${RPS_LIST[@]}"; do
        for n_tpcs in "${TPC_COUNT_LIST[@]}"; do
            run_tpc_count "$TPC_TASK" "$rps" "$n_tpcs" \
                || echo "[run.sh] WARNING: tpc${n_tpcs}_${TPC_TASK} rps=$rps failed — continuing"
        done
    done
fi

echo ""
echo "[run.sh] All done. Results in $RESULTS_BASE"
