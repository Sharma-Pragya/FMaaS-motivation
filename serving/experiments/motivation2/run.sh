#!/usr/bin/env bash
# motivation2 — the motivation figure (memory_isolation_vision/motivation_plot.py)
# reproduced on the device server (device/main.py), as sharing_benefit/vision:
# this script starts and stops the servers, run.py deploys and sends the load,
# plot.py draws the figure. Memory is the device server's own measurement
# (gpu_alloc_peak_mb in every response); see run.py.
#
#   isolation   (b, c, d): noisy task B ramps RATE_B -> RATE_B_MAX req/s at
#               RAMP_AT s; victims A1.. stay at RATE_A. Conditions:
#                 shared      one server, every task, --staging gpu, FIFO,
#                             cross-task batches (S-BE)
#                 shared_bfq  shared, with the STFQ (BFQ-style) scheduler, equal
#                             weights: performance isolation only
#                 fmvisor     one server in process-isolation mode (a backbone
#                             process + one process per task), STFQ: each task
#                             gets FM_SLOTS GPU slots, then waits on the CPU
#                             (at most FM_BACKLOG_CAP), then is rejected
#                 mig         one server per MIG slice (needs N_TASKS slices)
#                 no_sharing  one server per task on the whole GPU (no MIG)
#   efficiency  (a): N in EFF_NS tasks, closed loop; shared (one server) vs
#               partitioned (one server per task, whole GPU).
#
# Run from serving/:
#   bash experiments/motivation2/run.sh
#   python experiments/motivation2/plot.py --exp-dir experiments/motivation2/results
#
# Overrides (defaults follow memory_isolation_vision/results_bw_n2_a110):
#   EXPERIMENTS=isolation,efficiency ISO_CONDITIONS=shared,mig N_TASKS=2
#   BACKBONE=dinobase-patch DECODER_PATH=nyudepth_dinobase_monocular RES=224 MAX_BATCH=32 STAGING=gpu SCHEDULER=fifo
#   RATE_A=110 RATE_B=20 RATE_B_MAX=1000 RAMP_AT=60 RAMP_S=15 DURATION=180 WARMUP_BURST_SECS=15
#   EFF_CONDITIONS=shared,partitioned EFF_NS=1,2,4,6,8 EFF_DURATION=60 EFF_CONC=1
#   (efficiency as motivation1: closed loop, one request in flight per task)
#   CUDA_DEVICE=cuda:0 MIG_UUIDS=MIG-...,MIG-... EXP_DIR=experiments/motivation2/results
#
# MIG: create the slices first (root), e.g. RTX PRO 6000 Blackwell:
#   sudo nvidia-smi -i 0 -mig 1
#   sudo nvidia-smi mig -i 0 -cgi 1g.24gb,1g.24gb,1g.24gb,1g.24gb -C
# then ISO_CONDITIONS=mig; run shared / no_sharing / efficiency with MIG off,
# into the same EXP_DIR (the plot takes the latest run of each condition).

set -euo pipefail

SERVING_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SERVING_DIR"

FMTK_DIR="${FMTK_DIR:-../../FMTK}"
FMAAS_DIR="${FMAAS_DIR:-..}"
[[ "$FMTK_DIR" = /* ]] || FMTK_DIR="$(cd "$SERVING_DIR/$FMTK_DIR" && pwd)"
[[ "$FMAAS_DIR" = /* ]] || FMAAS_DIR="$(cd "$SERVING_DIR/$FMAAS_DIR" && pwd)"
export PYTHONPATH="${FMTK_DIR}/src:${FMAAS_DIR}:${PYTHONPATH:-}"

CONDA_ENV="${CONDA_ENV:-fmtk-bw}"
if [[ -z "${PYTHON:-}" ]]; then
    if command -v conda &> /dev/null; then
        PYTHON="conda run --no-capture-output -n ${CONDA_ENV} python"
    else
        PYTHON=python
    fi
fi

EXPERIMENTS="${EXPERIMENTS:-isolation,efficiency}"
ISO_CONDITIONS="${ISO_CONDITIONS:-shared,mig}"
N_TASKS="${N_TASKS:-2}"
BACKBONE="${BACKBONE:-dinobase-patch}"
RES="${RES:-224}"
MAX_BATCH="${MAX_BATCH:-32}"
MAX_BATCH_WAIT_MS="${MAX_BATCH_WAIT_MS:-0}"
QUEUE_CAPACITY="${QUEUE_CAPACITY:-10000000}"
STAGING="${STAGING:-gpu}"
SCHEDULER="${SCHEDULER:-fifo}"
FM_SLOTS="${FM_SLOTS:-64}"                 # fmvisor: GPU slots per task (2 x MAX_BATCH)
FM_BACKLOG_CAP="${FM_BACKLOG_CAP:-500}"    # fmvisor: requests per task waiting on the CPU
RATE_A="${RATE_A:-110}"
RATE_B="${RATE_B:-20}"
RATE_B_MAX="${RATE_B_MAX:-1000}"
RAMP_AT="${RAMP_AT:-60}"
RAMP_S="${RAMP_S:-15}"
DURATION="${DURATION:-180}"
WARMUP="${WARMUP:-10}"
WARMUP_BURST_SECS="${WARMUP_BURST_SECS:-15}"   # closed-loop GPU warm-up before each isolation run
EFF_CONDITIONS="${EFF_CONDITIONS:-shared,partitioned}"
EFF_NS="${EFF_NS:-1,2,4,6,8}"
EFF_DURATION="${EFF_DURATION:-60}"
EFF_CONC="${EFF_CONC:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
GPU_INDEX="${CUDA_DEVICE##*:}"; [[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || GPU_INDEX=0
BASE_PORT="${BASE_PORT:-8100}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
EXP_DIR="${EXP_DIR:-experiments/motivation2/results}"
LOG_DIR="$EXP_DIR/logs"
mkdir -p "$LOG_DIR"

# Tasks: B (noisy) then A1..A{N-1} (victims).
ISO_TASKS=(B)
for ((i = 1; i < N_TASKS; i++)); do ISO_TASKS+=("A$i"); done

echo "================================================================"
echo "  motivation2 (device server)"
echo "  experiments=$EXPERIMENTS  backbone=$BACKBONE  res=$RES  max_batch=$MAX_BATCH"
echo "  isolation: $ISO_CONDITIONS  tasks=${ISO_TASKS[*]}  staging=$STAGING"
echo "             B $RATE_B -> $RATE_B_MAX req/s at ${RAMP_AT}s (+${RAMP_S}s), A $RATE_A req/s, ${DURATION}s"
echo "  efficiency: $EFF_CONDITIONS  n=$EFF_NS  conc=$EFF_CONC  ${EFF_DURATION}s"
echo "  results: $EXP_DIR"
echo "================================================================"

# ---------------------------------------------------------------------------
# Servers
# ---------------------------------------------------------------------------
PIDS=()
PORTS=()
LOGS=()

stop_servers() {
    for pid in "${PIDS[@]:-}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done
    # Wait up to STOP_TIMEOUT s for each server, then kill it: a process-mode
    # server can log "Stopped" and still not exit.
    local deadline=$((SECONDS + STOP_TIMEOUT))
    for pid in "${PIDS[@]:-}"; do
        [[ -n "$pid" ]] || continue
        while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
        if kill -0 "$pid" 2>/dev/null; then
            echo "[run.sh] server pid=$pid did not exit after ${STOP_TIMEOUT}s; killing"
            pkill -9 -P "$pid" 2>/dev/null || true
            kill -9 "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    done
    for port in "${PORTS[@]:-}"; do
        [[ -n "$port" ]] && pkill -f "device/main.py.*--port ${port}( |$)" 2>/dev/null || true
    done
    PIDS=(); PORTS=(); LOGS=()
    sleep 2
}
trap stop_servers EXIT

# start_server PORT LOG [CUDA_VISIBLE_DEVICES] [SCHEDULER] [ISOLATION_MODE]
start_server() {
    local port="$1" log="$2" visible="${3:-}" sched="${4:-$SCHEDULER}" mode="${5:-shared}"
    local cuda="$CUDA_DEVICE" extra=()
    [[ "$mode" == process ]] && extra=(--slots-per-task "$FM_SLOTS" --task-backlog-cap "$FM_BACKLOG_CAP")
    pkill -f "device/main.py.*--port ${port}( |$)" 2>/dev/null || true
    local envs=()
    if [[ -n "$visible" ]]; then envs=(env "CUDA_VISIBLE_DEVICES=$visible"); cuda="cuda:0"; fi
    ${envs[@]+"${envs[@]}"} $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port "$port" --runtime-type pytorch --cuda "$cuda" \
        --isolation-mode "$mode" --staging "$STAGING" ${extra[@]+"${extra[@]}"} \
        --scheduler-policy "$sched" \
        --max-batch-size "$MAX_BATCH" --max-batch-wait-ms "$MAX_BATCH_WAIT_MS" \
        --queue-capacity "$QUEUE_CAPACITY" \
        > "$log" 2>&1 &
    PIDS+=("$!"); PORTS+=("$port"); LOGS+=("$log")
    echo "[run.sh] server port=$port pid=$! ${visible:+CUDA_VISIBLE_DEVICES=$visible }log=$log"
}

# Wait until every started server is listening (or one has exited).
wait_servers() {
    local deadline=$((SECONDS + STARTUP_TIMEOUT)) i
    for i in "${!PIDS[@]}"; do
        until grep -q "gRPC runtime listening" "${LOGS[$i]}" 2>/dev/null; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                echo "[run.sh] server on port ${PORTS[$i]} exited during startup"; return 1
            fi
            if (( SECONDS > deadline )); then
                echo "[run.sh] server on port ${PORTS[$i]} not up after ${STARTUP_TIMEOUT}s"; return 1
            fi
            sleep 1
        done
    done
}

# Report any server that is no longer running after a run.
check_servers() {
    local i
    for i in "${!PIDS[@]}"; do
        kill -0 "${PIDS[$i]}" 2>/dev/null \
            || echo "[run.sh] WARNING: server on port ${PORTS[$i]} exited during the run (see ${LOGS[$i]})"
    done
}

mig_uuids() {
    if [[ -n "${MIG_UUIDS:-}" ]]; then tr ',' ' ' <<< "$MIG_UUIDS"; return; fi
    nvidia-smi -L 2>/dev/null | grep -o 'MIG-[0-9a-fA-F-]*' | tr '\n' ' '
}

gpu_total_mb() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU_INDEX" 2>/dev/null \
        | head -1 | tr -d ' ' || true
}

# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------
run_isolation() {
    local cond="$1" urls="" sep="" i port mig_arg=() uuids=() sched="$SCHEDULER"
    stop_servers
    case "$cond" in
        shared|shared_bfq)
            # shared_bfq: STFQ, the device server's BFQ-style policy; no
            # --task-rates, so equal weights.
            [[ "$cond" == shared_bfq ]] && sched=stfq
            start_server "$BASE_PORT" "$LOG_DIR/iso_${cond}_p${BASE_PORT}.log" "" "$sched"
            for t in "${ISO_TASKS[@]}"; do urls+="${sep}${t}=localhost:${BASE_PORT}"; sep=","; done
            ;;
        fmvisor)
            sched=stfq
            start_server "$BASE_PORT" "$LOG_DIR/iso_${cond}_p${BASE_PORT}.log" "" "$sched" process
            for t in "${ISO_TASKS[@]}"; do urls+="${sep}${t}=localhost:${BASE_PORT}"; sep=","; done
            ;;
        no_sharing|mig)
            if [[ "$cond" == mig ]]; then
                read -ra uuids <<< "$(mig_uuids)"
                if (( ${#uuids[@]} < N_TASKS )); then
                    echo "[run.sh] mig: ${#uuids[@]} MIG slices found, need $N_TASKS; skipping"
                    return 1
                fi
                mig_arg=(--mig-uuid "${uuids[0]}")
            fi
            for i in "${!ISO_TASKS[@]}"; do
                port=$((BASE_PORT + i))
                start_server "$port" "$LOG_DIR/iso_${cond}_p${port}.log" "${uuids[$i]:-}"
                urls+="${sep}${ISO_TASKS[$i]}=localhost:${port}"; sep=","
            done
            ;;
        *) echo "[run.sh] unknown isolation condition: $cond"; return 1 ;;
    esac
    wait_servers || { stop_servers; return 1; }
    local total; total="$(gpu_total_mb)"
    $PYTHON -u experiments/motivation2/run.py --experiment isolation --condition "$cond" \
        --task-urls "$urls" --backbone "$BACKBONE" --res "$RES" \
        ${DECODER_PATH:+--decoder-path "$DECODER_PATH"} \
        --rate-a "$RATE_A" --rate-b "$RATE_B" --rate-b-max "$RATE_B_MAX" \
        --ramp-at "$RAMP_AT" --ramp-s "$RAMP_S" --duration "$DURATION" --warmup "$WARMUP" \
        --warmup-burst-secs "$WARMUP_BURST_SECS" --warmup-concurrency "$MAX_BATCH" \
        --gpu-index "$GPU_INDEX" ${total:+--gpu-total-mb "$total"} ${mig_arg[@]+"${mig_arg[@]}"} \
        --extra "{\"max_batch\": $MAX_BATCH, \"staging\": \"$STAGING\", \"scheduler\": \"$sched\", \"fm_slots\": $FM_SLOTS, \"fm_backlog_cap\": $FM_BACKLOG_CAP}" \
        --out-dir "$EXP_DIR/isolation" \
        || echo "[run.sh] WARNING: isolation/$cond failed"
    check_servers
    stop_servers
}

# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------
run_efficiency() {
    local cond="$1" n="$2" urls="" sep="" i port
    stop_servers
    if [[ "$cond" == shared ]]; then
        start_server "$BASE_PORT" "$LOG_DIR/eff_${cond}_n${n}_p${BASE_PORT}.log"
        for ((i = 0; i < n; i++)); do urls+="${sep}T${i}=localhost:${BASE_PORT}"; sep=","; done
    else
        for ((i = 0; i < n; i++)); do
            port=$((BASE_PORT + i))
            start_server "$port" "$LOG_DIR/eff_${cond}_n${n}_p${port}.log"
            urls+="${sep}T${i}=localhost:${port}"; sep=","
        done
    fi
    if ! wait_servers; then
        # A server that cannot even start (e.g. out of memory) is recorded as OOM.
        mkdir -p "$EXP_DIR/efficiency"
        local f="$EXP_DIR/efficiency/summary.csv"
        [[ -f "$f" ]] || echo "condition,n_tasks,n_servers,backbone,res,concurrency,duration,warmup,status,throughput_rps,avg_gpu_mem_mb,peak_gpu_mem_mb,avg_latency_ms,p99_latency_ms,fail" > "$f"
        echo "$cond,$n,${#PIDS[@]},$BACKBONE,$RES,$EFF_CONC,$EFF_DURATION,$WARMUP,start_failed,,,,,," >> "$f"
        stop_servers; return 1
    fi
    $PYTHON -u experiments/motivation2/run.py --experiment efficiency --condition "$cond" \
        --task-urls "$urls" --backbone "$BACKBONE" --res "$RES" \
        ${DECODER_PATH:+--decoder-path "$DECODER_PATH"} \
        --duration "$EFF_DURATION" --warmup "$WARMUP" --concurrency "$EFF_CONC" \
        --out-dir "$EXP_DIR/efficiency" \
        || echo "[run.sh] WARNING: efficiency/$cond n=$n failed"
    stop_servers
}

IFS=',' read -ra EXPS <<< "$EXPERIMENTS"
for exp in "${EXPS[@]}"; do
    case "$exp" in
        isolation)
            IFS=',' read -ra CONDS <<< "$ISO_CONDITIONS"
            for c in "${CONDS[@]}"; do
                echo; echo "######## isolation / $c ########"
                run_isolation "$c" || true
            done ;;
        efficiency)
            IFS=',' read -ra CONDS <<< "$EFF_CONDITIONS"
            IFS=',' read -ra NS <<< "$EFF_NS"
            for c in "${CONDS[@]}"; do
                for n in "${NS[@]}"; do
                    echo; echo "######## efficiency / $c / n=$n ########"
                    run_efficiency "$c" "$n" || true
                done
            done ;;
        *) echo "[run.sh] unknown experiment: $exp" ;;
    esac
done

echo
echo "[run.sh] done. Plot: $PYTHON experiments/motivation2/plot.py --exp-dir $EXP_DIR"
