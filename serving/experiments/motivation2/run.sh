#!/usr/bin/env bash
# motivation2 — the motivation figure (memory_isolation_vision/motivation_plot.py)
# reproduced on the device server (device/main.py), as sharing_benefit/vision:
# this script starts and stops the servers, run.py deploys and sends the load,
# plot.py draws the figure. Memory is the device server's own measurement
# (gpu_alloc_peak_mb in every response); see run.py.
#
# Every arm stages inputs on the GPU (STAGING=gpu) and, except fmvisor, runs
# the task heads inline (WORKER_MODE=inline: one after another on the backbone
# thread, as Nexus / Mainstream run per-model suffixes). Per arm:
#
#   arm          servers           --isolation-mode  policy  staging  worker mode
#   shared       1                 shared            fifo    gpu      inline
#   shared_bfq   1                 shared            stfq    gpu      inline
#   fmvisor      1                 process           stfq    gpu slots task processes
#   no_sharing   1 per task        shared            fifo    gpu      inline
#   mig          1 per MIG slice   shared            fifo    gpu      inline
#   partitioned  1 per task        shared            fifo    gpu      inline   (efficiency)
#   mig, mig_tpc 1 per task        shared            fifo    gpu      inline   (efficiency)
#
#   isolation   (b, c, d): noisy task B ramps RATE_B -> RATE_B_MAX req/s at
#               RAMP_AT s; victims A1.. stay at RATE_A. Conditions:
#                 shared      one server, every task, --staging gpu, FIFO,
#                             cross-task batches (S-BE; Nexus / Mainstream style)
#                 shared_bfq  shared, with the STFQ (BFQ-style) scheduler, equal
#                             weights: performance isolation only
#                 fmvisor     one server in process-isolation mode (a backbone
#                             process + one process per task), STFQ: each task
#                             gets FM_SLOTS GPU slots (or as many as fit in
#                             FM_MEM_MB), then waits on the CPU
#                             (at most FM_BACKLOG_CAP), then is rejected
#                 mig         one server per MIG slice (needs N_TASKS slices)
#                 no_sharing  one server per task on the whole GPU (no MIG)
#   efficiency  (a): N in EFF_NS tasks, closed loop; shared (one server) vs
#               partitioned (one server per task, whole GPU) vs mig (one
#               server per task, task i on MIG slice i mod #slices) vs mig_tpc
#               (mig, and the k servers on a slice each pinned to 1/k of its
#               TPCs: --tpc-mode libsmctrl --tpc-share j/k).
#
# Run from serving/:
#   bash experiments/motivation2/run.sh
#   python experiments/motivation2/plot.py --exp-dir experiments/motivation2/results
#
# Overrides (defaults follow memory_isolation_vision/results_bw_n2_a110):
#   EXPERIMENTS=isolation,efficiency ISO_CONDITIONS=shared,mig N_TASKS=2
#   BACKBONE=dinobase-patch DECODER_PATH=nyudepth_dinobase_monocular RES=224 MAX_BATCH=32 STAGING=gpu SCHEDULER=fifo
#   WORKER_MODE=inline (shared-mode arms; threaded = per-task head threads, an ablation)
#   RATE_A=110 RATE_B=20 RATE_B_MAX=1000 RAMP_AT=60 RAMP_S=15 DURATION=180 WARMUP_BURST_SECS=15
#   EFF_CONDITIONS=shared,partitioned EFF_NS=1,2,4,6,8 EFF_DURATION=60 EFF_CONC=1
#   (efficiency as motivation1: closed loop, one request in flight per task)
#   CUDA_DEVICE=cuda:0 MIG_UUIDS=MIG-...,MIG-... EXP_DIR=experiments/motivation2/results
#
# MIG is handled here (MIG_MANAGE=1, the default; runs nvidia-smi with sudo
# unless the script is root):
# every other run goes first with MIG off (turned off if it is on), then MIG
# is turned on with N_TASKS slices of MIG_PROFILE, the mig runs go, and MIG
# is turned off again (also if the script stops early). MIG_PROFILE: one from
# `sudo nvidia-smi mig -i 0 -lgip` that allows N_TASKS instances (RTX PRO 4500
# Blackwell: 1g.16gb, 2 of them; RTX PRO 6000: 1g.24gb, 4). MIG_MANAGE=0:
# create / remove the slices yourself, as before.

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
WORKER_MODE="${WORKER_MODE:-inline}"       # shared-mode arms: task heads inline | threaded
FM_SLOTS="${FM_SLOTS:-64}"                 # fmvisor: GPU slots per task (2 x MAX_BATCH)
FM_MEM_MB="${FM_MEM_MB:-}"                 # fmvisor: per-task memory budget in MB ("MB" or
                                           # "task=MB,...,*=MB"); set, it replaces FM_SLOTS
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
MIG_MANAGE="${MIG_MANAGE:-1}"
MIG_PROFILE="${MIG_PROFILE:-1g.16gb}"
MIG_WAIT="${MIG_WAIT:-120}"               # s to wait for the GPU to be free before creating slices
EXP_DIR="${EXP_DIR:-experiments/motivation2/results}"
LOG_DIR="$EXP_DIR/logs"
mkdir -p "$LOG_DIR"

# Tasks: B (noisy) then A1..A{N-1} (victims).
ISO_TASKS=(B)
for ((i = 1; i < N_TASKS; i++)); do ISO_TASKS+=("A$i"); done

echo "================================================================"
echo "  motivation2 (device server)"
echo "  experiments=$EXPERIMENTS  backbone=$BACKBONE  res=$RES  max_batch=$MAX_BATCH"
echo "  isolation: $ISO_CONDITIONS  tasks=${ISO_TASKS[*]}  staging=$STAGING  worker_mode=$WORKER_MODE (fmvisor: task processes)"
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
MIG_TURNED_ON=0
on_exit() {
    stop_servers
    if [[ "$MIG_TURNED_ON" == 1 ]]; then mig_off || true; fi
}
trap on_exit EXIT

# start_server PORT LOG [CUDA_VISIBLE_DEVICES] [SCHEDULER] [ISOLATION_MODE] [TPC_SHARE i/k]
start_server() {
    local port="$1" log="$2" visible="${3:-}" sched="${4:-$SCHEDULER}" mode="${5:-shared}"
    local tpc_share="${6:-}" cuda="$CUDA_DEVICE" extra=()
    if [[ "$mode" == process ]]; then
        if [[ -n "$FM_MEM_MB" ]]; then extra=(--task-mem-mb "$FM_MEM_MB")
        else extra=(--slots-per-task "$FM_SLOTS"); fi
        extra+=(--task-backlog-cap "$FM_BACKLOG_CAP")
    fi
    [[ -n "$tpc_share" ]] && extra+=(--tpc-mode libsmctrl --tpc-share "$tpc_share")
    pkill -f "device/main.py.*--port ${port}( |$)" 2>/dev/null || true
    local envs=()
    if [[ -n "$visible" ]]; then envs=(env "CUDA_VISIBLE_DEVICES=$visible"); cuda="cuda:0"; fi
    ${envs[@]+"${envs[@]}"} $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port "$port" --runtime-type pytorch --cuda "$cuda" \
        --isolation-mode "$mode" --staging "$STAGING" --worker-mode "$WORKER_MODE" \
        ${extra[@]+"${extra[@]}"} \
        --scheduler-policy "$sched" \
        --max-batch-size "$MAX_BATCH" --max-batch-wait-ms "$MAX_BATCH_WAIT_MS" \
        --queue-capacity "$QUEUE_CAPACITY" \
        > "$log" 2>&1 &
    PIDS+=("$!"); PORTS+=("$port"); LOGS+=("$log")
    echo "[run.sh] server port=$port pid=$! ${visible:+CUDA_VISIBLE_DEVICES=$visible }${tpc_share:+tpc=$tpc_share }log=$log"
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

mig_enabled() {
    nvidia-smi -i "$GPU_INDEX" --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null \
        | grep -qi enabled
}

have_slices() {         # at least N_TASKS MIG slices exist already
    (( $(nvidia-smi -L 2>/dev/null | grep -c 'MIG-' || true) >= N_TASKS ))
}

# nvidia-smi as root for MIG changes: directly if this script is root, else sudo.
SUDO=""; [[ "$(id -u)" == 0 ]] || SUDO=sudo

mig_on() {
    local profiles="" i n
    for ((i = 0; i < N_TASKS; i++)); do profiles+="${profiles:+,}$MIG_PROFILE"; done
    echo "[run.sh] MIG on: GPU $GPU_INDEX, $N_TASKS x $MIG_PROFILE"
    $SUDO nvidia-smi -i "$GPU_INDEX" -mig 1
    if ! mig_enabled; then
        echo "[run.sh] MIG mode is pending (a GPU reset or reboot may be needed); not running mig"
        return 1
    fi
    MIG_TURNED_ON=1
    # Slices can only be created while no process has the GPU open ("In use
    # by another client"): wait for the earlier runs' processes to go, then
    # retry a few times.
    local deadline=$((SECONDS + MIG_WAIT)) apps try
    while :; do
        apps=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name \
               --format=csv,noheader 2>/dev/null || true)
        # With MIG mode on, --query-compute-apps misses processes that still
        # hold the device open (e.g. an orphaned server from a killed run), so
        # also look for this user's processes on /dev/nvidiaN.
        if [[ -z "$apps" ]]; then
            local pids
            pids=$(fuser "/dev/nvidia$GPU_INDEX" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' \
                   | while read -r p; do [[ "$(stat -c %U /proc/$p 2>/dev/null)" == "$USER" ]] && echo "$p"; done || true)
            [[ -n "$pids" ]] && apps=$(ps -o pid=,cmd= -p "$(paste -sd, <<< "$pids")" 2>/dev/null || true)
        fi
        [[ -z "$apps" ]] && break
        if (( SECONDS > deadline )); then
            echo "[run.sh] GPU still in use after ${MIG_WAIT}s by:"; echo "$apps"; break
        fi
        sleep 2
    done
    for try in 1 2 3 4 5; do
        $SUDO nvidia-smi mig -i "$GPU_INDEX" -cgi "$profiles" -C && break
        echo "[run.sh] creating MIG slices failed (try $try/5); retrying in 10 s"
        nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true
        sleep 10
    done
    n=$(nvidia-smi -L | grep -c 'MIG-' || true)
    echo "[run.sh] $n MIG slice(s):"; nvidia-smi -L | grep 'MIG-' || true
    (( n >= N_TASKS )) || { echo "[run.sh] need $N_TASKS slices"; return 1; }
}

mig_off() {
    echo "[run.sh] MIG off: GPU $GPU_INDEX"
    $SUDO nvidia-smi mig -i "$GPU_INDEX" -dci 2>/dev/null || true
    $SUDO nvidia-smi mig -i "$GPU_INDEX" -dgi 2>/dev/null || true
    $SUDO nvidia-smi -i "$GPU_INDEX" -mig 0
    MIG_TURNED_ON=0
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
        --extra "{\"max_batch\": $MAX_BATCH, \"staging\": \"$STAGING\", \"scheduler\": \"$sched\", \"isolation_mode\": \"$([[ $cond == fmvisor ]] && echo process || echo shared)\", \"worker_mode\": \"$([[ $cond == fmvisor ]] && echo task_processes || echo "$WORKER_MODE")\", \"fm_slots\": $FM_SLOTS, \"fm_mem_mb\": \"$FM_MEM_MB\", \"fm_backlog_cap\": $FM_BACKLOG_CAP}" \
        --out-dir "$EXP_DIR/isolation" \
        || echo "[run.sh] WARNING: isolation/$cond failed"
    check_servers
    stop_servers
}

# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------
run_efficiency() {
    local cond="$1" n="$2" urls="" sep="" i port uuids=()
    stop_servers
    if [[ "$cond" == mig || "$cond" == mig_tpc ]]; then
        # One server per task, as partitioned, on the MIG slices: task i on
        # slice i mod (number of slices), so with more tasks than slices a
        # slice holds several tasks' servers (they share its SMs and memory).
        # mig_tpc: those servers also split the slice's TPCs, j-th of k
        # (libsmctrl stream masks on their backbone stream).
        read -ra uuids <<< "$(mig_uuids)"
        if (( ${#uuids[@]} == 0 )); then
            echo "[run.sh] efficiency/mig: no MIG slices; skipping"; return 1
        fi
    fi
    if [[ "$cond" == shared ]]; then
        start_server "$BASE_PORT" "$LOG_DIR/eff_${cond}_n${n}_p${BASE_PORT}.log"
        for ((i = 0; i < n; i++)); do urls+="${sep}T${i}=localhost:${BASE_PORT}"; sep=","; done
    else
        for ((i = 0; i < n; i++)); do
            port=$((BASE_PORT + i))
            if [[ "$cond" == mig || "$cond" == mig_tpc ]]; then
                local ns=${#uuids[@]} sl=$((i % ${#uuids[@]})) share=""
                if [[ "$cond" == mig_tpc ]]; then
                    local k=$(( (n - sl + ns - 1) / ns )) j=$(( i / ns ))
                    share="$j/$k"
                fi
                start_server "$port" "$LOG_DIR/eff_${cond}_n${n}_p${port}.log" \
                    "${uuids[$sl]}" "" shared "$share"
            else
                start_server "$port" "$LOG_DIR/eff_${cond}_n${n}_p${port}.log"
            fi
            urls+="${sep}T${i}=localhost:${port}"; sep=","
        done
    fi
    if ! wait_servers; then
        # A server that cannot even start (e.g. out of memory) is recorded as OOM.
        mkdir -p "$EXP_DIR/efficiency"
        local f="$EXP_DIR/efficiency/summary.csv"
        [[ -f "$f" ]] || echo "condition,n_tasks,n_servers,backbone,res,concurrency,duration,warmup,status,throughput_rps,avg_gpu_mem_mb,peak_gpu_mem_mb,avg_latency_ms,p99_latency_ms,fail,avg_device_used_mb,peak_device_used_mb" > "$f"
        echo "$cond,$n,${#PIDS[@]},$BACKBONE,$RES,$EFF_CONC,$EFF_DURATION,$WARMUP,start_failed,,,,,,,," >> "$f"
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

# Every mig condition (isolation and efficiency) runs last, with MIG on; the
# rest first, with MIG off.
IFS=',' read -ra ALL_ISO <<< "$ISO_CONDITIONS"
ISO_NOMIG=(); RUN_MIG=0
for c in "${ALL_ISO[@]}"; do [[ "$c" == mig ]] && RUN_MIG=1 || ISO_NOMIG+=("$c"); done
[[ ",$EXPERIMENTS," == *,isolation,* ]] || RUN_MIG=0
IFS=',' read -ra ALL_EFF <<< "$EFF_CONDITIONS"
EFF_NOMIG=(); RUN_MIG_EFF=0
EFF_MIG=()
for c in "${ALL_EFF[@]}"; do
    if [[ "$c" == mig || "$c" == mig_tpc ]]; then RUN_MIG_EFF=1; EFF_MIG+=("$c"); else EFF_NOMIG+=("$c"); fi
done
[[ ",$EXPERIMENTS," == *,efficiency,* ]] || RUN_MIG_EFF=0
IFS=',' read -ra NS <<< "$EFF_NS"

if [[ "$MIG_MANAGE" == 1 ]]; then
    NEED_MIG_OFF=0
    [[ ",$EXPERIMENTS," == *,isolation,* ]] && (( ${#ISO_NOMIG[@]} )) && NEED_MIG_OFF=1
    [[ ",$EXPERIMENTS," == *,efficiency,* ]] && (( ${#EFF_NOMIG[@]} )) && NEED_MIG_OFF=1
    if [[ "$NEED_MIG_OFF" == 1 ]] && mig_enabled; then
        echo "[run.sh] MIG is on; turning it off for the non-MIG runs"
        mig_off
    fi
fi

IFS=',' read -ra EXPS <<< "$EXPERIMENTS"
for exp in "${EXPS[@]}"; do
    case "$exp" in
        isolation)
            for c in "${ISO_NOMIG[@]+"${ISO_NOMIG[@]}"}"; do
                echo; echo "######## isolation / $c ########"
                run_isolation "$c" || true
            done ;;
        efficiency)
            for c in "${EFF_NOMIG[@]+"${EFF_NOMIG[@]}"}"; do
                for n in "${NS[@]}"; do
                    echo; echo "######## efficiency / $c / n=$n ########"
                    run_efficiency "$c" "$n" || true
                done
            done ;;
        *) echo "[run.sh] unknown experiment: $exp" ;;
    esac
done

if [[ "$RUN_MIG" == 1 || "$RUN_MIG_EFF" == 1 ]]; then
    if [[ "$MIG_MANAGE" != 1 ]] || have_slices || mig_on; then
        if [[ "$RUN_MIG" == 1 ]]; then
            echo; echo "######## isolation / mig ########"
            run_isolation mig || true
        fi
        if [[ "$RUN_MIG_EFF" == 1 ]]; then
            for c in "${EFF_MIG[@]}"; do
                for n in "${NS[@]}"; do
                    echo; echo "######## efficiency / $c / n=$n ########"
                    run_efficiency "$c" "$n" || true
                done
            done
        fi
    fi
    if [[ "$MIG_MANAGE" == 1 && "$MIG_TURNED_ON" == 1 ]]; then mig_off; fi
fi

echo
echo "[run.sh] done. Plot: $PYTHON experiments/motivation2/plot.py --exp-dir $EXP_DIR"
