#!/bin/bash
# Diagnostic: why does 1-TPC ecgclass show ~52ms p50 in RTVSntask but ~100ms p50
# in tpc_closed_loop_ecg? Both pin via libsmctrl on a torch stream.
#
# This script starts the device server with the RTVSntask launch flags
# (max_batch_size=100, task-rates=ecgclass:N, stfq or fifo), then drives it
# with the tpc_closed_loop_ecg client (closed loop, concurrency=1). It then
# repeats with the closed-loop launch flags (max_batch_size=1, no task-rates,
# fifo) for control. All with tpc_partition=[0] (1 TPC).
#
# Expected outcome: if RTVSntask-style server yields ~100ms under closed-loop
# load, then the discrepancy in the original data is purely a load-pattern
# effect. If it yields ~52ms, then one of the launch flags (max_batch_size,
# task-rates, scheduler) is bypassing TPC enforcement somehow.

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

export PYTHONPATH="${FMTK_DIR}/src:${FMAAS_DIR}:${PYTHONPATH:-}"

if command -v conda &> /dev/null; then
    PYTHON="${PYTHON:-conda run --no-capture-output -n ${CONDA_ENV} python}"
else
    PYTHON="${PYTHON:-python}"
fi

CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:-momentlarge}"
DEVICE_PORT="${DEVICE_PORT:-8000}"
DURATION="${DURATION:-60}"
WARMUP_SECS="${WARMUP_SECS:-5}"
TPC_MODE="${TPC_MODE:-libsmctrl}"
TPC_PARTITION="${TPC_PARTITION:-0}"
RESULTS_BASE="${RESULTS_BASE:-experiments/RTVSntask/vision/diagnose_tpc_results}"

LOG_DIR="${RESULTS_BASE}/logs"
mkdir -p "$LOG_DIR"

DEVICE_PID=""

stop_device() {
    if [[ -n "${DEVICE_PID:-}" ]]; then
        kill "$DEVICE_PID" 2>/dev/null || true
        wait "$DEVICE_PID" 2>/dev/null || true
    fi
    DEVICE_PID=""
    pkill -f "device/main.py.*--port ${DEVICE_PORT}" 2>/dev/null || true
    sleep 2
}
trap 'stop_device' EXIT

# ---------------------------------------------------------------------------
# A) RTVSntask-style server: max_batch_size=100, task-rates, scheduler=fifo
#    (mimics the no_sharing_tpc server in RTVSntask/tpc/run.sh)
# ---------------------------------------------------------------------------
run_case_rtvsntask_style() {
    local case_name="A_rtvsntask_style_bs100_taskrates"
    local out_dir="${RESULTS_BASE}/${case_name}"
    local log_file="${LOG_DIR}/device_${case_name}.log"
    mkdir -p "$out_dir"

    stop_device
    echo ""
    echo "================================================================"
    echo "  CASE A: RTVSntask-style launch flags"
    echo "    max_batch_size=100, task-rates=ecgclass:1, scheduler=fifo"
    echo "    tpc_partition=[${TPC_PARTITION}]"
    echo "================================================================"

    $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port              "$DEVICE_PORT" \
        --runtime-type      pytorch \
        --cuda              "$CUDA_DEVICE" \
        --scheduler-policy  fifo \
        --max-batch-size    100 \
        --max-batch-wait-ms 0 \
        --task-rates        "ecgclass:1" \
        --tpc-mode          "$TPC_MODE" \
        --tpc-partition     $TPC_PARTITION \
        > "$log_file" 2>&1 &
    DEVICE_PID=$!
    sleep 5

    $PYTHON -u experiments/tpc_closed_loop_ecg/run.py \
        --device-url "localhost:${DEVICE_PORT}" \
        --backbone "$BACKBONE" \
        --mode closed \
        --concurrency 1 \
        --target-rps 10 \
        --duration "$DURATION" \
        --warmup-secs "$WARMUP_SECS" \
        --tpc-count 1 \
        --exp-dir "$out_dir"

    stop_device
}

# ---------------------------------------------------------------------------
# B) Closed-loop-style server: max_batch_size=1, no task-rates, scheduler=fifo
#    (mimics the tpc_closed_loop_ecg/run.sh launch)
# ---------------------------------------------------------------------------
run_case_closedloop_style() {
    local case_name="B_closedloop_style_bs1_notaskrates"
    local out_dir="${RESULTS_BASE}/${case_name}"
    local log_file="${LOG_DIR}/device_${case_name}.log"
    mkdir -p "$out_dir"

    stop_device
    echo ""
    echo "================================================================"
    echo "  CASE B: closed-loop-style launch flags (control)"
    echo "    max_batch_size=1, no task-rates, scheduler=fifo"
    echo "    tpc_partition=[${TPC_PARTITION}]"
    echo "================================================================"

    $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port              "$DEVICE_PORT" \
        --runtime-type      pytorch \
        --cuda              "$CUDA_DEVICE" \
        --scheduler-policy  fifo \
        --max-batch-size    1 \
        --max-batch-wait-ms 0 \
        --tpc-mode          "$TPC_MODE" \
        --tpc-partition     $TPC_PARTITION \
        > "$log_file" 2>&1 &
    DEVICE_PID=$!
    sleep 5

    $PYTHON -u experiments/tpc_closed_loop_ecg/run.py \
        --device-url "localhost:${DEVICE_PORT}" \
        --backbone "$BACKBONE" \
        --mode closed \
        --concurrency 1 \
        --target-rps 10 \
        --duration "$DURATION" \
        --warmup-secs "$WARMUP_SECS" \
        --tpc-count 1 \
        --exp-dir "$out_dir"

    stop_device
}

# ---------------------------------------------------------------------------
# C) RTVSntask-style server + OPEN-LOOP rps=1 traffic (the exact RTVSntask regime)
# ---------------------------------------------------------------------------
run_case_rtvsntask_openloop() {
    local case_name="C_rtvsntask_style_openloop_rps1"
    local out_dir="${RESULTS_BASE}/${case_name}"
    local log_file="${LOG_DIR}/device_${case_name}.log"
    mkdir -p "$out_dir"

    stop_device
    echo ""
    echo "================================================================"
    echo "  CASE C: RTVSntask launch flags + open-loop rps=1 traffic"
    echo "    (replicates the exact RTVSntask no_sharing_tpc regime)"
    echo "================================================================"

    $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port              "$DEVICE_PORT" \
        --runtime-type      pytorch \
        --cuda              "$CUDA_DEVICE" \
        --scheduler-policy  fifo \
        --max-batch-size    100 \
        --max-batch-wait-ms 0 \
        --task-rates        "ecgclass:1" \
        --tpc-mode          "$TPC_MODE" \
        --tpc-partition     $TPC_PARTITION \
        > "$log_file" 2>&1 &
    DEVICE_PID=$!
    sleep 5

    $PYTHON -u experiments/tpc_closed_loop_ecg/run.py \
        --device-url "localhost:${DEVICE_PORT}" \
        --backbone "$BACKBONE" \
        --mode open \
        --concurrency 1 \
        --target-rps 1 \
        --duration "$DURATION" \
        --warmup-secs "$WARMUP_SECS" \
        --tpc-count 1 \
        --exp-dir "$out_dir"

    stop_device
}

# ---------------------------------------------------------------------------
# D) RTVSntask-style server + open-loop rps=1 + 5 TPCs (full GPU)
#    Compare against case C (1 TPC) to see if TPC count matters under rps=1.
# ---------------------------------------------------------------------------
run_case_rtvsntask_openloop_5tpc() {
    local case_name="D_rtvsntask_style_openloop_rps1_5tpc"
    local out_dir="${RESULTS_BASE}/${case_name}"
    local log_file="${LOG_DIR}/device_${case_name}.log"
    mkdir -p "$out_dir"

    stop_device
    echo ""
    echo "================================================================"
    echo "  CASE D: RTVSntask launch flags + open-loop rps=1 + 5 TPCs"
    echo "    tpc_partition=[0 1 2 3 4] (full GPU)"
    echo "================================================================"

    $PYTHON -u "$SERVING_DIR/device/main.py" \
        --port              "$DEVICE_PORT" \
        --runtime-type      pytorch \
        --cuda              "$CUDA_DEVICE" \
        --scheduler-policy  fifo \
        --max-batch-size    100 \
        --max-batch-wait-ms 0 \
        --task-rates        "ecgclass:1" \
        --tpc-mode          "$TPC_MODE" \
        --tpc-partition     0 1 2 3 4 \
        > "$log_file" 2>&1 &
    DEVICE_PID=$!
    sleep 5

    $PYTHON -u experiments/tpc_closed_loop_ecg/run.py \
        --device-url "localhost:${DEVICE_PORT}" \
        --backbone "$BACKBONE" \
        --mode open \
        --concurrency 1 \
        --target-rps 1 \
        --duration "$DURATION" \
        --warmup-secs "$WARMUP_SECS" \
        --tpc-count 5 \
        --exp-dir "$out_dir"

    stop_device
}



run_case_rtvsntask_style       || echo "[diagnose] CASE A failed"
run_case_closedloop_style      || echo "[diagnose] CASE B failed"
run_case_rtvsntask_openloop    || echo "[diagnose] CASE C failed"
run_case_rtvsntask_openloop_5tpc || echo "[diagnose] CASE D failed"


# ---------------------------------------------------------------------------
# Summary: print p50 server_exec_ms for each case
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  DIAGNOSTIC SUMMARY"
echo "================================================================"
$PYTHON - <<PYEOF
import pandas as pd
from pathlib import Path

base = Path("${RESULTS_BASE}")
cases = [
    ("A_rtvsntask_style_bs100_taskrates",     "RTVSntask-style server + closed-loop cc=1  [1 TPC]"),
    ("B_closedloop_style_bs1_notaskrates",    "closed-loop-style server + closed-loop cc=1  [1 TPC]"),
    ("C_rtvsntask_style_openloop_rps1",       "RTVSntask-style server + open-loop rps=1  [1 TPC]"),
    ("D_rtvsntask_style_openloop_rps1_5tpc",  "RTVSntask-style server + open-loop rps=1  [5 TPC]"),
]
for case, desc in cases:
    p = base / case / "latencies.csv"
    if not p.exists():
        print(f"{case}: MISSING")
        continue
    df = pd.read_csv(p)
    warm = df[df["send_elapsed_sec"] > ${WARMUP_SECS}] if "send_elapsed_sec" in df else df
    n = len(warm)
    p50 = warm["server_exec_ms"].quantile(0.5)
    p99 = warm["server_exec_ms"].quantile(0.99)
    mean = warm["server_exec_ms"].mean()
    print(f"{case}")
    print(f"  {desc}")
    print(f"  n={n}  p50_exec={p50:.2f}ms  mean_exec={mean:.2f}ms  p99_exec={p99:.2f}ms")
    print()

print("Reference numbers from existing data:")
print("  tpc_closed_loop_ecg/results/a2/momentlarge/tpc_1   p50 ~= 100.7ms")
print("  RTVSntask/.../ntasks_3/rps_1/no_sharing_tpc ecgclass  p50 ~= 52.7ms")
PYEOF
