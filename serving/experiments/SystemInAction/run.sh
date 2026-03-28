#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  FMaaS System-in-Action Experiment  (local mode — no MQTT)
#
#  Setup: 3 GPUs, 1 initial task (ecgclass @ REQ_RATE req/s)
#  No runtime events — shows the system deploying and serving a
#  trace end-to-end in a single process.
#
#  Usage:
#    cd serving
#    bash experiments/SystemInAction/run.sh
#
#    # Override scheduler or rate:
#    SCHEDULERS="fmaas_share" REQ_RATE=10 bash experiments/SystemInAction/run.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Schedulers to run ────────────────────────────────────────────────
SCHEDULERS="${SCHEDULERS:-fmaas_place clipper_place}"

# ── Shared configuration ─────────────────────────────────────────────
REQ_RATE="${REQ_RATE:-150}"
TRACE="${TRACE:-poisson_per_task}"
DURATION="${DURATION:-60}"
SEED="${SEED:-42}"
EXP_DIR="${EXP_DIR:-experiments/SystemInAction/results}"
EXP_TYPE="${EXP_TYPE:-SystemInAction}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-5}"
MAX_BATCH_WAIT_MS="${MAX_BATCH_WAIT_MS:-0}"
ISOLATION_MODE="${ISOLATION_MODE:-shared}"
WARMUP_GAP="${WARMUP_GAP:-2.0}"

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVING_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Colors ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}   $*"; }
error()   { echo -e "${RED}[ERROR]${NC}  $*"; }
section() { echo -e "${CYAN}[RUN]${NC}    $*"; }

# ── Per-scheduler run function ────────────────────────────────────────
run_scheduler() {
    local SCHEDULER="$1"
    local LOG="$SERVING_DIR/$EXP_DIR/$SCHEDULER/$REQ_RATE/orchestrator.log"
    local RUNNER_PID=""

    cleanup_scheduler() {
        if [[ -n "$RUNNER_PID" ]] && kill -0 "$RUNNER_PID" 2>/dev/null; then
            info "Interrupting local runner [$SCHEDULER]..."
            kill "$RUNNER_PID" 2>/dev/null || true
            wait "$RUNNER_PID" 2>/dev/null || true
        fi
    }
    trap cleanup_scheduler EXIT INT TERM

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    section "SCHEDULER: $SCHEDULER"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    info "Setup:    3 GPUs, ecgclass @ ${REQ_RATE} req/s, ${DURATION}s trace"
    info "Mode:     local (single process, no MQTT)"
    info "Timeline:"
    info "  t=0s    Deploy + run ecgclass @ ${REQ_RATE} req/s for ${DURATION}s"
    echo ""

    # ── Run experiment (single process) ──────────────────────────────
    info "Starting orchestrator (local mode)..."
    mkdir -p "$SERVING_DIR/$EXP_DIR/$SCHEDULER/$REQ_RATE"
    python -u -m orchestrator.server \
        --mode              local \
        --exp-type          "$EXP_TYPE" \
        --scheduler         "$SCHEDULER" \
        --req-rate          "$REQ_RATE" \
        --duration          "$DURATION" \
        --trace             "$TRACE" \
        --seed              "$SEED" \
        --exp-dir           "$EXP_DIR" \
        --max-batch-size    "$MAX_BATCH_SIZE" \
        --max-batch-wait-ms "$MAX_BATCH_WAIT_MS" \
        --isolation-mode    "$ISOLATION_MODE" \
        --warmup-gap        "$WARMUP_GAP" \
        2>&1 | tee "$LOG" &
    RUNNER_PID=$!

    # Wait for it to finish
    wait "$RUNNER_PID"
    local exit_code=$?
    RUNNER_PID=""
    trap - EXIT INT TERM

    if [[ $exit_code -ne 0 ]]; then
        error "Local runner failed (exit=$exit_code). See $LOG"
        return 1
    fi

    # ── Results summary ───────────────────────────────────────────────
    local result_dir="$EXP_DIR/$SCHEDULER/$REQ_RATE"
    if [[ -f "$result_dir/request_latency_results.csv" ]]; then
        local nrows
        nrows=$(wc -l < "$result_dir/request_latency_results.csv")
        info "Results: $((nrows - 1)) requests → $result_dir"
    else
        warn "Results CSV not found in $result_dir"
    fi

    info "[$SCHEDULER] done."
}

# ── Main: iterate schedulers ──────────────────────────────────────────
cd "$SERVING_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  FMaaS System-in-Action Experiment  (local mode)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "Schedulers: $SCHEDULERS"
info "Duration:   ${DURATION}s  |  Rate: ${REQ_RATE} req/s  |  Trace: $TRACE"
info "Exp dir:    $EXP_DIR"

for SCHEDULER in $SCHEDULERS; do
    run_scheduler "$SCHEDULER"
    if [[ "${SCHEDULERS}" == *" "* ]]; then
        info "Pausing 15s before next scheduler run..."
        sleep 15
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "All schedulers complete."
info "Results in: $EXP_DIR/<scheduler>/$REQ_RATE/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
