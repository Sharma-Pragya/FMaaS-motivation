#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  FMaaS System-in-Action Experiment
#
#  Setup: 3 GPUs, 1 initial task (ecgclass @ REQ_RATE req/s)
#  No runtime events — shows the system deploying and serving a
#  5-minute trace end-to-end.
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
SCHEDULERS="${SCHEDULERS:-fmaas_share}"

# ── Shared configuration ─────────────────────────────────────────────
REQ_RATE="${REQ_RATE:-150}"
TRACE="${TRACE:-deterministic}"
DURATION="${DURATION:-360}"
SEED="${SEED:-42}"
EXP_DIR="${EXP_DIR:-experiments/SystemInAction/results}"
EXP_TYPE="${EXP_TYPE:-SystemInAction}"
ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-8080}"

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVING_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ORCHESTRATOR_URL="http://localhost:$ORCHESTRATOR_PORT"

# ── Colors ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}   $*"; }
error()   { echo -e "${RED}[ERROR]${NC}  $*"; }
section() { echo -e "${CYAN}[RUN]${NC}    $*"; }

# ── Per-scheduler run function ────────────────────────────────────────
run_scheduler() {
    local SCHEDULER="$1"
    local SITE_LOG="$SERVING_DIR/site_manager.log"
    local ORCHESTRATOR_LOG="$SERVING_DIR/orchestrator.log"
    local SITE_PID="" ORCHESTRATOR_PID=""

    cleanup_scheduler() {
        info "Cleaning up [$SCHEDULER]..."
        if [[ -n "$ORCHESTRATOR_PID" ]] && kill -0 "$ORCHESTRATOR_PID" 2>/dev/null; then
            curl -s -X POST "$ORCHESTRATOR_URL/cleanup" > /dev/null 2>&1 || true
            sleep 2
            kill "$ORCHESTRATOR_PID" 2>/dev/null || true
            wait "$ORCHESTRATOR_PID" 2>/dev/null || true
        fi
        if [[ -n "$SITE_PID" ]] && kill -0 "$SITE_PID" 2>/dev/null; then
            kill "$SITE_PID" 2>/dev/null || true
            wait "$SITE_PID" 2>/dev/null || true
        fi
    }
    trap cleanup_scheduler EXIT INT TERM

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    section "SCHEDULER: $SCHEDULER"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    info "Setup:    3 GPUs, ecgclass @ ${REQ_RATE} req/s, ${DURATION}s trace"
    info "Timeline:"
    info "  t=0s    Deploy ecgclass @ ${REQ_RATE} req/s and run for ${DURATION}s"
    echo ""

    # ── Kill any leftover processes from previous runs ────────────────
    pkill -f "site_manager.main" 2>/dev/null || true
    pkill -f "orchestrator.server" 2>/dev/null || true
    sleep 1

    # ── Start orchestrator ────────────────────────────────────────────
    info "Starting orchestrator..."
    python -u -m orchestrator.server --port "$ORCHESTRATOR_PORT" \
        > "$ORCHESTRATOR_LOG" 2>&1 &
    ORCHESTRATOR_PID=$!

    local elapsed=0
    while ! curl -s "$ORCHESTRATOR_URL/" > /dev/null 2>&1; do
        if ! kill -0 "$ORCHESTRATOR_PID" 2>/dev/null; then
            error "Orchestrator exited unexpectedly:"; tail -20 "$ORCHESTRATOR_LOG"; return 1
        fi
        if (( elapsed >= 30 )); then
            error "Timed out waiting for orchestrator."; return 1
        fi
        sleep 1; (( elapsed += 1 ))
    done
    info "Orchestrator ready (PID=$ORCHESTRATOR_PID)."

    # ── Start site manager ────────────────────────────────────────────
    info "Starting site_manager..."
    python -u -m site_manager.main > "$SITE_LOG" 2>&1 &
    SITE_PID=$!

    elapsed=0
    while ! grep -q "ready. Entering MQTT loop" "$SITE_LOG" 2>/dev/null; do
        if ! kill -0 "$SITE_PID" 2>/dev/null; then
            error "site_manager exited unexpectedly:"; tail -20 "$SITE_LOG"; return 1
        fi
        if (( elapsed >= 300 )); then
            error "Timed out waiting for site_manager."; return 1
        fi
        sleep 2; (( elapsed += 2 ))
    done
    info "Site manager ready (PID=$SITE_PID)."

    # ── Initial deployment ────────────────────────────────────────────
    info "Deploying initial task (ecgclass)..."
    local deploy_resp
    deploy_resp=$(curl -s -X POST "$ORCHESTRATOR_URL/deploy" \
        -H "Content-Type: application/json" \
        -d "{\"exp_type\":\"$EXP_TYPE\",\"trace\":\"$TRACE\",\"req_rate\":$REQ_RATE,\"duration\":$DURATION,\"seed\":$SEED,\"scheduler\":\"$SCHEDULER\",\"exp_dir\":\"$EXP_DIR\"}")
    if echo "$deploy_resp" | grep -q '"detail"'; then
        error "Deployment failed: $deploy_resp"; return 1
    fi
    info "Initial deployment complete."

    # ── Start inference ───────────────────────────────────────────────
    local run_resp
    run_resp=$(curl -s -X POST "$ORCHESTRATOR_URL/run")
    if echo "$run_resp" | grep -q '"detail"'; then
        error "Failed to start inference: $run_resp"; return 1
    fi
    info "Inference started at $(date '+%H:%M:%S') — running for ${DURATION}s."

    # ── Wait for inference to complete ────────────────────────────────
    info "Waiting for inference to complete..."
    local wait_resp
    wait_resp=$(curl -s -X POST "$ORCHESTRATOR_URL/wait")
    if echo "$wait_resp" | grep -q "completed"; then
        info "Inference completed."
    else
        warn "Wait response: $wait_resp"
    fi

    # ── Cleanup ───────────────────────────────────────────────────────
    info "Cleaning up..."
    curl -s -X POST "$ORCHESTRATOR_URL/cleanup" > /dev/null 2>&1 || true
    sleep 10

    kill "$SITE_PID" 2>/dev/null || true
    wait "$SITE_PID" 2>/dev/null || true
    kill "$ORCHESTRATOR_PID" 2>/dev/null || true
    wait "$ORCHESTRATOR_PID" 2>/dev/null || true

    trap - EXIT INT TERM

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
echo "  FMaaS System-in-Action Experiment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "Schedulers: $SCHEDULERS"
info "Duration:   ${DURATION}s  |  Rate: ${REQ_RATE} req/s  |  Trace: $TRACE"
info "Exp dir:    $EXP_DIR"

for SCHEDULER in $SCHEDULERS; do
    run_scheduler "$SCHEDULER"
    info "Pausing 15s before next scheduler run..."
    sleep 15
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "All schedulers complete."
info "Results in: $EXP_DIR/<scheduler>/$REQ_RATE/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
