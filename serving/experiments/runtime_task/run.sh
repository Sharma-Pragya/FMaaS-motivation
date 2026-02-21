#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  Runtime Task Addition Experiment (Using Stateful Orchestrator Server)
#
#  This experiment demonstrates adding new tasks during runtime:
#  1. Start orchestrator server (stateful)
#  2. Start site_manager
#  3. Deploy initial set of tasks
#  4. Run inference in background
#  5. Add a new task (DURING inference)
#  6. Continue inference with the new task
#  7. Cleanup
#
#  Usage:
#    cd serving
#    bash experiments/runtime/run.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
SCHEDULER="${SCHEDULER:-fmaas_share}"
REQ_RATE="${REQ_RATE:-10}"
TRACE="${TRACE:-lmsyschat}"
DURATION="${DURATION:-60}"
SEED="${SEED:-42}"
EXP_DIR="${EXP_DIR:-experiments/runtime/results}"
EXP_TYPE="${EXP_TYPE:-runtime}"
ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-8080}"

# New task to add at runtime
NEW_TASK="${NEW_TASK:-gestureclass}"
NEW_TASK_TYPE="${NEW_TASK_TYPE:-classification}"
NEW_TASK_WORKLOAD="${NEW_TASK_WORKLOAD:-8.0}"
ADD_TASK_AFTER="${ADD_TASK_AFTER:-10}"  # seconds

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVING_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SITE_LOG="$SERVING_DIR/site_manager.log"
ORCHESTRATOR_LOG="$SERVING_DIR/orchestrator.log"
ORCHESTRATOR_URL="http://localhost:$ORCHESTRATOR_PORT"

# ── Colors ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Cleanup trap ─────────────────────────────────────────────────────
SITE_PID=""
ORCHESTRATOR_PID=""

cleanup() {
    info "Cleaning up processes..."

    # Cleanup via API if orchestrator is running
    if [[ -n "$ORCHESTRATOR_PID" ]] && kill -0 "$ORCHESTRATOR_PID" 2>/dev/null; then
        info "Sending cleanup request to orchestrator..."
        curl -s -X POST "$ORCHESTRATOR_URL/cleanup" || warn "Cleanup request failed"
        sleep 2
    fi

    # Kill site manager
    if [[ -n "$SITE_PID" ]] && kill -0 "$SITE_PID" 2>/dev/null; then
        warn "Killing site_manager (PID $SITE_PID)..."
        kill "$SITE_PID" 2>/dev/null || true
        wait "$SITE_PID" 2>/dev/null || true
    fi

    # Kill orchestrator server
    if [[ -n "$ORCHESTRATOR_PID" ]] && kill -0 "$ORCHESTRATOR_PID" 2>/dev/null; then
        warn "Killing orchestrator server (PID $ORCHESTRATOR_PID)..."
        kill "$ORCHESTRATOR_PID" 2>/dev/null || true
        wait "$ORCHESTRATOR_PID" 2>/dev/null || true
    fi

    info "Logs:"
    info "  - Site manager: $SITE_LOG"
    info "  - Orchestrator: $ORCHESTRATOR_LOG"
}
trap cleanup EXIT INT TERM

# ── Setup ────────────────────────────────────────────────────────────
cd "$SERVING_DIR"
info "Working directory: $(pwd)"
info "Scheduler: $SCHEDULER"
info "Request rate: $REQ_RATE"
info "Trace: $TRACE  |  Duration: ${DURATION}s"
info "Exp dir: $EXP_DIR"
info "New task: $NEW_TASK (type=$NEW_TASK_TYPE, workload=$NEW_TASK_WORKLOAD)"
info "Will add task after ${ADD_TASK_AFTER}s"
echo ""

# ── Step 1: Start Orchestrator Server ────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "PHASE 0: Starting Orchestrator Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

info "Starting orchestrator server on port $ORCHESTRATOR_PORT..."
python -u -m orchestrator.server --port "$ORCHESTRATOR_PORT" > "$ORCHESTRATOR_LOG" 2>&1 &
ORCHESTRATOR_PID=$!
info "Orchestrator server PID: $ORCHESTRATOR_PID  (log: $ORCHESTRATOR_LOG)"

# Wait for orchestrator to be ready
info "Waiting for orchestrator server to start..."
MAX_WAIT=30
ELAPSED=0
POLL=1

while ! curl -s "$ORCHESTRATOR_URL/" > /dev/null 2>&1; do
    if ! kill -0 "$ORCHESTRATOR_PID" 2>/dev/null; then
        error "Orchestrator server exited unexpectedly. Last 20 lines:"
        tail -20 "$ORCHESTRATOR_LOG"
        exit 1
    fi
    if (( ELAPSED >= MAX_WAIT )); then
        error "Timed out waiting for orchestrator server after ${MAX_WAIT}s."
        tail -20 "$ORCHESTRATOR_LOG"
        exit 1
    fi
    sleep "$POLL"
    (( ELAPSED += POLL ))
done
info "Orchestrator server ready after ~${ELAPSED}s."
echo ""

# ── Step 2: Start site_manager ───────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "PHASE 1: Starting Site Manager"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

info "Starting site_manager..."
python -u -m site_manager.main > "$SITE_LOG" 2>&1 &
SITE_PID=$!
info "Site manager PID: $SITE_PID  (log: $SITE_LOG)"

# Wait for site_manager to be ready
READY_SENTINEL="ready. Entering MQTT loop"
MAX_WAIT=300
ELAPSED=0
POLL=2

info "Waiting for site_manager to initialize..."
while ! grep -q "$READY_SENTINEL" "$SITE_LOG" 2>/dev/null; do
    if ! kill -0 "$SITE_PID" 2>/dev/null; then
        error "site_manager exited unexpectedly. Last 20 lines:"
        tail -20 "$SITE_LOG"
        exit 1
    fi
    if (( ELAPSED >= MAX_WAIT )); then
        error "Timed out waiting for site_manager after ${MAX_WAIT}s."
        tail -20 "$SITE_LOG"
        exit 1
    fi
    sleep "$POLL"
    (( ELAPSED += POLL ))
done
info "Site manager ready after ~${ELAPSED}s."
echo ""

# ── Step 3: Deploy Initial Tasks ─────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "PHASE 2: Initial Deployment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "Deploying initial tasks..."

DEPLOY_PAYLOAD=$(cat <<EOF
{
  "exp_type": "$EXP_TYPE",
  "trace": "$TRACE",
  "req_rate": $REQ_RATE,
  "duration": $DURATION,
  "seed": $SEED,
  "scheduler": "$SCHEDULER",
  "exp_dir": "$EXP_DIR"
}
EOF
)

DEPLOY_RESPONSE=$(curl -s -X POST "$ORCHESTRATOR_URL/deploy" \
    -H "Content-Type: application/json" \
    -d "$DEPLOY_PAYLOAD")

if [[ $? -ne 0 ]] || echo "$DEPLOY_RESPONSE" | grep -q "detail"; then
    error "Deployment FAILED: $DEPLOY_RESPONSE"
    exit 1
fi

info "Deployment complete."
echo ""

# ── Step 4: Start Inference in Background ────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "PHASE 3: Runtime Inference (background)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "Starting inference in background (will run for ${DURATION}s)..."

# Start inference (returns immediately, runs in background on server)
INFERENCE_START_TIME=$(date +%s)
RUN_RESPONSE=$(curl -s -X POST "$ORCHESTRATOR_URL/run")

if echo "$RUN_RESPONSE" | grep -q "detail"; then
    error "Failed to start inference: $RUN_RESPONSE"
    exit 1
fi

info "Inference started successfully at $(date '+%H:%M:%S')."
echo ""

# ── Step 5: Wait, then Add New Task ──────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "PHASE 4: Runtime Task Addition"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Calculate how long to sleep to add task at the right experiment time
CURRENT_TIME=$(date +%s)
ELAPSED_SINCE_INFERENCE_START=$((CURRENT_TIME - INFERENCE_START_TIME))
SLEEP_TIME=$((ADD_TASK_AFTER - ELAPSED_SINCE_INFERENCE_START))

if (( SLEEP_TIME > 0 )); then
    info "Waiting ${SLEEP_TIME}s to add new task at experiment time ${ADD_TASK_AFTER}s..."
    sleep "$SLEEP_TIME"
else
    warn "Already past target time (${ADD_TASK_AFTER}s), adding task immediately (elapsed=${ELAPSED_SINCE_INFERENCE_START}s)"
fi

info "Adding new task: $NEW_TASK (type=$NEW_TASK_TYPE, workload=$NEW_TASK_WORKLOAD)"

ADD_TASK_PAYLOAD=$(cat <<EOF
{
  "task_name": "$NEW_TASK",
  "task_type": "$NEW_TASK_TYPE",
  "task_workload": $NEW_TASK_WORKLOAD,
  "elapsed_time": $ADD_TASK_AFTER
}
EOF
)

ADD_TASK_RESPONSE=$(curl -s -X POST "$ORCHESTRATOR_URL/add-task" \
    -H "Content-Type: application/json" \
    -d "$ADD_TASK_PAYLOAD")

if [[ $? -ne 0 ]] || echo "$ADD_TASK_RESPONSE" | grep -q "detail"; then
    warn "Task addition FAILED: $ADD_TASK_RESPONSE (inference may still be running)"
else
    info "Task added successfully! Inference continues with new task."
fi
echo ""

# ── Step 6: Wait for Inference to Complete ───────────────────────────
info "Waiting for inference to complete..."

# Call the /wait endpoint which blocks until all site managers send runtime ACKs
WAIT_RESPONSE=$(curl -s -X POST "$ORCHESTRATOR_URL/wait")

if echo "$WAIT_RESPONSE" | grep -q "completed"; then
    info "Inference completed successfully."
else
    warn "Wait response: $WAIT_RESPONSE"
fi
echo ""

# ── Step 7: Cleanup ──────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "PHASE 5: Cleanup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "Cleaning up devices..."

CLEANUP_RESPONSE=$(curl -s -X POST "$ORCHESTRATOR_URL/cleanup")

if [[ $? -ne 0 ]] || echo "$CLEANUP_RESPONSE" | grep -q "detail"; then
    warn "Cleanup had issues: $CLEANUP_RESPONSE (non-fatal)"
else
    info "Cleanup complete."
fi
sleep 10
echo ""

# ── Summary ──────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "Runtime Task Addition Experiment Complete!"
info "Results in: $EXP_DIR/$SCHEDULER/$REQ_RATE/"
info "Logs:"
info "  - Site manager: $SITE_LOG"
info "  - Orchestrator: $ORCHESTRATOR_LOG"
info "  - Inference: $EXP_DIR/runtime_inference.log"

# Check results
RESULT_DIR="$EXP_DIR/$SCHEDULER/$REQ_RATE"
if [[ -f "$RESULT_DIR/request_latency_results.csv" ]]; then
    NROWS=$(wc -l < "$RESULT_DIR/request_latency_results.csv")
    info "Results CSV: $((NROWS - 1)) requests logged"
else
    warn "Results CSV not found in $RESULT_DIR"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
