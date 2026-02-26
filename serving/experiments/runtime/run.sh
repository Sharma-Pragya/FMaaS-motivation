#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  FMaaS Runtime "System in Action" Experiment
#
#  Setup: 2 GPUs, 1 initial task (ecgclass @ REQ_RATE req/s)
#
#  Nine-event timeline (easy → hard):
#    t=0s    Initial deploy: ecgclass @ REQ_RATE req/s
#    t=60s   [EVENT 1] New task:       gestureclass @ 8 req/s   → add_decoder (shares backbone)
#    t=120s  [EVENT 2] Workload ramp:  ecgclass     +5 req/s    → rebalance (step 1)
#    t=180s  [EVENT 3] Workload ramp:  ecgclass     +5 req/s    → rebalance (step 2)
#    t=240s  [EVENT 4] Workload ramp:  ecgclass     +5 req/s    → rebalance (step 3)
#    t=300s  [EVENT 5] New task:       sysbp        @ 8 req/s   → runtime_add (new backbone on device2)
#    t=360s  [EVENT 6] New task:       diasbp       @ 20 req/s  → add_decoder (shares device2 backbone)
#    t=420s  [EVENT 7] Workload spike: ecgclass     +15 req/s   → fit (device2 backbone downsize)
#    t=480s  [EVENT 8] New task:       heartrate    @ 8 req/s   → add_decoder (shares device1 backbone)
#    t=540s  [EVENT 9] Workload spike: heartrate    +15 req/s   → fit (device1 backbone downsize)
#
#  Usage:
#    cd serving
#    bash experiments/runtime/run.sh
#
#    # Run a single scheduler:
#    SCHEDULERS="fmaas_share" bash experiments/runtime/run.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Schedulers to run ────────────────────────────────────────────────
SCHEDULERS="${SCHEDULERS:-fmaas_share}"

# ── Shared configuration ─────────────────────────────────────────────
REQ_RATE="${REQ_RATE:-10}"
TRACE="${TRACE:-lmsyschat}"
DURATION="${DURATION:-480}"
SEED="${SEED:-42}"
EXP_DIR="${EXP_DIR:-experiments/runtime/results}"
EXP_TYPE="${EXP_TYPE:-runtime}"
ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-8080}"

# Event 1: new task — add_decoder (shares existing momentbase backbone)
EVENT1_TASK="${EVENT1_TASK:-gestureclass}"
EVENT1_TASK_TYPE="${EVENT1_TASK_TYPE:-classification}"
EVENT1_WORKLOAD="${EVENT1_WORKLOAD:-8.0}"
EVENT1_AT="${EVENT1_AT:-60}"

# Event 2: workload ramp step 1 — rebalance
EVENT2_TASK="${EVENT2_TASK:-ecgclass}"
EVENT2_TASK_TYPE="${EVENT2_TASK_TYPE:-classification}"
EVENT2_SPIKE="${EVENT2_SPIKE:-5}"
EVENT2_AT="${EVENT2_AT:-120}"

# Event 3: workload ramp step 2 — rebalance
EVENT3_TASK="${EVENT3_TASK:-ecgclass}"
EVENT3_TASK_TYPE="${EVENT3_TASK_TYPE:-classification}"
EVENT3_SPIKE="${EVENT3_SPIKE:-10}"
EVENT3_AT="${EVENT3_AT:-180}"

# Event 4: workload ramp step 3 — rebalance
EVENT4_TASK="${EVENT4_TASK:-ecgclass}"
EVENT4_TASK_TYPE="${EVENT4_TASK_TYPE:-classification}"
EVENT4_SPIKE="${EVENT4_SPIKE:-5}"
EVENT4_AT="${EVENT4_AT:-240}"

# Event 5: new task — runtime_add (new backbone on device2)
EVENT5_TASK="${EVENT5_TASK:-sysbp}"
EVENT5_TASK_TYPE="${EVENT5_TASK_TYPE:-regression}"
EVENT5_WORKLOAD="${EVENT5_WORKLOAD:-8.0}"
EVENT5_AT="${EVENT5_AT:-300}"

# Event 6: new task — add_decoder (shares device2 backbone)
EVENT6_TASK="${EVENT6_TASK:-diasbp}"
EVENT6_TASK_TYPE="${EVENT6_TASK_TYPE:-regression}"
EVENT6_WORKLOAD="${EVENT6_WORKLOAD:-20}"
EVENT6_AT="${EVENT6_AT:-360}"

# Event 7: large workload spike — fit (device2 backbone downsize)
EVENT7_TASK="${EVENT7_TASK:-ecgclass}"
EVENT7_TASK_TYPE="${EVENT7_TASK_TYPE:-classification}"
EVENT7_SPIKE="${EVENT7_SPIKE:-15}"
EVENT7_AT="${EVENT7_AT:-420}"

# # Event 8: new task — add_decoder (shares device1 momentbase backbone)
# EVENT8_TASK="${EVENT8_TASK:-heartrate}"
# EVENT8_TASK_TYPE="${EVENT8_TASK_TYPE:-regression}"
# EVENT8_WORKLOAD="${EVENT8_WORKLOAD:-8.0}"
# EVENT8_AT="${EVENT8_AT:-480}"

# # Event 9: large workload spike — fit (device1 backbone downsize)
# EVENT9_TASK="${EVENT9_TASK:-heartrate}"
# EVENT9_TASK_TYPE="${EVENT9_TASK_TYPE:-regression}"
# EVENT9_SPIKE="${EVENT9_SPIKE:-15}"
# EVENT9_AT="${EVENT9_AT:-540}"

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVING_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ORCHESTRATOR_URL="http://localhost:$ORCHESTRATOR_PORT"

# ── Colors ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}   $*"; }
error()   { echo -e "${RED}[ERROR]${NC}  $*"; }
event()   { echo -e "${BLUE}[EVENT]${NC}  $*"; }
section() { echo -e "${CYAN}[RUN]${NC}    $*"; }

# ── Helper: sleep until target experiment wall-clock time ─────────────
sleep_until_exp_time() {
    local target_s="$1"
    local start_epoch="$2"
    local now elapsed remaining
    now=$(date +%s)
    elapsed=$(( now - start_epoch ))
    remaining=$(( target_s - elapsed ))
    if (( remaining > 0 )); then
        info "  Sleeping ${remaining}s until experiment t=${target_s}s..."
        sleep "$remaining"
    else
        warn "  Already past t=${target_s}s (elapsed=${elapsed}s), firing immediately."
    fi
}

# ── Helper: POST add-task ─────────────────────────────────────────────
fire_add_task() {
    local task_name="$1" task_type="$2" workload="$3" elapsed_time="$4"
    event "add-task → ${task_name} (${task_type}) @ ${workload} req/s at t=${elapsed_time}s"
    local resp
    resp=$(curl -s -X POST "$ORCHESTRATOR_URL/add-task" \
        -H "Content-Type: application/json" \
        -d "{\"task_name\":\"$task_name\",\"task_type\":\"$task_type\",\"task_workload\":$workload,\"elapsed_time\":$elapsed_time}") || true
    if echo "$resp" | grep -q '"detail"'; then
        warn "  add-task failed: $resp"
    else
        info "  add-task accepted."
    fi
}

# ── Helper: POST add-workload ─────────────────────────────────────────
fire_add_workload() {
    local task_name="$1" task_type="$2" delta="$3" elapsed_time="$4"
    event "add-workload → ${task_name} delta=+${delta} req/s at t=${elapsed_time}s"
    local resp
    resp=$(curl -s -X POST "$ORCHESTRATOR_URL/add-workload" \
        -H "Content-Type: application/json" \
        -d "{\"task_name\":\"$task_name\",\"task_type\":\"$task_type\",\"task_workload\":$delta,\"elapsed_time\":$elapsed_time}") || true
    if echo "$resp" | grep -q '"detail"'; then
        warn "  add-workload failed: $resp"
    else
        info "  add-workload accepted."
    fi
}

# ── Per-scheduler run function ────────────────────────────────────────
run_scheduler() {
    local SCHEDULER="$1"
    local SITE_LOG="$SERVING_DIR/site_manager.log"
    local ORCHESTRATOR_LOG="$SERVING_DIR/orchestrator.log"
    local SITE_PID="" ORCHESTRATOR_PID=""

    # Per-scheduler cleanup
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
    info "Setup:    2 GPUs, 1 initial task (ecgclass) @ ${REQ_RATE} req/s"
    info "Timeline:"
    info "  t=0s       Initial deploy: ecgclass @ ${REQ_RATE} req/s"
    info "  t=${EVENT1_AT}s      [EVENT 1] add-task:     ${EVENT1_TASK} @ ${EVENT1_WORKLOAD} req/s        (add_decoder)"
    info "  t=${EVENT2_AT}s     [EVENT 2] add-workload:  ${EVENT2_TASK} +${EVENT2_SPIKE} req/s             (ramp step 1)"
    info "  t=${EVENT3_AT}s     [EVENT 3] add-workload:  ${EVENT3_TASK} +${EVENT3_SPIKE} req/s             (ramp step 2)"
    info "  t=${EVENT4_AT}s     [EVENT 4] add-workload:  ${EVENT4_TASK} +${EVENT4_SPIKE} req/s             (ramp step 3)"
    info "  t=${EVENT5_AT}s     [EVENT 5] add-task:      ${EVENT5_TASK} @ ${EVENT5_WORKLOAD} req/s        (runtime_add, device2)"
    info "  t=${EVENT6_AT}s     [EVENT 6] add-task:      ${EVENT6_TASK} @ ${EVENT6_WORKLOAD} req/s        (add_decoder, device2)"
    info "  t=${EVENT7_AT}s     [EVENT 7] add-workload:  ${EVENT7_TASK} +${EVENT7_SPIKE} req/s            (fit/device2 backbone downsize)"
    # info "  t=${EVENT8_AT}s     [EVENT 8] add-task:      ${EVENT8_TASK} @ ${EVENT8_WORKLOAD} req/s        (add_decoder, device1)"
    # info "  t=${EVENT9_AT}s     [EVENT 9] add-workload:  ${EVENT9_TASK} +${EVENT9_SPIKE} req/s            (fit/device1 backbone downsize)"
    echo ""

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
    local INFERENCE_START
    INFERENCE_START=$(date +%s)
    local run_resp
    run_resp=$(curl -s -X POST "$ORCHESTRATOR_URL/run")
    if echo "$run_resp" | grep -q '"detail"'; then
        error "Failed to start inference: $run_resp"; return 1
    fi
    info "Inference started at $(date '+%H:%M:%S') (t=0s)."

    # ── EVENT 1: New task — add_decoder (shares momentbase) ──────────
    sleep_until_exp_time "$EVENT1_AT" "$INFERENCE_START"
    fire_add_task "$EVENT1_TASK" "$EVENT1_TASK_TYPE" "$EVENT1_WORKLOAD" "$EVENT1_AT"

    # ── EVENT 2: Workload ramp step 1 — rebalance ────────────────────
    sleep_until_exp_time "$EVENT2_AT" "$INFERENCE_START"
    fire_add_workload "$EVENT2_TASK" "$EVENT2_TASK_TYPE" "$EVENT2_SPIKE" "$EVENT2_AT"

    # ── EVENT 3: Workload ramp step 2 — rebalance ────────────────────
    sleep_until_exp_time "$EVENT3_AT" "$INFERENCE_START"
    fire_add_workload "$EVENT3_TASK" "$EVENT3_TASK_TYPE" "$EVENT3_SPIKE" "$EVENT3_AT"

    # ── EVENT 4: Workload ramp step 3 — rebalance ────────────────────
    sleep_until_exp_time "$EVENT4_AT" "$INFERENCE_START"
    fire_add_workload "$EVENT4_TASK" "$EVENT4_TASK_TYPE" "$EVENT4_SPIKE" "$EVENT4_AT"

    # ── EVENT 5: New task — runtime_add (new backbone on device2) ────
    sleep_until_exp_time "$EVENT5_AT" "$INFERENCE_START"
    fire_add_task "$EVENT5_TASK" "$EVENT5_TASK_TYPE" "$EVENT5_WORKLOAD" "$EVENT5_AT"

    # ── EVENT 6: New task — add_decoder (shares device2 backbone) ────
    sleep_until_exp_time "$EVENT6_AT" "$INFERENCE_START"
    fire_add_task "$EVENT6_TASK" "$EVENT6_TASK_TYPE" "$EVENT6_WORKLOAD" "$EVENT6_AT"

    # ── EVENT 7: Large spike — fit (device2 backbone downsize) ───────
    sleep_until_exp_time "$EVENT7_AT" "$INFERENCE_START"
    fire_add_workload "$EVENT7_TASK" "$EVENT7_TASK_TYPE" "$EVENT7_SPIKE" "$EVENT7_AT"

    # # ── EVENT 8: New task — add_decoder (shares device1 momentbase) ──
    # sleep_until_exp_time "$EVENT8_AT" "$INFERENCE_START"
    # fire_add_task "$EVENT8_TASK" "$EVENT8_TASK_TYPE" "$EVENT8_WORKLOAD" "$EVENT8_AT"

    # # ── EVENT 9: Large spike — fit (device1 backbone downsize) ────────
    # sleep_until_exp_time "$EVENT9_AT" "$INFERENCE_START"
    # fire_add_workload "$EVENT9_TASK" "$EVENT9_TASK_TYPE" "$EVENT9_SPIKE" "$EVENT9_AT"

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

    # Reset trap for next iteration
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
