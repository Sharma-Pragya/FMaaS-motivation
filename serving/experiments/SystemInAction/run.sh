#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  FMaaS System-in-Action Experiment  (local mode — no MQTT)
#
#  All experiment configuration lives in user_config.py.
#  This script reads from it, passes values to the orchestrator,
#  and saves a copy of the config in the results folder.
#
#  Usage:
#    cd serving
#    bash experiments/SystemInAction/run.sh
#
#    # Override scheduler:
#    SCHEDULERS="fmaas_share" bash experiments/SystemInAction/run.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVING_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Read experiment config from user_config.py ──────────────────────
read_cfg() {
    python3 -c "
from experiments.SystemInAction.user_config import experiment
value = experiment.get('$1', '$2')
if isinstance(value, list):
    print(','.join(str(v) for v in value))
else:
    print(value)
"
}

REQ_RATE="$(read_cfg req_rate 10)"
TRACE="$(read_cfg trace poisson_per_task)"
DURATION="$(read_cfg duration 20)"
MAX_BATCH_WAIT_MS="$(read_cfg max_batch_wait_ms 0)"
ISOLATION_MODE="$(read_cfg isolation_mode shared)"
WARMUP_GAP="$(read_cfg warmup_gap 2.0)"
MAX_MODEL_LEN="$(read_cfg max_model_len 256)"
BATCH_MODE="$(read_cfg batch_mode util_dummy)"

# ── Overridable from env ─────────────────────────────────────────────
SCHEDULERS="${SCHEDULERS:-clipper_place}"
EXP_DIR="${EXP_DIR:-experiments/SystemInAction/results}"
EXP_TYPE="${EXP_TYPE:-SystemInAction}"

# ── Colors ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}   $*"; }
error()   { echo -e "${RED}[ERROR]${NC}  $*"; }
section() { echo -e "${CYAN}[RUN]${NC}    $*"; }

# ── Save user_config snapshot to results folder ─────────────────────
save_config() {
    local dest="$1/user_config.json"
    python3 -c "
import json, importlib
mod = importlib.import_module('experiments.SystemInAction.user_config')
cfg = {
    'devices': mod.devices,
    'tasks': {k: {kk: vv for kk, vv in v.items()} for k, v in mod.tasks.items()},
    'experiment': mod.experiment,
    'scheduler': '$2',
}
with open('$dest', 'w') as f:
    json.dump(cfg, f, indent=2, default=str)
print('[INFO]   Config saved to $dest')
"
}

# ── Per-scheduler run function ────────────────────────────────────────
run_scheduler() {
    local SCHEDULER="$1"
    local OUT_DIR="$SERVING_DIR/$EXP_DIR/$SCHEDULER"
    local LOG="$OUT_DIR/orchestrator.log"
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
    info "Setup:    SystemInAction tasks @ ${REQ_RATE} req/s, ${DURATION}s trace"
    info "Mode:     local (single process, no MQTT)"
    info "Timeline:"
    info "  t=0s    Deploy + run for ${DURATION}s"
    echo ""

    # ── Save config ──────────────────────────────────────────────────
    mkdir -p "$OUT_DIR"
    save_config "$OUT_DIR" "$SCHEDULER"

    # ── Run experiment (single process) ──────────────────────────────
    info "Starting orchestrator (local mode)..."
    python -u -m orchestrator.server \
        --mode              local \
        --exp-type          "$EXP_TYPE" \
        --scheduler         "$SCHEDULER" \
        --req-rate          "$REQ_RATE" \
        --duration          "$DURATION" \
        --trace             "$TRACE" \
        --exp-dir           "$EXP_DIR" \
        --output-dir        "$OUT_DIR" \
        --max-batch-wait-ms "$MAX_BATCH_WAIT_MS" \
        --isolation-mode    "$ISOLATION_MODE" \
        --warmup-gap        "$WARMUP_GAP" \
        --max-model-len     "$MAX_MODEL_LEN" \
        --batch-mode        "$BATCH_MODE" \
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
    if [[ -f "$OUT_DIR/request_latency_results.csv" ]]; then
        local nrows
        nrows=$(wc -l < "$OUT_DIR/request_latency_results.csv")
        info "Results: $((nrows - 1)) requests → $OUT_DIR"
    else
        warn "Results CSV not found in $OUT_DIR"
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
info "Batch mode: $BATCH_MODE"
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
info "Results in: $EXP_DIR/<scheduler>/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
