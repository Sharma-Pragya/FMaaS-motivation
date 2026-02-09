#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  run_baselines.sh — End-to-end experiment runner
#
#  Launches site_manager, waits for it to be ready, then runs every
#  scheduler × request-rate combination sequentially. Cleans up on exit.
#
#  Usage:
#    cd serving
#    bash experiments/baselines/run_baselines.sh
#
#  To customise:
#    SCHEDULERS="fmaas clipper"  RATES="10 50"  bash experiments/baselines/run_baselines.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configurable parameters (override via environment) ───────────────
SCHEDULERS="${SCHEDULERS:- fmaas}"
RATES="${RATES:-200}"
TRACE="${TRACE:-lmsyschat}"
DURATION="${DURATION:-360}"
SEED="${SEED:-42}"
EXP_DIR="${EXP_DIR:-experiments/baselines/results}"
PLOT_DIR="${PLOT_DIR:-experiments/baselines/plots}"

# ── Derived paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVING_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SITE_LOG="$SERVING_DIR/site_manager.log"

# ── Colours for output ──────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Cleanup trap: kill site_manager on exit ──────────────────────────
SITE_PID=""
cleanup() {
    if [[ -n "$SITE_PID" ]] && kill -0 "$SITE_PID" 2>/dev/null; then
        warn "Killing site_manager (PID $SITE_PID)..."
        kill "$SITE_PID" 2>/dev/null || true
        wait "$SITE_PID" 2>/dev/null || true
    fi
    info "Done. Site manager log: $SITE_LOG"
}
trap cleanup EXIT INT TERM

# ── Step 0: Ensure we're in the serving directory ────────────────────
cd "$SERVING_DIR"
info "Working directory: $(pwd)"
info "Schedulers: $SCHEDULERS"
info "Rates:      $RATES"
info "Trace:      $TRACE  |  Duration: ${DURATION}s  |  Seed: $SEED"
info "Exp dir:    $EXP_DIR"
echo ""

# ── Step 1: Launch site_manager in the background ────────────────────
info "Starting site_manager ..."
python -u site_manager/main.py > "$SITE_LOG" 2>&1 &
SITE_PID=$!
info "Site manager PID: $SITE_PID  (log: $SITE_LOG)"

# ── Step 2: Wait for site_manager to be ready ───────────────────────
READY_SENTINEL="ready. Entering MQTT loop"
MAX_WAIT=300   # seconds
ELAPSED=0
POLL=2

info "Waiting for site_manager to initialise dataloaders ..."
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

# ── Step 3: Run each scheduler × rate combination ───────────────────
TOTAL=0
FAILED=0

for sched in $SCHEDULERS; do
    for rate in $RATES; do
        (( TOTAL += 1 ))
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        info "[$TOTAL] scheduler=$sched  req_rate=$rate"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        COMMON_ARGS="--scheduler $sched --req-rate $rate --duration $DURATION --trace $TRACE --seed $SEED --exp-dir $EXP_DIR"

        # ── Deploy ───────────────────────────────────────────────────
        info "  Deploying models ..."
        if ! python -m orchestrator.main $COMMON_ARGS --deploy-only; then
            error "  Deploy FAILED for $sched @ $rate"
            (( FAILED += 1 ))
            # Try cleanup even if deploy failed
            python -m orchestrator.main $COMMON_ARGS --cleanup-only 2>/dev/null || true
            continue
        fi
        info "  Deploy complete."

        # ── Run ──────────────────────────────────────────────────────
        info "  Running workload ..."
        if ! python -m orchestrator.main $COMMON_ARGS --run-only; then
            error "  Runtime FAILED for $sched @ $rate"
            (( FAILED += 1 ))
            # --run-only already triggers cleanup, but just in case:
            python -m orchestrator.main $COMMON_ARGS --cleanup-only 2>/dev/null || true
            continue
        fi
        info "  Runtime + cleanup complete."
        sleep 10
        # ── Verify output ────────────────────────────────────────────
        RESULT_DIR="$EXP_DIR/$sched/$rate"
        if [[ -f "$RESULT_DIR/request_latency_results.csv" ]]; then
            NROWS=$(wc -l < "$RESULT_DIR/request_latency_results.csv")
            info "  Results: $RESULT_DIR/request_latency_results.csv ($((NROWS - 1)) requests)"
        else
            warn "  Results CSV not found in $RESULT_DIR"
        fi

        # ── Cooldown: let GPU memory + shared mem fully release ───
        info "  Cooldown before next experiment ..."
        sleep 30
        echo ""
    done
done

# ── Step 4: Summary ─────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "All experiments finished.  Total: $TOTAL  Failed: $FAILED"
info "Results in: $EXP_DIR/<scheduler>/<rate>/"
echo ""

# ── Step 5: Generate comparison plots ────────────────────────────────
if (( FAILED < TOTAL )); then
    info "Generating comparison plots ..."
    python experiments/baselines/compare_schedulers.py --exp-dir "$EXP_DIR" --output-dir "$PLOT_DIR" || warn "Plot generation failed (non-fatal)."
else
    warn "All experiments failed; skipping plots."
fi
