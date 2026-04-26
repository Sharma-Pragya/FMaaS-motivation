#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  cluster_sharing_benefit sweep: vary #apps and measure cluster latency.
#
#  Regenerates plans from user_config.py then runs each (N, condition)
#  and writes results to results/N{N}/<condition>/.
#
#  Usage:
#    cd serving
#    bash experiments/cluster_sharing_benefit/run.sh
#
#  Env overrides:
#    N_APPS_LIST   "8 16 24 32"    (defaults to user_config.n_apps_list)
#    CONDITIONS    "sharing no_sharing no_sharing_tpc"
#    DURATION      180
#    RESULTS_DIR   experiments/cluster_sharing_benefit/results
#    SKIP_GEN      1   # set to 1 to skip plan regeneration
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVING_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SERVING_DIR"

N_APPS_LIST="${N_APPS_LIST:-}"
CONDITIONS="${CONDITIONS:-}"
DURATION="${DURATION:-}"
RESULTS_DIR="${RESULTS_DIR:-experiments/cluster_sharing_benefit/results}"
SKIP_GEN="${SKIP_GEN:-0}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}   $*"; }
error()   { echo -e "${RED}[ERROR]${NC}  $*"; }
section() { echo -e "${CYAN}[RUN]${NC}    $*"; }

if [[ -z "$N_APPS_LIST" ]]; then
    N_APPS_LIST=$(python3 -c "
from experiments.cluster_sharing_benefit.user_config import n_apps_list
print(' '.join(str(n) for n in n_apps_list))
")
fi
if [[ -z "$CONDITIONS" ]]; then
    CONDITIONS=$(python3 -c "
from experiments.cluster_sharing_benefit.user_config import conditions
print(' '.join(conditions))
")
fi
if [[ -z "$DURATION" ]]; then
    DURATION=$(python3 -c "
from experiments.cluster_sharing_benefit.user_config import experiment
print(experiment['duration'])
")
fi
info "N values:   $N_APPS_LIST"
info "Conditions: $CONDITIONS"
info "Duration:   ${DURATION}s"
info "Results:    $RESULTS_DIR"

if [[ "$SKIP_GEN" != "1" ]]; then
    info "Regenerating deployment plans..."
    python -m experiments.cluster_sharing_benefit.deployments.generate
fi

run_one() {
    local n="$1"
    local cond="$2"
    local out_dir="$SERVING_DIR/$RESULTS_DIR/N${n}/${cond}"
    local log="$out_dir/run.log"
    mkdir -p "$out_dir"

    if [[ -f "$out_dir/request_latency_results.csv" ]]; then
        warn "[N=$n $cond] result exists — skipping. Delete $out_dir to re-run."
        return 0
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    section "N=${n}  CONDITION: $cond"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python -u -m experiments.cluster_sharing_benefit.run \
        --n-apps     "$n" \
        --condition  "$cond" \
        --duration   "$DURATION" \
        --output-dir "$out_dir" \
        2>&1 | tee "$log"

    if [[ -f "$out_dir/request_latency_results.csv" ]]; then
        local nrows
        nrows=$(wc -l < "$out_dir/request_latency_results.csv")
        info "[N=$n $cond] $((nrows - 1)) requests -> $out_dir"
    else
        error "[N=$n $cond] results CSV not found"
        return 1
    fi
}

for n in $N_APPS_LIST; do
    for cond in $CONDITIONS; do
        run_one "$n" "$cond" || warn "[N=$n $cond] failed — continuing"
        info "Pausing 10s..."
        sleep 10
    done
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "Sweep complete. Results in $RESULTS_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
