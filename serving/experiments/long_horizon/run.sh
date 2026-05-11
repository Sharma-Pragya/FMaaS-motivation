#!/usr/bin/env bash
# Run all long-horizon conditions sequentially.
#
# Usage (from serving/):
#   bash experiments/long_horizon/run.sh
#
# Override a single condition:
#   CONDITION=fmaas bash experiments/long_horizon/run.sh
#
# Results land in experiments/long_horizon/results/<condition>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVING_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

RESULTS_ROOT="${RESULTS_DIR:-$SCRIPT_DIR/results}"
SEED="${SEED:-42}"

# Re-generate deployment plans before running
echo "[run.sh] Generating deployment plans..."
cd "$SERVING_DIR"
python -m experiments.long_horizon.deployments.generate > experiments/long_horizon/deployments/deployment.log

# Conditions to run — override with CONDITION=<name>
if [[ -n "${CONDITION:-}" ]]; then
    CONDITIONS=("$CONDITION")
else
    CONDITIONS=(fmaas no_sharing no_sharing_tpc)
fi

for cond in "${CONDITIONS[@]}"; do
    out="$RESULTS_ROOT/$cond"
    echo ""
    echo "================================================================"
    echo "[run.sh] condition=$cond  output=$out"
    echo "================================================================"
    python -u -m experiments.long_horizon.run \
        --condition "$cond" \
        --output-dir "$out" \
        --seed "$SEED"
done

echo ""
echo "[run.sh] All done. Results in $RESULTS_ROOT"
