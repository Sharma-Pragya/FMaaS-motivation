#!/usr/bin/env bash
# Sweep all (mix, condition) scenarios for the end-to-end real-world (mix)
# experiment.
#
# Usage (from serving/):
#   bash experiments/end_to_end_realworld_mix/run.sh
#
# Restrict the sweep:
#   MIX_LABELS="mix_L8_M8_H8"  bash experiments/end_to_end_realworld_mix/run.sh
#   CONDITIONS="fmaas"          bash experiments/end_to_end_realworld_mix/run.sh
#   FORCE_RERUN=1              bash experiments/end_to_end_realworld_mix/run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVING_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

RESULTS_ROOT="${RESULTS_DIR:-$SCRIPT_DIR/results}"

cd "$SERVING_DIR"

_py() { python -c "
import sys; sys.path.insert(0, '.')
from experiments.end_to_end_realworld_mix import user_config as uc
$1
"; }

if [ -z "${MIX_LABELS+x}" ]; then
    MIX_LABELS="$(_py "print(' '.join(uc.mix_label(m) for m in uc.experiment['mix_sweep']))")"
fi

if [ -z "${CONDITIONS+x}" ]; then
    CONDITIONS="$(_py "print(' '.join(uc.conditions))")"
fi

# ── Preprocess MAF days if needed ─────────────────────────────────────────────
MAF_N_DAYS="$(_py "print(int(uc.experiment.get('maf_n_days', 1)))")"
MAF_NPZ_DIR="$SERVING_DIR/traces/azurefunctions/preprocessed"
for d in $(seq 1 "$MAF_N_DAYS"); do
    npz="$MAF_NPZ_DIR/hashowner_day$(printf '%02d' "$d").npz"
    if [ ! -f "$npz" ]; then
        echo "[run.sh] Preprocessing MAF day $d..."
        python -m traces.maf_preprocess "$d"
    fi
done

echo "[run.sh] Generating deployment plans..."
python -m experiments.end_to_end_realworld_mix.deployments.generate \
    > experiments/end_to_end_realworld_mix/deployments/deployment.log

for label in $MIX_LABELS; do
    for cond in $CONDITIONS; do
        out="$RESULTS_ROOT/${label}/${cond}"
        echo ""
        echo "================================================================"
        echo "[run.sh] ${label}  condition=$cond  output=$out"
        echo "================================================================"

        if [ "${FORCE_RERUN:-0}" != "1" ] && \
           [ -f "$out/request_latency_results.csv" ] && \
           [ -f "$out/serving_timing_summary.json" ]; then
            echo "[run.sh] SKIP: results already exist at $out"
            continue
        fi

        scenario_dir="$SCRIPT_DIR/deployments/${label}"
        if ! python - "$scenario_dir" "$cond" <<'PY'
import json, sys
from pathlib import Path
scenario_dir, cond = Path(sys.argv[1]), sys.argv[2]
task_meta = json.loads((scenario_dir / "task_meta.json").read_text())
slots     = json.loads((scenario_dir / f"{cond}_slots.json").read_text())
if cond == "fmaas":
    placed = {t["task"] for s in slots for t in s["tasks"]}
else:
    placed = {s["task"] for s in slots}
rejected = sorted(set(task_meta) - placed)
if rejected:
    print(f"[run.sh] SKIP: {len(rejected)}/{len(task_meta)} tasks not placed "
          f"by {cond} for {scenario_dir.name}: {rejected}")
    sys.exit(1)
PY
        then
            continue
        fi

        python -u -m experiments.end_to_end_realworld_mix.run \
            --mix-label "$label" \
            --condition "$cond" \
            --output-dir "$out"
    done
done

echo ""
echo "[run.sh] All done. Results in $RESULTS_ROOT"
