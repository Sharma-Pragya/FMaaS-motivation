#!/usr/bin/env bash
# Sweep all (regime, N, condition) scenarios for the end-to-end real-world
# experiment.
#
# Usage (from serving/):
#   bash experiments/end_to_end_realworld/run.sh
#
# Restrict the sweep:
#   REGIMES="high"        bash experiments/end_to_end_realworld/run.sh
#   NS="8 16"             bash experiments/end_to_end_realworld/run.sh
#   CONDITIONS="fmaas"    bash experiments/end_to_end_realworld/run.sh
#
# Results land in experiments/end_to_end_realworld/results/<regime>_N<N>/<condition>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVING_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

RESULTS_ROOT="${RESULTS_DIR:-$SCRIPT_DIR/results}"

# ── Pull defaults from user_config.py (env-var overrides still work) ──────────
cd "$SERVING_DIR"   # needed so the Python import path resolves

_py() { python -c "
import sys; sys.path.insert(0, '.')
from experiments.end_to_end_realworld import user_config as uc
$1
"; }

if [ -z "${REGIMES+x}" ]; then
    REGIMES="$(_py "print(' '.join(uc.experiment['load_regimes']))")"
fi

if [ -z "${NS+x}" ]; then
    NS="$(_py "print(' '.join(str(n) for n in uc.experiment['n_tasks_sweep']))")"
fi

if [ -z "${CONDITIONS+x}" ]; then
    CONDITIONS="$(_py "print(' '.join(uc.conditions))")"
fi

cd "$SERVING_DIR"

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
python -m experiments.end_to_end_realworld.deployments.generate \
    > experiments/end_to_end_realworld/deployments/deployment.log

for regime in $REGIMES; do
    for n in $NS; do
        for cond in $CONDITIONS; do
            out="$RESULTS_ROOT/${regime}_N${n}/${cond}"
            echo ""
            echo "================================================================"
            echo "[run.sh] regime=$regime  N=$n  condition=$cond  output=$out"
            echo "================================================================"

            if [ -f "$out/request_latency_results.csv" ] && [ -f "$out/serving_timing_summary.json" ]; then
                echo "[run.sh] SKIP: results already exist at $out"
                continue
            fi

            scenario_dir="$SCRIPT_DIR/deployments/${regime}_N${n}"
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

            python -u -m experiments.end_to_end_realworld.run \
                --regime "$regime" \
                --n "$n" \
                --condition "$cond" \
                --output-dir "$out"
        done
    done
done

echo ""
echo "[run.sh] All done. Results in $RESULTS_ROOT"
