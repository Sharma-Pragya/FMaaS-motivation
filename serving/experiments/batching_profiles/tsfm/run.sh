#!/bin/bash
# Usage (from serving/): bash experiments/batching_profiles/tsfm/run.sh [N_RUNS] [extra args]
N_RUNS=${1:-1}
for i in $(seq 0 $((N_RUNS - 1))); do
    echo "=== Run $i ==="
    python experiments/batching_profiles/tsfm/run.py --run_idx $i "${@:2}"
done
