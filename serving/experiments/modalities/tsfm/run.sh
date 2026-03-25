#!/bin/bash
N_RUNS=${1:-5}
for backbone in momentbase chronosbase; do
    echo "=== Backbone: $backbone ==="
    for i in $(seq 0 $((N_RUNS - 1))); do
        echo "=== Run $i ==="
        python experiments/modalities/tsfm/run.py --run_idx $i --backbone $backbone "${@:2}"
    done
done
