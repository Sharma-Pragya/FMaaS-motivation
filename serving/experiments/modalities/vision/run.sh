#!/bin/bash
N_RUNS=${1:-5}
for backbone in dinobase swinsmall; do
    echo "=== Backbone: $backbone ==="
    for i in $(seq 0 $((N_RUNS - 1))); do
        echo "=== Run $i ==="
        python experiments/modalities/vision/run.py --run_idx $i --backbone $backbone "${@:2}"
    done
done
