#!/bin/bash
# Vision batching profile experiment.
# Usage (from serving/): bash experiments/batching_profiles/vision/run.sh [N_RUNS] [extra args]
#
# Environment variables (all optional):
#   BACKBONE        dinobase-patch (default)
#   TASK            nyudepth (default); set to "vocseg" for segmentation
#   BATCH_SIZES     1,2,4,8,16,32 (default)
#   N_REQUESTS      200 (default)
#   CUDA_DEVICE     cuda:0 (default)

N_RUNS=${1:-1}
for i in $(seq 0 $((N_RUNS - 1))); do
    echo "=== Run $i ==="
    python experiments/batching_profiles/vision/run.py --run_idx $i "${@:2}"
done
