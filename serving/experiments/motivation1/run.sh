#!/bin/bash
# Motivation Experiment #1 — task_sharing vs deploy_sharing (in-process, backbone-only)
# Run from the serving/ directory, or it will cd there automatically.
#
# Environment variables (all optional, shown with defaults):
#   CUDA_DEVICE           cuda:0
#   BACKBONE              chronosbase
#   N_TASKS               10,20,30,40,50
#   PHASE_DURATION        60          seconds per run
#   MAX_BATCH_SIZE        32
#   MAX_BATCH_WAIT_MS     10
#   STRATEGIES            task_sharing,deploy_sharing
#   EXP_DIR               experiments/motivation1/results

set -e
cd "$(dirname "$0")/../.."   # go to serving/

CUDA_DEVICE=${CUDA_DEVICE:-"cuda:0"}
BACKBONE=${BACKBONE:-"chronosbase"}
N_TASKS=${N_TASKS:-"1,2,4,6,8,10"}
PHASE_DURATION=${PHASE_DURATION:-60}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-32}
MAX_BATCH_WAIT_MS=${MAX_BATCH_WAIT_MS:-10}
STRATEGIES=${STRATEGIES:-"task_sharing,deploy_sharing"}
EXP_DIR=${EXP_DIR:-"experiments/motivation1/results"}

echo "=========================================="
echo "  Motivation Experiment #1"
echo "  Backbone        : ${BACKBONE}"
echo "  CUDA device     : ${CUDA_DEVICE}"
echo "  N tasks sweep   : ${N_TASKS}"
echo "  Strategies      : ${STRATEGIES}"
echo "  Duration/run    : ${PHASE_DURATION}s"
echo "  Workers/task    : 1 (closed-loop, N tasks concurrent)"
echo "  Max batch size  : ${MAX_BATCH_SIZE}"
echo "  Max batch wait  : ${MAX_BATCH_WAIT_MS} ms"
echo "  Results         : ${EXP_DIR}"
echo "=========================================="

CUDA_DEVICE=${CUDA_DEVICE} \
BACKBONE=${BACKBONE} \
python experiments/motivation1/run.py \
    --n-tasks          "${N_TASKS}" \
    --duration         "${PHASE_DURATION}" \
    --strategies       "${STRATEGIES}" \
    --backbone         "${BACKBONE}" \
    --cuda             "${CUDA_DEVICE}" \
    --max-batch-size   "${MAX_BATCH_SIZE}" \
    --max-batch-wait-ms "${MAX_BATCH_WAIT_MS}" \
    --exp-dir          "${EXP_DIR}"
