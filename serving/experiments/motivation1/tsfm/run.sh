#!/bin/bash
# Motivation Experiment #1 — task_sharing vs deploy_sharing (in-process)
# Run from the serving/ directory, or it will cd there automatically.
#
# Environment variables (all optional, shown with defaults):
#   CUDA_DEVICE           cuda:0
#   BACKBONE              chronosbase
#   DECODER_DIR           (empty = backbone-only; set to run full pipeline)
#                         e.g. /path/to/FMTK/models/tsfm/finetuned
#   N_TASKS               1,2,4,6,8,10
#   PHASE_DURATION        60          seconds per run
#   STRATEGIES            task_sharing,deploy_sharing
#   EXP_DIR               experiments/motivation1/results

set -e
cd "$(dirname "$0")/../.."   # go to serving/

CUDA_DEVICE=${CUDA_DEVICE:-"cuda:0"}
BACKBONE=${BACKBONE:-"momentbase"}
DECODER_DIR=${DECODER_DIR:-"/project/pi_shenoy_umass_edu/hshastri/FMTK/models/tsfm/finetuned"}
N_TASKS=${N_TASKS:-"1,2,4,6,8,10"}
PHASE_DURATION=${PHASE_DURATION:-180}
STRATEGIES=${STRATEGIES:-"task_sharing,deploy_sharing"}
EXP_DIR=${EXP_DIR:-"experiments/motivation1/results"}

echo "=========================================="
echo "  Motivation Experiment #1"
echo "  Backbone        : ${BACKBONE}"
echo "  CUDA device     : ${CUDA_DEVICE}"
echo "  Decoder dir     : ${DECODER_DIR:-'(backbone-only)'}"
echo "  N tasks sweep   : ${N_TASKS}"
echo "  Strategies      : ${STRATEGIES}"
echo "  Duration/run    : ${PHASE_DURATION}s"
echo "  Workers/task    : 1 (closed-loop, N tasks concurrent)"
echo "  Results         : ${EXP_DIR}"
echo "=========================================="

DECODER_DIR_ARG=""
if [ -n "${DECODER_DIR}" ]; then
    DECODER_DIR_ARG="--decoder-dir ${DECODER_DIR}"
fi

CUDA_DEVICE=${CUDA_DEVICE} \
BACKBONE=${BACKBONE} \
python experiments/motivation1/run.py \
    --n-tasks          "${N_TASKS}" \
    --duration         "${PHASE_DURATION}" \
    --strategies       "${STRATEGIES}" \
    --backbone         "${BACKBONE}" \
    --cuda             "${CUDA_DEVICE}" \
    ${DECODER_DIR_ARG} \
    --exp-dir          "${EXP_DIR}"
