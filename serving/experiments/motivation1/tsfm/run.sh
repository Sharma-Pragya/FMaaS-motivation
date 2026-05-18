#!/bin/bash
# Motivation Experiment #1 — task_sharing vs deploy_sharing (in-process)
# Run from the serving/ directory, or it will cd there automatically.
#
# Environment variables (all optional):
#   CONDA_ENV          fmtk (conda environment name)
#   FMTK_DIR           ../../../FMTK (relative path or absolute)
#   FMAAS_DIR          ../.. (relative path or absolute)
#   CUDA_DEVICE        cuda:0
#   BACKBONE           momentbase
#   DECODER_DIR        ${FMTK_DIR}/models/tsfm/finetuned
#   N_TASKS            1,2,4,6,8,10
#   PHASE_DURATION     180 seconds per run
#   STRATEGIES         task_sharing,deploy_sharing
#   EXP_DIR            experiments/motivation1/tsfm/results
#   DATASET_DIR        (auto-set from site_manager if not provided)

set -euo pipefail

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SERVING_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$SERVING_DIR"

# Conda environment for Python
CONDA_ENV="${CONDA_ENV:-fmtk}"

# Project directories (can be relative or absolute)
# Default: FMTK is sibling directory of FMaaS-motivation, FMAAS_DIR is parent of serving
FMTK_DIR="${FMTK_DIR:-../../FMTK}"
FMAAS_DIR="${FMAAS_DIR:-..}"

# Convert to absolute paths if relative
if [[ ! "$FMTK_DIR" = /* ]]; then
    # Try relative to SERVING_DIR first
    if [[ -d "$SERVING_DIR/$FMTK_DIR" ]]; then
        FMTK_DIR="$SERVING_DIR/$FMTK_DIR"
    # Try from workspace root (FMTK is sibling of FMaaS-motivation)
    elif [[ -d "$(dirname "$(dirname "$SERVING_DIR")")/../FMTK" ]]; then
        FMTK_DIR="$(cd "$(dirname "$(dirname "$SERVING_DIR")")/../FMTK" && pwd)"
    fi
fi

if [[ ! "$FMAAS_DIR" = /* ]]; then
    # Try relative to SERVING_DIR first
    if [[ -d "$SERVING_DIR/$FMAAS_DIR" ]]; then
        FMAAS_DIR="$SERVING_DIR/$FMAAS_DIR"
    # Try parent directory
    elif [[ -d "$(dirname "$SERVING_DIR")" ]]; then
        FMAAS_DIR="$(dirname "$SERVING_DIR")"
    fi
fi

# Validate paths exist
if [[ ! -d "$FMTK_DIR" ]]; then
    echo "ERROR: FMTK_DIR not found at: $FMTK_DIR"
    echo "Please set FMTK_DIR environment variable:"
    echo "  export FMTK_DIR=/path/to/FMTK"
    echo "  bash experiments/motivation1/tsfm/run.sh"
    exit 1
fi
if [[ ! -d "$FMAAS_DIR" ]]; then
    echo "ERROR: FMAAS_DIR not found at: $FMAAS_DIR"
    echo "Please set FMAAS_DIR environment variable:"
    echo "  export FMAAS_DIR=/path/to/FMaaS-motivation"
    echo "  bash experiments/motivation1/tsfm/run.sh"
    exit 1
fi

# Set up PYTHONPATH
export PYTHONPATH="${FMTK_DIR}/src:${FMAAS_DIR}:${PYTHONPATH:-}"

# Export dataset directory for FMTK to find datasets
export DATASET_DIR

# Python executable from conda environment
# Try conda run first, fall back to explicit PYTHON variable
if command -v conda &> /dev/null; then
    PYTHON="${PYTHON:-conda run -n ${CONDA_ENV} python}"
else
    # If conda not in PATH, try to find Python from environment
    PYTHON="${PYTHON:-python}"
fi

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
CUDA_DEVICE=${CUDA_DEVICE:-"cuda:0"}
BACKBONE=${BACKBONE:-"momentlarge"}
DECODER_DIR=${DECODER_DIR:-"${FMTK_DIR}/models/tsfm/finetuned"}
N_TASKS=${N_TASKS:-"1,2,4,6,8"}
PHASE_DURATION=${PHASE_DURATION:-"60"}
STRATEGIES=${STRATEGIES:-"task_sharing,deploy_sharing"}
EXP_DIR=${EXP_DIR:-"experiments/motivation1/tsfm/results"}

echo "================================================================"
echo "  Motivation Experiment #1 — Task Sharing vs Deploy Sharing"
echo "  Conda env      : $CONDA_ENV"
echo "  FMTK_DIR       : $FMTK_DIR"
echo "  FMAAS_DIR      : $FMAAS_DIR"
echo "  Backbone       : $BACKBONE"
echo "  CUDA device    : $CUDA_DEVICE"
echo "  Decoder dir    : ${DECODER_DIR:-'(backbone-only)'}"
echo "  N tasks sweep  : $N_TASKS"
echo "  Duration/run   : ${PHASE_DURATION}s"
echo "  Strategies     : $STRATEGIES"
echo "  Results        : $EXP_DIR"
echo "================================================================"

DECODER_DIR_ARG=""
if [ -n "${DECODER_DIR:-}" ] && [ -d "${DECODER_DIR}" ]; then
    DECODER_DIR_ARG="--decoder-dir ${DECODER_DIR}"
fi

$PYTHON experiments/motivation1/tsfm/run.py \
    --n-tasks          "${N_TASKS}" \
    --duration         "${PHASE_DURATION}" \
    --strategies       "${STRATEGIES}" \
    --backbone         "${BACKBONE}" \
    --cuda             "${CUDA_DEVICE}" \
    ${DECODER_DIR_ARG} \
    --exp-dir          "${EXP_DIR}"
