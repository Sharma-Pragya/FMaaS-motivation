#!/bin/bash
# Motivation LLM Experiment — task_sharing vs deploy_sharing with in-process vLLM
#
# Run from serving/ directory.
#
# Benchmark modes:
#   BENCHMARK_MODE=closed_loop  (default) — fixed CONCURRENCY workers per task
#   BENCHMARK_MODE=open_loop              — Poisson arrivals at TARGET_RPS per task
#   BENCHMARK_MODE=total_rps              — fixed total load split evenly across N tasks
#
# Environment variables (all optional):
#   CONDA_ENV          fmtk (conda environment name)
#   FMTK_DIR           ../../../FMTK (relative path or absolute)
#   FMAAS_DIR          ../.. (relative path or absolute)
#   CUDA_DEVICE        cuda:0
#   BACKBONE           qwen2.5-0.5b
#   N_TASKS            1,2,4,6,8,10
#   PHASE_DURATION     60
#   STRATEGIES         task_sharing,deploy_sharing
#   BENCHMARK_MODE     closed_loop
#   CONCURRENCY        1
#   TARGET_RPS         2.0
#   TOTAL_RPS          20.0
#   MAX_SAMPLES        50
#   UNIFORM_MAX_NEW_TOKENS  64
#   PROMPT_SOURCE_TASK ag_news
#   EXP_DIR            experiments/motivation1/llm/results
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
    echo "  bash experiments/motivation1/llm/run.sh"
    exit 1
fi
if [[ ! -d "$FMAAS_DIR" ]]; then
    echo "ERROR: FMAAS_DIR not found at: $FMAAS_DIR"
    echo "Please set FMAAS_DIR environment variable:"
    echo "  export FMAAS_DIR=/path/to/FMaaS-motivation"
    echo "  bash experiments/motivation1/llm/run.sh"
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
BACKBONE=${BACKBONE:-"qwen2.5-0.5b"}
PHASE_DURATION=${PHASE_DURATION:-10}
EXP_DIR=${EXP_DIR:-"experiments/motivation1/llm/results"}
N_TASKS=${N_TASKS:-"1,4"}
STRATEGIES=${STRATEGIES:-"task_sharing"}
BENCHMARK_MODE=${BENCHMARK_MODE:-"closed_loop"}
CONCURRENCY=${CONCURRENCY:-1}
TARGET_RPS=${TARGET_RPS:-2.0}
TOTAL_RPS=${TOTAL_RPS:-20.0}
MAX_SAMPLES=${MAX_SAMPLES:-50}
UNIFORM_MAX_NEW_TOKENS=${UNIFORM_MAX_NEW_TOKENS:-64}
PROMPT_SOURCE_TASK=${PROMPT_SOURCE_TASK:-"ag_news"}

echo "================================================================"
echo "  Motivation LLM Experiment #1"
echo "  Conda env       : $CONDA_ENV"
echo "  FMTK_DIR        : $FMTK_DIR"
echo "  FMAAS_DIR       : $FMAAS_DIR"
echo "  Backbone        : ${BACKBONE} (vLLM, in-process)"
echo "  CUDA device     : ${CUDA_DEVICE}"
echo "  Duration/run    : ${PHASE_DURATION}s"
echo "  N tasks         : ${N_TASKS}"
echo "  Strategies      : ${STRATEGIES}"
echo "  Benchmark mode  : ${BENCHMARK_MODE}"
echo "  Uniform max out : ${UNIFORM_MAX_NEW_TOKENS} tokens"
echo "  Prompt source   : ${PROMPT_SOURCE_TASK:-'(per-task datasets)'}"
if [ "${BENCHMARK_MODE}" = "closed_loop" ]; then
  echo "  Concurrency/task: ${CONCURRENCY} workers"
elif [ "${BENCHMARK_MODE}" = "total_rps" ]; then
  echo "  Total RPS       : ${TOTAL_RPS} req/s (split across N tasks)"
else
  echo "  Target RPS/task : ${TARGET_RPS} req/s"
fi
echo "  Results         : ${EXP_DIR}"
echo "================================================================"


# Run each (strategy, n_tasks) pair in its own fresh Python process for full GPU isolation
IFS=',' read -ra STRATEGY_LIST <<< "${STRATEGIES}"
IFS=',' read -ra N_TASKS_LIST <<< "${N_TASKS}"
for N in "${N_TASKS_LIST[@]}"; do
    for STRATEGY in "${STRATEGY_LIST[@]}"; do
        echo ""
        echo ">>> [run.sh] strategy=${STRATEGY} n_tasks=${N} — fresh Python process"
        $PYTHON experiments/motivation1/llm/run.py \
            --n-tasks "${N}" \
            --duration "${PHASE_DURATION}" \
            --exp-dir "${EXP_DIR}" \
            --strategies "${STRATEGY}" \
            --backbone "${BACKBONE}" \
            --cuda "${CUDA_DEVICE}" \
            --benchmark-mode "${BENCHMARK_MODE}" \
            --uniform-max-new-tokens "${UNIFORM_MAX_NEW_TOKENS}" \
            --prompt-source-task "${PROMPT_SOURCE_TASK}" \
            --concurrency "${CONCURRENCY}" \
            --target-rps "${TARGET_RPS}" \
            --total-rps "${TOTAL_RPS}" \
            --max-samples "${MAX_SAMPLES}"
        if [ $? -ne 0 ]; then
            echo "Error occurred for strategy=${STRATEGY} n_tasks=${N}. Continuing with next iteration."
            continue
        fi
    done
done