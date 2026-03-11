#!/bin/bash
# Motivation LLM Experiment — task_sharing vs deploy_sharing with qwen2.5-0.5b vLLM
# Run from serving/ directory.
#
# Benchmark modes:
#   BENCHMARK_MODE=closed_loop  (default) — fixed CONCURRENCY workers per task
#   BENCHMARK_MODE=open_loop              — Poisson arrivals at TARGET_RPS per task
#   BENCHMARK_MODE=total_rps              — fixed total load split evenly across N tasks

set -e
cd "$(dirname "$0")/../.."  # go to serving/

PHASE_DURATION=${PHASE_DURATION:-60}
EXP_DIR=${EXP_DIR:-"experiments/motivation2_llm/results"}
N_TASKS=${N_TASKS:-"2,4,8,10"}
STRATEGIES=${STRATEGIES:-"deploy_sharing,task_sharing"}
BENCHMARK_MODE=${BENCHMARK_MODE:-"closed_loop"}
CONCURRENCY=${CONCURRENCY:-4}
TARGET_RPS=${TARGET_RPS:-2.0}
TOTAL_RPS=${TOTAL_RPS:-20.0}
MAX_SAMPLES=${MAX_SAMPLES:-50}

echo "=========================================="
echo "  Motivation LLM Experiment"
echo "  Backbone        : qwen2.5-0.5b (vLLM)"
echo "  Duration/run    : ${PHASE_DURATION}s"
echo "  N tasks         : ${N_TASKS}"
echo "  Strategies      : ${STRATEGIES}"
echo "  Benchmark mode  : ${BENCHMARK_MODE}"
if [ "${BENCHMARK_MODE}" = "closed_loop" ]; then
echo "  Concurrency/task: ${CONCURRENCY} workers"
elif [ "${BENCHMARK_MODE}" = "total_rps" ]; then
echo "  Total RPS       : ${TOTAL_RPS} req/s (split across N tasks)"
else
echo "  Target RPS/task : ${TARGET_RPS} req/s"
fi
echo "  Results         : ${EXP_DIR}"
echo "=========================================="

python experiments/motivation2_llm/run.py \
    --n-tasks "${N_TASKS}" \
    --duration "${PHASE_DURATION}" \
    --exp-dir "${EXP_DIR}" \
    --strategies "${STRATEGIES}" \
    --benchmark-mode "${BENCHMARK_MODE}" \
    --concurrency "${CONCURRENCY}" \
    --target-rps "${TARGET_RPS}" \
    --total-rps "${TOTAL_RPS}" \
    --max-samples "${MAX_SAMPLES}"
