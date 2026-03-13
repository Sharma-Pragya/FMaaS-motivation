#!/bin/bash
# Motivation LLM Experiment — task_sharing vs deploy_sharing with in-process vLLM
# Run from serving/ directory.
#
# Benchmark modes:
#   BENCHMARK_MODE=closed_loop  (default) — fixed CONCURRENCY workers per task
#   BENCHMARK_MODE=open_loop              — Poisson arrivals at TARGET_RPS per task
#   BENCHMARK_MODE=total_rps              — fixed total load split evenly across N tasks

cd "$(dirname "$0")/../../.."  # go to serving/

CUDA_DEVICE=${CUDA_DEVICE:-"cuda:0"}
BACKBONE=${BACKBONE:-"qwen2.5-0.5b"}
PHASE_DURATION=${PHASE_DURATION:-60}
EXP_DIR=${EXP_DIR:-"experiments/motivation1/llm/results"}
N_TASKS=${N_TASKS:-"4,6,8,10"}
STRATEGIES=${STRATEGIES:-"task_sharing,deploy_sharing"}
BENCHMARK_MODE=${BENCHMARK_MODE:-"closed_loop"}
CONCURRENCY=${CONCURRENCY:-1}
TARGET_RPS=${TARGET_RPS:-2.0}
TOTAL_RPS=${TOTAL_RPS:-20.0}
MAX_SAMPLES=${MAX_SAMPLES:-50}
UNIFORM_MAX_NEW_TOKENS=${UNIFORM_MAX_NEW_TOKENS:-64}
PROMPT_SOURCE_TASK=${PROMPT_SOURCE_TASK:-"ag_news"}

echo "=========================================="
echo "  Motivation LLM Experiment #1"
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
echo "=========================================="


# Run each (strategy, n_tasks) pair in its own fresh Python process for full GPU isolation
IFS=',' read -ra STRATEGY_LIST <<< "${STRATEGIES}"
IFS=',' read -ra N_TASKS_LIST <<< "${N_TASKS}"
for N in "${N_TASKS_LIST[@]}"; do
    for STRATEGY in "${STRATEGY_LIST[@]}"; do
        echo ""
        echo ">>> [run.sh] strategy=${STRATEGY} n_tasks=${N} — fresh Python process"
        python experiments/motivation1/llm/run.py \
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