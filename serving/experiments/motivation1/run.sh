#!/bin/bash
# Motivation Experiment #2 — task_sharing vs deploy_sharing benchmark
# Run from serving/ directory.
#
# Benchmark modes:
#   BENCHMARK_MODE=closed_loop  (default) — fixed CONCURRENCY workers per task,
#                                           fire as fast as possible → shows max throughput
#   BENCHMARK_MODE=open_loop              — Poisson arrivals at TARGET_RPS per task

set -e
cd "$(dirname "$0")/../.."  # go to serving/

PHASE_DURATION=${PHASE_DURATION:-60}
EXP_DIR=${EXP_DIR:-"experiments/motivation2/results"}
N_TASKS=${N_TASKS:-"2,4,8,10"}
STRATEGIES=${STRATEGIES:-"deploy_sharing,task_sharing"}
BENCHMARK_MODE=${BENCHMARK_MODE:-"closed_loop"}
# closed_loop params
CONCURRENCY=${CONCURRENCY:-8}
# open_loop params
TARGET_RPS=${TARGET_RPS:-8.0}
# total_rps params (fixed total load split evenly across N tasks)
TOTAL_RPS=${TOTAL_RPS:-48.0}
# server batcher params
SERVER_MAX_BATCH_SIZE=${SERVER_MAX_BATCH_SIZE:-32}
SERVER_MAX_BATCH_WAIT_MS=${SERVER_MAX_BATCH_WAIT_MS:-10}

echo "=========================================="
echo "  Motivation Experiment #2"
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
echo "  Server batch cfg: size=${SERVER_MAX_BATCH_SIZE}, wait=${SERVER_MAX_BATCH_WAIT_MS} ms"
echo "  Results         : ${EXP_DIR}"
echo "=========================================="

python experiments/motivation2/run.py \
    --n-tasks "${N_TASKS}" \
    --duration "${PHASE_DURATION}" \
    --exp-dir "${EXP_DIR}" \
    --strategies "${STRATEGIES}" \
    --benchmark-mode "${BENCHMARK_MODE}" \
    --concurrency "${CONCURRENCY}" \
    --target-rps "${TARGET_RPS}" \
    --total-rps "${TOTAL_RPS}" \
    --server-max-batch-size "${SERVER_MAX_BATCH_SIZE}" \
    --server-max-batch-wait-ms "${SERVER_MAX_BATCH_WAIT_MS}"
