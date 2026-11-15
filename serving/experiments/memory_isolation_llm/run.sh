#!/usr/bin/env bash
# memory_isolation_llm — a load surge from tenant B vs. tenant A's KV cache
# and throughput, equal-sized requests: S-BE, BE, SP (see run.py).
#
# Run from serving/:
#   bash experiments/memory_isolation_llm/run.sh
#
# Overrides:
#   MODES=s_be,be,sp RATE_A=8 RATE_B=8 RATE_B_MAX=100 RAMP_AT=15 RAMP_S=20 \
#   WARMUP=10 DURATION=60 PROMPT_LEN=128 OUT_MIN=32 OUT_MAX=64 \
#   KV_TOKENS=10240 ISO_KV_TOKENS=5120 BATCH_CAP=256 GPU_MEM=0.25 ISO_GPU_MEM=0.20 \
#   EXP_DIR=experiments/memory_isolation_llm/results_l40s bash experiments/memory_isolation_llm/run.sh
#
# KV pools are set exactly (KV_TOKENS for s_be, ISO_KV_TOKENS per be/sp
# engine, default half); each engine prints its pool at startup. GPU_MEM /
# ISO_GPU_MEM only need room for weights, activations and that pool.
#
# SP needs libsmctrl (TPC_LIB_DIR, default <repo parent>/TPC_controller/tpc_controller).

set -euo pipefail

SERVING_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SERVING_DIR"
EXP="experiments/memory_isolation_llm"

CONDA_ENV="${CONDA_ENV:-}"
if [[ -n "$CONDA_ENV" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

EXP_DIR="${EXP_DIR:-$EXP/results_l40s}"
mkdir -p "$EXP_DIR/logs"

python -u "$EXP/run.py" \
    --modes                  "${MODES:-s_be,be,sp}" \
    --model                  "${MODEL:-Qwen/Qwen2.5-3B}" \
    --rate-a                 "${RATE_A:-8}" \
    --rate-b                 "${RATE_B:-8}" \
    --rate-b-max             "${RATE_B_MAX:-100}" \
    --ramp-at                "${RAMP_AT:-15}" \
    --ramp-s                 "${RAMP_S:-20}" \
    --warmup                 "${WARMUP:-10}" \
    --duration               "${DURATION:-60}" \
    --prompt-len             "${PROMPT_LEN:-128}" \
    --out-min                "${OUT_MIN:-32}" \
    --out-max                "${OUT_MAX:-64}" \
    --kv-tokens              "${KV_TOKENS:-10240}" \
    ${ISO_KV_TOKENS:+--iso-kv-tokens "$ISO_KV_TOKENS"} \
    --batch-cap              "${BATCH_CAP:-256}" \
    --gpu-memory-utilization "${GPU_MEM:-0.25}" \
    --iso-gpu-mem            "${ISO_GPU_MEM:-0.20}" \
    --exp-dir                "$EXP_DIR" \
    2>&1 | tee "$EXP_DIR/logs/run.log"

python "$EXP/plot.py" --exp-dir "$EXP_DIR"
