#!/usr/bin/env bash
# memory_isolation_vision — a load surge from task B, with requests staged on
# the GPU at arrival, vs. the other tasks' service: S-BE, BE, FMVisor (see run.py).
# SP is also available (MODES=s_be,be,sp,fmvisor) but not plotted by default.
# Defaults follow RTVSntask/vision (DINOv2-base, 224 px, 8 tasks, max batch
# 32), at 8 req/s per task (64 req/s in all): meant to sit between what BE
# serves (~50 req/s, RTVSntask) and what one shared backbone serves (~75 req/s
# at batch 32), so S-BE and FMVisor keep up before the surge and BE does not.
# Those capacities are estimates: check each mode's victims' req/s before the
# surge and adjust RATE_A / RATE_B. FM_CHUNK: FMVisor's slots per allocation (1: per request, as S-BE/BE).
#
# Run from serving/:
#   bash experiments/memory_isolation_vision/run.sh
#
# Overrides:
#   MODES=s_be,be,fmvisor N_TASKS=8 BACKBONE=dino-base RES=224 STAGING=gpu MAX_BATCH=32 \
#   FM_CHUNK=1 RATE_A=8 RATE_B=8 RATE_B_MAX=1000 RAMP_AT=15 RAMP_S=15 WARMUP=10 \
#   DURATION=60 CUDA_DEVICE=cuda:0 EXP_DIR=experiments/memory_isolation_vision/results_t4_n8_r8 \
#   TPC_LIB_DIR=... bash experiments/memory_isolation_vision/run.sh
#
# MIG (RTX PRO 6000 Blackwell: 4 x 1g.24gb, so N_TASKS=4): create the slices
# first (root),
#   sudo nvidia-smi -i 0 -mig 1
#   sudo nvidia-smi mig -i 0 -cgi 1g.24gb,1g.24gb,1g.24gb,1g.24gb -C
# and remove them after: sudo nvidia-smi mig -dci && sudo nvidia-smi mig -dgi
#   && sudo nvidia-smi -i 0 -mig 0
# then MODES=mig ...; MIG_UUIDS (comma-separated) defaults to every slice
# `nvidia-smi -L` lists. Run the other modes with MIG off, into the same
# EXP_DIR: the plot takes the latest run of each mode. PLOT_MODES picks what
# the plot draws (e.g. s_be,mig).
#
# BFQ_RATES (FMVisor): BFQ declared rates task:rate,... (weight = 1 / rate),
# e.g. A1:1,B:3 favours the victim 3:1; default: each task's base load rate.
# SP_SPLIT (SP / SP (MPS)): share of the GPU per task, task:weight,...; set it
# to the same ratio as BFQ_RATES (A1:3,B:1 for A1:1,B:3), as fair_share does.
#
# STAGING=host is the control: inputs wait in host RAM until their batch runs.
# SP needs libsmctrl (TPC_LIB_DIR, default ../../TPC_controller/tpc_controller).
# Each serving process prints its warm-up times per batch size at startup.

set -euo pipefail

SERVING_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SERVING_DIR"
EXP="experiments/memory_isolation_vision"

CONDA_ENV="${CONDA_ENV:-fmtk}"
EXP_DIR="${EXP_DIR:-$EXP/results_t4_n8_r8}"
STAGING="${STAGING:-gpu}"

# FMTK provides the backbone and heads (same default as isolation_mechanisms).
FMTK_DIR="${FMTK_DIR:-$(cd "$SERVING_DIR/../.." && pwd)/FMTK}"
if [[ ! -d "$FMTK_DIR/src/fmtk" ]]; then
    echo "ERROR: FMTK not found at $FMTK_DIR (set FMTK_DIR)"
    exit 1
fi
export PYTHONPATH="${FMTK_DIR}/src:${PYTHONPATH:-}"

if command -v conda &> /dev/null; then
    PYTHON="${PYTHON:-conda run --no-capture-output -n ${CONDA_ENV} python}"
else
    PYTHON="${PYTHON:-python}"
fi

mkdir -p "$EXP_DIR/logs"

$PYTHON -u "$EXP/run.py" \
    --modes          "${MODES:-s_be,be,fmvisor}" \
    --n-tasks        "${N_TASKS:-8}" \
    --backbone       "${BACKBONE:-dino-base}" \
    --res            "${RES:-224}" \
    --staging        "$STAGING" \
    --max-batch      "${MAX_BATCH:-32}" \
    --fm-chunk       "${FM_CHUNK:-1}" \
    ${MIG_UUIDS:+--mig-uuids "$MIG_UUIDS"} \
    ${BFQ_RATES:+--bfq-rates "$BFQ_RATES"} \
    ${SP_SPLIT:+--sp-split "$SP_SPLIT"} \
    --rate-a         "${RATE_A:-8}" \
    --rate-b         "${RATE_B:-8}" \
    --rate-b-max     "${RATE_B_MAX:-1000}" \
    --ramp-at        "${RAMP_AT:-15}" \
    --ramp-s         "${RAMP_S:-15}" \
    --warmup         "${WARMUP:-10}" \
    --duration       "${DURATION:-60}" \
    --device         "${CUDA_DEVICE:-cuda:0}" \
    --exp-dir        "$EXP_DIR" \
    2>&1 | tee "$EXP_DIR/logs/run_${STAGING}.log"

$PYTHON "$EXP/plot.py" --exp-dir "$EXP_DIR" --staging "$STAGING" \
    --modes "${PLOT_MODES:-s_be,sp,fmvisor}"
