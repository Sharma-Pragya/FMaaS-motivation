#!/bin/bash
# MemoryVSCompute/run.sh — capacity sweeps across backbones and GPUs.
#
# Answers: how many concurrent tasks does each method sustain under a latency
# SLO, and does it stop because the GPU ran out of compute or out of memory?
#
# This is a dispatcher, not a runner. It picks the RTVSntask runner that matches
# the backbone's task set and hands it a results directory under this folder:
#
#   vision backbones (swinlarge, dinobase-patch, ...) -> RTVSntask/vision/run.sh
#   tsfm   backbones (momentlarge, momentbase)        -> RTVSntask/tpc/run.sh
#
# Results land in:
#   MemoryVSCompute/results/<backbone>/<device>/ntasks_<N>/rps_<R>/<condition>/
#
# BACKBONE and DEVICE are required — there is no default. RESULTS_BASE is
# computed from them and must not be set by hand: the runners default it to a
# path inside RTVSntask, so a wrapper that forgets to override it writes into
# the frozen results tree.
#
# DEVICE is a label for the results path only (a10, l4, l40, t4). It does not
# select a GPU — use CUDA_DEVICE for that.
#
# Every other variable passes through to the underlying runner untouched:
#   NUM_TASKS_SWEEP  RPS_SWEEP  PHASE_DURATION  CONDITIONS  CUDA_DEVICE
#   MAX_BATCH_SIZE   TPC_MODE   DEVICE_PORT     WARMUP_BURST_SECS  ...
#
# Usage:
#   BACKBONE=swinlarge   DEVICE=a10 NUM_TASKS_SWEEP=15,16 RPS_SWEEP=5 \
#       bash experiments/MemoryVSCompute/run.sh
#
#   BACKBONE=momentlarge DEVICE=a10 NUM_TASKS_SWEEP=2,4,6 RPS_SWEEP=5 \
#       CONDITIONS=sharing,no_sharing bash experiments/MemoryVSCompute/run.sh
#
#   DRY_RUN=1 BACKBONE=momentlarge DEVICE=l4 bash experiments/MemoryVSCompute/run.sh
#       — print the dispatch and exit without launching anything.

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RTVS_DIR="$(cd "${EXP_DIR}/../RTVSntask" && pwd)"

# ---------------------------------------------------------------------------
# Backbone -> task set. Explicit rather than pattern-matched: an unrecognised
# backbone must fail loudly, not get routed to whichever runner a substring
# happens to match.
# ---------------------------------------------------------------------------
backbone_task_set() {
    case "$1" in
        swinlarge|swinbase|dinobase-patch|dinobase|dinolarge) echo "vision" ;;
        momentlarge|momentbase)                               echo "tsfm"   ;;
        *)                                                    echo ""       ;;
    esac
}

# Which runner serves each task set.
runner_for() {
    case "$1" in
        vision) echo "${RTVS_DIR}/vision/run.sh" ;;
        tsfm)   echo "${RTVS_DIR}/tpc/run.sh"    ;;
    esac
}

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [[ -z "${BACKBONE:-}" || -z "${DEVICE:-}" ]]; then
    echo "ERROR: BACKBONE and DEVICE are both required." >&2
    echo "  e.g. BACKBONE=swinlarge DEVICE=a10 NUM_TASKS_SWEEP=15,16 RPS_SWEEP=5 \\" >&2
    echo "       bash experiments/MemoryVSCompute/run.sh" >&2
    exit 1
fi

TASK_SET_RESOLVED="$(backbone_task_set "$BACKBONE")"
if [[ -z "$TASK_SET_RESOLVED" ]]; then
    echo "ERROR: unknown BACKBONE '$BACKBONE' — cannot tell which task set it belongs to." >&2
    echo "  vision: swinlarge, swinbase, dinobase-patch, dinobase, dinolarge" >&2
    echo "  tsfm:   momentlarge, momentbase" >&2
    echo "  (add it to backbone_task_set() in $0 if it is new)" >&2
    exit 1
fi

# TASK_SET is derived from BACKBONE. Honouring an inherited one would let a
# stale export silently route a vision backbone through the tsfm branch.
if [[ -n "${TASK_SET:-}" && "$TASK_SET" != "$TASK_SET_RESOLVED" ]]; then
    echo "ERROR: TASK_SET='$TASK_SET' contradicts BACKBONE='$BACKBONE' (implies '$TASK_SET_RESOLVED')." >&2
    echo "  Unset TASK_SET — this script derives it from BACKBONE." >&2
    exit 1
fi

if [[ -n "${RESULTS_BASE:-}" ]]; then
    echo "ERROR: do not set RESULTS_BASE — it is derived from BACKBONE and DEVICE." >&2
    exit 1
fi

# Device labels are directory names, and the filesystem is case-sensitive, so
# DEVICE=T4 and DEVICE=t4 are two unrelated result trees. Normalise rather than
# reject: the intent is never "make me a second, differently-cased device".
DEVICE_RAW="$DEVICE"
DEVICE="$(printf '%s' "$DEVICE" | tr '[:upper:]' '[:lower:]')"
if [[ "$DEVICE" != "$DEVICE_RAW" ]]; then
    echo "[MemoryVSCompute] NOTE: DEVICE='$DEVICE_RAW' normalised to '$DEVICE'" >&2
    echo "[MemoryVSCompute]       (results directories are lowercase by convention)" >&2
fi

RUNNER="$(runner_for "$TASK_SET_RESOLVED")"
if [[ ! -f "$RUNNER" ]]; then
    echo "ERROR: runner not found: $RUNNER" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
# Absolute, so it survives the runner's internal `cd "$SERVING_DIR"`.
RESULTS_BASE="${EXP_DIR}/results/${BACKBONE}/${DEVICE}"

export TASK_SET="$TASK_SET_RESOLVED"
export BACKBONE
export RESULTS_BASE

echo "================================================================"
echo "  MemoryVSCompute"
echo "  Backbone  : $BACKBONE  (task set: $TASK_SET)"
echo "  Device    : $DEVICE    (label only; CUDA_DEVICE=${CUDA_DEVICE:-cuda:0})"
echo "  Runner    : ${RUNNER#${RTVS_DIR}/}"
echo "  ntasks    : ${NUM_TASKS_SWEEP:-<runner default>}"
echo "  rps       : ${RPS_SWEEP:-<runner default>}"
echo "  duration  : ${PHASE_DURATION:-<runner default>}s"
echo "  conditions: ${CONDITIONS:-<runner default: all three>}"
echo "  Results   : $RESULTS_BASE"
echo "================================================================"

if [[ -n "${DRY_RUN:-}" ]]; then
    echo "[MemoryVSCompute] DRY_RUN set — not launching. Would exec:"
    echo "    bash $RUNNER"
    exit 0
fi

# Created only for a real run — a dry run must leave no trace on disk.
mkdir -p "$RESULTS_BASE"

exec bash "$RUNNER"
