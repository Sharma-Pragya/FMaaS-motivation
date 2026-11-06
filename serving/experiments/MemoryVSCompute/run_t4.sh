#!/bin/bash
# Capacity search on the T4 (16 GB, 40 SMs / 20 TPCs, ~320 GB/s, Turing).
#
#   bash experiments/MemoryVSCompute/run_t4.sh                    # momentlarge
#   BACKBONE=swinlarge bash experiments/MemoryVSCompute/run_t4.sh # swinlarge
#   DRY_RUN=1 bash experiments/MemoryVSCompute/run_t4.sh          # show probes, run nothing
#
# Bracket starts are per-backbone. swinlarge's capacity on a given GPU is roughly
# 2.5x lower than momentlarge's (measured on the A10: 18 vs 45 for sharing), so
# one shared set of starts would overshoot badly for swinlarge — the first probe
# would fail and the search would bisect down from zero, wasting probes.
#
# Starts are ~0.8x the expected answer so the first probe is likely to PASS,
# which gives the bracket a lower bound immediately. Being wrong is cheap: the
# bracket just doubles (or halves) until it finds the crossing.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_device_guard.sh"
check_gpu "T4" "t4"

BACKBONE="${BACKBONE:-momentlarge}"
DEVICE=t4

case "$BACKBONE" in
    momentlarge)
        # Measured on this T4: sharing 14 (n=15 fails at 307 ms), SP bracketed
        # 3..5. no_sharing has only failures on disk (12 and 16 are INFEASIBLE),
        # so its start is unused — the search bisects the known 0..12 bracket.
        SHARING_ARGS="--start 12 --max-ntasks 48"
        NS_ARGS="--start 6  --max-ntasks 24"
        TPC_ARGS="--start 4  --max-ntasks 16"
        ;;
    swinlarge)
        # Measured on this T4: no_sharing n=5 = 199 ms (1 ms under the SLO),
        # sharing n=5 = 122 ms, SP INFEASIBLE at n=4 (TPC starvation, requests
        # time out rather than OOM) so its answer is <= 3.
        SHARING_ARGS="--start 7 --max-ntasks 24"
        NS_ARGS="--start 6 --max-ntasks 20"
        TPC_ARGS="--start 2 --max-ntasks 12"
        ;;
    *)
        echo "[run_t4.sh] NOTE: no tuned starts for BACKBONE='$BACKBONE' —" >&2
        echo "[run_t4.sh]       bracketing from n=4. Add a case above once measured." >&2
        SHARING_ARGS="--start 4 --max-ntasks 48"
        NS_ARGS="--start 4 --max-ntasks 24"
        TPC_ARGS="--start 2 --max-ntasks 16"
        ;;
esac

SEARCH="python3 ${HERE}/search.py --backbone ${BACKBONE} --device ${DEVICE}"
[[ -n "${DRY_RUN:-}" ]] && SEARCH="$SEARCH --dry-run"

echo "[run_t4.sh] backbone=${BACKBONE}  device=${DEVICE}"

# sharing first: one device server, so cheapest and least likely to leave
# orphaned processes holding GPU memory for the runs that follow.
$SEARCH --conditions sharing        $SHARING_ARGS
$SEARCH --conditions no_sharing     $NS_ARGS
$SEARCH --conditions no_sharing_tpc $TPC_ARGS

[[ -z "${DRY_RUN:-}" ]] && python3 "${HERE}/plot.py"
