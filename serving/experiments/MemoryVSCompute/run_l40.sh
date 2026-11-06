#!/bin/bash
# Capacity search on the L40S (48 GB, 142 SMs / 71 TPCs, ~864 GB/s).
#
#   bash experiments/MemoryVSCompute/run_l40.sh
#
# See run_a10.sh for how the starting points are derived. Compute scaled from the
# A10's MEASURED momentlarge sharing capacity of 45 by the swinlarge ratio
# L40S/A10 = 33/18 = 1.83, giving ~82. Memory doubles with the 48 GB card, so the
# no_sharing wall should be roughly twice the A10's.
#
# NOTE: sharing may run past 96 tasks here. With only 10 base tsfm tasks that is
# 10+ __appN copies of each, all on one server. If latency stops responding to n,
# check MAX_BATCH_SIZE (default 32) before trusting the number — the batcher, not
# the GPU, may have become the limit.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_device_guard.sh"
check_gpu "L40" "l40"

BACKBONE="${BACKBONE:-momentlarge}"
DEVICE=l40
SEARCH="python3 ${HERE}/search.py --backbone ${BACKBONE} --device ${DEVICE}"

$SEARCH --conditions sharing         --start 68  --max-ntasks 160 --grow 1.3
$SEARCH --conditions no_sharing      --start 14  --max-ntasks 40
$SEARCH --conditions no_sharing_tpc  --start 14  --max-ntasks 40

python3 "${HERE}/plot.py"
