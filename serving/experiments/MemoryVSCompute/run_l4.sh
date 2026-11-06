#!/bin/bash
# Capacity search on the L4 (24 GB, 58 SMs / 29 TPCs, ~300 GB/s).
#
#   bash experiments/MemoryVSCompute/run_l4.sh
#
# See run_a10.sh for how the starting points are derived. Compute scaled from the
# A10's MEASURED momentlarge sharing capacity of 45 by the swinlarge ratio
# L4/A10 = 12/18 = 0.67, giving ~30. Memory is unchanged —
# same 24 GB card, same ~2.06 GiB per process — so the no_sharing wall should
# land near the A10's, and on this GPU compute may well bind first (as it did
# for swinlarge, where L4 was the one compute-limited BE result).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_device_guard.sh"
check_gpu "L4" "l4"

BACKBONE="${BACKBONE:-momentlarge}"
DEVICE=l4
SEARCH="python3 ${HERE}/search.py --backbone ${BACKBONE} --device ${DEVICE}"

$SEARCH --conditions sharing         --start 24  --max-ntasks 64
$SEARCH --conditions no_sharing      --start 8   --max-ntasks 24
$SEARCH --conditions no_sharing_tpc  --start 8   --max-ntasks 24

python3 "${HERE}/plot.py"
