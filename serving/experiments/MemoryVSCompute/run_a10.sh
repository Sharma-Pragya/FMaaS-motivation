#!/bin/bash
# Capacity search on the A10 (24 GB, 72 SMs / 36 TPCs, ~600 GB/s).
#
#   bash experiments/MemoryVSCompute/run_a10.sh
#
# Runs all three methods through search.py and plots at the end. Probes already
# on disk are reused, so this is safe to re-run or resume after an interrupt.
#
# BACKBONE defaults to momentlarge; override to reuse this for another model.
#
# Starting points are priors, not constraints — if one is wrong the bracket just
# keeps doubling. They come from measured data:
#   sharing     MEASURED: 45. The knee is abrupt — 44 -> 143.9 ms, 45 -> 195.8 ms,
#               46 -> 1263.5 ms. Linear extrapolation from n=20/40 predicted ~95
#               and was badly wrong; these curves go vertical, they do not bend.
#   no_sharing  MOMENT-large 341M params -> ~2.06 GiB/process against the
#               swinlarge-measured 1.375 GiB that put the A10 wall at 16
#               => expect a memory wall near 10-11
#   no_sharing_tpc  36 TPCs; swinlarge managed 4. momentlarge is cheaper per
#               request, but the same memory wall caps it, so expect 6-8.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_device_guard.sh"
check_gpu "A10" "a10"

BACKBONE="${BACKBONE:-momentlarge}"
DEVICE=a10
SEARCH="python3 ${HERE}/search.py --backbone ${BACKBONE} --device ${DEVICE}"

# sharing first: one device server, so cheapest and least likely to leave
# orphaned processes holding GPU memory for the runs that follow.
$SEARCH --conditions sharing         --start 40  --max-ntasks 96
$SEARCH --conditions no_sharing      --start 8   --max-ntasks 24
$SEARCH --conditions no_sharing_tpc  --start 8   --max-ntasks 24

python3 "${HERE}/plot.py"
