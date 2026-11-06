#!/usr/bin/env python3
"""MemoryVSCompute/search.py — find each method's task capacity without sweeping by hand.

For every condition, finds the largest ntasks that meets the SLO by bracketing
geometrically upward and then bisecting, instead of walking n one at a time.

    phase 1  bracket   4 -> 8 -> 16 -> 32 ...   until a probe fails
    phase 2  bisect    halve the surviving gap until it is one task wide
    capacity           the largest n that passed

Probes already on disk are free: the predicate reads existing results first and
only launches run.sh for values never measured. Re-running after adding points
by hand costs nothing for what is already there.

WHY THE PREDICATE IS NOT JUST LATENCY
    A run counts as passing only if worst-task mean latency <= threshold AND it
    delivered >= 99% of its offered requests. Latency alone is not monotonic in
    n: once servers hit CUDA OOM they shed requests, the client drops the failed
    RPCs, and the surviving samples are biased fast. On the A10, no_sharing
    reports 164 / 142 / 113 / 75 ms at n = 15 / 16 / 17 / 18 while delivery
    falls 100% -> 94% -> 73% -> 33%. Bisecting on latency would confidently
    return 18. The delivery gate is what makes the predicate monotonic, and
    monotonicity is what makes bisection valid.

ONE DURATION PER SEARCH
    PHASE_DURATION is deliberately not varied between bracketing and
    confirmation. run.py caches trace.json per ntasks/rps directory and shares
    it across all three conditions, consulting --duration only when generating a
    new one — so a 600 s probe silently replays a 300 s trace if one exists.
    Working around that means deleting traces mid-search, which would break
    comparability between conditions at the same n. Cheap short probes are not
    worth results you cannot defend.

Usage:
    python experiments/MemoryVSCompute/search.py --backbone momentlarge --device a10
    python experiments/MemoryVSCompute/search.py --backbone momentlarge --device a10 \\
        --conditions sharing --start 8 --max-ntasks 48
    python experiments/MemoryVSCompute/search.py --backbone momentlarge --device a10 --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

# From this experiment's own plot.py, not RTVSntask's — the search predicate and
# the figures have to agree on what passes, and this experiment requires complete
# delivery where the RTVSntask sweeps allow 1% loss.
from plot import (                                 # noqa: E402
    MIN_DELIVERY,
    SERIES_LABELS,
    failed,
    read_run,
)

RESULTS_DIR = EXP_DIR / "results"
RUN_SH = EXP_DIR / "run.sh"
CONDITIONS_ALL = ["no_sharing_tpc", "no_sharing", "sharing"]

# Per-condition bracket starts, filled from --start-per-condition in main().
START: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# Probe evaluation
# ---------------------------------------------------------------------------

def run_dir_for(backbone: str, device: str, cond: str, n: int, rps: float) -> Path:
    return RESULTS_DIR / backbone / device / f"ntasks_{n}" / f"rps_{int(rps)}" / cond


def measured(run_dir: Path, n: int, rps: float, duration: float) -> Optional[bool]:
    """True/False if this config has a verdict on disk, None if never run.

    An infeasible marker is a verdict: the backbone copies did not fit, which is
    a failure at that n, not missing data.
    """
    if (run_dir / "infeasible.json").is_file():
        return False
    rec = read_run(run_dir, n, rps, duration)
    if rec is None:
        return None
    return not failed(rec, THRESHOLD, MIN_DELIVERY)


def describe(run_dir: Path, n: int, rps: float, duration: float) -> str:
    if (run_dir / "infeasible.json").is_file():
        return "INFEASIBLE (did not fit)"
    rec = read_run(run_dir, n, rps, duration)
    if rec is None:
        return "no data"
    return f"{rec[0]:.0f} ms at {rec[1]*100:.1f}% delivered"


def launch(backbone: str, device: str, cond: str, n: int, rps: float,
           duration: float, dry_run: bool) -> None:
    """Run one (condition, ntasks) probe through run.sh.

    RESULTS_BASE is never set here — run.sh derives it and refuses an override.
    """
    env = dict(os.environ)
    env.update({
        "BACKBONE":        backbone,
        "DEVICE":          device,
        "CONDITIONS":      cond,
        "NUM_TASKS_SWEEP": str(n),
        "RPS_SWEEP":       str(int(rps)),
        "PHASE_DURATION":  str(int(duration)),
    })
    env.pop("RESULTS_BASE", None)
    env.pop("TASK_SET", None)

    if dry_run:
        print(f"      [dry-run] BACKBONE={backbone} DEVICE={device} CONDITIONS={cond} "
              f"NUM_TASKS_SWEEP={n} RPS_SWEEP={int(rps)} PHASE_DURATION={int(duration)} "
              f"bash {RUN_SH.name}")
        return

    t0 = time.time()
    # run.sh warns and continues on a failed condition rather than exiting, so
    # its return code is not a verdict — the result directory is.
    subprocess.run(["bash", str(RUN_SH)], env=env, check=False)
    print(f"      (probe took {(time.time() - t0) / 60:.1f} min)")


def probe(backbone: str, device: str, cond: str, n: int, rps: float,
          duration: float, dry_run: bool) -> Optional[bool]:
    """Verdict for one n, running it only if it has never been measured."""
    rd = run_dir_for(backbone, device, cond, n, rps)
    verdict = measured(rd, n, rps, duration)
    if verdict is not None:
        print(f"    n={n:<3} cached  -> {'PASS' if verdict else 'FAIL'}  "
              f"({describe(rd, n, rps, duration)})")
        return verdict

    print(f"    n={n:<3} running ...")
    launch(backbone, device, cond, n, rps, duration, dry_run)
    if dry_run:
        return None

    verdict = measured(rd, n, rps, duration)
    if verdict is None:
        print(f"    n={n:<3} ERROR: run produced no result at {rd}", file=sys.stderr)
        return None
    print(f"    n={n:<3} -> {'PASS' if verdict else 'FAIL'}  "
          f"({describe(rd, n, rps, duration)})")
    return verdict


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def find_capacity(backbone: str, device: str, cond: str, rps: float, duration: float,
                  start: int, max_n: int, dry_run: bool,
                  confirm: bool = True, grow: float = 2.0) -> Tuple[Optional[int], str]:
    """Largest passing n. Returns (capacity, note)."""
    print(f"\n  --- {SERIES_LABELS[cond]} ({cond}) ---")

    lo = 0          # largest known PASS
    hi: Optional[int] = None   # smallest known FAIL

    # Seed from anything already on disk so a rerun costs nothing.
    d = RESULTS_DIR / backbone / device
    if d.is_dir():
        for nd in sorted(d.glob("ntasks_*"), key=lambda p: int(p.name.split("_")[1])):
            n = int(nd.name.split("_")[1])
            v = measured(run_dir_for(backbone, device, cond, n, rps), n, rps, duration)
            if v is True:
                lo = max(lo, n)
            elif v is False and (hi is None or n < hi):
                hi = n
    if lo or hi:
        print(f"    seeded from disk: largest pass={lo or '-'}, smallest fail={hi or '-'}")

    # Phase 1 — bracket upward until something fails.
    #
    # An explicit `start` above the largest known pass is a caller's prior and is
    # honoured as-is: overriding it with lo*grow would overshoot a tuned start and
    # cost probes on the way back down. Only when `start` is already known to pass
    # do we need to invent the next value ourselves.
    n = start if start > lo else max(lo + 1, int(lo * grow))
    while hi is None:
        if n > max_n:
            return (lo or None), f"hit --max-ntasks={max_n} without failing; capacity >= {lo}"
        v = probe(backbone, device, cond, n, rps, duration, dry_run)
        if v is None:
            if dry_run:
                return (lo or None), f"dry-run: would start probing at n={n}"
            return (lo or None), "aborted: probe produced no result"
        if v:
            lo = max(lo, n)
            # Gentler than doubling when `grow` is small: at n~80 a 600 s probe
            # costs ~11 min, so overshooting to 160 to prove a failure wastes
            # far more than the extra bisection step it saves.
            n = max(n + 1, int(n * grow))
        else:
            hi = n

    # Phase 2 — bisect the gap.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        v = probe(backbone, device, cond, mid, rps, duration, dry_run)
        if v is None:
            if dry_run:
                return (lo or None), f"dry-run: would next probe n={mid} (bracket {lo}..{hi})"
            return (lo or None), "aborted: probe produced no result"
        if v:
            lo = mid
        else:
            hi = mid

    # Phase 3 — fill the point just below the crossing. Bisection lands on
    # whatever n the halving happens to pick, so n-1 is usually never run; the
    # capacity plot does not need it but the latency-vs-ntasks curve does, and
    # a reader wants to see the trend on both sides of the crossing.
    if confirm and lo > 1:
        below = lo - 1
        rd = run_dir_for(backbone, device, cond, below, rps)
        if measured(rd, below, rps, duration) is None:
            print(f"    filling neighbourhood point n={below} (for the curve, "
                  f"not the capacity)")
            probe(backbone, device, cond, below, rps, duration, dry_run)

    return lo, f"crossing measured: n={lo} passes, n={hi} fails"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

THRESHOLD = 200.0   # set from args in main(); module-level for measured()


def main() -> int:
    global THRESHOLD
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--device", required=True, help="results-path label (a10, l4, l40)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS_ALL),
                    help="comma-separated (default: all three)")
    ap.add_argument("--rps", type=float, default=5)
    ap.add_argument("--threshold", type=float, default=200.0, help="mean-latency SLO in ms")
    ap.add_argument("--duration", type=float, default=600.0,
                    help="PHASE_DURATION for every probe (kept constant on purpose)")
    ap.add_argument("--start", type=int, default=4,
                    help="first n to probe (default: 4). Starting near the expected "
                         "answer removes bracketing probes at no risk to the result.")
    ap.add_argument("--start-per-condition", default=None,
                    help="per-condition starts, e.g. 'no_sharing=10,sharing=20'. "
                         "Overrides --start for the named conditions.")
    ap.add_argument("--grow", type=float, default=2.0,
                    help="bracket growth factor (default: 2.0 = doubling). Use ~1.3 "
                         "when probes are expensive and the prior is trusted.")
    ap.add_argument("--no-confirm", action="store_true",
                    help="skip filling n-1 after the crossing is found")
    ap.add_argument("--max-ntasks", type=int, default=48,
                    help="refuse to probe above this (default: 48). Guards against "
                         "launching a huge number of device servers.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the probes that would run, launch nothing")
    args = ap.parse_args()

    THRESHOLD = args.threshold
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    global START
    START = {}
    if args.start_per_condition:
        for item in args.start_per_condition.split(","):
            if "=" not in item:
                print(f"ERROR: --start-per-condition entry '{item}' is not cond=n",
                      file=sys.stderr)
                return 1
            k, v = item.split("=", 1)
            k = k.strip()
            if k not in CONDITIONS_ALL:
                print(f"ERROR: unknown condition '{k}' in --start-per-condition",
                      file=sys.stderr)
                return 1
            START[k] = int(v)
    bad = [c for c in conds if c not in CONDITIONS_ALL]
    if bad:
        print(f"ERROR: unknown condition(s): {', '.join(bad)}", file=sys.stderr)
        return 1

    print("=" * 64)
    print(f"  capacity search — {args.backbone} on {args.device}")
    print(f"  SLO {args.threshold:g} ms mean + >={MIN_DELIVERY*100:g}% delivery, "
          f"rps={args.rps:g}, duration={args.duration:g}s")
    print(f"  bracket from n={args.start}, ceiling n={args.max_ntasks}")
    print("=" * 64)

    results: Dict[str, Tuple[Optional[int], str]] = {}
    for cond in conds:
        results[cond] = find_capacity(args.backbone, args.device, cond, args.rps,
                                      args.duration, START.get(cond, args.start),
                                      args.max_ntasks, args.dry_run,
                                      confirm=not args.no_confirm, grow=args.grow)

    print("\n" + "=" * 64)
    print(f"  RESULT — {args.backbone} on {args.device}")
    for cond in conds:
        cap, note = results[cond]
        print(f"    {SERIES_LABELS[cond]:<10} capacity={cap if cap is not None else '?'}   {note}")
    print("=" * 64)
    print("\nPlot with:  python experiments/MemoryVSCompute/plot.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
