#!/usr/bin/env python3
"""staging_ablation/run.py: does it matter WHERE a waiting request sits?

Question: if the shared-backbone baseline staged each request on the GPU when
it arrives (instead of keeping it on the CPU until its batch runs, as the
device server does today), would its latency change?

One process, one real backbone (loaded exactly as the device server loads it,
device.model_loader._build_pipeline), N tasks sending open-loop Poisson
traffic. Everything is identical between the two modes except where a
waiting request's input lives:

    host  request waits as a numpy array in host RAM. When its batch runs:
          np.concatenate -> torch.from_numpy -> backbone forward (which does
          the host->GPU copy). Same path as the device server today
          (batch_former._prepare_batch + runtime.run_backbone).
    gpu   request is copied to the GPU as soon as it arrives (on a separate
          copy stream, synchronized before it is queued). When its batch
          runs: torch.cat on the GPU -> backbone forward (no copy).

Batching follows the device server: one worker, FIFO across tasks, up to
--max-batch requests, --batch-window-ms collection window (default 0). Decoders
are skipped: they are the same in both modes, so they cannot explain a
difference. Arrival times come from one seeded stream per task, so both modes
see identical arrivals.

Latency = batch completion (backbone synchronized) - scheduled arrival time.

Usage (from serving/):
    python experiments/staging_ablation/run.py --backbone momentlarge \
        --rps 5,10,15,20 --modes host,gpu --duration 30
Output: <out-dir>/summary.csv (one row per backbone x rps x mode) and
        <out-dir>/requests_<backbone>_rps<r>_<mode>.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import torch  # noqa: E402

from device.model_loader import _build_pipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Inputs: one request = one sample, shaped as the sharing_benefit datasets.
# ---------------------------------------------------------------------------

def make_request(backbone: str, res: int, rng: np.random.Generator):
    """Return (x, mask) for ONE request, leading dim 1, as numpy (what the
    device server holds after decoding a gRPC request)."""
    if backbone.startswith("moment"):
        x = rng.standard_normal((1, 1, 512), dtype=np.float32)   # [B, channels, seq_len]
        mask = np.ones((1, 512), dtype=np.float64)                 # as ECG5000Dataset
        return x, mask
    # vision: [B, C, H, W]
    x = rng.standard_normal((1, 3, res, res), dtype=np.float32)
    return x, None


def poisson_arrivals(rate: float, duration: float, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    t, out = 0.0, []
    while True:
        t += rng.exponential(1.0 / rate)
        if t >= duration:
            return out
        out.append(t)


# ---------------------------------------------------------------------------
# One run: (backbone, rps, mode)
# ---------------------------------------------------------------------------

class Req:
    __slots__ = ("task", "rid", "t_sched", "x", "mask", "t_enq", "t_done")

    def __init__(self, task, rid, t_sched):
        self.task, self.rid, self.t_sched = task, rid, t_sched
        self.x = self.mask = None
        self.t_enq = self.t_done = None


def run_once(model, backbone: str, mode: str, rps: float, args, dev) -> tuple[dict, list[Req]]:
    n_tasks = args.n_tasks
    tasks = [f"task{i}" for i in range(n_tasks)]

    # Same arrivals and same input values for both modes.
    sched = []
    for i, t in enumerate(tasks):
        for k, at in enumerate(poisson_arrivals(rps, args.duration, seed=args.seed + 1000 * i)):
            sched.append((at, t, k))
    sched.sort()
    rng = np.random.default_rng(args.seed)
    proto = {t: make_request(backbone, args.res, rng) for t in tasks}

    queue: deque[Req] = deque()
    cv = threading.Condition()
    done_reqs: list[Req] = []
    arrivals_over = threading.Event()
    gpu_queued_bytes = [0, 0]  # current, peak (gpu mode only)

    copy_stream = torch.cuda.Stream(device=dev)
    comp_stream = torch.cuda.Stream(device=dev)
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    base_alloc = torch.cuda.memory_allocated(dev)

    t0 = time.perf_counter() + 0.2   # small start delay

    # --- arrival thread: plays the role of the server's infer() handler -----
    def arrivals():
        for at, task, k in sched:
            now = time.perf_counter() - t0
            if at > now:
                time.sleep(at - now)
            r = Req(task, k, at)
            x, m = proto[task]
            # A fresh copy, as decoding a gRPC request produces a new array.
            x, m = x.copy(), (m.copy() if m is not None else None)
            if mode == "gpu":
                with torch.cuda.stream(copy_stream):
                    r.x = torch.from_numpy(x).to(dev, non_blocking=False)
                    r.mask = torch.from_numpy(m).to(dev, non_blocking=False) if m is not None else None
                copy_stream.synchronize()   # input is on the GPU before it is queued
                nbytes = x.nbytes + (m.nbytes if m is not None else 0)
            else:
                r.x, r.mask = x, m
                nbytes = 0
            r.t_enq = time.perf_counter() - t0
            with cv:
                queue.append(r)
                if nbytes:
                    gpu_queued_bytes[0] += nbytes
                    gpu_queued_bytes[1] = max(gpu_queued_bytes[1], gpu_queued_bytes[0])
                cv.notify()
        arrivals_over.set()
        with cv:
            cv.notify()

    # --- worker: plays the role of the batcher + runtime.run_backbone -------
    batch_sizes = []

    def worker():
        window = args.batch_window_ms / 1000.0
        while True:
            with cv:
                while not queue and not arrivals_over.is_set():
                    cv.wait(timeout=0.1)
                if not queue and arrivals_over.is_set():
                    return
            if window > 0:
                deadline = time.perf_counter() + window
                while time.perf_counter() < deadline:
                    with cv:
                        if len(queue) >= args.max_batch:
                            break
                    time.sleep(0.0005)
            with cv:
                batch = [queue.popleft() for _ in range(min(args.max_batch, len(queue)))]
                if mode == "gpu":
                    gpu_queued_bytes[0] -= sum(
                        r.x.numel() * r.x.element_size()
                        + (r.mask.numel() * r.mask.element_size() if r.mask is not None else 0)
                        for r in batch)
            if not batch:
                continue
            with torch.cuda.stream(comp_stream), torch.no_grad():
                if mode == "gpu":
                    bx = torch.cat([r.x for r in batch], dim=0)
                    bm = torch.cat([r.mask for r in batch], dim=0) if batch[0].mask is not None else None
                else:
                    bx = torch.from_numpy(np.concatenate([r.x for r in batch], axis=0))
                    bm = (torch.from_numpy(np.concatenate([r.mask for r in batch], axis=0))
                          if batch[0].mask is not None else None)
                model.forward(bx, bm)
            comp_stream.synchronize()
            t_done = time.perf_counter() - t0
            for r in batch:
                r.t_done = t_done
                r.x = r.mask = None      # release (GPU) input memory
            done_reqs.extend(batch)
            batch_sizes.append(len(batch))

    ta = threading.Thread(target=arrivals, daemon=True)
    tw = threading.Thread(target=worker, daemon=True)
    ta.start()
    tw.start()
    ta.join()
    tw.join()
    torch.cuda.synchronize(dev)

    lat = np.array([(r.t_done - r.t_sched) * 1000 for r in done_reqs])
    wall = max(r.t_done for r in done_reqs)
    row = {
        "backbone": backbone, "mode": mode, "n_tasks": n_tasks, "rps_per_task": rps,
        "n_requests": len(done_reqs),
        "mean_ms": round(float(lat.mean()), 3),
        "p50_ms": round(float(np.percentile(lat, 50)), 3),
        "p99_ms": round(float(np.percentile(lat, 99)), 3),
        "throughput_rps": round(len(done_reqs) / wall, 2),
        "mean_batch": round(float(np.mean(batch_sizes)), 2),
        "peak_gpu_mb_above_model": round(
            (torch.cuda.max_memory_allocated(dev) - base_alloc) / 2**20, 1),
        "peak_queued_gpu_mb": round(gpu_queued_bytes[1] / 2**20, 2),
    }
    return row, done_reqs


def warmup(model, backbone: str, args, dev):
    rng = np.random.default_rng(0)
    for b in range(1, args.max_batch + 1):
        xs = [make_request(backbone, args.res, rng) for _ in range(b)]
        bx = torch.from_numpy(np.concatenate([x for x, _ in xs]))
        bm = (torch.from_numpy(np.concatenate([m for _, m in xs]))
              if xs[0][1] is not None else None)
        with torch.no_grad():
            for _ in range(3):
                model.forward(bx, bm)
    torch.cuda.synchronize(dev)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="momentlarge",
                    help="as the device server names it: momentlarge, dinobase-patch, swinlarge, ...")
    ap.add_argument("--rps", default="5,10,15,20", help="per-task request rates, comma-separated")
    ap.add_argument("--modes", default="host,gpu")
    ap.add_argument("--n-tasks", type=int, default=2)
    ap.add_argument("--duration", type=float, default=30.0, help="seconds of arrivals per run")
    ap.add_argument("--max-batch", type=int, default=5, help="device server default: 5")
    ap.add_argument("--batch-window-ms", type=float, default=0.0, help="device server default: 0")
    ap.add_argument("--res", type=int, default=224, help="vision input resolution")
    ap.add_argument("--repeats", type=int, default=1, help="runs per (rps, mode), alternating modes")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cuda", default="cuda:0")
    ap.add_argument("--out-dir", default="experiments/staging_ablation/results_t4")
    args = ap.parse_args()

    dev = torch.device(args.cuda)
    torch.cuda.set_device(dev)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[staging] GPU={torch.cuda.get_device_name(dev)} backbone={args.backbone}")
    pipeline = _build_pipeline(args.backbone, str(dev), None)
    model = pipeline.model_instance
    warmup(model, args.backbone, args, dev)

    rps_list = [float(r) for r in args.rps.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    summary = out / "summary.csv"
    new_file = not summary.exists()
    rows = []
    with summary.open("a", newline="") as f:
        w = None
        for rps in rps_list:
            for rep in range(args.repeats):
                # alternate mode order per repeat so drift does not favor one mode
                order = modes if rep % 2 == 0 else modes[::-1]
                for mode in order:
                    row, reqs = run_once(model, args.backbone, mode, rps, args, dev)
                    row["repeat"] = rep
                    row["gpu"] = torch.cuda.get_device_name(dev)
                    rows.append(row)
                    if w is None:
                        w = csv.DictWriter(f, fieldnames=list(row.keys()))
                        if new_file:
                            w.writeheader()
                    w.writerow(row)
                    f.flush()
                    with (out / f"requests_{args.backbone}_rps{rps:g}_{mode}_r{rep}.csv").open(
                            "w", newline="") as rf:
                        rw = csv.writer(rf)
                        rw.writerow(["task", "rid", "t_sched", "t_enq", "t_done", "latency_ms"])
                        for r in reqs:
                            rw.writerow([r.task, r.rid, f"{r.t_sched:.6f}", f"{r.t_enq:.6f}",
                                         f"{r.t_done:.6f}", f"{(r.t_done - r.t_sched) * 1000:.3f}"])
                    print(f"[staging] {args.backbone} rps={rps:g} mode={mode:4s} rep={rep}: "
                          f"mean={row['mean_ms']:.2f}ms p99={row['p99_ms']:.2f}ms "
                          f"tput={row['throughput_rps']} batch={row['mean_batch']} "
                          f"queued_gpu_peak={row['peak_queued_gpu_mb']}MB")

    # host vs gpu, averaged over repeats
    if {"host", "gpu"} <= set(modes):
        print("\n[staging] gpu vs host (positive = gpu staging is slower)")
        print(f"{'rps':>6} {'mean host':>10} {'mean gpu':>10} {'diff%':>7} "
              f"{'p99 host':>10} {'p99 gpu':>10} {'diff%':>7}")
        for rps in rps_list:
            def avg(mode, key):
                v = [r[key] for r in rows if r["rps_per_task"] == rps and r["mode"] == mode]
                return sum(v) / len(v)
            mh, mg = avg("host", "mean_ms"), avg("gpu", "mean_ms")
            ph, pg = avg("host", "p99_ms"), avg("gpu", "p99_ms")
            print(f"{rps:>6g} {mh:>10.2f} {mg:>10.2f} {100 * (mg - mh) / mh:>+7.1f} "
                  f"{ph:>10.2f} {pg:>10.2f} {100 * (pg - ph) / ph:>+7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
