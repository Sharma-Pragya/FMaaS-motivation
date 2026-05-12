#!/usr/bin/env python3
"""Closed-loop interference calibration via direct PyTorchRuntime.run_batch.

Bypasses gRPC + device servers + the scheduler. Each task runs in its own
subprocess pinned to a TPC partition (libsmctrl mask), repeatedly calling
runtime.run_batch with a batch of size BATCH_SIZE for DURATION seconds.
Throughput = batches_completed * batch_size / window.

Lets us measure the GPU+runtime ceiling per partition without serving-stack
overheads. Compare against the gRPC `no_sharing_tpc` numbers to see how much
of the 2:1 → 1.54:1 gap is overhead vs hardware scaling.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

SERVING_DIR = Path(__file__).resolve().parents[3]


def _setup_tpc_stream(tpc_partition: List[int], tpc_mode: str,
                      cuda_device: str):
    """Create a CUDA stream pinned to `tpc_partition` via libsmctrl.

    Mirrors planner/profiler/worker.py::_setup_tpc_stream but accepts an
    explicit partition list (so two workers can use non-overlapping TPCs).
    """
    import torch
    if not tpc_partition or tpc_mode == "none":
        return None
    if tpc_mode != "libsmctrl":
        raise ValueError(f"Unsupported tpc_mode: {tpc_mode}")

    candidates = []
    if "TPC_LIB_DIR" in os.environ:
        candidates.append(Path(os.environ["TPC_LIB_DIR"]) / "libsmctrl" / "libsmctrl.so")
    candidates += [
        SERVING_DIR.parent / "TPC_controller" / "tpc_controller" / "libsmctrl" / "libsmctrl.so",
        Path("/NFS/TPC_controller/tpc_controller/libsmctrl/libsmctrl.so"),
        Path("/NFS/TPC_controller/build/lib/tpc_controller/libsmctrl/libsmctrl.so"),
    ]
    so_path = next((p for p in candidates if p.exists()), None)
    if so_path is None:
        raise FileNotFoundError(
            "libsmctrl.so not found. Tried: " + ", ".join(str(c) for c in candidates))

    lib = ctypes.CDLL(str(so_path))
    lib.libsmctrl_set_stream_mask.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.libsmctrl_set_stream_mask.restype = None

    stream = torch.cuda.Stream(device=cuda_device)
    enable_bits = 0
    for tid in tpc_partition:
        enable_bits |= (1 << tid)
    disable_mask = (~enable_bits) & 0xFFFFFFFFFFFFFFFF
    lib.libsmctrl_set_stream_mask(
        ctypes.c_void_p(stream.cuda_stream),
        ctypes.c_uint64(disable_mask),
    )
    print(f"[worker pid={os.getpid()}] TPC mask: partition={tpc_partition}")
    return stream


def _build_batch(task: str, batch_size: int):
    import numpy as np
    from torch.utils.data import DataLoader
    from fmtk.datasetloaders.ecg5000 import ECG5000Dataset
    from fmtk.datasetloaders.uwavegesture import UWaveGestureLibraryALLDataset
    from site_manager.config import DATASET_DIR as _DATASET_DIR
    d = _DATASET_DIR
    if task == "ecgclass":
        ds = ECG5000Dataset({"dataset_path": f"{d}/ECG5000"},
                            {"task_type": "classification"}, "test")
    elif task == "gestureclass":
        ds = UWaveGestureLibraryALLDataset(
            {"dataset_path": f"{d}/UWaveGestureLibraryAll", "seq_len": 512},
            {"task_type": "classification"}, "test")
    else:
        raise ValueError(f"unknown task {task}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)
    batch = next(iter(loader))
    x = batch["x"].numpy().astype(np.float32)
    m = batch.get("mask")
    m = m.numpy().astype(np.float32) if m is not None else None
    return x, m


def worker_main(task: str, backbone: str, tpc_partition: List[int],
                tpc_mode: str, cuda_device: str,
                batch_size: int, duration: float, warmup: float,
                ready_barrier, start_event,
                out_conn) -> None:
    """Run run_batch in a tight loop for `duration` seconds; report throughput.

    Uses a barrier so both T1 and T2 finish initialization before either
    starts the timed loop, eliminating startup skew across subprocesses.
    """
    if str(SERVING_DIR) not in sys.path:
        sys.path.insert(0, str(SERVING_DIR))
    os.environ.setdefault("CUDA_DEVICE", cuda_device)

    import torch
    from device.runtime import PyTorchRuntime

    TASK_TYPES = {"ecgclass": "classification", "gestureclass": "classification"}

    stream = _setup_tpc_stream(tpc_partition, tpc_mode, cuda_device)
    runtime = PyTorchRuntime(cuda_stream=stream)
    runtime.load(backbone, [{"task": task, "type": TASK_TYPES[task],
                             "path": f"{task}_{backbone}_mlp"}])

    x, m = _build_batch(task, batch_size)
    tasks = [task] * batch_size
    print(f"[worker pid={os.getpid()}] task={task} x.shape={x.shape} "
          f"tpc_partition={tpc_partition}")

    # Warmup: a couple of forward passes (also primes any caches).
    for _ in range(2):
        runtime.run_batch(x, tasks, mask=m)

    # Sync: every worker reports ready, then blocks until parent fires.
    ready_barrier.wait()
    start_event.wait()

    t_start = time.time()
    t_warm  = t_start + warmup
    t_end   = t_start + duration

    counted_batches = 0
    while time.time() < t_end:
        result = runtime.run_batch(x, tasks, mask=m)
        done_at = time.time()
        if t_warm <= done_at < t_end:
            counted_batches += 1

    if stream is not None:
        stream.synchronize()

    eff = max(duration - warmup, 1e-6)
    rps = counted_batches * batch_size / eff
    out_conn.send({
        "task":          task,
        "tpc_partition": tpc_partition,
        "n_batches":     counted_batches,
        "rps":           rps,
    })
    out_conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone",    default="momentlarge")
    ap.add_argument("--cuda-device", default="cuda:0")
    ap.add_argument("--tpc-mode",    default="libsmctrl")
    ap.add_argument("--t1-task",     required=True)
    ap.add_argument("--t1-tpcs",     type=int, required=True,
                    help="Number of TPCs for T1 (partition = [0..t1_tpcs-1]).")
    ap.add_argument("--t2-task",     default="",
                    help="Empty = T1 alone (no T2 process).")
    ap.add_argument("--t2-tpcs",     type=int, default=0)
    ap.add_argument("--batch-size",  type=int, default=16)
    ap.add_argument("--duration",    type=float, default=15.0)
    ap.add_argument("--warmup-secs", type=float, default=3.0)
    ap.add_argument("--out-csv",     required=True)
    args = ap.parse_args()

    print(f"[Calib2] T1={args.t1_task}@{args.t1_tpcs}TPCs  "
          f"T2={args.t2_task or '<none>'}@{args.t2_tpcs}TPCs  "
          f"batch={args.batch_size}  dur={args.duration}s")

    t1_partition = list(range(args.t1_tpcs))
    t2_partition = list(range(args.t1_tpcs, args.t1_tpcs + args.t2_tpcs))

    # Spawn (not fork) — required for CUDA in subprocesses.
    ctx = mp.get_context("spawn")
    parent_conns = {}
    procs = []
    n_workers = 1 + (1 if (args.t2_task and args.t2_tpcs > 0) else 0)
    ready_barrier = ctx.Barrier(n_workers + 1)  # +1 = main waits with workers
    start_event   = ctx.Event()

    p_conn, c_conn = ctx.Pipe(duplex=False)
    parent_conns["t1"] = p_conn
    p = ctx.Process(target=worker_main,
                    args=(args.t1_task, args.backbone, t1_partition,
                          args.tpc_mode, args.cuda_device,
                          args.batch_size, args.duration, args.warmup_secs,
                          ready_barrier, start_event, c_conn))
    procs.append(p)

    if args.t2_task and args.t2_tpcs > 0:
        p_conn, c_conn = ctx.Pipe(duplex=False)
        parent_conns["t2"] = p_conn
        p = ctx.Process(target=worker_main,
                        args=(args.t2_task, args.backbone, t2_partition,
                              args.tpc_mode, args.cuda_device,
                              args.batch_size, args.duration, args.warmup_secs,
                              ready_barrier, start_event, c_conn))
        procs.append(p)

    for p in procs:
        p.start()
    # Wait until all workers finish init + warmup, then fire the start event
    # so both begin the timed loop at the same wall-clock instant.
    ready_barrier.wait()
    print(f"[Calib2] all {n_workers} worker(s) ready — starting timed loop")
    start_event.set()
    # Receive results before joining (Pipe send blocks if not drained).
    results = {name: conn.recv() for name, conn in parent_conns.items()}
    for p in procs:
        p.join()

    t1 = results["t1"]
    t2 = results.get("t2")
    t1_rps = t1["rps"]
    t2_rps = t2["rps"] if t2 else 0.0
    print(f"[Calib2] T1={t1_rps:.2f} RPS   T2={t2_rps:.2f} RPS")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["t1_task", "t1_tpcs", "t2_task", "t2_tpcs",
                        "batch_size", "duration_s", "t1_rps", "t2_rps"])
        w.writerow([args.t1_task, args.t1_tpcs,
                    args.t2_task or "", args.t2_tpcs,
                    args.batch_size, args.duration,
                    round(t1_rps, 4), round(t2_rps, 4)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
