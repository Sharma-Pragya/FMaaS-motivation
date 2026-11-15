#!/usr/bin/env python3
"""memory_isolation_vision/run.py — does one task's load surge take the GPU
memory the other tasks need?

--n-tasks depth-estimation tasks (default 8, as RTVSntask/vision) on one
DINOv2 backbone (FMTK), each with its own depth head sized to --res (default
224, the size RTVSntask/vision's NYU / VOC loaders resize to):
    B         the surging task: open-loop, --rate-b req/s, then from
              --ramp-at s a linear ramp to --rate-b-max over --ramp-s s,
              held to the end.
    A1..A{N-1} the victims: open-loop, constant --rate-a req/s each.
Arrivals are Poisson, generated up front from a seeded stream per task, so
every mode sees the same arrival times. Every request is one 3 x res x res
fp32 image (the same size for every task).

Requests are staged on the GPU when they arrive (--staging gpu): a serving
process copies each request's input into VRAM as soon as it receives it
(side stream, from one pinned host buffer per task: the values do not
matter), and the request waits there until a batch takes it. A backlog
therefore holds GPU memory: queued requests x 3 x res^2 x 4 B (3.1 MB at
518). --staging host is the control: inputs wait in host RAM and reach the
GPU when their batch runs, as the device server does today.

Batching follows the device server (device/batcher.py): with --batch-window-ms
0 a batch is whatever is staged, up to --max-batch, taken first come first
served; one backbone forward per batch; the features are split by task and
each task's head runs on its own thread and CUDA stream, once per batch, and
copies its output to the host (as runtime.run_decoders does).

Modes (what differs is only who shares a process and a batch):
    s_be  one process serves every task: one staging queue, cross-task
          batches, first come first served.
    be    one process per task, each with its own backbone copy, staging
          queue and batches of its own task only (sharing_benefit no_sharing:
          one device server per task, same --max-batch).
    sp    be, with each process's compute streams masked to a share of the
          GPU's TPCs through libsmctrl: by default an even one, total // N
          each, the remainder unused (fault_isolation_tsfm/run.py:
          _masked_stream; T4, 8 tasks: 2 TPCs each); with --sp-split, shares
          in that ratio (fair_share splits TPCs by the BFQ weight ratio).
          Compute only: memory is not partitioned, and without MPS the
          processes still take turns on the GPU.
    fmvisor  as fault_isolation_tsfm's fmvisor: one backbone process with
          cross-task batching, and one process per task running its head in
          its own CUDA context. Every request is staged on the GPU at
          arrival, as in s_be, but into a slot in its own task process's
          memory (a slot per request as it arrives, as s_be and be stage
          theirs; --fm-chunk > 1 allocates that many at a time; shared with
          the backbone through CUDA IPC), so a task's backlog is that task's
          memory. A
          batch is the staged requests across tasks, up to --max-batch, with
          no per-task limit, chosen by BFQ (the device server's STFQPolicy,
          weights 1 / declared rate: equal by default). The backbone copies a
          batch's inputs out of the slots and writes its features to a ring
          of RING batches in its own memory, which the task processes read.
          A task process that hits an OOM ends; the backbone reports it and
          stops serving it; the other tasks go on.

    sp_mps  be, with the processes as MPS clients (the MPS daemon is
          started for this mode only and stopped after), each limited to
          100 // N % of the SMs, or --sp-split's share
          (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE, as
          sharing_benefit's no_sharing_mps): the processes run at the same
          time on their shares of the SMs, rather than taking turns as in
          sp. Memory is not partitioned.
    mig   be, with each process on its own MIG slice (CUDA_VISIBLE_DEVICES =
          the slice's UUID, set before CUDA starts): memory and SMs are
          partitioned in hardware. Needs one slice per task, created
          beforehand (RTX PRO 6000 Blackwell: at most 4 x 1g.24gb, 46 of 188
          SMs each, so --n-tasks 4; A100-40GB: 7 x 1g.5gb); slices come from
          --mig-uuids or `nvidia-smi -L`. Memory is sampled per
          slice through NVML.

Out-of-memory is not caught: the process that hits it reports which thread
failed and exits, as a server that treats OOM as fatal would. Nothing
restarts. A request fails if its process dies with it in flight, or if it is
sent while its process is down.

Measured every --mem-period s:
    by each serving process, per task: requests staged and their MB, plus
        features waiting for the task's head and their MB (exact per-task
        memory even in s_be, where NVML sees one process); and the process's
        torch allocated / reserved MB.
    by the parent (no CUDA context), through NVML: device used MB and MB per
        serving process.

Output: <exp-dir>/<staging>/events_<mode>.csv, summary.jsonl. Plot with plot.py.

Usage (from serving/, FMTK/src on PYTHONPATH; run.sh sets it up):
    python experiments/memory_isolation_vision/run.py --modes s_be,be,sp \\
        --exp-dir experiments/memory_isolation_vision/results_t4_n8
"""
from __future__ import annotations

import argparse
import collections
import csv
import ctypes
import json
import multiprocessing as mp
import os
import queue
import random
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
SERVING_DIR = EXP_DIR.parents[1]
TPC_LIB_DIR = Path(os.environ.get(
    "TPC_LIB_DIR", SERVING_DIR.parents[1] / "TPC_controller" / "tpc_controller"))
SURGE = "B"                # the surging task; the victims are A1..A{N-1}
MODES = ("s_be", "be", "sp", "sp_mps", "fmvisor", "mig")
MPS_CONTROL = Path(os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")) / "control"
STAGINGS = ("gpu", "host")
JOIN_S = 30.0              # parent: each serving process, at stop
WARM_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)


# ---------------------------------------------------------------------------
# Arrivals (as memory_isolation_llm/run.py)
# ---------------------------------------------------------------------------

def task_names(n: int) -> tuple:
    """The surging task first, then the victims."""
    return (SURGE,) + tuple(f"A{i}" for i in range(1, n))


def rate_at(task: str, t: float, a) -> float:
    """Offered load (req/s) at t s after t = 0; warmup (t < 0) at base rates."""
    if task != SURGE:
        return a.rate_a
    if t < a.ramp_at:
        return a.rate_b
    frac = min(1.0, (t - a.ramp_at) / a.ramp_s) if a.ramp_s > 0 else 1.0
    return a.rate_b + frac * (a.rate_b_max - a.rate_b)


def arrivals(task: str, a) -> list:
    """Poisson arrival times (s, relative to t = 0) by thinning, seeded per task."""
    rng = random.Random(f"{a.seed}-{task}")
    peak = max(a.rate_a, a.rate_b, a.rate_b_max)
    out, t = [], -a.warmup
    while True:
        t += rng.expovariate(peak)
        if t >= a.duration:
            return out
        if rng.random() < rate_at(task, t, a) / peak:
            out.append(t)


# ---------------------------------------------------------------------------
# Serving process: one backbone, the given tasks' heads
# ---------------------------------------------------------------------------

def _backbone(cfg: dict):
    """DINOv2 (FMTK): (forward, embed_dim, attention implementation)."""
    import torch
    from fmtk.components.backbones.dinov2 import DinoV2Model
    torch.manual_seed(0)
    fm = DinoV2Model(cfg["device"], cfg["backbone"].split("-", 1)[1],
                     model_config={"return_all_tokens": True})
    fm.model.eval()
    attn = getattr(fm.model.config, "_attn_implementation", "unknown")
    return (lambda x: fm.forward(x)), fm.embed_dim, attn


def _head(cfg: dict, task: str, embed_dim: int):
    """Task's depth head sized to cfg["res"], seeded by the task's index: the
    same head in every mode."""
    import torch
    from fmtk.components.decoders.regression.monocular_depth import MonocularDepthDecoder
    res = cfg["res"]
    torch.manual_seed(1000 + cfg["all_tasks"].index(task))
    dec = MonocularDepthDecoder(device=cfg["device"], cfg={
        "input_dim": embed_dim, "height": res // 14, "width": res // 14,
        "pixel_height": res, "pixel_width": res, "mode": "PATCH"})
    dec.to_device()
    dec.eval()
    return dec.forward


def _models(cfg: dict, tasks: tuple):
    """DINOv2 and one depth head per task in tasks."""
    backbone, embed_dim, attn = _backbone(cfg)
    heads = {t: _head(cfg, t, embed_dim) for t in cfg["all_tasks"] if t in tasks}
    return backbone, heads, attn


def _fatal_hook(evt_q, comp: str, dying=None):
    """threading.excepthook: report an uncaught exception (an OOM, typically)
    as comp's error, then end the process, as a crashed server would.
    dying: an Event set first, so a process that maps this one's memory (the
    fmvisor backbone) stops using it before this process is gone."""
    def hook(hook_args) -> None:
        if dying is not None:
            dying.set()
        traceback.print_exception(hook_args.exc_type, hook_args.exc_value,
                                  hook_args.exc_traceback)
        name = getattr(hook_args.thread, "name", "?")
        msg = f"{hook_args.exc_type.__name__} in {name}: {str(hook_args.exc_value)[:160]}"
        try:
            evt_q.put(("error", comp, time.monotonic(), msg))
            evt_q.close()
            evt_q.join_thread()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
    return hook


def _tpc_range(n_tpcs: int, i: int, n_inst: int, weights=None) -> tuple:
    """Instance i's contiguous TPCs [lo, hi). Without weights: an even share,
    n_tpcs // n_inst each, the remainder unused (fault_isolation_tsfm,
    sharing_benefit's no_sharing_tpc). With weights (one per instance, in
    instance order): shares in that ratio, all TPCs used, at least one each
    (fair_share's no_sharing_tpc_3_1 splits TPCs by the weight ratio)."""
    if not weights:
        per = n_tpcs // n_inst
        if per == 0:
            raise ValueError(f"{n_inst} instances but only {n_tpcs} TPCs")
        return i * per, (i + 1) * per
    if n_tpcs < n_inst:
        raise ValueError(f"{n_inst} instances but only {n_tpcs} TPCs")
    total, cum, bounds = float(sum(weights)), 0.0, [0]
    for k, w in enumerate(weights):
        cum += w
        b = round(n_tpcs * cum / total)
        # at least one TPC for this instance and for each one after it
        b = max(b, bounds[-1] + 1)
        b = min(b, n_tpcs - (n_inst - 1 - k))
        bounds.append(b)
    return bounds[i], bounds[i + 1]


def _tpc_mask(stream, i: int, n_inst: int, device, weights=None) -> str:
    """Mask stream to instance i's TPCs (_tpc_range) through libsmctrl. As
    fault_isolation_tsfm/run.py:_masked_stream, for a stream that exists."""
    lib = ctypes.CDLL(str(TPC_LIB_DIR / "libsmctrl" / "libsmctrl.so"))
    lib.libsmctrl_set_stream_mask_ext.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
    lib.libsmctrl_set_stream_mask_ext.restype = None
    lib.libsmctrl_get_tpc_info_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int]
    lib.libsmctrl_get_tpc_info_cuda.restype = ctypes.c_int
    n = ctypes.c_uint32()
    if lib.libsmctrl_get_tpc_info_cuda(ctypes.byref(n), device.index or 0) != 0:
        raise RuntimeError("libsmctrl_get_tpc_info_cuda failed")
    lo, hi = _tpc_range(n.value, i, n_inst, weights)
    enable = sum(1 << k for k in range(lo, hi))
    disable = ~enable & ((1 << 128) - 1)   # set bit = TPC disabled
    lib.libsmctrl_set_stream_mask_ext(
        ctypes.c_void_p(stream.cuda_stream),
        ctypes.c_uint64(disable & 0xFFFFFFFFFFFFFFFF),
        ctypes.c_uint64(disable >> 64))
    return f"TPCs {lo}..{hi - 1} of {n.value}"


def server_main(comp: str, tasks: tuple, cfg: dict, req_qs: dict, evt_q, status_q,
                tpc=None, mig=None, mps_pct=None) -> None:
    """tpc: (i, n) in sp mode: every compute stream of this process is masked
    to instance i's share of n (copy streams only copy and stay unmasked).
    mps_pct: in sp_mps mode, this MPS client's share of the SMs."""
    if mps_pct is not None:
        # sp_mps: an MPS client limited to mps_pct % of the SMs (as
        # device/main.py --mps-thread-pct). Set before CUDA starts.
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_pct)
    if mig is not None:
        # mig mode: this process sees only its own MIG slice, as cuda:0. Must
        # be set before CUDA starts (torch is first imported just below).
        os.environ["CUDA_VISIBLE_DEVICES"] = mig
        cfg = {**cfg, "device": "cuda:0"}
    import torch
    dev = torch.device(cfg["device"])
    cuda = dev.type == "cuda"
    if tpc is not None and not cuda:
        raise RuntimeError("sp needs a CUDA device")
    if cuda:
        torch.cuda.set_device(dev)
    torch.set_num_threads(1)
    parent_pid = os.getppid()
    threading.excepthook = _fatal_hook(evt_q, comp)

    tpc_note = f"MPS client, {mps_pct}% of SMs" if mps_pct is not None else None

    def stream(compute: bool = True):
        nonlocal tpc_note
        if not cuda:
            return None
        s = torch.cuda.Stream(device=dev)
        if compute and tpc is not None:
            tpc_note = _tpc_mask(s, tpc[0], tpc[1], dev, tpc[2])
        return s

    def in_stream(s):
        return torch.cuda.stream(s) if s is not None else nullcontext()

    backbone, heads, attn = _models(cfg, tasks)
    shape = (3, cfg["res"], cfg["res"])
    req_mb = 3 * cfg["res"] ** 2 * 4 / 2**20

    # Warm-up: every power-of-two batch size up to max_batch, timed, with the
    # forward's peak memory above the weights (calibration for the rates).
    calib = {}
    bb_stream = stream()
    with in_stream(bb_stream), torch.inference_mode():
        for b in [s for s in WARM_SIZES if s < cfg["max_batch"]] + [cfg["max_batch"]]:
            x = torch.randn((b, *shape), device=dev)
            if cuda:
                torch.cuda.synchronize(dev)
                torch.cuda.reset_peak_memory_stats(dev)
            base = torch.cuda.memory_allocated(dev) if cuda else 0
            for rep in range(2):          # the second run is the timed one
                t0 = time.monotonic()
                feats = backbone(x)
                for h in heads.values():
                    h(feats).cpu()
                if cuda:
                    torch.cuda.synchronize(dev)
            peak = (torch.cuda.max_memory_allocated(dev) - base) / 2**20 if cuda else None
            calib[b] = {"ms": round(1e3 * (time.monotonic() - t0), 1),
                        "peak_mb": round(peak, 1) if peak is not None else None}
            del x, feats
    weights_mb = round(torch.cuda.memory_allocated(dev) / 2**20, 1) if cuda else None

    host_x = {t: torch.randn((1, *shape)) for t in tasks}
    if cuda:
        host_x = {t: x.pin_memory() for t, x in host_x.items()}
    task_streams = {t: stream() for t in tasks}
    copy_streams = {t: stream(compute=False) for t in tasks}

    # Staging queue, first come first served across this process's tasks.
    staged = collections.deque()      # (task, rid, gpu tensor or None, event)
    cond = threading.Condition()
    lock = threading.Lock()           # the counters below
    n_staged = {t: 0 for t in tasks}
    n_feat = {t: 0 for t in tasks}    # requests whose features wait for the head
    feat_mb = {t: 0.0 for t in tasks}
    head_qs = {t: queue.Queue() for t in tasks}
    stop = threading.Event()

    def next_rid(t: str):
        while True:
            try:
                return req_qs[t].get(timeout=0.5)
            except queue.Empty:
                if os.getppid() != parent_pid:
                    return None

    def intake(t: str) -> None:
        cs = copy_streams[t]
        while True:
            rid = next_rid(t)
            if rid is None:
                return
            g = ev = None
            if cfg["staging"] == "gpu":
                with in_stream(cs):
                    g = host_x[t].to(dev, non_blocking=True)   # in VRAM on arrival
                    if cuda:
                        ev = torch.cuda.Event()
                        ev.record(cs)
            with cond:
                staged.append((t, rid, g, ev))
                with lock:
                    n_staged[t] += 1
                cond.notify()

    def take_batch() -> list:
        with cond:
            while not staged and not stop.is_set():
                cond.wait(timeout=0.05)
            if not staged:
                return []
        if cfg["batch_window_s"] > 0:
            deadline = time.monotonic() + cfg["batch_window_s"]
            while len(staged) < cfg["max_batch"] and time.monotonic() < deadline:
                time.sleep(0.0002)
        with cond:
            batch = [staged.popleft() for _ in range(min(cfg["max_batch"], len(staged)))]
            with lock:
                for t, *_ in batch:
                    n_staged[t] -= 1
        return batch

    def backbone_loop() -> None:
        with in_stream(bb_stream), torch.inference_mode():
            while not stop.is_set():
                batch = take_batch()
                if not batch:
                    continue
                if cfg["staging"] == "gpu":
                    for _, _, g, ev in batch:
                        if ev is not None:
                            bb_stream.wait_event(ev)
                            g.record_stream(bb_stream)
                    x = torch.cat([g for _, _, g, _ in batch])
                else:
                    x = torch.cat([host_x[t] for t, *_ in batch]).to(dev, non_blocking=True)
                feats = backbone(x)
                groups: dict = {}
                for j, (t, rid, _, _) in enumerate(batch):
                    groups.setdefault(t, ([], []))
                    groups[t][0].append(j)
                    groups[t][1].append(rid)
                batch.clear()
                del x
                for t, (idx, rids) in groups.items():
                    ft = feats.index_select(0, torch.tensor(idx, device=dev))
                    ev = None
                    if cuda:
                        ev = torch.cuda.Event()
                        ev.record(bb_stream)
                    with lock:
                        n_feat[t] += len(rids)
                        feat_mb[t] += ft.numel() * ft.element_size() / 2**20
                    head_qs[t].put((ft, ev, rids))
                del feats

    def head_worker(t: str) -> None:
        s = task_streams[t]
        with in_stream(s), torch.inference_mode():
            while True:
                item = head_qs[t].get()
                if item is None:
                    return
                ft, ev, rids = item
                if ev is not None:
                    s.wait_event(ev)
                    ft.record_stream(s)
                heads[t](ft).cpu()        # synchronizes this task's stream
                now = time.monotonic()
                with lock:
                    n_feat[t] -= len(rids)
                    feat_mb[t] -= ft.numel() * ft.element_size() / 2**20
                del ft, item
                for rid in rids:
                    evt_q.put(("done", t, rid, now))

    def reporter() -> None:
        while not stop.wait(cfg["mem_period"]):
            with lock:
                per = {t: [n_staged[t], round(n_staged[t] * req_mb if cfg["staging"] == "gpu"
                                              else 0.0, 1),
                           n_feat[t], round(feat_mb[t], 1)] for t in tasks}
            alloc = torch.cuda.memory_allocated(dev) / 2**20 if cuda else None
            resv = torch.cuda.memory_reserved(dev) / 2**20 if cuda else None
            evt_q.put(("stats", comp, time.monotonic(), per,
                       None if alloc is None else round(alloc, 1),
                       None if resv is None else round(resv, 1)))

    workers = [threading.Thread(target=backbone_loop, name="backbone", daemon=True)]
    workers += [threading.Thread(target=head_worker, args=(t,), name=f"head-{t}", daemon=True)
                for t in tasks]
    workers.append(threading.Thread(target=reporter, name="reporter", daemon=True))
    intakes = [threading.Thread(target=intake, args=(t,), name=f"intake-{t}", daemon=True)
               for t in tasks]
    for th in workers + intakes:
        th.start()
    status_q.put(("up", comp, time.monotonic(),
                  {"calib": calib, "weights_mb": weights_mb, "attn": attn,
                   "req_mb": round(req_mb, 3), "tasks": list(tasks), "tpc": tpc_note}))
    for th in intakes:          # the parent stops each task with None
        th.join()
    stop.set()
    with cond:
        cond.notify_all()
    for q in head_qs.values():
        q.put(None)
    for th in workers:
        th.join(timeout=5)


# ---------------------------------------------------------------------------
# fmvisor: backbone process + one process per task (fault_isolation_tsfm's
# fmvisor), every request on the GPU in a slot its own task process owns
# ---------------------------------------------------------------------------

RING = 4          # fmvisor: batches of features the backbone can have out at once


def fm_task_main(t: str, i: int, cfg: dict, ring, req_q, evt_q, sched_q, done_q, ack_q,
                 ready_q, run_stop, dying) -> None:
    """Task t's own process, so its own CUDA context. Every request of t is
    copied to the GPU on arrival, into a slot in this process's own memory
    (a new slot only when every slot it has is in use, --fm-chunk at a time:
    one by default, so per request as it arrives; each allocation shared
    once with the backbone through CUDA IPC). So t's backlog is
    t's memory: if it outgrows the GPU, this process hits the OOM and ends
    (not caught, as everywhere in this script), and nothing else does.

    The backbone reads the slots of each batch, writes the features to its
    own `ring`, and says which rows are t's; this process runs t's head on
    them, acknowledges, replies, and frees the slots.

    Every wait is bounded and gives up once the run stops (run_stop: the
    backlog is not drained) or the backbone process is gone."""
    import torch
    import torch.multiprocessing  # noqa: F401  (CUDA tensors through queues)
    dev = torch.device(cfg["device"])
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.set_device(dev)
    torch.set_num_threads(1)
    comp = f"task-{t}"
    threading.excepthook = _fatal_hook(evt_q, comp, dying)
    backbone_pid = os.getppid()
    shape = (3, cfg["res"], cfg["res"])
    chunk_n = cfg["fm_chunk"]
    req_mb = 3 * cfg["res"] ** 2 * 4 / 2**20
    head = _head(cfg, t, ring.shape[-1])
    x = torch.randn(shape)
    if cuda:
        x = x.pin_memory()
    with torch.inference_mode():            # warm the head at the largest batch
        head(ring[0]).cpu()

    lock = threading.Lock()
    chunks: list = []                       # this task's slot memory, kept alive
    free: list = []                         # (chunk id, row)
    rid_of: dict = {}                       # (chunk id, row) -> request in it

    def new_chunk() -> None:
        """Called with lock held. An OOM here ends this process only."""
        c = torch.empty((chunk_n, *shape), device=dev)
        if not cuda:
            c.share_memory_()
        cid = len(chunks)
        chunks.append(c)
        sched_q.put(("chunk", cid, c))      # before any slot of it
        free.extend((cid, j) for j in range(chunk_n))

    with lock:
        new_chunk()
    ready_q.put((i, os.getpid()))
    stop = threading.Event()

    def over() -> bool:
        return stop.is_set() or run_stop.is_set() or os.getppid() != backbone_pid

    def reporter() -> None:
        while not stop.wait(cfg["mem_period"]):
            with lock:
                n_gpu, n_slots = len(rid_of), len(chunks) * chunk_n
            per = {t: [n_gpu, round(n_gpu * req_mb, 1), n_slots, 0.0]}
            alloc = torch.cuda.memory_allocated(dev) / 2**20 if cuda else None
            resv = torch.cuda.memory_reserved(dev) / 2**20 if cuda else None
            evt_q.put(("stats", comp, time.monotonic(), per,
                       None if alloc is None else round(alloc, 1),
                       None if resv is None else round(resv, 1)))

    def intake() -> None:
        copy_stream = torch.cuda.Stream(device=dev) if cuda else None
        while not over():
            try:
                rid = req_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if rid is None:
                return
            with lock:
                if not free:
                    new_chunk()
                slot = free.pop()
                rid_of[slot] = rid
            cid, j = slot
            if cuda:
                with torch.cuda.stream(copy_stream):
                    chunks[cid][j].copy_(x, non_blocking=True)
                # Host-side queues do not order kernels across contexts: the
                # input must be in VRAM before the backbone reads it.
                copy_stream.synchronize()
            else:
                chunks[cid][j].copy_(x)
            sched_q.put(("slot", cid, j, time.monotonic()))

    threading.Thread(target=reporter, name="reporter", daemon=True).start()
    threading.Thread(target=intake, name="intake", daemon=True).start()
    with torch.inference_mode():
        while not over():
            try:
                r, rows, slots = done_q.get(timeout=0.5)
            except queue.Empty:
                continue
            idx = torch.tensor(rows, dtype=torch.long, device=dev)
            head(ring[r].index_select(0, idx)).cpu()     # synchronizes this context
            ack_q.put((r, i))
            now = time.monotonic()
            with lock:
                done = [rid_of.pop(tuple(s)) for s in slots]
                free.extend(tuple(s) for s in slots)
            for rid in done:
                evt_q.put(("done", t, rid, now))
    stop.set()


def fm_backbone_main(cfg: dict, tasks: tuple, req_qs: dict, evt_qs: dict, status_q,
                     run_stop) -> None:
    """The backbone process: the backbone, and RING batches of features in its
    VRAM. Batches are the slots the task processes hand in, across tasks, up
    to --max-batch, with no per-task limit, chosen by BFQ: the device
    server's STFQPolicy (device/scheduler.py) over one queue per task, driven
    as device/batch_former.py and device/batcher.py drive it (start time when
    a request reaches the backbone, select() per batch,
    update_after_execution() with the batch's duration). Weights are 1 / each
    task's declared rate (--rate-a for victims, --rate-b for B), as
    RTVSntask's sharing condition sets them. For each batch it copies the inputs
    out of the tasks' slots (the only time it touches a task's memory), runs
    the backbone, writes the features to a free ring entry, and tells each
    task which rows are its. It starts the task processes, reports one that
    dies (and never batches it again), and goes on."""
    import torch
    import torch.multiprocessing as tmp
    if str(SERVING_DIR) not in sys.path:
        sys.path.insert(0, str(SERVING_DIR))
    from device.scheduler import RequestEnvelope, STFQPolicy
    dev = torch.device(cfg["device"])
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.set_device(dev)
    torch.set_num_threads(1)
    evt_q = evt_qs["backbone"]
    threading.excepthook = _fatal_hook(evt_q, "backbone")
    parent_pid = os.getppid()
    n, max_b = len(tasks), cfg["max_batch"]
    shape = (3, cfg["res"], cfg["res"])

    backbone, embed_dim, attn = _backbone(cfg)
    bb_stream = torch.cuda.Stream(device=dev) if cuda else None

    def in_stream():
        return torch.cuda.stream(bb_stream) if cuda else nullcontext()

    calib = {}
    with in_stream(), torch.inference_mode():
        for b in [s for s in WARM_SIZES if s < max_b] + [max_b]:
            xb = torch.randn((b, *shape), device=dev)
            if cuda:
                torch.cuda.synchronize(dev)
                torch.cuda.reset_peak_memory_stats(dev)
            base = torch.cuda.memory_allocated(dev) if cuda else 0
            for rep in range(2):            # the second run is the timed one
                t0 = time.monotonic()
                fb = backbone(xb)
                if cuda:
                    torch.cuda.synchronize(dev)
            peak = (torch.cuda.max_memory_allocated(dev) - base) / 2**20 if cuda else None
            calib[b] = {"ms": round(1e3 * (time.monotonic() - t0), 1),
                        "peak_mb": round(peak, 1) if peak is not None else None}
        feat_shape = tuple(fb.shape[1:])
        del xb, fb
    weights_mb = round(torch.cuda.memory_allocated(dev) / 2**20, 1) if cuda else None
    ring = torch.zeros((RING, max_b, *feat_shape), device=dev)
    if not cuda:
        ring.share_memory_()

    ctx = tmp.get_context("spawn")
    sched_qs = [ctx.Queue() for _ in range(n)]   # task i -> backbone: chunks, filled slots
    done_qs = [ctx.Queue() for _ in range(n)]    # backbone -> task i: ring rows to read
    ack_q = ctx.Queue()                          # tasks -> backbone: ring rows read
    ready_q = ctx.Queue()
    dying = [ctx.Event() for _ in range(n)]      # task i failed: stop using its memory
    procs = [ctx.Process(target=fm_task_main, name=f"task-{t}", daemon=True,
                         args=(t, i, cfg, ring, req_qs[t], evt_qs[f"task-{t}"], sched_qs[i],
                               done_qs[i], ack_q, ready_q, run_stop, dying[i]))
             for i, t in enumerate(tasks)]
    for p in procs:
        p.start()
    pids = {}
    while len(pids) < n:
        try:
            i, pid = ready_q.get(timeout=1.0)
            pids[f"task-{tasks[i]}"] = pid
        except queue.Empty:
            if any(p.exitcode is not None for p in procs):
                raise RuntimeError("a task process died during startup")
    up = [True] * n
    now = time.monotonic()
    for i, t in enumerate(tasks):
        status_q.put(("up", f"task-{t}", now, {"pid": pids[f"task-{t}"]}))
    status_q.put(("up", "backbone", now,
                  {"calib": calib, "weights_mb": weights_mb, "attn": attn, "pids": pids,
                   "fm_chunk": cfg["fm_chunk"],
                   "ring_mb": round(ring.numel() * 4 / 2**20, 1)}))

    chunks: list = [dict() for _ in range(n)]   # task i: chunk id -> its slots (IPC)
    # BFQ: one queue per task of RequestEnvelope(req_id=(task, chunk id, row)).
    policy = STFQPolicy(weights={t: 1.0 / cfg["rates"][t] for t in tasks})
    task_qs = {t: collections.deque() for t in tasks}

    def n_pending() -> int:
        return sum(map(len, task_qs.values()))
    unread = [set() for _ in range(RING)]       # ring entry -> tasks yet to read it
    next_r = 0

    def drop_task(i: int, why) -> None:
        """Task i is gone: never batch it again, release its memory, and
        report it (why None: it stopped at the end of the run)."""
        up[i] = False
        task_qs[tasks[i]].clear()
        chunks[i].clear()
        for u in unread:
            u.discard(i)
        if why is not None:
            evt_q.put(("error", f"task-{tasks[i]}", time.monotonic(), why))

    def poll() -> None:
        for i in range(n):
            while up[i]:
                if dying[i].is_set():
                    drop_task(i, "task process failed")
                    break
                try:
                    msg = sched_qs[i].get_nowait()
                except queue.Empty:
                    break
                except RuntimeError as exc:
                    # Opening a chunk of a task that has just died (its
                    # memory is gone): drop the task, not the backbone.
                    drop_task(i, f"task's memory unavailable: {str(exc)[:80]}")
                    break
                if msg[0] == "chunk":
                    chunks[i][msg[1]] = msg[2]
                else:
                    # x=None: every request is one batch shape (same image size).
                    env = RequestEnvelope(req_id=(i, msg[1], msg[2]), task=tasks[i], x=None,
                                          mask=None, question=None, enqueued_at=msg[3],
                                          future=None)
                    task_qs[tasks[i]].append(env)
                    policy.assign_start_time(env)

    def check_tasks() -> None:
        # A dead task is reported once and never batched again.
        for i, p in enumerate(procs):
            if not up[i]:
                continue
            if dying[i].is_set():
                drop_task(i, "task process failed")
            elif p.exitcode is not None:    # 0: stopped at the end of the run
                drop_task(i, f"task process exit code {p.exitcode}" if p.exitcode else None)

    def drain_acks() -> None:
        while True:
            try:
                r, i = ack_q.get_nowait()
            except queue.Empty:
                return
            unread[r].discard(i)

    window = cfg["batch_window_s"]
    with in_stream(), torch.inference_mode():
        while os.getppid() == parent_pid and not run_stop.is_set():
            check_tasks()
            if not any(up):
                break                       # every task has stopped
            drain_acks()
            poll()
            if not n_pending() or unread[next_r]:
                time.sleep(0.0002)
                continue
            deadline = time.monotonic() + window
            while n_pending() < max_b and time.monotonic() < deadline:
                poll()
            check_tasks()                   # last look before touching tasks' memory
            batch = [e.req_id for e in policy.select(task_qs, max_b)]   # (task, chunk, row)
            if not batch:
                continue
            t0 = time.monotonic()
            x = torch.stack([chunks[i][cid][j] for i, cid, j in batch])
            feats = backbone(x)
            r = next_r
            next_r = (next_r + 1) % RING
            ring[r][:len(batch)].copy_(feats)
            # Inputs are copied and features written before any task reads
            # them or reuses a slot.
            if cuda:
                bb_stream.synchronize()
            del x, feats
            # As device/batcher.py: the batch's duration corrects the virtual
            # times of the tasks in it.
            policy.update_after_execution([tasks[i] for i, _, _ in batch],
                                          time.monotonic() - t0, task_qs)
            by_task: dict = {}
            for k, (i, cid, j) in enumerate(batch):
                rows, slots = by_task.setdefault(i, ([], []))
                rows.append(k)
                slots.append((cid, j))
            unread[r] = {i for i in by_task if up[i]}
            for i, (rows, slots) in by_task.items():
                done_qs[i].put((r, rows, slots))
    deadline = time.monotonic() + 10.0
    for p in procs:
        p.join(timeout=max(0.0, deadline - time.monotonic()))
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join(timeout=1)


# ---------------------------------------------------------------------------
# Parent: GPU memory through NVML (fault_isolation_tsfm/run.py:GpuMemSampler)
# ---------------------------------------------------------------------------

class GpuMemSampler:
    """Device used memory and per-process used memory on the run's GPU. The
    parent never creates a CUDA context. Per-process numbers include each
    process's CUDA context and everything its caching allocator reserved; they
    can be missing in a container with its own PID namespace. Assumes the NVML
    index is the CUDA index (CUDA_VISIBLE_DEVICES does not reorder GPUs)."""

    def __init__(self, device: str, mig_uuids=None):
        """mig_uuids: in mig mode, the slices in use. Each slice's used memory
        is then sampled on its own (per_mig), and device used memory is their
        sum (the whole GPU's numbers are not meaningful with MIG on)."""
        self.ok = False
        self.mig = {}
        self.mig_total_mb = None
        if not device.startswith("cuda"):
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            idx = int(device.split(":")[1]) if ":" in device else 0
            self.h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            self.total_mb = pynvml.nvmlDeviceGetMemoryInfo(self.h).total / 2**20
            for u in mig_uuids or ():
                try:
                    self.mig[u] = pynvml.nvmlDeviceGetHandleByUUID(u)
                except Exception:
                    self.mig[u] = pynvml.nvmlDeviceGetHandleByUUID(u.encode())
            if self.mig:
                self.mig_total_mb = min(pynvml.nvmlDeviceGetMemoryInfo(h).total
                                        for h in self.mig.values()) / 2**20
            self.ok = True
        except Exception as exc:
            print(f"[mem] NVML unavailable ({exc}); no NVML samples", flush=True)

    def sample(self) -> tuple:
        """(device used MB, {pid: used MB}, {MIG uuid: used MB})."""
        per_mig = {u: self.nv.nvmlDeviceGetMemoryInfo(h).used / 2**20
                   for u, h in self.mig.items()}
        used = (sum(per_mig.values()) if per_mig
                else self.nv.nvmlDeviceGetMemoryInfo(self.h).used / 2**20)
        per_pid = {}
        try:
            for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.h):
                if p.usedGpuMemory:
                    per_pid[p.pid] = per_pid.get(p.pid, 0.0) + p.usedGpuMemory / 2**20
        except Exception:
            pass
        return used, per_pid, per_mig


def _mps_ctl(cmd: str) -> bool:
    import subprocess
    try:
        return subprocess.run(["nvidia-cuda-mps-control"], input=cmd + "\n", text=True,
                              capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def mps_start() -> None:
    """Start the MPS control daemon and wait until it answers (as
    sharing_benefit/tpc/run.sh:ensure_mps_running)."""
    import subprocess
    try:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot start the MPS daemon: {exc}")
    for _ in range(40):
        if MPS_CONTROL.exists() and _mps_ctl("get_server_list"):
            print("    MPS daemon started", flush=True)
            return
        time.sleep(0.5)
    raise RuntimeError("MPS control daemon did not come up")


def mps_stop() -> None:
    """Stop the MPS daemon if it runs, so other modes get the GPU without it
    (as sharing_benefit/tpc/run.sh:stop_devices)."""
    if not MPS_CONTROL.exists():
        return
    _mps_ctl("quit")
    for _ in range(20):
        if not MPS_CONTROL.exists():
            break
        time.sleep(0.5)
    time.sleep(1.0)
    print("    MPS daemon stopped", flush=True)


def mig_slices(given: str) -> list:
    """MIG slice UUIDs: --mig-uuids if given, else every MIG device that
    `nvidia-smi -L` lists (the parent never starts CUDA itself)."""
    if given:
        return [u.strip() for u in given.split(",") if u.strip()]
    import re
    import subprocess
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    return re.findall(r"UUID:\s*(MIG-[^\s)]+)", out)


# ---------------------------------------------------------------------------
# Parent: open-loop load, outcomes, liveness
# ---------------------------------------------------------------------------

def _pct(xs: list, q: float):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1)


def sp_weights(a, tasks: tuple):
    """--sp-split task:weight,... as a list in task order (tasks not named
    get weight 1), or None (even split)."""
    if not a.sp_split:
        return None
    w = {t: 1.0 for t in tasks}
    for item in filter(None, a.sp_split.split(",")):
        t, x = item.split(":")
        if t.strip() not in w:
            raise SystemExit(f"--sp-split: no task {t!r} (tasks: {', '.join(tasks)})")
        w[t.strip()] = float(x)
    return [w[t] for t in tasks]


def bfq_rates(a, tasks: tuple) -> dict:
    """fmvisor's BFQ declared rates (weight = 1 / rate, device/server.py):
    --bfq-rates task:rate,... where given (e.g. A1:1,B:3 favours A1 3:1, as
    fair_share's bfq_3_1), else each task's base load rate."""
    rates = {t: (a.rate_b if t == SURGE else a.rate_a) for t in tasks}
    for item in filter(None, a.bfq_rates.split(",")):
        t, r = item.split(":")
        if t.strip() not in rates:
            raise SystemExit(f"--bfq-rates: no task {t!r} (tasks: {', '.join(tasks)})")
        rates[t.strip()] = float(r)
    return rates


def run_mode(mode: str, a) -> tuple:
    ctx = mp.get_context("spawn")
    tasks = task_names(a.n_tasks)
    victims = tasks[1:]
    cfg = {"device": a.device, "backbone": a.backbone, "res": a.res,
           "max_batch": a.max_batch, "batch_window_s": a.batch_window_ms / 1000,
           "staging": a.staging, "mem_period": a.mem_period, "all_tasks": tasks,
           "fm_chunk": a.fm_chunk,
           # fmvisor's BFQ: declared rates, weight = 1 / rate (device/server.py)
           "rates": bfq_rates(a, tasks)}
    req_qs = {t: ctx.Queue() for t in tasks}
    status_q = ctx.Queue()
    if mode == "s_be":
        layout = {"server": tasks}
        comp_of = {t: "server" for t in tasks}
    elif mode == "fmvisor":
        # The backbone serves every task; each task's own process holds its
        # requests and runs its head. Requests go to (and fail with) the task
        # process; the backbone going down takes every task down.
        layout = {"backbone": tasks, **{f"task-{t}": (t,) for t in tasks}}
        comp_of = {t: f"task-{t}" for t in tasks}
    else:
        layout = {f"inst-{t}": (t,) for t in tasks}
        comp_of = {t: f"inst-{t}" for t in tasks}
    slices = {}
    if mode == "mig":
        # One task per MIG slice, as be with hardware-partitioned memory and SMs.
        found = mig_slices(a.mig_uuids)
        if len(found) < len(tasks):
            raise RuntimeError(f"mig: {len(tasks)} tasks but {len(found)} MIG slices "
                               f"({found}); create them or lower --n-tasks")
        slices = {f"inst-{t}": found[i] for i, t in enumerate(tasks)}
    # sp / sp_mps: each instance's share of the GPU, in instance (= task) order.
    split = sp_weights(a, tasks)            # None: even
    if split:
        mps_pcts = [max(1, round(100 * w / sum(split))) for w in split]
    else:
        mps_pcts = [100 // len(tasks)] * len(tasks)
    evt_qs = {c: ctx.Queue() for c in layout}
    run_stop = ctx.Event()
    if mode == "fmvisor":
        procs = {"backbone": ctx.Process(target=fm_backbone_main, name="backbone",
                                         args=(cfg, tasks, req_qs, evt_qs, status_q,
                                               run_stop))}
    else:
        procs = {c: ctx.Process(target=server_main, name=c,
                                args=(c, ts, cfg, {t: req_qs[t] for t in ts}, evt_qs[c],
                                      status_q,
                                      (i, len(layout), split) if mode == "sp" else None,
                                      slices.get(c),
                                      mps_pcts[i] if mode == "sp_mps" else None))
                 for i, (c, ts) in enumerate(layout.items())}
    up = {c: False for c in layout}
    info, downs, events = {}, [], []
    inflight: dict = {}               # rid -> (task, t_send)
    t_zero = None

    def rel(ts):
        return None if t_zero is None else round(ts - t_zero, 4)

    def mark_down(comp: str, ts: float, why: str) -> None:
        if not up[comp]:
            return
        up[comp] = False
        downs.append({"comp": comp, "t_down": rel(ts), "why": why})
        events.append({"kind": "down", "comp": comp, "t": rel(ts), "note": why})
        print(f"    {comp} down at t={rel(ts)}s: {why}", flush=True)
        for rid, (t, _) in list(inflight.items()):
            if comp_of[t] == comp:
                del inflight[rid]
                events.append({"kind": "fail", "comp": comp, "tenant": t, "rid": rid,
                               "t": rel(ts)})
        if mode == "fmvisor" and comp == "backbone":   # no backbone, no task is served
            for c in layout:
                mark_down(c, ts, "backbone down")

    def drain() -> None:
        for comp, q in evt_qs.items():
            while True:
                try:
                    msg = q.get_nowait()
                except queue.Empty:
                    break
                except Exception:         # left half-written by a dead process
                    break
                if msg[0] == "done":
                    _, t, rid, ts = msg
                    hit = inflight.pop(rid, None)
                    if hit is not None:
                        events.append({"kind": "ok", "tenant": t, "rid": rid, "t": rel(ts),
                                       "lat_ms": round((ts - hit[1]) * 1e3, 2)})
                elif msg[0] == "stats":
                    _, c, ts, per, alloc, resv = msg
                    for t, (n_s, mb_s, n_f, mb_f) in per.items():
                        events.append({"kind": "staged", "comp": c, "tenant": t, "t": rel(ts),
                                       "n": n_s, "mb": mb_s})
                        events.append({"kind": "feat", "comp": c, "tenant": t, "t": rel(ts),
                                       "n": n_f, "mb": mb_f})
                    events.append({"kind": "torch", "comp": c, "t": rel(ts), "mb": alloc,
                                   "note": resv})
                elif msg[0] == "error":
                    _, c, ts, why = msg
                    mark_down(c, ts, why)

    def check_exits() -> None:
        for comp, p in procs.items():
            if p.exitcode is not None and up[comp]:
                drain()                   # its "error" message, if it sent one
                mark_down(comp, time.monotonic(), f"exit code {p.exitcode}")

    def shutdown() -> None:
        run_stop.set()                    # fmvisor: task processes stop without draining
        for q in req_qs.values():
            q.put(None)
        deadline = time.monotonic() + JOIN_S
        for p in procs.values():
            p.join(timeout=max(0.0, deadline - time.monotonic()))
        for p in procs.values():
            if p.is_alive():
                print(f"    {p.name} did not stop; killing", flush=True)
                p.kill()
                p.join(timeout=2)
        # A dead process leaves requests in its queue that nobody will read;
        # without this, the parent waits forever at exit to flush them.
        for q in req_qs.values():
            q.cancel_join_thread()

    print(f"    starting {mode}: {', '.join(procs)}", flush=True)
    for p in procs.values():
        p.start()
    deadline = time.time() + a.startup_timeout
    while not all(up.values()):
        try:
            kind, comp, ts, data = status_q.get(timeout=1.0)
            up[comp] = True
            info[comp] = data
            if "calib" in data:             # fmvisor's task processes carry no model info
                cal = ", ".join(f"b{b}: {v['ms']} ms/+{v['peak_mb']} MB"
                                for b, v in data["calib"].items())
                print(f"      {comp} up ({data['attn']} attention, weights {data['weights_mb']} MB"
                      f"{', ' + data['tpc'] if data.get('tpc') else ''}): {cal}", flush=True)
        except queue.Empty:
            pass
        if any(p.exitcode is not None for p in procs.values()) or time.time() > deadline:
            shutdown()
            raise RuntimeError(f"{mode}: serving failed to start (up: {[c for c, u in up.items() if u]})")

    arr = sorted((ts, t) for t in tasks for ts in arrivals(t, a))
    mem = GpuMemSampler(a.device, list(slices.values()))
    mig_comp = {u: c for c, u in slices.items()}
    pid_comp = {p.pid: c for c, p in procs.items()}
    for c, d in info.items():               # fmvisor's task processes
        if "pid" in d:
            pid_comp[d["pid"]] = c
    peak_used = 0.0
    sent = {t: 0 for t in tasks}
    send_lag: list = []
    offered = {t: [0] * int(a.duration) for t in tasks}   # sent per 1 s bin, t >= 0
    t_zero = time.monotonic() + 0.5 + a.warmup
    t_end = t_zero + a.duration
    next_mem = t_zero - a.warmup
    i = 0
    while True:
        now = time.monotonic()
        if now >= t_end:
            break
        while i < len(arr) and t_zero + arr[i][0] <= now:
            ts_rel, t = arr[i]
            rid = f"{t}-{i}"
            i += 1
            send_lag.append(now - (t_zero + ts_rel))   # how late the generator is
            sent[t] += 1
            if 0 <= ts_rel < a.duration:
                offered[t][int(ts_rel)] += 1
            comp = comp_of[t]
            if up[comp]:
                inflight[rid] = (t, now)
                req_qs[t].put(rid)
            else:
                events.append({"kind": "fail", "comp": comp, "tenant": t, "rid": rid,
                               "t": rel(now)})
        if mem.ok and now >= next_mem:
            used, per_pid, per_mig = mem.sample()
            peak_used = max(peak_used, used)
            events.append({"kind": "mem", "comp": "gpu_used", "t": rel(now), "mb": round(used, 1)})
            if per_mig:                     # mig: a slice is one task's process
                for u, mb in per_mig.items():
                    events.append({"kind": "mem", "comp": mig_comp[u], "t": rel(now),
                                   "mb": round(mb, 1)})
            else:
                for pid, mb in per_pid.items():
                    if pid in pid_comp:
                        events.append({"kind": "mem", "comp": pid_comp[pid], "t": rel(now),
                                       "mb": round(mb, 1)})
            next_mem += a.mem_period
        drain()
        check_exits()
        time.sleep(0.0005)
    unfinished = {t: sum(1 for tt, _ in inflight.values() if tt == t) for t in tasks}
    shutdown()
    drain()

    def stats(ts: tuple, lo: float, hi: float) -> dict:
        """Completed req/s (summed over ts) and latency percentiles (pooled)."""
        ok = [e for e in events if e["kind"] == "ok" and e["tenant"] in ts
              and e["t"] is not None and lo <= e["t"] < hi]
        lat = [e["lat_ms"] for e in ok]
        return {"rps": round(len(ok) / (hi - lo), 2), "lat_p50_ms": _pct(lat, 0.5),
                "lat_p99_ms": _pct(lat, 0.99)}

    def count(t: str, kind: str) -> int:
        return sum(e["kind"] == kind and e.get("tenant") == t and e["t"] is not None
                   and 0 <= e["t"] < a.duration for e in events)

    summary = {
        "mode": mode, "tasks": list(tasks), "victims": list(victims), "surging": SURGE,
        "layout": {c: list(ts) for c, ts in layout.items()},
        "n_tasks": a.n_tasks, "backbone": a.backbone, "res": a.res, "staging": a.staging,
        "max_batch": a.max_batch, "batch_window_ms": a.batch_window_ms,
        "fm_chunk": a.fm_chunk if mode == "fmvisor" else None,
        "fm_policy": "bfq" if mode == "fmvisor" else None,
        "sp_split": ((dict(zip(tasks, split)) if split else "even")
                     if mode in ("sp", "sp_mps") else None),
        "sp_mps_pct": dict(zip(tasks, mps_pcts)) if mode == "sp_mps" else None,
        "fm_bfq_weights": ({t: 1.0 / r for t, r in cfg["rates"].items()}
                           if mode == "fmvisor" else None),
        "rate_a": a.rate_a, "rate_b": a.rate_b, "rate_b_max": a.rate_b_max,
        "ramp_at": a.ramp_at, "ramp_s": a.ramp_s, "warmup": a.warmup,
        "duration": a.duration, "seed": a.seed, "device": a.device,
        "servers": info,
        "gpu_total_mb": round(mem.total_mb, 1) if mem.ok else None,
        "mig_slices": slices or None,
        "mig_slice_mb": round(mem.mig_total_mb, 1) if mem.ok and mem.mig_total_mb else None,
        "gpu_peak_used_mb": round(peak_used, 1) if mem.ok else None,
        "downs": downs,
        "sent": sent, "offered_per_s": offered,
        # Request sent this late after its scheduled arrival: if large, the
        # load generator could not keep up with the offered rate.
        "send_lag_ms": {"p50": _pct([1e3 * x for x in send_lag], 0.5),
                        "p99": _pct([1e3 * x for x in send_lag], 0.99),
                        "max": round(1e3 * max(send_lag), 1) if send_lag else None},
        "ok": {t: count(t, "ok") for t in tasks},
        "fail": {t: count(t, "fail") for t in tasks},
        "unfinished_at_end": unfinished,
        "stats": {f"{name}_{w}": stats(ts, lo, hi)
                  for name, ts in (("surging", (SURGE,)), ("victims", victims))
                  for w, (lo, hi) in (("before", (0.0, a.ramp_at)),
                                      ("surge", (a.ramp_at, a.duration)))},
    }
    return events, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--backbone", default="dino-base", choices=["dino-base", "dino-large"])
    ap.add_argument("--n-tasks", type=int, default=8,
                    help="1 surging task + n-1 victims (RTVSntask/vision: 8)")
    ap.add_argument("--res", type=int, default=224,
                    help="image side in px (multiple of 14; RTVSntask/vision: 224)")
    ap.add_argument("--staging", choices=STAGINGS, default="gpu")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--sp-split", default="",
                    help="sp / sp_mps: share of the GPU per task, task:weight,... (e.g. "
                         "A1:5,B:1, the ratio --bfq-rates A1:1,B:5 gives FMVisor; "
                         "unnamed tasks weigh 1); default: an even split")
    ap.add_argument("--bfq-rates", default="",
                    help="fmvisor: BFQ declared rates task:rate,... (weight = 1 / rate; "
                         "e.g. A1:1,B:3); default: each task's base load rate")
    ap.add_argument("--mig-uuids", default="",
                    help="mig: comma-separated MIG slice UUIDs, one per task "
                         "(default: every slice `nvidia-smi -L` lists)")
    ap.add_argument("--fm-chunk", type=int, default=1,
                    help="fmvisor: slots a task process allocates at a time (1: one "
                         "per request as it arrives, as s_be / be stage theirs)")
    ap.add_argument("--batch-window-ms", type=float, default=0.0,
                    help="0 = batch whatever is staged (device server default)")
    ap.add_argument("--rate-a", type=float, default=5.0, help="req/s per victim")
    ap.add_argument("--rate-b", type=float, default=5.0)
    ap.add_argument("--rate-b-max", type=float, default=1000.0)
    ap.add_argument("--ramp-at", type=float, default=15.0, help="s after t = 0")
    ap.add_argument("--ramp-s", type=float, default=15.0)
    ap.add_argument("--warmup", type=float, default=10.0, help="load at base rates before t = 0")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--mem-period", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--startup-timeout", type=float, default=900.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--exp-dir", type=Path, default=EXP_DIR / "results")
    a = ap.parse_args()

    modes = a.modes.split(",")
    assert all(m in MODES for m in modes), modes
    assert a.res % 14 == 0, "--res must be a multiple of 14 (DINOv2 patch size)"
    assert 0 < a.ramp_at < a.duration
    assert a.n_tasks >= 2
    bfq_rates(a, task_names(a.n_tasks))          # check --bfq-rates before any mode runs
    sp_weights(a, task_names(a.n_tasks))         # and --sp-split
    out_dir = a.exp_dir / a.staging
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{a.backbone} at {a.res}px ({3 * a.res ** 2 * 4 / 2**20:.2f} MB/request), "
          f"staging={a.staging}, max batch {a.max_batch}; {a.n_tasks} tasks: "
          f"{a.n_tasks - 1} victims at {a.rate_a} req/s each; "
          f"B {a.rate_b} -> {a.rate_b_max} req/s over [{a.ramp_at}, {a.ramp_at + a.ramp_s}] s "
          f"of {a.duration} s -> {out_dir}", flush=True)

    for mode in modes:
        print(f"--- {mode}", flush=True)
        try:
            # MPS only for sp_mps: every other mode runs without the daemon.
            mps_stop()
            if mode == "sp_mps":
                mps_start()
            events, summary = run_mode(mode, a)
        except RuntimeError as e:
            print(f"    FAILED: {e}; skipping", flush=True)
            continue
        finally:
            if mode == "sp_mps":
                mps_stop()
        with open(out_dir / f"events_{mode}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["kind", "comp", "tenant", "rid", "t",
                                               "lat_ms", "n", "mb", "note"])
            wr.writeheader()
            wr.writerows(events)
        with open(out_dir / "summary.jsonl", "a") as f:
            f.write(json.dumps(summary) + "\n")
        st = summary["stats"]
        v_fail = sum(summary["fail"][t] for t in summary["victims"])
        print(f"    victims: {st['victims_before']['rps']} -> {st['victims_surge']['rps']} req/s, "
              f"p50 {st['victims_before']['lat_p50_ms']} -> {st['victims_surge']['lat_p50_ms']} ms, "
              f"{v_fail} failed; "
              f"B: {st['surging_before']['rps']} -> {st['surging_surge']['rps']} req/s, "
              f"{summary['fail'][SURGE]} failed; "
              f"peak GPU used={summary['gpu_peak_used_mb']} MB "
              f"send lag p99/max={summary['send_lag_ms']['p99']}/{summary['send_lag_ms']['max']} ms "
              f"down={[(d['comp'], d['t_down']) for d in summary['downs']]}", flush=True)


if __name__ == "__main__":
    main()
