#!/usr/bin/env python3
"""fault_isolation_tsfm/run.py — does a fault in one task's decoder reach the others?

N tasks (default 8) on one MOMENT-large backbone, each with its own decoder,
built with isolation_mechanisms/run.py (ecgclass-shaped MLP heads, random
weights seeded per task, random inputs of the real shape, so every task
batches with every other). P (task 0) faults; A1..A{N-1} are the victims.

Client in the parent, serving in children. The parent never touches CUDA. It
runs --clients closed-loop clients per task (default 1): each sends its next
request as soon as the last one completes, or --retry-backoff s after it
failed. It records every request's outcome with a timestamp and watches the
serving processes. Requests and replies travel on one queue per task in each
direction, so a process that dies mid-write can only corrupt its own task's
channels.

At --fault-at s, P's decoder faults (--fault-kind):
    gpu   once: an out-of-bounds embedding lookup in the decoder: on CUDA a
          device-side assert, which poisons the CUDA context it runs in.
    segv  once: a null dereference on the CPU before the decoder runs: SIGSEGV.
    leak  from then on: the decoder keeps GPU memory it never frees, at
          --leak-mb-per-s of wall time (so the ramp is the same in every
          mode, whatever P's request rate), and otherwise works normally.
          When an allocation fails, the CUDA out-of-memory error is not
          caught: the process serving it stops, as an engine that treats
          OOM as fatal would. Which process that is, and who else it
          serves, is what differs between modes.

GPU memory is sampled by the parent through NVML every --mem-period s:
device used, and per serving process ("mem" events, MB).

Modes (where P's decoder runs is what the fault takes down):
    s_be     one process: a backbone thread with cross-task batching and one
             thread + CUDA stream per decoder (isolation_mechanisms
             thread_stream; device server --isolation-mode shared). An
             unhandled error in any thread ends the process, as a crashed
             server would.
    be       one process per task, each with its own CUDA context, backbone
             copy and decoder (isolation_mechanisms instance_per_task;
             sharing_benefit no_sharing).
    sp       be, with each process's stream masked to an even share of the
             GPU's TPCs through libsmctrl: total // N each, the remainder
             unused (sharing_benefit no_sharing_tpc; T4, 8 tasks: 2 each).
             No MPS, so the contexts still time-slice.
    fmvisor  one backbone process with cross-task batching and one process
             per task running its decoder in its own CUDA context. Input and
             feature slots live in the backbone process's VRAM and reach the
             task processes through CUDA IPC (isolation_mechanisms
             process_context_ipc). The backbone process reports a dead task
             process and stops serving it; the other tasks go on.

No recovery: whatever dies stays down. A request fails if a component serving
it dies with it in flight, or if it is sent while one is down. Nothing is
replayed.

Output, same format as lora_fair_share/fault.py so its fault_plot.py plots it
(--exp-dir <this exp-dir>; the req/s plot is the meaningful one, n_tok = 1):
    <exp-dir>/fault/          --fault-kind gpu
    <exp-dir>/fault_segv/     --fault-kind segv
    <exp-dir>/fault_leak/     --fault-kind leak (plot with mem_plot.py)
        events_<mode>.csv, summary.jsonl

Usage (from serving/, FMTK/src on PYTHONPATH; run.sh sets it up):
    python experiments/fault_isolation_tsfm/run.py --modes s_be,be,sp,fmvisor
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import importlib.util
import json
import os
import queue
import random
import sys
import threading
import time
import traceback
from pathlib import Path

import torch
import torch.multiprocessing as mp

EXP_DIR = Path(__file__).resolve().parent
SERVING_DIR = EXP_DIR.parents[1]
TPC_LIB_DIR = Path(os.environ.get(
    "TPC_LIB_DIR", SERVING_DIR.parents[1] / "TPC_controller" / "tpc_controller"))

# isolation_mechanisms/run.py supplies the models (backbone, heads, inputs,
# warm-up). Loaded by path: both scripts are named run.py.
_spec = importlib.util.spec_from_file_location(
    "isolation_mechanisms_run", SERVING_DIR / "experiments" / "isolation_mechanisms" / "run.py")
im = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = im
_spec.loader.exec_module(im)

FAULT_TASK = "P"
MODES = ("s_be", "be", "sp", "fmvisor")
FAULT_KINDS = ("gpu", "segv", "leak")
TASK_JOIN_S = 10.0       # fmvisor backbone: all task processes, at stop
BACKBONE_JOIN_S = 30.0   # parent: any serving process, at stop


def task_names(n: int) -> list:
    return [FAULT_TASK] + [f"A{i}" for i in range(1, n)]


def fault_dir(exp_dir: Path, kind: str) -> Path:
    """Mirrors lora_fair_share/fault_plot.py --fault-kind."""
    return exp_dir / ("fault" if kind == "gpu" else f"fault_{kind}")


# ---------------------------------------------------------------------------
# The faulty decoder: same wrapper in every mode, only where it runs differs
# ---------------------------------------------------------------------------

class FaultyDecoder:
    """Wraps P's decoder. gpu / segv fire once, on the first call at or after
    t_fault; leak runs on every call from t_fault on.

    gpu / segv: the "fault" event goes on P's own reply queue and is flushed
    first: a SIGSEGV (and os._exit in s_be) skips the queue's feeder thread,
    so an unflushed event would be lost. The queue is not used again: the
    process dies next.

    leak: each call allocates leak_mb_s x (seconds since the last allocation)
    MB on the GPU and keeps it, so the leak follows wall time even though it
    only grows when P's decoder runs. Reports ("leak", t, total MB)."""

    def __init__(self, head, kind: str, t_fault, rep_q, device: str, leak_mb_s: float = 0.0):
        self.head, self.kind, self.t_fault = head, kind, t_fault
        self.rep_q, self.device = rep_q, device
        self.fired = False
        self.leak_mb_s, self.leaked, self.leaked_mb, self.t_leak = leak_mb_s, [], 0.0, None

    def _leak(self, feat: torch.Tensor, tf: float, now: float) -> None:
        if self.t_leak is None:
            self.t_leak = tf
            self.rep_q.put(("fault", now))
        mb = self.leak_mb_s * (now - self.t_leak)
        if mb < 1.0:
            return
        n = int(mb * 2**20) // 4
        # Not caught: an OOM here (or anywhere else) stops the process.
        self.leaked.append(torch.empty(n, dtype=torch.float32, device=feat.device))
        self.leaked_mb += n * 4 / 2**20
        self.t_leak = now
        self.rep_q.put(("leak", now, self.leaked_mb))

    def __call__(self, feat: torch.Tensor):
        tf = self.t_fault.value
        now = time.monotonic()
        if self.kind == "leak":
            if tf > 0 and now >= tf:
                self._leak(feat, tf, now)
            return self.head(feat)
        if not self.fired and tf > 0 and now >= tf:
            self.fired = True
            self.rep_q.put(("fault", time.monotonic()))
            self.rep_q.close()
            self.rep_q.join_thread()
            if self.kind == "segv":
                ctypes.string_at(0)          # the process dies here
            table = feat.reshape(-1, feat.shape[-1])
            bad = torch.full((1,), 1 << 30, dtype=torch.long, device=feat.device)
            torch.nn.functional.embedding(bad, table)   # out of bounds
            im._sync(self.device)            # surfaces the assert (IndexError on CPU)
        return self.head(feat)


def _decoder(t: int, cfg: dict, t_fault, rep_q):
    head = im.make_head(t, cfg["backbone"], cfg["classes"], cfg["device"])
    if t == 0:
        return head, FaultyDecoder(head, cfg["fault_kind"], t_fault, rep_q, cfg["device"],
                                   cfg["leak_mb_s"])
    return head, head


def _set_device(device: str) -> None:
    if im._is_cuda(device):
        torch.cuda.set_device(device)


def _next_request(req_q, parent_pid: int):
    """Block for the next request id; None means stop. Also None once the
    parent is gone, so a serving process never outlives the run (an orphan
    keeps its CUDA context and holds run.sh's tee open)."""
    while True:
        try:
            return req_q.get(timeout=0.5)
        except queue.Empty:
            if os.getppid() != parent_pid:
                return None


# ---------------------------------------------------------------------------
# s_be: one process, threads + streams
# ---------------------------------------------------------------------------

def _die_on_thread_error(hook_args) -> None:
    traceback.print_exception(hook_args.exc_type, hook_args.exc_value, hook_args.exc_traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def s_be_main(cfg: dict, req_qs, rep_qs, status_q, t_fault) -> None:
    device, n = cfg["device"], cfg["n"]
    cuda = im._is_cuda(device)
    _set_device(device)
    backbone = im.make_backbone(cfg["backbone"], device)
    decs = [_decoder(t, cfg, t_fault, rep_qs[t]) for t in range(n)]
    xs = [im.make_input(t, cfg["backbone"], device) for t in range(n)]
    im.warm_up(backbone, [h for h, _ in decs], device, cfg["max_batch"], cfg["backbone"])
    bb_stream = torch.cuda.Stream(device=device) if cuda else None
    task_streams = [torch.cuda.Stream(device=device) if cuda else None for _ in range(n)]
    local_q: queue.Queue = queue.Queue()
    task_qs = [queue.Queue() for _ in range(n)]
    stop = threading.Event()
    threading.excepthook = _die_on_thread_error
    parent_pid = os.getppid()

    def intake(t: int):
        while True:
            rid = _next_request(req_qs[t], parent_pid)
            if rid is None:
                return
            local_q.put((t, rid))

    def backbone_loop():
        ctx = torch.cuda.stream(bb_stream) if cuda else im.nullcontext()
        with ctx, torch.inference_mode():
            while not stop.is_set():
                batch = im._collect_batch(local_q, cfg["max_batch"], cfg["batch_window_s"], stop)
                if not batch:
                    continue
                x = torch.cat([xs[t].to(device, non_blocking=True) for t, _ in batch])
                feats = backbone(x)
                event = None
                if cuda:
                    event = torch.cuda.Event()
                    event.record(bb_stream)
                for j, (t, rid) in enumerate(batch):
                    task_qs[t].put((feats[j:j + 1], event, rid))

    def task_worker(t: int):
        stream = task_streams[t]
        if stream is not None:
            torch.cuda.set_stream(stream)
        dec = decs[t][1]
        with torch.inference_mode():
            while True:
                item = task_qs[t].get()
                if item is None:
                    return
                feat, event, rid = item
                if stream is not None:
                    stream.wait_event(event)
                    feat.record_stream(stream)
                dec(feat)
                im._sync(device, stream)
                rep_qs[t].put(("done", rid, time.monotonic()))

    intakes = [threading.Thread(target=intake, args=(t,), daemon=True) for t in range(n)]
    workers = [threading.Thread(target=backbone_loop, daemon=True, name="backbone")]
    workers += [threading.Thread(target=task_worker, args=(t,), daemon=True, name=f"task-{t}")
                for t in range(n)]
    for th in intakes + workers:
        th.start()
    status_q.put(("up", "server", time.monotonic()))
    for th in intakes:          # the parent stops each task with None
        th.join()
    stop.set()
    for q in task_qs:
        q.put(None)
    for th in workers:
        th.join(timeout=5)


# ---------------------------------------------------------------------------
# be / sp: one whole deployment per task
# ---------------------------------------------------------------------------

def _masked_stream(i: int, n_inst: int, device: str):
    """A CUDA stream masked to instance i's even share of the GPU's TPCs
    (libsmctrl): total // n each, contiguous, the remainder left unused, as
    sharing_benefit/tpc/run.sh splits them (T4, 8 tasks: 2 TPCs each, 4 idle).
    lora_fair_share/run.py:pin_tpcs without its vLLM graph-capture patch."""
    lib = ctypes.CDLL(str(TPC_LIB_DIR / "libsmctrl" / "libsmctrl.so"))
    lib.libsmctrl_set_stream_mask_ext.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
    lib.libsmctrl_set_stream_mask_ext.restype = None
    lib.libsmctrl_get_tpc_info_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int]
    lib.libsmctrl_get_tpc_info_cuda.restype = ctypes.c_int
    n = ctypes.c_uint32()
    if lib.libsmctrl_get_tpc_info_cuda(ctypes.byref(n), torch.device(device).index or 0) != 0:
        raise RuntimeError("libsmctrl_get_tpc_info_cuda failed")
    per = n.value // n_inst
    if per == 0:
        raise ValueError(f"{n_inst} instances but only {n.value} TPCs")
    tpcs = list(range(i * per, (i + 1) * per))
    stream = torch.cuda.Stream(device=device)
    enable = sum(1 << i for i in tpcs)
    disable = ~enable & ((1 << 128) - 1)   # set bit = TPC disabled
    lib.libsmctrl_set_stream_mask_ext(
        ctypes.c_void_p(stream.cuda_stream),
        ctypes.c_uint64(disable & 0xFFFFFFFFFFFFFFFF),
        ctypes.c_uint64(disable >> 64))
    print(f"[sp] TPCs {tpcs[0]}..{tpcs[-1]} of {n.value}", flush=True)
    return stream


def instance_main(t: int, comp: str, cfg: dict, req_q, rep_q, status_q, t_fault, tpcs) -> None:
    device = cfg["device"]
    cuda = im._is_cuda(device)
    _set_device(device)
    torch.set_num_threads(1)
    if tpcs is not None:
        stream = _masked_stream(*tpcs, device)
    else:
        stream = torch.cuda.Stream(device=device) if cuda else None
    stream_ctx = torch.cuda.stream(stream) if stream is not None else im.nullcontext()
    with stream_ctx:
        backbone = im.make_backbone(cfg["backbone"], device)
        head, dec = _decoder(t, cfg, t_fault, rep_q)
        x = im.make_input(t, cfg["backbone"], device)
        im.warm_up(backbone, [head], device, 1, cfg["backbone"])
    status_q.put(("up", comp, time.monotonic()))
    parent_pid = os.getppid()
    with stream_ctx, torch.inference_mode():
        while True:
            rid = _next_request(req_q, parent_pid)
            if rid is None:
                return
            dec(backbone(x.to(device, non_blocking=True)))
            im._sync(device, stream)
            rep_q.put(("done", rid, time.monotonic()))


# ---------------------------------------------------------------------------
# fmvisor: backbone process + one process per task, slots through CUDA IPC
# ---------------------------------------------------------------------------

def fm_task_main(t: int, cfg: dict, inputs, features, req_q, rep_q, sched_q, done,
                 ready_q, t_fault, stop) -> None:
    """Own process => own CUDA context. `inputs`/`features` live in the
    backbone process's VRAM and arrived through CUDA IPC; slot t is ours.

    Every wait is bounded and gives up once the run stops or the backbone
    process is gone: a task process left waiting would outlive the run, keep
    its CUDA context and the IPC mappings, and hold the parent's stdout open
    (run.sh's tee would never finish)."""
    device = cfg["device"]
    _set_device(device)
    torch.set_num_threads(1)
    backbone_pid = os.getppid()
    head, dec = _decoder(t, cfg, t_fault, rep_q)
    x = im.make_input(t, cfg["backbone"], device)
    with torch.inference_mode():
        head(features[t:t + 1])      # warm the decoder's kernels in this context
    im._sync(device)
    ready_q.put(t)

    def over() -> bool:
        return stop.is_set() or os.getppid() != backbone_pid

    with torch.inference_mode():
        while not over():
            try:
                rid = req_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if rid is None:
                return
            inputs[t].copy_(x[0], non_blocking=True)
            # Host-side queues do not order kernels across contexts: the input
            # must be in VRAM before the backbone reads it.
            im._sync(device)
            sched_q.put(t)
            while not done.wait(0.5):
                if over():
                    return
            done.clear()
            dec(features[t:t + 1])
            im._sync(device)
            rep_q.put(("done", rid, time.monotonic()))


def fm_backbone_main(cfg: dict, req_qs, rep_qs, status_q, t_fault, stop) -> None:
    device, n, names = cfg["device"], cfg["n"], cfg["names"]
    cuda = im._is_cuda(device)
    _set_device(device)
    ctx = mp.get_context("spawn")
    # Slot t belongs to task t (one outstanding request per task and client).
    inputs = torch.empty((n, *im.input_shape(cfg["backbone"])), device=device)
    features = torch.zeros((n, *im.feature_shape(cfg["backbone"])), device=device)
    if not cuda:
        inputs.share_memory_()
        features.share_memory_()
    sched_qs = [ctx.Queue() for _ in range(n)]   # task t -> backbone, one per task
    done = [ctx.Event() for _ in range(n)]
    ready_q = ctx.Queue()
    procs = [ctx.Process(target=fm_task_main, name=f"task-{names[t]}", daemon=True,
                         args=(t, cfg, inputs, features, req_qs[t], rep_qs[t], sched_qs[t],
                               done[t], ready_q, t_fault if t == 0 else None, stop))
             for t in range(n)]
    for p in procs:
        p.start()
    backbone = im.make_backbone(cfg["backbone"], device)
    im.warm_up(backbone, [], device, cfg["max_batch"], cfg["backbone"])
    bb_stream = torch.cuda.Stream(device=device) if cuda else None
    for _ in range(n):
        while True:
            try:
                ready_q.get(timeout=1.0)
                break
            except queue.Empty:
                if any(p.exitcode is not None for p in procs):
                    raise RuntimeError("a task process died during startup")
    up = [True] * n
    for t in range(n):
        status_q.put(("up", f"task-{names[t]}", time.monotonic()))
    status_q.put(("up", "backbone", time.monotonic()))

    def poll(slots: list) -> None:
        for t in range(n):
            if not up[t]:
                continue
            try:
                slots.append(sched_qs[t].get_nowait())
            except queue.Empty:
                pass

    window = cfg["batch_window_s"]
    parent_pid = os.getppid()
    ctx_stream = torch.cuda.stream(bb_stream) if cuda else im.nullcontext()
    with ctx_stream, torch.inference_mode():
        while not stop.is_set() and os.getppid() == parent_pid:
            # A dead task is reported once and never batched again.
            for t in range(n):
                if up[t] and procs[t].exitcode is not None:
                    up[t] = False
                    status_q.put(("down", f"task-{names[t]}", time.monotonic()))
            slots: list = []
            poll(slots)
            if not slots:
                time.sleep(0.0002)
                continue
            deadline = time.monotonic() + window
            while len(slots) < cfg["max_batch"] and time.monotonic() < deadline:
                poll(slots)
            slots = slots[:cfg["max_batch"]]
            idx = torch.tensor(slots, dtype=torch.long, device=device)
            feats = backbone(inputs.index_select(0, idx))
            features.index_copy_(0, idx, feats)
            # Features must be written before a task process reads them.
            im._sync(device, bb_stream)
            for s in slots:
                done[s].set()
    # Tasks see `stop` within 0.5 s; one shared deadline, well inside the
    # parent's wait for this process (BACKBONE_JOIN_S).
    deadline = time.monotonic() + TASK_JOIN_S
    for p in procs:
        p.join(timeout=max(0.0, deadline - time.monotonic()))
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join(timeout=1)


# ---------------------------------------------------------------------------
# Parent: GPU memory through NVML (no CUDA context in the parent)
# ---------------------------------------------------------------------------

class GpuMemSampler:
    """Device used memory and per-process used memory on the run's GPU.

    NVML's per-process numbers include each process's CUDA context and
    everything its caching allocator has reserved. They can be missing in a
    container whose PID namespace differs from the host's; device used memory
    still works then. The NVML index is the CUDA index, so this assumes
    CUDA_VISIBLE_DEVICES does not reorder GPUs."""

    def __init__(self, device: str):
        self.ok = False
        if not im._is_cuda(device):
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(torch.device(device).index or 0)
            self.total_mb = pynvml.nvmlDeviceGetMemoryInfo(self.h).total / 2**20
            self.ok = True
        except Exception as exc:
            print(f"[mem] NVML unavailable ({exc}); no memory samples", flush=True)

    def sample(self) -> tuple[float, dict]:
        """(device used MB, {pid: used MB})."""
        used = self.nv.nvmlDeviceGetMemoryInfo(self.h).used / 2**20
        per_pid = {}
        try:
            for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.h):
                if p.usedGpuMemory:
                    per_pid[p.pid] = per_pid.get(p.pid, 0.0) + p.usedGpuMemory / 2**20
        except Exception:
            pass
        return used, per_pid


def _child_pids(pid: int) -> list:
    """Children of pid (fmvisor's task processes), via psutil if installed."""
    try:
        import psutil
        return [c.pid for c in psutil.Process(pid).children()]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Parent: closed-loop clients, outcomes, liveness
# ---------------------------------------------------------------------------

def run_mode(mode: str, args) -> tuple[list, dict]:
    ctx = mp.get_context("spawn")
    n = args.n_tasks
    names = task_names(n)
    cfg = {"device": args.device, "backbone": args.backbone, "classes": 10, "n": n,
           "names": names, "max_batch": args.max_batch or n * args.clients,
           "batch_window_s": args.batch_window_ms / 1000, "fault_kind": args.fault_kind,
           "leak_mb_s": args.leak_mb_per_s}
    if mode == "fmvisor" and args.clients != 1:
        raise RuntimeError("fmvisor has one IPC slot per task: --clients must be 1")
    req_qs = [ctx.Queue() for _ in range(n)]
    rep_qs = [ctx.Queue() for _ in range(n)]
    status_q = ctx.Queue()
    t_fault = ctx.Value("d", 0.0)
    stop = ctx.Event()

    procs: dict = {}      # direct children, watched through exitcode
    if mode == "s_be":
        procs["server"] = ctx.Process(target=s_be_main, name="server",
                                      args=(cfg, req_qs, rep_qs, status_q, t_fault))
        chain = {t: ["server"] for t in range(n)}
    elif mode in ("be", "sp"):
        for t in range(n):
            comp = f"inst-{names[t]}"
            tpcs = (t, n) if mode == "sp" else None   # instance t of n
            procs[comp] = ctx.Process(target=instance_main, name=comp,
                                      args=(t, comp, cfg, req_qs[t], rep_qs[t], status_q,
                                            t_fault if t == 0 else None, tpcs))
        chain = {t: [f"inst-{names[t]}"] for t in range(n)}
    else:
        procs["backbone"] = ctx.Process(target=fm_backbone_main, name="backbone",
                                        args=(cfg, req_qs, rep_qs, status_q, t_fault, stop))
        chain = {t: ["backbone", f"task-{names[t]}"] for t in range(n)}
    comps = sorted({c for cs in chain.values() for c in cs})
    status = {c: False for c in comps}

    events, inflight, downs = [], {}, []   # inflight: rid -> (t, client, t_send)
    t_zero = None

    def rel(ts):
        return None if t_zero is None else ts - t_zero

    def mark(comp: str, up: bool, ts: float) -> None:
        if status.get(comp) == up:
            return
        status[comp] = up
        events.append({"kind": "up" if up else "down", "comp": comp, "t": rel(ts)})
        if up:
            return
        downs.append({"comp": comp, "t_down": rel(ts), "t_up": None})
        print(f"    {comp} down at t={rel(ts) if t_zero else float('nan'):.1f}s", flush=True)
        for rid, (t, ci, _) in list(inflight.items()):
            if comp in chain[t]:
                del inflight[rid]
                events.append({"kind": "fail", "comp": comp, "tenant": names[t],
                               "rid": rid, "t": rel(ts)})
                clients[ci]["busy"] = False
                clients[ci]["next"] = ts + args.retry_backoff
        if comp == "backbone":            # its task processes die with it
            for c in comps:
                if c.startswith("task-") and status[c]:
                    mark(c, False, ts)

    def drain_status() -> None:
        while True:
            try:
                kind, comp, ts = status_q.get_nowait()
            except queue.Empty:
                return
            mark(comp, kind == "up", ts)

    def drain_replies() -> None:
        for t in range(n):
            while True:
                try:
                    msg = rep_qs[t].get_nowait()
                except queue.Empty:
                    break
                except Exception:   # a channel left half-written by a dead process
                    break
                if msg[0] == "fault":
                    events.append({"kind": "fault", "tenant": names[t], "t": rel(msg[1])})
                    continue
                if msg[0] == "leak":
                    events.append({"kind": "leak", "tenant": names[t], "t": rel(msg[1]),
                                   "mb": round(msg[2], 1)})
                    continue
                _, rid, ts = msg
                hit = inflight.pop(rid, None)
                if hit is None:
                    continue
                _, ci, t_send = hit
                events.append({"kind": "ok", "tenant": names[t], "rid": rid, "t": rel(ts),
                               "n_tok": 1, "lat_ms": round((ts - t_send) * 1e3, 3)})
                clients[ci]["busy"] = False
                clients[ci]["next"] = ts + args.think_time

    def check_exits() -> None:
        for comp, p in procs.items():
            if p.exitcode is not None and status[comp]:
                print(f"    {comp} exited ({p.exitcode})", flush=True)
                mark(comp, False, time.monotonic())

    def shutdown() -> None:
        stop.set()
        for q in req_qs:
            q.put(None)
        deadline = time.monotonic() + BACKBONE_JOIN_S
        for p in procs.values():
            p.join(timeout=max(0.0, deadline - time.monotonic()))
        for p in procs.values():
            if p.is_alive():
                print(f"    {p.name} did not stop; killing", flush=True)
                p.kill()
                p.join(timeout=2)

    clients = [{"t": t, "busy": False, "next": None}
               for t in range(n) for _ in range(args.clients)]

    print(f"    starting {mode}: {', '.join(procs)}", flush=True)
    for p in procs.values():
        p.start()
    deadline = time.time() + args.startup_timeout
    while not all(status.values()):
        try:
            kind, comp, ts = status_q.get(timeout=1.0)
            mark(comp, kind == "up", ts)
        except queue.Empty:
            pass
        if any(p.exitcode is not None for p in procs.values()) or time.time() > deadline:
            shutdown()
            raise RuntimeError(f"{mode}: serving failed to start "
                               f"(up: {[c for c, s in status.items() if s]})")
    events = [e for e in events if e["kind"] != "up"]   # startup noise

    mem = GpuMemSampler(args.device)
    pid_comp = {p.pid: comp for comp, p in procs.items()}
    if mode == "fmvisor":   # task processes are the backbone's children
        for pid in _child_pids(procs["backbone"].pid):
            pid_comp.setdefault(pid, "task-?")
    peak_used = 0.0

    def sample_mem(now: float) -> None:
        nonlocal peak_used
        used, per_pid = mem.sample()
        peak_used = max(peak_used, used)
        events.append({"kind": "mem", "comp": "gpu_used", "t": rel(now), "mb": round(used, 1)})
        for pid, mb in per_pid.items():
            if pid in pid_comp:
                events.append({"kind": "mem", "comp": pid_comp[pid], "t": rel(now),
                               "mb": round(mb, 1)})

    t_start = time.monotonic() + 0.5
    t_zero = t_start + args.warmup
    t_end = t_zero + args.duration
    t_fault.value = t_zero + args.fault_at
    stagger = random.Random(args.seed)
    for c in clients:
        c["next"] = t_start + 0.05 * stagger.random()
    seq = 0
    next_mem = t_start

    while True:
        now = time.monotonic()
        if now >= t_end:
            break
        if mem.ok and now >= next_mem:
            sample_mem(now)
            next_mem += args.mem_period
        for ci, c in enumerate(clients):
            if c["busy"] or c["next"] > now:
                continue
            t = c["t"]
            rid = f"{names[t]}-{seq}"
            seq += 1
            if all(status[x] for x in chain[t]):
                c["busy"] = True
                inflight[rid] = (t, ci, now)
                req_qs[t].put(rid)
            else:
                down = next(x for x in chain[t] if not status[x])
                events.append({"kind": "fail", "comp": down, "tenant": names[t],
                               "rid": rid, "t": rel(now)})
                c["next"] = now + args.retry_backoff
        drain_status()
        drain_replies()
        check_exits()
        time.sleep(0.0002)

    shutdown()

    def count(t: str, kind: str, lo: float = 0.0, hi: float = args.duration) -> int:
        return sum(e.get("tenant") == t and e["kind"] == kind and e["t"] is not None
                   and lo <= e["t"] < hi for e in events)

    victims = names[1:]
    pre, post = args.fault_at, args.duration - args.fault_at
    summary = {
        "mode": mode, "n_tasks": n, "tasks": names, "fault_tenant": FAULT_TASK,
        "victims": victims, "chain": {names[t]: chain[t] for t in range(n)},
        "fault_kind": args.fault_kind, "fault_at": args.fault_at,
        "duration": args.duration, "warmup": args.warmup, "clients": args.clients,
        "think_time": args.think_time, "retry_backoff": args.retry_backoff,
        "backbone": args.backbone, "max_batch": cfg["max_batch"],
        "batch_window_ms": args.batch_window_ms, "device": args.device,
        "leak_mb_per_s": args.leak_mb_per_s if args.fault_kind == "leak" else None,
        "gpu_total_mb": round(mem.total_mb, 1) if mem.ok else None,
        "gpu_peak_used_mb": round(peak_used, 1) if mem.ok else None,
        "restarts": downs,   # every component that went down; nothing restarts
        "ok": {t: count(t, "ok") for t in names},
        "fail": {t: count(t, "fail") for t in names},
        "victims_ok": sum(count(t, "ok") for t in victims),
        "victims_fail": sum(count(t, "fail") for t in victims),
        "victims_rps_before": round(sum(count(t, "ok", 0, pre) for t in victims) / pre, 2),
        "victims_rps_after": round(sum(count(t, "ok", pre, args.duration)
                                       for t in victims) / post, 2),
        "still_in_flight_at_end": len(inflight),
    }
    return events, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--n-tasks", type=int, default=8, help="P + N-1 victims")
    ap.add_argument("--backbone", default="momentlarge", choices=im.BACKBONES)
    ap.add_argument("--clients", type=int, default=1, help="closed-loop clients per task")
    ap.add_argument("--max-batch", type=int, default=None,
                    help="cross-task backbone batch (s_be, fmvisor); default all requests")
    ap.add_argument("--batch-window-ms", type=float, default=1.0)
    ap.add_argument("--think-time", type=float, default=0.0)
    ap.add_argument("--retry-backoff", type=float, default=0.5)
    ap.add_argument("--fault-kind", choices=FAULT_KINDS, default="gpu")
    ap.add_argument("--fault-at", type=float, default=15.0, help="s after t = 0")
    ap.add_argument("--leak-mb-per-s", type=float, default=400.0,
                    help="leak: GPU memory P's decoder keeps per second of wall time")
    ap.add_argument("--mem-period", type=float, default=0.5,
                    help="s between NVML memory samples")
    ap.add_argument("--warmup", type=float, default=10.0, help="load before t = 0")
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--startup-timeout", type=float, default=600.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--exp-dir", type=Path, default=EXP_DIR / "results")
    args = ap.parse_args()

    modes = args.modes.split(",")
    assert all(m in MODES for m in modes), modes
    assert args.n_tasks >= 2, "need the faulty task and at least one victim"
    assert 0 < args.fault_at < args.duration
    out_dir = fault_dir(args.exp_dir, args.fault_kind)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.n_tasks} tasks {task_names(args.n_tasks)} on {args.backbone}, "
          f"{args.clients} closed-loop client(s) each; {args.fault_kind} fault in "
          f"{FAULT_TASK}'s decoder at t={args.fault_at}s of {args.duration}s -> {out_dir}")

    for mode in modes:
        print(f"--- {mode}", flush=True)
        try:
            events, summary = run_mode(mode, args)
        except RuntimeError as e:
            print(f"    FAILED: {e}; skipping", flush=True)
            continue
        with open(out_dir / f"events_{mode}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["kind", "comp", "tenant", "rid", "t",
                                               "n_tok", "lat_ms", "mb"])
            wr.writeheader()
            wr.writerows(events)
        with open(out_dir / "summary.jsonl", "a") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"    victims: ok={summary['victims_ok']} fail={summary['victims_fail']} "
              f"rps {summary['victims_rps_before']} -> {summary['victims_rps_after']}  "
              f"{FAULT_TASK}: ok={summary['ok'][FAULT_TASK]} fail={summary['fail'][FAULT_TASK]}  "
              f"down={[(d['comp'], d['t_down'] and round(d['t_down'], 1)) for d in summary['restarts']]}",
              flush=True)


if __name__ == "__main__":
    main()
