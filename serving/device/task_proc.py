"""task_proc.py — Per-task (per-vFM) process for process-isolation mode (design 3).

This process BORROWS GPU memory from the backbone: its lane's pool (inputs,
masks, features; see backbone_proc.py), shared once through CUDA IPC. It owns
no GPU memory the backbone reads, so its death cannot hurt the backbone.

  1. Intake thread: read TaskInferMsg from req_q. Every request first joins
     this process's CPU overflow queue (bounded by --task-backlog-cap; beyond
     it the request fails with queue_full). The first request also asks the
     backbone for the pool: ("alloc", x_shape, x_dtype, m_shape, m_dtype).
  2. pump(): while a slot j of the pool is free, take the oldest overflow
     request, copy its input host -> pinned buffer -> pool x[j] (and mask[j])
     on a copy stream, synchronize (the backbone must not read before the
     copy ends), and tell the backbone ("slot", j, enqueued_at).
  3. Head thread: read done_q. ("pool", ...) / ("pool_error", why) answer the
     alloc; ("feat", feat) replaces the feature pool after a backbone swap;
     ("done", [j..], ...) means feat[j] holds the features: run the decoder on
     them, reply, free the slots and pump(); ("error", [j..], why) fails them.

So up to N requests of this task wait on the GPU (in its pool), the rest wait
on the CPU in this process. One sample per request (leading dim 1). Requests
whose input shape or dtype differs from the first one fail (the pool has one
shape). An uncaught exception in either thread (an OOM, typically) sets
`dying` and ends this process only.

Queue topology (one lane per task, see isolated_app.py):
    req_q      gRPC process → this proc   (TaskInferMsg / TaskControlMsg)
    sched_q    this proc → backbone_proc  (alloc, filled slots)
    done_q     backbone_proc → this proc  (pool, feat, done, error)
    result_q   all task procs → gRPC process  (result dicts)
"""

import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

@dataclass
class TaskInferMsg:
    """gRPC process → task_proc: raw proto tensor bytes."""
    type: str = "infer"
    req_id: int = 0
    x_bytes: bytes = b""
    x_shape: tuple = ()
    x_dtype: str = "float32"
    mask_bytes: bytes = b""
    mask_shape: tuple = ()
    mask_dtype: str = "float32"
    enqueued_at: float = 0.0   # wall time set by gRPC process on arrival


@dataclass
class TaskControlMsg:
    """gRPC process → task_proc: control commands."""
    type: str = "control"
    command: str = ""          # "load_decoder"; type="stop" ends the process
    payload_json: str = ""
    seq: int = 0               # matched to reply on ctrl_reply_q


def _fatal_hook(task: str, dying):
    """threading.excepthook: set `dying` first, so the backbone stops using
    this process's lane, then end the process."""
    def hook(hook_args) -> None:
        dying.set()
        traceback.print_exception(hook_args.exc_type, hook_args.exc_value,
                                  hook_args.exc_traceback)
        name = getattr(hook_args.thread, "name", "?")
        print(f"[TaskProc:{task}] {hook_args.exc_type.__name__} in {name}, exiting", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    return hook


@dataclass
class _Pending:
    """A request waiting in this process's CPU overflow queue."""
    rid: int
    key: tuple                 # (x_shape, x_dtype, m_shape, m_dtype), per sample
    x: np.ndarray              # [1, *x_shape]
    m: object                  # np.ndarray [1, *m_shape] or None
    enqueued_at: float


# ---------------------------------------------------------------------------
# Task process entry point
# ---------------------------------------------------------------------------

def task_proc_main(
    task: str,
    lane: int,                 # this task's lane id (isolated_app.py)
    device: str,
    backlog_cap: int,          # max requests waiting on the CPU in this process
    req_q,                     # gRPC process → this proc
    ctrl_reply_q,              # this proc → isolated_app (control replies)
    sched_q,                   # this proc → backbone_proc
    done_q,                    # backbone_proc → this proc
    result_q,                  # this proc → gRPC process
    dying,                     # set when this proc fails
    run_stop,                  # set when the server stops
):
    import json

    import torch
    import torch.multiprocessing  # noqa: F401  (CUDA tensors through queues)
    import torch.nn as nn
    os.environ["CUDA_DEVICE"] = device

    dev = torch.device(device)
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.set_device(dev)
    torch.set_num_threads(1)
    threading.excepthook = _fatal_hook(task, dying)

    print(f"[TaskProc:{task}] Started on device={device} backlog_cap={backlog_cap}")

    decoder = None
    lock = threading.Lock()          # pool state, free, overflow, rid_of
    stage_lock = threading.Lock()    # one stager at a time: pinned buffers, copy stream, FIFO
    stop = threading.Event()

    pool = {"key": None, "x": None, "mask": None, "feat": None}
    pool_state = ["none"]            # "none" | "pending" | "ready"
    free: list = []                  # free slot numbers of the pool
    rid_of: dict = {}                # slot j -> req_id of the request in it
    overflow: deque = deque()        # _Pending, oldest first
    pinned = {"x": None, "mask": None}
    copy_stream = torch.cuda.Stream(device=dev) if cuda else None

    def torch_dtype(np_dtype):
        return torch.from_numpy(np.empty(0, dtype=np_dtype)).dtype

    def fail(rid: int, why: str) -> None:
        result_q.put({"req_id": rid, "error": why})

    # -----------------------------------------------------------------------
    # Control
    # -----------------------------------------------------------------------

    def load_decoder(spec: dict) -> float:
        """Load decoder via pipeline.add_decoder using a dummy model_instance.

        The dummy only needs .model_category so Pipeline can resolve the path
        to the decoder weights. No backbone weights are loaded here — the real
        backbone lives in backbone_proc.
        """
        nonlocal decoder
        from fmtk.pipeline import Pipeline
        from device.model_loader import _build_decoder

        bb   = spec["backbone"]
        dtype = spec["type"]
        path  = spec["path"]
        base_task = spec.get("base_task", task)
        model_category = spec.get("model_category", "tsfm")
        print(f"[TaskProc:{task}] Loading decoder backbone={bb} type={dtype} path={path} model_category={model_category}")

        class _DummyModelInstance:
            pass
        dummy = _DummyModelInstance()
        dummy.model_category = model_category
        dummy.device = device   # Pipeline.__init__ reads model_instance.device

        if cuda:
            torch.cuda.reset_peak_memory_stats()

        pipeline = Pipeline(dummy)
        decoder_obj = _build_decoder(bb, task, dtype, device, base_task=base_task)
        pipeline.add_decoder(decoder_obj, load=True, train=False, path=path)
        decoder_obj.model.eval()
        decoder = decoder_obj

        mem_mb = 0.0
        if cuda:
            torch.cuda.synchronize(dev)
            mem_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
        print(f"[TaskProc:{task}] Decoder loaded  gpu_alloc_peak={mem_mb:.1f} MB")
        return mem_mb

    def handle_control(msg: TaskControlMsg) -> None:
        payload = json.loads(msg.payload_json) if msg.payload_json else {}
        status = "ok"
        gpu_decoder_mem_mb = 0.0
        try:
            if msg.command == "load_decoder":
                gpu_decoder_mem_mb = load_decoder(payload)
                status = f"loaded_decoder_{task}"
            else:
                status = f"unknown_command_{msg.command}"
        except Exception as e:
            status = f"error_{e}"
            print(f"[TaskProc:{task}] Control error: {e}")
        ctrl_reply_q.put({"seq": msg.seq, "status": status,
                          "gpu_decoder_mem_mb": gpu_decoder_mem_mb})

    # -----------------------------------------------------------------------
    # Requests: CPU overflow -> GPU slots
    # -----------------------------------------------------------------------

    def submit(msg: TaskInferMsg) -> None:
        """Intake: validate, queue on the CPU, ask for the pool once, pump."""
        x = np.frombuffer(msg.x_bytes, dtype=msg.x_dtype).reshape(msg.x_shape)
        m = (np.frombuffer(msg.mask_bytes, dtype=msg.mask_dtype).reshape(msg.mask_shape)
             if msg.mask_bytes else None)
        if x.shape[0] != 1 or (m is not None and m.shape[0] != 1):
            fail(msg.req_id, "isolation_mode=process serves one sample per request")
            return
        key = (tuple(x.shape[1:]), torch_dtype(x.dtype),
               tuple(m.shape[1:]) if m is not None else None,
               torch_dtype(m.dtype) if m is not None else None)
        with lock:
            if pool_state[0] == "ready" and key != pool["key"]:
                why = f"input {key} does not match this vFM's pool {pool['key']}"
            elif len(overflow) >= backlog_cap:
                why = "queue_full"
            else:
                why = None
                overflow.append(_Pending(msg.req_id, key, x, m, msg.enqueued_at))
                if pool_state[0] == "none":
                    pool_state[0] = "pending"
                    sched_q.put(("alloc", *key))
        if why is not None:
            fail(msg.req_id, why)
            return
        pump()

    def pump() -> None:
        """Move the oldest overflow requests into free GPU slots, in order."""
        with stage_lock:
            while True:
                with lock:
                    if pool_state[0] != "ready" or not overflow or not free:
                        return
                    p = overflow.popleft()
                    if p.key != pool["key"]:
                        bad, j = p, None
                    else:
                        bad, j = None, free.pop()
                        rid_of[j] = p.rid
                if bad is not None:
                    fail(bad.rid, f"input {bad.key} does not match this vFM's pool {pool['key']}")
                    continue
                stage(j, p)

    def stage(j: int, p: _Pending) -> None:
        """Called with stage_lock held: copy p into slot j, then hand it over."""
        pinned["x"].numpy()[...] = p.x[0]
        if p.m is not None:
            pinned["mask"].numpy()[...] = p.m[0]
        with (torch.cuda.stream(copy_stream) if cuda else nullcontext()):
            pool["x"][j].copy_(pinned["x"], non_blocking=cuda)
            if p.m is not None:
                pool["mask"][j].copy_(pinned["mask"], non_blocking=cuda)
        if cuda:
            # Host-side queues do not order kernels across CUDA contexts: the
            # input must be in the pool before the backbone reads it.
            copy_stream.synchronize()
        sched_q.put(("slot", j, p.enqueued_at))

    def on_pool(x, mask, feat) -> None:
        with stage_lock:
            pinned["x"] = torch.empty(tuple(x.shape[1:]), dtype=x.dtype, pin_memory=cuda)
            pinned["mask"] = (torch.empty(tuple(mask.shape[1:]), dtype=mask.dtype, pin_memory=cuda)
                              if mask is not None else None)
            with lock:
                pool["x"], pool["mask"], pool["feat"] = x, mask, feat
                pool["key"] = (tuple(x.shape[1:]), x.dtype,
                               tuple(mask.shape[1:]) if mask is not None else None,
                               mask.dtype if mask is not None else None)
                free[:] = list(range(x.shape[0]))
                pool_state[0] = "ready"
        print(f"[TaskProc:{task}] Pool ready: {x.shape[0]} slots, x{tuple(x.shape[1:])} "
              f"feat{tuple(feat.shape[1:])}")
        pump()

    def on_pool_error(why: str) -> None:
        with lock:
            waiting = list(overflow)
            overflow.clear()
            pool_state[0] = "none"         # the next request asks again
        print(f"[TaskProc:{task}] Pool allocation failed: {why}")
        for p in waiting:
            fail(p.rid, f"pool_error: {why}")

    def release(js: list) -> list:
        """Free slots js; return the req_ids that were in them."""
        with lock:
            rids = [rid_of.pop(j) for j in js]
            free.extend(js)
        return rids

    # -----------------------------------------------------------------------
    # Threads
    # -----------------------------------------------------------------------

    def intake_loop() -> None:
        while not run_stop.is_set():
            try:
                msg = req_q.get(timeout=0.5)
            except queue.Empty:
                continue
            t = getattr(msg, "type", None)
            if t == "stop":
                print(f"[TaskProc:{task}] Received stop, exiting")
                return
            if t == "control":
                handle_control(msg)
            elif t == "infer":
                submit(msg)
            else:
                print(f"[TaskProc:{task}] Unknown message type: {t}")

    def head_loop() -> None:
        stream = torch.cuda.Stream(device=dev) if cuda else None
        while not stop.is_set():
            try:
                msg = done_q.get(timeout=0.5)
            except queue.Empty:
                continue
            kind = msg[0]
            if kind == "pool":
                on_pool(*msg[1:])
                continue
            if kind == "pool_error":
                on_pool_error(msg[1])
                continue
            if kind == "feat":
                with lock:
                    pool["feat"] = msg[1]
                continue
            if kind == "error":
                _, js, why = msg
                for rid in release(js):
                    fail(rid, why)
                pump()
                continue

            _, js, start_ns, proc_ns = msg          # "done"
            dec_start = time.time_ns()
            with (torch.cuda.stream(stream) if cuda else nullcontext()), torch.no_grad():
                idx = torch.tensor(js, dtype=torch.long, device=dev)
                feats = pool["feat"].index_select(0, idx)
                if decoder is not None:
                    logit = decoder.forward(feats)
                    if isinstance(decoder.criterion, nn.CrossEntropyLoss):
                        logit = torch.argmax(logit, dim=1)
                    out = logit.detach().cpu().numpy()         # synchronizes this stream
                else:
                    out = feats.detach().cpu().float().numpy()
            end_ns = time.time_ns()
            dec_ns = (end_ns - dec_start) // max(len(js), 1)
            for k, rid in enumerate(release(js)):
                result_q.put({
                    "req_id": rid,
                    "output": np.asarray(out[k:k + 1]).reshape(-1),
                    "start_time_ns": start_ns,
                    "end_time_ns": end_ns,
                    "proc_time_ns": proc_ns,
                    "swap_time_ns": 0,
                    "decoder_time_ns": dec_ns,
                })
            pump()

    intake = threading.Thread(target=intake_loop, name="intake", daemon=True)
    head = threading.Thread(target=head_loop, name="head", daemon=True)
    intake.start()
    head.start()
    intake.join()
    stop.set()
    head.join(timeout=5)
    print(f"[TaskProc:{task}] Exited")


# ---------------------------------------------------------------------------
# Helper to spawn
# ---------------------------------------------------------------------------

def start_task_process(ctx, task: str, lane: int, device: str, backlog_cap: int, req_q,
                       ctrl_reply_q, sched_q, done_q, result_q, dying, run_stop):
    p = ctx.Process(
        target=task_proc_main,
        args=(task, lane, device, backlog_cap, req_q, ctrl_reply_q, sched_q, done_q,
              result_q, dying, run_stop),
        daemon=True,
        name=f"task_proc_{task}",
    )
    p.start()
    return p
