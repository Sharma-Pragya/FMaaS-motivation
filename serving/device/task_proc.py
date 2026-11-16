"""task_proc.py — Per-task process for process-isolation mode.

As fmvisor's task process (experiments/memory_isolation_vision/run.py:
fm_task_main), with the input slots in host memory:

  1. Intake thread: read TaskInferMsg from req_q, copy the input (and its
     mask, if any) into a slot in this process's own host memory, and tell
     the backbone which slot: ("slot", cid, j, enqueued_at) on sched_q. A new
     chunk of --slot-chunk slots is allocated only when every slot of that
     shape is in use (one per request as it arrives, by default), and shared
     once with the backbone: ("chunk", cid, x_chunk, mask_chunk) on sched_q.
  2. Head thread: read done_q. ("ring", ring) is the backbone's feature ring
     (GPU, CUDA IPC); ("batch", r, items, ...) says which rows of ring[r] are
     this task's. Run the decoder once on those rows, ack ring[r] on ack_q,
     free the slots, and put one result per request on result_q.

One sample per request (leading dim 1). An uncaught exception in either
thread (an OOM, typically) sets `dying` and ends this process only.

Queue topology (one lane per task, see isolated_app.py):
    req_q      gRPC process → this proc   (TaskInferMsg / TaskControlMsg)
    sched_q    this proc → backbone_proc  (chunks, filled slots)
    done_q     backbone_proc → this proc  (ring, batches)
    ack_q      all task procs → backbone_proc (ring entry read)
    result_q   all task procs → gRPC process  (result dicts)
"""

import os
import queue
import sys
import threading
import time
import traceback
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
    this process's memory, then end the process (as fmvisor's task process)."""
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


# ---------------------------------------------------------------------------
# Task process entry point
# ---------------------------------------------------------------------------

def task_proc_main(
    task: str,
    lane: int,                 # this task's lane id (isolated_app.py)
    device: str,
    slot_chunk: int,
    req_q,                     # gRPC process → this proc
    ctrl_reply_q,              # this proc → isolated_app (control replies)
    sched_q,                   # this proc → backbone_proc
    done_q,                    # backbone_proc → this proc
    ack_q,                     # this proc → backbone_proc
    result_q,                  # this proc → gRPC process
    dying,                     # set when this proc fails
    run_stop,                  # set when the server stops
):
    import json

    import torch
    import torch.multiprocessing as tmp
    import torch.nn as nn
    # One fd per shared chunk otherwise; a large backlog would exhaust them.
    tmp.set_sharing_strategy("file_system")
    os.environ["CUDA_DEVICE"] = device

    dev = torch.device(device)
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.set_device(dev)
    torch.set_num_threads(1)
    threading.excepthook = _fatal_hook(task, dying)

    print(f"[TaskProc:{task}] Started on device={device} slot_chunk={slot_chunk}")

    decoder = None
    lock = threading.Lock()
    chunks: list = []          # cid -> (x chunk, mask chunk or None), kept alive
    chunk_key: list = []       # cid -> slot shape key
    free: dict = {}            # slot shape key -> [(cid, j)]
    rid_of: dict = {}          # (cid, j) -> req_id of the request in it
    stop = threading.Event()

    # -----------------------------------------------------------------------
    # Helpers
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

    def new_chunk(key) -> None:
        """Called with lock held: slot_chunk slots of this shape key, in
        shared host memory, handed to the backbone before any slot of them."""
        x_shape, x_dtype, m_shape, m_dtype = key
        xc = torch.empty((slot_chunk, *x_shape), dtype=x_dtype).share_memory_()
        mc = (torch.empty((slot_chunk, *m_shape), dtype=m_dtype).share_memory_()
              if m_shape is not None else None)
        cid = len(chunks)
        chunks.append((xc, mc))
        chunk_key.append(key)
        sched_q.put(("chunk", cid, xc, mc))
        free.setdefault(key, []).extend((cid, j) for j in range(slot_chunk))

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

    def stage(msg: TaskInferMsg) -> None:
        """Copy one request into a free slot and tell the backbone."""
        x = np.frombuffer(msg.x_bytes, dtype=msg.x_dtype).reshape(msg.x_shape)
        m = (np.frombuffer(msg.mask_bytes, dtype=msg.mask_dtype).reshape(msg.mask_shape)
             if msg.mask_bytes else None)
        if x.shape[0] != 1 or (m is not None and m.shape[0] != 1):
            result_q.put({"req_id": msg.req_id,
                          "error": "isolation_mode=process serves one sample per request"})
            return
        key = (tuple(x.shape[1:]), torch.from_numpy(x[:0]).dtype,
               tuple(m.shape[1:]) if m is not None else None,
               torch.from_numpy(m[:0]).dtype if m is not None else None)
        with lock:
            if not free.get(key):
                new_chunk(key)
            slot = free[key].pop()
            rid_of[slot] = msg.req_id
        cid, j = slot
        xc, mc = chunks[cid]
        xc.numpy()[j] = x[0]
        if m is not None:
            mc.numpy()[j] = m[0]
        sched_q.put(("slot", cid, j, msg.enqueued_at))

    def release(slots: list) -> list:
        """Free slots back to their shape's free list; return their req_ids."""
        with lock:
            rids = [rid_of.pop(s) for s in slots]
            for s in slots:
                free[chunk_key[s[0]]].append(s)
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
                stage(msg)
            else:
                print(f"[TaskProc:{task}] Unknown message type: {t}")

    def head_loop() -> None:
        stream = torch.cuda.Stream(device=dev) if cuda else None
        ring = None
        while not stop.is_set():
            try:
                msg = done_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if msg[0] == "ring":
                ring = msg[1]
                continue
            if msg[0] == "error":
                _, slots, why = msg
                for rid in release(slots):
                    result_q.put({"req_id": rid, "error": why})
                continue

            _, r, items, start_ns, proc_ns = msg
            slots = [s for s, _ in items]
            rows = [k for _, k in items]
            dec_start = time.time_ns()
            with (torch.cuda.stream(stream) if cuda else nullcontext()), torch.no_grad():
                idx = torch.tensor(rows, dtype=torch.long, device=dev)
                feats = ring[r].index_select(0, idx)
                if decoder is not None:
                    logit = decoder.forward(feats)
                    if isinstance(decoder.criterion, nn.CrossEntropyLoss):
                        logit = torch.argmax(logit, dim=1)
                    out = logit.detach().cpu().numpy()         # synchronizes this stream
                else:
                    out = feats.detach().cpu().float().numpy()
            ack_q.put((r, lane))
            end_ns = time.time_ns()
            dec_ns = (end_ns - dec_start) // max(len(rows), 1)
            for k, rid in enumerate(release(slots)):
                result_q.put({
                    "req_id": rid,
                    "output": np.asarray(out[k:k + 1]).reshape(-1),
                    "start_time_ns": start_ns,
                    "end_time_ns": end_ns,
                    "proc_time_ns": proc_ns,
                    "swap_time_ns": 0,
                    "decoder_time_ns": dec_ns,
                })

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

def start_task_process(ctx, task: str, lane: int, device: str, slot_chunk: int, req_q,
                       ctrl_reply_q, sched_q, done_q, ack_q, result_q, dying, run_stop):
    p = ctx.Process(
        target=task_proc_main,
        args=(task, lane, device, slot_chunk, req_q, ctrl_reply_q, sched_q, done_q,
              ack_q, result_q, dying, run_stop),
        daemon=True,
        name=f"task_proc_{task}",
    )
    p.start()
    return p
