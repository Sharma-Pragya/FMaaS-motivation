"""backbone_proc.py — Backbone process for process-isolation mode.

Owns:
  - FM backbone weights (GPU)
  - TenantQueues: one deque per task, populated from sched_q
  - Scheduling policy (FifoPolicy / RoundRobinPolicy) across all task queues
    — same logic as DeviceBatcher in shared mode
  - Backbone forward pass

Flow:
  1. Continuously drain sched_q → push PreprocessedMsg into TenantQueues
  2. Scheduler loop: wait for requests, open batch window, apply policy to
     select a cross-task batch (shape-homogeneous, up to max_batch_size)
  3. Read tensors from SharedMemory, run backbone forward
  4. Write feature slices to SharedMemory, put FeatMsg on feat_queues[task]

Control messages (load / swap_backbone) arrive on ctrl_q (separate from sched_q
so they are never mixed with infer messages and are handled immediately).

Queue topology (matches isolated_app.py):
    task_proc[task] → task_sched_queues[task]  (PreprocessedMsg — one queue per task)
    isolated_app    → ctrl_q                   (ControlMsg / SentinelMsg)
    backbone_proc   → feat_q[task]             (feat result dicts per task)
"""

import time
from collections import deque
from dataclasses import dataclass
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

@dataclass
class ControlMsg:
    type: str = "control"
    command: str = ""          # "load" | "swap_backbone"
    payload_json: str = ""
    seq: int = 0               # matched to reply on ctrl_reply_q


@dataclass
class SentinelMsg:
    type: str = "stop"


# ---------------------------------------------------------------------------
# Backbone process entry point
# ---------------------------------------------------------------------------

def backbone_proc_main(
    task_sched_queues: dict,   # {task: Queue} — one dedicated queue per task_proc
    ctrl_q: Queue,             # ControlMsg / SentinelMsg from isolated_app
    ctrl_reply_q: Queue,       # backbone_proc → isolated_app (control replies)
    feat_queues: dict,         # {task: Queue} — pre-created, passed in
    max_batch_size: int,
    max_batch_wait_ms: float,
    device: str,
    scheduler_policy: str,     # "fifo" | "round_robin"
):
    import gc
    import json
    import os

    import torch
    os.environ["CUDA_DEVICE"] = device

    # Import scheduler policies — same ones used in shared mode
    from device.model_loader import _build_pipeline
    from device.scheduler import FifoPolicy, RoundRobinPolicy, TenantQueues
    from device.task_proc import PreprocessedMsg
    from fmtk.logger import Logger

    print(f"[BackboneProc] Started device={device} policy={scheduler_policy} "
          f"max_batch_size={max_batch_size} max_batch_wait_ms={max_batch_wait_ms}")

    pipeline = None
    backbone_name = None
    logger = Logger(device, "backbone_proc")
    max_wait_s = max_batch_wait_ms / 1000.0

    # TenantQueues: same class as shared mode — one deque per task
    tenant_queues = TenantQueues()
    policy = RoundRobinPolicy() if scheduler_policy == "round_robin" else FifoPolicy()

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def drain_sched_q():
        """Non-blocking: poll each task's dedicated queue, push into TenantQueues."""
        for task_q in task_sched_queues.values():
            while True:
                try:
                    msg = task_q.get_nowait()
                except Exception:
                    break
                if isinstance(msg, PreprocessedMsg) or getattr(msg, "type", None) == "preprocessed":
                    tenant_queues.push(_make_envelope(msg))

    def _make_envelope(msg: PreprocessedMsg):
        """Wrap PreprocessedMsg into a thin object TenantQueues/policies can use.
        Policies only need: .task, .enqueued_at, .x (for shape check as x_shape)."""
        class _Env:
            __slots__ = ("req_id", "task", "enqueued_at", "x",
                         "shm_name", "shape", "dtype",
                         "mask_shm_name", "mask_shape", "mask_dtype")
        e = _Env()
        e.req_id        = msg.req_id
        e.task          = msg.task
        e.enqueued_at   = msg.enqueued_at
        e.shm_name      = msg.shm_name
        e.shape         = msg.shape
        e.dtype         = msg.dtype
        e.mask_shm_name = msg.mask_shm_name
        e.mask_shape    = msg.mask_shape
        e.mask_dtype    = msg.mask_dtype
        # Policies check .x.shape for homogeneity — use a dummy array with right shape
        e.x = np.empty(msg.shape, dtype=msg.dtype)
        return e

    def poll_ctrl_q():
        """Non-blocking check for control/stop messages. Returns True if stop."""
        while True:
            try:
                msg = ctrl_q.get_nowait()
            except Exception:
                return False
            t = getattr(msg, "type", None)
            if t == "stop":
                return True
            if t == "control":
                handle_control(msg)
        return False

    def handle_control(msg: ControlMsg):
        nonlocal pipeline, backbone_name
        payload = json.loads(msg.payload_json) if msg.payload_json else {}
        status = "ok"
        gpu_backbone_mem_mb = 0.0
        try:
            if msg.command == "load":
                backbone = payload["backbone"]
                print(f"[BackboneProc] Loading backbone={backbone}")
                torch.cuda.reset_peak_memory_stats()
                pipeline = _build_pipeline(backbone, device, logger,
                                           model_config=payload.get("model_config"))
                backbone_name = backbone
                torch.cuda.synchronize(torch.device(device))
                gpu_backbone_mem_mb = torch.cuda.max_memory_allocated(torch.device(device)) / (1024 ** 2)
                print(f"[BackboneProc] Backbone loaded  gpu_alloc_peak={gpu_backbone_mem_mb:.1f} MB")
                status = f"loaded_{backbone}"
            elif msg.command == "swap_backbone":
                backbone = payload["backbone"]
                print(f"[BackboneProc] Swapping backbone → {backbone}")
                del pipeline
                pipeline = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                pipeline = _build_pipeline(backbone, device, logger,
                                           model_config=payload.get("model_config"))
                backbone_name = backbone
                torch.cuda.synchronize(torch.device(device))
                gpu_backbone_mem_mb = torch.cuda.max_memory_allocated(torch.device(device)) / (1024 ** 2)
                status = f"swapped_{backbone}"
            else:
                status = f"unknown_command_{msg.command}"
            model_category = pipeline.model_instance.model_category if pipeline is not None else "tsfm"
        except Exception as e:
            status = f"error_{e}"
            print(f"[BackboneProc] Control error: {e}")
        gpu_backbone_reserved_mb = torch.cuda.memory_reserved(torch.device(device)) / (1024 ** 2)
        summary = {"backbone_process": {"gpu peak": gpu_backbone_mem_mb,
                                        "gpu reserved": gpu_backbone_reserved_mb}}
        ctrl_reply_q.put({"seq": msg.seq, "status": status,
                          "logger_summary": str(summary),
                          "model_category": model_category})

    def run_batch(batch: list):
        """Read tensors from SharedMemory, backbone forward, write feats back."""
        arrays, masks = [], []
        for env in batch:
            shm = SharedMemory(name=env.shm_name)
            x = np.ndarray(env.shape, dtype=env.dtype, buffer=shm.buf).copy()
            shm.close()
            shm.unlink()
            arrays.append(x)

            if env.mask_shm_name:
                mshm = SharedMemory(name=env.mask_shm_name)
                m = np.ndarray(env.mask_shape, dtype=env.mask_dtype, buffer=mshm.buf).copy()
                mshm.close()
                mshm.unlink()
                masks.append(m)
            else:
                masks.append(None)

        x_batch = np.concatenate(arrays, axis=0)
        mask_batch = (
            np.concatenate([m for m in masks if m is not None], axis=0)
            if all(m is not None for m in masks)
            else None
        )

        bx = torch.from_numpy(x_batch).to(device)
        bmask = torch.from_numpy(mask_batch).to(device) if mask_batch is not None else None

        start_ns = time.time_ns()
        with torch.no_grad():
            feats = pipeline.model_instance.forward(bx, bmask)
        proc_time_ns = time.time_ns() - start_ns

        # Fan out per-sample feature slices to each task's feat_q
        offset = 0
        for i, env in enumerate(batch):
            n = arrays[i].shape[0]
            feat_slice = feats[offset: offset + n].detach().cpu().numpy()
            offset += n

            shm_out = SharedMemory(create=True, size=max(feat_slice.nbytes, 1))
            np.copyto(
                np.ndarray(feat_slice.shape, dtype=feat_slice.dtype, buffer=shm_out.buf),
                feat_slice,
            )
            shm_out.close()  # task_proc will unlink

            feat_queues[env.task].put({
                "req_id": env.req_id,
                "shm_name": shm_out.name,
                "shape": feat_slice.shape,
                "dtype": str(feat_slice.dtype),
                "start_time_ns": start_ns,
                "proc_time_ns": proc_time_ns,
            })

        print(f"[BackboneProc] Batch done size={len(batch)} "
              f"tasks={[e.task for e in batch]} proc_time_ns={proc_time_ns}")

    # -----------------------------------------------------------------------
    # Scheduler loop — mirrors DeviceBatcher.run_forever() / _next_batch()
    # but reads from TenantQueues populated from sched_q
    # -----------------------------------------------------------------------
    while True:
        # Check for stop/control before doing anything else
        if poll_ctrl_q():
            print("[BackboneProc] Received stop, exiting")
            break

        drain_sched_q()

        # --- Phase 1: wait until at least one request is queued ---
        while tenant_queues.pending_count() == 0:
            if poll_ctrl_q():
                print("[BackboneProc] Received stop while idle, exiting")
                return
            drain_sched_q()
            time.sleep(0.001)

        # --- Phase 2: batch collection window ---
        # Same as DeviceBatcher._next_batch(): wait up to max_batch_wait_ms
        # or until max_batch_size requests are available across all task queues.
        deadline = time.time() + max_wait_s
        while tenant_queues.pending_count() < max_batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            if poll_ctrl_q():
                # Handle control but keep collecting — don't discard queued requests
                pass
            drain_sched_q()
            time.sleep(min(0.001, max(0, remaining)))

        # --- Phase 3: apply scheduling policy, select cross-task batch ---
        batch = tenant_queues.select_batch(policy, max_batch_size)
        if not batch:
            continue

        print(f"[BackboneProc] Scheduled batch_size={len(batch)} "
              f"req_ids={[e.req_id for e in batch]} tasks={[e.task for e in batch]}")

        if pipeline is None:
            for env in batch:
                feat_queues[env.task].put({"req_id": env.req_id, "error": "backbone_not_loaded"})
            continue

        try:
            run_batch(batch)
        except Exception as e:
            print(f"[BackboneProc] Forward error: {e}")
            for env in batch:
                feat_queues[env.task].put({"req_id": env.req_id, "error": str(e)})

    print("[BackboneProc] Exited")


# ---------------------------------------------------------------------------
# Helper to spawn
# ---------------------------------------------------------------------------

def start_backbone_process(
    task_sched_queues: dict,
    ctrl_q: Queue,
    ctrl_reply_q: Queue,
    feat_queues: dict,
    max_batch_size: int,
    max_batch_wait_ms: float,
    device: str,
    scheduler_policy: str = "fifo",
) -> Process:
    p = Process(
        target=backbone_proc_main,
        args=(task_sched_queues, ctrl_q, ctrl_reply_q, feat_queues,
              max_batch_size, max_batch_wait_ms, device, scheduler_policy),
        daemon=True,
        name="backbone_proc",
    )
    p.start()
    return p
