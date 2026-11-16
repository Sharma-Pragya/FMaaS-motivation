"""backbone_proc.py — Backbone process for process-isolation mode.

As fmvisor's backbone process (experiments/memory_isolation_vision/run.py:
fm_backbone_main), with the input slots in the task processes' host memory.

Owns:
  - FM backbone weights (GPU), run on bb_stream: the TPC-masked stream from
    EdgeRuntimeApplication._setup_tpc when --tpc-mode is set, as shared mode
  - TenantQueues + the scheduling policy from EdgeRuntimeApplication._make_policy,
    driven as batch_former.py / batcher.py drive it
  - The feature ring: ring_size batches of features in VRAM, shared with the
    task processes through CUDA IPC

Loop:
  1. Poll each attached lane's sched_q: ("chunk", cid, x, mask) maps a task's
     slots; ("slot", cid, j, enqueued_at) becomes a RequestEnvelope whose x /
     mask are that slot's tensors.
  2. Wait until ring[next_r] has been read by every task it was sent to and
     the batch window has passed; policy.select picks a cross-task batch.
  3. Stack the batch's slots, one copy to the GPU, backbone forward, features
     into ring[r], bb_stream.synchronize().
  4. Tell each task in the batch which rows of ring[r] are its:
     ("batch", r, [((cid, j), row), ...], start_ns, proc_ns) on its done_q.

The ring is allocated on the first batch (the feature shape is known then),
and again, once every task has read it, when a batch's feature shape differs
(after swap_backbone); each attached lane gets ("ring", ring) first.

Control messages arrive on ctrl_q: load / swap_backbone (replied on
ctrl_reply_q), attach (a lane now carries a task; replied) and detach (the
lane's task process is gone; not replied).
"""

import json
import queue
import time
from contextlib import nullcontext
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

@dataclass
class ControlMsg:
    type: str = "control"
    command: str = ""          # "load" | "swap_backbone" | "attach" | "detach"
    payload_json: str = ""
    seq: int = 0               # matched to reply on ctrl_reply_q


@dataclass
class SentinelMsg:
    type: str = "stop"


# ---------------------------------------------------------------------------
# Backbone process entry point
# ---------------------------------------------------------------------------

def backbone_proc_main(
    config,                    # RuntimeServerConfig
    device: str,
    sched_qs: list,            # lane i: task proc → backbone
    done_qs: list,             # lane i: backbone → task proc
    dyings: list,              # lane i: set when its task proc fails
    ctrl_q,                    # ControlMsg / SentinelMsg from isolated_app
    ctrl_reply_q,              # backbone_proc → isolated_app (control replies)
    ack_q,                     # task procs → backbone: (ring entry, lane) read
    run_stop,                  # set when the server stops
):
    import gc
    import os

    import torch
    import torch.multiprocessing  # noqa: F401  (CUDA tensors through queues)
    os.environ["CUDA_DEVICE"] = device

    from device.model_loader import _build_pipeline
    from device.scheduler import RequestEnvelope, SABAPolicy, STFQPolicy, TenantQueues
    from device.server import EdgeRuntimeApplication
    from fmtk.logger import Logger

    dev = torch.device(device)
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.set_device(dev)
    torch.set_num_threads(1)

    max_b = config.max_batch_size
    window_s = config.max_batch_wait_ms / 1000.0
    ring_size = config.ring_size
    n = len(sched_qs)

    # Same stream as shared mode: TPC-masked if configured, else a dedicated one.
    bb_stream = EdgeRuntimeApplication._setup_tpc(config) if cuda else None
    if cuda and bb_stream is None:
        bb_stream = torch.cuda.Stream(device=dev)
    policy = EdgeRuntimeApplication._make_policy(config)
    tenant_queues = TenantQueues()

    print(f"[BackboneProc] Started device={device} policy={type(policy).__name__} "
          f"max_batch_size={max_b} max_batch_wait_ms={config.max_batch_wait_ms} "
          f"lanes={n} ring_size={ring_size}")

    pipeline = None
    logger = Logger(device, "backbone_proc")

    task_of = [None] * n                    # lane -> task name (None: not attached)
    chunks = [dict() for _ in range(n)]     # lane -> cid -> (x chunk, mask chunk)
    ring = None
    next_r = 0
    unread = [set() for _ in range(ring_size)]  # ring entry -> lanes yet to read it

    # -----------------------------------------------------------------------
    # Lanes
    # -----------------------------------------------------------------------

    def drop_lane(i: int, why: str) -> None:
        """Lane i's task is gone: never batch it again, release its memory."""
        task = task_of[i]
        if task is None:
            return
        task_of[i] = None
        if task in tenant_queues._queues:
            q = tenant_queues._queues[task]
            # Only this lane's requests: a re-deployed task may be on another lane.
            for env in [e for e in q if e.req_id[0] == i]:
                q.remove(env)
        chunks[i].clear()
        for u in unread:
            u.discard(i)
        print(f"[BackboneProc] Dropped lane {i} (task={task}): {why}")

    def poll() -> None:
        for i in range(n):
            if task_of[i] is None:
                continue
            if dyings[i].is_set():
                drop_lane(i, "task process failed")
                continue
            while True:
                try:
                    msg = sched_qs[i].get_nowait()
                except queue.Empty:
                    break
                except RuntimeError as exc:
                    # Opening a chunk of a task that has just died: drop the
                    # task, not the backbone.
                    drop_lane(i, f"task's memory unavailable: {str(exc)[:80]}")
                    break
                if msg[0] == "chunk":
                    chunks[i][msg[1]] = (msg[2], msg[3])
                else:
                    _, cid, j, enqueued_at = msg
                    xc, mc = chunks[i][cid]
                    env = RequestEnvelope(req_id=(i, cid, j), task=task_of[i], x=xc[j],
                                          mask=mc[j] if mc is not None else None,
                                          question=None, enqueued_at=enqueued_at,
                                          future=None)
                    tenant_queues.push(env)
                    if isinstance(policy, STFQPolicy):
                        policy.assign_start_time(env)

    def drain_acks() -> None:
        while True:
            try:
                r, i = ack_q.get_nowait()
            except queue.Empty:
                return
            unread[r].discard(i)

    def new_ring(feats) -> None:
        """Wait until every task has read the current ring, then allocate one
        for feats' shape and hand it to every attached lane."""
        nonlocal ring, next_r
        while any(unread):
            drain_acks()
            for i in range(n):
                if task_of[i] is not None and dyings[i].is_set():
                    drop_lane(i, "task process failed")
            time.sleep(0.0002)
        ring = torch.zeros((ring_size, max_b, *feats.shape[1:]), dtype=feats.dtype, device=dev)
        next_r = 0
        for i in range(n):
            if task_of[i] is not None:
                done_qs[i].put(("ring", ring))
        print(f"[BackboneProc] Feature ring {tuple(ring.shape)} "
              f"{ring.numel() * ring.element_size() / 2**20:.1f} MB")

    # -----------------------------------------------------------------------
    # Control
    # -----------------------------------------------------------------------

    def poll_ctrl_q() -> bool:
        """Non-blocking check for control/stop messages. Returns True if stop."""
        while True:
            try:
                msg = ctrl_q.get_nowait()
            except queue.Empty:
                return False
            t = getattr(msg, "type", None)
            if t == "stop":
                return True
            if t == "control":
                handle_control(msg)

    def handle_control(msg: ControlMsg) -> None:
        nonlocal pipeline
        payload = json.loads(msg.payload_json) if msg.payload_json else {}

        if msg.command == "attach":
            i, task = payload["lane"], payload["task"]
            task_of[i] = task
            chunks[i].clear()
            if ring is not None:
                done_qs[i].put(("ring", ring))
            print(f"[BackboneProc] Attached lane {i} to task={task}")
            ctrl_reply_q.put({"seq": msg.seq, "status": f"attached_{task}"})
            return
        if msg.command == "detach":
            drop_lane(payload["lane"], "task process exited")
            return

        status = "ok"
        gpu_backbone_mem_mb = 0.0
        model_category = "tsfm"
        try:
            if msg.command == "load":
                backbone = payload["backbone"]
                print(f"[BackboneProc] Loading backbone={backbone}")
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                pipeline = _build_pipeline(backbone, device, logger,
                                           model_config=payload.get("model_config"))
                if cuda:
                    torch.cuda.synchronize(dev)
                    gpu_backbone_mem_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
                print(f"[BackboneProc] Backbone loaded  gpu_alloc_peak={gpu_backbone_mem_mb:.1f} MB")
                status = f"loaded_{backbone}"
            elif msg.command == "swap_backbone":
                backbone = payload["backbone"]
                print(f"[BackboneProc] Swapping backbone → {backbone}")
                pipeline = None
                gc.collect()
                if cuda:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                pipeline = _build_pipeline(backbone, device, logger,
                                           model_config=payload.get("model_config"))
                if cuda:
                    torch.cuda.synchronize(dev)
                    gpu_backbone_mem_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
                status = f"swapped_{backbone}"
            else:
                status = f"unknown_command_{msg.command}"
            if pipeline is not None:
                model_category = pipeline.model_instance.model_category
        except Exception as e:
            status = f"error_{e}"
            print(f"[BackboneProc] Control error: {e}")
        gpu_backbone_reserved_mb = torch.cuda.memory_reserved(dev) / (1024 ** 2) if cuda else 0.0
        summary = {"backbone_process": {"gpu peak": gpu_backbone_mem_mb,
                                        "gpu reserved": gpu_backbone_reserved_mb}}
        ctrl_reply_q.put({"seq": msg.seq, "status": status,
                          "logger_summary": str(summary),
                          "model_category": model_category})

    # -----------------------------------------------------------------------
    # Batches
    # -----------------------------------------------------------------------

    def by_lane(batch: list) -> dict:
        lanes: dict = {}
        for k, env in enumerate(batch):
            i, cid, j = env.req_id
            lanes.setdefault(i, []).append(((cid, j), k))
        return lanes

    def fail_batch(batch: list, why: str) -> None:
        for i, items in by_lane(batch).items():
            done_qs[i].put(("error", [s for s, _ in items], why))

    def run_batch(batch: list) -> None:
        nonlocal next_r
        start_ns = time.time_ns()
        t0 = time.monotonic()
        stream_ctx = torch.cuda.stream(bb_stream) if cuda else nullcontext()
        with stream_ctx, torch.no_grad():
            bx = torch.stack([env.x for env in batch]).to(dev, non_blocking=True)
            masks = [env.mask for env in batch]
            bmask = (torch.stack(masks).to(dev, non_blocking=True)
                     if all(m is not None for m in masks) else None)
            feats = pipeline.model_instance.forward(bx, bmask)
            if (ring is None or tuple(ring.shape[2:]) != tuple(feats.shape[1:])
                    or ring.dtype != feats.dtype):
                new_ring(feats)
            r = next_r
            next_r = (next_r + 1) % ring_size
            ring[r][:len(batch)].copy_(feats)
            # Features must be written before any task reads them (host-side
            # queues do not order kernels across CUDA contexts).
            if cuda:
                bb_stream.synchronize()
            del bx, bmask, feats
        proc_ns = time.time_ns() - start_ns

        # As device/batcher.py: the batch's duration corrects the policy.
        duration_s = time.monotonic() - t0
        if isinstance(policy, SABAPolicy):
            policy.update_batch_duration(duration_s)
        if isinstance(policy, STFQPolicy):
            policy.update_after_execution([env.task for env in batch], duration_s,
                                          tenant_queues._queues)

        lanes = by_lane(batch)
        unread[r] = {i for i in lanes if task_of[i] is not None}
        for i, items in lanes.items():
            if task_of[i] is not None:
                done_qs[i].put(("batch", r, items, start_ns, proc_ns))

        if config.verbose_batch_logs:
            print(f"[BackboneProc] Batch done size={len(batch)} "
                  f"tasks={[env.task for env in batch]} proc_time_ns={proc_ns}")

    # -----------------------------------------------------------------------
    # Scheduler loop
    # -----------------------------------------------------------------------
    while not run_stop.is_set():
        if poll_ctrl_q():
            print("[BackboneProc] Received stop, exiting")
            break
        drain_acks()
        poll()
        if tenant_queues.pending_count() == 0 or unread[next_r]:
            time.sleep(0.0002)
            continue

        # Batch collection window, as BatchFormer.next_batch().
        deadline = time.monotonic() + window_s
        while tenant_queues.pending_count() < max_b and time.monotonic() < deadline:
            time.sleep(0.0002)
            poll()

        # Last look before touching the tasks' memory.
        for i in range(n):
            if task_of[i] is not None and dyings[i].is_set():
                drop_lane(i, "task process failed")
        batch = tenant_queues.select_batch(policy, max_b)
        if not batch:
            continue

        if pipeline is None:
            fail_batch(batch, "backbone_not_loaded")
            continue
        try:
            run_batch(batch)
        except Exception as e:
            print(f"[BackboneProc] Forward error: {e}")
            fail_batch(batch, str(e))

    print("[BackboneProc] Exited")


# ---------------------------------------------------------------------------
# Helper to spawn
# ---------------------------------------------------------------------------

def start_backbone_process(ctx, config, device: str, sched_qs: list, done_qs: list,
                           dyings: list, ctrl_q, ctrl_reply_q, ack_q, run_stop):
    p = ctx.Process(
        target=backbone_proc_main,
        args=(config, device, sched_qs, done_qs, dyings, ctrl_q, ctrl_reply_q,
              ack_q, run_stop),
        daemon=True,
        name="backbone_proc",
    )
    p.start()
    return p
