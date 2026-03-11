"""
fmvisor.py — FM Supervisor: queue, batch assembly, worker thread, and dispatch.

Sits between fmapi and fm. Only in the inference path.
Lifecycle operations (load, add_decoder, swap_backbone) go directly from fmapi to fm.

Owns:
  - Per-task request queues (via scheduler)
  - Persistent worker thread — the single gate to fm.run_batch()
  - Async batch assembly loop — assembles batch N+1 while worker executes batch N
  - Future resolution — resolves each request's future from the worker thread

The pipelining design (from old batcher.py):
  async loop:  [assemble N] → wait worker free → hand off N → [assemble N+1] → ...
  worker:                      [run_batch N ...................done] → signal
This overlaps batch assembly with GPU execution, eliminating inter-batch gaps.
"""

import asyncio
import threading
import time

import numpy as np

from device.fm import FoundationModel
from device.scheduler import FifoPolicy, RequestEnvelope, TenantQueues


class FMVisor:
    """
    Supervises inference request flow: queue → batch → worker thread → fm → futures.

    fmapi calls enqueue() for each incoming Infer RPC.
    The internal async loop assembles batches and hands them to the persistent
    worker thread, which calls fm.run_batch() and resolves futures directly —
    exactly as the old DeviceBatcher did.
    """

    def __init__(
        self,
        fm: FoundationModel,
        max_batch_size: int = 1,
        max_batch_wait_ms: float = 1.0,
        queue_capacity: int = 1024,
    ):
        self._fm = fm
        self._queues = TenantQueues()
        self._policy = FifoPolicy()
        self._max_batch_size = max_batch_size
        self._max_batch_wait_s = max_batch_wait_ms / 1000.0
        self._queue_capacity = queue_capacity
        self._condition = asyncio.Condition()
        self._stopped = False

        # Persistent worker thread — mirrors old batcher.py exactly
        self._work_ready = threading.Event()                    # async → worker: batch ready
        self._work_done: asyncio.Event | None = None            # worker → async: batch done
        self._next_prepared = None                              # the batch handed to worker
        self._worker_thread: threading.Thread | None = None
        self._worker_loop_ref: asyncio.AbstractEventLoop | None = None

    async def start(self):
        """
        Start the persistent worker thread and the async batch assembly loop.

        Captures the running event loop so the worker thread can signal back
        via call_soon_threadsafe.
        """
        loop = asyncio.get_running_loop()
        self._worker_loop_ref = loop
        self._work_done = asyncio.Event()

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="fmvisor-worker")
        self._worker_thread.start()
        asyncio.create_task(self._batch_loop())
        print("[FMVisor] Started")

    async def stop(self):
        """Signal the batch loop to stop and drain the worker thread."""
        async with self._condition:
            self._stopped = True
            self._condition.notify_all()

    async def enqueue(self, request: RequestEnvelope):
        """
        Accept one inference request from fmapi and add it to the queue.

        Raises RuntimeError("queue_full") if at capacity — fmapi translates
        this to a gRPC RESOURCE_EXHAUSTED response.
        The caller awaits request.future to receive the inference result.
        """
        async with self._condition:
            pending = self._queues.pending_count()
            if pending >= self._queue_capacity:
                print(
                    f"[FMVisor] Queue full req={request.req_id} task={request.task} "
                    f"(capacity={self._queue_capacity})"
                )
                raise RuntimeError("queue_full")
            self._queues.push(request)
            pending_after = pending + 1
            if pending_after == 1 or pending_after % 50 == 0:
                print(
                    f"[FMVisor] Enqueued req={request.req_id} task={request.task} "
                    f"total_pending={pending_after} per_task={self._queues.snapshot_depths()}"
                )
            self._condition.notify_all()

    # ------------------------------------------------------------------
    # Async batch assembly loop — mirrors old run_forever()
    # ------------------------------------------------------------------

    async def _batch_loop(self):
        """
        Continuous loop: assemble batch → wait for worker free → hand off → repeat.

        Pipelining: _next_batch() runs concurrently with the worker executing
        the previous batch, so batch assembly overlaps GPU execution.
        """
        print("[FMVisor] Batch loop started")
        while True:
            # Assemble next batch (runs while worker executes previous batch)
            prepared = await self._next_batch()
            if prepared is None:
                print("[FMVisor] Batch loop stopping")
                self._next_prepared = None
                self._work_ready.set()
                if self._worker_thread:
                    self._worker_thread.join(timeout=10)
                return

            # Wait for worker to be free, then hand off immediately
            await self._work_done.wait()
            self._work_done.clear()

            self._next_prepared = prepared
            self._work_ready.set()   # unblocks worker thread

    async def _next_batch(self):
        """
        Wait until at least one request is queued, then form a batch.

        Waits up to max_batch_wait_ms for the batch to fill before dispatching
        a partial batch. Returns None if stopped.
        """
        async with self._condition:
            while self._queues.pending_count() == 0 and not self._stopped:
                await self._condition.wait()
            if self._stopped:
                return None

        deadline = time.monotonic() + self._max_batch_wait_s
        while time.monotonic() < deadline:
            async with self._condition:
                if self._queues.pending_count() >= self._max_batch_size:
                    break
            await asyncio.sleep(0)

        async with self._condition:
            requests = self._policy.select(self._queues, self._max_batch_size)
        if not requests:
            return None
        return self._prepare_batch(requests)

    def _prepare_batch(self, requests: list[RequestEnvelope]):
        """Concatenate per-request tensors into a single batch for fm.run_batch()."""
        batch_ids = [r.req_id for r in requests]
        task_names = [r.task for r in requests]
        print(f"[FMVisor] Prepared batch size={len(requests)} req_ids={batch_ids} tasks={task_names}")
        x = np.concatenate([r.x for r in requests], axis=0)
        masks = [r.mask for r in requests if r.mask is not None]
        mask = np.concatenate(masks, axis=0) if len(masks) == len(requests) and masks else None
        return _PreparedBatch(requests=requests, x=x, task_names=task_names, mask=mask)

    # ------------------------------------------------------------------
    # Worker thread — the single gate to fm.run_batch()
    # ------------------------------------------------------------------

    def _worker_loop(self):
        """
        Persistent worker thread body.

        Waits for a batch via _work_ready, calls fm.run_batch(), resolves
        each request's future via call_soon_threadsafe, then signals _work_done
        so the async loop can hand off the next batch.

        Futures are resolved here in the worker thread — no result needs to be
        passed back to the async side, keeping the design simple.
        """
        loop = self._worker_loop_ref
        # Signal async side that worker is ready for the first batch
        loop.call_soon_threadsafe(self._work_done.set)

        while True:
            self._work_ready.wait()
            self._work_ready.clear()

            prepared = self._next_prepared
            if prepared is None:
                return  # shutdown sentinel

            self._execute(prepared)
            loop.call_soon_threadsafe(self._work_done.set)

    def _execute(self, prepared):
        """
        Call fm.run_batch() and resolve each request's future.

        Runs entirely in the worker thread. Futures are resolved via
        call_soon_threadsafe so the asyncio event loop handles them safely.
        """
        print(
            f"[FMVisor] Executing batch size={len(prepared.requests)} "
            f"tasks={prepared.task_names}"
        )
        loop = self._worker_loop_ref
        result = self._fm.run_batch(prepared.x, prepared.task_names, prepared.mask)
        for i, request in enumerate(prepared.requests):
            payload = {
                "output": result.outputs[i],
                "start_time_ns": result.start_time_ns,
                "end_time_ns": result.end_time_ns,
                "proc_time_ns": result.proc_time_ns,
                "swap_time_ns": result.swap_time_ns[i],
                "decoder_time_ns": result.decoder_time_ns[i],
            }
            loop.call_soon_threadsafe(_resolve_future, request.future, payload)
        print(
            f"[FMVisor] Batch done size={len(prepared.requests)} "
            f"start={result.start_time_ns} end={result.end_time_ns}"
        )


class _PreparedBatch:
    """Internal container for a batch ready to hand to the worker thread."""
    __slots__ = ("requests", "x", "task_names", "mask")

    def __init__(self, requests, x, task_names, mask):
        self.requests = requests
        self.x = x
        self.task_names = task_names
        self.mask = mask


def _resolve_future(future, payload):
    """Set future result if not already resolved. Called via call_soon_threadsafe."""
    if not future.done():
        future.set_result(payload)
