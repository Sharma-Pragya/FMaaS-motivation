import asyncio
import threading
import time
from dataclasses import dataclass

import numpy as np

from device.runtime import SharedModelRuntime
from device.scheduler import FifoPolicy, RequestEnvelope, TenantQueues


@dataclass
class PreparedBatch:
    requests: list[RequestEnvelope]
    x: np.ndarray
    task_names: list[str]
    mask: np.ndarray | None


class DeviceBatcher:
    """Owns per-task queues and a single shared-model execution loop.

    The async scheduler loop assembles batches and hands them to a persistent
    worker thread via threading.Event signals.  The worker runs inference and
    immediately signals back so _next_batch can run concurrently with GPU work,
    eliminating asyncio.to_thread dispatch latency (~10 ms) between batches.
    """

    def __init__(
        self,
        runtime: SharedModelRuntime,
        max_batch_size: int = 1,
        max_batch_wait_ms: float = 1.0,
        queue_capacity: int = 1024,
    ):
        self._runtime = runtime
        self._queues = TenantQueues()
        self._policy = FifoPolicy()
        self._max_batch_size = max_batch_size
        self._max_batch_wait_s = max_batch_wait_ms / 1000.0
        self._queue_capacity = queue_capacity
        self._condition = asyncio.Condition()
        self._stopped = False

        # Persistent worker thread state
        self._work_ready = threading.Event()   # async → worker: batch is ready
        self._work_done = None                 # asyncio.Event set by worker when done
        self._next_prepared: PreparedBatch | None = None
        self._worker_thread: threading.Thread | None = None
        self._worker_loop_ref: asyncio.AbstractEventLoop | None = None

    async def enqueue(self, request: RequestEnvelope):
        async with self._condition:
            pending_before = self._queues.pending_count()
            if pending_before >= self._queue_capacity:
                print(
                    f"[DeviceBatcher] Queue full for req={request.req_id} task={request.task} "
                    f"(capacity={self._queue_capacity})"
                )
                raise RuntimeError("queue_full")
            self._queues.push(request)
            pending_after = pending_before + 1
            if pending_after == 1 or pending_after % 50 == 0:
                print(
                    f"[DeviceBatcher] Enqueued req={request.req_id} task={request.task} "
                    f"total_pending={pending_after} per_task={self._queues.snapshot_depths()}"
                )
            self._condition.notify_all()

    async def run_forever(self):
        print("[DeviceBatcher] Scheduler loop started")
        loop = asyncio.get_running_loop()
        self._worker_loop_ref = loop
        self._work_done = asyncio.Event()

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        print("[DeviceBatcher] Persistent worker thread started")

        while True:
            # Assemble next batch (runs while worker executes previous batch)
            prepared = await self._next_batch()
            if prepared is None:
                print("[DeviceBatcher] Scheduler loop stopping")
                # Signal worker to exit
                self._next_prepared = None
                self._work_ready.set()
                self._worker_thread.join(timeout=10)
                return

            # Wait for worker to be free, then hand off immediately
            await self._work_done.wait()
            self._work_done.clear()

            self._next_prepared = prepared
            self._work_ready.set()  # unblocks worker thread

    async def stop(self):
        async with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def _worker_loop(self):
        loop = self._worker_loop_ref
        # Signal async side that worker is ready for first batch
        loop.call_soon_threadsafe(self._work_done.set)

        while True:
            self._work_ready.wait()
            self._work_ready.clear()

            prepared = self._next_prepared
            if prepared is None:
                # Shutdown sentinel
                return

            self._execute_prepared(prepared)
            # Signal async side that we're done — it can dispatch the next batch
            loop.call_soon_threadsafe(self._work_done.set)

    async def _next_batch(self):
        async with self._condition:
            while self._queues.pending_count() == 0 and not self._stopped:
                await self._condition.wait()
            if self._stopped:
                return None

        deadline = time.time() + self._max_batch_wait_s
        while time.time() < deadline:
            async with self._condition:
                if self._queues.pending_count() >= self._max_batch_size:
                    break
            remaining = deadline - time.time()
            if remaining > 0:
                await asyncio.sleep(min(0.001, remaining))

        async with self._condition:
            requests = self._queues.select_batch(self._policy, self._max_batch_size)
        if not requests:
            return None
        return self._prepare_batch(requests)

    def _prepare_batch(self, requests: list[RequestEnvelope]) -> PreparedBatch:
        batch_ids = [request.req_id for request in requests]
        task_names = [request.task for request in requests]
        print(
            f"[DeviceBatcher] Prepared batch_size={len(requests)} "
            f"req_ids={batch_ids} tasks={task_names}"
        )
        x = np.concatenate([request.x for request in requests], axis=0)
        masks = [request.mask for request in requests if request.mask is not None]
        mask = np.concatenate(masks, axis=0) if len(masks) == len(requests) and masks else None
        return PreparedBatch(
            requests=requests,
            x=x,
            task_names=task_names,
            mask=mask,
        )

    def _execute_prepared(self, prepared: PreparedBatch):
        batch_ids = [request.req_id for request in prepared.requests]
        print(
            f"[DeviceBatcher] Executing batch_size={len(prepared.requests)} "
            f"req_ids={batch_ids} tasks={prepared.task_names}"
        )
        result = self._runtime.run_batch(prepared.x, prepared.task_names, prepared.mask)
        for index, request in enumerate(prepared.requests):
            payload = {
                "output": result.outputs[index],
                "start_time_ns": result.start_time_ns,
                "end_time_ns": result.end_time_ns,
                "proc_time_ns": result.proc_time_ns,
                "swap_time_ns": result.swap_time_ns[index],
                "decoder_time_ns": result.decoder_time_ns[index],
            }
            request.future.get_loop().call_soon_threadsafe(self._set_result_if_pending, request.future, payload)
        print(
            f"[DeviceBatcher] Finished batch_size={len(prepared.requests)} "
            f"start={result.start_time_ns} end={result.end_time_ns}"
        )

    @staticmethod
    def _set_result_if_pending(future, payload):
        if not future.done():
            future.set_result(payload)
