import asyncio
import time
from dataclasses import dataclass

import numpy as np

from device.scheduler import (
    FifoPolicy,
    RequestEnvelope,
    RoundRobinPolicy,
    STFQPolicy,
    SABAPolicy,
    TenantQueues,
    TokenBucketPolicy,
    WFQPolicy,
)


@dataclass
class PreparedBatch:
    requests: list[RequestEnvelope]
    x: np.ndarray
    task_names: list[str]
    mask: np.ndarray | None
    questions: list[str | None] | None = None  # per-request question strings for VLM


class BatchFormer:
    """Owns per-task queues, the policy, and batch selection.

    No threads, no CUDA. Produces PreparedBatch objects on demand for the
    orchestrator to dispatch.
    """

    def __init__(
        self,
        max_batch_size: int = 1,
        max_batch_wait_ms: float = 1.0,
        queue_capacity: int = 1024,
        policy: "FifoPolicy | RoundRobinPolicy | WFQPolicy | TokenBucketPolicy | None" = None,
    ):
        self._queues = TenantQueues()
        self._policy = policy if policy is not None else FifoPolicy()
        self._max_batch_size = max_batch_size
        self._max_batch_wait_s = max_batch_wait_ms / 1000.0
        self._queue_capacity = queue_capacity
        self._condition = asyncio.Condition()
        self._stopped = False

    @property
    def policy(self):
        return self._policy

    @property
    def queues(self):
        return self._queues

    async def enqueue(self, request: RequestEnvelope):
        async with self._condition:
            pending_before = self._queues.pending_count()
            if pending_before >= self._queue_capacity:
                print(
                    f"[BatchFormer] Queue full for req={request.req_id} task={request.task} "
                    f"(capacity={self._queue_capacity})"
                )
                raise RuntimeError("queue_full")
            self._queues.push(request)
            if isinstance(self._policy, STFQPolicy):
                self._policy.assign_start_time(request)
            # pending_after = pending_before + 1
            # if pending_after == 1 or pending_after % 50 == 0:
            #     print(
            #         f"[BatchFormer] Enqueued req={request.req_id} task={request.task} "
            #         f"total_pending={pending_after} per_task={self._queues.snapshot_depths()}"
            #     )
            self._condition.notify_all()

    async def stop(self):
        async with self._condition:
            self._stopped = True
            self._condition.notify_all()

    async def next_batch(self) -> PreparedBatch | None:
        # Wait until at least one request is queued after the worker becomes
        # free, then accumulate for at most max_batch_wait before dispatch.
        async with self._condition:
            while self._queues.pending_count() == 0 and not self._stopped:
                await self._condition.wait()
            if self._stopped:
                return None

        # Brief accumulation window — lets a few more requests arrive before
        # scheduling, so the policy has something to choose between.
        # Once the batch is full we stop waiting early.
        deadline = time.time() + self._max_batch_wait_s
        while time.time() < deadline:
            async with self._condition:
                if self._queues.pending_count() >= self._max_batch_size:
                    break
            remaining = deadline - time.time()
            if remaining > 0:
                await asyncio.sleep(min(0.001, remaining))

        # Policy selects which requests to run next
        async with self._condition:
            requests = self._queues.select_batch(self._policy, self._max_batch_size)
        if not requests:
            return None
        return self._prepare_batch(requests)

    def _prepare_batch(self, requests: list[RequestEnvelope]) -> PreparedBatch:
        batch_ids = [request.req_id for request in requests]
        task_names = [request.task for request in requests]
        print(
            f"[BatchFormer] Prepared batch_size={len(requests)} "
            f"req_ids={batch_ids} tasks={task_names}"
        )
        xs = [request.x for request in requests]
        x = None if xs[0] is None else np.concatenate(xs, axis=0)
        masks = [request.mask for request in requests if request.mask is not None]
        mask = np.concatenate(masks, axis=0) if len(masks) == len(requests) and masks else None
        questions_raw = [request.question for request in requests]
        questions = questions_raw if any(q is not None for q in questions_raw) else None
        return PreparedBatch(
            requests=requests,
            x=x,
            task_names=task_names,
            mask=mask,
            questions=questions,
        )
