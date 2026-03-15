from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class RequestEnvelope:
    req_id: int
    task: str
    x: Any
    mask: Any
    question: Any
    enqueued_at: float
    future: Any


class FifoPolicy:
    """Simple cross-task FIFO baseline."""

    def select(self, queues: dict[str, deque], max_batch_size: int) -> list[RequestEnvelope]:
        picked: list[RequestEnvelope] = []
        while len(picked) < max_batch_size:
            next_item = None
            next_task = None
            for task, queue in queues.items():
                if not queue:
                    continue
                candidate = queue[0]
                #need to check shape of the request input here to make sure the batch is homogeneous, otherwise we can end up with a batch of 2 requests where one has shape (1, 512) and the other has shape (1, 1024) which will cause an error when we try to concatenate them
                if picked and candidate.x.shape != picked[0].x.shape:
                    continue
                if next_item is None or candidate.enqueued_at < next_item.enqueued_at:
                    next_item = candidate
                    next_task = task
            if next_item is None or next_task is None:
                break
            picked.append(queues[next_task].popleft())
        return picked


class RoundRobinPolicy:
    """Fair cross-task scheduling via round-robin over task queues.

    Rotates through tasks in a fixed order, taking at most one request per
    task per round. Each task gets an equal number of batch slots regardless
    of its queue depth, so a high-RPS aggressor cannot starve a low-RPS
    victim.

    Shape constraint: once the first request in a batch fixes the shape,
    tasks whose head request has a different shape are skipped for this batch
    (same rule as FifoPolicy).
    """

    def __init__(self):
        # Tracks which task to start from in the next select() call so that
        # no task always gets priority at the start of a new batch.
        self._last_task_index: int = 0

    def select(self, queues: dict[str, deque], max_batch_size: int) -> list[RequestEnvelope]:
        task_names = [t for t in queues if queues[t]]   # only non-empty queues
        if not task_names:
            return []

        # Rotate starting task so no single task always leads the batch.
        start = self._last_task_index % len(task_names)
        ordered = task_names[start:] + task_names[:start]

        picked: list[RequestEnvelope] = []
        # Keep cycling through tasks until the batch is full or all queues
        # are exhausted / shape-incompatible.
        made_progress = True
        while len(picked) < max_batch_size and made_progress:
            made_progress = False
            for task in ordered:
                if len(picked) >= max_batch_size:
                    break
                queue = queues[task]
                if not queue:
                    continue
                candidate = queue[0]
                if picked and candidate.x.shape != picked[0].x.shape:
                    continue
                picked.append(queue.popleft())
                made_progress = True

        if task_names:
            self._last_task_index = (start + 1) % len(task_names)

        return picked


class TenantQueues:
    """Holds one logical queue per task and exposes batch selection."""

    def __init__(self):
        self._queues: dict[str, deque] = {}

    def push(self, request: RequestEnvelope):
        self._queues.setdefault(request.task, deque()).append(request)

    def pending_count(self) -> int:
        return sum(len(queue) for queue in self._queues.values())

    def select_batch(self, policy: "FifoPolicy | RoundRobinPolicy", max_batch_size: int) -> list[RequestEnvelope]:
        return policy.select(self._queues, max_batch_size)

    def snapshot_depths(self) -> dict[str, int]:
        return {task: len(queue) for task, queue in self._queues.items() if queue}

