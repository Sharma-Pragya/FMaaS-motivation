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


class TenantQueues:
    """Holds one logical queue per task and exposes batch selection."""

    def __init__(self):
        self._queues: dict[str, deque] = {}

    def push(self, request: RequestEnvelope):
        self._queues.setdefault(request.task, deque()).append(request)

    def pending_count(self) -> int:
        return sum(len(queue) for queue in self._queues.values())

    def select_batch(self, policy: FifoPolicy, max_batch_size: int) -> list[RequestEnvelope]:
        return policy.select(self._queues, max_batch_size)

    def snapshot_depths(self) -> dict[str, int]:
        return {task: len(queue) for task, queue in self._queues.items() if queue}

