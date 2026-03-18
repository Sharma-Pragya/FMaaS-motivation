import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict


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


class WFQPolicy:
    """Weighted Fair Queuing across task queues.

    Each task is assigned a weight (default 1.0 for all tasks).  The scheduler
    maintains a virtual finish time (VFT) per task and always picks the request
    from the task with the smallest VFT.  After picking a request its VFT is
    advanced by 1/weight, so high-weight tasks get proportionally more slots.

    Effect on noisy-neighbor:
      - All tasks equal weight (default): each task gets slots proportional to
        its queue depth but victim wait is bounded to O(1/weight) aggressor
        batches regardless of aggressor RPS.
      - With explicit weights = target RPS ratios: aggressor gets more slots
        but victim wait is still bounded, maximising GPU utilisation.
    """

    def __init__(self, weights: Dict[str, float] | None = None):
        # weights: {task: weight}  — missing tasks default to 1.0
        self._weights: Dict[str, float] = weights or {}
        # virtual finish time per task — starts at 0, advances per request served
        self._vft: Dict[str, float] = {}

    def set_weight(self, task: str, weight: float) -> None:
        self._weights[task] = max(weight, 1e-6)

    def _weight(self, task: str) -> float:
        return self._weights.get(task, 1.0)

    def select(self, queues: dict[str, deque], max_batch_size: int) -> list[RequestEnvelope]:
        picked: list[RequestEnvelope] = []

        while len(picked) < max_batch_size:
            best_task = None
            best_vft  = None

            for task, queue in queues.items():
                if not queue:
                    continue
                candidate = queue[0]
                if picked and candidate.x.shape != picked[0].x.shape:
                    continue
                vft = self._vft.get(task, 0.0)
                if best_task is None or vft < best_vft:
                    best_task = task
                    best_vft  = vft

            if best_task is None:
                break

            req = queues[best_task].popleft()
            picked.append(req)
            # Advance VFT by 1/weight.
            # weight = 1/rps so high-RPS tasks get weight=small → VFT advances
            # fast → they get fewer slots per unit time, preserving fairness.
            # Low-RPS victim has high weight → VFT advances slowly → stays
            # near front of queue and is served promptly.
            self._vft[best_task] = best_vft + 1.0 / self._weight(best_task)

        return picked


class TokenBucketPolicy:
    """Credit-based fair scheduling across task queues.

    Each task accrues credits at its registered credit rate over wall time.
    The scheduler always picks from the task with the most accumulated credit,
    so a victim at 10 rps is always served before an aggressor at 100 rps when
    both have queued requests — regardless of queue depths.

    This works even when max_batch_wait_ms=1 because credit is tracked ACROSS
    batches, not just within a single batch window.  The aggressor burns its
    credit fast; the victim's credit accumulates until it arrives.

    Usage:
        policy = TokenBucketPolicy()
        # credit_rate = 1/rps so low-RPS tenants accrue more credit
        policy.set_rate("ecgclass",     1.0 / 10.0)    # victim @ 10 rps
        policy.set_rate("gestureclass", 1.0 / 100.0)   # aggressor @ 100 rps
    """

    def __init__(self):
        self._rates:   Dict[str, float] = {}   # task -> registered rps
        self._credits: Dict[str, float] = {}   # task -> accumulated credit
        self._last_update: float = time.monotonic()

    def set_rate(self, task: str, credit_rate: float) -> None:
        self._rates[task]   = max(credit_rate, 1e-6)
        self._credits.setdefault(task, 0.0)

    def _accrue(self) -> None:
        now = time.monotonic()
        dt  = now - self._last_update
        self._last_update = now
        for task, rate in self._rates.items():
            self._credits[task] = self._credits.get(task, 0.0) + rate * dt

    def select(self, queues: dict[str, deque], max_batch_size: int) -> list[RequestEnvelope]:
        self._accrue()
        picked: list[RequestEnvelope] = []

        while len(picked) < max_batch_size:
            best_task   = None
            best_credit = None

            for task, queue in queues.items():
                if not queue:
                    continue
                candidate = queue[0]
                if picked and candidate.x.shape != picked[0].x.shape:
                    continue
                # Unknown tasks (not registered) get credit=0 — treated fairly
                credit = self._credits.get(task, 0.0)
                if best_task is None or credit > best_credit:
                    best_task   = task
                    best_credit = credit

            if best_task is None:
                break

            req = queues[best_task].popleft()
            picked.append(req)
            # Deduct one credit per slot consumed
            self._credits[best_task] = self._credits.get(best_task, 0.0) - 1.0

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
