"""
scheduler.py — Pure scheduling policy and queue data structures.

No I/O, no gRPC, no torch. Stateless logic only.
Answers one question: given the current queues, which requests form the next batch?

Classes:
  RequestEnvelope  — container for a single inference request as it moves through the system
  PreparedBatch    — a batch of requests ready to be handed to the FM for execution
  TenantQueues     — per-task FIFO queues; tracks all pending requests
  FifoPolicy       — batch selection policy: oldest-first, shape-homogeneous
"""

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RequestEnvelope:
    """
    A single inference request moving through the system.

    Created by fmapi when a gRPC Infer call arrives.
    Enqueued into TenantQueues by fmvisor.
    Carried through batch assembly and returned to fmapi via the future.

    Fields:
      req_id      — unique request identifier from the client
      task        — task name (e.g. "ecgclass"), used to select the right decoder
      x           — input tensor as numpy array
      mask        — optional attention mask
      question    — optional text question (for LLaVA-style models)
      enqueued_at — wall-clock time when enqueued, used for FIFO ordering
      future      — asyncio.Future resolved by fmvisor when inference completes
    """
    req_id: int
    task: str
    x: Any
    mask: Any
    question: Any
    enqueued_at: float
    future: Any


@dataclass
class PreparedBatch:
    """
    A batch of requests ready for FM execution.

    Built by FifoPolicy.select() from TenantQueues.
    Passed from fmvisor to fm for inference.

    Fields:
      requests   — original RequestEnvelope objects (needed to resolve futures)
      x          — concatenated input tensors, shape (batch_size, ...)
      task_names — task name per request, parallel to requests
      mask       — concatenated masks if all requests have one, else None
    """
    requests: list[RequestEnvelope]
    x: np.ndarray
    task_names: list[str]
    mask: np.ndarray | None


class TenantQueues:
    """
    Holds one FIFO deque per task name.

    fmvisor pushes incoming requests here and polls for pending count.
    FifoPolicy reads from these queues to select batches.
    """

    def __init__(self):
        self._queues: dict[str, deque] = {}

    def push(self, request: RequestEnvelope):
        """Add a request to its task's queue, creating the queue if needed."""
        self._queues.setdefault(request.task, deque()).append(request)

    def pending_count(self) -> int:
        """Total number of requests waiting across all task queues."""
        return sum(len(q) for q in self._queues.values())

    def snapshot_depths(self) -> dict[str, int]:
        """Per-task queue depths, for logging."""
        return {task: len(q) for task, q in self._queues.items() if q}

    def queues(self) -> dict[str, deque]:
        """Direct access to the underlying queues, used by policy classes."""
        return self._queues


class FifoPolicy:
    """
    Batch selection policy: oldest-first, shape-homogeneous.

    Picks up to max_batch_size requests in arrival order.
    Skips requests whose input shape differs from the first picked request,
    since the FM requires a homogeneous batch for concatenation.

    This is intentionally stateless — swap it for a priority or SLO-aware
    policy without changing any other component.
    """

    def select(self, queues: TenantQueues, max_batch_size: int) -> list[RequestEnvelope]:
        """
        Select the next batch from the given queues.

        Returns up to max_batch_size requests. May return fewer if queues are
        sparse or shapes are heterogeneous.
        """
        raw = queues.queues()
        picked: list[RequestEnvelope] = []

        while len(picked) < max_batch_size:
            next_item = None
            next_task = None

            for task, queue in raw.items():
                if not queue:
                    continue
                candidate = queue[0]
                # Enforce shape homogeneity: skip requests that would break concatenation
                if picked and candidate.x.shape != picked[0].x.shape:
                    continue
                if next_item is None or candidate.enqueued_at < next_item.enqueued_at:
                    next_item = candidate
                    next_task = task

            if next_item is None or next_task is None:
                break
            picked.append(raw[next_task].popleft())

        return picked
