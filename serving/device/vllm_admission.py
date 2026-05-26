"""STFQ-based admission scheduler in front of vLLM's AsyncLLMEngine.

vLLM owns continuous batching internally — we only control the *order* in
which requests reach the engine. This module reuses the same `STFQPolicy`
the device batcher uses for the PyTorch path ([device/scheduler.py]),
guaranteeing identical fairness semantics between the TSFM/vision BFQ runs
and the LLM admission runs.

The control flow mirrors `device/batcher.py` + `device/batch_former.py`:

    enqueue:        queues.push + policy.assign_start_time + condition.notify
    coordinator:    sem.acquire (≡ worker-free) → wait for pending → policy.select
                    (advances STFQ's virtual clock _v) → spawn _dispatch
    _dispatch:      await runtime.infer → policy.update_after_execution
                    (re-chains queued VFTs with real duration) → sem.release

The only structural difference from the PyTorch path is that up to
`max_in_flight` dispatches run concurrently inside vLLM (the PyTorch worker
is strictly serial). STFQ remains correct under concurrency — `_v` simply
advances faster, and each completion re-chains its own task's queue
independently. Service-time passed to `update_after_execution` is wall-clock
latency, which over-estimates true per-request service share under sharing
(small bias toward the *other* task; conservative).

Activated when:
    --runtime-type vllm  AND  --scheduler-policy {stfq,wfq}  AND  --task-rates …

Weight convention (matches device/server.py:85): `weight_t = 1/rps_t`. Low-RPS
victim → high weight → slow virtual-finish-time advance → stays near the front.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Dict, Optional

from device.scheduler import RequestEnvelope, STFQPolicy, TenantQueues


class VLLMAdmissionScheduler:
    """Per-task queues drained by a coordinator that calls STFQPolicy.select()
    — the same call DeviceBatcher uses — to admit one request at a time into
    vLLM. Up to `max_in_flight` concurrent dispatches inside vLLM, which does
    its own continuous batching across that set."""

    def __init__(self, runtime, weights: Dict[str, float], max_in_flight: int = 32):
        if max_in_flight < 1:
            raise ValueError(f"max_in_flight must be >= 1, got {max_in_flight}")
        self._runtime = runtime
        self._policy = STFQPolicy(weights=dict(weights))
        self._queues = TenantQueues()
        # Pre-create per-task deques so policy.select() sees them from the start.
        for task in weights:
            self._queues._queues.setdefault(task, deque())
        self._sem = asyncio.Semaphore(max_in_flight)
        self._condition = asyncio.Condition()
        self._stop = False
        self._coordinator_task: Optional[asyncio.Task] = None
        print(f"[VLLMAdmission] STFQ weights={dict(weights)} max_in_flight={max_in_flight}")

    async def start(self) -> None:
        if self._coordinator_task is None:
            self._coordinator_task = asyncio.create_task(self._coordinator())

    async def stop(self) -> None:
        async with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._coordinator_task is not None:
            try:
                await asyncio.wait_for(self._coordinator_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._coordinator_task.cancel()
            self._coordinator_task = None

    async def submit(self, req_id: int, prompt: str, task: str) -> dict:
        # Unknown tasks bypass the scheduler entirely.
        if task not in self._queues._queues:
            return await self._runtime.infer(req_id, prompt, task)
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        env = RequestEnvelope(
            req_id=req_id, task=task, x=None, mask=None,
            question=prompt, enqueued_at=time.time(), future=fut,
        )
        async with self._condition:
            self._queues.push(env)
            # Eagerly assign STFQ virtual_start (uses current _v as floor)
            # and advance the per-task speculative _vft — same as BatchFormer.
            self._policy.assign_start_time(env)
            self._condition.notify_all()
        return await fut

    async def _coordinator(self) -> None:
        """One pick per loop iteration, exactly like DeviceBatcher.run_forever:
            wait for a free 'worker' (sem slot) → wait for pending work →
            policy.select(queues, 1) → dispatch concurrently."""
        try:
            while not self._stop:
                # Analog of `await self._work_done.wait()` — block until there
                # is room in vLLM's in-flight window.
                await self._sem.acquire()
                # Analog of BatchFormer.next_batch wait loop — block until at
                # least one request is queued.
                async with self._condition:
                    while self._queues.pending_count() == 0 and not self._stop:
                        await self._condition.wait()
                    if self._stop:
                        self._sem.release()
                        return
                    # Policy selection — advances _v and records _last_dispatch_s
                    # so update_after_execution can re-chain queued VFTs.
                    picked = self._queues.select_batch(self._policy, max_batch_size=1)
                if not picked:
                    self._sem.release()
                    continue
                asyncio.create_task(self._dispatch(picked[0]))
        except asyncio.CancelledError:
            pass

    async def _dispatch(self, env: RequestEnvelope) -> None:
        """Run one request in vLLM, then correct VFTs with the real service
        time. Wall-clock duration under N-way concurrency overestimates the
        per-request share but is conservative for fairness."""
        t0 = time.time()
        try:
            result = await self._runtime.infer(env.req_id, env.question, env.task)
            if not env.future.done():
                env.future.set_result(result)
        except Exception as e:
            if not env.future.done():
                env.future.set_exception(e)
        finally:
            duration_s = time.time() - t0
            async with self._condition:
                self._policy.update_after_execution(
                    [env.task], duration_s, self._queues._queues,
                )
            # Releasing the sem unblocks the coordinator's sem.acquire(); no
            # explicit condition notify needed unless the queue was empty,
            # in which case the next submit() will notify.
            self._sem.release()
