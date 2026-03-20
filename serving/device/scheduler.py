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
    virtual_start: float = 0.0


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


class STFQPolicy:
    """Start-time Fair Queuing (STFQ) across task queues.

    Each request gets a virtual start time S_i and finish time F_i:
        S_i = max(V, F_i_prev)   where V is the virtual clock
        F_i = S_i + 1/w_i        (weight-based increment, no size estimate)

    Dispatch: always pick the request with the smallest S_i.
    Virtual clock V advances to S_i of the just-dispatched request.

    Weights = 1/rps of each task (same convention as WFQPolicy). Low-RPS
    victim gets high weight → 1/w is small → F advances slowly → task stays
    near front of virtual timeline → served promptly even under aggressor load.
    High-RPS aggressor gets low weight → F advances quickly → served less
    often per unit virtual time, proportional to its share.

    Idle flow reset: if a task had no pending requests, F_i_prev is stale.
    On its next arrival, S_i = max(V, F_i_prev) naturally resets to V
    (current virtual time) if F_i_prev < V, preventing backlog priority.
    """

    def __init__(self, weights: Dict[str, float] | None = None):
        self._weights: Dict[str, float] = weights or {}
        # Virtual finish time of the last request served/assigned per task
        self._vft: Dict[str, float] = {}
        # System virtual clock — updated to S_i of each dispatched request
        self._v: float = 0.0
        # S_i of the last dispatched request per task — used for post-execution correction
        self._last_dispatch_s: Dict[str, float] = {}

    def assign_start_time(self, request: RequestEnvelope) -> None:
        """Assign virtual_start eagerly at enqueue time and advance _vft speculatively."""
        s = max(self._v, self._vft.get(request.task, 0.0))
        request.virtual_start = s
        self._vft[request.task] = s + 1.0 / self._weight(request.task)

    def set_weight(self, task: str, weight: float) -> None:
        self._weights[task] = max(weight, 1e-6)

    def _weight(self, task: str) -> float:
        return self._weights.get(task, 1.0)

    def select(self, queues: dict[str, deque], max_batch_size: int) -> list[RequestEnvelope]:
        picked: list[RequestEnvelope] = []

        # Clear per-batch tracking at the start of each select() call
        self._last_dispatch_s = {}

        while len(picked) < max_batch_size:
            best_task = None
            best_s = None

            for task, queue in queues.items():
                if not queue:
                    continue
                candidate = queue[0]
                if picked and candidate.x.shape != picked[0].x.shape:
                    continue
                # S_i was frozen at enqueue time — just read it
                s = candidate.virtual_start
                if best_task is None or s < best_s:
                    best_task = task
                    best_s = s

            if best_task is None:
                break

            req = queues[best_task].popleft()
            picked.append(req)

            # Store LAST S_i per task — the tail of the dispatched chain for
            # this task. Queued requests re-chain off this in update_after_execution,
            # preserving each task's virtual timeline correctly.
            # (Alternative: store FIRST S_i only — uncomment below and comment above)
            # if best_task not in self._last_dispatch_s:
            #     self._last_dispatch_s[best_task] = best_s
            self._last_dispatch_s[best_task] = best_s

            # _v advances to the largest S_i picked — represents how far through
            # the virtual timeline this batch has consumed.
            self._v = best_s

        return picked

    def update_after_execution(
        self,
        tasks: list[str],
        actual_duration_s: float,
        queues: dict,
    ) -> None:
        """Correct VFT and re-chain queued requests for tasks that just ran.

        Only tasks present in the batch need correction — other task queues
        were not affected by this batch's speculative VFT assignments.

        For each affected task:
          1. Recompute real finish time: F = S_dispatched + actual_duration / w
          2. Re-chain all still-queued requests using STFQ formula:
               S_i = max(V, F_{i-1})
               F_i = S_i + 1/w
          3. Update _vft[task] to F of the last queued request (for next arrival)
        """
        for task in set(tasks):
            s_dispatched = self._last_dispatch_s.get(task)
            if s_dispatched is None:
                continue
            w = self._weight(task)
            # Corrected finish time of the dispatched request
            corrected_f = s_dispatched + actual_duration_s / w
            # Re-chain remaining queued requests for this task
            running_f = corrected_f
            for req in queues.get(task, []):
                s_i = max(self._v, running_f)
                req.virtual_start = s_i
                running_f = s_i + 1.0 / w
            # Update _vft so next arrival chains off the corrected end
            self._vft[task] = running_f


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


class SABAPolicy:
    """SLO-Aware Batch slot Allocator (SABA).

    Allocates batch slots in proportion to each task's registered arrival rate,
    guaranteeing the victim gets exactly enough slots to drain its queue at
    steady state while giving surplus capacity to the aggressor.

    Per-batch allocation:
      need[t] = round(rps[t] * estimated_batch_duration_s), capped at queue depth
      If sum(need) <= B: allocate need[t] each, fill spare from deepest queue
      If sum(need) >  B: scale down proportionally (each task gets >= 1 slot)

    Batch duration is tracked via an exponential moving average updated after
    every executed batch by calling update_batch_duration(seconds).

    Why this is FM-specific:
      The batch is the atomic unit of GPU execution — you cannot preempt
      mid-batch. SABA decides the entire batch composition upfront based on
      throughput targets rather than scheduling requests one-at-a-time (WFQ)
      or alternating tasks blindly (RR).
    """

    def __init__(self, rates: Dict[str, float], initial_batch_duration_s: float = 0.035):
        # rates: {task: rps} — registered arrival rates
        self._rates: Dict[str, float] = dict(rates)
        self._batch_duration_s: float = initial_batch_duration_s

    def update_batch_duration(self, duration_s: float) -> None:
        """EMA update called after each executed batch."""
        self._batch_duration_s = 0.8 * self._batch_duration_s + 0.2 * duration_s

    def set_rate(self, task: str, rps: float) -> None:
        self._rates[task] = max(rps, 1e-6)

    def select(self, queues: dict[str, deque], max_batch_size: int) -> list[RequestEnvelope]:
        B = max_batch_size

        # Determine reference shape from first available request
        ref_shape = None
        eligible: dict[str, deque] = {}
        for task, queue in queues.items():
            if not queue:
                continue
            if ref_shape is None:
                ref_shape = queue[0].x.shape
            if queue[0].x.shape == ref_shape:
                eligible[task] = queue

        if not eligible:
            return []

        # Step 1: compute how many slots each task needs this batch
        need: dict[str, int] = {}
        for task, queue in eligible.items():
            rps = self._rates.get(task, 1.0)
            raw = rps * self._batch_duration_s
            need[task] = min(max(1, round(raw)), len(queue))

        total_need = sum(need.values())

        # Step 2: scale down proportionally if over budget
        if total_need <= B:
            alloc = dict(need)
            spare = B - total_need
        else:
            # Scale proportionally, guarantee at least 1 slot per task
            alloc = {}
            for task, n in need.items():
                alloc[task] = max(1, int(n * B / total_need))
            # Re-check after rounding — trim from largest if still over budget
            while sum(alloc.values()) > B:
                heaviest = max(alloc, key=lambda t: alloc[t])
                if alloc[heaviest] <= 1:
                    break
                alloc[heaviest] -= 1
            # Guarantee each protectable task's throughput covers its arrival
            # rate.  A task is "protectable" if its steady-state need fits within
            # B slots (rps * batch_duration_s <= B) — i.e. it is a low-rate
            # tenant that proportional scaling may have under-served.  High-rate
            # tasks whose need already exceeds B are inherently limited and are
            # not candidates for this correction.
            for task in list(alloc.keys()):
                rps = self._rates.get(task, 1.0)
                steady_need = rps * self._batch_duration_s
                if steady_need > B:
                    continue  # cannot fully serve this task in one batch anyway
                if alloc[task] / self._batch_duration_s < rps:
                    donor = max(
                        (t for t in alloc if t != task and alloc[t] > 1
                         and self._rates.get(t, 1.0) * self._batch_duration_s > B),
                        key=lambda t: alloc[t],
                        default=None,
                    )
                    if donor is not None:
                        alloc[donor] -= 1
                        alloc[task]  += 1
                    elif sum(alloc.values()) < B:
                        alloc[task] += 1
            spare = max(0, B - sum(alloc.values()))

        # Step 3: pick allocated slots (FIFO within each task)
        picked: list[RequestEnvelope] = []
        for task, slots in alloc.items():
            queue = eligible[task]
            for _ in range(slots):
                if queue:
                    picked.append(queue.popleft())

        # Step 4: fill spare slots from the deepest queue
        if spare > 0:
            by_depth = sorted(eligible.items(), key=lambda kv: -len(kv[1]))
            for task, queue in by_depth:
                while spare > 0 and queue:
                    picked.append(queue.popleft())
                    spare -= 1

        return picked


class DeadlineSplitPolicy:
    """Deadline-driven batch splitting for FM serving.

    Core idea: instead of mixing victim requests into a large aggressor batch
    (which inflates exec time to ~54ms), fire a small dedicated victim-only
    batch on a timer, interleaved between full aggressor batches.

    Per select() call:
      - Track when each task last had a batch fired for it.
      - If a protected task (rps[t] <= split_rps_threshold) has requests queued
        AND its deadline has elapsed (now >= last_fired[t] + 1/rps[t]):
            → fire a small batch of up to split_max_size requests for that task only
      - Otherwise:
            → fire a full aggressor batch of max_batch_size

    Result at phase4 (victim=20rps, aggressor=300rps, bsize=8):
      - Victim batch: size~1, exec~9ms, fires every ~50ms  → victim latency ~9ms
      - Aggressor batch: size~8, exec~54ms, fires in between → aggressor gets
        ~41ms of GPU time per 50ms cycle = ~82% of max throughput
      - Victim latency stays at single-batch baseline regardless of aggressor load

    Parameters:
      rates             : {task: rps} — registered arrival rates
      split_rps_threshold: tasks with rps <= this are "protected" (get dedicated batches)
      split_max_size    : max requests in a dedicated victim batch (default 2)
    """

    def __init__(
        self,
        rates: Dict[str, float],
        split_rps_threshold: float = 50.0,
        split_max_size: int = 2,
    ):
        self._rates = dict(rates)
        self._split_rps_threshold = split_rps_threshold
        self._split_max_size = split_max_size
        # Wall time of last batch fired per task (0 = never fired)
        self._last_fired: Dict[str, float] = {}

    def _is_protected(self, task: str) -> bool:
        return self._rates.get(task, 1.0) <= self._split_rps_threshold

    def select(self, queues: dict[str, deque], max_batch_size: int) -> list[RequestEnvelope]:
        now = time.monotonic()

        # Determine reference shape from first available request
        ref_shape = None
        for queue in queues.values():
            if queue:
                ref_shape = queue[0].x.shape
                break
        if ref_shape is None:
            return []

        # Check if any protected task has hit its deadline
        for task, queue in queues.items():
            if not queue or not self._is_protected(task):
                continue
            if queue[0].x.shape != ref_shape:
                continue
            rps = self._rates.get(task, 1.0)
            deadline_interval = 1.0 / rps
            last = self._last_fired.get(task, 0.0)
            if now >= last + deadline_interval:
                # Fire a small dedicated batch for this task only
                picked: list[RequestEnvelope] = []
                while queue and len(picked) < self._split_max_size:
                    picked.append(queue.popleft())
                self._last_fired[task] = now
                return picked

        # No protected task is overdue — fire a full aggressor batch
        picked = []
        # Fill from non-protected (high-rate) queues first, then protected if room
        for protected in [False, True]:
            for task, queue in queues.items():
                if not queue:
                    continue
                if self._is_protected(task) != protected:
                    continue
                candidate = queue[0]
                if picked and candidate.x.shape != picked[0].x.shape:
                    continue
                while queue and len(picked) < max_batch_size:
                    if queue[0].x.shape != ref_shape:
                        break
                    picked.append(queue.popleft())
                if len(picked) >= max_batch_size:
                    break

        if picked:
            # Record fire time for all tasks present
            tasks_fired = {r.task for r in picked}
            for task in tasks_fired:
                self._last_fired[task] = now

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
