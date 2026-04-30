import asyncio
import threading
import time
from dataclasses import dataclass

from device.batch_former import BatchFormer, PreparedBatch
from device.runtime import SharedModelRuntime
from device.scheduler import (
    FifoPolicy,
    RequestEnvelope,
    RoundRobinPolicy,
    SABAPolicy,
    STFQPolicy,
    TokenBucketPolicy,
    WFQPolicy,
)
from device.task_worker import (
    InlineTaskWorker,
    TaskWorker,
    ThreadedTaskWorker,
)


@dataclass
class _DecoderJob:
    task_name: str
    indices: list[int]              # positions in the original batch
    feats: list                     # feature tensor slices
    n_total: int                    # size of the original batch
    requests: list                  # RequestEnvelope refs for this task only
    backbone_start_ns: int
    backbone_end_ns: int
    proc_time_ns: int
    peak_bytes_bb: int
    backbone_event: object = None   # CUDA event the decoder stream waits on


class DeviceBatcher:
    """Owns per-task queues and a single shared-model execution loop.

    Pipeline:
        backbone (shared, batched)  ->  per-task worker (encoder?+decoder)

    The post-backbone per-task pipeline runs on a TaskWorker. Today that's
    just the decoder; later, encoder + decoder for the same task share one
    worker thread and one CUDA stream so cross-step ordering is implicit.

    The async scheduler loop waits for the backbone worker to become free,
    then asks the BatchFormer for the next batch and hands it to a persistent
    worker thread via threading.Event signals. This keeps batch selection
    aligned with the freshest queue contents at dispatch time.
    """

    def __init__(
        self,
        runtime: SharedModelRuntime,
        max_batch_size: int = 1,
        max_batch_wait_ms: float = 1.0,
        queue_capacity: int = 1024,
        policy: "FifoPolicy | RoundRobinPolicy | WFQPolicy | TokenBucketPolicy | None" = None,
        worker_mode: str = "threaded",   # "threaded" | "inline"
        verbose_batch_logs: bool = False,
    ):
        self._runtime = runtime
        self._batch_former = BatchFormer(
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            queue_capacity=queue_capacity,
            policy=policy,
            verbose_batch_logs=verbose_batch_logs,
        )
        self._worker_mode = worker_mode
        self._verbose_batch_logs = verbose_batch_logs

        # Persistent worker thread state
        self._work_ready = threading.Event()   # async → worker: batch is ready
        self._work_done = None                 # asyncio.Event set by worker when done
        self._next_prepared: PreparedBatch | None = None
        self._worker_thread: threading.Thread | None = None
        self._worker_loop_ref: asyncio.AbstractEventLoop | None = None

        # One per-task worker for the post-backbone pipeline. Today it runs
        # the decoder step; when encoders land, the same worker will run
        # encoder->decoder on its own thread + stream.
        device = getattr(self._runtime, "_loader").device
        if worker_mode == "inline":
            self._task_worker: TaskWorker = InlineTaskWorker(run_fn=self._run_task_pipeline)
        else:
            self._task_worker = ThreadedTaskWorker(
                worker_name="task",
                device=device,
                run_fn=self._run_task_pipeline,
            )

    # -- Backwards-compat shims so external callers (server.py control path)
    # that reach into the batcher's policy/queues keep working.
    @property
    def _policy(self):
        return self._batch_former.policy

    @property
    def _queues(self):
        return self._batch_former.queues

    async def enqueue(self, request: RequestEnvelope):
        await self._batch_former.enqueue(request)

    async def run_forever(self):
        print("[DeviceBatcher] Scheduler loop started")
        loop = asyncio.get_running_loop()
        self._worker_loop_ref = loop
        self._work_done = asyncio.Event()

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        print("[DeviceBatcher] Persistent worker thread started")

        while True:
            # Wait for the worker to be free before selecting the next batch.
            # This allows late arrivals to join the batch about to dispatch.
            await self._work_done.wait()
            self._work_done.clear()

            prepared = await self._batch_former.next_batch()
            if prepared is None:
                print("[DeviceBatcher] Scheduler loop stopping")
                # Signal worker to exit
                self._next_prepared = None
                self._work_ready.set()
                self._worker_thread.join(timeout=10)
                # Tear down per-task workers
                self._task_worker.shutdown()
                return

            self._next_prepared = prepared
            self._work_ready.set()  # unblocks worker thread

    async def stop(self):
        await self._batch_former.stop()

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

            try:
                self._execute_prepared(prepared)
            except Exception as exc:
                import traceback
                print(f"[DeviceBatcher] ERROR in _execute_prepared: {exc}")
                traceback.print_exc()
                for request in prepared.requests:
                    loop.call_soon_threadsafe(self._fail_future_if_pending, request.future, exc)
            # Signal async side that we're done — it can dispatch the next batch
            loop.call_soon_threadsafe(self._work_done.set)

    def _execute_prepared(self, prepared: PreparedBatch):
        if self._verbose_batch_logs:
            batch_ids = [request.req_id for request in prepared.requests]
            print(
                f"[DeviceBatcher] Executing batch_size={len(prepared.requests)} "
                f"req_ids={batch_ids} tasks={prepared.task_names}"
            )
        # Stage 1: backbone on this worker thread.
        bb = self._runtime.run_backbone(
            prepared.x, prepared.task_names, prepared.mask, prepared.questions
        )

        # Policy bookkeeping uses backbone duration as a proxy for batch duration
        # (decoder runs concurrently on per-task threads).
        duration_s = (bb.backbone_end_time_ns - bb.start_time_ns) / 1e9
        policy = self._batch_former.policy
        if isinstance(policy, SABAPolicy):
            policy.update_batch_duration(duration_s)
        if isinstance(policy, STFQPolicy):
            policy.update_after_execution(prepared.task_names, duration_s, self._batch_former.queues._queues)

        # Stage 2: group by task and dispatch the per-task pipeline.
        task_groups: dict[str, list[int]] = {}
        for idx, task_name in enumerate(prepared.task_names):
            task_groups.setdefault(task_name, []).append(idx)

        n_total = len(prepared.task_names)
        for task_name, indices in task_groups.items():
            feats = [bb.feats_by_idx[i] for i in indices]
            reqs  = [prepared.requests[i] for i in indices]
            job = _DecoderJob(
                task_name=task_name,
                indices=indices,
                feats=feats,
                n_total=n_total,
                requests=reqs,
                backbone_start_ns=bb.start_time_ns,
                backbone_end_ns=bb.backbone_end_time_ns,
                proc_time_ns=bb.proc_time_ns,
                peak_bytes_bb=bb.peak_bytes,
                backbone_event=bb.backbone_event,
            )
            self._task_worker.submit(task_name, job)

        if self._verbose_batch_logs:
            print(
                f"[DeviceBatcher] Dispatched batch_size={len(prepared.requests)} "
                f"backbone_end={bb.backbone_end_time_ns} "
                f"dispatched_tasks={list(task_groups.keys())}"
            )

    def _run_task_pipeline(self, task_name: str, job: _DecoderJob, stream) -> None:
        # Today: one step (decoder). Later this same function will run
        # encoder->decoder on the same `stream` so the cross-step dependency
        # is implicit (no event needed inside a task).
        outs, swaps, decs, peak_bytes = self._runtime.run_decoders(
            job.task_name, job.indices, job.feats, job.n_total,
            decoder_stream=stream,
            backbone_event=job.backbone_event,
        )
        end_ns = time.time_ns()
        batch_peak_mb = (max(job.peak_bytes_bb, peak_bytes)) / (1024 ** 2)
        pairs = []
        for i, request in zip(job.indices, job.requests):
            payload = {
                "output": outs[i],
                "start_time_ns": job.backbone_start_ns,
                "end_time_ns": end_ns,
                "proc_time_ns": job.proc_time_ns,
                "swap_time_ns": swaps[i],
                "decoder_time_ns": decs[i],
                "gpu_alloc_peak_mb": batch_peak_mb,
            }
            pairs.append((request.future, payload))
        loop = self._worker_loop_ref
        if loop is not None:
            loop.call_soon_threadsafe(self._resolve_batch, pairs)

    @staticmethod
    def _resolve_batch(pairs):
        for future, payload in pairs:
            if not future.done():
                future.set_result(payload)

    @staticmethod
    def _set_result_if_pending(future, payload):
        if not future.done():
            future.set_result(payload)

    @staticmethod
    def _fail_future_if_pending(future, exc):
        if not future.done():
            future.set_exception(exc)
