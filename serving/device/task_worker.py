import queue as pyqueue
import threading
from abc import ABC, abstractmethod
from typing import Callable


class TaskWorker(ABC):
    """A per-task pipeline worker.

    A worker runs the post-backbone per-task pipeline steps (today: decoder
    only; later: encoder + decoder if/when per-task encoders land) for jobs
    submitted to it. All steps for a given task share the same thread and the
    same CUDA stream, so cross-step dependencies inside a task are implicit
    and there is no per-stage thread/stream proliferation.

    Two implementations:
      - ThreadedTaskWorker: each task gets its own worker thread + own CUDA
        stream so kernels from different tasks can overlap and per-stream
        syncs don't block other tasks.
      - InlineTaskWorker: runs steps synchronously on the caller's thread,
        no own thread, no own queue, no own stream.
    """

    @abstractmethod
    def submit(self, task_name: str, job) -> None:
        """Hand a job off to the worker for the given task."""

    @abstractmethod
    def shutdown(self) -> None:
        """Stop any worker threads and release per-task resources."""


class ThreadedTaskWorker(TaskWorker):
    """Per-task worker thread + per-task CUDA stream.

    `run_fn(task_name, job, stream) -> None` runs the task's pipeline steps
    on the task's stream. Today there's a single step (decoder); later this
    same function runs `encoder -> decoder` on the same stream.
    """

    def __init__(
        self,
        worker_name: str,
        device,
        run_fn: Callable[[str, object, object], None],
    ):
        self._worker_name = worker_name
        self._device = device
        self._run_fn = run_fn
        self._threads: dict[str, threading.Thread] = {}
        self._queues: dict[str, pyqueue.Queue] = {}
        self._streams: dict[str, object] = {}
        self._lock = threading.Lock()

    def submit(self, task_name: str, job) -> None:
        q = self._get_queue(task_name)
        q.put(job)

    def shutdown(self) -> None:
        with self._lock:
            for _, q in self._queues.items():
                q.put(None)
            for _, t in self._threads.items():
                t.join(timeout=5)

    def _get_queue(self, task_name: str) -> pyqueue.Queue:
        with self._lock:
            q = self._queues.get(task_name)
            if q is not None:
                return q
            q = pyqueue.Queue()
            self._queues[task_name] = q

            # Create a dedicated CUDA stream for this task's worker.
            stream = None
            try:
                import torch
                if str(self._device).startswith("cuda"):
                    stream = torch.cuda.Stream(device=self._device)
            except Exception as exc:
                print(f"[{self._worker_name}-{task_name}] stream creation failed: {exc}")
            self._streams[task_name] = stream

            t = threading.Thread(
                target=self._worker_loop,
                args=(task_name, q, stream),
                name=f"{self._worker_name}-{task_name}",
                daemon=True,
            )
            self._threads[task_name] = t
            t.start()
            print(f"[{self._worker_name}] Started worker thread for task={task_name} "
                  f"(own_stream={stream is not None})")
            return q

    def _worker_loop(self, task_name: str, q: pyqueue.Queue, stream):
        # Pin this thread's current CUDA stream to its OWN stream so workers
        # from different tasks can overlap on the GPU and per-stream syncs
        # don't stall each other.
        try:
            import torch
            if stream is not None:
                torch.cuda.set_stream(stream)
        except Exception as exc:
            print(f"[{self._worker_name}-{task_name}] set_stream failed: {exc}")

        while True:
            job = q.get()
            if job is None:
                return
            try:
                self._run_fn(task_name, job, stream)
            except Exception as exc:
                import traceback
                print(f"[{self._worker_name}-{task_name}] ERROR: {exc}")
                traceback.print_exc()


class InlineTaskWorker(TaskWorker):
    """Runs the pipeline steps synchronously on the caller's thread.

    No own thread, no own queue, no own CUDA stream. Use when batching is
    desired but per-task threading/streams are not.
    """

    def __init__(self, run_fn: Callable[[str, object, object], None]):
        self._run_fn = run_fn

    def submit(self, task_name: str, job) -> None:
        try:
            self._run_fn(task_name, job, None)
        except Exception as exc:
            import traceback
            print(f"[InlineTaskWorker] ERROR task={task_name}: {exc}")
            traceback.print_exc()

    def shutdown(self) -> None:
        return
