"""isolated_app.py — IsolatedRuntimeApplication for process-isolation mode.

Drop-in replacement for EdgeRuntimeApplication. Same public interface:
    start(bootstrap_json), stop(), infer(request), control(command, payload_json)

Queue topology:
    raw_q[task]            : gRPC process → task_proc  (TaskInferMsg / TaskControlMsg)
    task_sched_queues[task]: task_proc → backbone_proc (PreprocessedMsg, one queue per task)
    ctrl_q                 : isolated_app → backbone_proc (ControlMsg / SentinelMsg)
    feat_q[task]           : backbone_proc → task_proc (feat result dicts)
    result_q               : ALL task_procs → gRPC process (inference result dicts)

Process roles:
    task_proc[task] — preprocess raw tensor → SharedMemory → task_sched_queues[task]
                      run decoder on features from feat_q → result_q
    backbone_proc   — polls each task_sched_queues[task] independently
                      feeds TenantQueues, applies FifoPolicy/RoundRobinPolicy
                      backbone forward, fans features to feat_q[task]
"""

import asyncio
import json
import threading
from multiprocessing import Queue

import numpy as np

from device.backbone_proc import ControlMsg, SentinelMsg, start_backbone_process
from device.task_proc import TaskControlMsg, TaskInferMsg, start_task_process


class IsolatedRuntimeApplication:

    def __init__(self, config):
        self.config = config
        self._device = _get_device()
        self._scheduler_policy = getattr(config, "scheduler_policy", "fifo")

        # ctrl_q: isolated_app → backbone_proc (control messages)
        self._ctrl_q: Queue = Queue()
        # ctrl_reply_q: backbone_proc → isolated_app (control replies)
        self._bb_ctrl_reply_q: Queue = Queue()
        # Single result_q: all task_procs → gRPC result reader thread
        self._result_q: Queue = Queue()

        # Per-task queues — populated at deployment time
        self._raw_queues: dict[str, Queue] = {}         # gRPC → task_proc
        self._task_ctrl_reply_queues: dict[str, Queue] = {}  # task_proc → isolated_app
        self._task_sched_queues: dict[str, Queue] = {}  # task_proc → backbone_proc (one per task)
        self._feat_queues: dict[str, Queue] = {}        # backbone_proc → task_proc

        # Process handles
        self._backbone_proc = None
        self._task_procs: dict[str, object] = {}

        # asyncio future registry: req_id → Future
        self._futures: dict[int, asyncio.Future] = {}
        self._futures_lock = threading.Lock()

        # Sequence counter for control message seq matching
        self._ctrl_seq = 0
        self._ctrl_seq_lock = threading.Lock()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader_thread: threading.Thread | None = None
        self._stopped = False
        self._current_backbone: str | None = None
        self._model_category: str = "tsfm"  # updated from backbone_proc reply

    def _next_seq(self) -> int:
        with self._ctrl_seq_lock:
            self._ctrl_seq += 1
            return self._ctrl_seq

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, bootstrap_json: str | None = None):
        self._loop = asyncio.get_running_loop()

        self._reader_thread = threading.Thread(
            target=self._result_reader_loop, daemon=True, name="result_reader"
        )
        self._reader_thread.start()

        if bootstrap_json:
            payload = json.loads(bootstrap_json)
            print(
                f"[IsolatedApp] Bootstrapping backbone={payload['backbone']} "
                f"decoders={len(payload.get('decoders', []))}"
            )
            await self._deploy(payload["backbone"], payload.get("decoders", []))

        print("[IsolatedApp] Started")

    async def stop(self):
        print("[IsolatedApp] Stopping")
        self._stopped = True

        # Stop each task process
        for task, q in self._raw_queues.items():
            q.put(TaskControlMsg(type="stop"))
        for proc in self._task_procs.values():
            proc.join(timeout=5)

        # Stop backbone process
        if self._backbone_proc is not None:
            self._ctrl_q.put(SentinelMsg())
            self._backbone_proc.join(timeout=10)

        # Stop result reader thread
        self._result_q.put({"req_id": -1, "__stop__": True})
        if self._reader_thread:
            self._reader_thread.join(timeout=5)

        print("[IsolatedApp] Stopped")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def infer(self, request):
        task = request.task
        if task not in self._raw_queues:
            raise RuntimeError(f"task_not_deployed:{task}")

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        with self._futures_lock:
            self._futures[request.req_id] = future

        x_array = _decode_tensor_to_numpy(request.x)
        mask_array = (
            _decode_tensor_to_numpy(request.mask)
            if request.HasField("mask")
            else None
        )

        import time
        self._raw_queues[task].put(TaskInferMsg(
            req_id=request.req_id,
            x_bytes=x_array.tobytes(),
            x_shape=x_array.shape,
            x_dtype=str(x_array.dtype),
            mask_bytes=mask_array.tobytes() if mask_array is not None else b"",
            mask_shape=mask_array.shape if mask_array is not None else (),
            mask_dtype=str(mask_array.dtype) if mask_array is not None else "float32",
            enqueued_at=time.time(),
        ))
        return await future

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    async def control(self, command: str, payload_json: str) -> dict:
        payload = json.loads(payload_json) if payload_json else {}
        status = "ok"
        logger_summary = "no_logger"
        try:
            if command == "load":
                backbone = payload["backbone"]
                decoders = payload.get("decoders", [])
                logger_summary = await self._deploy(backbone, decoders)
                status = f"loaded_{backbone}"

            elif command == "swap_backbone":
                backbone = payload["backbone"]
                decoders = payload.get("decoders", [])
                seq = self._next_seq()
                self._ctrl_q.put(ControlMsg(
                    command="swap_backbone",
                    payload_json=json.dumps({"backbone": backbone}),
                    seq=seq,
                ))
                resp = await asyncio.to_thread(
                    _wait_for_seq, self._bb_ctrl_reply_q, seq, 60.0
                )
                logger_summary = resp.get("logger_summary", "")
                await self._reload_decoders(backbone, decoders)
                status = f"swapped_{backbone}"

            elif command == "add_decoder":
                for dec in payload.get("decoders", []):
                    task = dec["task"]
                    if task not in self._raw_queues:
                        await self._spawn_task_proc(task, dec, self._current_backbone)
                status = f"added_{len(payload.get('decoders', []))}_decoders"

            else:
                raise ValueError(f"unknown_command_{command}")

        except Exception as exc:
            status = f"error_{exc}"
            print(f"[IsolatedApp] Control error: {exc}")

        return {"status": status, "logger_summary": logger_summary}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _deploy(self, backbone: str, decoder_specs: list) -> str:
        """Returns logger_summary string with backbone + decoder memory appended."""
        self._current_backbone = backbone

        # Create per-task queues before spawning any process
        for dec in decoder_specs:
            task = dec["task"]
            if task not in self._raw_queues:
                self._raw_queues[task] = Queue()
                self._task_ctrl_reply_queues[task] = Queue()
                self._task_sched_queues[task] = Queue()
                self._feat_queues[task] = Queue()

        # Spawn backbone process if not already running
        if self._backbone_proc is None or not self._backbone_proc.is_alive():
            self._backbone_proc = start_backbone_process(
                task_sched_queues=self._task_sched_queues,
                ctrl_q=self._ctrl_q,
                ctrl_reply_q=self._bb_ctrl_reply_q,
                feat_queues=self._feat_queues,
                max_batch_size=self.config.max_batch_size,
                max_batch_wait_ms=self.config.max_batch_wait_ms,
                device=self._device,
                scheduler_policy=self._scheduler_policy,
            )

        # Tell backbone proc to load the backbone weights
        seq = self._next_seq()
        self._ctrl_q.put(ControlMsg(
            command="load",
            payload_json=json.dumps({"backbone": backbone}),
            seq=seq,
        ))
        resp = await asyncio.to_thread(
            _wait_for_seq, self._bb_ctrl_reply_q, seq, 120.0
        )
        logger_summary = resp.get("logger_summary", "")
        self._model_category = resp.get("model_category", "tsfm")
        print(f"[IsolatedApp] Backbone load: {resp['status']}  model_category={self._model_category}")

        # Spawn one task process per decoder, accumulate decoder memory
        total_decoder_mem_mb = 0.0
        for dec in decoder_specs:
            task = dec["task"]
            if task not in self._task_procs or not self._task_procs[task].is_alive():
                total_decoder_mem_mb += await self._spawn_task_proc(task, dec, backbone)

        # Append decoder memory into logger_summary so run.py can parse it
        try:
            summary_dict = eval(logger_summary) if logger_summary else {}
        except Exception:
            summary_dict = {}
        summary_dict["decoder_processes"] = {"gpu peak": total_decoder_mem_mb,
                                              "gpu reserved": total_decoder_mem_mb}
        return str(summary_dict)

    async def _spawn_task_proc(self, task: str, dec_spec: dict, backbone: str):
        if task not in self._raw_queues:
            self._raw_queues[task] = Queue()
            self._task_ctrl_reply_queues[task] = Queue()
            self._task_sched_queues[task] = Queue()
            self._feat_queues[task] = Queue()

        proc = start_task_process(
            task=task,
            raw_q=self._raw_queues[task],
            ctrl_reply_q=self._task_ctrl_reply_queues[task],
            task_sched_q=self._task_sched_queues[task],
            feat_q=self._feat_queues[task],
            result_q=self._result_q,
            device=self._device,
        )
        self._task_procs[task] = proc

        # Tell task proc to load its decoder
        seq = self._next_seq()
        self._raw_queues[task].put(TaskControlMsg(
            command="load_decoder",
            payload_json=json.dumps({**dec_spec, "backbone": backbone,
                                     "model_category": self._model_category}),
            seq=seq,
        ))
        resp = await asyncio.to_thread(
            _wait_for_seq, self._task_ctrl_reply_queues[task], seq, 60.0
        )
        mem_mb = resp.get("gpu_decoder_mem_mb", 0.0)
        print(f"[IsolatedApp] Task proc {task} decoder: {resp['status']}  gpu_alloc={mem_mb:.1f} MB")
        return mem_mb

    async def _reload_decoders(self, backbone: str, decoder_specs: list):
        for dec in decoder_specs:
            task = dec["task"]
            if task in self._raw_queues:
                seq = self._next_seq()
                self._raw_queues[task].put(TaskControlMsg(
                    command="load_decoder",
                    payload_json=json.dumps({**dec, "backbone": backbone, "model_category": self._model_category}),
                    seq=seq,
                ))
                resp = await asyncio.to_thread(
                    _wait_for_seq, self._task_ctrl_reply_queues[task], seq, 60.0
                )
                print(f"[IsolatedApp] Task proc {task} decoder reload: {resp['status']}")

    def _result_reader_loop(self):
        """Background thread: result_q → asyncio futures."""
        while not self._stopped:
            try:
                result = self._result_q.get(timeout=1.0)
            except Exception:
                continue
            if result.get("__stop__"):
                break
            req_id = result["req_id"]
            with self._futures_lock:
                future = self._futures.pop(req_id, None)
            if future is None:
                print(f"[IsolatedApp] No future for req_id={req_id}")
                continue
            if "error" in result:
                self._loop.call_soon_threadsafe(
                    _set_exception_if_pending, future, RuntimeError(result["error"])
                )
            else:
                self._loop.call_soon_threadsafe(
                    _set_result_if_pending, future, result
                )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_device() -> str:
    import os
    return os.environ.get("CUDA_DEVICE", "cuda:0")


def _decode_tensor_to_numpy(payload) -> np.ndarray:
    dtype = np.dtype(payload.dtype)
    return np.frombuffer(payload.data, dtype=dtype).reshape(tuple(payload.shape))


def _set_result_if_pending(future: asyncio.Future, payload):
    if not future.done():
        future.set_result(payload)


def _set_exception_if_pending(future: asyncio.Future, exc: Exception):
    if not future.done():
        future.set_exception(exc)


def _wait_for_seq(reply_q: Queue, seq: int, timeout: float) -> dict:
    """Block until a reply with matching seq arrives. Stash out-of-order replies."""
    import time
    stash: list = []
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for seq={seq}")
        try:
            resp = reply_q.get(timeout=min(remaining, 1.0))
        except Exception:
            continue
        if resp.get("seq") == seq:
            # Put stashed replies back for other waiters
            for r in stash:
                reply_q.put(r)
            return resp
        stash.append(resp)
