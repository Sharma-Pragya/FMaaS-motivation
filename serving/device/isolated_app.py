"""isolated_app.py — IsolatedRuntimeApplication for process-isolation mode.

Drop-in replacement for EdgeRuntimeApplication. Same public interface:
    start(bootstrap_json), stop(), infer(request), control(command, payload_json)

This process never touches CUDA. At start it creates every queue (spawn
context, torch.multiprocessing, so CUDA tensors can cross them) and starts the
backbone process; it starts one task process per task as tasks are deployed.

Lanes: a queue can only be handed to a process when that process starts, so
config.max_tasks lanes are created up front and the backbone gets them all.
Each task process takes the next unused lane; a lane is never reused (a
process that dies can leave a lane's queue locks held), so max_tasks bounds
the task processes ever started, restarts included.

Queue topology:
    req_q[lane]         : gRPC process → task_proc   (TaskInferMsg / TaskControlMsg)
    task_reply_q[lane]  : task_proc → gRPC process   (control replies)
    sched_q[lane]       : task_proc → backbone_proc  (alloc, filled slots)
    done_q[lane]        : backbone_proc → task_proc  (pool, feat, done, error)
    dying[lane]         : Event, set by a failing task_proc
    ctrl_q / bb_reply_q : gRPC process ↔ backbone_proc (ControlMsg / SentinelMsg)
    result_q            : ALL task_procs → gRPC process (inference result dicts)

GPU memory crossing processes (design 3): each lane's pool is allocated by the
backbone process and lent to that lane's task process only (backbone_proc.py).
"""

import asyncio
import json
import threading
import time

import numpy as np
import torch.multiprocessing as tmp

from device.backbone_proc import ControlMsg, SentinelMsg, start_backbone_process
from device.task_proc import TaskControlMsg, TaskInferMsg, start_task_process


class IsolatedRuntimeApplication:

    def __init__(self, config):
        self.config = config
        self._device = _get_device()
        self._ctx = tmp.get_context("spawn")
        ctx, k = self._ctx, config.max_tasks

        # Shared queues
        self._ctrl_q = ctx.Queue()
        self._bb_ctrl_reply_q = ctx.Queue()
        self._result_q = ctx.Queue()
        self._run_stop = ctx.Event()

        # Lanes, all created before any process starts
        self._req_qs = [ctx.Queue() for _ in range(k)]
        self._task_reply_qs = [ctx.Queue() for _ in range(k)]
        self._sched_qs = [ctx.Queue() for _ in range(k)]
        self._done_qs = [ctx.Queue() for _ in range(k)]
        self._dyings = [ctx.Event() for _ in range(k)]
        self._next_lane = 0

        # Process handles
        self._backbone_proc = None
        self._task_procs: dict[int, object] = {}   # lane → process
        self._lane_task: dict[int, str] = {}       # lane → task
        self._task_lane: dict[str, int] = {}       # task → its current lane
        self._dead_lanes: set[int] = set()
        self._down_tasks: set[str] = set()
        self._backbone_down = False

        # asyncio future registry: internal request id → (Future, task).
        # Keyed by our own id, not the client's req_id, which need not be unique.
        self._futures: dict[int, tuple] = {}
        self._next_rid = 0
        self._futures_lock = threading.Lock()

        # Open requests per task (handed to its process, no result yet), and
        # the most a task may have open: its GPU slots plus its CPU backlog.
        # submit() fails a request with queue_full beyond this, before copying it.
        self._open: dict[str, int] = {}
        self._open_cap = ((config.slots_per_task or 2 * config.max_batch_size)
                          + config.task_backlog_cap)

        # Sequence counter for control message seq matching
        self._ctrl_seq = 0
        self._ctrl_seq_lock = threading.Lock()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None
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

        self._backbone_proc = start_backbone_process(
            self._ctx, self.config, self._device, self._sched_qs, self._done_qs,
            self._dyings, self._ctrl_q, self._bb_ctrl_reply_q, self._run_stop,
        )

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="proc_monitor"
        )
        self._monitor_thread.start()

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
        self._run_stop.set()

        for lane in self._task_procs:
            self._req_qs[lane].put(TaskControlMsg(type="stop"))
        self._ctrl_q.put(SentinelMsg())

        procs = list(self._task_procs.values()) + [self._backbone_proc]
        deadline = time.time() + 10.0
        for proc in procs:
            if proc is not None:
                proc.join(timeout=max(0.0, deadline - time.time()))
        for proc in procs:
            if proc is not None and proc.is_alive():
                print(f"[IsolatedApp] {proc.name} did not stop; killing")
                proc.kill()
                proc.join(timeout=2)
        # A dead process leaves messages in its queues that nobody will read;
        # without this, exit waits forever to flush them.
        for q in self._req_qs + [self._ctrl_q]:
            q.cancel_join_thread()

        # Stop result reader thread
        self._result_q.put({"req_id": -1, "__stop__": True})
        if self._reader_thread:
            self._reader_thread.join(timeout=5)

        print("[IsolatedApp] Stopped")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def infer(self, request):
        return await (await self.submit(request))

    async def submit(self, request) -> asyncio.Future:
        """Hand one request to its task process and return the future its
        result lands on. The input is copied into the message for the task
        process, and the returned future holds no reference to request, so
        InferStream can drop request once this returns (as
        EdgeRuntimeApplication.submit)."""
        task = request.task
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        with self._futures_lock:
            if self._backbone_down:
                raise RuntimeError("backbone_down")
            if task in self._down_tasks:
                raise RuntimeError(f"task_down:{task}")
            if task not in self._task_lane:
                raise RuntimeError(f"task_not_deployed:{task}")
            # Per-task cap, checked here, before the input is copied: the
            # queue to the task process never blocks, so a task sending
            # faster than its process reads would otherwise pile messages up
            # in this process's RAM (the task process's --task-backlog-cap
            # only sees a message once it has been read).
            if self._open.get(task, 0) >= self._open_cap:
                raise RuntimeError("queue_full")
            self._open[task] = self._open.get(task, 0) + 1
            lane = self._task_lane[task]
            rid = self._next_rid
            self._next_rid += 1
            self._futures[rid] = (future, task)
        future.add_done_callback(lambda _f, t=task: self._close_open(t))

        x_array = _decode_tensor_to_numpy(request.x)
        mask_array = (
            _decode_tensor_to_numpy(request.mask)
            if request.HasField("mask")
            else None
        )

        self._req_qs[lane].put(TaskInferMsg(
            req_id=rid,
            x_bytes=x_array.tobytes(),
            x_shape=x_array.shape,
            x_dtype=str(x_array.dtype),
            mask_bytes=mask_array.tobytes() if mask_array is not None else b"",
            mask_shape=mask_array.shape if mask_array is not None else (),
            mask_dtype=str(mask_array.dtype) if mask_array is not None else "float32",
            enqueued_at=time.time(),
        ))
        return future

    def _close_open(self, task: str) -> None:
        """A request of task got its result (or failed): it is no longer open."""
        with self._futures_lock:
            self._open[task] = max(0, self._open.get(task, 0) - 1)

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
                resp = await self._backbone_control("swap_backbone", {"backbone": backbone}, 60.0)
                logger_summary = resp.get("logger_summary", "")
                self._current_backbone = backbone
                await self._reload_decoders(backbone, decoders)
                status = f"swapped_{backbone}"

            elif command == "add_decoder":
                for dec in payload.get("decoders", []):
                    task = dec["task"]
                    if not self._task_alive(task):
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

    async def _backbone_control(self, command: str, payload: dict, timeout: float) -> dict:
        seq = self._next_seq()
        self._ctrl_q.put(ControlMsg(command=command, payload_json=json.dumps(payload), seq=seq))
        return await asyncio.to_thread(_wait_for_seq, self._bb_ctrl_reply_q, seq, timeout)

    def _task_alive(self, task: str) -> bool:
        lane = self._task_lane.get(task)
        return lane is not None and lane not in self._dead_lanes

    async def _deploy(self, backbone: str, decoder_specs: list) -> str:
        """Returns logger_summary string with backbone + decoder memory appended."""
        self._current_backbone = backbone

        # Tell backbone proc to load the backbone weights
        resp = await self._backbone_control("load", {"backbone": backbone}, 120.0)
        logger_summary = resp.get("logger_summary", "")
        self._model_category = resp.get("model_category", "tsfm")
        print(f"[IsolatedApp] Backbone load: {resp['status']}  model_category={self._model_category}")

        # Spawn one task process per decoder, accumulate decoder memory
        total_decoder_mem_mb = 0.0
        for dec in decoder_specs:
            task = dec["task"]
            if not self._task_alive(task):
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
        if self._next_lane >= len(self._req_qs):
            raise RuntimeError(f"no_free_lane (max_tasks={len(self._req_qs)})")
        lane = self._next_lane
        self._next_lane += 1

        await self._backbone_control("attach", {"lane": lane, "task": task}, 30.0)
        proc = start_task_process(
            self._ctx, task, lane, self._device, self.config.task_backlog_cap,
            self._req_qs[lane], self._task_reply_qs[lane], self._sched_qs[lane],
            self._done_qs[lane], self._result_q, self._dyings[lane],
            self._run_stop,
        )
        self._task_procs[lane] = proc
        self._lane_task[lane] = task

        # Tell task proc to load its decoder
        seq = self._next_seq()
        self._req_qs[lane].put(TaskControlMsg(
            command="load_decoder",
            payload_json=json.dumps({**dec_spec, "backbone": backbone,
                                     "model_category": self._model_category}),
            seq=seq,
        ))
        resp = await asyncio.to_thread(
            _wait_for_seq, self._task_reply_qs[lane], seq, 60.0
        )
        mem_mb = resp.get("gpu_decoder_mem_mb", 0.0)
        print(f"[IsolatedApp] Task proc {task} (lane {lane}) decoder: {resp['status']}  "
              f"gpu_alloc={mem_mb:.1f} MB")

        # Requests are accepted only once the decoder is loaded.
        with self._futures_lock:
            self._task_lane[task] = lane
            self._down_tasks.discard(task)
        return mem_mb

    async def _reload_decoders(self, backbone: str, decoder_specs: list):
        for dec in decoder_specs:
            task = dec["task"]
            if self._task_alive(task):
                lane = self._task_lane[task]
                seq = self._next_seq()
                self._req_qs[lane].put(TaskControlMsg(
                    command="load_decoder",
                    payload_json=json.dumps({**dec, "backbone": backbone, "model_category": self._model_category}),
                    seq=seq,
                ))
                resp = await asyncio.to_thread(
                    _wait_for_seq, self._task_reply_qs[lane], seq, 60.0
                )
                print(f"[IsolatedApp] Task proc {task} decoder reload: {resp['status']}")

    def _fail_futures(self, task: str | None, why: str):
        """Fail pending futures of task (None: every task). Called with
        _futures_lock held."""
        for req_id, (future, t) in list(self._futures.items()):
            if task is None or t == task:
                del self._futures[req_id]
                self._loop.call_soon_threadsafe(
                    _set_exception_if_pending, future, RuntimeError(why)
                )

    def _monitor_loop(self):
        """Background thread: a task process that exits is detached from the
        backbone and its task's pending requests fail; the backbone process
        exiting fails every pending request."""
        while not self._stopped:
            time.sleep(0.5)
            if self._stopped:
                return
            for lane, proc in list(self._task_procs.items()):
                if lane in self._dead_lanes or proc.exitcode is None:
                    continue
                self._dead_lanes.add(lane)
                task = self._lane_task[lane]
                print(f"[IsolatedApp] Task proc {task} (lane {lane}) exited "
                      f"code={proc.exitcode}")
                self._ctrl_q.put(ControlMsg(command="detach",
                                            payload_json=json.dumps({"lane": lane})))
                with self._futures_lock:
                    if self._task_lane.get(task) == lane:
                        self._down_tasks.add(task)
                        self._fail_futures(task, f"task_down:{task}")
            if (not self._backbone_down and self._backbone_proc is not None
                    and self._backbone_proc.exitcode is not None):
                print(f"[IsolatedApp] Backbone proc exited code={self._backbone_proc.exitcode}")
                with self._futures_lock:
                    self._backbone_down = True
                    self._fail_futures(None, "backbone_down")

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
                entry = self._futures.pop(req_id, None)
            if entry is None:
                print(f"[IsolatedApp] No future for req_id={req_id}")
                continue
            future = entry[0]
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


def _wait_for_seq(reply_q, seq: int, timeout: float) -> dict:
    """Block until a reply with matching seq arrives. Stash out-of-order replies."""
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
