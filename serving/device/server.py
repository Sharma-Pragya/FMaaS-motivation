import asyncio
import json
import logging
import os
import signal
import time
from dataclasses import dataclass

import grpc
import numpy as np

from device.batcher import DeviceBatcher
from device.proto import edge_runtime_pb2, edge_runtime_pb2_grpc
from device.runtime import PyTorchRuntime, VLLMRuntime
from device.scheduler import FifoPolicy, RequestEnvelope, RoundRobinPolicy, WFQPolicy, STFQPolicy, TokenBucketPolicy, SABAPolicy, DeadlineSplitPolicy
from device.vllm_admission import VLLMAdmissionScheduler


LOGGER = logging.getLogger(__name__)


def _decode_tensor(payload: edge_runtime_pb2.TensorPayload | None) -> np.ndarray | None:
    if payload is None or not payload.shape:
        return None
    dtype = np.dtype(payload.dtype)
    array = np.frombuffer(payload.data, dtype=dtype)
    return array.reshape(tuple(payload.shape))


def _encode_output(output: np.ndarray) -> tuple[list[float], list[int]]:
    # Normalize to ndarray first; callers may pass Python lists.
    array = np.asarray(output)
    # VLM outputs are string/object arrays — return empty numeric output;
    # text travels via InferResponse.text_output.
    if array.dtype == object:
        return [], []
    array = array.astype(np.float32, copy=False)
    return array.reshape(-1).tolist(), list(array.shape)


@dataclass
class RuntimeServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 1
    max_batch_wait_ms: float = 1.0
    queue_capacity: int = 1024
    stop_grace_s: float = 5.0
    runtime_type: str = "pytorch"
    scheduler_policy: str = "fifo"
    isolation_mode: str = "shared"   # "shared" | "process" | "none"
    task_rates: dict = None          # {task: rps} — used by WFQ/TokenBucket
    gpu_memory_utilization: float = 0.9  # vLLM GPU memory fraction (0.0–1.0)
    max_model_len: int = None            # vLLM max sequence length (None = vLLM default)
    max_new_tokens: int = None           # vLLM SamplingParams.max_tokens (None = backbone default)
    max_num_seqs: int = None             # vLLM max concurrent sequences in the batch (None = vLLM default 256)
    ignore_eos: bool = False             # vLLM SamplingParams.ignore_eos (force exactly max_tokens)
    tpc_mode: str = "none"               # "none" | "libsmctrl" | "green"
    tpc_partition: list = None           # list of TPC IDs to pin this server to
    tpc_share: str = None                # "i/k": the i-th of k equal TPC ranges of the
                                         # device this process sees (libsmctrl), e.g. one
                                         # of k servers sharing a MIG slice
    worker_mode: str = "threaded"        # "threaded" | "inline" — per-task pipeline worker mode
    verbose_batch_logs: bool = False      # high-frequency per-batch request logging
    max_tasks: int = 16                  # isolation_mode=process: task lanes created at start
    slots_per_task: int = None           # isolation_mode=process: GPU slots per vFM pool (None = 2 x max_batch_size)
    task_mem_mb: str = None              # isolation_mode=process: vFM memory budget, "MB" or
                                         # "task=MB,...,*=MB"; sets the pool's slots (overrides
                                         # slots_per_task). Pool only for now.
    task_backlog_cap: int = 10000        # isolation_mode=process: requests a vFM may hold waiting on the CPU
    staging: str = "host"                # shared mode: "host" = inputs wait on the CPU, copied at batch time;
                                         # "gpu" = copied to the GPU on arrival and wait there (like process mode)


class EdgeRuntimeApplication:
    """Owns the shared-model runtime, queueing, and gRPC entry points."""

    def __init__(self, config: RuntimeServerConfig):
        self.config = config
        self.runtime_type = config.runtime_type
        self.isolation_mode = config.isolation_mode
        if config.isolation_mode == "process":
            raise ValueError("Use IsolatedRuntimeApplication for isolation_mode=process")
        # Admission scheduler in front of vLLM (STFQ/WFQ). For runtime_type=vllm
        # we never construct a DeviceBatcher; the admission scheduler — when
        # active — reuses STFQPolicy to order requests before they enter the
        # AsyncLLMEngine. vLLM keeps doing its own continuous batching across
        # the in-flight set bounded by max_batch_size.
        self.admission: VLLMAdmissionScheduler | None = None
        # TPC pinning applies to both runtimes: it masks a CUDA stream to a
        # subset of SMs/TPCs before any model code runs on it. Safe to call
        # here for vllm too now that CUDA_VISIBLE_DEVICES is fixed up front
        # in main.py's pre-parser, before this constructor's first CUDA call.
        cuda_stream = self._setup_tpc(config)
        if config.runtime_type == "vllm":
            self.runtime = VLLMRuntime(cuda_stream=cuda_stream)
            self.batcher = None
            task_rates = config.task_rates or {}
            if config.scheduler_policy in ("stfq", "wfq") and task_rates:
                weights = {t: 1.0/r if r > 0 else 1.0 for t, r in task_rates.items()}
                self.admission = VLLMAdmissionScheduler(
                    runtime=self.runtime,
                    weights=weights,
                    max_in_flight=max(1, config.max_batch_size),
                )
        else:
            self.runtime = PyTorchRuntime(cuda_stream=cuda_stream)
            if config.isolation_mode == "none":
                # No batcher — requests go directly to runtime.run_batch()
                self.batcher = None
            else:
                policy = self._make_policy(config)
                self.batcher = DeviceBatcher(
                    runtime=self.runtime,
                    max_batch_size=config.max_batch_size,
                    max_batch_wait_ms=config.max_batch_wait_ms,
                    queue_capacity=config.queue_capacity,
                    policy=policy,
                    worker_mode=config.worker_mode,
                    verbose_batch_logs=config.verbose_batch_logs,
                )
        self._batch_task: asyncio.Task | None = None
        self._green_partition = None  # held for cleanup
        # staging=gpu: inputs are copied to the GPU when they arrive, on a
        # separate copy stream so the copy never queues behind backbone work.
        self._stage_device = None
        self._copy_stream = None
        if config.staging == "gpu" and config.runtime_type == "pytorch" and self.batcher is not None:
            import torch
            from device.config import DEVICE
            if str(DEVICE).startswith("cuda"):
                self._stage_device = DEVICE
                self._copy_stream = torch.cuda.Stream(device=DEVICE)
                print(f"[Device] staging=gpu: inputs copied to {DEVICE} on arrival")

    def _stage(self, array: np.ndarray | None):
        """staging=gpu: copy one input to the GPU and wait for the copy to land
        before the request is queued. staging=host: return it unchanged."""
        if array is None or self._copy_stream is None:
            return array
        import torch
        with torch.cuda.stream(self._copy_stream):
            t = torch.from_numpy(array).to(self._stage_device, non_blocking=False)
        self._copy_stream.synchronize()
        return t

    @staticmethod
    def _make_policy(config: RuntimeServerConfig):
        """Batch scheduling policy from config. Also used by the backbone
        process in isolation_mode=process (backbone_proc.py)."""
        task_rates = config.task_rates or {}
        if config.scheduler_policy == "round_robin":
            policy = RoundRobinPolicy()
        elif config.scheduler_policy == "stfq":
            # STFQ weight = 1/rps so low-RPS victim gets high weight
            # (slow VFT advance = stays near front = served promptly)
            inv_weights = {t: 1.0/r for t, r in task_rates.items()} if task_rates else None
            policy = STFQPolicy(weights=inv_weights)
        elif config.scheduler_policy == "wfq":
            # WFQ weight = 1/rps so low-RPS victim gets high weight
            # (slow VFT advance = higher priority = served promptly)
            inv_weights = {t: 1.0/r for t, r in task_rates.items()} if task_rates else None
            policy = WFQPolicy(weights=inv_weights)
        elif config.scheduler_policy == "token_bucket":
            # TokenBucket: accrue at 1/rps so victim accrues faster
            # → victim always has more credit → served first when present
            policy = TokenBucketPolicy()
            for task, rate in task_rates.items():
                policy.set_rate(task, 1.0 / rate if rate > 0 else 1.0)
        elif config.scheduler_policy == "saba":
            # SABA: allocate batch slots proportional to arrival rate
            policy = SABAPolicy(rates=task_rates)
        elif config.scheduler_policy == "deadline_split":
            # DeadlineSplit: fire small dedicated batches for low-rate
            # protected tasks on a per-task deadline timer; high-rate
            # tasks get full batches in between.
            policy = DeadlineSplitPolicy(rates=task_rates)
        else:
            policy = FifoPolicy()
        return policy

    @staticmethod
    def _setup_tpc(config: RuntimeServerConfig):
        """Create and TPC-pin a CUDA stream if configured. Returns stream or None."""
        if config.tpc_mode == "none" or not (config.tpc_partition or config.tpc_share):
            return None

        import ctypes
        import torch
        from pathlib import Path

        cuda_device = os.environ.get("CUDA_DEVICE", "cuda:0")
        device_id = int(cuda_device.split(":")[-1]) if ":" in cuda_device else 0
        sm_count = torch.cuda.get_device_properties(cuda_device).multi_processor_count

        if config.tpc_mode == "libsmctrl":
            TPC_LIB_DIR = Path(os.environ.get(
                "TPC_LIB_DIR",
                "../../TPC_controller/tpc_controller"
            ))
            so_path = TPC_LIB_DIR / "libsmctrl" / "libsmctrl.so"
            if not so_path.exists():
                raise FileNotFoundError(f"libsmctrl.so not found at {so_path}")
            lib = ctypes.CDLL(str(so_path))
            # Use the _ext (128-bit) entry point: this GPU has 71 TPCs, and the
            # plain uint64 libsmctrl_set_stream_mask() unconditionally disables
            # every TPC with ID >= 64 (libsmctrl.c:276-282), which would silently
            # shrink any partition reaching above 63. `unsigned __int128` is
            # passed in a register pair on x86-64 SysV, so declaring it as two
            # consecutive uint64 args (low, high) matches the C ABI.
            lib.libsmctrl_set_stream_mask_ext.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64
            ]
            lib.libsmctrl_set_stream_mask_ext.restype = None
            lib.libsmctrl_get_tpc_info_cuda.argtypes = [
                ctypes.POINTER(ctypes.c_uint32), ctypes.c_int
            ]
            lib.libsmctrl_get_tpc_info_cuda.restype = ctypes.c_int

            num_tpcs = ctypes.c_uint32()
            ret = lib.libsmctrl_get_tpc_info_cuda(ctypes.byref(num_tpcs), device_id)
            total_tpcs = num_tpcs.value if ret == 0 else sm_count // 2
            if config.tpc_share:
                # --tpc-share i/k: TPCs [i*T/k, (i+1)*T/k) of the T this process
                # sees (under MIG, the slice's).
                i, k = (int(v) for v in config.tpc_share.split("/"))
                lo, hi = i * total_tpcs // k, (i + 1) * total_tpcs // k
                config.tpc_partition = list(range(lo, max(hi, lo + 1)))

            stream = torch.cuda.Stream(device=cuda_device)
            enable_bits = 0
            for tid in config.tpc_partition:
                enable_bits |= (1 << tid)
            # A *set* bit means the TPC is disabled, so invert over 128 bits.
            disable_mask = (~enable_bits) & ((1 << 128) - 1)
            lib.libsmctrl_set_stream_mask_ext(
                ctypes.c_void_p(stream.cuda_stream),
                ctypes.c_uint64(disable_mask & 0xFFFFFFFFFFFFFFFF),
                ctypes.c_uint64((disable_mask >> 64) & 0xFFFFFFFFFFFFFFFF),
            )
            print(f"[TPC] libsmctrl: pinned stream to TPCs {config.tpc_partition} "
                  f"(total={total_tpcs})")
            return stream

        elif config.tpc_mode == "green":
            import sys
            tpc_iso_dir = str(Path(__file__).resolve().parents[1] / "experiments" / "tpc_isolation")
            if tpc_iso_dir not in sys.path:
                sys.path.insert(0, tpc_iso_dir)
            from run import GreenContextPartition

            partition = GreenContextPartition(device_id=device_id)
            streams = partition.create_partitions(num_partitions=2)
            # Use partition index: partition 0 for low TPC IDs, 1 for high
            idx = 0 if config.tpc_partition[0] == 0 else 1
            print(f"[TPC] green context: using partition {idx} for TPCs {config.tpc_partition}")
            # Store partition ref so it doesn't get GC'd
            return streams[idx]

        return None

    async def start(self, bootstrap_json: str | None = None):
        if bootstrap_json:
            payload = json.loads(bootstrap_json)
            print(
                f"[Device] Bootstrapping backbone={payload['backbone']} "
                f"decoders={len(payload['decoders'])}"
            )
            model_config = {"gpu_memory_utilization": self.config.gpu_memory_utilization}
            if self.config.max_model_len is not None:
                model_config["max_model_len"] = self.config.max_model_len
            if self.config.max_new_tokens is not None:
                model_config["max_new_tokens"] = self.config.max_new_tokens
            if self.config.max_num_seqs is not None:
                model_config["max_num_seqs"] = self.config.max_num_seqs
            if self.config.ignore_eos:
                model_config["ignore_eos"] = True
            await asyncio.to_thread(
                self.runtime.load,
                payload["backbone"],
                payload.get("decoders", []),
                model_config=model_config,
            ) 
        if self.runtime_type == "pytorch" and self.batcher is not None and self._batch_task is None:
            self._batch_task = asyncio.create_task(self.batcher.run_forever())
        if self.admission is not None:
            await self.admission.start()
        print("[Device] Runtime application started")

    async def stop(self):
        print("[Device] Stopping runtime application")
        if self.admission is not None:
            await self.admission.stop()
        if self.batcher is not None:
            await self.batcher.stop()
        if self._batch_task is not None:
            await self._batch_task
            self._batch_task = None
        print("[Device] Runtime application stopped")

    async def infer(self, request: edge_runtime_pb2.InferRequest):
        # print(f"[Device] Received infer req_id={request.req_id} task={request.task}")
        if self.runtime_type == "vllm":
            prompt = request.question if request.HasField("question") else ""
            # x is the image for VLM backbones (None for text LLMs). The
            # runtime decides how to use it from the backbone's model_category.
            x = _decode_tensor(request.x)
            if self.admission is not None:
                return await self.admission.submit(request.req_id, prompt, request.task, x=x)
            return await self.runtime.infer(request.req_id, prompt, request.task, x=x)
        if self.isolation_mode == "none":
            # Direct path: no queue, call runtime.run_batch() inline
            x        = _decode_tensor(request.x)
            mask     = _decode_tensor(request.mask) if request.HasField("mask") else None
            question = request.question if request.HasField("question") else None
            questions = [question] if question is not None else None
            result = await asyncio.to_thread(
                self.runtime.run_batch, x, [request.task], mask, questions
            )
            return {
                "req_id":          request.req_id,
                "output":          result.outputs[0],
                "start_time_ns":   result.start_time_ns,
                "end_time_ns":     result.end_time_ns,
                "proc_time_ns":    result.proc_time_ns,
                "swap_time_ns":    result.swap_time_ns[0],
                "decoder_time_ns": result.decoder_time_ns[0],
                "gpu_alloc_peak_mb": result.gpu_alloc_peak_mb,
            }
        return await (await self.submit(request))

    async def submit(self, request: edge_runtime_pb2.InferRequest) -> asyncio.Future:
        """Queue one request and return the future its result lands on.

        Batched path (pytorch, isolation_mode shared): the input is staged
        and queued now, and the returned future holds no reference to
        request, so once the caller drops request, the queued request keeps
        only its staged input (on the GPU with staging=gpu; with
        staging=host the queued input is a view of request's bytes).
        InferStream relies on this. Other paths: a task running infer()."""
        if self.runtime_type == "vllm" or self.isolation_mode == "none":
            return asyncio.ensure_future(self.infer(request))
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        enqueued_at = time.time()   # before staging, so the copy counts as queueing time
        envelope = RequestEnvelope(
            req_id=request.req_id,
            task=request.task,
            x=self._stage(_decode_tensor(request.x)),
            mask=self._stage(_decode_tensor(request.mask)) if request.HasField("mask") else None,
            question=request.question if request.HasField("question") else None,
            enqueued_at=enqueued_at,
            future=future,
        )
        await self.batcher.enqueue(envelope)
        return future

    def _stats(self) -> dict:
        """Control("stats"): this process's GPU memory (device.memstats.cuda_mem)
        and, per task, the requests waiting in the batch queue and their bytes
        (with staging=gpu, inputs already on the GPU)."""
        from device.config import DEVICE
        from device.memstats import MB, cuda_mem
        out = {"mode": "shared", "time_ns": time.time_ns(), **cuda_mem(DEVICE), "tasks": {},
               # the GPU or MIG slice this process sees ("" = as configured)
               "visible": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
        queues = getattr(getattr(self.batcher, "_queues", None), "_queues", None) or {}
        for task, q in list(queues.items()):
            n = len(q)
            head = q[0] if n else None
            per = getattr(getattr(head, "x", None), "nbytes", 0) or 0
            out["tasks"][task] = {"queued_n": n, "queued_mb": n * per / MB}
        return out

    async def control(self, command: str, payload_json: str) -> dict:
        if command == "stats":          # polled often: no log line
            return {"status": "ok", "logger_summary": json.dumps(self._stats())}
        logger = None
        status = "ok"
        try:
            payload = json.loads(payload_json) if payload_json else {}
            print(f"[Device] Control command={command}")

            if command == "load":
                model_config = {"gpu_memory_utilization": self.config.gpu_memory_utilization}
                if self.config.max_model_len is not None:
                    model_config["max_model_len"] = self.config.max_model_len
                if self.config.max_new_tokens is not None:
                    model_config["max_new_tokens"] = self.config.max_new_tokens
                if self.config.max_num_seqs is not None:
                    model_config["max_num_seqs"] = self.config.max_num_seqs
                if self.config.ignore_eos:
                    model_config["ignore_eos"] = True
                logger = await asyncio.to_thread(
                            self.runtime.load, payload["backbone"], payload.get("decoders", []),
                            model_config=model_config
                        )
                status = f"loaded_{payload['backbone']}"
            elif command == "swap_backbone":
                logger = await asyncio.to_thread(
                    self.runtime.swap_backbone, payload["backbone"], payload.get("decoders", [])
                )
                status = f"swapped_{payload['backbone']}"
            elif command == "add_decoder":
                logger = await asyncio.to_thread(self.runtime.add_decoders, payload["decoders"])
                status = f"added_{len(payload['decoders'])}_decoders"
            elif command == "remove_decoder":
                task_names = payload.get("tasks", [])
                logger = await asyncio.to_thread(self.runtime.remove_decoders, task_names)
                status = f"removed_{len(task_names)}_decoders"
            elif command == "add_adapter":
                logger = await asyncio.to_thread(self.runtime.add_adapters, payload["adapters"])
                status = f"added_{len(payload['adapters'])}_adapters"
            elif command == "set_rates":
                # payload: {"rates": {"task": rps, ...}}
                # Registers per-task offered rates (rps) with TokenBucketPolicy
                # if active. We invert here so low-RPS tasks accrue more credit
                # and get protected under noisy-neighbor overload.
                if self.batcher and isinstance(self.batcher._policy, TokenBucketPolicy):
                    for task, rate in payload.get("rates", {}).items():
                        r = float(rate)
                        self.batcher._policy.set_rate(task, 1.0 / r if r > 0 else 1.0)
                    status = f"rates_set_{list(payload.get('rates', {}).keys())}"
                else:
                    status = "rates_ignored_policy_not_token_bucket"
            else:
                raise ValueError(f"unknown_command_{command}")
        except Exception as exc:
            LOGGER.exception("Control operation failed")
            status = f"error_{exc}"
        print(f"[Device] Control result status={status}")
        logger_summary = str(logger.summary()) if logger else "no_logger"
        return {"status": status, "logger_summary": logger_summary}


class EdgeRuntimeServicer(edge_runtime_pb2_grpc.EdgeRuntimeServicer):
    def __init__(self, app: EdgeRuntimeApplication):
        self._app = app

    async def Infer(self, request, context):
        t_recv_ns = time.time_ns()
        try:
            response = await self._app.infer(request)
        except RuntimeError as exc:
            if str(exc) == "queue_full":
                await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "device queue is full")
            raise
        except Exception as exc:
            LOGGER.exception("Infer failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
        return self._to_response(response, t_recv_ns)

    async def InferStream(self, request_iterator, context):
        """Many requests on one stream; responses in completion order, each
        carrying its req_id. A failed request gets a response with status
        "queue_full" or "error: ..." (the stream goes on).

        Why a stream: gRPC keeps an open unary call's message bytes until the
        call ends (2-3x the payload in host RAM, measured), so with Infer a
        request waiting in the queue holds its input in host memory even when
        staged on the GPU. A message read off a stream is released once read;
        submit() keeps only the staged input."""
        out: asyncio.Queue = asyncio.Queue()
        n_open = 0
        reading = True
        submit = getattr(self._app, "submit", None)

        def error_response(req_id: int, exc: BaseException) -> edge_runtime_pb2.InferResponse:
            status = "queue_full" if str(exc) == "queue_full" else f"error: {exc}"[:500]
            return edge_runtime_pb2.InferResponse(req_id=req_id, status=status)

        async def finish(req_id: int, future, t_recv_ns: int) -> None:
            nonlocal n_open
            try:
                item = self._to_response(await future, t_recv_ns, req_id)
            except Exception as exc:
                item = error_response(req_id, exc)
            n_open -= 1                     # before the put: the writer checks n_open after a get
            out.put_nowait(item)

        async def reader() -> None:
            nonlocal n_open, reading
            try:
                async for request in request_iterator:
                    t_recv_ns, req_id = time.time_ns(), request.req_id
                    try:
                        future = (await submit(request) if submit is not None
                                  else asyncio.ensure_future(self._app.infer(request)))
                    except Exception as exc:
                        out.put_nowait(error_response(req_id, exc))
                        continue
                    finally:
                        request = None      # the queued request keeps only its staged input
                    n_open += 1
                    asyncio.ensure_future(finish(req_id, future, t_recv_ns))
            finally:
                reading = False
                out.put_nowait(None)        # wake the writer to check for the end

        reader_task = asyncio.ensure_future(reader())
        try:
            while True:
                item = await out.get()
                if item is not None:
                    yield item
                if not reading and n_open == 0 and out.empty():
                    break
            await reader_task               # re-raise a stream read error, if any
        finally:
            reader_task.cancel()

    def _to_response(self, response: dict, t_recv_ns: int, req_id: int = 0):
        raw_output = response.get("output")
        text_output = response.get("text_output", "")
        # VLM outputs are object-dtype arrays containing the generated text string
        if raw_output is not None and isinstance(raw_output, np.ndarray) and raw_output.dtype == object:
            text_output = str(raw_output.flat[0]) if raw_output.size > 0 else ""
            raw_output = None
        output_values, output_shape = _encode_output(raw_output) if raw_output is not None else ([], [])
        return edge_runtime_pb2.InferResponse(
            output=output_values,
            output_shape=output_shape,
            start_time_ns=response["start_time_ns"],
            end_time_ns=response["end_time_ns"],
            proc_time_ns=response["proc_time_ns"],
            swap_time_ns=response["swap_time_ns"],
            decoder_time_ns=response["decoder_time_ns"],
            status="ok",
            text_output=text_output,
            gpu_alloc_peak_mb=float(response.get("gpu_alloc_peak_mb", 0.0) or 0.0),
            server_time_ns=time.time_ns() - t_recv_ns,
            req_id=req_id,
        )

    async def Control(self, request, context):
        response = await self._app.control(request.command, request.payload_json)
        return edge_runtime_pb2.ControlResponse(
            status=response["status"],
            logger_summary=response["logger_summary"],
        )


async def serve(config: RuntimeServerConfig, bootstrap_json: str | None = None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if config.isolation_mode == "process":
        from device.isolated_app import IsolatedRuntimeApplication
        app = IsolatedRuntimeApplication(config)
    else:
        app = EdgeRuntimeApplication(config)  # handles both "shared" and "none"
    await app.start(bootstrap_json=bootstrap_json)

    server = grpc.aio.server()
    edge_runtime_pb2_grpc.add_EdgeRuntimeServicer_to_server(EdgeRuntimeServicer(app), server)
    bind_addr = f"{config.host}:{config.port}"
    server.add_insecure_port(bind_addr)
    await server.start()
    print(
        f"[Device] gRPC runtime listening on {bind_addr} "
        f"(max_batch_size={config.max_batch_size}, max_batch_wait_ms={config.max_batch_wait_ms})"
    )

    stop_event = asyncio.Event()

    def _request_stop():
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            pass

    await stop_event.wait()
    await server.stop(config.stop_grace_s)
    await app.stop()
