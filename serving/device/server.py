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
    tpc_mode: str = "none"               # "none" | "libsmctrl" | "green"
    tpc_partition: list = None           # list of TPC IDs to pin this server to
    worker_mode: str = "threaded"        # "threaded" | "inline" — per-task pipeline worker mode
    verbose_batch_logs: bool = False      # high-frequency per-batch request logging


class EdgeRuntimeApplication:
    """Owns the shared-model runtime, queueing, and gRPC entry points."""

    def __init__(self, config: RuntimeServerConfig):
        self.config = config
        self.runtime_type = config.runtime_type
        self.isolation_mode = config.isolation_mode
        if config.isolation_mode == "process":
            raise ValueError("Use IsolatedRuntimeApplication for isolation_mode=process")
        if config.runtime_type == "vllm":
            self.runtime = VLLMRuntime()
            self.batcher = None
        else:
            cuda_stream = self._setup_tpc(config)
            self.runtime = PyTorchRuntime(cuda_stream=cuda_stream)
            if config.isolation_mode == "none":
                # No batcher — requests go directly to runtime.run_batch()
                self.batcher = None
            else:
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

    @staticmethod
    def _setup_tpc(config: RuntimeServerConfig):
        """Create and TPC-pin a CUDA stream if configured. Returns stream or None."""
        if config.tpc_mode == "none" or not config.tpc_partition:
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
            lib.libsmctrl_set_stream_mask.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
            lib.libsmctrl_set_stream_mask.restype = None
            lib.libsmctrl_get_tpc_info_cuda.argtypes = [
                ctypes.POINTER(ctypes.c_uint32), ctypes.c_int
            ]
            lib.libsmctrl_get_tpc_info_cuda.restype = ctypes.c_int

            num_tpcs = ctypes.c_uint32()
            ret = lib.libsmctrl_get_tpc_info_cuda(ctypes.byref(num_tpcs), device_id)
            total_tpcs = num_tpcs.value if ret == 0 else sm_count // 2

            stream = torch.cuda.Stream(device=cuda_device)
            enable_bits = 0
            for tid in config.tpc_partition:
                enable_bits |= (1 << tid)
            disable_mask = (~enable_bits) & 0xFFFFFFFFFFFFFFFF
            lib.libsmctrl_set_stream_mask(
                ctypes.c_void_p(stream.cuda_stream),
                ctypes.c_uint64(disable_mask),
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
            await asyncio.to_thread(
                self.runtime.load,
                payload["backbone"],
                payload.get("decoders", []),
                model_config=model_config,
            ) 
        if self.runtime_type == "pytorch" and self.batcher is not None and self._batch_task is None:
            self._batch_task = asyncio.create_task(self.batcher.run_forever())
        print("[Device] Runtime application started")

    async def stop(self):
        print("[Device] Stopping runtime application")
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
            return await self.runtime.infer(request.req_id, prompt)
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
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        envelope = RequestEnvelope(
            req_id=request.req_id,
            task=request.task,
            x=_decode_tensor(request.x),
            mask=_decode_tensor(request.mask) if request.HasField("mask") else None,
            question=request.question if request.HasField("question") else None,
            enqueued_at=time.time(),
            future=future,
        )
        await self.batcher.enqueue(envelope)
        return await future

    async def control(self, command: str, payload_json: str) -> dict:
        logger = None
        status = "ok"
        try:
            payload = json.loads(payload_json) if payload_json else {}
            print(f"[Device] Control command={command}")

            if command == "load":
                model_config = {"gpu_memory_utilization": self.config.gpu_memory_utilization}
                if self.config.max_model_len is not None:
                    model_config["max_model_len"] = self.config.max_model_len
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
