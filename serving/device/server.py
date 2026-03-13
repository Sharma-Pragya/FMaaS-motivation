import asyncio
import json
import logging
import signal
import time
from dataclasses import dataclass

import grpc
import numpy as np

from device.batcher import DeviceBatcher
from device.proto import edge_runtime_pb2, edge_runtime_pb2_grpc
from device.runtime import PyTorchRuntime, VLLMRuntime
from device.scheduler import RequestEnvelope


LOGGER = logging.getLogger(__name__)


def _decode_tensor(payload: edge_runtime_pb2.TensorPayload | None) -> np.ndarray | None:
    if payload is None or not payload.shape:
        return None
    dtype = np.dtype(payload.dtype)
    array = np.frombuffer(payload.data, dtype=dtype)
    return array.reshape(tuple(payload.shape))


def _encode_output(output: np.ndarray) -> tuple[list[float], list[int]]:
    array = np.asarray(output, dtype=np.float32)
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


class EdgeRuntimeApplication:
    """Owns the shared-model runtime, queueing, and gRPC entry points."""

    def __init__(self, config: RuntimeServerConfig):
        self.config = config
        self.runtime_type = config.runtime_type
        if config.runtime_type == "vllm":
            self.runtime = VLLMRuntime()
            self.batcher = None
        else:
            self.runtime = PyTorchRuntime()
            self.batcher = DeviceBatcher(
                runtime=self.runtime,
                max_batch_size=config.max_batch_size,
                max_batch_wait_ms=config.max_batch_wait_ms,
                queue_capacity=config.queue_capacity,
            )
        self._batch_task: asyncio.Task | None = None

    async def start(self, bootstrap_json: str | None = None):
        if bootstrap_json:
            payload = json.loads(bootstrap_json)
            print(
                f"[Device] Bootstrapping backbone={payload['backbone']} "
                f"decoders={len(payload['decoders'])}"
            )
            await asyncio.to_thread(self.runtime.load, payload["backbone"], payload.get("decoders", []))
        if self.runtime_type == "pytorch" and self._batch_task is None:
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
        print(f"[Device] Received infer req_id={request.req_id} task={request.task}")
        if self.runtime_type == "vllm":
            prompt = request.question if request.HasField("question") else ""
            return await self.runtime.infer(request.req_id, prompt)
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
                logger = await asyncio.to_thread(
                    self.runtime.load, payload["backbone"], payload.get("decoders", [])
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
        try:
            response = await self._app.infer(request)
        except RuntimeError as exc:
            if str(exc) == "queue_full":
                await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "device queue is full")
            raise
        except Exception as exc:
            LOGGER.exception("Infer failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))

        output_values, output_shape = _encode_output(response["output"]) if response.get("output") is not None else ([], [])
        return edge_runtime_pb2.InferResponse(
            output=output_values,
            output_shape=output_shape,
            start_time_ns=response["start_time_ns"],
            end_time_ns=response["end_time_ns"],
            proc_time_ns=response["proc_time_ns"],
            swap_time_ns=response["swap_time_ns"],
            decoder_time_ns=response["decoder_time_ns"],
            status="ok",
            text_output=response.get("text_output", ""),
        )

    async def Control(self, request, context):
        response = await self._app.control(request.command, request.payload_json)
        return edge_runtime_pb2.ControlResponse(
            status=response["status"],
            logger_summary=response["logger_summary"],
        )


async def serve(config: RuntimeServerConfig, bootstrap_json: str | None = None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = EdgeRuntimeApplication(config)
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
