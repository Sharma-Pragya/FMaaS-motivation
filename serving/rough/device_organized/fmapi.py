"""
fmapi.py — FMApi: the public gRPC interface to the device.

This is the only layer that knows about gRPC and wire formats.
It exposes two RPCs:

  Infer(InferRequest) → InferResponse
    Decode the proto tensor payload → RequestEnvelope → hand to fmvisor → encode result.
    fmvisor handles queuing, batching, and dispatch to the FM.

  Control(ControlRequest) → ControlResponse
    Dispatch lifecycle commands directly to fm (no fmvisor involvement):
      load           — cold-start: load backbone + decoders
      add_decoder    — hot-add a new task decoder to the running backbone
      swap_backbone  — swap the backbone in-place without restarting

Also owns:
  - RuntimeServerConfig — server bind parameters
  - serve()             — gRPC server lifecycle (start, signal handling, shutdown)

Functions:
  decode_tensor(payload)   — proto TensorPayload → numpy array
  encode_output(output)    — numpy array → (flat float list, shape list) for proto

Classes:
  RuntimeServerConfig  — dataclass for host/port/batch/queue settings
  FMApiServicer        — gRPC servicer: wires Infer and Control to fmvisor and fm
"""

import asyncio
import json
import logging
import signal
import time
from dataclasses import dataclass

import grpc
import numpy as np

from device.fm import FoundationModel
from device.fmvisor import FMVisor
from device.proto import edge_runtime_pb2, edge_runtime_pb2_grpc
from device.scheduler import RequestEnvelope


LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Wire format translation
# ------------------------------------------------------------------

def decode_tensor(payload: edge_runtime_pb2.TensorPayload | None) -> np.ndarray | None:
    """
    Decode a proto TensorPayload into a numpy array.

    Returns None if the payload is absent or has no shape (optional fields).
    Used to convert the x and mask fields of an InferRequest.
    """
    if payload is None or not payload.shape:
        return None
    dtype = np.dtype(payload.dtype)
    array = np.frombuffer(payload.data, dtype=dtype)
    return array.reshape(tuple(payload.shape))


def encode_output(output: np.ndarray) -> tuple[list[float], list[int]]:
    """
    Encode a numpy array into the flat float list + shape list expected by InferResponse.

    Converts to float32 and flattens, since proto repeated float fields are 1-D.
    The client reconstructs the original shape using output_shape.
    """
    array = np.asarray(output, dtype=np.float32)
    return array.reshape(-1).tolist(), list(array.shape)


# ------------------------------------------------------------------
# Server configuration
# ------------------------------------------------------------------

@dataclass
class RuntimeServerConfig:
    """
    Parameters for the gRPC server and the FMVisor queue/batch settings.

    Fields:
      host              — bind address (default 0.0.0.0)
      port              — gRPC listen port
      max_batch_size    — maximum requests per batch passed to fm
      max_batch_wait_ms — max wait time before dispatching a partial batch
      queue_capacity    — max total queued requests; beyond this, Infer returns RESOURCE_EXHAUSTED
      stop_grace_s      — seconds to wait for in-flight RPCs when shutting down
    """
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 1
    max_batch_wait_ms: float = 1.0
    queue_capacity: int = 1024
    stop_grace_s: float = 5.0


# ------------------------------------------------------------------
# gRPC servicer
# ------------------------------------------------------------------

class FMApiServicer(edge_runtime_pb2_grpc.EdgeRuntimeServicer):
    """
    gRPC servicer: the entry point for all external calls to the device.

    Infer  → decode proto → enqueue in fmvisor → await future → encode response
    Control → parse command → call fm directly → return status
    """

    def __init__(self, fm: FoundationModel, fmvisor: FMVisor):
        """
        Args:
          fm      — the FoundationModel; receives Control commands directly
          fmvisor — the FMVisor; receives Infer requests for queuing and dispatch
        """
        self._fm = fm
        self._fmvisor = fmvisor

    async def Infer(self, request, context):
        """
        Handle an inference request.

        Decodes the proto tensor payload into a RequestEnvelope, enqueues it
        in fmvisor, and awaits the future that fmvisor resolves after GPU execution.
        Returns the result encoded as an InferResponse proto.

        Errors:
          RESOURCE_EXHAUSTED — queue is full (fmvisor rejects the request)
          INTERNAL           — any unexpected error during inference
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        print(f"[FMApi] Infer req_id={request.req_id} task={request.task}")
        envelope = RequestEnvelope(
            req_id=request.req_id,
            task=request.task,
            x=decode_tensor(request.x),
            mask=decode_tensor(request.mask) if request.HasField("mask") else None,
            question=request.question if request.HasField("question") else None,
            enqueued_at=time.time(),
            future=future,
        )
        try:
            await self._fmvisor.enqueue(envelope)
        except RuntimeError as exc:
            if str(exc) == "queue_full":
                await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "device queue is full")
            raise

        try:
            result = await future
        except Exception as exc:
            LOGGER.exception("Infer failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            return

        output_values, output_shape = encode_output(result["output"])
        return edge_runtime_pb2.InferResponse(
            output=output_values,
            output_shape=output_shape,
            start_time_ns=result["start_time_ns"],
            end_time_ns=result["end_time_ns"],
            proc_time_ns=result["proc_time_ns"],
            swap_time_ns=result["swap_time_ns"],
            decoder_time_ns=result["decoder_time_ns"],
            status="ok",
        )

    async def Control(self, request, context):
        """
        Handle a lifecycle control command.

        Dispatches directly to fm (bypassing fmvisor) since lifecycle ops
        are synchronous, not batched. Supported commands:
          load           — {"backbone": str, "decoders": [...]}
          add_decoder    — {"decoders": [...]}
          swap_backbone  — {"backbone": str, "decoders": [...]}

        Returns a ControlResponse with status and logger summary.
        """
        status = "ok"
        logger = None
        try:
            payload = json.loads(request.payload_json) if request.payload_json else {}
            print(f"[FMApi] Control command={request.command}")
            if request.command == "load":
                logger = await asyncio.to_thread(
                    self._fm.load, payload["backbone"], payload["decoders"]
                )
                status = f"loaded_{payload['backbone']}"
            elif request.command == "add_decoder":
                logger = await asyncio.to_thread(
                    self._fm.add_decoder, payload["decoders"]
                )
                status = f"added_{len(payload['decoders'])}_decoders"
            elif request.command == "swap_backbone":
                logger = await asyncio.to_thread(
                    self._fm.swap_backbone, payload["backbone"], payload["decoders"]
                )
                status = f"swapped_{payload['backbone']}"
            else:
                raise ValueError(f"unknown_command_{request.command}")
        except Exception as exc:
            LOGGER.exception("Control operation failed")
            status = f"error_{exc}"
        print(f"[FMApi] Control result status={status}")
        logger_summary = str(logger.summary()) if logger else "no_logger"
        return edge_runtime_pb2.ControlResponse(status=status, logger_summary=logger_summary)


# ------------------------------------------------------------------
# Server lifecycle
# ------------------------------------------------------------------

async def serve(config: RuntimeServerConfig, bootstrap_json: str | None = None):
    """
    Start the gRPC server, bootstrap the FM if requested, and run until stopped.

    Wires together fm, fmvisor, and fmapi into a running server:
      1. Create and start FoundationModel (starts worker thread)
      2. If bootstrap_json provided, load initial backbone + decoders into fm
      3. Create and start FMVisor (starts batch loop)
      4. Create FMApiServicer and register it with the gRPC server
      5. Listen for SIGINT/SIGTERM and shut down gracefully

    Args:
      config         — server bind and queue configuration
      bootstrap_json — optional JSON string {"backbone": str, "decoders": [...]}
                       used for initial model load at startup
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    fm = FoundationModel()

    if bootstrap_json:
        payload = json.loads(bootstrap_json)
        print(
            f"[FMApi] Bootstrapping backbone={payload['backbone']} "
            f"decoders={len(payload['decoders'])}"
        )
        await asyncio.to_thread(fm.load, payload["backbone"], payload["decoders"])

    fmvisor = FMVisor(
        fm=fm,
        max_batch_size=config.max_batch_size,
        max_batch_wait_ms=config.max_batch_wait_ms,
        queue_capacity=config.queue_capacity,
    )
    await fmvisor.start()

    server = grpc.aio.server()
    edge_runtime_pb2_grpc.add_EdgeRuntimeServicer_to_server(
        FMApiServicer(fm=fm, fmvisor=fmvisor), server
    )
    bind_addr = f"{config.host}:{config.port}"
    server.add_insecure_port(bind_addr)
    await server.start()
    print(
        f"[FMApi] gRPC server listening on {bind_addr} "
        f"(max_batch_size={config.max_batch_size}, "
        f"max_batch_wait_ms={config.max_batch_wait_ms})"
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    await stop_event.wait()
    print("[FMApi] Shutting down...")
    await server.stop(config.stop_grace_s)
    await fmvisor.stop()
    fm.stop()
    print("[FMApi] Shutdown complete")
