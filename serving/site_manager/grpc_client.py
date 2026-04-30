"""gRPC client wrapper for the custom device runtime."""

import time

import numpy as np
import grpc

from device.proto import edge_runtime_pb2, edge_runtime_pb2_grpc


def _encode_tensor(array: np.ndarray | None) -> edge_runtime_pb2.TensorPayload | None:
    if array is None:
        return None
    data = np.ascontiguousarray(array)
    return edge_runtime_pb2.TensorPayload(
        shape=list(data.shape),
        dtype=str(data.dtype),
        data=data.tobytes(),
    )


def encode_infer_request(task: str, x: np.ndarray | None, mask: np.ndarray | None = None,
                         question: str | None = None) -> edge_runtime_pb2.InferRequest:
    """Pre-encode a reusable InferRequest with all fixed fields (task, x, mask).

    The returned object has req_id=0; callers must set req_id before each send.
    Since proto objects are mutable and not thread-safe, each concurrent sender
    should call this once and reuse the same object (one per coroutine is safe
    because coroutines don't run truly concurrently).
    """
    rpc_request = edge_runtime_pb2.InferRequest(task=task, req_id=0)
    x_tensor = _encode_tensor(x)
    if x_tensor is not None:
        rpc_request.x.CopyFrom(x_tensor)
    mask_tensor = _encode_tensor(mask)
    if mask_tensor is not None:
        rpc_request.mask.CopyFrom(mask_tensor)
    if question is not None:
        rpc_request.question = str(question)
    return rpc_request


def _decode_output(response: edge_runtime_pb2.InferResponse) -> np.ndarray:
    output = np.asarray(response.output, dtype=np.float32)
    if response.output_shape:
        return output.reshape(tuple(response.output_shape))
    return output


class EdgeRuntimeClient:
    def __init__(self, url: str):
        self.url = url
        self._channel = grpc.aio.insecure_channel(url)
        self._stub = edge_runtime_pb2_grpc.EdgeRuntimeStub(self._channel)
        self._ready = False

    async def _ensure_ready(self):
        if not self._ready:
            await self._channel.channel_ready()
            self._ready = True

    async def wait_ready(self):
        await self._ensure_ready()
        return True

    async def infer(self, request: dict):
        await self._ensure_ready()
        rpc_request = edge_runtime_pb2.InferRequest(
            req_id=request["req_id"],
            task=request["task"],
        )
        x_tensor = _encode_tensor(request.get("x"))
        if x_tensor is not None:
            rpc_request.x.CopyFrom(x_tensor)
        mask = _encode_tensor(request.get("mask"))
        if mask is not None:
            rpc_request.mask.CopyFrom(mask)
        question = request.get("question")
        if question is not None:
            if isinstance(question, list):
                question = question[0] if question else ""
            elif isinstance(question, np.ndarray):
                if question.size == 1:
                    question = question.item()
                else:
                    question = question.reshape(-1)[0]
            rpc_request.question = str(question)

        rpc_start_time_ns = time.time_ns()
        response = await self._stub.Infer(rpc_request)
        rpc_end_time_ns = time.time_ns()
        if response.status and response.status != "ok":
            raise RuntimeError(response.status)
        return {
            "output": _decode_output(response),
            "text_output": response.text_output or "",
            "rpc_start_time_ns": rpc_start_time_ns,
            "rpc_end_time_ns": rpc_end_time_ns,
            "start_time_ns": response.start_time_ns,
            "end_time_ns": response.end_time_ns,
            "proc_time_ns": response.proc_time_ns,
            "swap_time_ns": response.swap_time_ns,
            "decoder_time_ns": response.decoder_time_ns,
            "status": response.status,
            "gpu_alloc_peak_mb": response.gpu_alloc_peak_mb,
        }

    async def control(self, command: str, payload_json: str, timeout_s: float | None = 120.0):
        await self._ensure_ready()
        response = await self._stub.Control(
            edge_runtime_pb2.ControlRequest(command=command, payload_json=payload_json),
            timeout=timeout_s,
        )
        return {
            "status": response.status,
            "logger_summary": response.logger_summary,
        }

    async def close(self):
        await self._channel.close()
