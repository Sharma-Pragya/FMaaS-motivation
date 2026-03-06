"""gRPC client wrapper for the custom device runtime."""

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
            x=_encode_tensor(request["x"]),
        )
        mask = _encode_tensor(request.get("mask"))
        if mask is not None:
            rpc_request.mask.CopyFrom(mask)
        question = request.get("question")
        if question is not None:
            if isinstance(question, np.ndarray):
                if question.size == 1:
                    question = question.item()
                else:
                    question = question.reshape(-1)[0]
            rpc_request.question = str(question)

        response = await self._stub.Infer(rpc_request)
        if response.status and response.status != "ok":
            raise RuntimeError(response.status)
        return {
            "output": _decode_output(response),
            "start_time_ns": response.start_time_ns,
            "end_time_ns": response.end_time_ns,
            "proc_time_ns": response.proc_time_ns,
            "swap_time_ns": response.swap_time_ns,
            "decoder_time_ns": response.decoder_time_ns,
            "status": response.status,
        }

    async def control(self, command: str, payload_json: str):
        await self._ensure_ready()
        response = await self._stub.Control(
            edge_runtime_pb2.ControlRequest(command=command, payload_json=payload_json)
        )
        return {
            "status": response.status,
            "logger_summary": response.logger_summary,
        }

    async def close(self):
        await self._channel.close()
