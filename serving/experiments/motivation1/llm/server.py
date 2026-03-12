#!/usr/bin/env python3
"""In-process LLM serving for motivation1.

Mirrors the TSFM experiment structure:
- task_sharing: one local vLLM runtime per task
- deploy_sharing: one shared local vLLM runtime for all tasks
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Dict, List

import torch

import sys

SERVING_DIR = Path(__file__).resolve().parents[3]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))


class _LocalVLLMEndpoint:
    """Single vLLM runtime + dedicated event loop thread for inference."""

    def __init__(self, backbone: str, device: str, model_config: dict, loop: asyncio.AbstractEventLoop):
        self.backbone = backbone
        self.device = device
        self.model_config = model_config
        self._loop = loop
        self.runtime = None

    async def _init_runtime(self) -> None:
        from device.vllm_runtime import VLLMRuntime

        runtime = VLLMRuntime()
        runtime.load(
            backbone=self.backbone,
            decoders=[],
            device=self.device,
            model_config=self.model_config,
        )
        self.runtime = runtime

    def start(self) -> None:
        try:
            init_future: Future = asyncio.run_coroutine_threadsafe(self._init_runtime(), self._loop)
            init_future.result()
        except Exception as e:
            print(f"[LocalVLLMEndpoint] failed to initialize runtime: {e}")
            raise

    async def infer(self, req_id: int, prompt: str) -> dict:
        if self.runtime is None:
            raise RuntimeError("runtime_not_started")
        future: Future = asyncio.run_coroutine_threadsafe(
            self.runtime.infer(req_id=req_id, prompt=prompt),
            self._loop,
        )
        try:
            return await asyncio.wrap_future(future)
        except Exception as e:
            print(f"[LocalVLLMEndpoint] infer failed req_id={req_id}: {type(e).__name__}: {e}")
            raise

    def stop(self) -> None:
        self.runtime = None

    def memory_stats(self) -> dict:
        if self.runtime is None:
            return {}
        return self.runtime.memory_stats or {}


class LocalRuntimeClient:
    """EdgeRuntimeClient-compatible interface for local in-process inference."""

    def __init__(self, endpoint: _LocalVLLMEndpoint, label: str):
        self._endpoint = endpoint
        self.label = label

    async def wait_ready(self) -> bool:
        return True

    async def infer(self, request: dict) -> dict:
        req_id = int(request["req_id"])
        prompt = str(request.get("question", ""))
        return await self._endpoint.infer(req_id=req_id, prompt=prompt)

    async def close(self) -> None:
        return None


class LLMInferenceServer:
    """In-process LLM server abstraction for motivation1 experiment."""

    def __init__(
        self,
        strategy: str,
        tasks: List[str],
        backbone: str,
        device: str = "cuda:0",
        model_config: dict | None = None,
    ):
        if strategy not in ("task_sharing", "deploy_sharing"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
        self.tasks = tasks
        self.backbone = backbone
        self.device = device
        self.model_config = model_config or {}

        self._endpoints: List[_LocalVLLMEndpoint] = []
        self._task_endpoint: Dict[str, _LocalVLLMEndpoint] = {}
        self._task_client: Dict[str, LocalRuntimeClient] = {}
        self._model_load_mem_mb = 0.0
        self._peak_mem_mb = 0.0
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_started = False

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def start(self) -> None:
        if not self._loop_started:
            self._loop_thread.start()
            self._loop_started = True

        dev = torch.device(self.device)
        if dev.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(dev)
            except Exception:
                pass

        if self.strategy == "deploy_sharing":
            endpoint = _LocalVLLMEndpoint(self.backbone, self.device, self.model_config, self._loop)
            endpoint.start()
            self._endpoints = [endpoint]
            for task in self.tasks:
                self._task_endpoint[task] = endpoint
        else:
            for task in self.tasks:
                endpoint = _LocalVLLMEndpoint(self.backbone, self.device, self.model_config, self._loop)
                endpoint.start()
                self._endpoints.append(endpoint)
                self._task_endpoint[task] = endpoint

        for idx, task in enumerate(self.tasks):
            self._task_client[task] = LocalRuntimeClient(self._task_endpoint[task], f"local://instance-{idx}")

        if dev.type == "cuda":
            self._model_load_mem_mb = torch.cuda.memory_allocated(dev) / (1024 ** 2)
            self._peak_mem_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

    def stop(self) -> None:
        for endpoint in self._endpoints:
            endpoint.stop()
        self._endpoints.clear()
        self._task_endpoint.clear()
        self._task_client.clear()
        if self._loop_started:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5)
            self._loop_started = False

        dev = torch.device(self.device)
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    def client(self, task: str) -> LocalRuntimeClient:
        return self._task_client[task]

    def model_load_mem_mb(self) -> float:
        return self._model_load_mem_mb

    def peak_gpu_mem_mb(self) -> float:
        return self._peak_mem_mb

    def memory_summary(self) -> dict:
        model_gb = 0.0
        reserved_gb = 0.0
        util_sum = 0.0
        total_gpu_gb = 0.0
        n = len(self._endpoints)
        for endpoint in self._endpoints:
            mem = endpoint.memory_stats()
            model_gb += float(mem.get("model_memory_gb", 0.0))
            reserved_gb += float(mem.get("reserved_gb", 0.0))
            util_sum += float(mem.get("gpu_memory_utilization", 0.0))
            total_gpu_gb = float(mem.get("total_gpu_gb", 0.0))  # same GPU, just take last
        return {
            "model_memory_mb": round(model_gb * 1024),
            "total_reserved_mb": round(reserved_gb * 1024),
            "avg_gpu_memory_utilization": round(util_sum / n, 4) if n else 0.0,
            "total_gpu_gb": total_gpu_gb,
        }

    def deployment_records(self) -> List[dict]:
        records: List[dict] = []
        for idx, endpoint in enumerate(self._endpoints):
            records.append(
                {
                    "endpoint": f"local://instance-{idx}",
                    "model_config": self.model_config,
                    "memory": endpoint.memory_stats(),
                }
            )
        return records
