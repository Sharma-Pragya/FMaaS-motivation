#!/usr/bin/env python3
"""In-process LLM serving for motivation1.

- deploy_sharing: 1 shared vLLM runtime, all tasks route to it via a thread-safe client
- task_sharing:   1 vLLM runtime per task in a separate process (vLLM constraint: one engine per process)
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import threading
import traceback
from concurrent.futures import Future
from pathlib import Path
from typing import Dict, List

import torch
import sys

SERVING_DIR = Path(__file__).resolve().parents[3]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _memory_record(runtime) -> dict:
    bb = runtime.logger.summary().get("load_backbone", {}) if runtime.logger else {}
    return {
        "model_weights_mb": runtime.model_weights_bytes / 1e6,
        "gpu_peak_mb":      bb.get("gpu peak",     0.0),
        "gpu_reserved_mb":  bb.get("gpu reserved", 0.0),
    }


# ---------------------------------------------------------------------------
# Client for deploy_sharing — owns a VLLMRuntime in a dedicated event-loop thread
# ---------------------------------------------------------------------------

class InProcessClient:
    """Loads and serves one VLLMRuntime inside a background event-loop thread."""

    def __init__(self, backbone: str, device: str, model_config: dict,
                 loop: asyncio.AbstractEventLoop):
        self._backbone     = backbone
        self._device       = device
        self._model_config = model_config
        self._loop         = loop
        self.runtime       = None

    def start(self) -> None:
        future: Future = asyncio.run_coroutine_threadsafe(self._load(), self._loop)
        future.result()

    async def _load(self) -> None:
        from device.runtime import VLLMRuntime
        self.runtime = VLLMRuntime()
        self.runtime.load(backbone=self._backbone, decoders=[],
                          device=self._device, model_config=self._model_config)

    async def infer(self, request: dict) -> dict:
        future: Future = asyncio.run_coroutine_threadsafe(
            self.runtime.infer(int(request["req_id"]), str(request.get("question", ""))),
            self._loop,
        )
        return await asyncio.wrap_future(future)

    async def wait_ready(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    def memory_stats(self) -> dict:
        return _memory_record(self.runtime) if self.runtime else {}

    def stop(self) -> None:
        self.runtime = None


# ---------------------------------------------------------------------------
# Client stub for task_sharing — holds memory stats, inference already done
# ---------------------------------------------------------------------------

class ProcessClient:
    """Thin stub representing a process-isolated runtime (task_sharing).
    Inference ran inside the process; this just exposes memory stats."""

    def __init__(self, memory: dict):
        self._memory = memory

    def memory_stats(self) -> dict:
        return self._memory

    def stop(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Process worker for task_sharing
# ---------------------------------------------------------------------------

def _process_worker(
    task: str,
    prompts: List[str],
    duration: float,
    backbone: str,
    device: str,
    model_config: dict,
    result_queue,
) -> None:
    import asyncio, time, traceback
    label = f"local://process-{task}"
    try:
        from device.runtime import VLLMRuntime
        runtime = VLLMRuntime()
        runtime.load(backbone=backbone, decoders=[], device=device, model_config=model_config)
        mem_stats = _memory_record(runtime)
    except Exception as e:
        result_queue.put({
            "task": task, "label": label,
            "n_requests": 0, "throughput_rps": 0.0, "avg_latency_ms": 0.0,
            "memory": {}, "error": f"load_failed: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })
        return

    try:
        async def _run():
            latencies, stop, req_id, lock = [], asyncio.Event(), 0, asyncio.Lock()
            deadline, cycle = time.time() + duration, len(prompts)

            async def _worker():
                nonlocal req_id
                while not stop.is_set():
                    async with lock:
                        rid = req_id; req_id += 1
                    t0 = time.time()
                    await runtime.infer(rid, prompts[rid % cycle])
                    latencies.append((time.time() - t0) * 1000)

            async def _timer():
                await asyncio.sleep(max(0, deadline - time.time()))
                stop.set()

            concurrency = model_config.get("concurrency", 4)
            await asyncio.gather(
                asyncio.create_task(_timer()),
                *[asyncio.create_task(_worker()) for _ in range(concurrency)],
                return_exceptions=True,
            )
            return latencies

        latencies = asyncio.run(_run())
        n = len(latencies)
        result_queue.put({
            "task": task, "label": label,
            "n_requests": n,
            "throughput_rps": n / duration if duration > 0 else 0.0,
            "avg_latency_ms": sum(latencies) / n if n else 0.0,
            "memory": mem_stats,
            "error": None,
        })
    except Exception as e:
        result_queue.put({
            "task": task, "label": label,
            "n_requests": 0, "throughput_rps": 0.0, "avg_latency_ms": 0.0,
            "memory": {}, "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class LLMInferenceServer:
    """Unified LLM server for both strategies.

    deploy_sharing: one InProcessClient shared across tasks.
    task_sharing:   one process per task (vLLM requires one engine per process);
                    inference runs inside each process, results collected here.
    """

    def __init__(self, strategy: str, tasks: List[str], backbone: str,
                 device: str = "cuda:0", model_config: dict | None = None):
        if strategy not in ("task_sharing", "deploy_sharing"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy     = strategy
        self.tasks        = tasks
        self.backbone     = backbone
        self.device       = device
        self.model_config = model_config or {}

        self._clients: Dict[str, InProcessClient | ProcessClient] = {}
        self._process_results: List[dict] = []   # task_sharing only

        # Shared event loop for deploy_sharing InProcessClients
        self._loop        = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=lambda: (
            asyncio.set_event_loop(self._loop), self._loop.run_forever()
        ), daemon=True)
        self._loop_started = False

    def _ensure_loop(self) -> None:
        if not self._loop_started:
            self._loop_thread.start()
            self._loop_started = True

    # ------------------------------------------------------------------
    # start
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.strategy == "deploy_sharing":
            self._ensure_loop()
            client = InProcessClient(self.backbone, self.device, self.model_config, self._loop)
            client.start()
            self._clients = {task: client for task in self.tasks}
        else:
            self._start_task_sharing()

    def _start_task_sharing(self) -> None:
        ctx          = mp.get_context("spawn")
        result_queue = ctx.Queue()
        duration     = self.model_config.get("duration", 60)

        procs = []
        for task in self.tasks:
            p = ctx.Process(
                target=_process_worker,
                args=(task, self.model_config.get("prompts", {}).get(task, []),
                      duration, self.backbone, self.device, self.model_config,
                      result_queue),
                daemon=True,
            )
            p.start()
            procs.append(p)

        print(f"[task_sharing] {len(self.tasks)} processes launched — waiting for results")

        # Collect results — use whatever arrived within duration+60s; partial ok
        results = []
        for _ in self.tasks:
            try:
                results.append(result_queue.get(timeout=duration + 60))
            except queue.Empty:
                break  # no more results within timeout; use what we have

        for p in procs:
            p.join(timeout=5)
            if p.is_alive(): p.terminate()

        self._process_results = results
        self._clients = {r["task"]: ProcessClient(r["memory"]) for r in results}

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def client(self, task: str) -> InProcessClient:
        return self._clients[task]

    def task_sharing_results(self) -> List[dict]:
        return self._process_results

    def stop(self) -> None:
        for c in self._clients.values():
            c.stop()
        self._clients.clear()
        if self._loop_started:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5)
            self._loop_started = False
        dev = torch.device(self.device)
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Memory / deployment
    # ------------------------------------------------------------------

    def memory_summary(self) -> dict:
        total_model_mb = total_reserved_mb = total_gpu_gb = 0.0
        try:
            dev = torch.device(self.device)
            if dev.type == "cuda":
                total_gpu_gb = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
        except Exception:
            pass
        seen = set()
        for task, client in self._clients.items():
            cid = id(client)
            if cid in seen:
                continue
            seen.add(cid)
            stats = client.memory_stats()
            total_model_mb    += float(stats.get("model_weights_mb", 0.0))
            total_reserved_mb += float(stats.get("gpu_reserved_mb",  0.0))
        return {
            "model_memory_mb":   round(total_model_mb),
            "total_reserved_mb": round(total_reserved_mb),
            "total_gpu_gb":      total_gpu_gb,
        }

    def deployment_records(self) -> List[dict]:
        seen, records = set(), []
        for idx, (task, client) in enumerate(self._clients.items()):
            cid = id(client)
            if cid in seen:
                continue
            seen.add(cid)
            records.append({
                "endpoint":     f"local://instance-{idx}",
                "model_config": self.model_config,
                "memory":       client.memory_stats(),
            })
        return records
