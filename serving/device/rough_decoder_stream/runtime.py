import threading
import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np

from fmtk.logger import Logger
from device.model_loader import ModelLoader


# ---------------------------------------------------------------------------
# Read-write lock: multiple readers (inference) OR one writer (load/swap/add)
# ---------------------------------------------------------------------------

class _RWLock:
    """Simple readers-writer lock.

    Multiple threads can hold the read lock concurrently (run_backbone and
    run_decoders on separate threads).  The write lock is exclusive — used by
    load / swap_backbone / add_decoders / add_adapters to safely mutate
    pipeline state without racing with inference.
    """

    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False

    def read_acquire(self):
        with self._cond:
            while self._writer:
                self._cond.wait()
            self._readers += 1

    def read_release(self):
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def write_acquire(self):
        with self._cond:
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._writer = True

    def write_release(self):
        with self._cond:
            self._writer = False
            self._cond.notify_all()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class BatchRunResult:
    outputs: list             # list[np.ndarray] per sample, each flattened
    start_time_ns: int
    end_time_ns: int
    proc_time_ns: int         # backbone forward time (ns)
    swap_time_ns: list[int]   # per-task decoder lookup time (ns)
    decoder_time_ns: list[int]# per-task decoder forward time (ns)
    gpu_alloc_peak_mb: float  # peak GPU memory allocated during this batch (MB)


@dataclass
class BackboneResult:
    """Intermediate result after backbone forward, before decoders."""
    feats_by_idx: dict        # {sample_index: feature_tensor}
    task_names: list[str]
    start_time_ns: int
    proc_time_ns: int
    peak_bytes: int
    cuda_event: object = None # CUDA event recorded after backbone forward on _cuda_stream


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseRuntime(ABC):
    """Shared lifecycle interface for all runtime backends.

    Concrete subclasses implement load / swap_backbone / add_decoders using
    whatever backend they own (PyTorch pipeline, vLLM engine, …).
    Inference is intentionally NOT on this base class — PyTorchRuntime exposes
    run_batch() (numpy in, numpy out) while VLLMRuntime exposes async infer()
    (prompt string in, text out). Forcing a common signature would lie about
    the contract.
    """

    @abstractmethod
    def load(self, backbone: str, decoders: list, **kwargs) -> Logger:
        """Load backbone + decoders. Returns op Logger with timing."""

    @abstractmethod
    def swap_backbone(self, backbone: str, decoders: list) -> Logger:
        """Release current backbone, load new one."""

    @abstractmethod
    def add_decoders(self, decoders: list) -> Logger:
        """Hot-add decoder heads to the loaded backbone."""

    @abstractmethod
    def add_adapters(self, adapters: list) -> Logger:
        """Hot-add LoRA adapters to the loaded backbone."""


# ---------------------------------------------------------------------------
# PyTorch runtime (TSFM backbones + MLP decoders)
# ---------------------------------------------------------------------------

class PyTorchRuntime(BaseRuntime):
    """Owns a ModelLoader and a single Logger. All operations — load, swap,
    add_decoder, run_batch — record into the same Logger automatically via
    the Pipeline (which holds the same Logger reference).

    Call self.logger.summary() for per-section averages, or .save() to persist.
    """

    def __init__(self, loader: ModelLoader | None = None, cuda_stream=None):
        self._rwlock = _RWLock()  # write for load/swap/add, read for inference
        self._loader = loader if loader is not None else ModelLoader()
        self.logger  = Logger(self._loader.device, "runtime")
        self._loader.logger = self.logger
        self.pipeline = None
        self.decoders = None
        self.adapters = None
        # Always use a dedicated (non-default) CUDA stream to avoid implicit
        # synchronisation overhead of the default stream under concurrent load.
        import torch
        if cuda_stream is None:
            device = self._loader.device
            if str(device).startswith("cuda"):
                cuda_stream = torch.cuda.Stream(device=device)
        self._cuda_stream = cuda_stream
        # Second stream for decoder work — enables overlapping decoder N with
        # backbone N+1 when the batcher pipelines execution.
        self._dec_stream = None
        device = self._loader.device
        if str(device).startswith("cuda"):
            self._dec_stream = torch.cuda.Stream(device=device)

    def _sync(self):
        self.pipeline = self._loader.pipeline
        self.decoders = self._loader.decoders
        self.adapters = self._loader.adapters

    def load(self, backbone: str, decoders: list, **kwargs) -> Logger:
        self._rwlock.write_acquire()
        try:
            op_log = self._loader.load_models(backbone, decoders, dec_stream=self._dec_stream)
            self._sync()
            return op_log
        finally:
            self._rwlock.write_release()

    def swap_backbone(self, backbone: str, decoders: list) -> Logger:
        self._rwlock.write_acquire()
        try:
            op_log = self._loader.swap_backbone(backbone, decoders, dec_stream=self._dec_stream)
            self._sync()
            return op_log
        finally:
            self._rwlock.write_release()

    def add_decoders(self, decoders: list) -> Logger:
        self._rwlock.write_acquire()
        try:
            op_log = self._loader.add_decoder(decoders, dec_stream=self._dec_stream)
            self._sync()
            return op_log
        finally:
            self._rwlock.write_release()

    def add_adapters(self, adapters: list) -> Logger:
        self._rwlock.write_acquire()
        try:
            op_log = self._loader.add_adapter(adapters)
            self._sync()
            return op_log
        finally:
            self._rwlock.write_release()

    def run_backbone(
        self,
        x: np.ndarray,
        task_names: list[str],
        mask: np.ndarray | None = None,
        questions: list[str | None] | None = None,
    ) -> BackboneResult:
        """Backbone forward only. Returns features for run_decoders().

        Runs on self._cuda_stream. Caller must hold no decoder work in flight
        on self._dec_stream that would conflict (the batcher handles this via
        thread join before the next backbone forward).

        Locking: holds a read lock so that load/swap/add (write lock) cannot
        mutate pipeline state mid-inference.  Multiple read locks can be held
        concurrently, so run_decoders on a separate thread is not blocked.
        """
        import torch
        self._rwlock.read_acquire()
        try:
            return self._run_backbone_inner(x, task_names, mask, questions)
        finally:
            self._rwlock.read_release()

    def _run_backbone_inner(self, x, task_names, mask, questions):
        import torch
        if self.pipeline is None:
            raise RuntimeError("pipeline_not_loaded: backbone has not been loaded yet")
        device   = self._loader.device
        is_cuda  = str(device).startswith("cuda")
        start_ns = time.time_ns()

        if x is None:
            bx     = None
            b_mask = None
        else:
            bx = torch.from_numpy(x)
            b_mask = torch.from_numpy(mask) if mask is not None else None

        adapters_map = self.adapters or {}
        groups: list[tuple[str | None, list[int]]] = []
        for idx, task in enumerate(task_names):
            adapter_name = adapters_map.get(task)
            if groups and groups[-1][0] == adapter_name:
                groups[-1][1].append(idx)
            else:
                groups.append((adapter_name, [idx]))

        feats_by_idx: dict[int, object] = {}
        proc_time_ns = 0
        peak_bytes   = 0

        stream_ctx = torch.cuda.stream(self._cuda_stream) if self._cuda_stream else nullcontext()
        with stream_ctx, torch.no_grad():
            for adapter_name, indices in groups:
                if adapter_name is not None:
                    self.pipeline.set_adapter(adapter_name)
                else:
                    self.pipeline.unload_adapter()

                sub_mask = b_mask[indices] if b_mask is not None else None

                bb_start = time.time_ns()
                if questions is not None:
                    sub_q = [questions[i] for i in indices]
                    if bx is not None:
                        sub_x = bx[indices]
                        sub_feats = self.pipeline.model_instance.forward((sub_x, sub_q), sub_mask)
                    else:
                        sub_feats = self.pipeline.model_instance.forward((None, questions), sub_mask)
                else:
                    sub_x = bx[indices]
                    sub_feats = self.pipeline.model_instance.forward(sub_x, sub_mask)
                if is_cuda:
                    # Sync only _cuda_stream (not the whole device) so we don't
                    # wait for a previous batch's decoder on _dec_stream.
                    if self._cuda_stream is not None:
                        self._cuda_stream.synchronize()
                    else:
                        torch.cuda.synchronize(device)
                proc_time_ns += time.time_ns() - bb_start

                if is_cuda:
                    cur_bytes = torch.cuda.memory_allocated(device)
                    if cur_bytes > peak_bytes:
                        peak_bytes = cur_bytes

                for out_pos, orig_idx in enumerate(indices):
                    feats_by_idx[orig_idx] = sub_feats[out_pos : out_pos + 1]

        # Record event so _dec_stream can wait for backbone output to be ready
        # without going through the default stream (which serializes everything).
        event = None
        if is_cuda and self._cuda_stream is not None:
            event = torch.cuda.Event()
            event.record(self._cuda_stream)

        return BackboneResult(
            feats_by_idx=feats_by_idx,
            task_names=task_names,
            start_time_ns=start_ns,
            proc_time_ns=proc_time_ns,
            peak_bytes=peak_bytes,
            cuda_event=event,
        )

    def run_decoders(
        self,
        bb: BackboneResult,
        use_dec_stream: bool = False,
    ) -> BatchRunResult:
        """Decoder pass using features from run_backbone().

        If use_dec_stream is True, runs on self._dec_stream so that the caller
        can overlap the next backbone forward on self._cuda_stream.  The caller
        must join the decoder thread before starting the next backbone forward
        to ensure feature memory can be safely reused.

        Locking: holds a read lock (same as run_backbone) — concurrent with
        other readers but exclusive with load/swap/add.
        """
        import torch
        import torch.nn as nn
        self._rwlock.read_acquire()
        try:
            return self._run_decoders_inner(bb, use_dec_stream)
        finally:
            self._rwlock.read_release()

    def _run_decoders_inner(self, bb, use_dec_stream):
        import torch
        import torch.nn as nn
        device   = self._loader.device
        is_cuda  = str(device).startswith("cuda")
        peak_bytes = bb.peak_bytes

        if use_dec_stream and self._dec_stream is not None:
            stream_ctx = torch.cuda.stream(self._dec_stream)
            # Make _dec_stream wait for backbone output (recorded on _cuda_stream)
            # so features are ready.  This avoids touching the default stream
            # which would serialize all streams.
            if bb.cuda_event is not None:
                self._dec_stream.wait_event(bb.cuda_event)
        else:
            stream_ctx = torch.cuda.stream(self._cuda_stream) if self._cuda_stream else nullcontext()

        outputs       = [None] * len(bb.task_names)
        swap_times    = [0] * len(bb.task_names)
        decoder_times = [0] * len(bb.task_names)

        task_groups: dict[str, list[int]] = {}
        for index, task_name in enumerate(bb.task_names):
            task_groups.setdefault(task_name, []).append(index)

        for task_name, indices in task_groups.items():
            swap_start = time.time_ns()
            decoder_name = self.decoders.get(task_name)
            active_decoder = self.pipeline.decoders[decoder_name] if decoder_name else None
            swap_elapsed = time.time_ns() - swap_start
            for index in indices:
                swap_times[index] = swap_elapsed

            dec_start = time.time_ns()
            feat_batch = [bb.feats_by_idx[index] for index in indices]

            if active_decoder is not None:
                with stream_ctx, torch.no_grad():
                    feat_input = torch.cat(feat_batch, dim=0)
                    logits = active_decoder.forward(feat_input)
                if is_cuda:
                    if use_dec_stream and self._dec_stream is not None:
                        self._dec_stream.synchronize()
                    else:
                        torch.cuda.synchronize(device)
                    dec_bytes = torch.cuda.memory_allocated(device)
                    if dec_bytes > peak_bytes:
                        peak_bytes = dec_bytes
                if isinstance(active_decoder.criterion, nn.CrossEntropyLoss):
                    logits = torch.argmax(logits, dim=1)
                result_batch = logits.detach().cpu().numpy()

                if result_batch.ndim == 0:
                    result_batch = result_batch.reshape(1)
                batched_results = [np.asarray(result_batch[i:i + 1]) for i in range(len(indices))]
            else:
                batched_results = []
                for feat_i in feat_batch:
                    if isinstance(feat_i, (list, str)):
                        text = feat_i[0] if isinstance(feat_i, list) else feat_i
                        batched_results.append(np.array([text], dtype=object))
                    else:
                        batched_results.append(feat_i.detach().cpu().float().numpy())

            dec_elapsed = time.time_ns() - dec_start
            per_item_dec_elapsed = dec_elapsed // len(indices)
            for index, result_i in zip(indices, batched_results):
                decoder_times[index] = per_item_dec_elapsed
                outputs[index] = result_i.reshape(-1)
        end_ns = time.time_ns()

        return BatchRunResult(
            outputs=outputs,
            start_time_ns=bb.start_time_ns,
            end_time_ns=end_ns,
            proc_time_ns=bb.proc_time_ns,
            swap_time_ns=swap_times,
            decoder_time_ns=decoder_times,
            gpu_alloc_peak_mb=peak_bytes / (1024 ** 2),
        )

    def run_batch(
        self,
        x: np.ndarray,
        task_names: list[str],
        mask: np.ndarray | None = None,
        questions: list[str | None] | None = None,
    ) -> BatchRunResult:
        """Convenience: backbone + decoders in one call (non-pipelined).

        Used by the direct path (isolation_mode=none) and any caller that
        doesn't need to overlap backbone N+1 with decoder N.
        Acquires read lock internally via run_backbone/run_decoders.
        """
        bb = self.run_backbone(x, task_names, mask, questions)
        return self.run_decoders(bb)


# ---------------------------------------------------------------------------
# vLLM runtime (LLM backbones, continuous batching via AsyncLLMEngine)
# ---------------------------------------------------------------------------

class VLLMRuntime(BaseRuntime):
    """Uses ModelLoader + Pipeline for lifecycle (load, swap, metrics) and
    delegates inference to pipeline.model_instance.async_forward() for true
    continuous batching via vLLM's AsyncLLMEngine.

    decoders is always [] for LLMs — accepted for interface compatibility.
    """

    def __init__(self):
        self._loader  = ModelLoader()
        self.logger   = Logger(self._loader.device, "runtime")
        self._loader.logger = self.logger
        self.pipeline = None
        self.backbone: str | None = None
        self.model_weights_bytes: int = 0  # set by load(), weights only (no KV cache)

    def _read_vllm_weights_bytes(self) -> int:
        """Read model weights memory from vLLM's internal model_runner.
        Path: async_engine → engine → model_executor → driver_worker → model_runner.model_memory_usage
        Returns bytes, or 0 if the path doesn't exist (vLLM version mismatch)."""
        try:
            engine = self.pipeline.model_instance._async_engine.engine
            return int(engine.model_executor.driver_worker.model_runner.model_memory_usage)
        except Exception:
            return 0

    def load(self, backbone: str, decoders: list, device: str = None,
             model_config: dict | None = None) -> Logger:
        import os as _os
        if device is not None:
            self._loader.device = device
        physical_device = str(self._loader.device)  # e.g. "cuda:1"
        # Set CUDA_VISIBLE_DEVICES so vLLM uses the right physical GPU.
        # After this, the logical device within this process is always cuda:0.
        gpu_index = int(physical_device.split(':')[1]) if ':' in physical_device else 0
        _os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
        # Use cuda:0 for the loader and logger — after CUDA_VISIBLE_DEVICES
        # remapping, cuda:1 is no longer a valid device in this process.
        self._loader.device = "cuda:0"
        self.logger = Logger("cuda:0", "runtime")
        self._loader.logger = self.logger
        op_log = self._loader.load_models(backbone, decoders, model_config=model_config)
        self.pipeline = self._loader.pipeline
        self.backbone = backbone
        self.model_weights_bytes = self._read_vllm_weights_bytes()
        return op_log

    def swap_backbone(self, backbone: str, decoders: list) -> Logger:
        op_log = self._loader.swap_backbone(backbone, decoders)
        self.pipeline = self._loader.pipeline
        self.backbone = backbone
        return op_log

    def add_decoders(self, decoders: list) -> Logger:
        # LLMs have no task-specific decoder heads — no-op
        return Logger(self._loader.device, "noop")

    def add_adapters(self, adapters: list) -> Logger:
        # LLMs managed by vLLM don't use hot-added LoRA adapters here — no-op
        return Logger(self._loader.device, "noop")

    async def infer(self, req_id: int, prompt: str) -> dict:
        """Single prompt → generated text. Multiple concurrent calls are
        batched at the iteration level by vLLM's AsyncLLMEngine — true
        continuous batching with no extra logic needed here."""
        if self.pipeline is None:
            raise RuntimeError("vllm_model_not_loaded")
        start_ns = time.time_ns()
        text = await self.pipeline.model_instance.async_forward(prompt)
        end_ns = time.time_ns()
        return {
            "output": [],
            "text_output": text,
            "start_time_ns": start_ns,
            "end_time_ns": end_ns,
            "proc_time_ns": end_ns - start_ns,
            "swap_time_ns": 0,
            "decoder_time_ns": 0,
        }


# ---------------------------------------------------------------------------
# Backwards-compat alias (server.py imports SharedModelRuntime by name)
# ---------------------------------------------------------------------------
SharedModelRuntime = PyTorchRuntime
