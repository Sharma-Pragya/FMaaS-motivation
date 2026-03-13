import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from fmtk.logger import Logger
from device.model_loader import ModelLoader


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


# ---------------------------------------------------------------------------
# PyTorch runtime (TSFM backbones + MLP decoders)
# ---------------------------------------------------------------------------

class PyTorchRuntime(BaseRuntime):
    """Owns a ModelLoader and a single Logger. All operations — load, swap,
    add_decoder, run_batch — record into the same Logger automatically via
    the Pipeline (which holds the same Logger reference).

    Call self.logger.summary() for per-section averages, or .save() to persist.
    """

    def __init__(self, loader: ModelLoader | None = None):
        self._lock   = threading.RLock()
        self._loader = loader if loader is not None else ModelLoader()
        self.logger  = Logger(self._loader.device, "runtime")
        self._loader.logger = self.logger
        self.pipeline = None
        self.decoders = None

    def _sync(self):
        self.pipeline = self._loader.pipeline
        self.decoders = self._loader.decoders

    def load(self, backbone: str, decoders: list, **kwargs) -> Logger:
        with self._lock:
            op_log = self._loader.load_models(backbone, decoders)
            self._sync()
            return op_log

    def swap_backbone(self, backbone: str, decoders: list) -> Logger:
        with self._lock:
            op_log = self._loader.swap_backbone(backbone, decoders)
            self._sync()
            return op_log

    def add_decoders(self, decoders: list) -> Logger:
        with self._lock:
            op_log = self._loader.add_decoder(decoders)
            self._sync()
            return op_log

    def run_batch(
        self,
        x: np.ndarray,
        task_names: list[str],
        mask: np.ndarray | None = None,
    ) -> BatchRunResult:
        """Run one backbone pass + per-task decoder passes."""
        import torch
        import torch.nn as nn
        with self._lock:
            device   = self._loader.device
            start_ns = time.time_ns()

            bx     = torch.from_numpy(x)
            b_mask = torch.from_numpy(mask) if mask is not None else None

            with torch.no_grad():
                backbone_start = time.time_ns()
                feats = self.pipeline.model_instance.forward(bx, b_mask)
                proc_time_ns = time.time_ns() - backbone_start
                peak_bytes = torch.cuda.memory_allocated(device) if str(device).startswith("cuda") else 0

            outputs        = []
            swap_times     = []
            decoder_times  = []
            active_decoder = None
            current_task   = None

            for index, task_name in enumerate(task_names):
                swap_start = time.time_ns()
                if current_task != task_name:
                    current_task   = task_name
                    decoder_name   = self.decoders.get(task_name)
                    active_decoder = self.pipeline.decoders[decoder_name] if decoder_name else None
                swap_times.append(time.time_ns() - swap_start)

                dec_start = time.time_ns()
                feat_i = feats[index : index + 1]
                if active_decoder is not None:
                    with torch.no_grad():
                        logit_i = active_decoder.forward(feat_i)
                    dec_bytes = torch.cuda.memory_allocated(device) if str(device).startswith("cuda") else 0
                    if dec_bytes > peak_bytes:
                        peak_bytes = dec_bytes
                    if isinstance(active_decoder.criterion, nn.CrossEntropyLoss):
                        logit_i = torch.argmax(logit_i, dim=1)
                    if (
                        hasattr(active_decoder, "requires_model")
                        and active_decoder.requires_model
                        and hasattr(self.pipeline.model_instance.model, "normalizer")
                    ):
                        logit_i = self.pipeline.model_instance.model.normalizer(x=logit_i, mode="denorm")
                    result_i = logit_i.detach().cpu().numpy()
                else:
                    result_i = feat_i.detach().cpu().float().numpy()

                decoder_times.append(time.time_ns() - dec_start)
                outputs.append(result_i.reshape(-1))
            end_ns = time.time_ns()

            return BatchRunResult(
                outputs=outputs,
                start_time_ns=start_ns,
                end_time_ns=end_ns,
                proc_time_ns=proc_time_ns,
                swap_time_ns=swap_times,
                decoder_time_ns=decoder_times,
                gpu_alloc_peak_mb=peak_bytes / (1024 ** 2),
            )


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

    def load(self, backbone: str, decoders: list, device: str = "cuda:0",
             model_config: dict | None = None) -> Logger:
        self._loader.device = device
        self.logger = Logger(device, "runtime")
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
