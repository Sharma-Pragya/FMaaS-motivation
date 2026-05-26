import inspect
import threading
import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np

from fmtk.logger import Logger
from device.model_loader import ModelLoader


# ---------------------------------------------------------------------------
# Backbone capability detection
# ---------------------------------------------------------------------------

_FORWARD_ADAPTERS_CACHE: dict[int, bool] = {}


def _backbone_supports_adapters(forward_callable) -> bool:
    """True iff `forward(...)` accepts an `adapters` kwarg (or **kwargs).

    Used to decide between the multi-LoRA path (one batched forward with
    `adapters=[...]`) and the fallback path (per-adapter set_adapter +
    sub-batched forward). Inspecting once per backbone class is cheap;
    we cache by id() of the bound method.
    """
    key = id(forward_callable)
    cached = _FORWARD_ADAPTERS_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        sig = inspect.signature(forward_callable)
    except (TypeError, ValueError):
        _FORWARD_ADAPTERS_CACHE[key] = False
        return False
    params = sig.parameters
    has_adapters = "adapters" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    _FORWARD_ADAPTERS_CACHE[key] = has_adapters
    return has_adapters


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
    feats_by_idx: dict        # {orig_idx: feature tensor slice}
    task_names: list[str]
    start_time_ns: int
    backbone_end_time_ns: int
    proc_time_ns: int
    peak_bytes: int
    backbone_event: object | None = None  # CUDA event recorded after backbone forward,
                                          # for decoder streams to wait on


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

    def remove_decoders(self, task_names: list) -> Logger:
        """Remove decoders for the given tasks and free GPU memory.
        Default no-op; override in concrete runtimes that support it."""
        from fmtk.logger import Logger
        return Logger(None, "noop")


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
        self._lock   = threading.RLock()
        self._loader = loader if loader is not None else ModelLoader()
        self.logger  = Logger(self._loader.device, "runtime")
        self._loader.logger = self.logger
        self.pipeline = None
        self.decoders = None
        self.adapters = None
        # Always use a dedicated (non-default) CUDA stream to avoid implicit
        # synchronisation overhead of the default stream under concurrent load.
        if cuda_stream is None:
            import torch
            device = self._loader.device
            if str(device).startswith("cuda"):
                cuda_stream = torch.cuda.Stream(device=device)
        self._cuda_stream = cuda_stream

    def _sync(self):
        self.pipeline = self._loader.pipeline
        self.decoders = self._loader.decoders
        self.adapters = self._loader.adapters

    def load(self, backbone: str, decoders: list, **kwargs) -> Logger:
        with self._lock:
            op_log = self._loader.load_models(
                backbone, decoders,
                model_config=kwargs.get("model_config"),
            )
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

    def add_adapters(self, adapters: list) -> Logger:
        with self._lock:
            op_log = self._loader.add_adapter(adapters)
            self._sync()
            return op_log

    def remove_decoders(self, task_names: list) -> Logger:
        with self._lock:
            op_log = self._loader.remove_decoder(task_names)
            self._sync()
            return op_log

    def run_backbone(
        self,
        x: np.ndarray,
        task_names: list[str],
        mask: np.ndarray | None = None,
        questions: list[str | None] | None = None,
    ) -> BackboneResult:
        """Run encoder+backbone forward(s) only; return features per request.

        Adapter-grouping preserved from the original run_batch path.
        """
        import torch
        with self._lock:
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
            # Per-sample adapter list: one entry per request in the batch
            # (in task_names order). The backbone forward routes each sample
            # through its own LoRA in a single batched pass — no per-adapter
            # set_adapter / forward splitting.
            adapters_list = [adapters_map.get(task) for task in task_names]
            indices = list(range(len(task_names)))
            has_any_adapter = any(a is not None for a in adapters_list)
            # Backbones that declare an `adapters=` kwarg (e.g. moment.py)
            # take the multi-LoRA fused path. Backbones without it (e.g.
            # Qwen-VLM) fall back to per-adapter set_adapter + sub-batched
            # forwards. Backbones without LoRA at all just get plain forward.
            backbone_accepts_adapters = _backbone_supports_adapters(
                self.pipeline.model_instance.forward
            )
            use_fallback_split = has_any_adapter and not backbone_accepts_adapters
            # Only pass `adapters=` when at least one sample actually maps to
            # an adapter slot AND the backbone accepts the kwarg.
            forward_kwargs = (
                {"adapters": adapters_list}
                if (has_any_adapter and backbone_accepts_adapters)
                else {}
            )

            feats_by_idx: dict[int, object] = {}
            proc_time_ns = 0
            peak_bytes   = 0

            stream_ctx = torch.cuda.stream(self._cuda_stream) if self._cuda_stream else nullcontext()
            with stream_ctx, torch.no_grad():
                if use_fallback_split:
                    # Fallback path for backbones without `adapters=` support:
                    # split the batch by adapter, set_adapter / unload_adapter
                    # between groups, and run one sub-batched forward per
                    # group. Same semantics as before the multi-LoRA change.
                    groups: list[tuple[str | None, list[int]]] = []
                    for idx, a in enumerate(adapters_list):
                        if groups and groups[-1][0] == a:
                            groups[-1][1].append(idx)
                        else:
                            groups.append((a, [idx]))

                    for adapter_name, group_indices in groups:
                        if adapter_name is not None:
                            self.pipeline.set_adapter(adapter_name)
                        else:
                            self.pipeline.unload_adapter()

                        sub_mask = b_mask[group_indices] if b_mask is not None else None

                        bb_start = time.time_ns()
                        if questions is not None:
                            sub_q = [questions[i] for i in group_indices]
                            if bx is not None:
                                sub_x = bx[group_indices]
                                sub_feats = self.pipeline.model_instance.forward(
                                    (sub_x, sub_q), sub_mask
                                )
                            else:
                                sub_feats = self.pipeline.model_instance.forward(
                                    (None, sub_q), sub_mask
                                )
                        else:
                            sub_x = bx[group_indices]
                            sub_feats = self.pipeline.model_instance.forward(
                                sub_x, sub_mask
                            )
                        if is_cuda:
                            self._cuda_stream.synchronize() if self._cuda_stream else torch.cuda.synchronize(device)
                        proc_time_ns += time.time_ns() - bb_start

                        if is_cuda:
                            cur_bytes = torch.cuda.memory_allocated(device)
                            if cur_bytes > peak_bytes:
                                peak_bytes = cur_bytes

                        for out_pos, orig_idx in enumerate(group_indices):
                            feats_by_idx[orig_idx] = sub_feats[out_pos : out_pos + 1]
                else:
                    sub_mask = b_mask if b_mask is not None else None

                    bb_start = time.time_ns()
                    if questions is not None:
                        if bx is not None:
                            sub_feats = self.pipeline.model_instance.forward(
                                (bx, questions), sub_mask, **forward_kwargs,
                            )
                        else:
                            sub_feats = self.pipeline.model_instance.forward(
                                (None, questions), sub_mask, **forward_kwargs,
                            )
                    else:
                        sub_feats = self.pipeline.model_instance.forward(
                            bx, sub_mask, **forward_kwargs,
                        )
                    if is_cuda:
                        # Stream-local sync, not device-wide — lets decoder
                        # threads on OTHER streams keep running.
                        self._cuda_stream.synchronize() if self._cuda_stream else torch.cuda.synchronize(device)
                    proc_time_ns += time.time_ns() - bb_start

                    if is_cuda:
                        cur_bytes = torch.cuda.memory_allocated(device)
                        if cur_bytes > peak_bytes:
                            peak_bytes = cur_bytes

                    for out_pos, orig_idx in enumerate(indices):
                        feats_by_idx[orig_idx] = sub_feats[out_pos : out_pos + 1]

            # Record an event on the backbone stream so decoder streams can
            # wait on it without a CPU-side barrier.
            backbone_event = None
            if is_cuda and self._cuda_stream is not None:
                backbone_event = torch.cuda.Event()
                backbone_event.record(self._cuda_stream)

            return BackboneResult(
                feats_by_idx=feats_by_idx,
                task_names=list(task_names),
                start_time_ns=start_ns,
                backbone_end_time_ns=time.time_ns(),
                proc_time_ns=proc_time_ns,
                peak_bytes=peak_bytes,
                backbone_event=backbone_event,
            )

    def run_decoders(
        self,
        task_name: str,
        indices: list[int],
        feats: list,
        n_total: int,
        decoder_stream=None,
        backbone_event=None,
    ) -> tuple[list, list[int], list[int], int]:
        """Run the decoder for one task across the given feature slices.

        Returns (outputs_sparse, swap_times_sparse, decoder_times_sparse, peak_bytes)
        where *_sparse are length-n_total lists with entries only at `indices`.
        Intended to be called on a per-task worker thread.

        If `decoder_stream` is provided, the decoder runs on that stream
        (instead of the backbone stream). `backbone_event` is an event recorded
        on the backbone stream; the decoder stream will wait on it so backbone
        feature writes are visible before the decoder reads them.
        """
        import torch
        import torch.nn as nn

        device  = self._loader.device
        is_cuda = str(device).startswith("cuda")
        outputs       = [None] * n_total
        swap_times    = [0] * n_total
        decoder_times = [0] * n_total
        peak_bytes    = 0

        # Prefer the per-decoder stream when provided; fall back to the shared one.
        stream = decoder_stream if decoder_stream is not None else self._cuda_stream
        stream_ctx = torch.cuda.stream(stream) if stream else nullcontext()

        # Cross-stream dependency: decoder must not read features before the
        # backbone has finished writing them. wait_event is async (CPU-side
        # it just enqueues a wait on this stream).
        if is_cuda and stream is not None and backbone_event is not None:
            stream.wait_event(backbone_event)

        swap_start = time.time_ns()
        decoder_name = self.decoders.get(task_name)
        active_decoder = self.pipeline.decoders[decoder_name] if decoder_name else None
        swap_elapsed = time.time_ns() - swap_start
        for index in indices:
            swap_times[index] = swap_elapsed

        dec_start = time.time_ns()
        if active_decoder is not None:
            feat_input = torch.cat(feats, dim=0)
            with stream_ctx, torch.no_grad():
                logits = active_decoder.forward(feat_input)
                if isinstance(active_decoder.criterion, nn.CrossEntropyLoss):
                    logits = torch.argmax(logits, dim=1)
            # Stream-local sync — waits only for THIS decoder's kernels on its
            # own stream, not the whole device. Other decoder threads on their
            # own streams are unaffected.
            if is_cuda and stream is not None:
                stream.synchronize()
            result_batch = logits.detach().cpu().numpy()
            if result_batch.ndim == 0:
                result_batch = result_batch.reshape(1)
            batched_results = [np.asarray(result_batch[i:i + 1]) for i in range(len(indices))]
        else:
            batched_results = []
            for feat_i in feats:
                if isinstance(feat_i, (list, str)):
                    text = feat_i[0] if isinstance(feat_i, list) else feat_i
                    batched_results.append(np.array([text], dtype=object))
                else:
                    batched_results.append(feat_i.detach().cpu().float().numpy())

        dec_elapsed = time.time_ns() - dec_start
        per_item_dec_elapsed = dec_elapsed // max(len(indices), 1)
        for index, result_i in zip(indices, batched_results):
            decoder_times[index] = per_item_dec_elapsed
            outputs[index] = result_i.reshape(-1)

        return outputs, swap_times, decoder_times, peak_bytes

    def run_batch(
        self,
        x: np.ndarray,
        task_names: list[str],
        mask: np.ndarray | None = None,
        questions: list[str | None] | None = None,
    ) -> BatchRunResult:
        """Backbone + decoder on one thread (used by isolation_mode=none).

        The batcher path uses run_backbone + per-task run_decoders directly
        so the decoder stage can run on a task-owned thread.
        """
        bb = self.run_backbone(x, task_names, mask, questions)

        outputs       = [None] * len(task_names)
        swap_times    = [0] * len(task_names)
        decoder_times = [0] * len(task_names)
        peak_bytes    = bb.peak_bytes

        task_groups: dict[str, list[int]] = {}
        for index, task_name in enumerate(task_names):
            task_groups.setdefault(task_name, []).append(index)

        with self._lock:
            for task_name, indices in task_groups.items():
                feats = [bb.feats_by_idx[i] for i in indices]
                outs, swaps, decs, pb = self.run_decoders(task_name, indices, feats, len(task_names))
                for i in indices:
                    outputs[i] = outs[i]
                    swap_times[i] = swaps[i]
                    decoder_times[i] = decs[i]
                if pb > peak_bytes:
                    peak_bytes = pb

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
        # Multi-LoRA: if any task spec requests an adapter, flip enable_lora on
        # the vLLM engine so per-request LoRARequest routing works downstream.
        if any((spec or {}).get("adapter") for spec in (decoders or [])):
            mc = dict(model_config or {})
            mc.setdefault("enable_lora", True)
            mc.setdefault("max_loras", max(4, len(decoders)))
            mc.setdefault("max_lora_rank", 64)
            model_config = mc
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
        """Hot-add LoRA adapters to a loaded vLLM backbone. The engine must
        have been built with enable_lora=True (set automatically when adapter
        specs were present at load time). Each spec:
            {"task": "...", "adapter": "lora", "path": "<adapter-dir-name>"}
        """
        return self._loader.add_adapter(adapters)

    async def infer(self, req_id: int, prompt: str, task: str | None = None) -> dict:
        """Single prompt → generated text. Multiple concurrent calls are
        batched at the iteration level by vLLM's AsyncLLMEngine — true
        continuous batching with no extra logic needed here. If `task` maps
        to a registered LoRA adapter, the request is routed through it."""
        if self.pipeline is None:
            raise RuntimeError("vllm_model_not_loaded")
        adapter_name = None
        if task and self._loader.adapters:
            adapter_name = self._loader.adapters.get(task)
        start_ns = time.time_ns()
        text = await self.pipeline.model_instance.async_forward(
            prompt, adapter_name=adapter_name,
        )
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
