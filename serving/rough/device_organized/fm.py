"""
fm.py — Foundation Model: pure synchronous model manager.

Owns model state only — pipeline, decoders, backbone name.
Has no threads, no asyncio, no events.

All methods are synchronous and called from exactly one place:
  - load / add_decoder / swap_backbone  → called by fmapi via asyncio.to_thread
  - run_batch                           → called by fmvisor's worker thread

Because fmapi's lifecycle calls go through asyncio.to_thread (a thread pool)
and fmvisor's run_batch goes through the persistent worker thread, and these
never run concurrently (fmapi holds the model lock during lifecycle ops),
no internal locking is needed here.
"""

import gc
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from device.config import DEVICE, DECODERS
from fmtk.pipeline import Pipeline
from fmtk.components.decoders.regression.mlp import MLPDecoder as RegressionMLP
from fmtk.components.decoders.classification.mlp import MLPDecoder as ClassificationMLP
from fmtk.components.decoders.forecasting.mlp import MLPDecoder as ForecastingMLP
from fmtk.logger import Logger


@dataclass(frozen=True)
class BatchResult:
    """
    Result of one run_batch() call.

    Returned to fmvisor's worker thread, which splits it per-request
    and resolves each RequestEnvelope's asyncio future.

    Fields:
      outputs         — concatenated outputs, shape (batch_size, ...)
      start_time_ns   — ns when backbone forward() started
      end_time_ns     — ns when last decoder finished
      proc_time_ns    — time in backbone forward() only
      swap_time_ns    — per-request decoder lookup time
      decoder_time_ns — per-request decoder forward() time
    """
    outputs: np.ndarray
    start_time_ns: int
    end_time_ns: int
    proc_time_ns: int
    swap_time_ns: list[int]
    decoder_time_ns: list[int]


class FoundationModel:
    """
    Manages the FM as a shared model resource.

    Pure synchronous — no threads, no asyncio.
    The caller is responsible for ensuring mutual exclusion:
      - fmvisor's worker thread calls run_batch() exclusively during inference
      - fmapi calls lifecycle ops via asyncio.to_thread, protected by _model_lock

    Methods:
      load(backbone, decoders)       — cold-start: load backbone + decoders onto GPU
      add_decoder(decoder_specs)     — hot-add decoders to running backbone
      swap_backbone(backbone, decoders) — free old backbone, load new one
      run_batch(batch)               — run backbone forward + per-task decoder passes
    """

    def __init__(self):
        self._pipeline: Pipeline | None = None
        self._decoders: dict[str, str] = {}
        self._backbone_name: str | None = None
        self._model_lock = threading.RLock()
        # Set by load() when the pipeline is ready for inference.
        # run_batch() blocks on this so requests arriving before load completes
        # are held rather than rejected — matches old runtime.py behavior.
        self._ready = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle — called by fmapi via asyncio.to_thread
    # ------------------------------------------------------------------

    def load(self, backbone: str, decoders: list[dict]) -> Logger:
        """
        Cold-start: load a backbone and its decoders onto the GPU.

        Args:
          backbone — backbone name (e.g. "momentlarge")
          decoders — list of {"task": str, "type": str, "path": str}

        Returns:
          Logger with timing measurements.
        """
        with self._model_lock:
            return self._load(backbone, decoders)

    def add_decoder(self, decoder_specs: list[dict]) -> Logger:
        """
        Hot-add new decoders to the already-loaded backbone (~0.3s).

        Does not reload the backbone. Raises RuntimeError if no backbone loaded.

        Args:
          decoder_specs — list of {"task": str, "type": str, "path": str}

        Returns:
          Logger with timing measurements.
        """
        with self._model_lock:
            if self._pipeline is None or self._backbone_name is None:
                raise RuntimeError("No backbone loaded. Call load() first.")
            logger = Logger(DEVICE, 'add_decoder_logger')
            for dec in decoder_specs:
                with logger.measure("add_decoder", device=DEVICE):
                    self._attach_decoder(dec)
            print(f"[FM] Hot-added {len(decoder_specs)} decoder(s). Total: {len(self._decoders)}")
            return logger

    def swap_backbone(self, backbone: str, decoders: list[dict]) -> Logger:
        """
        Free old backbone from GPU and load a new one in-place.

        Args:
          backbone — new backbone name (e.g. "chronosbase")
          decoders — list of {"task": str, "type": str, "path": str}

        Returns:
          Logger with timing measurements from the load operation.
        """
        with self._model_lock:
            self._ready.clear()   # block run_batch during the swap
            if self._pipeline is not None:
                print(f"[FM] Releasing backbone '{self._backbone_name}' from GPU...")
                del self._pipeline
                self._pipeline = None
            self._decoders = {}
            self._backbone_name = None
            torch.cuda.empty_cache()
            gc.collect()
            print("[FM] GPU memory released. Loading new backbone...")
            return self._load(backbone, decoders)

    # ------------------------------------------------------------------
    # Inference — called by fmvisor's worker thread
    # ------------------------------------------------------------------

    def run_batch(self, x: np.ndarray, task_names: list[str], mask: np.ndarray | None) -> BatchResult:
        """
        Run one batch through the shared backbone and per-task decoders.

        Called synchronously by fmvisor's persistent worker thread.
        The backbone runs once for the whole batch (shared forward pass).
        Each request then passes through its task-specific decoder.

        Args:
          x          — concatenated input tensors, shape (batch_size, ...)
          task_names — task name per request, parallel to x
          mask       — concatenated masks or None

        Returns:
          BatchResult with per-request outputs and timing breakdowns.

        Raises:
          RuntimeError("pipeline_not_loaded") if no model loaded yet.
        """
        # Block until load() or swap_backbone() signals the pipeline is ready.
        # This handles requests arriving before the first load completes.
        self._ready.wait()

        with self._model_lock:
            if self._pipeline is None or not self._decoders:
                raise RuntimeError("pipeline_not_loaded")

            start_ns = time.time_ns()
            bx = torch.from_numpy(x)
            b_mask = torch.from_numpy(mask) if mask is not None else None

            feats = self._pipeline.model_instance.forward(bx, b_mask)
            proc_time_ns = time.time_ns() - start_ns

            outputs, swap_times, decoder_times = [], [], []
            active_decoder = None
            current_task = None

            for i, task_name in enumerate(task_names):
                swap_start = time.time_ns()
                if current_task != task_name:
                    current_task = task_name
                    decoder_name = self._decoders.get(task_name)
                    active_decoder = self._pipeline.decoders[decoder_name] if decoder_name else None
                swap_times.append(time.time_ns() - swap_start)

                decoder_start = time.time_ns()
                feat_i = feats[i: i + 1]
                if active_decoder is not None:
                    logit_i = active_decoder.forward(feat_i)
                    if isinstance(active_decoder.criterion, nn.CrossEntropyLoss):
                        logit_i = torch.argmax(logit_i, dim=1)
                    if (
                        hasattr(active_decoder, "requires_model")
                        and active_decoder.requires_model
                        and hasattr(self._pipeline.model_instance.model, "normalizer")
                    ):
                        logit_i = self._pipeline.model_instance.model.normalizer(x=logit_i, mode="denorm")
                    result_i = logit_i.detach().cpu().numpy()
                else:
                    embeddings = self._pipeline.model_instance.forward((feat_i, None))
                    result_i = self._pipeline.model_instance.postprocess(embeddings)
                outputs.append(result_i)
                decoder_times.append(time.time_ns() - decoder_start)

            end_ns = time.time_ns()
            output_array = np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.float32)
            return BatchResult(
                outputs=output_array,
                start_time_ns=start_ns,
                end_time_ns=end_ns,
                proc_time_ns=proc_time_ns,
                swap_time_ns=swap_times,
                decoder_time_ns=decoder_times,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, backbone: str, decoders: list[dict]) -> Logger:
        logger = Logger(DEVICE, 'deploymentlogger')
        print(f"[FM] Loading backbone: {backbone}")
        with logger.measure("load_backbone", device=DEVICE):
            self._pipeline = self._build_backbone(backbone, logger)
        self._backbone_name = backbone
        self._decoders = {}
        for dec in decoders:
            self._attach_decoder(dec)
        print(f"[FM] Loaded {self._pipeline.model_instance.__class__.__name__} "
              f"with {len(self._decoders)} decoders.")
        self._ready.set()
        return logger

    def _attach_decoder(self, dec: dict):
        task, dtype, path = dec["task"], dec["type"], dec["path"]
        backbone = self._backbone_name
        print(f"[FM] Attaching decoder: {task} ({dtype}) from {path}")
        if dtype == "regression":
            cfg = DECODERS[f'mlp_{backbone}_{dtype}']['decoder_config']
            decoder = RegressionMLP(device=cfg['DEVICE'], cfg=cfg['cfg'])
        elif dtype == "classification":
            cfg = DECODERS[f'mlp_{backbone}_{task}']['decoder_config']
            decoder = ClassificationMLP(device=cfg['DEVICE'], cfg=cfg['cfg'])
        elif dtype == "forecasting":
            cfg = DECODERS[f'mlp_{backbone}_{dtype}']['decoder_config']
            decoder = ForecastingMLP(device=cfg['DEVICE'], cfg=cfg['cfg'])
        else:
            raise ValueError(f"Unknown decoder type: {dtype}")
        self._decoders[task] = self._pipeline.add_decoder(decoder, load=True, train=False, path=path)

    def _build_backbone(self, backbone: str, logger: Logger) -> Pipeline:
        if backbone == "momentlarge":
            from fmtk.components.backbones.moment import MomentModel
            return Pipeline(MomentModel(DEVICE, "large"), logger=logger)
        elif backbone == "momentsmall":
            from fmtk.components.backbones.moment import MomentModel
            return Pipeline(MomentModel(DEVICE, "small"), logger=logger)
        elif backbone == "momentbase":
            from fmtk.components.backbones.moment import MomentModel
            return Pipeline(MomentModel(DEVICE, "base"), logger=logger)
        elif backbone == "chronostiny":
            from fmtk.components.backbones.chronos import ChronosModel
            return Pipeline(ChronosModel(DEVICE, "tiny"), logger=logger)
        elif backbone == "chronosmini":
            from fmtk.components.backbones.chronos import ChronosModel
            return Pipeline(ChronosModel(DEVICE, "mini"), logger=logger)
        elif backbone == "chronossmall":
            from fmtk.components.backbones.chronos import ChronosModel
            return Pipeline(ChronosModel(DEVICE, "small"), logger=logger)
        elif backbone == "chronosbase":
            from fmtk.components.backbones.chronos import ChronosModel
            return Pipeline(ChronosModel(DEVICE, "base"), logger=logger)
        elif backbone == "chronoslarge":
            from fmtk.components.backbones.chronos import ChronosModel
            return Pipeline(ChronosModel(DEVICE, "large"), logger=logger)
        elif backbone == "papageis":
            from fmtk.components.backbones.papagei import PapageiModel
            cfg = {'in_channels': 1, 'base_filters': 32, 'kernel_size': 3, 'stride': 2,
                   'groups': 1, 'n_block': 18, 'n_classes': 512, 'n_experts': 3}
            return Pipeline(PapageiModel(DEVICE, "papagei_s", model_config=cfg), logger=logger)
        elif backbone == "papageip":
            from fmtk.components.backbones.papagei import PapageiModel
            cfg = {'in_channels': 1, 'base_filters': 32, 'kernel_size': 3, 'stride': 2,
                   'groups': 1, 'n_block': 18, 'n_classes': 512}
            return Pipeline(PapageiModel(DEVICE, "papagei_p", model_config=cfg), logger=logger)
        elif backbone == "papageissvri":
            from fmtk.components.backbones.papagei import PapageiModel
            cfg = {'in_channels': 1, 'base_filters': 32, 'kernel_size': 3, 'stride': 2,
                   'groups': 1, 'n_block': 18, 'n_classes': 512}
            return Pipeline(PapageiModel(DEVICE, "papagei_s_svri", model_config=cfg), logger=logger)
        elif backbone == "llava":
            from fmtk.components.backbones.llava import LlavaModel
            return Pipeline(LlavaModel(DEVICE, "llava-1.5-7b-hf"), logger=logger)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
