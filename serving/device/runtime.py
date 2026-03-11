import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from fmtk.logger import Logger
from device.model_loader import ModelLoader


@dataclass
class BatchRunResult:
    outputs: list             # list[np.ndarray] per sample, each flattened (variable size ok)
    start_time_ns: int
    end_time_ns: int
    proc_time_ns: int         # backbone forward time (ns)
    swap_time_ns: list[int]   # per-task decoder lookup time (ns)
    decoder_time_ns: list[int]# per-task decoder forward time (ns)
    gpu_alloc_peak_mb: float  # peak GPU memory allocated during this batch (MB)


class SharedModelRuntime:
    """Owns a ModelLoader and a single Logger. All operations — load, swap,
    add_decoder, run_batch — record into the same Logger automatically via
    the Pipeline (which holds the same Logger reference).

    Call self.logger.summary() for per-section averages, or .save() to persist.
    """

    def __init__(self, loader: ModelLoader | None = None):
        self._lock  = threading.RLock()
        # If no loader passed (device/server.py legacy path), create a default one
        self._loader = loader if loader is not None else ModelLoader()
        self.logger  = Logger(self._loader.device, "runtime")
        # Give the loader our logger so Pipeline records into it automatically
        self._loader.logger = self.logger
        self.pipeline = None
        self.decoders = None

    def _sync(self):
        self.pipeline = self._loader.pipeline
        self.decoders = self._loader.decoders

    def load(self, backbone: str, decoders: list[dict]) -> Logger:
        with self._lock:
            op_log = self._loader.load_models(backbone, decoders)
            self._sync()
            return op_log

    def swap_backbone(self, backbone: str, decoders: list[dict]) -> Logger:
        with self._lock:
            op_log = self._loader.swap_backbone(backbone, decoders)
            self._sync()
            return op_log

    def add_decoders(self, decoders: list[dict]) -> Logger:
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
        with self._lock:
            device   = self._loader.device
            start_ns = time.time_ns()

            bx     = torch.from_numpy(x)
            b_mask = torch.from_numpy(mask) if mask is not None else None

            with torch.no_grad():
                backbone_start = time.time_ns()
                feats = self.pipeline.model_instance.forward(bx, b_mask)
                proc_time_ns = time.time_ns() - backbone_start
                # weights + input + feats all live — backbone inference working set
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
                    # weights + feats + decoder activations all live — take max
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
