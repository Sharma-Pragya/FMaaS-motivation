import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from device.model_loader import add_decoder, get_loaded_pipeline, load_models, swap_backbone


@dataclass(frozen=True)
class BatchRunResult:
    outputs: np.ndarray
    start_time_ns: int
    end_time_ns: int
    proc_time_ns: int
    swap_time_ns: list[int]
    decoder_time_ns: list[int]


class SharedModelRuntime:
    """Owns the single shared backbone/decoder state on a device."""

    def __init__(self):
        self._lock = threading.RLock()
        self.pipeline = None
        self.decoders = None

    def _refresh_loaded_locked(self):
        self.pipeline, self.decoders, _ = get_loaded_pipeline()
        if self.pipeline is None or self.decoders is None:
            raise RuntimeError("pipeline_not_loaded")

    def ensure_loaded(self):
        with self._lock:
            if self.pipeline is None or self.decoders is None:
                self._refresh_loaded_locked()

    def load(self, backbone: str, decoders: list[dict]):
        with self._lock:
            logger = load_models(backbone, decoders)
            self._refresh_loaded_locked()
            return logger

    def swap_backbone(self, backbone: str, decoders: list[dict]):
        with self._lock:
            logger = swap_backbone(backbone, decoders)
            self._refresh_loaded_locked()
            return logger

    def add_decoders(self, decoders: list[dict]):
        with self._lock:
            logger = add_decoder(decoders)
            self._refresh_loaded_locked()
            return logger

    def run_batch(
        self,
        x: np.ndarray,
        task_names: list[str],
        mask: np.ndarray | None = None,
    ) -> BatchRunResult:
        """Run one shared backbone pass and task-specific decoder passes."""
        with self._lock:
            if self.pipeline is None or self.decoders is None:
                self._refresh_loaded_locked()

            pipeline = self.pipeline
            decoders = self.decoders

            start_ns = time.time_ns()
            bx = torch.from_numpy(x)
            b_mask = torch.from_numpy(mask) if mask is not None else None
            feats = pipeline.model_instance.forward(bx, b_mask)
            proc_time_ns = time.time_ns() - start_ns

            outputs = []
            swap_times = []
            decoder_times = []
            active_decoder = None
            current_task = None

            for index, task_name in enumerate(task_names):
                swap_start = time.time_ns()
                if current_task != task_name:
                    current_task = task_name
                    decoder_name = decoders.get(task_name)
                    active_decoder = pipeline.decoders[decoder_name] if decoder_name else None
                swap_times.append(time.time_ns() - swap_start)

                decoder_start = time.time_ns()
                feat_i = feats[index : index + 1]
                if active_decoder is not None:
                    logit_i = active_decoder.forward(feat_i)
                    if isinstance(active_decoder.criterion, nn.CrossEntropyLoss):
                        logit_i = torch.argmax(logit_i, dim=1)
                    if (
                        hasattr(active_decoder, "requires_model")
                        and active_decoder.requires_model
                        and hasattr(pipeline.model_instance.model, "normalizer")
                    ):
                        logit_i = pipeline.model_instance.model.normalizer(x=logit_i, mode="denorm")
                    result_i = logit_i.detach().cpu().numpy()
                else:
                    embeddings = pipeline.model_instance.forward((feat_i, None))
                    result_i = pipeline.model_instance.postprocess(embeddings)

                outputs.append(result_i)
                decoder_times.append(time.time_ns() - decoder_start)

            end_ns = time.time_ns()
            output_array = np.concatenate([o.reshape(-1) for o in outputs], axis=0) if outputs else np.empty((0,), dtype=np.float32)
            return BatchRunResult(
                outputs=output_array,
                start_time_ns=start_ns,
                end_time_ns=end_ns,
                proc_time_ns=proc_time_ns,
                swap_time_ns=swap_times,
                decoder_time_ns=decoder_times,
            )
