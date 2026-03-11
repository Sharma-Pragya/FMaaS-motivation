#!/usr/bin/env python3
"""motivation1/server.py — In-process serving infrastructure.

Two strategies exposed via InferenceServer:

  task_sharing   — N separate backbones, one per task. Each call is
                   batch_size=1 (no other task to batch with).

  deploy_sharing — 1 backbone shared by all N tasks. A threading.Barrier
                   ensures all N threads arrive together → one GPU forward
                   pass per round. Guaranteed batch_size=N.

Uses device/model_loader.ModelLoader and device/runtime.SharedModelRuntime.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import numpy as np
import torch

from device.model_loader import ModelLoader
from device.runtime import SharedModelRuntime


# Task name → decoder type (matches device/config.py DECODERS keys)
TASK_TYPES: Dict[str, str] = {
    "ecgclass":     "classification",
    "gestureclass": "classification",
    "sysbp":        "regression",
    "diasbp":       "regression",
    "heartrate":    "regression",
    "eclfore":      "forecasting",
    "etth1fore":    "forecasting",
    "exchangefore": "forecasting",
    "trafficfore":  "forecasting",
    "weatherfore":  "forecasting",
}


# ---------------------------------------------------------------------------
# _SharedBackbone — barrier-based batching over a SharedModelRuntime
# ---------------------------------------------------------------------------

class _SharedBackbone:
    def __init__(self, backbone: str, device: torch.device, n_tasks: int):
        self.backbone = backbone
        self.device = device
        self.n_tasks = n_tasks

        self._barrier = threading.Barrier(n_tasks)
        self._slot_lock = threading.Lock()
        self._inputs:  List[Optional[tuple]]      = [None] * n_tasks
        self._outputs: List[Optional[np.ndarray]] = [None] * n_tasks
        self._slot_index: Dict[int, int] = {}
        self._slot_counter = 0

        self.batch_count = 0
        self.batch_size_sum = 0
        self.mixed_batch_count = 0
        self.peak_gpu_mb_sum = 0.0

        self._runtime: Optional[SharedModelRuntime] = None

    def load(self, decoder_specs: List[dict]):
        loader = ModelLoader(device=self.device)
        self._runtime = SharedModelRuntime(loader=loader)   # Logger created here
        self._runtime.load(self.backbone, decoder_specs)    # logger passed into loader
        print(f"[Backbone] {self.backbone} on {self.device} "
              f"(n_tasks={self.n_tasks}, decoders={len(loader.decoders)})")

    def _get_slot(self) -> int:
        tid = threading.get_ident()
        with self._slot_lock:
            if tid not in self._slot_index:
                self._slot_index[tid] = self._slot_counter
                self._slot_counter += 1
            return self._slot_index[tid]

    def infer(self, task: str, x: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        slot = self._get_slot()
        self._inputs[slot] = (task, x, mask)

        arrival = self._barrier.wait()
        if arrival == 0:
            self._run_batch()
        self._barrier.wait()

        result = self._outputs[slot]
        self._inputs[slot] = None
        self._outputs[slot] = None
        return result

    def _run_batch(self):
        inputs = self._inputs[:self.n_tasks]
        tasks  = [inp[0] for inp in inputs]
        xs     = [inp[1] for inp in inputs]
        masks  = [inp[2] for inp in inputs]

        x_batch    = np.concatenate(xs, axis=0)
        mask_batch = np.concatenate(masks, axis=0) if all(m is not None for m in masks) else None

        result = self._runtime.run_batch(x_batch, tasks, mask_batch)
        for i, out in enumerate(result.outputs):
            self._outputs[i] = out

        self.batch_count      += 1
        self.batch_size_sum   += self.n_tasks
        self.peak_gpu_mb_sum  += result.gpu_alloc_peak_mb
        if len(set(tasks)) > 1:
            self.mixed_batch_count += 1

    def stats(self) -> Dict:
        n = self.batch_count
        return {
            "batch_count":          n,
            "avg_batch_size":       (self.batch_size_sum / n) if n else 0.0,
            "mixed_batch_fraction": (self.mixed_batch_count / n) if n else 0.0,
            "avg_peak_gpu_mb":      (self.peak_gpu_mb_sum / n) if n else 0.0,
        }


# ---------------------------------------------------------------------------
# InferenceServer
# ---------------------------------------------------------------------------

class InferenceServer:
    """
    strategy="task_sharing"   → N _SharedBackbone instances, each n_tasks=1.
    strategy="deploy_sharing" → 1 _SharedBackbone with n_tasks=N.

    decoder_dir: root dir for finetuned decoder checkpoints.
                 Expects: {decoder_dir}/{task}_{backbone}_mlp/decoder.pth
                 Pass None for backbone-only inference.
    """

    def __init__(
        self,
        strategy: str,
        tasks: List[str],
        backbone: str,
        device: str = "cuda:0",
        decoder_dir: Optional[str] = None,
    ):
        if strategy not in ("task_sharing", "deploy_sharing"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy     = strategy
        self.tasks        = tasks
        self.backbone     = backbone
        self.decoder_dir  = decoder_dir

        self._dev = torch.device(device)
        self._backbones:     List[_SharedBackbone]       = []
        self._task_backbone: Dict[str, _SharedBackbone]  = {}
        self._model_load_mem_mb: float = 0.0
        self._peak_mem_mb:       float = 0.0

    def _decoder_specs(self, tasks: List[str]) -> List[dict]:
        if not self.decoder_dir:
            return []
        return [
            {"task": t, "type": TASK_TYPES[t], "path": f"{t}_{self.backbone}_mlp"}
            for t in tasks
        ]

    def start(self):
        """Synchronous — call from main thread before launching worker threads."""
        if self._dev.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass

        if self.strategy == "deploy_sharing":
            bb = _SharedBackbone(self.backbone, self._dev, len(self.tasks))
            bb.load(self._decoder_specs(self.tasks))
            self._backbones = [bb]
            for task in self.tasks:
                self._task_backbone[task] = bb
        else:
            for task in self.tasks:
                bb = _SharedBackbone(self.backbone, self._dev, 1)
                bb.load(self._decoder_specs([task]))
                self._backbones.append(bb)
                self._task_backbone[task] = bb

        # Read total GPU allocation after all loads complete (not per-backbone,
        # since each backbone's logger peak includes all previously loaded backbones)
        if self._dev.type == "cuda":
            self._model_load_mem_mb = torch.cuda.memory_allocated(self._dev) / (1024 ** 2)
            self._peak_mem_mb       = torch.cuda.max_memory_allocated(self._dev) / (1024 ** 2)

        mode = "backbone+decoders" if self.decoder_dir else "backbone-only"
        print(f"[Server:{self.strategy}] {len(self._backbones)} backbone(s), "
              f"{len(self.tasks)} task(s), {mode}. "
              f"load_mem={self._model_load_mem_mb:.0f} MB")

    def stop(self):
        self._backbones.clear()
        self._task_backbone.clear()
        if self._dev.type == "cuda":
            torch.cuda.empty_cache()

    def infer(self, task: str, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self._task_backbone[task].infer(task, x, mask)

    def model_load_mem_mb(self) -> float:
        return self._model_load_mem_mb

    def peak_gpu_mem_mb(self) -> float:
        return self._peak_mem_mb

    def avg_inference_gpu_mem_mb(self) -> float:
        """Average peak GPU memory during inference. All backbones share the same
        GPU so memory_allocated already reflects total — just use any one backbone."""
        for bb in self._backbones:
            n = bb.batch_count
            if n > 0:
                return bb.peak_gpu_mb_sum / n
        return 0.0

    def batch_stats(self) -> Dict:
        total_batches = total_items = total_mixed = 0
        for bb in self._backbones:
            s = bb.stats()
            n = s["batch_count"]
            total_batches += n
            total_items   += s["avg_batch_size"] * n
            total_mixed   += s["mixed_batch_fraction"] * n
        return {
            "batch_count":          total_batches,
            "avg_batch_size":       (total_items  / total_batches) if total_batches else 0.0,
            "mixed_batch_fraction": (total_mixed  / total_batches) if total_batches else 0.0,
        }
