#!/usr/bin/env python3
"""motivation1/server.py — In-process serving infrastructure.

Two strategies exposed via InferenceServer:

  task_sharing   — N separate _SharedBackbone instances (one per task).
                   Each task has its own backbone copy in GPU memory.
                   Each call is batch_size=1 (no other task to batch with).

  deploy_sharing — 1 _SharedBackbone shared by all N tasks.
                   Uses a threading.Barrier so all N task threads arrive
                   together, get batched into one GPU forward pass, then
                   all return simultaneously. Guaranteed batch_size=N,
                   mixed_batch_fraction=1.0.

Backbone-only inference: no decoders. forward() produces feature embeddings.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Backbone pipeline factory
# ---------------------------------------------------------------------------

def _build_pipeline(backbone: str, device: torch.device):
    """Instantiate a fresh Pipeline for `backbone` on `device`."""
    from fmtk.pipeline import Pipeline

    if backbone in ("chronosbase", "chronossmall", "chronostiny", "chronosmini", "chronoslarge"):
        size = backbone.replace("chronos", "")
        from fmtk.components.backbones.chronos import ChronosModel
        return Pipeline(ChronosModel(device, size))
    elif backbone in ("momentbase", "momentsmall", "momentlarge"):
        size = backbone.replace("moment", "")
        from fmtk.components.backbones.moment import MomentModel
        return Pipeline(MomentModel(device, size))
    elif backbone == "papageis":
        from fmtk.components.backbones.papagei import PapageiModel
        cfg = {"in_channels": 1, "base_filters": 32, "kernel_size": 3, "stride": 2,
               "groups": 1, "n_block": 18, "n_classes": 512, "n_experts": 3}
        return Pipeline(PapageiModel(device, "papagei_s", model_config=cfg))
    elif backbone == "papageip":
        from fmtk.components.backbones.papagei import PapageiModel
        cfg = {"in_channels": 1, "base_filters": 32, "kernel_size": 3, "stride": 2,
               "groups": 1, "n_block": 18, "n_classes": 512}
        return Pipeline(PapageiModel(device, "papagei_p", model_config=cfg))
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")


# ---------------------------------------------------------------------------
# _SharedBackbone
#
# Shared by `n_tasks` worker threads. Uses a Barrier to synchronise:
#   1. All n_tasks threads call infer() and submit their (task, x, mask).
#   2. The last thread to arrive assembles the batch and runs the GPU forward.
#   3. All threads pick up their result and return.
#
# For task_sharing n_tasks=1, so batch_size=1 always.
# For deploy_sharing n_tasks=N, so batch_size=N always.
# ---------------------------------------------------------------------------

class _SharedBackbone:
    def __init__(self, backbone: str, device: torch.device, n_tasks: int):
        self.backbone = backbone
        self.device = device
        self.n_tasks = n_tasks
        self.pipeline = None

        # Barrier synchronises n_tasks threads per round
        self._barrier = threading.Barrier(n_tasks)

        # Shared slot for the current round's inputs / outputs
        self._lock = threading.Lock()
        self._inputs: List[Optional[tuple]] = [None] * n_tasks   # (task, x, mask)
        self._outputs: List[Optional[np.ndarray]] = [None] * n_tasks
        self._slot_index: Dict[int, int] = {}  # thread_id → slot index
        self._slot_counter = 0

        # Stats
        self.batch_count = 0
        self.batch_size_sum = 0
        self.mixed_batch_count = 0

    def load(self):
        self.pipeline = _build_pipeline(self.backbone, self.device)
        print(f"[Backbone] {self.backbone} loaded on {self.device} (n_tasks={self.n_tasks})")

    def _get_slot(self) -> int:
        tid = threading.get_ident()
        with self._lock:
            if tid not in self._slot_index:
                idx = self._slot_counter
                self._slot_index[tid] = idx
                self._slot_counter += 1
            return self._slot_index[tid]

    def infer(self, task: str, x: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Called from worker thread. Returns features after the batched forward pass.
        All n_tasks threads must call this simultaneously (barrier-synchronized).
        """
        slot = self._get_slot()
        self._inputs[slot] = (task, x, mask)

        # Wait for all n_tasks threads to submit their input
        arrival = self._barrier.wait()

        # The last thread to arrive (arrival == 0 in Python's Barrier) runs the batch
        if arrival == 0:
            self._run_batch()

        # Wait for the batch result to be written
        self._barrier.wait()

        result = self._outputs[slot]
        self._inputs[slot] = None
        self._outputs[slot] = None
        return result

    def _run_batch(self):
        inputs = self._inputs[:self.n_tasks]
        tasks = [inp[0] for inp in inputs]
        xs    = [inp[1] for inp in inputs]
        masks = [inp[2] for inp in inputs]

        # All shapes must match for concatenation (guaranteed by same dataset)
        x_batch = np.concatenate(xs, axis=0)
        mask_batch = (np.concatenate(masks, axis=0)
                      if all(m is not None for m in masks) else None)

        # Pass CPU tensors — ChronosModel.forward moves internals to device itself
        bx = torch.from_numpy(x_batch)
        bm = torch.from_numpy(mask_batch) if mask_batch is not None else None

        with torch.no_grad():
            feats = self.pipeline.model_instance.forward(bx, bm)

        feats_np = feats.detach().cpu().float().numpy()  # (n_tasks, feat_dim...)

        for i in range(self.n_tasks):
            self._outputs[i] = feats_np[i]

        # Track stats
        n = self.n_tasks
        self.batch_count += 1
        self.batch_size_sum += n
        if len(set(tasks)) > 1:
            self.mixed_batch_count += 1

    def stats(self) -> Dict:
        n = self.batch_count
        return {
            "batch_count": n,
            "avg_batch_size": (self.batch_size_sum / n) if n else 0.0,
            "mixed_batch_fraction": (self.mixed_batch_count / n) if n else 0.0,
        }


# ---------------------------------------------------------------------------
# InferenceServer
# ---------------------------------------------------------------------------

class InferenceServer:
    """
    strategy="task_sharing"   → N _SharedBackbone instances, each with n_tasks=1.
                                N separate backbone copies in GPU memory.
                                batch_size=1, mixed_batch_fraction=0 always.

    strategy="deploy_sharing" → 1 _SharedBackbone with n_tasks=N.
                                1 backbone copy in GPU memory.
                                batch_size=N, mixed_batch_fraction=1 always.
    """

    def __init__(
        self,
        strategy: str,
        tasks: List[str],
        backbone: str,
        device: str = "cuda:0",
        max_batch_size: int = 32,   # kept for API compat, unused (barrier controls batch size)
        max_batch_wait_ms: float = 10.0,  # kept for API compat, unused
    ):
        if strategy not in ("task_sharing", "deploy_sharing"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
        self.tasks = tasks
        self.backbone = backbone
        self.device = device

        self._dev = torch.device(device)
        self._backbones: List[_SharedBackbone] = []
        self._task_backbone: Dict[str, _SharedBackbone] = {}
        self._peak_mem_mb: float = 0.0

    def _gpu_mem_mb(self) -> float:
        if self._dev.type != "cuda":
            return 0.0
        try:
            return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Memory logging
    # ------------------------------------------------------------------

    def start_memory_logging(self, log_path: str, interval_s: float = 1.0):
        """Start a background thread that samples current GPU memory every `interval_s` seconds."""
        import csv as _csv
        self._mem_log_stop = threading.Event()
        self._mem_log_path = log_path

        def _loop():
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "w", newline="") as f:
                writer = _csv.writer(f)
                writer.writerow(["elapsed_s", "allocated_mb", "reserved_mb"])
                t0 = time.time()
                while not self._mem_log_stop.is_set():
                    if self._dev.type == "cuda":
                        alloc = torch.cuda.memory_allocated(self._dev) / (1024 ** 2)
                        resrv = torch.cuda.memory_reserved(self._dev) / (1024 ** 2)
                    else:
                        alloc = resrv = 0.0
                    writer.writerow([f"{time.time() - t0:.2f}", f"{alloc:.1f}", f"{resrv:.1f}"])
                    f.flush()
                    self._mem_log_stop.wait(interval_s)

        self._mem_log_thread = threading.Thread(target=_loop, daemon=True)
        self._mem_log_thread.start()
        print(f"[Server] Memory logging → {log_path}")

    def stop_memory_logging(self):
        if hasattr(self, "_mem_log_stop"):
            self._mem_log_stop.set()
            self._mem_log_thread.join(timeout=5)

    def start(self):
        """Synchronous — call from main thread before launching worker threads."""
        if self._dev.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass

        if self.strategy == "deploy_sharing":
            bb = _SharedBackbone(self.backbone, self._dev, len(self.tasks))
            bb.load()
            self._backbones = [bb]
            for task in self.tasks:
                self._task_backbone[task] = bb
            print(f"[Server:deploy_sharing] 1 backbone, {len(self.tasks)} tasks batched together")

        else:  # task_sharing
            for task in self.tasks:
                bb = _SharedBackbone(self.backbone, self._dev, 1)
                bb.load()
                self._backbones.append(bb)
                self._task_backbone[task] = bb
            print(f"[Server:task_sharing] {len(self.tasks)} separate backbones on {self.device}")

        self._peak_mem_mb = self._gpu_mem_mb()
        self._model_load_mem_mb = (
            torch.cuda.memory_allocated(self._dev) / (1024 ** 2)
            if self._dev.type == "cuda" else 0.0
        )
        print(f"[Server] GPU mem after model load: {self._model_load_mem_mb:.1f} MB  "
              f"(peak so far: {self._peak_mem_mb:.1f} MB)")

    def stop(self):
        """Synchronous cleanup."""
        self._backbones.clear()
        self._task_backbone.clear()
        if self._dev.type == "cuda":
            torch.cuda.empty_cache()

    def infer(self, task: str, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Called from worker thread. Blocks until the batched forward completes."""
        return self._task_backbone[task].infer(task, x, mask)

    def peak_gpu_mem_mb(self) -> float:
        return self._peak_mem_mb

    def model_load_mem_mb(self) -> float:
        return self._model_load_mem_mb

    def batch_stats(self) -> Dict:
        seen = set()
        total_batches = 0
        total_items = 0.0
        total_mixed = 0.0
        for bb in self._backbones:
            if id(bb) in seen:
                continue
            seen.add(id(bb))
            s = bb.stats()
            n = s["batch_count"]
            total_batches += n
            total_items += s["avg_batch_size"] * n
            total_mixed += s["mixed_batch_fraction"] * n
        return {
            "batch_count": total_batches,
            "avg_batch_size": (total_items / total_batches) if total_batches else 0.0,
            "mixed_batch_fraction": (total_mixed / total_batches) if total_batches else 0.0,
        }
