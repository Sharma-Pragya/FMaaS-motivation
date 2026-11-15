#!/usr/bin/env python3
"""isolation_mechanisms — overhead of three ways to run N tasks on one shared backbone.

Standalone: needs only torch + torchvision (+ pynvml for GPU memory). Nothing
here imports serving/device, so the mechanisms can be compared in isolation.

Modes
  model_only       One thread, no queues, one CUDA context. Each iteration runs
                   one request from every task as a single backbone batch, then
                   every head inline. Lower bound: the model's own cost.
  thread_stream    Current FMvisor design. One process / CUDA context. Client
                   threads enqueue; a backbone thread forms cross-task batches
                   on its own stream; each task's head runs on that task's own
                   worker thread + CUDA stream.
  process_context  Hard isolation, plain data path. Backbone in the main process;
                   each task is a spawned process with its own CUDA context and
                   head. Slot buffers are CPU shared memory, so every request
                   crosses the host: task writes input to its CPU slot, backbone
                   copies it H2D, runs, copies features D2H into the CPU slot,
                   task copies them H2D and runs its head. Only slot ids cross
                   the queue.
  process_context_ipc
                   process_context with the slot buffers in VRAM, owned by the
                   main process and mapped into tasks through CUDA IPC. Inputs
                   and features never leave the GPU between processes.
  process_context_ipc_mps
                   process_context_ipc with every context attached to a CUDA MPS
                   server, so kernels from different processes can run
                   concurrently instead of time-slicing. run.py starts a private
                   MPS daemon (own pipe/log dirs, so other GPU jobs are not
                   attached) and shuts it down at exit.
  instance_per_task
                   No sharing: the deployment-per-task baseline. Each task is a
                   spawned process with its own CUDA context, its own full copy
                   of the backbone and its head, serving its own requests at
                   batch size 1. Nothing is batched across tasks; the main
                   process only coordinates and never touches CUDA.
  instance_per_task_mps
                   instance_per_task with the contexts attached to a private
                   MPS server, so the separate deployments' kernels can overlap.

Workload: closed loop, one outstanding request per task, for --duration seconds
after --warmup seconds. Backbones: the paper's PyTorch FMs through FMTK
(momentlarge / papageip with the ecgclass MLP head, dinobase / swinlarge with
the nyudepth monocular-depth head), or a torchvision ResNet trunk with random
weights and an avgpool + Linear head. Inputs are random tensors of each task's
real shape, so compute matches real data. The vLLM backbones (Qwen2.5-3B,
Qwen2-VL-7B) are not covered: vLLM owns batching and the KV cache, so the
backbone/head split these modes isolate does not exist there.

Memory: NVML per-process usage for every process of the run (this is what
exposes per-CUDA-context overhead; torch.cuda.memory_* cannot see it), the
device-level delta from before the run, and torch allocator stats.

Usage (from serving/):
    python experiments/isolation_mechanisms/run.py --mode thread_stream --ntasks 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import nn


MODES = ("model_only", "thread_stream", "process_context", "process_context_ipc",
         "process_context_ipc_mps", "instance_per_task", "instance_per_task_mps")
# Modes whose task-shared slot buffers live in VRAM (CUDA IPC) rather than CPU shm.
IPC_MODES = ("process_context_ipc", "process_context_ipc_mps")
RESNET_DIMS = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048}

# Paper backbones, built through FMTK (FMTK/src must be on PYTHONPATH). Each is
# paired with the task the paper serves on it; head configs are copied from
# serving/device/config.py so this script stays standalone. Shapes are per
# request (no batch dim); "feature" is checked against the real backbone output
# at startup.
FMTK_MODELS = {
    "momentlarge": {"task": "ecgclass", "input": (1, 512), "feature": (1, 64, 1024),
                    "head": ("mlp", {"input_dim": 1024, "output_dim": 5, "hidden_dim": 128})},
    "papageip":    {"task": "ecgclass", "input": (1, 512), "feature": (1, 512),
                    "head": ("mlp", {"input_dim": 512, "output_dim": 5, "hidden_dim": 128})},
    "dinobase":    {"task": "nyudepth", "input": (3, 224, 224), "feature": (257, 768),
                    "head": ("monodepth", {"input_dim": 768, "height": 16, "width": 16,
                                           "pixel_height": 224, "pixel_width": 224,
                                           "mode": "PATCH"})},
    "swinlarge":   {"task": "nyudepth", "input": (3, 224, 224), "feature": (49, 1536),
                    "head": ("monodepth", {"input_dim": 1536, "height": 7, "width": 7,
                                           "pixel_height": 224, "pixel_width": 224,
                                           "mode": "ALL"})},
}
BACKBONES = sorted(RESNET_DIMS) + sorted(FMTK_MODELS)


def input_shape(backbone: str) -> tuple:
    return FMTK_MODELS[backbone]["input"] if backbone in FMTK_MODELS else (3, 224, 224)


def feature_shape(backbone: str) -> tuple:
    if backbone in FMTK_MODELS:
        return FMTK_MODELS[backbone]["feature"]
    return (RESNET_DIMS[backbone], 7, 7)


def task_name(backbone: str) -> str:
    return FMTK_MODELS[backbone]["task"] if backbone in FMTK_MODELS else "classification"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Head(nn.Module):
    def __init__(self, dim: int, classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.pool(x), 1))


def make_backbone(name: str, device: str):
    """Return a callable: [B, *input_shape] -> [B, *feature_shape]."""
    torch.manual_seed(0)
    if name in RESNET_DIMS:
        import torchvision.models as tvm
        full = getattr(tvm, name)(weights=None)
        # Drop avgpool + fc: output is [B, dim, 7, 7], heads do the pooling.
        return nn.Sequential(*list(full.children())[:-2]).to(device).eval()

    from fmtk.components.backbones.moment import MomentModel
    from fmtk.components.backbones.papagei import PapageiModel
    from fmtk.components.backbones.dinov2 import DinoV2Model
    from fmtk.components.backbones.swin import SwinModel
    if name == "momentlarge":
        fm = MomentModel(device, "large")
    elif name == "papageip":
        fm = PapageiModel(device, "papagei_p", model_config={
            "in_channels": 1, "base_filters": 32, "kernel_size": 3, "stride": 2,
            "groups": 1, "n_block": 18, "n_classes": 512})
    elif name == "dinobase":
        fm = DinoV2Model(device, "base", model_config={"return_all_tokens": True})
    elif name == "swinlarge":
        fm = SwinModel(device, "large", model_config={})
    fm.model.eval()
    # FMTK wrappers move inputs to the device themselves; mask=None means every
    # timestep is observed, which costs the same as a real all-ones mask.
    return lambda x: fm.forward(x)


def make_head(task_id: int, backbone: str, classes: int, device: str):
    """Return a callable: [B, *feature_shape] -> task output."""
    torch.manual_seed(1000 + task_id)
    if backbone in RESNET_DIMS:
        return Head(RESNET_DIMS[backbone], classes).to(device).eval()
    kind, cfg = FMTK_MODELS[backbone]["head"]
    if kind == "mlp":
        from fmtk.components.decoders.classification.mlp import MLPDecoder
        dec = MLPDecoder(device=device, cfg=cfg)
        dec.model.to(device).eval()
    else:
        from fmtk.components.decoders.regression.monocular_depth import MonocularDepthDecoder
        dec = MonocularDepthDecoder(device=device, cfg=cfg)
        dec.to_device()
        dec.eval()
    return dec.forward


def make_input(task_id: int, backbone: str, device: str) -> torch.Tensor:
    gen = torch.Generator().manual_seed(task_id)
    x = torch.randn((1, *input_shape(backbone)), generator=gen)
    return x.pin_memory() if _is_cuda(device) else x


def check_feature_shape(backbone_fn, backbone: str, device: str) -> None:
    """Fail fast if the declared feature shape (used to size slot buffers
    before the backbone is loaded) doesn't match what the backbone returns."""
    with torch.inference_mode():
        out = backbone_fn(torch.randn((1, *input_shape(backbone)), device=device))
    if tuple(out.shape[1:]) != feature_shape(backbone):
        raise RuntimeError(f"{backbone}: backbone returns per-request features of shape "
                           f"{tuple(out.shape[1:])}, but feature_shape() declares "
                           f"{feature_shape(backbone)}; fix FMTK_MODELS")


def warm_up(backbone, heads: list, device: str, max_batch: int, name: str) -> None:
    """Run every backbone batch size once before timing starts.

    The first call in a fresh process pays one-off costs (lazy CUDA module
    loading, cuDNN init and per-shape algorithm selection) that can exceed the
    time-based --warmup and would otherwise land in the measured window.
    """
    check_feature_shape(backbone, name, device)
    with torch.inference_mode():
        for b in range(1, max_batch + 1):
            backbone(torch.randn((b, *input_shape(name)), device=device))
        feat = torch.randn((1, *feature_shape(name)), device=device)
        for head in heads:
            head(feat)
    _sync(device)


def _is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def _sync(device: str, stream=None) -> None:
    if not _is_cuda(device):
        return
    if stream is not None:
        stream.synchronize()
    else:
        torch.cuda.current_stream().synchronize()


def _torch_mem(device: str) -> tuple[float, float]:
    if not _is_cuda(device):
        return 0.0, 0.0
    return (torch.cuda.memory_allocated(device) / 2**20,
            torch.cuda.memory_reserved(device) / 2**20)


# ---------------------------------------------------------------------------
# GPU memory probe (NVML)
# ---------------------------------------------------------------------------

class GpuMemProbe:
    """Reads per-process and per-device GPU memory through NVML.

    Per-process numbers can be missing under MPS or when the container's PID
    namespace differs from the host's; the device-level delta still works then.
    """

    def __init__(self):
        self.ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                             for i in range(pynvml.nvmlDeviceGetCount())]
            self.ok = True
        except Exception as exc:
            print(f"[mem] NVML unavailable ({exc}); GPU memory columns will be 0")

    def device_used(self) -> list[int]:
        if not self.ok:
            return []
        return [self._nvml.nvmlDeviceGetMemoryInfo(h).used for h in self._handles]

    def pid_used(self, pids: set[int]) -> dict[int, int]:
        if not self.ok:
            return {}
        used: dict[int, int] = {}
        for h in self._handles:
            try:
                procs = self._nvml.nvmlDeviceGetComputeRunningProcesses(h)
            except Exception:
                continue
            for p in procs:
                if p.pid in pids and p.usedGpuMemory:
                    used[p.pid] = used.get(p.pid, 0) + p.usedGpuMemory
        return used


class MemMonitor:
    """Background sampler: keeps the peak of per-pid usage and device delta."""

    def __init__(self, probe: GpuMemProbe, baseline: list[int], period_s: float = 0.25):
        self._probe = probe
        self._baseline = baseline
        self._period_s = period_s
        self._pids: set[int] = {os.getpid()}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mem_monitor")
        self.peak_pids_total = 0
        self.peak_per_pid: dict[int, int] = {}
        self.peak_device_delta = 0

    def add_pids(self, pids) -> None:
        self._pids.update(pids)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._sample()
        self._stop.set()
        self._thread.join(timeout=2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self._period_s)

    def _sample(self) -> None:
        per_pid = self._probe.pid_used(set(self._pids))
        total = sum(per_pid.values())
        if total > self.peak_pids_total:
            self.peak_pids_total = total
            self.peak_per_pid = per_pid
        used = self._probe.device_used()
        if used and self._baseline:
            delta = max(u - b for u, b in zip(used, self._baseline))
            self.peak_device_delta = max(self.peak_device_delta, delta)


# ---------------------------------------------------------------------------
# Mode: model_only
# ---------------------------------------------------------------------------

def run_model_only(args, monitor: MemMonitor) -> dict:
    device = args.device
    t0 = time.monotonic()
    backbone = make_backbone(args.backbone, device)
    heads = [make_head(i, args.backbone, args.classes, device) for i in range(args.ntasks)]
    xs = [make_input(i, args.backbone, device) for i in range(args.ntasks)]
    stream = torch.cuda.Stream(device=device) if _is_cuda(device) else None
    chunks = [list(range(i, min(i + args.max_batch, args.ntasks)))
              for i in range(0, args.ntasks, args.max_batch)]
    warm_up(backbone, heads, device, args.max_batch, args.backbone)
    ready_s = time.monotonic() - t0

    lat: list[list[float]] = [[] for _ in range(args.ntasks)]
    t_go = time.monotonic()
    t_meas, t_end = t_go + args.warmup, t_go + args.warmup + args.duration
    stream_ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with stream_ctx, torch.inference_mode():
        k = 0
        while True:
            s = time.monotonic()
            if s >= t_end:
                break
            tasks = chunks[k % len(chunks)]
            k += 1
            x = torch.cat([xs[t].to(device, non_blocking=True) for t in tasks])
            feats = backbone(x)
            for j, t in enumerate(tasks):
                heads[t](feats[j:j + 1])
            _sync(device, stream)
            e = time.monotonic()
            if s >= t_meas:
                for t in tasks:
                    lat[t].append(e - s)

    alloc, reserved = _torch_mem(device)
    return {"lat": lat, "ready_s": ready_s, "torch_alloc_mb": alloc,
            "torch_reserved_mb": reserved, "n_contexts": 1}


# ---------------------------------------------------------------------------
# Mode: thread_stream
# ---------------------------------------------------------------------------

class _Req:
    __slots__ = ("task", "x", "done")

    def __init__(self, task: int, x: torch.Tensor):
        self.task = task
        self.x = x
        self.done = threading.Event()


def _collect_batch(q: queue.Queue, max_batch: int, window_s: float, stop: threading.Event) -> list:
    while not stop.is_set():
        try:
            first = q.get(timeout=0.05)
            break
        except queue.Empty:
            continue
    else:
        return []
    batch = [first]
    deadline = time.monotonic() + window_s
    while len(batch) < max_batch:
        remaining = deadline - time.monotonic()
        try:
            batch.append(q.get(timeout=remaining) if remaining > 0 else q.get_nowait())
        except queue.Empty:
            break
    return batch


def run_thread_stream(args, monitor: MemMonitor) -> dict:
    device = args.device
    cuda = _is_cuda(device)
    t0 = time.monotonic()
    backbone = make_backbone(args.backbone, device)
    heads = [make_head(i, args.backbone, args.classes, device) for i in range(args.ntasks)]
    xs = [make_input(i, args.backbone, device) for i in range(args.ntasks)]
    bb_stream = torch.cuda.Stream(device=device) if cuda else None
    task_streams = [torch.cuda.Stream(device=device) if cuda else None for _ in range(args.ntasks)]
    req_q: queue.Queue = queue.Queue()
    task_qs = [queue.Queue() for _ in range(args.ntasks)]
    stop = threading.Event()
    warm_up(backbone, heads, device, args.max_batch, args.backbone)
    ready_s = time.monotonic() - t0

    def backbone_loop():
        ctx = torch.cuda.stream(bb_stream) if cuda else nullcontext()
        with ctx, torch.inference_mode():
            while not stop.is_set():
                batch = _collect_batch(req_q, args.max_batch, args.batch_window_ms / 1000, stop)
                if not batch:
                    continue
                x = torch.cat([r.x.to(device, non_blocking=True) for r in batch])
                feats = backbone(x)
                event = None
                if cuda:
                    event = torch.cuda.Event()
                    event.record(bb_stream)
                # Hand each task its slice; the backbone thread moves straight
                # on to the next batch while heads run on their own streams.
                for j, r in enumerate(batch):
                    task_qs[r.task].put((feats[j:j + 1], event, r))

    def task_worker(t: int):
        stream = task_streams[t]
        if stream is not None:
            torch.cuda.set_stream(stream)
        with torch.inference_mode():
            while True:
                item = task_qs[t].get()
                if item is None:
                    return
                feat, event, r = item
                if stream is not None:
                    stream.wait_event(event)
                    feat.record_stream(stream)  # don't let the allocator recycle it early
                heads[t](feat)
                _sync(device, stream)
                r.done.set()

    lat: list[list[float]] = [[] for _ in range(args.ntasks)]
    t_go = time.monotonic() + 0.2
    t_meas, t_end = t_go + args.warmup, t_go + args.warmup + args.duration

    def client(t: int):
        while time.monotonic() < t_go:
            time.sleep(0.001)
        while True:
            s = time.monotonic()
            if s >= t_end:
                return
            r = _Req(t, xs[t])
            req_q.put(r)
            r.done.wait()
            if s >= t_meas:
                lat[t].append(time.monotonic() - s)

    threads = [threading.Thread(target=backbone_loop, daemon=True, name="backbone")]
    threads += [threading.Thread(target=task_worker, args=(t,), daemon=True, name=f"task-{t}")
                for t in range(args.ntasks)]
    clients = [threading.Thread(target=client, args=(t,), daemon=True, name=f"client-{t}")
               for t in range(args.ntasks)]
    for th in threads + clients:
        th.start()
    for th in clients:
        th.join()
    stop.set()
    for q in task_qs:
        q.put(None)
    for th in threads:
        th.join(timeout=5)

    alloc, reserved = _torch_mem(device)
    return {"lat": lat, "ready_s": ready_s, "torch_alloc_mb": alloc,
            "torch_reserved_mb": reserved, "n_contexts": 1}


# ---------------------------------------------------------------------------
# Mode: process_context
# ---------------------------------------------------------------------------

class PrivateMps:
    """A CUDA MPS daemon only this run's processes attach to.

    Clients find the daemon through CUDA_MPS_PIPE_DIRECTORY, so pointing it at
    a fresh directory keeps other jobs on the GPU off this server. Must start
    before this process touches CUDA; spawned children inherit the env.
    """

    def __init__(self):
        if shutil.which("nvidia-cuda-mps-control") is None:
            raise RuntimeError("nvidia-cuda-mps-control not found; MPS mode needs it on PATH")
        root = Path(tempfile.mkdtemp(prefix="mps_isolation_"))
        self.env = dict(os.environ,
                        CUDA_MPS_PIPE_DIRECTORY=str(root / "pipe"),
                        CUDA_MPS_LOG_DIRECTORY=str(root / "log"))
        for key in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY"):
            Path(self.env[key]).mkdir(parents=True)
        self.root = root

    def start(self) -> None:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], env=self.env, check=True)
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = self.env["CUDA_MPS_PIPE_DIRECTORY"]
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = self.env["CUDA_MPS_LOG_DIRECTORY"]
        print(f"[mps] private daemon started (dirs under {self.root})")

    def server_pids(self) -> list[int]:
        # The server process is spawned on the first client connection.
        out = subprocess.run(["nvidia-cuda-mps-control"], input="get_server_list\n",
                             env=self.env, capture_output=True, text=True)
        return [int(tok) for tok in out.stdout.split() if tok.isdigit()]

    def stop(self) -> None:
        # `quit` blocks until every client context is gone, and this process is
        # itself a client until it exits, so waiting on it here deadlocks. Hand
        # the quit to a detached watcher that issues it once we have exited.
        subprocess.Popen(
            ["bash", "-c",
             f"while kill -0 {os.getpid()} 2>/dev/null; do sleep 0.2; done; "
             "echo quit | nvidia-cuda-mps-control"],
            env=self.env, start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[mps] daemon will stop once this process exits")


def _task_proc_main(t: int, cfg: dict, inputs, features, req_q, done, ready_q,
                    result_q, go, t_go_val) -> None:
    # Runs in its own spawned process => its own CUDA context.
    device = cfg["device"]
    t0 = time.monotonic()
    if _is_cuda(device):
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    head = make_head(t, cfg["backbone"], cfg["classes"], device)
    x = make_input(t, cfg["backbone"], device)
    with torch.inference_mode():
        head(features[t:t + 1].to(device))  # warm the head's kernels in this context
    _sync(device)
    ready_q.put((t, os.getpid(), time.monotonic() - t0))

    go.wait()
    t_go = t_go_val.value
    t_meas, t_end = t_go + cfg["warmup"], t_go + cfg["warmup"] + cfg["duration"]
    while time.monotonic() < t_go:
        time.sleep(0.001)
    lat: list[float] = []
    with torch.inference_mode():
        while True:
            s = time.monotonic()
            if s >= t_end:
                break
            inputs[t].copy_(x[0], non_blocking=True)
            # Host-side sync: host events don't order kernels across contexts,
            # so the copy must be complete before the backbone may read it.
            _sync(device)
            req_q.put(t)
            done.wait()
            done.clear()
            # IPC slots are already on this GPU (no-op); CPU slots copy H2D here.
            head(features[t:t + 1].to(device))
            _sync(device)
            if s >= t_meas:
                lat.append(time.monotonic() - s)
    alloc, reserved = _torch_mem(device)
    result_q.put({"task": t, "lat": lat, "torch_alloc_mb": alloc,
                  "torch_reserved_mb": reserved})


def run_process_context(args, monitor: MemMonitor) -> dict:
    device = args.device
    cuda = _is_cuda(device)
    n = args.ntasks
    t0 = time.monotonic()
    ctx = mp.get_context("spawn")

    # Slot t belongs to task t (one outstanding request per task), owned by
    # this process. In VRAM, CUDA IPC handles are passed to children at spawn;
    # on the CPU they are shared-memory tensors.
    slot_device = device if (cuda and args.mode in IPC_MODES) else "cpu"
    inputs = torch.empty((n, *input_shape(args.backbone)), device=slot_device)
    features = torch.zeros((n, *feature_shape(args.backbone)), device=slot_device)
    if slot_device == "cpu":
        inputs.share_memory_()
        features.share_memory_()

    req_q = ctx.Queue()
    ready_q = ctx.Queue()
    result_q = ctx.Queue()
    done = [ctx.Event() for _ in range(n)]
    go = ctx.Event()
    t_go_val = ctx.Value("d", 0.0)
    cfg = {"device": device, "backbone": args.backbone, "classes": args.classes,
           "warmup": args.warmup, "duration": args.duration}
    procs = [ctx.Process(target=_task_proc_main, name=f"task-{t}",
                         args=(t, cfg, inputs, features, req_q, done[t], ready_q,
                               result_q, go, t_go_val))
             for t in range(n)]
    for p in procs:
        p.start()
    monitor.add_pids(p.pid for p in procs)

    backbone = make_backbone(args.backbone, device)
    bb_stream = torch.cuda.Stream(device=device) if cuda else None
    warm_up(backbone, [], device, args.max_batch, args.backbone)

    for _ in range(n):
        while True:
            try:
                ready_q.get(timeout=1.0)
                break
            except queue.Empty:
                if any(not p.is_alive() for p in procs):
                    raise RuntimeError("a task process died during startup")
    ready_s = time.monotonic() - t0
    if getattr(args, "mps", None) is not None:
        # Under MPS the server holds device memory on the clients' behalf.
        monitor.add_pids(args.mps.server_pids())

    t_go_val.value = time.monotonic() + 0.2
    go.set()

    results: dict[int, dict] = {}
    ctx_stream = torch.cuda.stream(bb_stream) if cuda else nullcontext()
    with ctx_stream, torch.inference_mode():
        while len(results) < n:
            while True:
                try:
                    r = result_q.get_nowait()
                    results[r["task"]] = r
                except queue.Empty:
                    break
            # exitcode 0 = finished normally (its result may still be in flight).
            if any(p.exitcode not in (None, 0) for p in procs):
                raise RuntimeError("a task process died during the run")
            slots = _collect_batch_mp(req_q, args.max_batch, args.batch_window_ms / 1000)
            if not slots:
                continue
            idx = torch.tensor(slots, dtype=torch.long, device=slot_device)
            # .to() calls are no-ops for VRAM slots; H2D / D2H for CPU slots.
            feats = backbone(inputs.index_select(0, idx).to(device))
            features.index_copy_(0, idx, feats.to(slot_device))
            # Features must be written before a task's process reads them.
            _sync(device, bb_stream)
            for s in slots:
                done[s].set()

    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    alloc, reserved = _torch_mem(device)
    return {
        "lat": [results[t]["lat"] for t in range(n)],
        "ready_s": ready_s,
        "torch_alloc_mb": alloc + sum(r["torch_alloc_mb"] for r in results.values()),
        "torch_reserved_mb": reserved + sum(r["torch_reserved_mb"] for r in results.values()),
        "n_contexts": 1 + n,
    }


# ---------------------------------------------------------------------------
# Mode: instance_per_task
# ---------------------------------------------------------------------------

def _instance_proc_main(t: int, cfg: dict, ready_q, result_q, go, t_go_val) -> None:
    # One whole deployment: own CUDA context, own backbone copy, own head.
    device = cfg["device"]
    t0 = time.monotonic()
    if _is_cuda(device):
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    backbone = make_backbone(cfg["backbone"], device)
    head = make_head(t, cfg["backbone"], cfg["classes"], device)
    x = make_input(t, cfg["backbone"], device)
    stream = torch.cuda.Stream(device=device) if _is_cuda(device) else None
    warm_up(backbone, [head], device, 1, cfg["backbone"])
    ready_q.put((t, os.getpid(), time.monotonic() - t0))

    go.wait()
    t_go = t_go_val.value
    t_meas, t_end = t_go + cfg["warmup"], t_go + cfg["warmup"] + cfg["duration"]
    while time.monotonic() < t_go:
        time.sleep(0.001)
    lat: list[float] = []
    stream_ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with stream_ctx, torch.inference_mode():
        while True:
            s = time.monotonic()
            if s >= t_end:
                break
            head(backbone(x.to(device, non_blocking=True)))
            _sync(device, stream)
            if s >= t_meas:
                lat.append(time.monotonic() - s)
    alloc, reserved = _torch_mem(device)
    result_q.put({"task": t, "lat": lat, "torch_alloc_mb": alloc,
                  "torch_reserved_mb": reserved})


def run_instance_per_task(args, monitor: MemMonitor) -> dict:
    n = args.ntasks
    t0 = time.monotonic()
    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue()
    result_q = ctx.Queue()
    go = ctx.Event()
    t_go_val = ctx.Value("d", 0.0)
    cfg = {"device": args.device, "backbone": args.backbone, "classes": args.classes,
           "warmup": args.warmup, "duration": args.duration}
    procs = [ctx.Process(target=_instance_proc_main, name=f"instance-{t}",
                         args=(t, cfg, ready_q, result_q, go, t_go_val))
             for t in range(n)]
    for p in procs:
        p.start()
    monitor.add_pids(p.pid for p in procs)

    for _ in range(n):
        while True:
            try:
                ready_q.get(timeout=1.0)
                break
            except queue.Empty:
                if any(not p.is_alive() for p in procs):
                    raise RuntimeError("an instance process died during startup")
    ready_s = time.monotonic() - t0
    if getattr(args, "mps", None) is not None:
        monitor.add_pids(args.mps.server_pids())

    t_go_val.value = time.monotonic() + 0.2
    go.set()

    results: dict[int, dict] = {}
    while len(results) < n:
        try:
            r = result_q.get(timeout=1.0)
            results[r["task"]] = r
        except queue.Empty:
            # exitcode 0 = finished normally (its result may still be in flight).
            if any(p.exitcode not in (None, 0) for p in procs):
                raise RuntimeError("an instance process died during the run")

    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    return {
        "lat": [results[t]["lat"] for t in range(n)],
        "ready_s": ready_s,
        "torch_alloc_mb": sum(r["torch_alloc_mb"] for r in results.values()),
        "torch_reserved_mb": sum(r["torch_reserved_mb"] for r in results.values()),
        "n_contexts": n,
    }


def _collect_batch_mp(q, max_batch: int, window_s: float) -> list[int]:
    try:
        first = q.get(timeout=0.05)
    except queue.Empty:
        return []
    batch = [first]
    deadline = time.monotonic() + window_s
    while len(batch) < max_batch:
        remaining = deadline - time.monotonic()
        try:
            batch.append(q.get(timeout=remaining) if remaining > 0 else q.get_nowait())
        except queue.Empty:
            break
    return batch


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "mode", "backbone", "ntasks", "max_batch", "batch_window_ms", "duration_s",
    "n_requests", "throughput_rps", "lat_mean_ms", "lat_p50_ms", "lat_p95_ms",
    "lat_p99_ms", "ready_s", "n_contexts", "gpu_mem_procs_mb",
    "gpu_mem_device_delta_mb", "torch_alloc_mb", "torch_reserved_mb",
    "gpu_mem_per_proc_mb",
]


def summarize(args, res: dict, monitor: MemMonitor) -> dict:
    all_lat = np.array([x for per_task in res["lat"] for x in per_task]) * 1e3
    n = len(all_lat)
    pct = (lambda q: round(float(np.percentile(all_lat, q)), 3)) if n else (lambda q: 0.0)
    return {
        "mode": args.mode,
        "backbone": args.backbone,
        "ntasks": args.ntasks,
        "max_batch": args.max_batch,
        "batch_window_ms": args.batch_window_ms,
        "duration_s": args.duration,
        "n_requests": n,
        "throughput_rps": round(n / args.duration, 2),
        "lat_mean_ms": round(float(all_lat.mean()), 3) if n else 0.0,
        "lat_p50_ms": pct(50),
        "lat_p95_ms": pct(95),
        "lat_p99_ms": pct(99),
        "ready_s": round(res["ready_s"], 2),
        "n_contexts": res["n_contexts"],
        "gpu_mem_procs_mb": round(monitor.peak_pids_total / 2**20, 1),
        "gpu_mem_device_delta_mb": round(monitor.peak_device_delta / 2**20, 1),
        "torch_alloc_mb": round(res["torch_alloc_mb"], 1),
        "torch_reserved_mb": round(res["torch_reserved_mb"], 1),
        "gpu_mem_per_proc_mb": json.dumps(
            sorted(round(v / 2**20, 1) for v in monitor.peak_per_pid.values())),
    }


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--ntasks", type=int, default=2)
    parser.add_argument("--backbone", choices=BACKBONES, default="resnet50",
                        help="torchvision ResNet (random weights) or a paper backbone via FMTK.")
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--max-batch", type=int, default=None,
                        help="Max cross-task backbone batch (default: ntasks).")
    parser.add_argument("--batch-window-ms", type=float, default=1.0,
                        help="Batch collection window (thread_stream / process_context).")
    parser.add_argument("--duration", type=float, default=30.0, help="Measured seconds.")
    parser.add_argument("--warmup", type=float, default=5.0, help="Unmeasured seconds first.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-csv", default="experiments/isolation_mechanisms/results/summary.csv")
    args = parser.parse_args()
    args.max_batch = args.max_batch or args.ntasks

    # MPS must be up before this process (or any child) creates a CUDA context.
    args.mps = None
    if args.mode.endswith("_mps"):
        if not _is_cuda(args.device):
            parser.error(f"{args.mode} needs a CUDA device")
        args.mps = PrivateMps()
        args.mps.start()

    try:
        # Baseline device memory BEFORE this process creates its CUDA context.
        probe = GpuMemProbe() if _is_cuda(args.device) else None
        baseline = probe.device_used() if probe else []
        monitor = MemMonitor(probe, baseline) if probe else _NullMonitor()
        monitor.start()

        runner = {"model_only": run_model_only,
                  "thread_stream": run_thread_stream,
                  "process_context": run_process_context,
                  "process_context_ipc": run_process_context,
                  "process_context_ipc_mps": run_process_context,
                  "instance_per_task": run_instance_per_task,
                  "instance_per_task_mps": run_instance_per_task}[args.mode]
        print(f"[run] mode={args.mode} backbone={args.backbone} ntasks={args.ntasks} "
              f"max_batch={args.max_batch} device={args.device}")
        res = runner(args, monitor)
        monitor.stop()
    finally:
        if args.mps is not None:
            args.mps.stop()

    row = summarize(args, res, monitor)
    append_csv(Path(args.out_csv), row)
    print("[run] " + "  ".join(f"{k}={row[k]}" for k in (
        "n_requests", "throughput_rps", "lat_p50_ms", "lat_p99_ms", "ready_s",
        "n_contexts", "gpu_mem_procs_mb", "gpu_mem_device_delta_mb",
        "torch_reserved_mb", "gpu_mem_per_proc_mb")))


class _NullMonitor:
    peak_pids_total = 0
    peak_per_pid: dict = {}
    peak_device_delta = 0

    def add_pids(self, pids) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


if __name__ == "__main__":
    main()
