"""task_proc.py — Per-task process for process-isolation mode.

Responsibilities:
  1. Preprocessing: read TaskInferMsg from gRPC process, write tensor to
     SharedMemory, put PreprocessedMsg on sched_q (backbone_proc reads this)
  2. Decoder: read features from feat_q (backbone_proc writes this),
     run decoder forward, put result on result_q (gRPC process reads this)

NO scheduling decisions here. backbone_proc owns TenantQueues and the
scheduling policy (FifoPolicy / RoundRobinPolicy) across all task queues.

Queue topology:
    gRPC process
        │  TaskInferMsg (req_id, raw tensor bytes)
        ▼
    raw_q[task]          ← task_proc reads, preprocesses, writes to sched_q
        │  PreprocessedMsg (req_id, task, shm_name, shape, dtype, enqueued_at)
        ▼
    sched_q              ← backbone_proc scheduler reads ALL tasks from here
        │  (shared single queue — backbone_proc polls per-task entries)
        ▼
    backbone_proc (scheduler + forward)
        │  FeatMsg (req_id, task, shm_name, shape, dtype)
        ▼
    feat_q[task]         ← task_proc reads, runs decoder
        │  result dict
        ▼
    result_q             ← gRPC process reads, resolves asyncio futures
"""

import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

@dataclass
class TaskInferMsg:
    """gRPC process → task_proc: raw proto tensor bytes."""
    type: str = "infer"
    req_id: int = 0
    x_bytes: bytes = b""
    x_shape: tuple = ()
    x_dtype: str = "float32"
    mask_bytes: bytes = b""
    mask_shape: tuple = ()
    mask_dtype: str = "float32"
    enqueued_at: float = 0.0   # wall time set by gRPC process on arrival


@dataclass
class TaskControlMsg:
    """gRPC process → task_proc: control commands."""
    type: str = "control"
    command: str = ""          # "load_decoder" | "stop"
    payload_json: str = ""
    seq: int = 0               # matched to reply on ctrl_reply_q


@dataclass
class PreprocessedMsg:
    """task_proc → backbone_proc scheduler queue: tensor in SharedMemory."""
    type: str = "preprocessed"
    req_id: int = 0
    task: str = ""
    shm_name: str = ""
    shape: tuple = ()
    dtype: str = "float32"
    mask_shm_name: str = ""
    mask_shape: tuple = ()
    mask_dtype: str = "float32"
    enqueued_at: float = 0.0   # preserved from TaskInferMsg for FIFO ordering


# ---------------------------------------------------------------------------
# Task process entry point
# ---------------------------------------------------------------------------

def task_proc_main(
    task: str,
    raw_q: Queue,              # gRPC process → this proc (TaskInferMsg / TaskControlMsg)
    ctrl_reply_q: Queue,       # this proc → isolated_app (control replies)
    task_sched_q: Queue,       # this proc → backbone_proc, dedicated per-task queue
    feat_q: Queue,             # backbone_proc → this proc (feat result dicts)
    result_q: Queue,           # this proc → gRPC process (inference result dicts)
    device: str,
):
    import json
    import os

    import torch
    import torch.nn as nn
    os.environ["CUDA_DEVICE"] = device

    from device.model_loader import _build_decoder

    print(f"[TaskProc:{task}] Started on device={device}")

    decoder = None
    backbone_name = None

    # req_id -> TaskInferMsg: requests sent to backbone, awaiting features
    pending: dict = {}

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def load_decoder(spec: dict) -> float:
        """Load decoder via pipeline.add_decoder using a dummy model_instance.

        The dummy only needs .model_category so Pipeline can resolve the path
        to the decoder weights. No backbone weights are loaded here — the real
        backbone lives in backbone_proc.
        """
        nonlocal decoder, backbone_name
        from fmtk.pipeline import Pipeline
        from device.model_loader import _build_decoder

        bb   = spec["backbone"]
        dtype = spec["type"]
        path  = spec["path"]
        model_category = spec.get("model_category", "tsfm")
        print(f"[TaskProc:{task}] Loading decoder backbone={bb} type={dtype} path={path} model_category={model_category}")

        class _DummyModelInstance:
            pass
        dummy = _DummyModelInstance()
        dummy.model_category = model_category

        if "cuda" in device:
            torch.cuda.reset_peak_memory_stats()

        pipeline = Pipeline(dummy)
        decoder_obj = _build_decoder(bb, task, dtype, device)
        pipeline.add_decoder(decoder_obj, load=True, train=False, path=path)
        # decoder_obj.model now has weights loaded and is on GPU (add_decoder handles this)
        decoder_obj.model.eval()
        decoder = decoder_obj
        backbone_name = bb

        mem_mb = 0.0
        if "cuda" in device:
            torch.cuda.synchronize(torch.device(device))
            mem_mb = torch.cuda.max_memory_allocated(torch.device(device)) / (1024 ** 2)
        print(f"[TaskProc:{task}] Decoder loaded  gpu_alloc_peak={mem_mb:.1f} MB")
        return mem_mb

    def preprocess_and_forward(msg: TaskInferMsg):
        """Write tensor to SharedMemory, put PreprocessedMsg on sched_q."""
        x = np.frombuffer(msg.x_bytes, dtype=msg.x_dtype).reshape(msg.x_shape)
        shm = SharedMemory(create=True, size=max(x.nbytes, 1))
        np.copyto(np.ndarray(x.shape, dtype=x.dtype, buffer=shm.buf), x)
        shm.close()  # backbone_proc will unlink after reading

        mask_shm_name, mask_shape, mask_dtype = "", (), "float32"
        if msg.mask_bytes:
            mask = np.frombuffer(msg.mask_bytes, dtype=msg.mask_dtype).reshape(msg.mask_shape)
            mshm = SharedMemory(create=True, size=max(mask.nbytes, 1))
            np.copyto(np.ndarray(mask.shape, dtype=mask.dtype, buffer=mshm.buf), mask)
            mshm.close()
            mask_shm_name = mshm.name
            mask_shape = mask.shape
            mask_dtype = str(mask.dtype)

        task_sched_q.put(PreprocessedMsg(
            req_id=msg.req_id,
            task=task,
            shm_name=shm.name,
            shape=x.shape,
            dtype=str(x.dtype),
            mask_shm_name=mask_shm_name,
            mask_shape=mask_shape,
            mask_dtype=mask_dtype,
            enqueued_at=msg.enqueued_at,
        ))
        pending[msg.req_id] = msg

    def drain_feat_q():
        """Non-blocking: read backbone features, run decoder, emit results."""
        while True:
            try:
                feat = feat_q.get_nowait()
            except Exception:
                break
            req_id = feat["req_id"]
            original = pending.pop(req_id, None)
            if original is None:
                print(f"[TaskProc:{task}] Unexpected feat req_id={req_id}")
                continue
            if "error" in feat:
                result_q.put({"req_id": req_id, "error": feat["error"]})
                continue
            try:
                result_q.put(_run_decoder(feat))
            except Exception as e:
                print(f"[TaskProc:{task}] Decoder error req_id={req_id}: {e}")
                result_q.put({"req_id": req_id, "error": str(e)})

    def _run_decoder(feat: dict) -> dict:
        shm = SharedMemory(name=feat["shm_name"])
        feats = np.ndarray(feat["shape"], dtype=feat["dtype"], buffer=shm.buf).copy()
        shm.close()
        shm.unlink()

        feat_tensor = torch.from_numpy(feats).to(device)

        swap_start = time.time_ns()
        swap_time_ns = time.time_ns() - swap_start

        dec_start = time.time_ns()
        with torch.no_grad():
            if decoder is not None:
                logit = decoder.forward(feat_tensor)
                if isinstance(decoder.criterion, nn.CrossEntropyLoss):
                    logit = torch.argmax(logit, dim=1)
                output = logit.detach().cpu().numpy().reshape(-1)
            else:
                output = feat_tensor.detach().cpu().float().numpy().reshape(-1)
        decoder_time_ns = time.time_ns() - dec_start

        return {
            "req_id": feat["req_id"],
            "output": output,
            "start_time_ns": feat["start_time_ns"],
            "end_time_ns": time.time_ns(),
            "proc_time_ns": feat["proc_time_ns"],
            "swap_time_ns": swap_time_ns,
            "decoder_time_ns": decoder_time_ns,
        }

    # -----------------------------------------------------------------------
    # Main loop: interleave raw_q (new requests) and feat_q (backbone responses)
    # -----------------------------------------------------------------------
    while True:
        drain_feat_q()

        try:
            msg = raw_q.get(timeout=0.005)
        except Exception:
            continue

        t = getattr(msg, "type", None)

        if t == "stop":
            print(f"[TaskProc:{task}] Received stop, exiting")
            break

        if t == "control":
            payload = json.loads(msg.payload_json) if msg.payload_json else {}
            status = "ok"
            gpu_decoder_mem_mb = 0.0
            try:
                if msg.command == "load_decoder":
                    gpu_decoder_mem_mb = load_decoder(payload)
                    status = f"loaded_decoder_{task}"
                else:
                    status = f"unknown_command_{msg.command}"
            except Exception as e:
                status = f"error_{e}"
                print(f"[TaskProc:{task}] Control error: {e}")
            ctrl_reply_q.put({"seq": msg.seq, "status": status,
                               "gpu_decoder_mem_mb": gpu_decoder_mem_mb})
            continue

        if t == "infer":
            preprocess_and_forward(msg)
            continue

        print(f"[TaskProc:{task}] Unknown message type: {t}")

    print(f"[TaskProc:{task}] Exited")


# ---------------------------------------------------------------------------
# Helper to spawn
# ---------------------------------------------------------------------------

def start_task_process(
    task: str,
    raw_q: Queue,
    ctrl_reply_q: Queue,
    task_sched_q: Queue,
    feat_q: Queue,
    result_q: Queue,
    device: str,
) -> Process:
    p = Process(
        target=task_proc_main,
        args=(task, raw_q, ctrl_reply_q, task_sched_q, feat_q, result_q, device),
        daemon=True,
        name=f"task_proc_{task}",
    )
    p.start()
    return p
