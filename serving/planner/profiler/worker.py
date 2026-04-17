# worker.py
import json
import os
import sys
import ctypes
from pathlib import Path

import torch
import gc

from inference_pipeline import InferencePipeline
from config import device as cfg_device


def _setup_tpc_stream(tpc_count, tpc_mode):
    """Create a CUDA stream pinned to `tpc_count` TPCs via libsmctrl.

    Returns the stream on success, or None if no partition is requested.
    Mirrors device/server.py::RuntimeServer._setup_tpc.
    """
    if tpc_count is None or tpc_mode == "none":
        return None

    cuda_device = os.environ.get("CUDA_DEVICE", cfg_device)
    device_id = int(cuda_device.split(":")[-1]) if ":" in cuda_device else 0

    if tpc_mode != "libsmctrl":
        raise ValueError(f"Unsupported tpc_mode in profiler: {tpc_mode}")

    TPC_LIB_DIR = Path(os.environ.get(
        "TPC_LIB_DIR",
        "../../../../TPC_controller/tpc_controller",
    ))
    so_path = TPC_LIB_DIR / "libsmctrl" / "libsmctrl.so"
    if not so_path.exists():
        raise FileNotFoundError(f"libsmctrl.so not found at {so_path}")

    lib = ctypes.CDLL(str(so_path))
    lib.libsmctrl_set_stream_mask.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.libsmctrl_set_stream_mask.restype = None
    lib.libsmctrl_get_tpc_info_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
    ]
    lib.libsmctrl_get_tpc_info_cuda.restype = ctypes.c_int

    num_tpcs = ctypes.c_uint32()
    ret = lib.libsmctrl_get_tpc_info_cuda(ctypes.byref(num_tpcs), device_id)
    total_tpcs = num_tpcs.value if ret == 0 else (
        torch.cuda.get_device_properties(cuda_device).multi_processor_count // 2
    )
    if tpc_count > total_tpcs:
        raise ValueError(f"tpc_count={tpc_count} exceeds total_tpcs={total_tpcs}")

    partition = list(range(tpc_count))
    stream = torch.cuda.Stream(device=cuda_device)
    enable_bits = 0
    for tid in partition:
        enable_bits |= (1 << tid)
    disable_mask = (~enable_bits) & 0xFFFFFFFFFFFFFFFF
    lib.libsmctrl_set_stream_mask(
        ctypes.c_void_p(stream.cuda_stream),
        ctypes.c_uint64(disable_mask),
    )
    print(f"[TPC] libsmctrl: pinned stream to {tpc_count} TPCs "
          f"(partition={partition}, total={total_tpcs})")
    return stream


if __name__ == "__main__":
    payload = json.loads(sys.argv[1])
    task_name  = payload["task_name"]
    task_info  = payload["task_info"]
    pipeline   = payload["pipeline"]
    log_file =   payload["file_name"]
    tpc_count  = payload.get("tpc_count")
    tpc_mode   = payload.get("tpc_mode", "none")

    stream = _setup_tpc_stream(tpc_count, tpc_mode)

    # The runtime owns its own dedicated CUDA stream. Passing the TPC-pinned
    # stream in via cuda_stream ensures every run_batch kernel launches on the
    # masked stream — equivalent to the device/server.py TPC path.
    pipe = InferencePipeline(
        task_name, task_info, pipeline, log_file,
        tpc_count=tpc_count, cuda_stream=stream,
    )
    pipe.run()
    if stream is not None:
        stream.synchronize()

    # FULL cleanup
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
