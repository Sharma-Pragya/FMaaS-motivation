"""
TPC Isolation experiment: two FMs on separate SM/TPC partitions.

Two models from the same family (momentbase + momentsmall) are loaded
on the same GPU. We compare:
  - baseline:  both models share all SMs (no pinning)
  - isolated:  each model pinned to its own SM partition

Two isolation backends are supported via --mode:

  --mode libsmctrl   (default)
      Uses libsmctrl to set per-stream TPC disable-bitmasks.
      Supports dynamic reallocation at runtime.
      Requires driver <= 528 (CUDA <= 12.1). Fails silently on driver 570+.

  --mode green
      Uses CUDA Green Contexts (official API, CUDA 12.4+, driver 550+).
      Splits SMs into two fixed partitions at startup.
      Static — partitions cannot be changed without recreating contexts.
      Works on Ampere+ (compute capability 8.x).

Usage (from serving/):
  python experiments/tpc_isolation/run.py [--n_requests 100] [--device cuda:0] [--mode green]
"""

import sys
import os
import argparse
import csv
import ctypes
import threading
import time
import numpy as np
import torch
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SERVING_DIR = Path(__file__).resolve().parents[3]
TPC_LIB_DIR = Path("/project/pi_shenoy_umass_edu/hshastri/TPC_controller/tpc_controller")

for p in [str(SERVING_DIR), str(TPC_LIB_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("CUDA_DEVICE", "cuda:0")

from serving.device.runtime import PyTorchRuntime
from serving.device.model_loader import ModelLoader

# ── libsmctrl direct interface (bypasses pycuda Stream requirement) ───────────

def _load_libsmctrl():
    so_path = TPC_LIB_DIR / "libsmctrl" / "libsmctrl.so"
    if not so_path.exists():
        raise FileNotFoundError(
            f"libsmctrl.so not found at {so_path}.\n"
            f"Build it with: cd {TPC_LIB_DIR}/libsmctrl && make libsmctrl.so"
        )
    lib = ctypes.CDLL(str(so_path))
    lib.libsmctrl_set_stream_mask.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.libsmctrl_set_stream_mask.restype  = None
    lib.libsmctrl_set_global_mask.argtypes = [ctypes.c_uint64]
    lib.libsmctrl_set_global_mask.restype  = None
    lib.libsmctrl_get_tpc_info_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int
    ]
    lib.libsmctrl_get_tpc_info_cuda.restype = ctypes.c_int
    return lib


def _query_num_tpcs(lib, device_id: int, sm_count: int) -> int:
    num_tpcs = ctypes.c_uint32()
    ret = lib.libsmctrl_get_tpc_info_cuda(ctypes.byref(num_tpcs), device_id)
    if ret != 0:
        # fallback: 2 SMs per TPC
        return sm_count // 2
    return num_tpcs.value


def _tpcs_to_disable_mask(tpc_ids, total_tpcs):
    """Convert list of enabled TPC IDs → libsmctrl disable-bitmask."""
    enable_bits = 0
    for tid in tpc_ids:
        enable_bits |= (1 << tid)
    return (~enable_bits) & 0xFFFFFFFFFFFFFFFF


def pin_stream_to_tpcs(lib, stream: torch.cuda.Stream, tpc_ids, total_tpcs):
    """Pin a torch CUDA stream to specific TPCs via libsmctrl."""
    mask = _tpcs_to_disable_mask(tpc_ids, total_tpcs)
    # torch.cuda.Stream.cuda_stream is the raw CUstream handle (int)
    lib.libsmctrl_set_stream_mask(
        ctypes.c_void_p(stream.cuda_stream), ctypes.c_uint64(mask)
    )


def clear_stream_pin(lib, stream: torch.cuda.Stream, total_tpcs):
    """Remove TPC restriction from stream (allow all TPCs)."""
    all_tpcs = list(range(total_tpcs))
    pin_stream_to_tpcs(lib, stream, all_tpcs, total_tpcs)


# ── Green Context isolation ───────────────────────────────────────────────────

class GreenContextPartition:
    """
    Splits the GPU SM pool into two equal partitions using CUDA Green Contexts.
    Each partition gets its own CUgreenCtx + CUstream.

    Requirements:
      - CUDA 12.4+ toolkit AND driver 550+ (libcuda.so.1 must expose the API)
      - Ampere or newer GPU (compute capability 8.x+)

    Limitations vs libsmctrl:
      - Partition sizes are fixed at creation — cannot be changed at runtime
        without destroying and recreating contexts (expensive ~ms operation)
      - Minimum SM granularity: 2 SMs on Ampere (must be even split)
    """

    def __init__(self, device_id: int = 0):
        self._cuda = self._load_libcuda()
        self._device_id = device_id
        self._green_ctxs = []   # list of CUgreenCtx handles (as c_void_p)
        self._streams    = []   # list of raw CUstream handles (as c_void_p)
        self._torch_streams = []

    @staticmethod
    def _load_libcuda():
        import ctypes.util
        path = ctypes.util.find_library("cuda")
        if path is None:
            # common fallback location
            path = "libcuda.so.1"
        lib = ctypes.CDLL(path)

        # cuInit
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuInit.restype  = ctypes.c_int

        # cuDeviceGet
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.cuDeviceGet.restype  = ctypes.c_int

        # cuDeviceGetDevResource  (CUDA 12.4+)
        lib.cuDeviceGetDevResource.argtypes = [
            ctypes.c_int,                   # CUdevice
            ctypes.c_void_p,                # CUdevResource*
            ctypes.c_uint,                  # CUdevResourceType
        ]
        lib.cuDeviceGetDevResource.restype = ctypes.c_int

        # cuDevSmResourceSplitByCount  (CUDA 12.4+)
        lib.cuDevSmResourceSplitByCount.argtypes = [
            ctypes.c_void_p,   # CUdevResource* result (array)
            ctypes.POINTER(ctypes.c_uint),  # unsigned int* nbGroups
            ctypes.c_void_p,   # const CUdevResource* input
            ctypes.c_void_p,   # CUdevResource* remainder (can be NULL)
            ctypes.c_uint,     # flags
            ctypes.c_uint,     # minCount
        ]
        lib.cuDevSmResourceSplitByCount.restype = ctypes.c_int

        # cuDevResourceGenerateDesc  (CUDA 12.4+)
        lib.cuDevResourceGenerateDesc.argtypes = [
            ctypes.c_void_p,   # CUdevResourceDesc* phDesc
            ctypes.c_void_p,   # CUdevResource* resources
            ctypes.c_uint,     # unsigned int nbResources
        ]
        lib.cuDevResourceGenerateDesc.restype = ctypes.c_int

        # cuGreenCtxCreate  (CUDA 12.4+)
        lib.cuGreenCtxCreate.argtypes = [
            ctypes.c_void_p,   # CUgreenCtx* phCtx
            ctypes.c_void_p,   # CUdevResourceDesc desc (passed by value — use pointer)
            ctypes.c_int,      # CUdevice
            ctypes.c_uint,     # unsigned int flags
        ]
        lib.cuGreenCtxCreate.restype = ctypes.c_int

        # cuGreenCtxStreamCreate  (CUDA 12.4+)
        lib.cuGreenCtxStreamCreate.argtypes = [
            ctypes.c_void_p,   # CUstream* phStream
            ctypes.c_void_p,   # CUgreenCtx
            ctypes.c_uint,     # flags
            ctypes.c_int,      # priority
        ]
        lib.cuGreenCtxStreamCreate.restype = ctypes.c_int

        # cuGreenCtxDestroy  (CUDA 12.4+)
        lib.cuGreenCtxDestroy.argtypes = [ctypes.c_void_p]
        lib.cuGreenCtxDestroy.restype  = ctypes.c_int

        return lib

    def _check(self, ret, name):
        if ret != 0:
            raise RuntimeError(f"{name} failed with CUresult={ret}. "
                               f"Requires CUDA 12.4+ driver (550+) and Ampere+ GPU.")

    def create_partitions(self, num_partitions: int = 2):
        """
        Split all GPU SMs into num_partitions equal groups.
        Creates one green context + one stream per partition.
        Returns list of torch.cuda.Stream objects, one per partition.
        """
        cuda = self._cuda
        CU_DEV_RESOURCE_TYPE_SM = 1          # from CUDA headers
        CU_GREEN_CTX_DEFAULT_STREAM = 0x1
        CU_STREAM_NON_BLOCKING = 0x1
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING = 0x1

        # CUdevResource is an opaque struct — allocate enough space (256 bytes is safe)
        RESOURCE_SIZE = 256

        # 1. Get device SM resource
        sm_resource = (ctypes.c_uint8 * RESOURCE_SIZE)()
        dev = ctypes.c_int(self._device_id)
        self._check(
            cuda.cuDeviceGetDevResource(dev, sm_resource, CU_DEV_RESOURCE_TYPE_SM),
            "cuDeviceGetDevResource"
        )

        # 2. Split into num_partitions groups
        result_array = (ctypes.c_uint8 * (RESOURCE_SIZE * num_partitions))()
        nb_groups = ctypes.c_uint(num_partitions)
        self._check(
            cuda.cuDevSmResourceSplitByCount(
                result_array,
                ctypes.byref(nb_groups),
                sm_resource,
                None,   # no remainder
                CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING,
                0,      # minCount = 0 (split evenly)
            ),
            "cuDevSmResourceSplitByCount"
        )
        actual_groups = nb_groups.value
        print(f"[GreenCtx] Split {self._device_id} into {actual_groups} SM partitions")

        torch_streams = []
        for i in range(actual_groups):
            # Slice out the i-th resource from the result array
            res_i = (ctypes.c_uint8 * RESOURCE_SIZE)()
            ctypes.memmove(res_i, ctypes.addressof(result_array) + i * RESOURCE_SIZE,
                           RESOURCE_SIZE)

            # 3. Generate resource descriptor
            desc = (ctypes.c_uint8 * RESOURCE_SIZE)()
            self._check(
                cuda.cuDevResourceGenerateDesc(desc, res_i, 1),
                "cuDevResourceGenerateDesc"
            )

            # 4. Create green context
            green_ctx = ctypes.c_void_p(0)
            self._check(
                cuda.cuGreenCtxCreate(
                    ctypes.byref(green_ctx),
                    desc,
                    dev,
                    CU_GREEN_CTX_DEFAULT_STREAM,
                ),
                "cuGreenCtxCreate"
            )
            self._green_ctxs.append(green_ctx)

            # 5. Create stream on this green context
            cu_stream = ctypes.c_void_p(0)
            self._check(
                cuda.cuGreenCtxStreamCreate(
                    ctypes.byref(cu_stream),
                    green_ctx,
                    CU_STREAM_NON_BLOCKING,
                    0,  # priority
                ),
                "cuGreenCtxStreamCreate"
            )
            self._streams.append(cu_stream)

            # 6. Wrap raw CUstream handle into a torch Stream
            torch_stream = torch.cuda.ExternalStream(cu_stream.value)
            torch_streams.append(torch_stream)
            print(f"[GreenCtx] Partition {i}: green_ctx={green_ctx.value:#x}  "
                  f"stream={cu_stream.value:#x}")

        self._torch_streams = torch_streams
        return torch_streams

    def destroy(self):
        for gc in self._green_ctxs:
            if gc.value:
                self._cuda.cuGreenCtxDestroy(gc)
        self._green_ctxs.clear()
        self._streams.clear()
        self._torch_streams.clear()


# ── Data loading ──────────────────────────────────────────────────────────────

def build_ecg_data(dataset_dir):
    from torch.utils.data import DataLoader
    from fmtk.datasetloaders.ecg5000 import ECG5000Dataset
    ds = ECG5000Dataset(
        {"dataset_path": f"{dataset_dir}/ECG5000"},
        {"task_type": "classification"},
        "test",
    )
    return DataLoader(ds, batch_size=1, shuffle=False)


def build_gesture_data(dataset_dir):
    from torch.utils.data import DataLoader
    from fmtk.datasetloaders.uwavegesture import UWaveGestureLibraryALLDataset
    ds = UWaveGestureLibraryALLDataset(
        {"dataset_path": f"{dataset_dir}/UWaveGestureLibraryAll"},
        {"task_type": "classification"},
        "test",
    )
    return DataLoader(ds, batch_size=1, shuffle=False)


# ── Runtime factory ───────────────────────────────────────────────────────────

def make_runtime(device: str) -> PyTorchRuntime:
    loader = ModelLoader(device=torch.device(device))
    return PyTorchRuntime(loader=loader)


# ── Worker thread ─────────────────────────────────────────────────────────────

def inference_worker(
    runtime: PyTorchRuntime,
    stream: torch.cuda.Stream,
    data_loader,
    task: str,
    n_requests: int,
    results: list,
    barrier: threading.Barrier,
):
    """Run n_requests inference calls, optionally on a pinned CUDA stream."""
    data_iter = iter(data_loader)
    latencies = []

    # Wait for both threads to be ready before starting
    barrier.wait()

    for _ in range(n_requests):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        x_i = batch["x"].numpy().astype(np.float32)
        m_i = batch["mask"].numpy().astype(np.float32) if "mask" in batch else None

        if stream is not None:
            with torch.cuda.stream(stream):
                r = runtime.run_batch(x_i, [task], mask=m_i)
        else:
            r = runtime.run_batch(x_i, [task], mask=m_i)

        lat_ms = (r.end_time_ns - r.start_time_ns) / 1e6
        latencies.append(lat_ms)

    results.extend(latencies)


# ── Run one condition ─────────────────────────────────────────────────────────

def run_condition(
    rt1: PyTorchRuntime,
    rt2: PyTorchRuntime,
    loader1,
    loader2,
    stream1: torch.cuda.Stream | None,
    stream2: torch.cuda.Stream | None,
    n_requests: int,
    condition_name: str,
) -> tuple[list, list]:
    results1, results2 = [], []
    barrier = threading.Barrier(2)

    t1 = threading.Thread(
        target=inference_worker,
        args=(rt1, stream1, loader1, "ecgclass",     n_requests, results1, barrier),
        daemon=True,
    )
    t2 = threading.Thread(
        target=inference_worker,
        args=(rt2, stream2, loader2, "gestureclass", n_requests, results2, barrier),
        daemon=True,
    )

    print(f"\n[{condition_name}] Starting concurrent inference ({n_requests} reqs each)...")
    t1.start(); t2.start()
    t1.join();  t2.join()
    print(f"[{condition_name}] Done.")
    return results1, results2


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_requests", type=int, default=100)
    p.add_argument("--device",     type=str, default="cuda:0")
    p.add_argument("--device_id",  type=int, default=0,
                   help="CUDA device ordinal (for TPC query)")
    p.add_argument("--mode", choices=["libsmctrl", "green"], default="libsmctrl",
                   help="Isolation backend: libsmctrl (driver<=528) or green (driver 550+, CUDA 12.4+)")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_DEVICE"] = args.device

    from site_manager.config import DATASET_DIR

    print(f"[tpc_isolation] Device: {args.device}  mode: {args.mode}")
    print(f"[tpc_isolation] GPU: {torch.cuda.get_device_name(args.device)}")

    sm_count = torch.cuda.get_device_properties(args.device).multi_processor_count
    print(f"[tpc_isolation] SMs: {sm_count}")

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("[tpc_isolation] Loading datasets...")
    loader1 = build_ecg_data(DATASET_DIR)
    loader2 = build_gesture_data(DATASET_DIR)

    # ── Load models ───────────────────────────────────────────────────────────
    print("[tpc_isolation] Loading FM1: momentbase + ecgclass...")
    rt1 = make_runtime(args.device)
    rt1.load("momentbase", [{"task": "ecgclass", "type": "classification",
                              "path": "ecgclass_momentbase_mlp"}])

    print("[tpc_isolation] Loading FM2: momentsmall + gestureclass...")
    rt2 = make_runtime(args.device)
    rt2.load("momentsmall", [{"task": "gestureclass", "type": "classification",
                               "path": "gestureclass_momentsmall_mlp"}])

    # ── Baseline streams (shared, no pinning) ─────────────────────────────────
    stream1_base = torch.cuda.Stream(device=args.device)
    stream2_base = torch.cuda.Stream(device=args.device)

    # ── Condition 1: Baseline ─────────────────────────────────────────────────
    lat1_base, lat2_base = run_condition(
        rt1, rt2, loader1, loader2,
        stream1_base, stream2_base,
        args.n_requests, "baseline"
    )

    # ── Condition 2: Isolated ─────────────────────────────────────────────────
    green_partition = None

    if args.mode == "libsmctrl":
        lib = _load_libsmctrl()
        num_tpcs = _query_num_tpcs(lib, args.device_id, sm_count)
        print(f"[tpc_isolation] TPCs: {num_tpcs}")
        partitions = [
            list(range(0, num_tpcs // 2)),
            list(range(num_tpcs // 2, num_tpcs)),
        ]
        print(f"[tpc_isolation] FM1 → TPCs {partitions[0]}")
        print(f"[tpc_isolation] FM2 → TPCs {partitions[1]}")

        stream1_iso = torch.cuda.Stream(device=args.device)
        stream2_iso = torch.cuda.Stream(device=args.device)
        pin_stream_to_tpcs(lib, stream1_iso, partitions[0], num_tpcs)
        pin_stream_to_tpcs(lib, stream2_iso, partitions[1], num_tpcs)
        print("[tpc_isolation] Streams pinned via libsmctrl.")

        lat1_iso, lat2_iso = run_condition(
            rt1, rt2, loader1, loader2,
            stream1_iso, stream2_iso,
            args.n_requests, "isolated"
        )

        clear_stream_pin(lib, stream1_iso, num_tpcs)
        clear_stream_pin(lib, stream2_iso, num_tpcs)

    else:  # green
        print("[tpc_isolation] Creating green context SM partitions...")
        green_partition = GreenContextPartition(device_id=args.device_id)
        iso_streams = green_partition.create_partitions(num_partitions=2)
        stream1_iso, stream2_iso = iso_streams[0], iso_streams[1]
        print(f"[tpc_isolation] FM1 → green context stream {stream1_iso}")
        print(f"[tpc_isolation] FM2 → green context stream {stream2_iso}")

        lat1_iso, lat2_iso = run_condition(
            rt1, rt2, loader1, loader2,
            stream1_iso, stream2_iso,
            args.n_requests, "isolated"
        )

        green_partition.destroy()

    # ── Print summary ─────────────────────────────────────────────────────────
    def stats(arr, label):
        a = np.array(arr)
        print(f"  {label:30s}  p50={np.percentile(a,50):.1f}ms  "
              f"p95={np.percentile(a,95):.1f}ms  p99={np.percentile(a,99):.1f}ms  "
              f"mean={a.mean():.1f}ms")

    iso_label = "libsmctrl TPC partitions" if args.mode == "libsmctrl" else "green context SM partitions"
    print("\n=== Results ===")
    print("Baseline (shared, no pinning):")
    stats(lat1_base, "FM1 momentbase/ecgclass")
    stats(lat2_base, "FM2 momentsmall/gestureclass")
    print(f"Isolated ({iso_label}):")
    stats(lat1_iso, "FM1 momentbase/ecgclass")
    stats(lat2_iso, "FM2 momentsmall/gestureclass")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    req_path = out_dir / "requests.csv"
    with open(req_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "condition", "fm", "backbone", "task", "req_idx", "latency_ms"])
        for i, lat in enumerate(lat1_base):
            w.writerow([args.mode, "baseline", "fm1", "momentbase",  "ecgclass",     i, round(lat, 4)])
        for i, lat in enumerate(lat2_base):
            w.writerow([args.mode, "baseline", "fm2", "momentsmall", "gestureclass", i, round(lat, 4)])
        for i, lat in enumerate(lat1_iso):
            w.writerow([args.mode, "isolated", "fm1", "momentbase",  "ecgclass",     i, round(lat, 4)])
        for i, lat in enumerate(lat2_iso):
            w.writerow([args.mode, "isolated", "fm2", "momentsmall", "gestureclass", i, round(lat, 4)])

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "condition", "fm", "backbone", "task",
                    "mean_ms", "p50_ms", "p95_ms", "p99_ms"])
        for cond, fm, bb, task, arr in [
            ("baseline", "fm1", "momentbase",  "ecgclass",     lat1_base),
            ("baseline", "fm2", "momentsmall", "gestureclass", lat2_base),
            ("isolated", "fm1", "momentbase",  "ecgclass",     lat1_iso),
            ("isolated", "fm2", "momentsmall", "gestureclass", lat2_iso),
        ]:
            a = np.array(arr)
            w.writerow([args.mode, cond, fm, bb, task,
                        round(a.mean(), 3),
                        round(np.percentile(a, 50), 3),
                        round(np.percentile(a, 95), 3),
                        round(np.percentile(a, 99), 3)])

    print(f"\n[tpc_isolation] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
