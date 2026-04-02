"""
TSFM batching profile: momentbase + ecgclass, sweep batch sizes.

Usage (from serving/):
  python experiments/batching_profiles/tsfm/run.py [--batch_sizes 1,2,4,8,16,32] \
      [--n_requests 200] [--device cuda:0] [--run_idx 0]
"""

import sys
import os
import argparse
import csv
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

SERVING_DIR = Path(__file__).resolve().parents[3]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

from device.runtime import PyTorchRuntime
from site_manager.config import DATASET_DIR

BACKBONE     = "momentbase"
TASK         = "ecgclass"
DECODER_TYPE = "classification"
DECODER_PATH = f"{TASK}_{BACKBONE}_mlp"


def build_loader(batch_size: int):
    from fmtk.datasetloaders.ecg5000 import ECG5000Dataset
    ds = ECG5000Dataset(
        {"dataset_path": f"{DATASET_DIR}/ECG5000"},
        {"task_type": "classification"},
        "test",
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_sizes", type=str, default="1,2,4,8,16,32,64,128")
    p.add_argument("--n_requests",  type=int, default=200)
    p.add_argument("--device",      type=str, default="cuda:0")
    p.add_argument("--run_idx",     type=int, default=0)
    return p.parse_args()


def run_batch_size(runtime, batch_size, n_requests):
    loader    = build_loader(batch_size)
    data_iter = iter(loader)

    latencies_ms = []
    backbone_ms  = []
    decoder_ms   = []
    gpu_peaks_mb = []

    for _ in range(n_requests):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x_i = batch["x"].numpy().astype(np.float32)
        m_i = batch["mask"].numpy().astype(np.float32) if "mask" in batch else None

        result = runtime.run_batch(x_i, [TASK] * batch_size, mask=m_i)
        lat = (result.end_time_ns - result.start_time_ns) / 1e6
        latencies_ms.append(lat)
        backbone_ms.append(result.proc_time_ns / 1e6)
        decoder_ms.append(sum(result.decoder_time_ns) / 1e6)
        gpu_peaks_mb.append(result.gpu_alloc_peak_mb)

    return latencies_ms, backbone_ms, decoder_ms, gpu_peaks_mb


def main():
    args = parse_args()
    os.environ["CUDA_DEVICE"] = args.device
    gpu_name = torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else args.device
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    req_path     = out_dir / "requests.csv"
    summary_path = out_dir / "summary.csv"
    write_req_hdr = not req_path.exists()
    write_sum_hdr = not summary_path.exists()

    # Load model once, reuse across batch sizes
    print(f"[tsfm] Loading {BACKBONE} + {TASK} decoder on {args.device}...")
    runtime = PyTorchRuntime()
    op_log = runtime.load(
        BACKBONE,
        [{"task": TASK, "type": DECODER_TYPE, "path": DECODER_PATH}],
    )
    load_summary = op_log.summary()

    def _s(section, metric):
        return load_summary.get(section, {}).get(metric, None)

    with open(req_path, "a", newline="") as rf, \
         open(summary_path, "a", newline="") as sf:

        rw = csv.writer(rf)
        sw = csv.writer(sf)

        if write_req_hdr:
            rw.writerow(["run_idx", "device", "backbone", "task", "batch_size", "req_idx",
                         "latency_ms", "backbone_ms", "decoder_ms", "gpu_peak_mb"])
        if write_sum_hdr:
            sw.writerow([
                "run_idx", "backbone", "task", "batch_size", "n_requests", "device",
                "backbone_load_time_ms", "model_load_mem_mb",
                "decoder_load_time_ms", "decoder_load_mem_mb",
                "avg_latency_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms",
                "backbone_mean_ms", "decoder_mean_ms",
                "peak_gpu_mem_mb", "avg_gpu_mem_mb",
                "throughput_rps", "avg_batch_size", "mixed_batch_fraction", "batch_count",
                "backbone_ms_per_sample",
            ])

        for bs in batch_sizes:
            print(f"[tsfm] batch_size={bs}  ({args.n_requests} requests)...")
            lats, bbs, decs, gpus = run_batch_size(runtime, bs, args.n_requests)
            lat_arr = np.array(lats)

            for i, (lat, bb, dec, gpu) in enumerate(zip(lats, bbs, decs, gpus)):
                rw.writerow([args.run_idx, gpu_name, BACKBONE, TASK, bs, i,
                             round(lat, 4), round(bb, 4), round(dec, 4), round(gpu, 3)])

            mean_lat_s = lat_arr.mean() / 1000.0
            throughput = bs / mean_lat_s if mean_lat_s > 0 else 0.0

            sw.writerow([
                args.run_idx, BACKBONE, TASK, bs, args.n_requests, gpu_name,
                _s("load_backbone",               "wall time"),
                _s("load_backbone",               "gpu peak"),
                _s(f"add_decoder_{DECODER_PATH}", "wall time"),
                _s(f"add_decoder_{DECODER_PATH}", "gpu peak"),
                round(lat_arr.mean(), 3),
                round(np.percentile(lat_arr, 50), 3),
                round(np.percentile(lat_arr, 95), 3),
                round(np.percentile(lat_arr, 99), 3),
                round(np.mean(bbs), 3),
                round(np.mean(decs), 3),
                round(max(gpus), 3),        # peak_gpu_mem_mb — max over all batches
                round(np.mean(gpus), 3),    # avg_gpu_mem_mb  — mean over all batches
                round(throughput, 3),
                bs,                         # avg_batch_size (fixed, no dynamic batching)
                0.0,                        # mixed_batch_fraction (single task, always 0)
                args.n_requests,            # batch_count
                round(np.mean(bbs) / bs, 4),  # backbone_ms_per_sample
            ])
            print(f"[tsfm]   p50={np.percentile(lat_arr,50):.1f}ms  "
                  f"p99={np.percentile(lat_arr,99):.1f}ms  "
                  f"throughput={throughput:.1f} samples/s")

    print(f"[tsfm] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
