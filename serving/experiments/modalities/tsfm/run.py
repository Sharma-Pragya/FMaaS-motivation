"""
TSFM microbenchmark: momentbase + ecgclass decoder, closed-loop client.

Usage (from serving/):
  python experiments/microbenchmark/tsfm/run.py [--n_requests 200] [--device cuda:0] [--run_idx 0]
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

from serving.device.runtime import PyTorchRuntime
from site_manager.config import DATASET_DIR

TASK         = "ecgclass"
DECODER_TYPE = "classification"


def build_data():
    from fmtk.datasetloaders.ecg5000 import ECG5000Dataset
    ds = ECG5000Dataset(
        {"dataset_path": f"{DATASET_DIR}/ECG5000"},
        {"task_type": "classification"},
        "test",
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    print(f"[tsfm] ECG5000 test set: {len(ds)} samples")
    return loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone",   type=str, default="momentbase")
    p.add_argument("--n_requests", type=int, default=200)
    p.add_argument("--device",     type=str, default="cuda:0")
    p.add_argument("--run_idx",    type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_DEVICE"] = args.device
    gpu_name = torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else args.device

    backbone     = args.backbone
    decoder_path = f"{TASK}_{backbone}_mlp"

    loader    = build_data()
    data_iter = iter(loader)
    runtime   = PyTorchRuntime()

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"[tsfm] Loading {backbone} + {TASK} decoder...")
    op_log = runtime.load(
        backbone,
        [{"task": TASK, "type": DECODER_TYPE, "path": decoder_path}],
    )
    load_summary = op_log.summary()

    # ── Inference ─────────────────────────────────────────────────────────────
    latencies_ms = []
    backbone_ms  = []
    decoder_ms   = []
    swap_ms      = []
    gpu_peaks_mb = []

    print(f"[tsfm] Running {args.n_requests} closed-loop requests...")
    for i in range(args.n_requests):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        x_i = batch["x"].numpy().astype(np.float32)
        m_i = batch["mask"].numpy().astype(np.float32) if "mask" in batch else None

        result = runtime.run_batch(x_i, [TASK], mask=m_i)
        lat = (result.end_time_ns - result.start_time_ns) / 1e6
        latencies_ms.append(lat)
        backbone_ms.append(result.proc_time_ns / 1e6)
        decoder_ms.append(sum(result.decoder_time_ns) / 1e6)
        swap_ms.append(sum(result.swap_time_ns) / 1e6)
        gpu_peaks_mb.append(result.gpu_alloc_peak_mb)

    lat_arr = np.array(latencies_ms)

    print(f"[tsfm] Done. "
          f"p50={np.percentile(lat_arr,50):.1f}ms  "
          f"p99={np.percentile(lat_arr,99):.1f}ms")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    def _s(summary, section, metric):
        return summary.get(section, {}).get(metric, None)

    req_path = out_dir / "requests.csv"
    write_header = not req_path.exists()
    with open(req_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["run_idx", "req_idx",
                        "latency_ms", "backbone_ms", "decoder_ms", "swap_ms", "gpu_peak_mb"])
        for i, (lat, bb, dec, sw, gpu) in enumerate(
            zip(latencies_ms, backbone_ms, decoder_ms, swap_ms, gpu_peaks_mb)
        ):
            w.writerow([args.run_idx, i,
                        round(lat, 4), round(bb, 4), round(dec, 4), round(sw, 4), round(gpu, 3)])

    summary_path = out_dir / "summary.csv"
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "run_idx", "backbone", "task", "n_requests", "device",
                # load phase — matches FMTK schema
                "backbone memory(MB)", "backbone load time(ms)",
                "decoder memory(MB)", "decoder load time(ms)",
                "encoder memory(MB)", "encoder load time(ms)",
                "adapter memory(MB)", "adapter load time(ms)",
                "train time(ms)", "train mem peak(MB)", "train energy(mJ)",
                "inference time(ms)", "inference mem peak(MB)", "inference energy(mJ)",
                "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms",
                "backbone_mean_ms", "decoder_mean_ms", "swap_mean_ms", "gpu_peak_mean_mb",
            ])
        w.writerow([
            args.run_idx, backbone, TASK, args.n_requests, gpu_name,
            _s(load_summary, "load_backbone",               "gpu peak"),
            _s(load_summary, "load_backbone",               "wall time"),
            _s(load_summary, f"add_decoder_{decoder_path}", "gpu peak"),
            _s(load_summary, f"add_decoder_{decoder_path}", "wall time"),
            None, None,   # encoder memory, encoder load time (N/A)
            None, None,   # adapter memory, adapter load time (N/A)
            None, None, None,  # train time, train mem peak, train energy (N/A)
            round(lat_arr.mean(), 3),        # inference time(ms) = mean per-request latency
            round(np.mean(gpu_peaks_mb), 3), # inference mem peak(MB) = mean per-batch GPU peak
            None,                            # inference energy(mJ) — not measured
            round(lat_arr.mean(), 3),
            round(np.percentile(lat_arr, 50), 3),
            round(np.percentile(lat_arr, 95), 3),
            round(np.percentile(lat_arr, 99), 3),
            round(np.mean(backbone_ms), 3),
            round(np.mean(decoder_ms), 3),
            round(np.mean(swap_ms), 3),
            round(np.mean(gpu_peaks_mb), 3),
        ])

    print(f"[tsfm] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
