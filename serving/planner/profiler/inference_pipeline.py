"""Profiler inference pipeline.

TSFM / CV paths use `PyTorchRuntime.run_batch` (same codepath as
`serving/experiments/batching_profiles/tsfm/run.py`) so backbone/decoder
timings match production numbers. VLM and LLM paths are currently not
supported in this profiler.
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from component_loader import get_model_class  # noqa: F401  (imported for side effects in config)
from dataset_loader import get_dataset_class
from fmtk.utils import control_randomness
from config import *

# serving/ must be on sys.path so `device.runtime` imports cleanly when this
# module is invoked as `worker.py` from the profiler directory.
_SERVING_DIR = Path(__file__).resolve().parents[2]
if str(_SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVING_DIR))


class InferencePipeline:
    def __init__(self, task_name, task_info, pipeline, log_file,
                 tpc_count=None, cuda_stream=None):
        self.tpc_count = tpc_count
        self.cuda_stream = cuda_stream
        self.backbone_cfg = backbones[pipeline['backbone']]
        self.dataset_cfg = datasets[task_info['datasets'][0]]
        self.task_cfg = task_info
        self.task_name = task_name
        self.pipeline = pipeline
        self.model_name = self.backbone_cfg['model_name']
        self.model_type = self.backbone_cfg['model_type']
        self.device = device
        self.log_file = log_file

        task_type = task_info.get('task_type')
        self.is_vlm = task_type == 'vlm'
        self.is_llm = task_type in (
            'sentiment', 'text_classification', 'ner', 'qa',
            'summarization', 'translation', 'math_reasoning',
            'code_generation', 'reading_comprehension', 'fact_verification',
        )
        if self.is_vlm or self.is_llm:
            raise NotImplementedError(
                "VLM/LLM profiling not supported in the runtime-based profiler. "
                "Use the fmtk-based path if accuracy metrics are required."
            )

        control_randomness(13)

        # Normalize batch_size to a list so we can sweep.
        raw_bs = self.task_cfg['inference_config']['batch_size']
        self.inference_batch_sizes = raw_bs if isinstance(raw_bs, list) else [raw_bs]

        dataset_class = get_dataset_class(self.dataset_cfg['dataset_type'])
        self.dataset_instance_test = dataset_class(self.dataset_cfg, self.task_cfg, split='test')

    def _build_loader(self, batch_size):
        return DataLoader(
            self.dataset_instance_test,
            batch_size=batch_size,
            shuffle=self.task_cfg['inference_config'].get('shuffle', False),
            drop_last=True,
        )

    def run(self):
        # Import here so CUDA_DEVICE env + TPC partition are applied before
        # ModelLoader picks up the device.
        from device.runtime import PyTorchRuntime

        os.environ.setdefault("CUDA_DEVICE", str(self.device))
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''
        backbone_name = self.pipeline['backbone']

        for path in self.pipeline['paths']:
            decoder_key = path.get('decoder')
            decoder_type = decoders[decoder_key]['decoder_type'] if decoder_key else None
            # path=None is intentional for timing-only runs: fmtk Pipeline.add_decoder
            # will random-init the decoder. The CSV 'decoder' column uses the real
            # decoder identity from the profiler config, not a synthesized path.
            decoder_path = path.get('path')

            # Fresh runtime per pipeline path so backbone/decoder load timings
            # are attributable.
            runtime = PyTorchRuntime(cuda_stream=self.cuda_stream)
            decoder_specs = [{
                "task": self.task_name,
                "type": self.task_cfg['task_type'],
                "path": decoder_path,
            }]
            print(f"[profiler] Loading backbone={backbone_name} decoder={decoder_path}")
            op_log = runtime.load(backbone_name, decoder_specs)
            load_summary = op_log.summary()

            def _s(section, metric):
                return load_summary.get(section, {}).get(metric)

            backbone_load_ms   = _s("load_backbone", "wall time")
            backbone_load_mem  = _s("load_backbone", "gpu peak")
            decoder_load_ms    = _s(f"add_decoder_{decoder_path}", "wall time")
            decoder_load_mem   = _s(f"add_decoder_{decoder_path}", "gpu peak")

            for batch_size in self.inference_batch_sizes:
                print(f"  -- inference batch_size={batch_size} tpc={self.tpc_count}")
                latencies_ms, backbone_ms, decoder_ms, gpu_peaks_mb = self._run_one(
                    runtime, batch_size, n_requests,
                )

                if not latencies_ms:
                    print(f"     [warn] no samples collected for bs={batch_size}")
                    continue

                lat_arr = np.array(latencies_ms)
                mean_lat_s = lat_arr.mean() / 1000.0
                throughput = batch_size / mean_lat_s if mean_lat_s > 0 else 0.0

                metrics = {
                    "backbone":              backbone_name,
                    "decoder":               decoder_type,
                    "encoder":               path.get('encoder'),
                    "adapter":               path.get('adapter'),
                    "dataset_name":          self.task_cfg['datasets'][0],
                    "device":                gpu_name,
                    "tpc_count":             self.tpc_count,
                    "task_name":             self.task_name,
                    "inference_batch_size":  batch_size,
                    "n_requests":            len(latencies_ms),
                    "backbone_load_time_ms": backbone_load_ms,
                    "backbone_load_mem_mb":  backbone_load_mem,
                    "decoder_load_time_ms":  decoder_load_ms,
                    "decoder_load_mem_mb":   decoder_load_mem,
                    "avg_latency_ms":        round(float(lat_arr.mean()), 4),
                    "p50_latency_ms":        round(float(np.percentile(lat_arr, 50)), 4),
                    "p95_latency_ms":        round(float(np.percentile(lat_arr, 95)), 4),
                    "p99_latency_ms":        round(float(np.percentile(lat_arr, 99)), 4),
                    "backbone_mean_ms":      round(float(np.mean(backbone_ms)), 4),
                    "decoder_mean_ms":       round(float(np.mean(decoder_ms)), 4),
                    "backbone_ms_per_sample": round(float(np.mean(backbone_ms)) / batch_size, 4),
                    "peak_gpu_mem_mb":       round(float(np.max(gpu_peaks_mb)), 3),
                    "avg_gpu_mem_mb":        round(float(np.mean(gpu_peaks_mb)), 3),
                    "throughput_rps":        round(float(throughput), 3),
                }

                write_header = not os.path.exists(self.log_file)
                with open(self.log_file, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(metrics)

    def _run_one(self, runtime, batch_size, n_requests_local):
        loader = self._build_loader(batch_size)
        data_iter = iter(loader)

        latencies_ms, backbone_ms, decoder_ms, gpu_peaks_mb = [], [], [], []
        for _ in range(n_requests_local):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            x_i = batch["x"].numpy().astype(np.float32)
            m_i = batch["mask"].numpy().astype(np.float32) if "mask" in batch else None

            result = runtime.run_batch(x_i, [self.task_name] * batch_size, mask=m_i)
            latencies_ms.append((result.end_time_ns - result.start_time_ns) / 1e6)
            backbone_ms.append(result.proc_time_ns / 1e6)
            decoder_ms.append(sum(result.decoder_time_ns) / 1e6)
            gpu_peaks_mb.append(result.gpu_alloc_peak_mb)

        return latencies_ms, backbone_ms, decoder_ms, gpu_peaks_mb
