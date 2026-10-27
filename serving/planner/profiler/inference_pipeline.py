"""Profiler inference pipeline.

Uses `PyTorchRuntime.run_batch` (same codepath as
`serving/experiments/batching_profiles/tsfm/run.py`) so backbone/decoder
timings match production numbers. Dispatches on `task_info['task_type']`:

  - TSFM / CV     : numpy `x` (+ optional `mask`) → run_batch(x, tasks, mask=...)
  - VLM           : image tensor + prompt string → run_batch(x, tasks, questions=...)
  - LLM (HF-text) : prompt string only          → run_batch(None, tasks, questions=...)

vLLM-backed backbones (`_VLLM_BACKED_BACKBONES`) instead go through
`VLLMRuntime.infer`, driven **strictly sequentially** under `asyncio.run`:
one awaited request at a time, never overlapping. That is deliberate. The
serving experiments (RTVSntask) drive vLLM open-loop with Poisson arrivals,
where requests overlap and get continuously batched, so their per-request
latency includes self-interference — at N=1/2 RPS the measured in-flight
concurrency is 1.06 (Qwen2.5-3B) and 2.25 (Qwen2-VL-7B), inflating mean
latency ~35% over an isolated request. The other rows of the paper's
backbone-cost table are closed-loop batch-1 costs, so the vLLM rows have to
be measured the same way to belong in the same column.

The `runtime` CSV column records which path produced each row; do not
aggregate across runtimes without it.

Adapter handling mirrors `ModelLoader.load_models`: a path entry like
`{'adapter': 'lora_vlm', 'path': '<saved_adapter_dir>'}` is forwarded as
`{'task': ..., 'adapter': 'lora_vlm', 'path': '...'}` so the runtime
loads the LoRA weights and `pipeline.set_adapter` fires during inference.
"""

import asyncio
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


# Backbones that ModelLoader._build_pipeline routes to a vLLM class. These run
# through VLLMRuntime instead of PyTorchRuntime. Keep in sync with
# serving/device/model_loader.py::_build_pipeline — a name here that the loader
# sends to a torch backbone (or vice versa) silently mismatches the runtime.
_VLLM_BACKED_BACKBONES = {
    "qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
    "qwen2-vl-2b", "qwen2-vl-7b",
    "phi3-mini", "phi3-small", "phi3-medium",
    "phi-vllm",
}

_LLM_TASK_TYPES = {
    'sentiment', 'text_classification', 'ner', 'qa',
    'summarization', 'translation', 'math_reasoning',
    'code_generation', 'reading_comprehension', 'fact_verification',
}


def _lora_weights_mb(runtime, task_name):
    """Size of the registered LoRA's weight file, in MB.

    Under vLLM the adapter is never materialised on the GPU at load time
    (register_adapter only builds a LoRARequest), so there is no allocator
    reading to take. The safetensors file is the same quantity the PyTorch
    path reports as adapter memory.
    """
    try:
        loras = runtime.pipeline.model_instance._loras
        name = runtime._loader.adapters.get(task_name) or next(iter(loras))
        weights = Path(loras[name].lora_path) / "adapter_model.safetensors"
        return round(weights.stat().st_size / 1e6, 3)
    except Exception as exc:
        print(f"     [warn] could not size LoRA for task {task_name!r}: {exc}")
        return None


def _llm_collate_fn(batch):
    """List-based collate for LLM datasets (`question` is str, `y` is anything)."""
    return {
        "question": [b["question"] for b in batch],
        "y":        [b["y"]        for b in batch],
    }


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
        self.is_llm = task_type in _LLM_TASK_TYPES

        # VLM: resolve the task-specific prompt and pull in the VLM collate.
        self._collate_fn = None
        if self.is_vlm:
            from fmtk.tasks.vlm_utils import TASK_REGISTRY
            from fmtk.datasetloaders.vlm_dataset import vlm_collate_fn
            vlm_task_key = self.task_cfg['vlm_task_key']
            self.task_cfg['prompt'] = TASK_REGISTRY[vlm_task_key]['prompt']
            self._collate_fn = vlm_collate_fn
        elif self.is_llm:
            self._collate_fn = _llm_collate_fn

        control_randomness(13)

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
            collate_fn=self._collate_fn,
        )

    def _build_task_specs(self, path):
        """Build ModelLoader-compatible task_specs from a config path entry.

        Returns [] when the path has neither a decoder nor an adapter
        (e.g. LLM `{'path': ''}`), which means load_models loads only the
        backbone with an empty decoder/adapter map.
        """
        spec = {"task": self.task_name, "path": path.get('path', '')}
        has_load = False
        if path.get('decoder') is not None:
            spec["type"] = self.task_cfg['task_type']
            has_load = True
        if path.get('adapter') is not None:
            spec["adapter"] = path['adapter']
            has_load = True
        return [spec] if has_load else []

    def run(self):
        # Import here so CUDA_DEVICE env + TPC partition are applied before
        # ModelLoader picks up the device.
        from device.runtime import PyTorchRuntime, VLLMRuntime

        backbone_name = self.pipeline['backbone']
        is_vllm = backbone_name in _VLLM_BACKED_BACKBONES
        runtime_name = "vllm" if is_vllm else "pytorch"

        os.environ.setdefault("CUDA_DEVICE", str(self.device))
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''
        modality = 'VLM' if self.is_vlm else ('LLM' if self.is_llm else 'TSFM/CV')
        print(f"[profiler] task={self.task_name} backbone={backbone_name} "
              f"modality={modality} runtime={runtime_name}")

        for path in self.pipeline['paths']:
            decoder_key = path.get('decoder')
            decoder_type = decoders[decoder_key]['decoder_type'] if decoder_key else None
            encoder_key = path.get('encoder')
            adapter_key = path.get('adapter')
            # path=None is intentional for timing-only TSFM runs: fmtk Pipeline.add_decoder
            # will random-init the decoder. The CSV 'decoder' column uses the real
            # decoder identity from the profiler config, not a synthesized path.
            saved_path = path.get('path', '')

            runtime = (VLLMRuntime(cuda_stream=self.cuda_stream) if is_vllm
                       else PyTorchRuntime(cuda_stream=self.cuda_stream))
            task_specs = self._build_task_specs(path)
            model_config = self.backbone_cfg.get('model_config')

            print(f"[profiler] Loading backbone={backbone_name} "
                  f"decoder={decoder_key} encoder={encoder_key} adapter={adapter_key} "
                  f"path={saved_path!r}")
            op_log = runtime.load(backbone_name, task_specs, model_config=model_config)
            load_summary = op_log.summary()

            def _s(section, metric):
                return load_summary.get(section, {}).get(metric)

            backbone_load_ms   = _s("load_backbone", "wall time")
            backbone_load_mem  = _s("load_backbone", "gpu peak")
            encoder_load_ms    = _s(f"add_encoder_{saved_path}", "wall time") if encoder_key else None
            encoder_load_mem   = _s(f"add_encoder_{saved_path}", "gpu peak") if encoder_key else None
            decoder_load_ms    = _s(f"add_decoder_{saved_path}", "wall time") if decoder_key else None
            decoder_load_mem   = _s(f"add_decoder_{saved_path}", "gpu peak") if decoder_key else None
            adapter_load_ms    = _s(f"add_adapter_{saved_path}", "wall time") if adapter_key else None
            adapter_load_mem   = _s(f"add_adapter_{saved_path}", "gpu peak") if adapter_key else None

            kv_cache_mem_mb = None
            if is_vllm:
                # vLLM allocates weights and the KV pool outside torch's caching
                # allocator, so the Logger's "gpu peak" reads near zero. Take the
                # engine's own accounting instead. Weights and pool are reported
                # separately: weights are the model's cost, the pool is a
                # deployment knob (gpu_memory_utilization) and is what actually
                # sets how many replicas fit on a card.
                backbone_load_mem = round(runtime.model_weights_bytes / 1e6, 3) or None
                kv_cache_mem_mb = runtime.kv_cache_bytes / 1e6
                kv_cache_mem_mb = round(kv_cache_mem_mb, 3) if kv_cache_mem_mb else None
                # register_adapter() only builds a LoRARequest; vLLM loads the
                # adapter lazily on first use, so there is no eager load to time.
                if adapter_key:
                    adapter_load_ms = None
                    adapter_load_mem = _lora_weights_mb(runtime, self.task_name)

            for batch_size in self.inference_batch_sizes:
                print(f"  -- inference batch_size={batch_size} tpc={self.tpc_count}")
                runner = self._run_one_vllm if is_vllm else self._run_one
                latencies_ms, backbone_ms, decoder_ms, gpu_peaks_mb = runner(
                    runtime, batch_size, n_requests,
                )
                # _run_one_vllm always measures isolated single requests, so the
                # per-sample and throughput derivations below must divide by 1
                # regardless of what the config asked for.
                if is_vllm:
                    batch_size = 1

                if not latencies_ms:
                    print(f"     [warn] no samples collected for bs={batch_size}")
                    continue

                lat_arr = np.array(latencies_ms)
                mean_lat_s = lat_arr.mean() / 1000.0
                throughput = batch_size / mean_lat_s if mean_lat_s > 0 else 0.0

                metrics = {
                    "backbone":              backbone_name,
                    "runtime":               runtime_name,
                    "decoder":               decoder_type,
                    "encoder":               encoder_key,
                    "adapter":               adapter_key,
                    "dataset_name":          self.task_cfg['datasets'][0],
                    "device":                gpu_name,
                    "tpc_count":             self.tpc_count,
                    "task_name":             self.task_name,
                    "inference_batch_size":  batch_size,
                    "n_requests":            len(latencies_ms),
                    # Generation length dominates decode-bound latency, so a
                    # latency here is uninterpretable without these two. Record
                    # them rather than leaving them to be reconstructed from
                    # logs later.
                    "max_new_tokens":        (model_config or {}).get('max_new_tokens'),
                    "ignore_eos":            (model_config or {}).get('ignore_eos'),
                    "backbone_load_time_ms": backbone_load_ms,
                    "backbone_load_mem_mb":  backbone_load_mem,
                    "kv_cache_mem_mb":       kv_cache_mem_mb,
                    "encoder_load_time_ms":  encoder_load_ms,
                    "encoder_load_mem_mb":   encoder_load_mem,
                    "decoder_load_time_ms":  decoder_load_ms,
                    "decoder_load_mem_mb":   decoder_load_mem,
                    "adapter_load_time_ms":  adapter_load_ms,
                    "adapter_load_mem_mb":   adapter_load_mem,
                    "avg_latency_ms":        round(float(lat_arr.mean()), 4),
                    "p50_latency_ms":        round(float(np.percentile(lat_arr, 50)), 4),
                    "p95_latency_ms":        round(float(np.percentile(lat_arr, 95)), 4),
                    "p99_latency_ms":        round(float(np.percentile(lat_arr, 99)), 4),
                    "backbone_mean_ms":      round(float(np.mean(backbone_ms)), 4),
                    "decoder_mean_ms":       round(float(np.mean(decoder_ms)), 4),
                    "backbone_ms_per_sample": round(float(np.mean(backbone_ms)) / batch_size, 4),
                    # Empty under vLLM: per-request torch peak is not meaningful
                    # once the KV pool is preallocated. See kv_cache_mem_mb.
                    "peak_gpu_mem_mb":       round(float(np.max(gpu_peaks_mb)), 3) if gpu_peaks_mb else None,
                    "avg_gpu_mem_mb":        round(float(np.mean(gpu_peaks_mb)), 3) if gpu_peaks_mb else None,
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

            tasks = [self.task_name] * batch_size

            if self.is_vlm:
                x_i = batch["x"].numpy().astype(np.float32)
                questions = batch["question"]
                result = runtime.run_batch(x_i, tasks, questions=questions)
            elif self.is_llm:
                questions = batch["question"]
                result = runtime.run_batch(None, tasks, questions=questions)
            else:
                x_i = batch["x"].numpy().astype(np.float32)
                m_i = batch["mask"].numpy().astype(np.float32) if "mask" in batch else None
                result = runtime.run_batch(x_i, tasks, mask=m_i)

            latencies_ms.append((result.end_time_ns - result.start_time_ns) / 1e6)
            backbone_ms.append(result.proc_time_ns / 1e6)
            decoder_ms.append(sum(result.decoder_time_ns) / 1e6)
            gpu_peaks_mb.append(result.gpu_alloc_peak_mb)

        return latencies_ms, backbone_ms, decoder_ms, gpu_peaks_mb

    def _run_one_vllm(self, runtime, batch_size, n_requests_local):
        """Closed-loop, strictly sequential vLLM measurement.

        `batch_size` is forced to 1 and each request is awaited to completion
        before the next is issued, so nothing is ever co-batched. This is what
        makes the resulting latency comparable to the PyTorch rows, which run
        one batch at a time. Do NOT convert this to asyncio.gather() to "speed
        it up" — overlapping requests is precisely the effect being excluded;
        see the module docstring.
        """
        if batch_size != 1:
            print(f"     [warn] vLLM path measures isolated requests; "
                  f"forcing batch_size {batch_size} -> 1")
        loader = self._build_loader(1)

        async def _drive():
            data_iter = iter(loader)
            latencies_ms, backbone_ms = [], []
            for req_id in range(n_requests_local):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)

                prompt = batch["question"][0]
                # VLLMRuntime.infer picks the (image, prompt) vs bare-prompt
                # call shape from the backbone's model_category; x is ignored
                # for LLMs. Qwen2VLVLLMModel._to_pil accepts the (C,H,W) tensor.
                x = batch["x"][0] if self.is_vlm else None

                result = await runtime.infer(
                    req_id, prompt, task=self.task_name, x=x,
                )
                latencies_ms.append(
                    (result["end_time_ns"] - result["start_time_ns"]) / 1e6)
                backbone_ms.append(result["proc_time_ns"] / 1e6)
            return latencies_ms, backbone_ms

        latencies_ms, backbone_ms = asyncio.run(_drive())
        # No separable decoder stage (LoRA compute is fused into the forward),
        # and no per-request torch peak once the KV pool is preallocated.
        decoder_ms = [0.0] * len(latencies_ms)
        return latencies_ms, backbone_ms, decoder_ms, []
