"""
VLM microbenchmark: phi3-mini via VLLMRuntime, closed-loop client.

Uses real OCR dataset (FMTK vlm/ocr) with the standard OCR prompt.

Usage (from serving/):
  python experiments/microbenchmark/vlm/run.py [--n_requests 100] [--device cuda:0] [--max_tokens 64] [--run_idx 0]
"""

import sys
import os
import argparse
import csv
import asyncio
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

SERVING_DIR = Path(__file__).resolve().parents[3]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

from serving.device.runtime import PyTorchRuntime

FMTK_DIR    = Path(__file__).resolve().parents[5] / "FMTK"
OCR_DATASET = str(FMTK_DIR / "dataset" / "vlm" / "ocr")


def build_data():
    from fmtk.datasetloaders.vlm_dataset import VLMDataset, vlm_collate_fn
    from fmtk.tasks.vlm_utils import TASK_REGISTRY
    prompt = TASK_REGISTRY["ocr"]["prompt"]
    ds = VLMDataset(
        {"dataset_path": OCR_DATASET, "json_file": "labels.json"},
        {"task_type": "vlm", "prompt": prompt},
        split="test",
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=vlm_collate_fn)
    print(f"[vlm] OCR dataset: {len(ds)} samples")
    return loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone",     type=str,  default="phi")
    p.add_argument("--n_requests",   type=int,  default=1)
    p.add_argument("--device",       type=str,  default="cuda:0")
    p.add_argument("--max_tokens",   type=int,  default=64)
    p.add_argument("--run_idx",      type=int,  default=0)
    p.add_argument("--adapter_path", type=str,  default=None,
                   help="Saved LoRA adapter name under models/vlm/finetuned/ (e.g. vlm_ocr_phi_lora)")
    return p.parse_args()


async def run_benchmark(args):
    gpu_name = torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else args.device
    loader    = build_data()
    data_iter = iter(loader)
    # runtime   = VLLMRuntime()
    runtime = PyTorchRuntime()

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"[vlm] Loading {args.backbone}...")
    op_log = runtime.load(args.backbone, [], device=args.device,
                          model_config={"max_tokens": args.max_tokens})

    adapter_summary = {}
    if args.adapter_path:
        from peft import LoraConfig
        print(f"[vlm] Loading adapter: {args.adapter_path}")
        peft_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                              lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        runtime.pipeline.logger = runtime.logger
        runtime.pipeline.add_adapter(peft_cfg, load=True, train=False, path=args.adapter_path)
        adapter_summary = runtime.logger.summary()

    load_summary = op_log.summary()

    # ── Inference ─────────────────────────────────────────────────────────────
    latencies_ms  = []
    backbone_ms   = []
    output_lens   = []
    gpu_peaks_mb  = []

    print(f"[vlm] Running {args.n_requests} closed-loop requests...")
    for i in range(args.n_requests):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        image  = batch["x"][0]
        prompt = batch["question"][0]

        result = runtime.run_vlm(image=image, prompt=prompt)
        lat = (result.end_time_ns - result.start_time_ns) / 1e6
        latencies_ms.append(lat)
        backbone_ms.append(result.proc_time_ns / 1e6)
        output_lens.append(len(result.outputs[0].split()) if result.outputs[0] else 0)
        gpu_peaks_mb.append(result.gpu_alloc_peak_mb)

    lat_arr = np.array(latencies_ms)

    print(f"[vlm] Done. "
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
            w.writerow(["run_idx", "req_idx", "latency_ms", "output_words"])
        for i, (lat, olen) in enumerate(zip(latencies_ms, output_lens)):
            w.writerow([args.run_idx, i, round(lat, 4), olen])

    summary_path = out_dir / "summary.csv"
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "run_idx", "backbone", "task", "n_requests", "max_tokens", "device",
                "backbone memory(MB)", "backbone load time(ms)",
                "decoder memory(MB)", "decoder load time(ms)",
                "encoder memory(MB)", "encoder load time(ms)",
                "adapter memory(MB)", "adapter load time(ms)",
                "train time(ms)", "train mem peak(MB)", "train energy(mJ)",
                "inference time(ms)", "inference mem peak(MB)", "inference energy(mJ)",
                "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms",
                "backbone_mean_ms", "avg_output_words",
            ])
        w.writerow([
            args.run_idx, args.backbone, "ocr", args.n_requests, args.max_tokens, gpu_name,
            _s(load_summary, "load_backbone", "gpu peak"),
            _s(load_summary, "load_backbone", "wall time"),
            None, None,
            None, None,
            _s(adapter_summary, f"add_adapter_{args.adapter_path}", "gpu peak"),
            _s(adapter_summary, f"add_adapter_{args.adapter_path}", "wall time"),
            None, None, None,
            round(lat_arr.mean(), 3),
            round(np.mean(gpu_peaks_mb), 2),
            None,
            round(lat_arr.mean(), 3),
            round(np.percentile(lat_arr, 50), 3),
            round(np.percentile(lat_arr, 95), 3),
            round(np.percentile(lat_arr, 99), 3),
            round(np.mean(backbone_ms), 3),
            round(np.mean(output_lens), 2),
        ])

    print(f"[vlm] Results saved to {out_dir}")


def main():
    args = parse_args()
    os.environ["CUDA_DEVICE"] = args.device
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
