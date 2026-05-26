#!/usr/bin/env python3
"""Generate two random-init PEFT LoRA adapters for the fair_share/llm experiment.

Writes them to FMTK's resolved adapter path so the runtime's deploy payload
(path="qwenA"/"qwenB") loads them correctly:

    <FMTK>/models/llm/finetuned/qwenA/   (adapter_config.json + adapter_model.safetensors)
    <FMTK>/models/llm/finetuned/qwenB/

Usage (from serving/):
    python experiments/fair_share/llm/make_loras.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

QWEN_IDS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b":   "Qwen/Qwen2.5-7B-Instruct",
}


def fmtk_finetuned_dir() -> Path:
    import fmtk.pipeline as _pl
    base = Path(_pl.__file__).resolve().parent
    return base.parent.parent / "models" / "llm" / "finetuned"


def make_one(model_id: str, cache_dir: str, out_dir: Path, rank: int, alpha: int) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[make_loras] {out_dir} already populated — skipping.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[make_loras] Loading {model_id} on CPU (slow, ~1 min)...")
    base = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir, torch_dtype=torch.float16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, cfg)
    peft_model.save_pretrained(str(out_dir))
    del peft_model, base
    print(f"[make_loras] Wrote → {out_dir}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="qwen2.5-1.5b", choices=list(QWEN_IDS))
    p.add_argument("--names", default="qwenA,qwenB")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    args = p.parse_args()

    import fmtk.pipeline as _pl
    cache_dir = str(Path(_pl.__file__).resolve().parent.parent.parent
                    / "models" / "llm" / "pretrained")
    root = fmtk_finetuned_dir()
    for name in [n.strip() for n in args.names.split(",") if n.strip()]:
        make_one(QWEN_IDS[args.backbone], cache_dir, root / name, args.rank, args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
