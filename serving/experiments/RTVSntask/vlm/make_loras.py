#!/usr/bin/env python3
"""Generate random-init PEFT LoRA adapters over Qwen2-VL for the RTVSntask/vlm experiment.

VLM analog of experiments/fair_share/llm/make_loras.py. Writes to FMTK's
*vlm* adapter path, which is where Pipeline.add_adapter resolves them from
once the backbone reports model_category='vlm':

    <FMTK>/models/vlm/finetuned/qwenvlA/   (adapter_config.json + adapter_model.safetensors)
    <FMTK>/models/vlm/finetuned/qwenvlB/

target_modules is q_proj/v_proj, matching device/config.py ADAPTERS['lora_vlm'].
In Qwen2-VL those names occur ONLY under model.layers.*.self_attn — the visual
tower uses visual.blocks.*.attn.qkv / attn.proj — so the adapters are
language-side only. That is deliberate: vLLM 0.6.6 lists exactly
{qkv_proj, o_proj, gate_up_proj, down_proj} in Qwen2VLForConditionalGeneration.
supported_lora_modules and cannot serve a LoRA over the vision encoder.

Weights are random-init: this experiment measures serving cost (adapter memory,
load time, multi-LoRA scheduling), not accuracy.

Usage (from serving/):
    python experiments/RTVSntask/vlm/make_loras.py --names qwenvlA,qwenvlB
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2VLForConditionalGeneration

QWEN_VL_IDS = {
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
}


def fmtk_finetuned_dir() -> Path:
    import fmtk.pipeline as _pl
    base = Path(_pl.__file__).resolve().parent
    return base.parent.parent / "models" / "vlm" / "finetuned"


def fmtk_pretrained_dir() -> Path:
    import fmtk.pipeline as _pl
    base = Path(_pl.__file__).resolve().parent
    return base.parent.parent / "models" / "vlm" / "pretrained"


def make_one(model_id: str, cache_dir: str, out_dir: Path, rank: int, alpha: int) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[make_loras] {out_dir} already populated — skipping.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[make_loras] Loading {model_id} on CPU (slow, ~1 min)...")
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, cache_dir=cache_dir, torch_dtype=torch.float16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, cfg)
    n_lora = sum(1 for n, _ in peft_model.named_parameters() if "lora_" in n)
    n_visual = sum(1 for n, _ in peft_model.named_parameters()
                   if "lora_" in n and "visual" in n)
    print(f"[make_loras] LoRA tensors: {n_lora} (visual-tower: {n_visual} — must be 0)")
    if n_visual:
        raise SystemExit("[make_loras] ERROR: adapter touches the vision encoder; "
                         "vLLM cannot serve it.")
    peft_model.save_pretrained(str(out_dir))
    del peft_model, base
    print(f"[make_loras] Wrote → {out_dir}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="qwen2-vl-2b", choices=list(QWEN_VL_IDS))
    p.add_argument("--names", default="qwenvlA,qwenvlB")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    args = p.parse_args()

    cache_dir = str(fmtk_pretrained_dir())
    root = fmtk_finetuned_dir()
    for name in [n.strip() for n in args.names.split(",") if n.strip()]:
        make_one(QWEN_VL_IDS[args.backbone], cache_dir, root / name, args.rank, args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
