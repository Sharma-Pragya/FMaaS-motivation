#!/usr/bin/env python3
"""lora_cost_attribution/make_adapters.py — Write synthetic PEFT LoRA adapters.

Compute cost does not depend on weight values, so we skip peft/model loading
and write adapter_model.safetensors + adapter_config.json directly, with
shapes taken from the base model's HF config. Both A and B are random
(peft's default B=0 would still cost the same, but this avoids any chance of
a zero-adapter shortcut).

Produces:
    adapters/r8_0
    adapters/r64_0 ... adapters/r64_{N-1}

Usage (from serving/):
    python experiments/lora_cost_attribution/make_adapters.py \
        --model Qwen/Qwen2.5-3B --num-r64 16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig

EXP_DIR = Path(__file__).resolve().parent

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def module_shapes(cfg):
    """(in_features, out_features) for each target module of a Qwen2 layer."""
    h = cfg.hidden_size
    head_dim = h // cfg.num_attention_heads
    kv = cfg.num_key_value_heads * head_dim
    inter = cfg.intermediate_size
    return {
        "q_proj":    ("self_attn", h, h),
        "k_proj":    ("self_attn", h, kv),
        "v_proj":    ("self_attn", h, kv),
        "o_proj":    ("self_attn", h, h),
        "gate_proj": ("mlp", h, inter),
        "up_proj":   ("mlp", h, inter),
        "down_proj": ("mlp", inter, h),
    }


def write_adapter(out_dir: Path, model: str, cfg, rank: int, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer in range(cfg.num_hidden_layers):
        for name, (block, fin, fout) in module_shapes(cfg).items():
            prefix = f"base_model.model.model.layers.{layer}.{block}.{name}"
            tensors[f"{prefix}.lora_A.weight"] = (
                torch.randn(rank, fin, generator=g) * 0.01).to(torch.bfloat16)
            tensors[f"{prefix}.lora_B.weight"] = (
                torch.randn(fout, rank, generator=g) * 0.01).to(torch.bfloat16)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "adapter_model.safetensors"))
    (out_dir / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": model,
        "r": rank,
        "lora_alpha": 2 * rank,
        "lora_dropout": 0.0,
        "target_modules": TARGET_MODULES,
        "bias": "none",
        "fan_in_fan_out": False,
        "use_rslora": False,
        "use_dora": False,
        "modules_to_save": None,
    }, indent=2))
    print(f"wrote {out_dir}  (rank={rank}, {len(tensors)} tensors)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--num-r64", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=EXP_DIR / "adapters")
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model)
    write_adapter(args.out_dir / "r8_0", args.model, cfg, rank=8, seed=8)
    for i in range(args.num_r64):
        write_adapter(args.out_dir / f"r64_{i}", args.model, cfg,
                      rank=64, seed=6400 + i)


if __name__ == "__main__":
    main()
