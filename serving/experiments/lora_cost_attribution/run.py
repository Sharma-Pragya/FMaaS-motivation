#!/usr/bin/env python3
"""lora_cost_attribution — Decode cost of a fixed batch under LoRA configs.

One invocation = one fresh vLLM offline engine, one (batch size, config)
point. All B requests are submitted at once with max_num_seqs=B, fixed
prompt length and fixed output length (ignore_eos), so every decode step
runs at exactly batch size B and the only thing that varies is the batch's
adapter contents.

Configs:
    A    enable_lora=False, no adapters                 pure base cost
    B    enable_lora=True (rank 64), no adapters        overhead of LoRA enabled
    C    rank-8 engine, all requests on one r8 adapter  rank-8 cost
    C64  rank-64 engine, all requests on one r8 adapter rank padding effect
    D    rank-64 engine, all requests on one r64        rank-64 cost
    E    half no-adapter, half r64                      do base reqs pay?
    F    k distinct r64 adapters, round-robin (--k)     diversity effect

Appends one JSON line per run to <exp-dir>/raw.jsonl.

Usage (from serving/):
    python experiments/lora_cost_attribution/run.py --batch-size 8 --config D
    python experiments/lora_cost_attribution/run.py --batch-size 32 --config F --k 4
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest

EXP_DIR = Path(__file__).resolve().parent

MAX_LORAS = 16   # fixed across all LoRA engines so only batch contents vary


def engine_kwargs(config: str, batch_size: int, prompt_len: int,
                  output_len: int) -> dict:
    max_model_len = prompt_len + output_len
    kw = dict(
        max_num_seqs=batch_size,
        max_model_len=max_model_len,
        # Whole batch prefills in one step -> every decode step is at size B.
        max_num_batched_tokens=max(max_model_len, batch_size * prompt_len),
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
    )
    if config == "A":
        kw["enable_lora"] = False
    else:
        kw.update(enable_lora=True, max_loras=MAX_LORAS,
                  max_lora_rank=8 if config == "C" else 64)
    return kw


def lora_assignment(config: str, batch_size: int, k: int,
                    adapter_dir: Path) -> list:
    """Per-request LoRARequest (or None), length batch_size."""
    def req(name: str, idx: int) -> LoRARequest:
        return LoRARequest(name, idx + 1, str(adapter_dir / name))

    if config in ("A", "B"):
        return [None] * batch_size
    if config in ("C", "C64"):
        return [req("r8_0", 0)] * batch_size
    if config == "D":
        return [req("r64_0", 0)] * batch_size
    if config == "E":
        half = batch_size // 2
        return [None] * (batch_size - half) + [req("r64_0", 0)] * half
    if config == "F":
        pool = [req(f"r64_{i}", i) for i in range(k)]
        return [pool[i % k] for i in range(batch_size)]
    raise ValueError(config)


def make_prompts(n: int, prompt_len: int, seed: int) -> list:
    # Distinct random token ids per request; stay clear of special tokens.
    rng = random.Random(seed)
    return [TokensPrompt(prompt_token_ids=[rng.randint(1000, 30000)
                                           for _ in range(prompt_len)])
            for _ in range(n)]


def run_batch(llm, prompts, sp, loras) -> dict:
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, lora_request=loras, use_tqdm=False)
    wall = time.perf_counter() - t0

    n_out = [len(o.outputs[0].token_ids) for o in outs]
    assert all(n == sp.max_tokens for n in n_out), n_out

    # The offline LLM runs with disable_log_stats=True, so last_token_time is
    # never advanced past arrival; use finished_time for the decode span.
    ttft, tpot = [], []
    for o in outs:
        m = o.metrics
        if m is None or m.first_token_time is None:
            continue
        ttft.append(m.first_token_time - m.arrival_time)
        if m.finished_time is not None and m.finished_time > m.first_token_time:
            tpot.append((m.finished_time - m.first_token_time) / (sp.max_tokens - 1))
    ttft_s = statistics.median(ttft) if ttft else 0.0
    if tpot:
        tpot_s, source = statistics.median(tpot), "finished_time"
    else:
        # Whole batch decodes in lockstep, so wall minus prefill is the decode span.
        tpot_s, source = (wall - ttft_s) / (sp.max_tokens - 1), "wall"
    return {
        "wall_s": wall,
        "ttft_ms_median": ttft_s * 1e3,
        "tpot_ms_median": tpot_s * 1e3,
        "tpot_source": source,
        "tokens_per_s": sum(n_out) / wall,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--config", required=True,
                    choices=["A", "B", "C", "C64", "D", "E", "F"])
    ap.add_argument("--k", type=int, default=1, help="distinct adapters for F")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--dtype", default="auto",
                    help="'half' on pre-Ampere GPUs (e.g. T4) without bf16")
    ap.add_argument("--adapter-dir", type=Path, default=EXP_DIR / "adapters")
    ap.add_argument("--exp-dir", type=Path, default=EXP_DIR / "results")
    args = ap.parse_args()

    if args.config == "F" and not 1 <= args.k <= min(args.batch_size, MAX_LORAS):
        raise SystemExit(f"F needs 1 <= k <= min(B, {MAX_LORAS})")
    if args.config == "E" and args.batch_size < 2:
        raise SystemExit("E needs batch size >= 2")

    # Pre-Ampere GPUs (T4, V100) have no bf16; Qwen2.5's default dtype is bf16.
    if args.dtype == "auto" and torch.cuda.get_device_capability()[0] < 8:
        args.dtype = "half"
    print(f"dtype={args.dtype} on {torch.cuda.get_device_name()}")

    llm = LLM(model=args.model, seed=0, dtype=args.dtype,
              gpu_memory_utilization=args.gpu_memory_utilization,
              **engine_kwargs(args.config, args.batch_size,
                              args.prompt_len, args.output_len))
    sp = SamplingParams(temperature=0.0, max_tokens=args.output_len,
                        min_tokens=args.output_len, ignore_eos=True)
    loras = lora_assignment(args.config, args.batch_size, args.k,
                            args.adapter_dir)
    if all(l is None for l in loras):
        loras = None

    # Warmup also pulls adapters into GPU slots, so load cost is excluded.
    run_batch(llm, make_prompts(args.batch_size, args.prompt_len, seed=999),
              sp, loras)

    reps = [run_batch(llm, make_prompts(args.batch_size, args.prompt_len, seed=r),
                      sp, loras)
            for r in range(args.reps)]

    row = {
        "model": args.model,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "config": args.config,
        "k": args.k if args.config == "F" else None,
        "prompt_len": args.prompt_len,
        "output_len": args.output_len,
        "reps": reps,
        **{f"{key}_median": statistics.median(r[key] for r in reps)
           for key in ("wall_s", "tokens_per_s")},
        "tpot_ms_median": statistics.median(r["tpot_ms_median"] for r in reps),
        "tpot_source": reps[0]["tpot_source"],
    }
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    with open(args.exp_dir / "raw.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")

    label = args.config + (f"(k={args.k})" if args.config == "F" else "")
    print(f"[B={args.batch_size} {label}] wall={row['wall_s_median']:.3f}s  "
          f"tpot={row['tpot_ms_median']:.2f}ms ({row['tpot_source']})  tok/s={row['tokens_per_s_median']:.0f}")


if __name__ == "__main__":
    main()
