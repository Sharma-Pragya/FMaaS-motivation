#!/usr/bin/env python3
"""lora_fair_share — VTC vs. BFQ with a plain and a rank-64 LoRA tenant.

Two backlogged tenants share one vLLM engine (LoRA enabled, rank 64):
    P  plain, no adapter
    A  one rank-64 adapter

We drive LLMEngine step by step and own admission: whenever a slot frees,
the scheduler picks which tenant's next request to add. vLLM's own waiting
queue is kept at <= K requests so its FCFS order never overrides ours.
Schedulers differ only in what they charge to a tenant's counter:

    fcfs        none; admits in arrival order. With --demand A:P = d_A:d_P,
                A's requests arrive d_A/d_P times as often as P's, so the
                arrival stream (and FCFS admission) interleaves them in that
                ratio (smooth weighted round robin; 1:1 = alternate)
    vtc         prefill 1 x input tokens, decode 2 x generated tokens
    tuned_vtc   vtc x m_A for A, m_A = A:P per-token cost at --tune-b
    bfq         decode tokens x (base(b)/b + a_i(b)), prefill tokens x
                profiled prefill cost per token
    bfq_stfq    fair_share BFQ's STFQ (start tag S = max(V, F_t), smallest
                S admitted, V <- S) charged with bfq's profiled costs, added
                to F_t at admission and on every decode step. Identical to
                bfq while both tenants are backlogged; V only matters once a
                tenant goes idle.
    standalone_P / standalone_A   one tenant alone (share baselines)

Demand (--demand) only changes fcfs. Both tenants send faster than their fair
share, so both always have queued requests: vtc/bfq pick between two
non-empty queues, and the arrival ratio does not enter their decision.

Two more modes give each tenant its own engine (separate processes, run
before the shared engine is built so the GPU is free):

    be          no sharing, best effort: both engines contend freely
    sp          no sharing, spatial partition: each engine is pinned to a
                weight-proportional slice of the GPU's TPCs via libsmctrl

Each isolated engine gets --iso-cap slots (default B/2) and --iso-gpu-mem
of memory (default half of --gpu-memory-utilization), as in fair_share/llm.

Costs come from the lora_cost_attribution sweep (config B = base(b),
F1 - B = adapter cost). Output lengths are drawn uniformly from
[--out-min, --out-max] so completions are staggered, not in waves.

Shared modes run back to back on one engine (same batch cap); results
append to <exp-dir>/summary.jsonl, per-step logs go to <exp-dir>/steps/.

Usage (from serving/):
    python experiments/lora_fair_share/run.py --batch-cap 64 --weights 1:1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest

from cost_model import CASES, DEFAULT_PROFILE, PROFILE_DIR, Profile

EXP_DIR = Path(__file__).resolve().parent
SERVING_DIR = EXP_DIR.parents[1]
TPC_LIB_DIR = Path(os.environ.get(
    "TPC_LIB_DIR", SERVING_DIR.parents[1] / "TPC_controller" / "tpc_controller"))

ISO_MODES = ["be", "sp"]
ALL_MODES = ["standalone_P", "standalone_A", "fcfs", "vtc", "tuned_vtc", "bfq",
             "bfq_stfq"] + ISO_MODES
TENANTS = ("P", "A")


# ---------------------------------------------------------------------------
# Schedulers: shared selection rule, per-mode charge
# ---------------------------------------------------------------------------

class Scheduler:
    def __init__(self, mode: str, weights: dict, profile: Profile, m_A: float,
                 demand: dict | None = None):
        self.mode = mode
        self.w = weights
        # fcfs: arrival ratio, replayed as a smooth weighted round robin.
        self.demand = demand or {t: 1.0 for t in TENANTS}
        self.credit = {t: 0.0 for t in TENANTS}
        self.profile = profile
        self.m = {"P": 1.0, "A": m_A}
        # Service counter; for bfq_stfq it is the finish tag F_t.
        self.counter = {t: 0.0 for t in TENANTS}
        self.V = 0.0      # bfq_stfq virtual clock
        self.last = "A"   # so the first tie / fcfs pick goes to P

    def pick(self, ready=TENANTS) -> str:
        """Next tenant to admit, among `ready` (tenants with waiting work;
        both, in the closed loop). Ties go to the tenant not picked last."""
        if self.mode == "standalone_P":
            return "P"
        if self.mode == "standalone_A":
            return "A"
        cand = [t for t in TENANTS if t in ready]
        if self.mode == "fcfs":
            # Next arrival in a stream where tenant c arrives at rate d_c:
            # add d_c to every credit, take the largest, charge it sum(d).
            # Ties go to the tenant not picked last, so 1:1 alternates P, A, ...
            for c in TENANTS:
                self.credit[c] += self.demand[c]
            t = max(cand, key=lambda c: (self.credit[c], c != self.last))
            self.credit[t] -= sum(self.demand.values())
        else:
            # bfq_stfq: start tag S_t = max(V, F_t); the others compare
            # counters directly (V never exceeds both, so ties break alike).
            v = self.V if self.mode == "bfq_stfq" else float("-inf")
            start = {c: max(v, self.counter[c]) for c in cand}
            t = min(cand, key=lambda c: (start[c], c == self.last))
            if self.mode == "bfq_stfq":
                self.V = start[t]
        self.last = t
        return t

    def on_admit(self, t: str, prompt_len: int) -> None:
        if self.mode in ("vtc", "tuned_vtc"):
            self.counter[t] += prompt_len * self.m_of(t) / self.w[t]
        elif self.mode in ("bfq", "bfq_stfq"):
            if self.mode == "bfq_stfq":
                self.counter[t] = max(self.V, self.counter[t])
            self.counter[t] += prompt_len * self.profile.prefill_tok_ms[t] / self.w[t]

    def on_decode(self, t: str, n_tok: int, b: int) -> None:
        if n_tok == 0:
            return
        if self.mode in ("vtc", "tuned_vtc"):
            self.counter[t] += 2 * n_tok * self.m_of(t) / self.w[t]
        elif self.mode in ("bfq", "bfq_stfq"):
            self.counter[t] += n_tok * self.profile.decode_tok_ms(t, b) / self.w[t]

    def m_of(self, t: str) -> float:
        return self.m[t] if self.mode == "tuned_vtc" else 1.0


# ---------------------------------------------------------------------------
# One mode on a live engine
# ---------------------------------------------------------------------------

def preemptions(engine) -> int | None:
    try:
        return sum(s.num_cumulative_preemption for s in engine.scheduler)
    except Exception:
        return None


def run_mode(engine, mode: str, args, profile: Profile, weights: dict,
             m_A: float, lora: LoRARequest, rep: int,
             t0: float | None = None) -> tuple[dict, list]:
    """t0: perf_counter() at which warmup starts (default now). perf_counter
    is CLOCK_MONOTONIC on Linux, so the be/sp workers can share one t0."""
    sched = Scheduler(mode, weights, profile, m_A, parse_ratio(args.demand))
    prompt_rng = random.Random(1000 + rep)
    len_rng = random.Random(2000 + rep)   # same length sequence for every mode

    tenant_of, prev_len, last_tok_t = {}, {}, {}
    admitted = {t: 0 for t in TENANTS}
    tokens = {t: 0 for t in TENANTS}
    p_tbt = []
    steps = []
    meas = dict(t_first=None, t_last=None, decode_ms=0.0, prefill_ms=0.0,
                pred_decode_ms=0.0, pred_prefill_ms=0.0, n_steps=0,
                n_prefill=0, b_sum=0, a_frac_sum=0.0)

    t_meas = (time.perf_counter() if t0 is None else t0) + args.warmup
    t_end = t_meas + args.duration
    pre0 = preemptions(engine)
    nid = 0

    while time.perf_counter() < t_end:
        # 1. Admission: fill free slots, at most K waiting inside vLLM.
        while engine.get_num_unfinished_requests() < args.batch_cap + args.queue_k:
            t = sched.pick()
            rid = f"{mode}-{rep}-{nid}-{t}"
            nid += 1
            n_out = len_rng.randint(args.out_min, args.out_max)
            sp = SamplingParams(temperature=0.0, max_tokens=n_out,
                                min_tokens=n_out, ignore_eos=True)
            prompt = TokensPrompt(prompt_token_ids=[
                prompt_rng.randint(1000, 30000) for _ in range(args.prompt_len)])
            engine.add_request(rid, prompt, sp,
                               lora_request=lora if CASES[args.case][t] else None)
            tenant_of[rid], prev_len[rid] = t, 0
            admitted[t] += 1
            sched.on_admit(t, args.prompt_len)

        # 2. One engine step, timed.
        t0 = time.perf_counter()
        outs = engine.step()
        t1 = time.perf_counter()
        if not outs:
            continue
        step_ms = (t1 - t0) * 1e3

        # 3. Accounting.
        new = {t: 0 for t in TENANTS}
        is_prefill = False
        for o in outs:
            rid = o.request_id
            t = tenant_of[rid]
            n = len(o.outputs[0].token_ids)
            if prev_len[rid] == 0:
                is_prefill = True
            new[t] += n - prev_len[rid]
            prev_len[rid] = n
            if t == "P" and t0 >= t_meas and rid in last_tok_t:
                p_tbt.append((t1 - last_tok_t[rid]) * 1e3)
            last_tok_t[rid] = t1
            if o.finished:
                for d in (tenant_of, prev_len, last_tok_t):
                    d.pop(rid, None)
        b = len(outs)
        if not is_prefill:
            for t in TENANTS:
                sched.on_decode(t, new[t], b)

        if t0 >= t_meas:
            if meas["t_first"] is None:
                meas["t_first"] = t0
            meas["t_last"] = t1
            meas["n_steps"] += 1
            for t in TENANTS:
                tokens[t] += new[t]
            if is_prefill:
                meas["n_prefill"] += 1
                meas["prefill_ms"] += step_ms
                meas["pred_prefill_ms"] += sum(
                    new[t] * args.prompt_len * profile.prefill_tok_ms[t] for t in TENANTS)
            else:
                meas["decode_ms"] += step_ms
                meas["pred_decode_ms"] += sum(
                    new[t] * profile.decode_tok_ms(t, b) for t in TENANTS)
                meas["b_sum"] += b
                meas["a_frac_sum"] += new["A"] / b
        steps.append({"t": round(t0, 4), "ms": round(step_ms, 3), "b": b,
                      "prefill": is_prefill, "P": new["P"], "A": new["A"],
                      "cP": round(sched.counter["P"], 2),
                      "cA": round(sched.counter["A"], 2)})

    # Drain so the next mode starts from an empty engine.
    engine.abort_request(list(tenant_of))
    while engine.has_unfinished_requests():
        engine.step()

    wall = meas["t_last"] - meas["t_first"]
    n_dec = meas["n_steps"] - meas["n_prefill"]
    pre1 = preemptions(engine)
    row = {
        "mode": mode, "rep": rep, "batch_cap": args.batch_cap,
        "weights_A_P": args.weights, "case": args.case, "demand_A_P": args.demand,
        "adapter": args.adapter.name,
        "m_A": m_A if mode == "tuned_vtc" else None,
        "wall_s": wall,
        "P_tok_s": tokens["P"] / wall, "A_tok_s": tokens["A"] / wall,
        "agg_tok_s": (tokens["P"] + tokens["A"]) / wall,
        "admitted_P": admitted["P"], "admitted_A": admitted["A"],
        "n_steps": meas["n_steps"], "n_prefill_steps": meas["n_prefill"],
        "prefill_time_frac": meas["prefill_ms"] / (wall * 1e3),
        "mean_b": meas["b_sum"] / n_dec if n_dec else None,
        "mean_A_frac_in_batch": meas["a_frac_sum"] / n_dec if n_dec else None,
        # Profiled cost of the tokens served / measured step time; ~1 if the
        # profile transfers to mixed batches.
        "gpu_check_decode": meas["pred_decode_ms"] / meas["decode_ms"],
        "gpu_check_total": (meas["pred_decode_ms"] + meas["pred_prefill_ms"])
                           / (meas["decode_ms"] + meas["prefill_ms"]),
        "P_tbt_ms_median": statistics.median(p_tbt) if p_tbt else None,
        "P_tbt_ms_p99": float(np.percentile(p_tbt, 99)) if p_tbt else None,
        "preemptions": (pre1 - pre0) if pre0 is not None else None,
    }
    return row, steps


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

def build_engine(args, batch_cap: int, gpu_mem: float, max_loras: int):
    dtype = args.dtype
    if dtype == "auto" and torch.cuda.get_device_capability()[0] < 8:
        dtype = "half"
    max_model_len = args.prompt_len + args.out_max
    # Prefill budget per step: a whole batch of prompts unless capped
    # (fault.py --max-batched-tokens; run.py has no such flag). vLLM's startup
    # profiling run is this many tokens, so with long prompts and large
    # batches the uncapped value leaves little or no memory for KV cache.
    # Without chunked prefill it must hold one full sequence.
    batched = batch_cap * args.prompt_len
    if getattr(args, "max_batched_tokens", None):
        batched = min(batched, args.max_batched_tokens)
    # Optional fixed KV pool (memory_isolation_llm sets it per engine; this
    # script never does): vLLM then allocates exactly this many blocks
    # instead of filling gpu_memory_utilization.
    extra = {}
    if getattr(args, "num_gpu_blocks", None):
        extra["num_gpu_blocks_override"] = args.num_gpu_blocks
    return LLMEngine.from_engine_args(EngineArgs(
        model=args.model, dtype=dtype, seed=0,
        gpu_memory_utilization=gpu_mem,
        max_num_seqs=batch_cap,
        max_model_len=max_model_len,
        max_num_batched_tokens=max(max_model_len, batched),
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        enable_lora=True, max_loras=max_loras, max_lora_rank=64,
        disable_log_stats=True, **extra,
    ))


def pin_tpcs(frac_lo: float, frac_hi: float):
    """Return a CUDA stream masked to TPCs [frac_lo, frac_hi) of the GPU, and
    point vLLM's graph capture at it. Must run before the engine is built.

    Same mechanism as device/server.py:_setup_tpc + device/runtime.py:
    _mask_graph_capture: libsmctrl bakes the mask into a CUDA graph at
    capture time, and vLLM captures on a fresh unmasked stream unless
    graph_capture is replaced.
    """
    import ctypes
    from contextlib import contextmanager

    lib = ctypes.CDLL(str(TPC_LIB_DIR / "libsmctrl" / "libsmctrl.so"))
    lib.libsmctrl_set_stream_mask_ext.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
    lib.libsmctrl_set_stream_mask_ext.restype = None
    lib.libsmctrl_get_tpc_info_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int]
    lib.libsmctrl_get_tpc_info_cuda.restype = ctypes.c_int

    n = ctypes.c_uint32()
    if lib.libsmctrl_get_tpc_info_cuda(ctypes.byref(n), 0) != 0:
        raise RuntimeError("libsmctrl_get_tpc_info_cuda failed")
    tpcs = list(range(round(n.value * frac_lo), round(n.value * frac_hi)))
    if not tpcs:
        raise ValueError(f"empty TPC slice [{frac_lo}, {frac_hi}) of {n.value}")

    stream = torch.cuda.Stream()
    enable = sum(1 << i for i in tpcs)
    disable = ~enable & ((1 << 128) - 1)   # set bit = TPC disabled
    lib.libsmctrl_set_stream_mask_ext(
        ctypes.c_void_p(stream.cuda_stream),
        ctypes.c_uint64(disable & 0xFFFFFFFFFFFFFFFF),
        ctypes.c_uint64(disable >> 64))

    import vllm.worker.model_runner as _model_runner
    from vllm.distributed.parallel_state import (
        GraphCaptureContext, get_pp_group, get_tp_group)
    if not hasattr(_model_runner, "graph_capture"):
        raise RuntimeError("vllm.worker.model_runner.graph_capture moved; "
                           "SP would run unmasked")

    @contextmanager
    def masked_graph_capture():
        context = GraphCaptureContext(stream)
        with get_tp_group().graph_capture(context), \
                get_pp_group().graph_capture(context):
            yield context

    _model_runner.graph_capture = masked_graph_capture
    print(f"[TPC] pinned to TPCs {tpcs[0]}-{tpcs[-1]} of {n.value}", flush=True)
    return stream, tpcs


# ---------------------------------------------------------------------------
# be / sp: one engine per tenant, each in its own process
# ---------------------------------------------------------------------------

def worker_main(args, profile: Profile, weights: dict, m_A: float) -> None:
    """One tenant on its own engine. Signals ready by touching
    <worker-out>.ready, then waits for t0 on stdin so both workers measure
    the same window."""
    t = args.worker
    stream, tpcs = None, None
    if args.worker_tpcs:
        stream, tpcs = pin_tpcs(*(float(x) for x in args.worker_tpcs.split(":")))
    engine = build_engine(args, args.batch_cap, args.gpu_memory_utilization,
                          args.iso_max_loras)
    if stream is not None:
        # LLMEngine.step runs the forward on this thread, so making the
        # masked stream current covers every eager kernel; graph replays
        # carry the mask from capture.
        step = engine.step

        def masked_step():
            with torch.cuda.stream(stream):
                return step()
        engine.step = masked_step
    lora = LoRARequest(args.adapter.name, 1, str(args.adapter))

    out = Path(args.worker_out)
    Path(f"{out}.ready").touch()
    t0 = float(sys.stdin.readline())
    row, steps = run_mode(engine, f"standalone_{t}", args, profile, weights,
                          m_A, lora, args.worker_rep, t0=t0)
    row["tpcs"] = tpcs
    with open(out, "w") as f:
        json.dump({"row": row, "steps": steps}, f)


def run_iso_mode(mode: str, args, weights: dict, rep: int):
    """Run both tenants' workers concurrently; return (row, {t: steps}) or
    None if a worker failed (e.g. two engines do not fit in GPU memory)."""
    cap = args.iso_cap or max(1, args.batch_cap // 2)
    mem = args.iso_gpu_mem or args.gpu_memory_utilization / 2
    f_p = weights["P"] / (weights["P"] + weights["A"])
    common = [
        "--model", args.model, "--dtype", args.dtype, "--weights", args.weights,
        "--case", args.case, "--demand", args.demand, "--warmup", str(args.warmup), "--duration", str(args.duration),
        "--prompt-len", str(args.prompt_len), "--out-min", str(args.out_min),
        "--out-max", str(args.out_max), "--queue-k", str(args.queue_k),
        "--tune-b", str(args.tune_b), "--adapter", str(args.adapter),
        "--profile", str(args.profile), "--exp-dir", str(args.exp_dir),
        "--iso-max-loras", str(args.iso_max_loras),
        "--batch-cap", str(cap), "--gpu-memory-utilization", str(mem),
        "--worker-rep", str(rep),
    ]
    procs, outs = {}, {}

    def fail(msg):
        print(f"    [{mode}] FAILED: {msg}; skipping", flush=True)
        for p in procs.values():
            if p.poll() is None:
                p.kill()
        for o in outs.values():
            for f in (o, Path(f"{o}.ready")):
                f.unlink(missing_ok=True)
        return None

    # Start sequentially: vLLM sizes its KV cache from free memory at
    # profiling time, so the second engine must see the first fully loaded.
    for t in TENANTS:
        outs[t] = args.exp_dir / "steps" / f".worker_{mode}_{t}_r{rep}.json"
        cmd = [sys.executable, str(Path(__file__).resolve()), *common,
               "--worker", t, "--worker-out", str(outs[t])]
        if mode == "sp":
            cmd += ["--worker-tpcs", f"0:{f_p}" if t == "P" else f"{f_p}:1"]
        print(f"    starting {t} worker: cap={cap} mem={mem:.2f}"
              + (f" tpcs={cmd[-1]}" if mode == "sp" else ""), flush=True)
        procs[t] = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
        deadline = time.time() + args.iso_startup_timeout
        while not Path(f"{outs[t]}.ready").exists():
            if procs[t].poll() is not None:
                return fail(f"{t} worker exited during startup "
                            f"(code {procs[t].returncode})")
            if time.time() > deadline:
                return fail(f"{t} worker not ready after {args.iso_startup_timeout}s")
            time.sleep(1)

    t0 = time.perf_counter() + 1.0
    for p in procs.values():
        p.stdin.write(f"{t0}\n")
        p.stdin.close()
    for t, p in procs.items():
        if p.wait() != 0:
            return fail(f"{t} worker exited with code {p.returncode}")

    res = {}
    for t in TENANTS:
        res[t] = json.loads(outs[t].read_text())
        outs[t].unlink()
        Path(f"{outs[t]}.ready").unlink()
    rp, ra = res["P"]["row"], res["A"]["row"]
    pre = (rp["preemptions"] + ra["preemptions"]
           if None not in (rp["preemptions"], ra["preemptions"]) else None)
    row = {
        "mode": mode, "rep": rep, "batch_cap": args.batch_cap,
        "weights_A_P": args.weights, "case": args.case, "demand_A_P": args.demand,
        "adapter": args.adapter.name,
        "m_A": None,
        "wall_s": (rp["wall_s"] + ra["wall_s"]) / 2,
        "P_tok_s": rp["P_tok_s"], "A_tok_s": ra["A_tok_s"],
        "agg_tok_s": rp["P_tok_s"] + ra["A_tok_s"],
        "admitted_P": rp["admitted_P"], "admitted_A": ra["admitted_A"],
        "n_steps": rp["n_steps"] + ra["n_steps"],
        "n_prefill_steps": rp["n_prefill_steps"] + ra["n_prefill_steps"],
        "prefill_time_frac": (rp["prefill_time_frac"] + ra["prefill_time_frac"]) / 2,
        "mean_b": None, "mean_A_frac_in_batch": None,
        # The profile assumes one engine owns the GPU, so the check has no
        # single meaning here; per-engine values are kept in iso_P / iso_A.
        "gpu_check_decode": None, "gpu_check_total": None,
        "P_tbt_ms_median": rp["P_tbt_ms_median"], "P_tbt_ms_p99": rp["P_tbt_ms_p99"],
        "preemptions": pre,
        "iso_cap": cap, "iso_gpu_mem": mem, "iso_P": rp, "iso_A": ra,
    }
    return row, {t: res[t]["steps"] for t in TENANTS}


def parse_ratio(s: str) -> dict:
    """'A:P' string -> {"A": a, "P": p}."""
    a, p = (float(x) for x in s.split(":"))
    return {"A": a, "P": p}


def save_row(args, row: dict) -> None:
    with open(args.exp_dir / "summary.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def save_steps(args, steps: list, tag: str) -> None:
    with open(args.exp_dir / "steps" / f"{tag}.jsonl", "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--batch-cap", type=int, default=64)
    ap.add_argument("--weights", default="1:1", help="A:P")
    ap.add_argument("--case", choices=list(CASES), default="mixed",
                    help="which tenants use the rank-64 adapter (cost_model.CASES)")
    ap.add_argument("--demand", default="1:1",
                    help="A:P arrival ratio; changes fcfs (S-BE) admission only")
    ap.add_argument("--modes", default=",".join(ALL_MODES))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--warmup", type=float, default=60.0)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--out-min", type=int, default=128)
    ap.add_argument("--out-max", type=int, default=384)
    ap.add_argument("--queue-k", type=int, default=1)
    ap.add_argument("--tune-b", type=int, default=8)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--adapter", type=Path,
                    default=PROFILE_DIR / "adapters" / "r64_0")
    ap.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    ap.add_argument("--exp-dir", type=Path, default=EXP_DIR / "results_t4")
    # be / sp: per-tenant engines
    ap.add_argument("--iso-cap", type=int, default=None,
                    help="slots per isolated engine (default batch-cap/2)")
    ap.add_argument("--iso-gpu-mem", type=float, default=None,
                    help="gpu_memory_utilization per isolated engine "
                         "(default half of --gpu-memory-utilization)")
    ap.add_argument("--iso-max-loras", type=int, default=1,
                    help="LoRA slots per isolated engine (shared engine uses 16, "
                         "as in profiling; 16 slots of rank 64 do not fit twice)")
    ap.add_argument("--iso-startup-timeout", type=float, default=1200.0)
    ap.add_argument("--worker", choices=TENANTS, help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", help=argparse.SUPPRESS)
    ap.add_argument("--worker-rep", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--worker-tpcs", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    modes = args.modes.split(",")
    assert all(m in ALL_MODES for m in modes), modes
    weights = parse_ratio(args.weights)
    parse_ratio(args.demand)   # fail early on a malformed ratio

    profile = Profile(args.profile, args.prompt_len, CASES[args.case])
    m_A = profile.m_A(args.tune_b)
    if args.worker:
        worker_main(args, profile, weights, m_A)
        return
    print(f"case={args.case} adapter on {[t for t in TENANTS if CASES[args.case][t]]}; "
          f"demand A:P={args.demand}; "
          f"profile: base(b)={dict(zip(profile.bs, map(round, profile.base_ms)))} "
          f"adapter(b)={[round(a, 3) for a in profile.adapter_ms]} "
          f"m_A(b={args.tune_b})={m_A:.3f}")

    (args.exp_dir / "steps").mkdir(parents=True, exist_ok=True)
    # Step-log tag; the original mixed case keeps its old file names.
    wtag = (args.weights.replace(":", "-")
            + ("" if args.case == "mixed" else f"_{args.case}")
            + ("" if args.demand == "1:1" else f"_d{args.demand.replace(':', '-')}"))

    # Isolated modes first, while this process holds no GPU memory.
    for rep in range(args.reps):
        for mode in (m for m in modes if m in ISO_MODES):
            print(f"--- {mode} rep{rep} (B={args.batch_cap}, weights A:P={args.weights}, demand A:P={args.demand})")
            res = run_iso_mode(mode, args, weights, rep)
            if res is None:
                continue
            row, steps = res
            save_row(args, row)
            for t in TENANTS:
                save_steps(args, steps[t], f"{mode}_B{args.batch_cap}_w{wtag}_r{rep}_{t}")
            print(f"    P={row['P_tok_s']:.0f} tok/s  A={row['A_tok_s']:.0f} tok/s  "
                  f"agg={row['agg_tok_s']:.0f}  preempt={row['preemptions']}")

    shared = [m for m in modes if m not in ISO_MODES]
    if not shared:
        return
    engine = build_engine(args, args.batch_cap, args.gpu_memory_utilization,
                          max_loras=16)   # as in profiling
    lora = LoRARequest(args.adapter.name, 1, str(args.adapter))

    for rep in range(args.reps):
        for mode in shared:
            print(f"--- {mode} rep{rep} (B={args.batch_cap}, weights A:P={args.weights}, demand A:P={args.demand})")
            row, steps = run_mode(engine, mode, args, profile, weights, m_A, lora, rep)
            save_row(args, row)
            save_steps(args, steps, f"{mode}_B{args.batch_cap}_w{wtag}_r{rep}")
            print(f"    P={row['P_tok_s']:.0f} tok/s  A={row['A_tok_s']:.0f} tok/s  "
                  f"agg={row['agg_tok_s']:.0f}  gpu_check={row['gpu_check_decode']:.2f}  "
                  f"prefill={row['prefill_time_frac']:.0%}  b={row['mean_b']:.1f}  "
                  f"preempt={row['preemptions']}")


if __name__ == "__main__":
    main()
