#!/usr/bin/env python3
"""memory_isolation_llm/run.py — does one tenant's load surge take another's KV cache?

Two tenants on Qwen2.5-3B, each with its own rank-64 LoRA adapter:
    A  the victim: open-loop, constant --rate-a req/s
    B  the surging tenant: open-loop, --rate-b req/s, then from --ramp-at s a
       linear ramp to --rate-b-max over --ramp-s s, held to the end.
Arrivals are Poisson, generated up front from a seeded stream per tenant, so
every mode sees the same arrival times. Both tenants send requests of the
same size distribution: --prompt-len prompt tokens (random ids) and an output
length drawn uniformly from [--out-min, --out-max] (seeded per tenant), as in
lora_fair_share, so completions are staggered rather than in waves.

KV cache, not slots, is the bottleneck by design. Each engine's KV pool is
set exactly (vLLM num_gpu_blocks_override, 16-token blocks): --kv-tokens for
s_be, --iso-kv-tokens (default half) for each be / sp engine. The pool holds
far fewer requests than --batch-cap, so when B surges it fills before the
slot limit does. --gpu-memory-utilization only has to leave room for the
weights, activations and that pool.

Modes:
    s_be  one engine for both tenants (both adapters), one KV pool. Requests
          enter the engine in arrival order (first come, first served); vLLM
          decides which admitted requests run, by available KV blocks.
    be    one engine per tenant (own weights, own KV pool, own adapter);
          engines contend freely for the GPU.
    sp    be, each engine pinned to half the GPU's TPCs via libsmctrl
          (lora_fair_share/run.py:pin_tpcs).
be/sp get --iso-cap slots, --iso-gpu-mem and --iso-kv-tokens each. Every
engine prints its KV capacity at startup.

A parent process (no CUDA) generates the arrivals and records every
completion. Each engine reports every --kv-period s, per tenant: KV tokens
held by its running requests, running and waiting request counts, plus the
pool's used/total blocks and vLLM's cumulative preemptions. "Running" is
vLLM V0's scheduler.running (the sequence groups that hold KV blocks; a
preempted request leaves it) with each sequence's current length; if this
vLLM does not expose it, a request counts as running if it produced a token
in the last --kv-window s, holding prompt + generated tokens.

Output: <exp-dir>/events_<mode>.csv (done / kv / pool rows) and
summary.jsonl. Plot with plot.py.

Usage (from serving/):
    python experiments/memory_isolation_llm/run.py --modes s_be,be,sp \\
        --exp-dir experiments/memory_isolation_llm/results_l40s
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import queue
import random
import sys
import time
from collections import deque
from pathlib import Path

import torch
import torch.multiprocessing as mp

EXP_DIR = Path(__file__).resolve().parent
LFS_DIR = EXP_DIR.parent / "lora_fair_share"
sys.path.insert(0, str(LFS_DIR))
from cost_model import PROFILE_DIR  # noqa: E402  (numpy only; no vLLM)

TENANTS = ("A", "B")       # A: victim, B: surging
MODES = ("s_be", "be", "sp")
ISO_MODES = ("be", "sp")
KV_BLOCK = 16              # vLLM's default block size on CUDA (tokens)


def _lfs_run():
    """lora_fair_share/run.py (build_engine, pin_tpcs). Imported only in engine
    processes: it imports vLLM. Loaded by path: this script is also run.py."""
    spec = importlib.util.spec_from_file_location("lfs_run", LFS_DIR / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Arrivals
# ---------------------------------------------------------------------------

def rate_at(tenant: str, t: float, args) -> float:
    """Offered load (req/s) at t s after t = 0; warmup (t < 0) at base rates."""
    if tenant == "A":
        return args.rate_a
    if t < args.ramp_at:
        return args.rate_b
    frac = min(1.0, (t - args.ramp_at) / args.ramp_s) if args.ramp_s > 0 else 1.0
    return args.rate_b + frac * (args.rate_b_max - args.rate_b)


def arrivals(tenant: str, args) -> list:
    """Poisson arrival times (s, relative to t = 0) by thinning, seeded per tenant."""
    rng = random.Random(f"{args.seed}-{tenant}")
    peak = max(args.rate_a, args.rate_b, args.rate_b_max)
    out, t = [], -args.warmup
    while True:
        t += rng.expovariate(peak)
        if t >= args.duration:
            return out
        if rng.random() < rate_at(tenant, t, args) / peak:
            out.append(t)


# ---------------------------------------------------------------------------
# Engine process
# ---------------------------------------------------------------------------

def _kv_bytes_per_token(engine) -> int:
    try:
        mc, pc = engine.model_config, engine.parallel_config
        elem = torch.tensor([], dtype=mc.dtype).element_size()
        return 2 * mc.get_num_layers(pc) * mc.get_num_kv_heads(pc) * mc.get_head_size() * elem
    except Exception:
        return 36 * 1024   # Qwen2.5-3B fp16: 36 layers x 2 KV heads x 128 x 2 x 2 B


def _pool(engine) -> tuple:
    """(used blocks, total blocks, block size, cumulative preemptions), None if
    this vLLM does not expose them (V0 scheduler / block manager)."""
    try:
        cc = engine.cache_config
        sched = engine.scheduler[0] if isinstance(engine.scheduler, list) else engine.scheduler
        total = cc.num_gpu_blocks
        used = total - sched.block_manager.get_num_free_gpu_blocks()
        return used, total, cc.block_size, getattr(sched, "num_cumulative_preemption", None)
    except Exception:
        return None, None, None, None


def _running_tokens(engine) -> dict | None:
    """{request id: tokens} for every request holding KV blocks (vLLM V0
    scheduler.running), or None if this vLLM does not expose it."""
    try:
        sched = engine.scheduler[0] if isinstance(engine.scheduler, list) else engine.scheduler
        return {sg.request_id: sum(s.get_len() for s in sg.get_seqs()) for sg in sched.running}
    except Exception:
        return None


def engine_main(name: str, cfg: dict, tenants: tuple, req_q, evt_q) -> None:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    lfs = _lfs_run()
    args = copy.copy(cfg["args"])
    args.num_gpu_blocks = cfg["kv_tokens"] // KV_BLOCK   # read by build_engine
    stream = None
    if cfg.get("tpcs"):   # sp: before the engine is built, so graphs carry the mask
        stream, _ = lfs.pin_tpcs(*cfg["tpcs"])
    engine = lfs.build_engine(args, cfg["cap"], cfg["gpu_mem"], len(tenants))
    if stream is not None:
        step = engine.step

        def masked_step():
            with torch.cuda.stream(stream):
                return step()
        engine.step = masked_step
    lora = {t: LoRARequest(f"{t}-{Path(cfg['adapter'][t]).name}", cfg["lora_id"][t],
                           str(cfg["adapter"][t])) for t in tenants}
    prompt_rng = random.Random(name)
    kv_bpt = _kv_bytes_per_token(engine)
    used, total, bs, _ = _pool(engine)
    evt_q.put(("ready", name, time.monotonic(), {
        "kv_bytes_per_token": kv_bpt,
        "kv_capacity_tokens": total * bs if total else None,
        "cap": cfg["cap"], "gpu_mem": cfg["gpu_mem"]}))

    waiting = deque()                     # (rid, tenant, n_out), arrival order
    tenant_of, gen, last_tok = {}, {}, {}
    last_kv = 0.0
    exposed = _running_tokens(engine) is not None
    evt_q.put(("note", name, time.monotonic(),
               "KV per tenant from vLLM scheduler.running" if exposed else
               f"KV per tenant estimated: token in the last {args.kv_window} s"))
    while True:
        while True:                       # 1. intake
            try:
                msg = req_q.get_nowait()
            except queue.Empty:
                break
            if msg is None:
                return
            waiting.append(msg)
        # 2. admission, first come first served, up to cap + queue_k in vLLM
        while waiting and engine.get_num_unfinished_requests() < cfg["cap"] + args.queue_k:
            rid, t, n_out = waiting.popleft()
            prompt = TokensPrompt(prompt_token_ids=[
                prompt_rng.randint(1000, 30000) for _ in range(args.prompt_len)])
            sp = SamplingParams(temperature=0.0, max_tokens=n_out, min_tokens=n_out,
                                ignore_eos=True)
            engine.add_request(rid, prompt, sp, lora_request=lora[t])
            tenant_of[rid], gen[rid] = t, 0
        # 3. one step
        if engine.has_unfinished_requests():
            for o in engine.step():
                rid = o.request_id
                t = tenant_of.get(rid)
                if t is None:
                    continue
                gen[rid] = len(o.outputs[0].token_ids)
                last_tok[rid] = time.monotonic()
                if o.finished:
                    evt_q.put(("done", rid, t, time.monotonic(), gen[rid]))
                    for d in (tenant_of, gen, last_tok):
                        d.pop(rid, None)
        else:
            time.sleep(0.001)
        # 4. KV report
        now = time.monotonic()
        if now - last_kv >= args.kv_period:
            last_kv = now
            kv = {t: 0 for t in tenants}
            run = {t: 0 for t in tenants}
            wait = {t: 0 for t in tenants}
            held = _running_tokens(engine)
            if held is None:   # fallback: produced a token recently
                held = {rid: args.prompt_len + gen[rid] for rid in tenant_of
                        if now - last_tok.get(rid, -1e9) <= args.kv_window}
            for rid, t in tenant_of.items():
                if rid in held:
                    kv[t] += held[rid]
                    run[t] += 1
                else:
                    wait[t] += 1
            for _, t, _ in waiting:
                wait[t] += 1
            used, total, bs, pre = _pool(engine)
            evt_q.put(("kv", name, now, kv, run, wait, used, total, pre))


# ---------------------------------------------------------------------------
# Parent: arrivals, completions
# ---------------------------------------------------------------------------

def run_mode(mode: str, args) -> tuple[list, dict]:
    ctx = mp.get_context("spawn")
    evt_q = ctx.Queue()
    adapter = {"A": args.adapter_a, "B": args.adapter_b}
    base = {"args": args, "adapter": adapter, "lora_id": {"A": 1, "B": 2}}
    if mode in ISO_MODES:
        groups = {f"engine-{t}": (t,) for t in TENANTS}
        cfgs = {f"engine-{t}": dict(base, cap=args.iso_cap or args.batch_cap // 2,
                                    gpu_mem=args.iso_gpu_mem,
                                    kv_tokens=args.iso_kv_tokens or args.kv_tokens // 2,
                                    tpcs=(i / 2, (i + 1) / 2) if mode == "sp" else None)
                for i, t in enumerate(TENANTS)}
    else:
        groups = {"engine": TENANTS}
        cfgs = {"engine": dict(base, cap=args.batch_cap, gpu_mem=args.gpu_memory_utilization,
                               kv_tokens=args.kv_tokens, tpcs=None)}
    engine_of = {t: name for name, ts in groups.items() for t in ts}
    req_qs, procs, info = {}, {}, {}
    events = []
    t_zero = None

    def rel(ts):
        return ts - t_zero

    def handle(evt) -> None:
        kind = evt[0]
        if kind == "ready":
            info[evt[1]] = evt[3]
        elif kind == "note":
            print(f"      {evt[1]}: {evt[3]}", flush=True)
        elif kind == "done":
            _, rid, t, ts, n = evt
            t_arr = sent.pop(rid, None)
            if t_arr is not None:
                events.append({"kind": "done", "tenant": t, "rid": rid, "t": rel(ts),
                               "lat_ms": round((ts - t_arr) * 1e3, 1), "n_tok": n})
        elif kind == "kv":
            _, name, ts, kv, run, wait, used, total, pre = evt
            for t in kv:
                events.append({"kind": "kv", "comp": name, "tenant": t, "t": rel(ts),
                               "tokens": kv[t], "running": run[t], "waiting": wait[t]})
            events.append({"kind": "pool", "comp": name, "t": rel(ts), "used_blocks": used,
                           "total_blocks": total, "preemptions": pre})

    # Engines one at a time: each sizes its KV pool from the memory left free.
    for name in groups:
        req_qs[name] = ctx.Queue()
        procs[name] = ctx.Process(target=engine_main, name=name,
                                  args=(name, cfgs[name], groups[name], req_qs[name], evt_q))
        c = cfgs[name]
        print(f"    starting {name} (cap={c['cap']}, mem={c['gpu_mem']:.3f}"
              + (f", tpcs={c['tpcs'][0]:.2f}:{c['tpcs'][1]:.2f}" if c["tpcs"] else "") + ")",
              flush=True)
        procs[name].start()
        deadline = time.time() + args.startup_timeout
        while name not in info:
            try:
                evt = evt_q.get(timeout=1.0)
                if evt[0] in ("ready", "note"):   # KV samples before t = 0 are dropped
                    handle(evt)
            except queue.Empty:
                pass
            if procs[name].exitcode is not None or time.time() > deadline:
                for p in procs.values():
                    if p.is_alive():
                        p.kill()
                raise RuntimeError(f"{mode}: {name} failed to start")
        cap_tok = info[name]["kv_capacity_tokens"]
        print(f"      KV pool: {cap_tok} tokens "
              f"({cap_tok / (args.prompt_len + args.out_max):.0f} requests at full length)"
              if cap_tok else "      KV pool: not exposed by this vLLM", flush=True)

    sched = {t: arrivals(t, args) for t in TENANTS}
    # Output lengths: one seeded stream per tenant, same in every mode.
    out_rng = {t: random.Random(f"{args.seed}-{t}-out") for t in TENANTS}
    t_zero = time.monotonic() + 0.5 + args.warmup
    t_end = t_zero + args.duration
    nxt = {t: 0 for t in TENANTS}
    sent, n_sent = {}, {t: 0 for t in TENANTS}
    dead = set()
    while True:
        now = time.monotonic()
        if now >= t_end:
            break
        for t in TENANTS:
            arr = sched[t]
            while nxt[t] < len(arr) and t_zero + arr[nxt[t]] <= now:
                rid = f"{t}-{nxt[t]}"
                sent[rid] = t_zero + arr[nxt[t]]
                n_out = out_rng[t].randint(args.out_min, args.out_max)
                req_qs[engine_of[t]].put((rid, t, n_out))
                n_sent[t] += 1
                nxt[t] += 1
        while True:
            try:
                handle(evt_q.get_nowait())
            except queue.Empty:
                break
        for name, p in procs.items():
            if p.exitcode is not None and name not in dead:
                dead.add(name)
                print(f"    {name} exited ({p.exitcode}) at t={rel(now):.1f}s", flush=True)
        time.sleep(0.001)

    for q in req_qs.values():
        q.put(None)
    for p in procs.values():
        p.join(timeout=30)
        if p.is_alive():
            p.kill()

    def done(t, lo, hi):
        return [e for e in events if e["kind"] == "done" and e["tenant"] == t
                and lo <= e["t"] < hi]

    def pct(xs, q):
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1) if xs else None

    phases = {"before": (0.0, args.ramp_at),
              "surge": (args.ramp_at + args.ramp_s, args.duration)}
    stats = {}
    for t in TENANTS:
        for ph, (lo, hi) in phases.items():
            d = done(t, lo, hi)
            stats[f"{t}_{ph}"] = {
                "rps": round(len(d) / (hi - lo), 2) if hi > lo else None,
                "lat_p50_ms": pct([e["lat_ms"] for e in d], 0.5),
                "lat_p99_ms": pct([e["lat_ms"] for e in d], 0.99)}
    summary = {
        "mode": mode, "tenants": list(TENANTS), "victim": "A", "surging": "B",
        "model": args.model, "prompt_len": args.prompt_len,
        "out_min": args.out_min, "out_max": args.out_max,
        "kv_tokens": {n: c["kv_tokens"] for n, c in cfgs.items()},
        "rate_a": args.rate_a, "rate_b": args.rate_b, "rate_b_max": args.rate_b_max,
        "ramp_at": args.ramp_at, "ramp_s": args.ramp_s,
        "warmup": args.warmup, "duration": args.duration, "seed": args.seed,
        "engines": {name: dict(info.get(name, {}), tenants=list(groups[name]))
                    for name in groups},
        "engine_of": engine_of, "died": sorted(dead),
        "sent": n_sent, "unfinished_at_end": len(sent), "stats": stats,
    }
    return events, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--out-min", type=int, default=32)
    ap.add_argument("--out-max", type=int, default=64,
                    help="output length uniform in [out-min, out-max], both tenants")
    ap.add_argument("--kv-tokens", type=int, default=10240,
                    help="s_be KV pool, tokens (multiple of 16)")
    ap.add_argument("--iso-kv-tokens", type=int, default=None,
                    help="be/sp KV pool per engine, tokens (default half of --kv-tokens)")
    ap.add_argument("--batch-cap", type=int, default=256,
                    help="s_be slots (max_num_seqs); larger than the KV pool holds")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.25,
                    help="s_be engine: room for weights, activations and the KV pool")
    ap.add_argument("--iso-cap", type=int, default=None, help="be/sp slots per engine")
    ap.add_argument("--iso-gpu-mem", type=float, default=0.20, help="be/sp per engine")
    ap.add_argument("--max-batched-tokens", type=int, default=8192,
                    help="prompt tokens per step (also vLLM's startup profiling size)")
    ap.add_argument("--queue-k", type=int, default=1)
    ap.add_argument("--rate-a", type=float, default=8.0, help="victim A, req/s")
    ap.add_argument("--rate-b", type=float, default=8.0, help="B before the ramp, req/s")
    ap.add_argument("--rate-b-max", type=float, default=100.0, help="B after the ramp, req/s")
    ap.add_argument("--ramp-at", type=float, default=15.0, help="s after t = 0")
    ap.add_argument("--ramp-s", type=float, default=20.0, help="ramp length, s")
    ap.add_argument("--warmup", type=float, default=10.0, help="base-rate load before t = 0")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--kv-period", type=float, default=0.5, help="s between KV reports")
    ap.add_argument("--kv-window", type=float, default=1.0,
                    help="KV fallback only: running = produced a token in the last s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--startup-timeout", type=float, default=1200.0)
    ap.add_argument("--adapter-a", type=Path, default=PROFILE_DIR / "adapters" / "r64_0")
    ap.add_argument("--adapter-b", type=Path, default=PROFILE_DIR / "adapters" / "r64_1")
    ap.add_argument("--exp-dir", type=Path, default=EXP_DIR / "results_l40s")
    args = ap.parse_args()

    modes = args.modes.split(",")
    assert all(m in MODES for m in modes), modes
    assert args.kv_tokens % KV_BLOCK == 0, f"--kv-tokens must be a multiple of {KV_BLOCK}"
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"A {args.rate_a} req/s; B {args.rate_b} -> {args.rate_b_max} req/s over "
          f"[{args.ramp_at}, {args.ramp_at + args.ramp_s}] s of {args.duration} s; "
          f"requests {args.prompt_len} + [{args.out_min}, {args.out_max}] tokens; "
          f"KV pool {args.kv_tokens} tokens (be/sp "
          f"{args.iso_kv_tokens or args.kv_tokens // 2} each)")

    for mode in modes:
        print(f"--- {mode}", flush=True)
        try:
            events, summary = run_mode(mode, args)
        except RuntimeError as e:
            print(f"    FAILED: {e}; skipping", flush=True)
            continue
        with open(args.exp_dir / f"events_{mode}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=[
                "kind", "comp", "tenant", "rid", "t", "lat_ms", "n_tok", "tokens",
                "running", "waiting", "used_blocks", "total_blocks", "preemptions"])
            wr.writeheader()
            wr.writerows(events)
        with open(args.exp_dir / "summary.jsonl", "a") as f:
            f.write(json.dumps(summary) + "\n")
        s = summary["stats"]
        for t in TENANTS:
            b, a = s[f"{t}_before"], s[f"{t}_surge"]
            print(f"    {t}: {b['rps']} -> {a['rps']} req/s, p50 latency "
                  f"{b['lat_p50_ms']} -> {a['lat_p50_ms']} ms", flush=True)
        print(f"    sent {summary['sent']}, unfinished at end {summary['unfinished_at_end']}",
              flush=True)


if __name__ == "__main__":
    main()
