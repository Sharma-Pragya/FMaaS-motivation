#!/usr/bin/env python3
"""lora_fair_share/fault.py — fault isolation: one task's fault vs. the others' service.

--n-tasks tasks (default 4) on one base model (run.py's engine settings),
each with its own rank-64 LoRA adapter: P, the task that faults, and victims
A1..A{N-1}. Load is closed loop: --clients per task, each sending its next
request as soon as the last one completes (after --retry-backoff s if it
failed), so the offered load adapts to what each deployment can serve and
no queue grows beyond the clients.

Every finished request is post-processed by its task's hook: a small GPU op
on the output tokens, standing in for task-specific code (a head, a
classifier, constrained decoding). At --fault-at s P's hook faults once,
in one of two ways (--fault-kind):

    gpu      a GPU context fault: the hook indexes out of bounds, on CUDA a
             device-side assert, which poisons the whole CUDA context it
             runs in.
    segv     a host/process fault: the hook dereferences a null pointer on
             the CPU (a bug in a native extension, say) before any GPU
             work, so the process dies of SIGSEGV with its CUDA context
             intact. No Python handler can catch it.

The hook is identical everywhere; only where it runs differs:

    s_be     one engine process for all tasks, hooks inline in it. vLLM
             multi-LoRA, Punica, S-LoRA; VTC or any other scheduler inside
             one engine shares the same fault domain. Admission in arrival
             order.
    be       one engine process per task (its own copy of the base model),
             running its task's hook; engines contend freely for the GPU.
    sp       as be, each engine pinned to 1/N of the GPU's TPCs via libsmctrl
             (run.py:pin_tpcs; TPC_LIB_DIR).
    fmvisor  isolation_mechanisms' process_context_ipc applied to the LLM:
             one engine process (BFQ-STFQ admission) that owns a per-task
             slot ring in VRAM, plus one process per task that receives its
             ring through CUDA IPC at spawn and runs the hook in its own
             context. The engine writes a finished request's tokens to the
             task's ring and passes only the slot index.

be/sp split --batch-cap and --gpu-memory-utilization N ways (override with
--iso-cap / --iso-gpu-mem), so every task gets the same slot share in every
mode. BE/SP need N copies of the base model: a 3B model in fp16 is ~6.2 GB,
so N = 4 fits an L40S (48 GB) at 0.85 and N = 5 is marginal.

A parent process generates load and records every request's outcome. By
default nothing is recovered: whatever dies (an engine, or an fmvisor task
process) stays down for the rest of the run, so the run shows only how far
the fault reaches. With --recover the parent restarts dead engines and the
fmvisor engine restarts its task processes; restarted components run clean.
A request fails if the component serving it dies with it in flight, or if
it is sent while that component is down. Nothing is replayed.

Output: <exp-dir>/fault/events_<mode>.csv and summary.jsonl for
--fault-kind gpu, <exp-dir>/fault_<kind>/ for the others. Plot with
fault_plot.py (same --fault-kind).

Usage (from serving/):
    python experiments/lora_fair_share/fault.py --modes s_be,be,sp,fmvisor \\
        --exp-dir experiments/lora_fair_share/results_l40s [--fault-kind segv]
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import queue
import random
import sys
import time
from collections import deque
from pathlib import Path

import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cost_model import DEFAULT_PROFILE, PROFILE_DIR, Profile  # noqa: E402

# run.py (and through it vLLM) is imported only inside engine_main: spawned
# task processes re-import this module, and importing vLLM there would add
# seconds to every task-process restart.
FAULT_TENANT = "P"
EXP_DIR = Path(__file__).resolve().parent
MODES = ("s_be", "be", "sp", "fmvisor")
ISO_MODES = ("be", "sp")
FAULT_KINDS = ("gpu", "segv")
HOOK_VOCAB = 152_064   # Qwen2.5 vocabulary: valid token ids index the table
RING = 256             # fmvisor slots per task
HEARTBEAT_S = 0.5


def task_names(n: int) -> list:
    return [FAULT_TENANT] + [f"A{i}" for i in range(1, n)]


def fault_dir(exp_dir: Path, kind: str) -> Path:
    """gpu keeps the original fault/ directory; fault_plot.py mirrors this."""
    return exp_dir / ("fault" if kind == "gpu" else f"fault_{kind}")


def flush_before_crash(evt_q, kind: str) -> None:
    """A SIGSEGV skips interpreter shutdown, so events still buffered in the
    queue's feeder thread (the "fault" event, earlier "done"s) would be lost.
    The queue is not used again: the process dies next."""
    if kind == "segv":
        evt_q.close()
        evt_q.join_thread()


# ---------------------------------------------------------------------------
# The task hook: same code in every deployment
# ---------------------------------------------------------------------------

class TenantHook:
    """Post-processing on a finished request's tokens. faulty=True:
    kind "gpu" indexes past the table: on CUDA a device-side assert, sticky
    for the whole context; on CPU an IndexError. kind "segv" reads address 0
    on the CPU before any GPU work: SIGSEGV. Either way the process running
    it dies."""

    def __init__(self, device: str, kind: str = "gpu"):
        g = torch.Generator().manual_seed(0)
        self.device, self.kind = device, kind
        self.table = torch.randn(HOOK_VOCAB, 64, generator=g).to(device)
        self.w = torch.randn(64, 8, generator=g).to(device)

    def __call__(self, tokens: torch.Tensor, faulty: bool = False) -> int:
        if faulty and self.kind == "segv":
            ctypes.string_at(0)   # null dereference: the process dies here
        idx = tokens.to(self.device, torch.long)
        if faulty:
            idx = idx + HOOK_VOCAB
        y = torch.nn.functional.embedding(idx, self.table).mean(0) @ self.w
        return int(y.argmax())   # syncs, so an assert surfaces here


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# fmvisor admission: BFQ-STFQ over N tasks
# ---------------------------------------------------------------------------

class Stfq:
    """run.py's bfq_stfq generalized to N tasks: start tag S_t = max(V, F_t),
    smallest S admitted, V <- S; profiled BFQ costs added to F_t at admission
    (prefill) and on every decode step. Every task carries a rank-64 adapter,
    so all are charged as run.py's adapter tenant ("A")."""

    def __init__(self, tasks, profile: Profile, prompt_len: int):
        self.tasks = list(tasks)
        self.profile, self.prompt_len = profile, prompt_len
        self.F = {t: 0.0 for t in tasks}
        self.V = 0.0
        self.rr = 0   # ties rotate

    def pick(self, ready: list) -> str:
        n = len(self.tasks)
        order = {t: (self.tasks.index(t) - self.rr) % n for t in ready}
        start = {t: max(self.V, self.F[t]) for t in ready}
        t = min(ready, key=lambda c: (start[c], order[c]))
        self.V = start[t]
        self.rr = (self.tasks.index(t) + 1) % n
        return t

    def on_admit(self, t: str) -> None:
        self.F[t] = max(self.V, self.F[t]) + self.prompt_len * self.profile.prefill_tok_ms["A"]

    def on_decode(self, t: str, n_tok: int, b: int) -> None:
        if n_tok:
            self.F[t] += n_tok * self.profile.decode_tok_ms("A", b)


# ---------------------------------------------------------------------------
# fmvisor task process
# ---------------------------------------------------------------------------

def tenant_main(t: str, ring: torch.Tensor, work_q, ready_q, evt_q, t_fault,
                kind: str) -> None:
    """Own process => own CUDA context. `ring` lives in the engine process's
    VRAM and arrived through CUDA IPC (CPU shared memory without a GPU)."""
    device = _device()
    hook = TenantHook(device, kind)
    hook(ring[0, :1])            # warm the hook's kernels in this context
    _sync(device)
    ready_q.put(t)
    while True:
        item = work_q.get()
        if item is None:
            return
        rid, k, n = item
        faulty = t_fault is not None and 0 < t_fault.value <= time.monotonic()
        if faulty:
            evt_q.put(("fault", f"task-{t}", t, time.monotonic()))
            flush_before_crash(evt_q, kind)
        hook(ring[k, :n], faulty)
        evt_q.put(("done", rid, t, time.monotonic(), n))


# ---------------------------------------------------------------------------
# Engine process (s_be/fmvisor: all tasks; be/sp: one)
# ---------------------------------------------------------------------------

def engine_main(name: str, cfg: dict, tenants: tuple, req_q, evt_q) -> None:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    from run import build_engine, pin_tpcs

    args, mode = cfg["args"], cfg["mode"]
    device = _device()
    stream = None
    if cfg.get("tpcs"):   # sp: before the engine is built, so graphs carry the mask
        stream, tpcs = pin_tpcs(*cfg["tpcs"])
    engine = build_engine(args, cfg["cap"], cfg["gpu_mem"], cfg["max_loras"])
    if stream is not None:
        step = engine.step

        def masked_step():
            with torch.cuda.stream(stream):
                return step()
        engine.step = masked_step
    # One adapter per task (same weights, distinct ids: the shared engine
    # holds N adapters, as it would for N fine-tunes).
    lora = {t: LoRARequest(f"{args.adapter.name}-{t}", cfg["lora_id"][t], str(args.adapter))
            for t in tenants}
    sched = Stfq(tenants, Profile(args.profile, args.prompt_len), args.prompt_len)
    prompt_rng = random.Random(name)

    fmvisor = mode == "fmvisor"
    if fmvisor:
        ctx = mp.get_context("spawn")
        rings = {t: torch.zeros((RING, args.out_max), dtype=torch.long, device=device)
                 for t in tenants}
        if device == "cpu":
            for r in rings.values():
                r.share_memory_()
        task_ready = ctx.Queue()
        t_fault_val = ctx.Value("d", 0.0)
        work_q, procs, up, slot = {}, {}, {t: False for t in tenants}, {t: 0 for t in tenants}

        def spawn(t: str, fault: bool) -> None:
            work_q[t] = ctx.Queue()
            procs[t] = ctx.Process(
                target=tenant_main, name=f"task-{t}", daemon=True,
                args=(t, rings[t], work_q[t], task_ready, evt_q,
                      t_fault_val if fault else None, args.fault_kind))
            procs[t].start()

        for t in tenants:
            spawn(t, fault=t == FAULT_TENANT)
        while not all(up.values()):
            t = task_ready.get()
            up[t] = True
            evt_q.put(("up", f"task-{t}", t, time.monotonic()))
    else:
        hook = TenantHook(device, args.fault_kind)
        hook(torch.zeros(1, dtype=torch.long))
        _sync(device)

    waiting = {t: deque() for t in tenants}   # (seq, rid, n_out)
    tenant_of, prev_len = {}, {}
    t_fault, fired = None, False
    last_hb = 0.0
    evt_q.put(("ready", name, None, time.monotonic()))

    def deliver(t: str, rid: str, ids: list) -> None:
        nonlocal fired
        if not fmvisor:
            faulty = (not fired and t == FAULT_TENANT and t_fault is not None
                      and time.monotonic() >= t_fault)
            if faulty:
                fired = True
                evt_q.put(("fault", name, t, time.monotonic()))
                flush_before_crash(evt_q, args.fault_kind)
            hook(torch.tensor(ids), faulty)   # an assert or SIGSEGV kills this process
            evt_q.put(("done", rid, t, time.monotonic(), len(ids)))
            return
        if not up[t]:
            return
        k = slot[t]
        slot[t] = (k + 1) % RING
        rings[t][k, :len(ids)].copy_(torch.tensor(ids, dtype=torch.long))
        # The tokens must be in VRAM before the task process reads them:
        # host-side queues do not order kernels across contexts.
        _sync(device)
        work_q[t].put((rid, k, len(ids)))

    while True:
        # 1. Intake.
        while True:
            try:
                msg = req_q.get_nowait()
            except queue.Empty:
                break
            if msg[0] == "stop":
                if fmvisor:
                    for t in tenants:
                        work_q[t].put(None)
                return
            if msg[0] == "go":
                t_fault = msg[1]
                if fmvisor and t_fault is not None:
                    t_fault_val.value = t_fault
            else:
                _, rid, t, n_out, seq = msg
                if not fmvisor or up[t]:
                    waiting[t].append((seq, rid, n_out))

        # 2. fmvisor: supervise task processes.
        if fmvisor:
            while True:
                try:
                    t = task_ready.get_nowait()
                except queue.Empty:
                    break
                up[t] = True
                evt_q.put(("up", f"task-{t}", t, time.monotonic()))
            for t in tenants:
                if up[t] and procs[t].exitcode is not None:
                    up[t] = False
                    evt_q.put(("down", f"task-{t}", t, time.monotonic()))
                    dead = [r for r, tt in tenant_of.items() if tt == t]
                    engine.abort_request(dead)
                    for r in dead:
                        tenant_of.pop(r)
                        prev_len.pop(r)
                    waiting[t].clear()
                    if args.recover:
                        spawn(t, fault=False)

        # 3. Admission: fill free slots, at most queue_k waiting inside vLLM.
        while engine.get_num_unfinished_requests() < cfg["cap"] + args.queue_k:
            ready = [t for t in tenants if waiting[t] and (not fmvisor or up[t])]
            if not ready:
                break
            if fmvisor:
                t = sched.pick(ready)
            else:   # arrival order
                t = min(ready, key=lambda c: waiting[c][0][0])
            _, rid, n_out = waiting[t].popleft()
            sp = SamplingParams(temperature=0.0, max_tokens=n_out,
                                min_tokens=n_out, ignore_eos=True)
            prompt = TokensPrompt(prompt_token_ids=[
                prompt_rng.randint(1000, 30000) for _ in range(args.prompt_len)])
            engine.add_request(rid, prompt, sp, lora_request=lora[t])
            tenant_of[rid], prev_len[rid] = t, 0
            if fmvisor:
                sched.on_admit(t)

        now = time.monotonic()
        if now - last_hb >= HEARTBEAT_S:
            evt_q.put(("hb", name, None, now))
            last_hb = now

        # 4. One engine step.
        if not engine.has_unfinished_requests():
            time.sleep(0.001)
            continue
        outs = engine.step()
        new = {t: 0 for t in tenants}
        is_prefill = False
        for o in outs:
            t = tenant_of.get(o.request_id)
            if t is None:
                continue
            ids = list(o.outputs[0].token_ids)
            if prev_len[o.request_id] == 0:
                is_prefill = True
            new[t] += len(ids) - prev_len[o.request_id]
            prev_len[o.request_id] = len(ids)
            if o.finished:
                del tenant_of[o.request_id], prev_len[o.request_id]
                deliver(t, o.request_id, ids)
        if fmvisor and not is_prefill:
            for t in tenants:
                sched.on_decode(t, new[t], len(outs))


# ---------------------------------------------------------------------------
# Parent: closed-loop clients, outcomes, engine restarts
# ---------------------------------------------------------------------------

def run_fault_mode(mode: str, args) -> tuple[list, dict]:
    ctx = mp.get_context("spawn")
    evt_q = ctx.Queue()
    tasks = task_names(args.n_tasks)
    n = len(tasks)
    lora_id = {t: i + 1 for i, t in enumerate(tasks)}
    base = {"args": args, "mode": mode, "lora_id": lora_id}
    if mode in ISO_MODES:
        groups = {f"engine-{t}": (t,) for t in tasks}
        cap = args.iso_cap or max(1, args.batch_cap // n)
        gpu_mem = args.iso_gpu_mem or args.gpu_memory_utilization / n
        cfgs = {f"engine-{t}": dict(base, cap=cap, gpu_mem=gpu_mem, max_loras=1,
                                    tpcs=(i / n, (i + 1) / n) if mode == "sp" else None)
                for i, t in enumerate(tasks)}
    else:
        groups = {"engine": tuple(tasks)}
        cap, gpu_mem = args.batch_cap, args.gpu_memory_utilization
        cfgs = {"engine": dict(base, cap=cap, gpu_mem=gpu_mem, max_loras=n, tpcs=None)}
    engine_of = {t: name for name, ts in groups.items() for t in ts}
    # Components that must all be up for a task to be served.
    chain = {t: [engine_of[t]] + ([f"task-{t}"] if mode == "fmvisor" else [])
             for t in tasks}

    procs, req_qs, status, last_hb = {}, {}, {}, {}
    events, inflight = [], {}   # rid -> (task, client index)
    restarts = []
    stalled = set()   # be/sp engines currently past --hang-timeout (not killed)
    t_zero = None   # plot t = 0 (end of warmup)

    # Closed-loop clients. Output lengths come from one seeded stream per
    # task, so every mode sees the same sequence of request sizes.
    len_rng = {t: random.Random(f"{args.seed}-{t}") for t in tasks}
    clients = [{"t": t, "next": None, "busy": False}
               for t in tasks for _ in range(args.clients)]
    seq = 0

    def rel(ts):
        return None if t_zero is None else ts - t_zero

    def free(ci: int, ts: float, backoff: float) -> None:
        clients[ci]["busy"] = False
        clients[ci]["next"] = ts + backoff

    def start(name: str) -> None:
        req_qs[name] = ctx.Queue()
        procs[name] = ctx.Process(target=engine_main, name=name,
                                  args=(name, cfgs[name], groups[name], req_qs[name], evt_q))
        procs[name].start()
        status[name] = False

    def mark(comp: str, up: bool, ts: float) -> None:
        if status.get(comp) == up:
            return
        status[comp] = up
        events.append({"kind": "up" if up else "down", "comp": comp, "t": rel(ts)})
        if up:
            for r in restarts:
                if r["comp"] == comp and r["t_up"] is None:
                    r["t_up"] = rel(ts)
            return
        restarts.append({"comp": comp, "t_down": rel(ts), "t_up": None})
        for rid, (t, ci) in list(inflight.items()):
            if comp in chain[t]:
                del inflight[rid]
                events.append({"kind": "fail", "comp": comp, "tenant": t,
                               "rid": rid, "t": rel(ts)})
                free(ci, ts, args.retry_backoff)
        if comp.startswith("engine"):   # its task processes die with it
            for t in groups[comp]:
                if f"task-{t}" in status:
                    status[f"task-{t}"] = False

    def handle(evt) -> None:
        kind, ts = evt[0], evt[3]
        if kind == "done":
            rid, t, n_tok = evt[1], evt[2], evt[4]
            hit = inflight.pop(rid, None)
            if hit is not None:
                events.append({"kind": "ok", "tenant": t, "rid": rid,
                               "t": rel(ts), "n_tok": n_tok})
                free(hit[1], ts, args.think_time)
        elif kind == "ready":
            mark(evt[1], True, ts)
            last_hb[evt[1]] = ts
            if t_zero is not None:   # a restart: run clean
                req_qs[evt[1]].put(("go", None))
        elif kind in ("up", "down"):
            mark(evt[1], kind == "up", ts)
        elif kind == "fault":
            events.append({"kind": "fault", "comp": evt[1], "tenant": evt[2], "t": rel(ts)})
        elif kind == "hb":
            last_hb[evt[1]] = ts

    def drain(block_s: float = 0.0) -> None:
        try:
            handle(evt_q.get(timeout=block_s) if block_s else evt_q.get_nowait())
        except queue.Empty:
            return
        while True:
            try:
                handle(evt_q.get_nowait())
            except queue.Empty:
                return

    def kill_all() -> None:
        for p in procs.values():
            if p.is_alive():
                p.kill()

    # Start engines one at a time (each sizes its KV cache from free memory).
    for name in groups:
        c = cfgs[name]
        print(f"    starting {name} (cap={c['cap']}, mem={c['gpu_mem']:.3f}"
              + (f", tpcs={c['tpcs'][0]:.3f}:{c['tpcs'][1]:.3f}" if c["tpcs"] else "")
              + ")", flush=True)
        start(name)
        deadline = time.time() + args.startup_timeout
        while not status[name]:
            drain(1.0)
            if procs[name].exitcode is not None or time.time() > deadline:
                kill_all()
                raise RuntimeError(f"{mode}: {name} failed to start")

    t_start = time.monotonic() + 0.5
    t_zero = t_start + args.warmup
    t_end = t_zero + args.duration
    t_fault = t_zero + args.fault_at
    # Stagger first sends over 1 s so the run does not open with one big prefill.
    stagger = random.Random(args.seed)
    for c in clients:
        c["next"] = t_start + stagger.random()
    for name in groups:
        req_qs[name].put(("go", t_fault))

    while True:
        now = time.monotonic()
        if now >= t_end:
            break
        for ci, c in enumerate(clients):
            if c["busy"] or c["next"] > now:
                continue
            t = c["t"]
            rid = f"{t}-{seq}"
            seq += 1
            if all(status.get(x) for x in chain[t]):
                c["busy"] = True
                inflight[rid] = (t, ci)
                n_out = len_rng[t].randint(args.out_min, args.out_max)
                req_qs[engine_of[t]].put(("req", rid, t, n_out, seq))
            else:
                down = next(x for x in chain[t] if not status.get(x))
                events.append({"kind": "fail", "comp": down, "tenant": t,
                               "rid": rid, "t": rel(now)})
                c["next"] = now + args.retry_backoff
        drain()
        for name, p in procs.items():
            if p.exitcode is not None and status[name]:
                mark(name, False, time.monotonic())
                print(f"    {name} died (exit {p.exitcode}) at t={rel(time.monotonic()):.1f}s; "
                      + ("restarting" if args.recover else "left down"), flush=True)
                if args.recover:
                    start(name)
            elif status[name] and now - last_hb.get(name, now) > args.hang_timeout:
                if mode in ISO_MODES:
                    # Isolation only: a be/sp victim engine stalled by P's
                    # fault is left alone (it resumes on its own); killing it
                    # would turn a stall into failures. Reported once per stall.
                    if name not in stalled:
                        stalled.add(name)
                        print(f"    {name} silent for {args.hang_timeout}s at "
                              f"t={rel(now):.1f}s; not killing ({mode})", flush=True)
                else:
                    print(f"    {name} silent for {args.hang_timeout}s; killing", flush=True)
                    p.kill()
            else:
                stalled.discard(name)
        time.sleep(0.002)

    for name in groups:
        req_qs[name].put(("stop",))
    for p in procs.values():
        p.join(timeout=10)
        if p.is_alive():
            p.kill()

    def count(t, kind):
        return sum(e.get("tenant") == t and e["kind"] == kind
                   and e["t"] is not None and 0 <= e["t"] < args.duration for e in events)

    victims = [t for t in tasks if t != FAULT_TENANT]
    summary = {
        "mode": mode, "n_tasks": n, "tasks": tasks, "fault_tenant": FAULT_TENANT,
        "victims": victims, "chain": chain,
        "fault_at": args.fault_at, "duration": args.duration, "warmup": args.warmup,
        "clients": args.clients, "think_time": args.think_time,
        "retry_backoff": args.retry_backoff, "batch_cap": cap, "gpu_mem": gpu_mem,
        "model": args.model, "fault_kind": args.fault_kind, "recover": args.recover,
        "restarts": restarts,   # every down; t_up stays None without --recover
        "ok": {t: count(t, "ok") for t in tasks},
        "fail": {t: count(t, "fail") for t in tasks},
        "victims_ok": sum(count(t, "ok") for t in victims),
        "victims_fail": sum(count(t, "fail") for t in victims),
        "still_in_flight_at_end": len(inflight),
    }
    return events, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--n-tasks", type=int, default=4, help="P + N-1 victims")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--batch-cap", type=int, default=64,
                    help="shared engine's slots; be/sp get batch-cap / N each")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                    help="all engines together; be/sp get 1/N each")
    ap.add_argument("--iso-cap", type=int, default=None, help="be/sp: slots per engine")
    ap.add_argument("--iso-gpu-mem", type=float, default=None, help="be/sp: memory per engine")
    ap.add_argument("--clients", type=int, default=16, help="closed-loop clients per task")
    ap.add_argument("--think-time", type=float, default=0.0,
                    help="s between a completion and the client's next request")
    ap.add_argument("--retry-backoff", type=float, default=0.5,
                    help="s a client waits after a failure before sending again")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--out-min", type=int, default=32)
    ap.add_argument("--out-max", type=int, default=64)
    ap.add_argument("--queue-k", type=int, default=1)
    ap.add_argument("--max-batched-tokens", type=int, default=None,
                    help="cap on prompt tokens per engine step (default: a whole "
                         "batch of prompts); needed for long prompts x large batches")
    ap.add_argument("--fault-at", type=float, default=20.0, help="s after t = 0")
    ap.add_argument("--recover", action="store_true",
                    help="restart dead engines / task processes (default: leave them down)")
    ap.add_argument("--fault-kind", choices=FAULT_KINDS, default="gpu",
                    help="gpu: out-of-bounds index, sticky CUDA error; "
                         "segv: null dereference on the CPU, process dies")
    ap.add_argument("--warmup", type=float, default=10.0, help="load before t = 0")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--startup-timeout", type=float, default=1200.0)
    ap.add_argument("--hang-timeout", type=float, default=10.0,
                    help="s without a heartbeat before an engine is killed "
                         "(s_be, fmvisor); be/sp engines are only reported")
    ap.add_argument("--adapter", type=Path, default=PROFILE_DIR / "adapters" / "r64_0")
    ap.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    ap.add_argument("--exp-dir", type=Path, default=EXP_DIR / "results_l40s")
    args = ap.parse_args()

    modes = args.modes.split(",")
    assert all(m in MODES for m in modes), modes
    assert args.n_tasks >= 2, "need the faulty task and at least one victim"
    out_dir = fault_dir(args.exp_dir, args.fault_kind)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = task_names(args.n_tasks)
    print(f"{args.n_tasks} tasks {tasks}, {args.clients} closed-loop clients each; "
          f"{args.fault_kind} fault in {FAULT_TENANT} at t={args.fault_at}s of "
          f"{args.duration}s -> {out_dir}")

    for mode in modes:
        print(f"--- {mode}", flush=True)
        try:
            events, summary = run_fault_mode(mode, args)
        except RuntimeError as e:
            print(f"    FAILED: {e}; skipping", flush=True)
            continue
        with open(out_dir / f"events_{mode}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["kind", "comp", "tenant", "rid", "t", "n_tok"])
            wr.writeheader()
            wr.writerows(events)
        with open(out_dir / "summary.jsonl", "a") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"    victims: ok={summary['victims_ok']} fail={summary['victims_fail']}  "
              f"{FAULT_TENANT}: ok={summary['ok'][FAULT_TENANT]} "
              f"fail={summary['fail'][FAULT_TENANT]}  restarts={summary['restarts']}",
              flush=True)


if __name__ == "__main__":
    main()
