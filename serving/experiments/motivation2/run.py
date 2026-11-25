#!/usr/bin/env python3
"""motivation2/run.py — the motivation figure's experiments, against the
device server (device/main.py). run.sh starts and stops the servers; this
script deploys the tasks, sends the load and records every request.

Memory is what the device server reports in every InferResponse:
gpu_alloc_peak_mb, the server process's torch.cuda.memory_allocated after
the batch's backbone and decoders (device/runtime.py, device/batcher.py).
With --staging gpu the queued inputs are torch tensors in that process, so a
backlog shows up in it. It leaves out the CUDA context and the allocator's
cached-but-free memory. One sample per completed request, timestamped with
the server's end_time_ns.

Every task is a depth task on one DINOv2 backbone (monocular decoder,
nyudepth weights, its own copy per task, named by task: B, A1, ... / T0, ...).
Inputs are one random fp32 3 x res x res image per task (the values do not
matter), encoded once and reused for every request.

Each task sends from its own process, on its own gRPC channel, so a flood
from one task cannot delay another task's sends in the client. Isolation
runs send on one InferStream per task, so a request waiting on the GPU holds
no host copy of its input (see _open_loop); efficiency runs use Infer.

--experiment isolation   (panels b, c, d)
    B         the noisy task: open-loop Poisson, --rate-b req/s, then from
              --ramp-at s a linear ramp to --rate-b-max over --ramp-s s,
              held to the end.
    A1..      the victims: open-loop Poisson, --rate-a req/s each.
    Arrivals come from one seeded stream per task, so every condition sees
    the same arrival times. Which server serves which task is --task-urls
    (run.sh: shared = one server for all; mig = one server per MIG slice;
    no_sharing = one server per task on the whole GPU).

--experiment efficiency  (panel a)
    --task-urls tasks, closed loop: each keeps --concurrency requests in
    flight for --duration s. Throughput and memory exclude --warmup s.
    shared = one server with every task; partitioned = one server per task.

Output, in --out-dir:
    isolation:  events_<condition>.csv (one row per request) and a line in
                summary.jsonl
    efficiency: a row in summary.csv (one per condition x n_tasks)

Usage (from serving/; run.sh does this):
    python experiments/motivation2/run.py --experiment isolation --condition shared \\
        --task-urls B=localhost:8000,A1=localhost:8000 --out-dir experiments/motivation2/results
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import numpy as np  # noqa: E402

DECODER_PATH = "nyudepth_{bb}_monocular"
SEED_BASE = 1000
START_DELAY_S = 3.0          # children start sending this long after the parent picks t0


# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

async def deploy(task_urls: dict, backbone: str, decoder_path: str | None = None) -> dict:
    """Load the backbone and one depth decoder per task on each server.
    decoder_path: FMTK's path under models/vision/finetuned (default
    DECODER_PATH for the backbone). Returns {url: status}."""
    from site_manager.grpc_client import EdgeRuntimeClient
    bb = backbone.replace("-patch", "")
    path = decoder_path or DECODER_PATH.format(bb=bb)
    by_url: dict = {}
    for t, u in task_urls.items():
        by_url.setdefault(u, []).append(t)
    out = {}
    for url, tasks in by_url.items():
        decoders = [{"task": t, "type": "monocular", "base_task": "nyudepth",
                     "path": path} for t in tasks]
        c = EdgeRuntimeClient(url)
        try:
            await asyncio.wait_for(c.wait_ready(), timeout=120)
            resp = await c.control("load", json.dumps({"backbone": backbone, "decoders": decoders}),
                                   timeout_s=600)
            out[url] = resp["status"]
        except Exception as exc:
            out[url] = f"error_{type(exc).__name__}: {exc}"
        finally:
            await c.close()
        print(f"[Deploy] {url}  tasks={tasks}  status={out[url]}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Arrivals
# ---------------------------------------------------------------------------

def poisson_times(seed: int, duration: float, rate_fn) -> list:
    """Arrival times of a Poisson process with rate rate_fn(t), by thinning."""
    rng = np.random.default_rng(seed)
    r_max = max(rate_fn(t) for t in np.linspace(0, duration, 1000))
    out, t = [], 0.0
    while True:
        t += rng.exponential(1.0 / r_max)
        if t >= duration:
            return out
        if rng.random() * r_max <= rate_fn(t):
            out.append(t)


def rate_fn_for(task: str, a) -> callable:
    if task != a.surging:
        return lambda t: a.rate_a

    def f(t: float) -> float:
        if t < a.ramp_at:
            return a.rate_b
        if a.ramp_s <= 0 or t >= a.ramp_at + a.ramp_s:
            return a.rate_b_max
        return a.rate_b + (a.rate_b_max - a.rate_b) * (t - a.ramp_at) / a.ramp_s
    return f


# ---------------------------------------------------------------------------
# One task's sender (its own process)
# ---------------------------------------------------------------------------

def _channel(url: str):
    import grpc
    # Own subchannel: tasks to the same server do not share a TCP connection.
    return grpc.aio.insecure_channel(url, options=[
        ("grpc.use_local_subchannel_pool", 1),
        ("grpc.max_send_message_length", 64 << 20),
        ("grpc.max_receive_message_length", 64 << 20),
    ])


def _request(task: str, res: int, seed: int):
    from site_manager.grpc_client import encode_infer_request
    x = np.random.default_rng(seed).standard_normal((1, 3, res, res)).astype(np.float32)
    return encode_infer_request(task, x)


async def _open_loop(task: str, url: str, times: list, t0: float, res: int, seed: int,
                     end_s: float, req_timeout: float) -> list:
    """Send at times[i] (s after t0) on one InferStream; responses come back
    in completion order, matched by req_id. Rows: (t_send, t_done, ok,
    lat_ms, mem_mb, mem_t, code).

    A stream, not a call per request: gRPC keeps an open unary call's input
    bytes in host RAM until it is answered (in the server, 2-3x the payload),
    so a GPU backlog of unary calls fills host RAM first. A request read off
    a stream holds no host copy (device/server.py InferStream).

    A request the server fails gets its status as code ("queue_full",
    "error: ..."). Requests still open at end_s are cut (UNFINISHED). If the
    stream breaks, open requests fail then and the rest fail at their send
    times, as sent while the server is down."""
    import grpc
    from device.proto import edge_runtime_pb2_grpc
    ch = _channel(url)
    stub = edge_runtime_pb2_grpc.EdgeRuntimeStub(ch)
    req = _request(task, res, seed)
    rows: list = []
    sent: dict = {}                 # req_id -> t_send, until answered
    n_written = 0
    call = stub.InferStream()

    def code_of(exc: BaseException) -> str:
        return exc.code().name if isinstance(exc, grpc.aio.AioRpcError) else type(exc).__name__

    async def receive() -> None:
        async for r in call:
            ts = sent.pop(r.req_id, None)
            if ts is None:
                continue
            now = time.time() - t0
            ok = not r.status or r.status == "ok"
            rows.append((ts, now, ok, 1e3 * (now - ts) if ok else None,
                         r.gpu_alloc_peak_mb if ok else None,
                         (r.end_time_ns / 1e9 - t0 if r.end_time_ns else now) if ok else None,
                         "" if ok else r.status[:80]))

    async def send_all() -> None:
        nonlocal n_written
        for i, ts in enumerate(times):
            wait = t0 + ts - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            if recv_task.done():        # the stream ended early
                return
            req.req_id = i              # serialized by write(), so reusing req is safe
            sent[i] = ts
            await call.write(req)
            n_written = i + 1
        await call.done_writing()
        await asyncio.shield(recv_task)

    recv_task = asyncio.create_task(receive())
    broken = None                       # (t, code): the stream failed at t
    try:
        await asyncio.wait_for(send_all(), timeout=max(0.1, t0 + end_s - time.time()))
    except asyncio.TimeoutError:
        pass
    except Exception as exc:
        broken = (time.time() - t0, code_of(exc))
    if broken is None and recv_task.done() and not recv_task.cancelled() \
            and recv_task.exception() is not None:
        broken = (time.time() - t0, code_of(recv_task.exception()))
    if broken is None and n_written < len(times) and recv_task.done():
        broken = (time.time() - t0, "STREAM_ENDED")
    call.cancel()
    recv_task.cancel()

    # Open requests: failed when the stream broke, else cut at end_s.
    t_cut, code_open = broken if broken else (end_s, "UNFINISHED")
    for ts in sent.values():
        rows.append((ts, max(ts, t_cut), False, None, None, None, code_open))
    # Never written: the stream broke, or writes were held back (the server
    # stopped reading) until end_s.
    for i in range(n_written, len(times)):
        if i not in sent:
            ts = times[i]
            rows.append((ts, max(ts, broken[0]) if broken else ts, False, None, None, None,
                         broken[1] if broken else "NOT_SENT"))
    await ch.close()
    return rows


async def _closed_loop(task: str, url: str, t0: float, res: int, seed: int,
                       duration: float, concurrency: int, req_timeout: float) -> list:
    import grpc
    from device.proto import edge_runtime_pb2_grpc
    ch = _channel(url)
    stub = edge_runtime_pb2_grpc.EdgeRuntimeStub(ch)
    req = _request(task, res, seed)
    rows: list = []

    async def worker() -> None:
        while time.time() - t0 < duration:
            ts = time.time() - t0
            try:
                r = await stub.Infer(req, timeout=req_timeout)
                now = time.time() - t0
                rows.append((ts, now, True, 1e3 * (now - ts), r.gpu_alloc_peak_mb,
                             r.end_time_ns / 1e9 - t0 if r.end_time_ns else now, ""))
            except grpc.aio.AioRpcError as exc:
                rows.append((ts, time.time() - t0, False, None, None, None, exc.code().name))
                await asyncio.sleep(0.1)

    wait = t0 - time.time()
    if wait > 0:
        await asyncio.sleep(wait)
    await asyncio.gather(*[worker() for _ in range(concurrency)])
    await ch.close()
    return rows


def _child(kind: str, kw: dict, out_q) -> None:
    try:
        coro = _open_loop(**kw) if kind == "open" else _closed_loop(**kw)
        out_q.put((kw["task"], asyncio.run(coro), None))
    except Exception as exc:
        out_q.put((kw["task"], [], f"{type(exc).__name__}: {exc}"))


class StatsPoller:
    """Polls Control("stats") on every server every period_s, in a thread of
    the parent while the senders run: GPU memory as the device server reports
    it (device/memstats.py), sampled on a timer, so it goes on when no
    request completes. Rows: {"t": s after t0, "round": k, "url": ..., **stats}."""

    def __init__(self, urls, t0: float, period_s: float = 0.5):
        import threading
        self.urls, self.t0, self.period_s = sorted(set(urls)), t0, period_s
        self.rows: list = []
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, name="stats", daemon=True)

    def _loop(self) -> None:
        import grpc
        from device.proto import edge_runtime_pb2 as pb, edge_runtime_pb2_grpc as pbg
        stubs = {u: pbg.EdgeRuntimeStub(grpc.insecure_channel(u)) for u in self.urls}
        req = pb.ControlRequest(command="stats", payload_json="")
        k, nxt = 0, time.time()
        while not self._stop.is_set():
            for u, stub in stubs.items():
                try:
                    r = stub.Control(req, timeout=2.0)
                    self.rows.append({"t": round(time.time() - self.t0, 3), "round": k,
                                      "url": u, **json.loads(r.logger_summary)})
                except Exception:
                    pass                # server down or busy: no sample
            k += 1
            nxt += self.period_s
            self._stop.wait(max(0.0, nxt - time.time()))

    def __enter__(self):
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join(timeout=5)


def append_csv(path: Path, row: dict) -> None:
    """Append row to path, rewriting the file if row has new columns."""
    old, fields = [], list(row)
    if path.exists():
        with path.open() as f:
            rd = csv.DictReader(f)
            old = list(rd)
            fields = list(rd.fieldnames or []) + [k for k in row if k not in (rd.fieldnames or [])]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(old + [row])


def run_senders(kind: str, per_task_kw: dict) -> dict:
    """One process per task; returns {task: rows}."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_child, args=(kind, kw, q), name=f"send-{t}", daemon=True)
             for t, kw in per_task_kw.items()]
    for p in procs:
        p.start()
    out = {}
    for _ in procs:
        task, rows, err = q.get()
        if err:
            print(f"[Send] {task}: sender failed: {err}", flush=True)
        out[task] = rows
    for p in procs:
        p.join()
    return out


# ---------------------------------------------------------------------------
# GPU capacity (for the plot's reference lines only; memory itself comes from
# the device server's responses)
# ---------------------------------------------------------------------------

def capacity_mb(gpu_index: int, mig_uuid: str | None) -> tuple:
    try:
        import pynvml
        pynvml.nvmlInit()
        total = pynvml.nvmlDeviceGetMemoryInfo(
            pynvml.nvmlDeviceGetHandleByIndex(gpu_index)).total / 2**20
        sl = None
        if mig_uuid:
            try:
                h = pynvml.nvmlDeviceGetHandleByUUID(mig_uuid)
            except Exception:
                h = pynvml.nvmlDeviceGetHandleByUUID(mig_uuid.encode())
            sl = pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**20
        return round(total, 1), (round(sl, 1) if sl else None)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def parse_urls(s: str) -> dict:
    out = {}
    for pair in s.split(","):
        t, u = pair.split("=", 1)
        out[t.strip()] = u.strip()
    return out


def isolation(a, task_urls: dict) -> None:
    tasks = list(task_urls)
    victims = [t for t in tasks if t != a.surging]
    times = {t: poisson_times(SEED_BASE + i, a.duration, rate_fn_for(t, a))
             for i, t in enumerate(tasks)}
    for t in tasks:
        print(f"[Trace] {t}: {len(times[t])} requests over {a.duration:g} s", flush=True)

    if a.warmup_burst_secs > 0:
        # As run_warmup_burst in sharing_benefit/tpc, RTVSntask/tpc, colocation:
        # a closed-loop burst, results discarded. Up to --warmup-concurrency
        # in flight per task, so batches of every size up to the max batch
        # run (cuDNN tunes per new batch shape) before the measured run.
        tw = time.time() + 1.0
        wkw = {t: dict(task=t, url=task_urls[t], t0=tw, res=a.res, seed=SEED_BASE + i,
                       duration=a.warmup_burst_secs, concurrency=a.warmup_concurrency,
                       req_timeout=a.req_timeout)
               for i, t in enumerate(tasks)}
        print(f"[Warmup] closed-loop burst for {a.warmup_burst_secs:g} s, "
              f"{a.warmup_concurrency} in flight per task ...", flush=True)
        wrows = run_senders("closed", wkw)
        print("[Warmup] done: " + ", ".join(
            f"{t}: {sum(r[2] for r in rs)} ok / {len(rs)}" for t, rs in wrows.items()), flush=True)

    t0 = time.time() + START_DELAY_S
    end_s = a.duration + a.grace
    kw = {t: dict(task=t, url=task_urls[t], times=times[t], t0=t0, res=a.res,
                  seed=SEED_BASE + i, end_s=end_s, req_timeout=a.req_timeout)
          for i, t in enumerate(tasks)}
    print(f"[Run] isolation/{a.condition}: sending for {a.duration:g} s ...", flush=True)
    with StatsPoller(task_urls.values(), t0, a.stats_period) as poller:
        rows = run_senders("open", kw)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    with (a.out_dir / f"mem_{a.condition}.jsonl").open("w") as f:
        for r in poller.rows:
            f.write(json.dumps(r) + "\n")
    print(f"[Mem] {len(poller.rows)} stats samples -> mem_{a.condition}.jsonl", flush=True)
    path = a.out_dir / f"events_{a.condition}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "url", "t_send", "t_done", "ok", "lat_ms", "mem_mb", "mem_t", "code"])
        for t in tasks:
            for ts, td, ok, lat, mem, mt, code in sorted(rows[t]):
                w.writerow([t, task_urls[t], round(ts, 4), round(td, 4), int(ok),
                            "" if lat is None else round(lat, 2),
                            "" if mem is None else round(mem, 1),
                            "" if mt is None else round(mt, 4), code])

    def rps(ts_, lo, hi):
        n = sum(1 for t in ts_ for r in rows[t] if r[2] and lo <= r[1] < hi)
        return round(n / max(1e-9, hi - lo), 2)

    per_task = {}
    for t in tasks:
        rs = rows[t]
        codes: dict = {}
        for r in rs:
            if not r[2]:
                codes[r[6]] = codes.get(r[6], 0) + 1
        mems = [r[4] for r in rs if r[2] and r[4]]
        per_task[t] = {"sent": len(times[t]), "ok": sum(r[2] for r in rs),
                       "fail": sum(not r[2] for r in rs), "fail_codes": codes,
                       "peak_mem_mb": round(max(mems), 1) if mems else None}
    gpu_total, mig_slice = capacity_mb(a.gpu_index, a.mig_uuid)
    summary = {
        "experiment": "isolation", "condition": a.condition, "time": time.strftime("%F %T"),
        "backbone": a.backbone, "res": a.res, "surging": a.surging, "victims": victims,
        "task_urls": task_urls, "rate_a": a.rate_a, "rate_b": a.rate_b,
        "rate_b_max": a.rate_b_max, "ramp_at": a.ramp_at, "ramp_s": a.ramp_s,
        "duration": a.duration, "deploy": a.deploy_status,
        "gpu_total_mb": a.gpu_total_mb or gpu_total, "mig_slice_mb": a.mig_slice_mb or mig_slice,
        "victim_rps_before": rps(victims, a.warmup, a.ramp_at),
        "victim_rps_after": rps(victims, a.ramp_at, a.duration),
        "surging_rps_before": rps([a.surging], a.warmup, a.ramp_at),
        "surging_rps_after": rps([a.surging], a.ramp_at, a.duration),
        "tasks": per_task, **json.loads(a.extra or "{}"),
    }
    with (a.out_dir / "summary.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"[Done] {path}")
    print(json.dumps({k: summary[k] for k in ("victim_rps_before", "victim_rps_after",
                                              "surging_rps_before", "surging_rps_after")}))
    for t, s in per_task.items():
        print(f"    {t}: {s}")


def efficiency(a, task_urls: dict) -> None:
    tasks = list(task_urls)
    n_servers = len(set(task_urls.values()))
    row = {"condition": a.condition, "n_tasks": len(tasks), "n_servers": n_servers,
           "backbone": a.backbone, "res": a.res, "concurrency": a.concurrency,
           "duration": a.duration, "warmup": a.warmup, "status": "ok",
           "throughput_rps": "", "avg_gpu_mem_mb": "", "peak_gpu_mem_mb": "",
           "avg_latency_ms": "", "p99_latency_ms": "", "fail": "",
           "avg_device_used_mb": "", "peak_device_used_mb": ""}
    bad = {u: s for u, s in a.deploy_status.items() if not str(s).startswith("loaded")}
    if bad:
        row["status"] = "oom" if any("memory" in str(s).lower() for s in bad.values()) else "load_failed"
        print(f"[Eff] deploy failed: {bad}", flush=True)
    else:
        t0 = time.time() + START_DELAY_S
        kw = {t: dict(task=t, url=task_urls[t], t0=t0, res=a.res, seed=SEED_BASE + i,
                      duration=a.duration, concurrency=a.concurrency, req_timeout=a.req_timeout)
              for i, t in enumerate(tasks)}
        print(f"[Run] efficiency/{a.condition} n={len(tasks)}: closed loop, "
              f"{a.concurrency} in flight per task, {a.duration:g} s ...", flush=True)
        with StatsPoller(task_urls.values(), t0, a.stats_period) as poller:
            rows = run_senders("closed", kw)
        win = a.duration - a.warmup
        # GPU used, CUDA contexts included: per poll round, the most any server
        # reports (one GPU, so every server sees the same device), in the window.
        rounds: dict = {}
        for r in poller.rows:
            if a.warmup <= r["t"] <= a.duration and r.get("device_used_mb"):
                rounds[r["round"]] = max(rounds.get(r["round"], 0.0), r["device_used_mb"])
        if rounds:
            row["avg_device_used_mb"] = round(float(np.mean(list(rounds.values()))), 1)
            row["peak_device_used_mb"] = round(max(rounds.values()), 1)
        ok = [(t, r) for t in tasks for r in rows[t] if r[2] and r[1] >= a.warmup]
        lats = [r[3] for _, r in ok]
        # Memory: per server, mean / max of its samples in the window; summed over servers.
        by_url: dict = {}
        for t, r in ok:
            if r[4]:
                by_url.setdefault(task_urls[t], []).append(r[4])
        row.update({
            "throughput_rps": round(len(ok) / win, 2),
            "avg_gpu_mem_mb": round(sum(float(np.mean(v)) for v in by_url.values()), 1),
            "peak_gpu_mem_mb": round(sum(max(v) for v in by_url.values()), 1),
            "avg_latency_ms": round(float(np.mean(lats)), 2) if lats else "",
            "p99_latency_ms": round(float(np.percentile(lats, 99)), 2) if lats else "",
            "fail": sum(1 for t in tasks for r in rows[t] if not r[2]),
        })
        if len(by_url) < n_servers:
            row["status"] = "partial"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    append_csv(a.out_dir / "summary.csv", row)
    print(f"[Done] {row}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["isolation", "efficiency"], required=True)
    ap.add_argument("--condition", required=True,
                    help="isolation: shared | mig | no_sharing; efficiency: shared | partitioned")
    ap.add_argument("--task-urls", required=True, help="task=host:port,... (order kept)")
    ap.add_argument("--backbone", default="dinobase-patch")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--decoder-path", default=None,
                    help="decoder under FMTK models/vision/finetuned "
                         "(default nyudepth_<backbone>_monocular)")
    ap.add_argument("--surging", default="B")
    ap.add_argument("--rate-a", type=float, default=110.0)
    ap.add_argument("--rate-b", type=float, default=20.0)
    ap.add_argument("--rate-b-max", type=float, default=1000.0)
    ap.add_argument("--ramp-at", type=float, default=60.0)
    ap.add_argument("--ramp-s", type=float, default=15.0)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--grace", type=float, default=10.0,
                    help="isolation: requests still open this long after --duration are cut")
    ap.add_argument("--req-timeout", type=float, default=60.0)
    ap.add_argument("--warmup-burst-secs", type=float, default=15.0,
                    help="isolation: closed-loop warm-up before the run, discarded (0: none)")
    ap.add_argument("--warmup-concurrency", type=int, default=32,
                    help="isolation: warm-up requests in flight per task (the max batch)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="efficiency: requests in flight per task (1, as motivation1)")
    ap.add_argument("--stats-period", type=float, default=0.5,
                    help="s between Control(\"stats\") polls of every server")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--mig-uuid", default=None, help="a slice in use, for its capacity")
    ap.add_argument("--gpu-total-mb", type=float, default=None)
    ap.add_argument("--mig-slice-mb", type=float, default=None)
    ap.add_argument("--extra", default=None, help="JSON merged into the summary line")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    if not a.out_dir.is_absolute():
        a.out_dir = SERVING_DIR / a.out_dir

    task_urls = parse_urls(a.task_urls)
    a.deploy_status = asyncio.run(deploy(task_urls, a.backbone, a.decoder_path))
    if a.experiment == "isolation":
        bad = {u: s for u, s in a.deploy_status.items() if not str(s).startswith("loaded")}
        if bad:
            print(f"[Run] deploy failed, not running: {bad}", flush=True)
            return 1
        if a.surging not in task_urls:
            print(f"[Run] --surging {a.surging} not in --task-urls", flush=True)
            return 1
        isolation(a, task_urls)
    else:
        efficiency(a, task_urls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
