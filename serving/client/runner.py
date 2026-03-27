"""Trace-driven inference client.

Generates requests and sends them directly to device gRPC servers at their
scheduled wall-clock timestamps. No MQTT, no pre-stored trace buffer, no
priority queue needed.

Key design:
  - Trace is generated once upfront (just req_id, task, req_time).
  - warmup() pre-establishes all gRPC connections before the clock starts,
    eliminating the cold-start latency spike at t=0.
  - run(start_epoch) dispatches each request when wall_time >= start_epoch + req_time.
    Routing is resolved lazily via live_plan at dispatch time, so add_task()
    updates are picked up automatically without re-generating trace.
  - Results are returned as a list of tuples (same format as runtime_executor)
    alongside a req_metadata dict for CSV writing.

Usage:
    runner = TraceRunner(live_plan, trace, output_dir)
    await runner.warmup()
    results, req_metadata = await runner.run()
"""

import asyncio
import gc
import json
import os
import statistics
import time
from collections import defaultdict
from typing import List, Optional

import numpy as np
from torch.utils.data import DataLoader

from site_manager.grpc_client import EdgeRuntimeClient
from orchestrator.router import parse_plan

_rng = np.random.default_rng(42)


# ── Dataset / dataloader cache ─────────────────────────────────────────────

_DATA: dict = {}


def _percentile(values: list, q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if q >= 100:
        return float(s[-1])
    k = (len(s) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def _initialize_data():
    """Load one batch per task into _DATA. Called once before run()."""
    global _DATA
    if _DATA:
        return
    from fmtk.datasetloaders.etth1 import ETTh1Dataset
    from fmtk.datasetloaders.weather import WeatherDataset
    from fmtk.datasetloaders.exchange import ExchangeDataset
    from fmtk.datasetloaders.ecg5000 import ECG5000Dataset
    from fmtk.datasetloaders.uwavegesture import UWaveGestureLibraryALLDataset
    from fmtk.datasetloaders.ppg import PPGDataset
    from fmtk.datasetloaders.ecl import ECLDataset
    from fmtk.datasetloaders.traffic import TrafficDataset
    from site_manager.config import DATASET_DIR, DEFAULT_BATCH_SIZE

    d = DATASET_DIR
    cfg = {"batch_size": DEFAULT_BATCH_SIZE, "shuffle": False}
    loaders = {
        "ecgclass":     DataLoader(ECG5000Dataset({"dataset_path": f"{d}/ECG5000"}, {"task_type": "classification"}, "test"), **cfg),
        # "heartrate":    DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data"}, {"task_type": "regression", "label": "hr"}, "test"), **cfg),
        # "diasbp":       DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data"}, {"task_type": "regression", "label": "diasbp"}, "test"), **cfg),
        # "sysbp":        DataLoader(PPGDataset({"dataset_path": f"{d}/PPG-data"}, {"task_type": "regression", "label": "sysbp"}, "test"), **cfg),
        "gestureclass": DataLoader(UWaveGestureLibraryALLDataset({"dataset_path": f"{d}/UWaveGestureLibraryAll"}, {"task_type": "classification"}, "test"), **cfg),
        "etth1fore":    DataLoader(ETTh1Dataset({"dataset_path": f"{d}/ETTh1"}, {"task_type": "forecasting"}, "test"), **cfg),
        "weatherfore":  DataLoader(WeatherDataset({"dataset_path": f"{d}/Weather"}, {"task_type": "forecasting"}, "test"), **cfg),
        "trafficfore":  DataLoader(TrafficDataset({"dataset_path": f"{d}/Traffic"}, {"task_type": "forecasting"}, "test"), **cfg),
        "eclfore":      DataLoader(ECLDataset({"dataset_path": f"{d}/ElectricityLoad-data"}, {"task_type": "forecasting"}, "test"), **cfg),
        "exchangefore": DataLoader(ExchangeDataset({"dataset_path": f"{d}/Exchange"}, {"task_type": "forecasting"}, "test"), **cfg),
    }
    _DATA = {task: next(iter(loader)) for task, loader in loaders.items()}
    print(f"[TraceRunner] Initialized {len(_DATA)} dataloaders.")


# ── gRPC client cache (per-runner asyncio loop) ────────────────────────────

_CLIENT_CACHE: dict = {}


async def _get_client(url: str, client_key: str = "default") -> Optional[EdgeRuntimeClient]:
    cache_key = (url, client_key)
    if cache_key not in _CLIENT_CACHE:
        client = EdgeRuntimeClient(url)
        try:
            await client.wait_ready()
            _CLIENT_CACHE[cache_key] = client
            print(f"[TraceRunner] Connected to {url} (client_key={client_key})")
        except Exception as e:
            print(f"[TraceRunner] Failed to connect to {url}: {e}")
            return None
    return _CLIENT_CACHE[cache_key]


async def _close_clients():
    for (_, _), client in list(_CLIENT_CACHE.items()):
        try:
            await client.close()
        except Exception:
            pass
    _CLIENT_CACHE.clear()


# ── Per-request send ───────────────────────────────────────────────────────

async def _send_request(req_id, device_url, inputs_dict, outputs_dict, client_key: str = "default"):
    """Send one inference request and return a result tuple."""
    send_time = time.time()
    client = await _get_client(device_url, client_key=client_key)
    if client is None:
        return None

    submit_time = time.time()
    try:
        response = await client.infer({
            "req_id": req_id,
            "task": inputs_dict["task"],
            "x": inputs_dict["x"],
            "mask": inputs_dict.get("mask"),
            "question": inputs_dict.get("question"),
        })
    except Exception as e:
        print(f"[TraceRunner] Inference error req {req_id}: {e}")
        return None

    recv_time = time.time()
    dev_start = response["start_time_ns"] / 1e9
    dev_end   = response["end_time_ns"]   / 1e9
    proc_time = response["proc_time_ns"]  / 1e9
    swap_time = response["swap_time_ns"]  / 1e9
    dec_time  = response["decoder_time_ns"] / 1e9
    result    = response["output"]
    pred  = result.item() if getattr(result, "size", 1) == 1 else result.flatten().tolist()
    true_val = outputs_dict.get("y")
    true_val = true_val.item() if true_val.size == 1 else true_val.flatten().tolist()

    return (
        req_id, device_url,
        send_time, submit_time, dev_start, dev_end, recv_time,
        recv_time - send_time,   # e2e
        proc_time, swap_time, dec_time,
        pred, true_val,
    )


# ── TraceRunner ────────────────────────────────────────────────────────────

class TraceRunner:
    """Drives a pre-generated trace against live device servers.

    Args:
        live_plan:  Deployment plan dict (same structure as deployment_plan.json).
                    Updated in-place by add_task() — routing picks up changes automatically.
        trace:      List of request objects or dicts with req_id, task, req_time fields.
        output_dir: Where to write request_latency_results.csv and serving_timing_summary.json.
    """

    def __init__(self, live_plan: dict, trace: list, output_dir: str, pretrace_warmup_secs: float = 0.0):
        self.live_plan = live_plan
        self.output_dir = output_dir
        self._pretrace_warmup_secs = max(0.0, float(pretrace_warmup_secs))

        # Normalise trace to plain dicts
        self._trace = [
            {"req_id": r.req_id, "task": r.task, "req_time": r.req_time}
            if hasattr(r, "req_id") else dict(r)
            for r in trace
        ]
        # Sort once upfront — no priority queue needed
        self._trace.sort(key=lambda r: r["req_time"])

        self._results: list = []
        self._req_metadata: dict = {}   # req_id → enriched req dict
        self._plan_version: int = 0          # bump when live_plan changes
        self._plan_cache: tuple = (-1, None, None) # (version, task_routes, route_table)
        self._input_cache: dict = {}         # task → (inputs_dict, outputs_dict), pre-built
        self._dispatch_lag_ms: list = []     # scheduling delay vs target send timestamp

    def _refresh_route_cache_if_needed(self):
        if self._plan_version != self._plan_cache[0]:
            task_routes, task_totals = parse_plan(self.live_plan)
            route_table = {}
            for task_name, routes in task_routes.items():
                total = task_totals[task_name]
                probs = np.array([r[3] for r in routes]) / total
                route_table[task_name] = (routes, probs)
            self._plan_cache = (self._plan_version, task_routes, route_table)

    async def _run_pretrace_warmup(self):
        """Replay an initial prefix of the trace before t=0 to warm real traffic paths."""
        if self._pretrace_warmup_secs <= 0:
            return

        warm_reqs = [r for r in self._trace if r["req_time"] <= self._pretrace_warmup_secs]
        if not warm_reqs:
            return

        print(
            f"[TraceRunner] Running pre-trace warmup for {self._pretrace_warmup_secs:.1f}s "
            f"({len(warm_reqs)} requests)..."
        )
        start_epoch = time.time()
        inflight = set()
        max_inflight = 500
        routed_rng = np.random.default_rng(2026)
        warmup_req_id = -1_000_000
        successful = 0

        def _collect_done(task: asyncio.Task):
            nonlocal successful
            inflight.discard(task)
            try:
                result = task.result()
                if result is not None:
                    successful += 1
            except Exception as e:
                print(f"[TraceRunner] Pre-trace warmup task error: {e}")

        self._refresh_route_cache_if_needed()
        route_table = self._plan_cache[2]

        for req in warm_reqs:
            target = start_epoch + req["req_time"]
            now = time.time()
            if target > now:
                await asyncio.sleep(target - now)

            entry = route_table.get(req["task"])
            if entry is None:
                continue
            routes, probs = entry
            idx = int(routed_rng.choice(len(routes), p=probs))
            _, device_url, _, _ = routes[idx]

            cached = self._input_cache.get(req["task"])
            if cached is None:
                continue
            inputs, outputs = cached

            while len(inflight) >= max_inflight:
                await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)

            t = asyncio.create_task(
                _send_request(warmup_req_id, device_url, inputs, outputs, client_key=req["task"])
            )
            warmup_req_id -= 1
            t.add_done_callback(_collect_done)
            inflight.add(t)

        if inflight:
            await asyncio.wait(inflight)
        print(f"[TraceRunner] Pre-trace warmup complete. {successful}/{len(warm_reqs)} succeeded.")

    # ── add_requests ──────────────────────────────────────────────────────

    def invalidate_plan_cache(self):
        """Call after live_plan is updated (add_task / add_workload) to force
        route_single and parse_plan to re-read the updated plan."""
        self._plan_version += 1

    def add_requests(self, new_trace: list):
        """Add new requests to the trace at runtime (called by add_task / add_workload).

        Thread-safe: appends to _trace and re-sorts. The run() loop will pick
        up the new entries naturally since it iterates over _trace in timestamp order.

        Note: only effective if called before run() has dispatched those timestamps.
        For requests with req_time >= current elapsed, they will be dispatched on time.
        """
        new_reqs = [
            {"req_id": r.req_id, "task": r.task, "req_time": r.req_time}
            if hasattr(r, "req_id") else dict(r)
            for r in new_trace
        ]
        self._trace.extend(new_reqs)
        self._trace.sort(key=lambda r: r["req_time"])
        print(f"[TraceRunner] Added {len(new_reqs)} requests. Total trace: {len(self._trace)}")

    # ── warmup ────────────────────────────────────────────────────────────

    async def warmup(self):
        """Load datasets and pre-establish gRPC connections to all devices in live_plan.

        Call this before run() so the clock starts with warm channels and data
        already loaded. Both operations happen here so start_epoch is set only
        after all slow one-time work is complete.
        """
        # Load dataloaders first — this can take 1-3 minutes and must complete
        # before start_epoch is set, otherwise all requests arrive "in the past"
        # and the entire trace is dispatched as an instant burst.
        await asyncio.get_event_loop().run_in_executor(None, _initialize_data)

        await _close_clients()

        task_routes, _ = parse_plan(self.live_plan)
        devices = {dev for routes in task_routes.values() for _, dev, _, _ in routes}

        if not devices:
            print("[TraceRunner] WARNING: no devices found in live_plan for warmup.")
            return

        print(f"[TraceRunner] Warming up gRPC connections to {len(devices)} device(s): {sorted(devices)}")
        try:
            trace_tasks = {req["task"] for req in self._trace}
            connect_jobs = []
            for task_name, routes in task_routes.items():
                if task_name not in trace_tasks:
                    continue
                for _, dev, _, _ in routes:
                    connect_jobs.append(_get_client(dev, client_key=task_name))
            await asyncio.wait_for(
                asyncio.gather(*connect_jobs),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            missing = []
            for task_name, routes in task_routes.items():
                if task_name not in trace_tasks:
                    continue
                for _, dev, _, _ in routes:
                    if (dev, task_name) not in _CLIENT_CACHE:
                        missing.append(f"{dev}[{task_name}]")
            raise RuntimeError(
                f"[TraceRunner] Warmup timed out after 60s. "
                f"These devices did not respond: {missing}. "
                f"Check that device servers started successfully."
            )
        print("[TraceRunner] All gRPC connections ready.")

        # Model warmup — send one dummy inference per (task, device) for only the
        # tasks actually present in the trace, triggering GPU kernel JIT compilation
        # and warming CUDA caches before the clock starts.
        trace_tasks = {req["task"] for req in self._trace}
        task_routes, _ = parse_plan(self.live_plan)
        warmup_jobs = []
        warmup_id = -1
        for task, routes in task_routes.items():
            if task not in trace_tasks:
                continue
            batch = _DATA.get(task)
            if batch is None:
                continue
            inputs = {"task": task, "x": batch["x"].numpy().astype(np.float32)}
            if "mask" in batch:
                inputs["mask"] = batch["mask"].numpy().astype(np.float32)
            if "question" in batch:
                inputs["question"] = batch["question"]
            outputs = {"y": batch["y"].numpy().astype(np.float32)}
            for _, dev, _, _ in routes:
                warmup_jobs.append(
                    _send_request(warmup_id, dev, inputs, outputs, client_key=task)
                )
                warmup_id -= 1

        if warmup_jobs:
            print(f"[TraceRunner] Running model warmup ({len(warmup_jobs)} inferences)...")
            await asyncio.gather(*warmup_jobs, return_exceptions=True)
            print("[TraceRunner] Model warmup complete.")

        # Pre-build numpy input arrays once per task — reused for every request.
        # Avoids repeated .numpy().astype() allocations in the hot dispatch loop.
        self._input_cache = {}
        for task in {req["task"] for req in self._trace}:
            batch = _DATA.get(task)
            if batch is None:
                continue
            inp = {"task": task, "x": batch["x"].numpy().astype(np.float32)}
            if "mask" in batch:
                inp["mask"] = batch["mask"].numpy().astype(np.float32)
            if "question" in batch:
                inp["question"] = batch["question"]
            out = {"y": batch["y"].numpy().astype(np.float32)}
            self._input_cache[task] = (inp, out)

        await self._run_pretrace_warmup()

    # ── run ───────────────────────────────────────────────────────────────

    async def run(self, start_epoch: float = None) -> tuple:
        """Dispatch all trace requests at their scheduled times.

        Args:
            start_epoch: Wall-clock time for t=0. Defaults to now.
                         Pass a future time to allow a buffer after warmup.

        Returns:
            (results, req_metadata) — same format as runtime_executor.
        """
        if start_epoch is None:
            start_epoch = time.time()

        inflight = set()
        completed: list = []
        req_metadata: dict = {}
        dispatched_idx = 0   # how many requests from _trace we have dispatched
        max_inflight = 500
        self._dispatch_lag_ms = []

        def _collect_done(task: asyncio.Task):
            inflight.discard(task)
            try:
                completed.append(task.result())
            except Exception as e:
                print(f"[TraceRunner] Task error: {e}")
                completed.append(None)

        self._refresh_route_cache_if_needed()

        print(f"[TraceRunner] Starting trace dispatch at epoch={start_epoch:.3f} "
              f"({len(self._trace)} requests)")

        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()

        try:
            while True:
                # Check if there are undispatched requests (add_requests may have added more)
                if dispatched_idx >= len(self._trace):
                    # Nothing left — but wait briefly in case add_requests() adds more
                    await asyncio.sleep(0.05)
                    if dispatched_idx >= len(self._trace):
                        break   # truly done
                    continue

                req = self._trace[dispatched_idx]

                # Sleep until this request is due
                target = start_epoch + req["req_time"]
                now = time.time()
                if target > now:
                    await asyncio.sleep(target - now)
                dispatch_now = time.time()
                self._dispatch_lag_ms.append(max(0.0, (dispatch_now - target) * 1000.0))

                dispatched_idx += 1

                # Refresh routing cache only when plan changes (add_task / add_workload)
                self._refresh_route_cache_if_needed()
                task_routes, route_table = self._plan_cache[1], self._plan_cache[2]

                # Route this request lazily against the live plan
                entry = route_table.get(req["task"])
                if entry is None:
                    print(f"[TraceRunner] Skipping req {req['req_id']}: no route for '{req['task']}'")
                    continue
                routes, probs = entry
                idx = int(_rng.choice(len(routes), p=probs))
                site_mgr, device_url, backbone, _ = routes[idx]

                req_metadata[req["req_id"]] = {
                    **req,
                    "backbone": backbone,
                    "site_manager": site_mgr,
                }

                # Look up pre-built numpy arrays (built once in warmup, reused every request)
                cached = self._input_cache.get(req["task"])
                if cached is None:
                    print(f"[TraceRunner] WARNING: no dataloader for '{req['task']}', skipping.")
                    continue
                inputs, outputs = cached

                while len(inflight) >= max_inflight:
                    await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)

                t = asyncio.create_task(
                    _send_request(req["req_id"], device_url, inputs, outputs, client_key=req["task"])
                )
                t.add_done_callback(_collect_done)
                inflight.add(t)
        finally:
            if gc_was_enabled:
                gc.enable()

        # Wait for all remaining in-flight requests
        if inflight:
            print(f"[TraceRunner] Waiting for {len(inflight)} in-flight requests...")
            await asyncio.wait(inflight)

        self._results = completed
        self._req_metadata = req_metadata
        if self._dispatch_lag_ms:
            print(
                "[TraceRunner] Dispatch lag ms "
                f"p50={_percentile(self._dispatch_lag_ms,50):.2f} "
                f"p95={_percentile(self._dispatch_lag_ms,95):.2f} "
                f"p99={_percentile(self._dispatch_lag_ms,99):.2f} "
                f"max={_percentile(self._dispatch_lag_ms,100):.2f}"
            )
        print(f"[TraceRunner] Done. {len([r for r in completed if r])} successful / "
              f"{len(completed)} total requests.")
        return completed, req_metadata

    # ── save_results ──────────────────────────────────────────────────────

    def save_results(self):
        """Write request_latency_results.csv and serving_timing_summary.json."""
        if not self._results:
            print("[TraceRunner] No results to save.")
            return

        _write_csv(self._results, self._req_metadata, self.output_dir)


# ── CSV / summary writer ───────────────────────────────────────────────────

def _write_csv(reqs_latency: list, req_metadata: dict, output_dir: str):
    """Write latency CSV and timing summary — identical columns to MQTT _save_results."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path     = os.path.join(output_dir, "request_latency_results.csv")
    summary_path = os.path.join(output_dir, "serving_timing_summary.json")

    valid = [e for e in reqs_latency if e is not None]
    per_device = defaultdict(list)
    for entry in valid:
        per_device[entry[1]].append(entry)

    # Correct cross-host clock skew for mixed timestamps:
    # client_submit_time (client clock) vs device_start/end_time (server clock).
    # We estimate one offset per device from median(device_start - client_submit).
    device_clock_offset = {}
    for device_url, entries in per_device.items():
        offsets = [e[4] - e[3] for e in entries]
        device_clock_offset[device_url] = statistics.median(offsets) if offsets else 0.0

    timing_summary = {"devices": {}}
    for device_url, entries in per_device.items():
        entries.sort(key=lambda x: x[4])  # sort by device_start_time
        count = len(entries)
        prep_ms           = sum((e[3] - e[2]) * 1000 for e in entries) / count
        offset = device_clock_offset.get(device_url, 0.0)
        submit_to_back_ms = sum((e[4] - e[3] - offset) * 1000 for e in entries) / count
        backend_exec_ms   = sum((e[5] - e[4]) * 1000 for e in entries) / count
        back_to_client_ms = sum((e[6] - e[5] + offset) * 1000 for e in entries) / count
        e2e_ms            = sum(e[7] * 1000           for e in entries) / count

        start_to_start_ms, idle_gap_ms, overlap_count = None, None, 0
        if count > 1:
            s2s, gaps = [], []
            prev_s, prev_e = entries[0][4], entries[0][5]
            for e in entries[1:]:
                s2s.append((e[4] - prev_s) * 1000)
                gap = (e[4] - prev_e) * 1000
                if gap < 0:
                    overlap_count += 1
                    gap = 0.0
                gaps.append(gap)
                prev_s, prev_e = e[4], e[5]
            start_to_start_ms = sum(s2s) / len(s2s)
            idle_gap_ms       = sum(gaps) / len(gaps)

        timing_summary["devices"][device_url] = {
            "request_count": count,
            "avg_client_prep_ms": prep_ms,
            "avg_client_submit_to_backend_start_ms": submit_to_back_ms,
            "avg_backend_exec_ms": backend_exec_ms,
            "avg_backend_to_client_return_ms": back_to_client_ms,
            "avg_end_to_end_ms": e2e_ms,
            "avg_backend_start_to_start_ms": start_to_start_ms,
            "avg_backend_idle_gap_ms": idle_gap_ms,
            "backend_overlap_pairs": overlap_count,
        }

    with open(csv_path, "w") as f:
        f.write(
            "req_id,req_time,site_manager,device,backbone,task,"
            "site_manager_send_time,client_infer_submit_time,"
            "device_start_time,device_end_time,client_receive_time,"
            "client_prep_time(ms),client_submit_to_backend_start(ms),"
            "backend_exec_time(ms),backend_to_client_return(ms),"
            "end_to_end_latency(ms),proc_time(ms),swap_time(ms),decoder_time(ms),"
            "pred,true\n"
        )
        for entry in valid:
            (req_id, device_url, send_t, submit_t, dev_start, dev_end,
             recv_t, e2e, proc, swap, dec, pred, true_val) = entry
            req       = req_metadata.get(req_id, {})
            req_time  = req.get("req_time", -1)
            task      = req.get("task", "unknown")
            backbone  = req.get("backbone", "unknown")
            site_mgr  = req.get("site_manager", "local")
            prep      = (submit_t - send_t) * 1000
            offset    = device_clock_offset.get(device_url, 0.0)
            sub_ms    = (dev_start - submit_t - offset) * 1000
            exec_ms   = (dev_end - dev_start) * 1000
            ret_ms    = (recv_t - dev_end + offset) * 1000
            f.write(
                f"{req_id},{req_time},{site_mgr},{device_url},{backbone},{task},"
                f"{send_t},{submit_t},{dev_start},{dev_end},{recv_t},"
                f"{prep},{sub_ms},{exec_ms},{ret_ms},{e2e*1000},"
                f"{proc*1000},{swap*1000},{dec*1000},{pred},{true_val}\n"
            )

    with open(summary_path, "w") as f:
        json.dump(timing_summary, f, indent=2)

    print(f"[TraceRunner] Saved {len(valid)} results → {csv_path}")
