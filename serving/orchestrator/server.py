#!/usr/bin/env python3
"""FMaaS Orchestrator — single entry point for both local and MQTT modes.

MQTT mode (geo-distributed):
    python -m orchestrator.server --mode mqtt --port 8080 --exp-type runtime
    # Then control via HTTP:
    curl -X POST http://localhost:8080/deploy
    curl -X POST http://localhost:8080/run
    curl -X POST http://localhost:8080/add-task -d '{"task_name": "gestureclass", ...}'
    curl -X POST http://localhost:8080/add-workload -d '{"task_name": "heartrate", "task_workload": 20}'
    curl -X POST http://localhost:8080/cleanup

Local mode (single process, no MQTT):
    python -m orchestrator.server --mode local --exp-type SystemInAction \
        --scheduler fmaas_share --req-rate 10 --duration 480 --trace deterministic \
        --exp-dir ./experiments/SystemInAction/results
"""

import argparse
import asyncio
import os
import threading
import time
import uvicorn
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from client.trace import generate_trace
from client.runner import TraceRunner
from orchestrator.main import Orchestrator
from orchestrator.router import route_trace
from site_manager.base import BaseSiteManager


# ============================================================================
# Pydantic Models (Request/Response schemas)
# ============================================================================

class DeployRequest(BaseModel):
    exp_type: str = "runtime"
    trace: str = "lmsyschat"
    req_rate: int = 10
    duration: int = 60
    seed: int = 42
    scheduler: str = "fmaas_share"
    exp_dir: str = "./experiments/runtime/results"


class RunRequest(BaseModel):
    pass


class AddTaskRequest(BaseModel):
    task_name: str
    task_type: str = "classification"
    task_workload: float = 8.0
    elapsed_time: float = 0.0


class AddWorkloadRequest(BaseModel):
    task_name: str
    task_type: str = "classification"
    task_workload: float = 10.0
    elapsed_time: float = 0.0


class CleanupRequest(BaseModel):
    pass


class StatusResponse(BaseModel):
    deployed: bool
    running: bool
    total_requests_generated: int
    tasks: list
    plan: Optional[Dict[str, Any]]


# ============================================================================
# Global State
# ============================================================================

class AppState:
    def __init__(self):
        self.orchestrator: Optional[Orchestrator] = None
        self.site_manager: Optional[BaseSiteManager] = None
        self.deployed = False
        self.running = False
        self.config = None

    def reset(self):
        self.deployed = False
        self.running = False


state = AppState()
app = FastAPI(title="FMaaS Orchestrator", version="1.0.0")


# ============================================================================
# Helper Functions
# ============================================================================

def load_config(exp_type: str):
    if exp_type == "baselines":
        from experiments.baselines.user_config import devices, tasks
    elif exp_type == "batching":
        from experiments.batching.user_config import devices, tasks
    elif exp_type == "runtime":
        from experiments.runtime.user_config import devices, tasks
    elif exp_type == "runtime_task":
        from experiments.runtime_task.user_config import devices, tasks
    elif exp_type == "SystemInAction":
        from experiments.SystemInAction.user_config import devices, tasks
    else:
        raise ValueError(f"Unknown exp_type: {exp_type}")
    return devices, tasks


def _fmaas_to_device_scheduler(scheduler_name: str) -> str:
    name = scheduler_name.lower()
    if name in ("fmaas", "fmaas_share"):
        return "stfq"
    if name in ("clipper-ht", "clipper-ha", "m4-ht", "m4-ha"):
        return "wfq"
    return "stfq"


def _incremental_plan_to_json(incremental_plan: dict) -> dict:
    deployments = {}
    for task, entries in incremental_plan.items():
        for site_manager, device_ip, backbone, rps in entries:
            key = (site_manager, device_ip, backbone)
            deployments.setdefault(key, {})[task] = rps

    sites = {}
    for (site_manager, device_ip, backbone), task_rates in deployments.items():
        deploy = {
            "device": device_ip,
            "backbone": backbone,
            "decoders": [{"task": t} for t in task_rates],
            "tasks": {t: {"request_per_sec": r} for t, r in task_rates.items()},
        }
        sites.setdefault(site_manager, []).append(deploy)

    return {
        "sites": [
            {"id": site_id, "deployments": deps}
            for site_id, deps in sites.items()
        ]
    }


def _normalized_backbone(backbone: str):
    if backbone and "__clipper__" in backbone:
        return backbone.split("__clipper__")[0]
    return backbone


def _shift_requests_to_elapsed(requests, target_elapsed):
    if not requests:
        return
    min_req_time = min(r.req_time for r in requests)
    shift = max(0.0, target_elapsed - min_req_time)
    if shift <= 0:
        return
    for req in requests:
        req.req_time += shift


def _send_incremental_batches(orchestrator, site_manager, trace, incremental_plan, diffs, seed, fallback_elapsed):
    """Route trace once, then send per-diff batches only after each diff ACKs."""
    inc_plan_json = _incremental_plan_to_json(incremental_plan)
    routed_trace = route_trace(trace, inc_plan_json, seed)

    diff_by_route = {}
    for diff in diffs:
        key = (diff.site_manager, diff.ip, diff.backbone)
        diff_by_route[key] = diff
        norm_key = (diff.site_manager, diff.ip, _normalized_backbone(diff.backbone))
        diff_by_route.setdefault(norm_key, diff)

    immediate = []
    pending = {}
    for req in routed_trace:
        key = (req.site_manager, req.device, req.backbone)
        diff = diff_by_route.get(key)
        if diff is not None:
            pending.setdefault(id(diff), {"diff": diff, "requests": []})["requests"].append(req)
        else:
            immediate.append(req)

    if immediate:
        site_manager.send_requests(immediate)
        orchestrator.increment_requests_generated(len(immediate))

    for batch_info in pending.values():
        diff = batch_info["diff"]
        requests = batch_info["requests"]
        if not requests:
            continue
        site_manager.apply_diff(diff)
        target_elapsed = site_manager.current_runtime_elapsed(fallback=fallback_elapsed)
        _shift_requests_to_elapsed(requests, target_elapsed)
        site_manager.send_requests(requests)
        orchestrator.increment_requests_generated(len(requests))


# ============================================================================
# Local mode: LocalExperiment
# ============================================================================

class LocalExperiment:
    """Runs a full FMaaS experiment in a single process, no MQTT.

    Typical usage:
        exp = LocalExperiment(exp_type, scheduler, req_rate, duration, trace, seed, exp_dir)
        exp.deploy()
        exp.run()

        # At runtime event times:
        exp.add_task("gestureclass", workload=8.0, elapsed=60.0)
        exp.add_task("sysbp",        workload=8.0, elapsed=300.0)
        exp.add_workload("ecgclass", delta=5.0,    elapsed=120.0)

        exp.wait()
        exp.cleanup()
    """

    def __init__(self, exp_type, scheduler, req_rate, duration, trace_type, seed, exp_dir,
                 max_batch_size=5, max_batch_wait_ms=0.0, isolation_mode="shared",
                 warmup_gap=2.0, pretrace_warmup_secs=20.0):
        self.exp_type = exp_type
        self.scheduler = scheduler
        self.req_rate = req_rate
        self.duration = duration
        self.trace_type = trace_type
        self.seed = seed
        self.max_batch_size = max_batch_size
        self.max_batch_wait_ms = max_batch_wait_ms
        self.isolation_mode = isolation_mode
        self._warmup_gap = warmup_gap
        self._pretrace_warmup_secs = pretrace_warmup_secs

        self.output_dir = os.path.abspath(os.path.join(exp_dir, scheduler, str(req_rate)))
        os.makedirs(self.output_dir, exist_ok=True)

        self.devices, self.tasks = load_config(exp_type)

        from site_manager.local import LocalSiteManager
        self._site_manager = LocalSiteManager(self.output_dir)
        self._orchestrator = Orchestrator(self.devices, self.tasks, scheduler, self.output_dir)

        self._runner: TraceRunner = None
        self._runner_thread = None

    def deploy(self):
        """Plan + deploy all devices. Blocks until hardware is ready."""
        print(f"\n{'='*60}\nDEPLOY\n{'='*60}")

        trace, avg_workload, _ = generate_trace(
            self.trace_type, self.req_rate, self.duration, self.tasks, self.seed)
        for task_name in self.tasks:
            if task_name in avg_workload:
                self.tasks[task_name]['peak_workload'] = avg_workload[task_name]

        plan = self._orchestrator.run_deployment_plan(
            self.devices, self.tasks, scheduler_name=self.scheduler, output_dir=self.output_dir)

        for site in plan.get("sites", []):
            for dep in site.get("deployments", []):
                dep.setdefault("scheduler_policy", _fmaas_to_device_scheduler(self.scheduler))
                dep.setdefault("max_batch_size", self.max_batch_size)
                dep.setdefault("max_batch_wait_ms", self.max_batch_wait_ms)
                dep.setdefault("isolation_mode", self.isolation_mode)

        self._site_manager.deploy(plan, self.output_dir)
        self._orchestrator.increment_requests_generated(len(trace))

        self._runner = TraceRunner(
            self._site_manager.live_plan,
            trace,
            self.output_dir,
            pretrace_warmup_secs=self._pretrace_warmup_secs,
        )
        print(f"[LocalExperiment] Deploy complete. {len(trace)} requests ready.")

    def run(self):
        """Warmup gRPC connections then start trace dispatch in background (non-blocking)."""
        print(f"\n{'='*60}\nRUN INFERENCE\n{'='*60}")

        runner = self

        def _run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(runner._run_async())
            loop.close()

        self._runner_thread = threading.Thread(target=_run_in_thread, daemon=True)
        self._runner_thread.start()
        self._runtime_start = time.time()

    async def _run_async(self):
        await self._runner.warmup()
        start_epoch = time.time() + self._warmup_gap
        print(f"[LocalExperiment] Warmup done. Trace starts in {self._warmup_gap}s...")
        await asyncio.sleep(self._warmup_gap)
        await self._runner.run(start_epoch=start_epoch)

    def add_task(self, task_name: str, workload: float, elapsed: float,
                 task_type: str = "classification"):
        print(f"\n[LocalExperiment] add_task: {task_name} @ {workload} req/s (t={elapsed}s)")

        _, tasks = load_config(self.exp_type)
        known = tasks.get(task_name, {})
        task_spec = {
            'type': task_type,
            'peak_workload': workload,
            'latency': known.get('latency', 1000),
            'metric': known.get('metric', 'accuracy' if task_type == 'classification' else 'mae'),
            'value': known.get('value', 0.9 if task_type == 'classification' else 0.5),
        }

        scheduler_mode = self.scheduler.lower() in ("fmaas_share", "clipper-ha", "m4-ha")
        ok, msg, diffs, _ = self._orchestrator.handle_add_task(task_name, task_spec, scheduler_mode)
        if not ok:
            raise RuntimeError(f"handle_add_task failed: {msg}")

        for diff in diffs:
            if diff.full_deployment is not None:
                diff.full_deployment.setdefault("scheduler_policy", _fmaas_to_device_scheduler(self.scheduler))
                diff.full_deployment.setdefault("max_batch_size", self.max_batch_size)
                diff.full_deployment.setdefault("max_batch_wait_ms", self.max_batch_wait_ms)
                diff.full_deployment.setdefault("isolation_mode", self.isolation_mode)
            self._site_manager.apply_diff(diff)

        self._site_manager.live_plan.clear()
        self._site_manager.live_plan.update(self._orchestrator.plan)
        self._runner.invalidate_plan_cache()

        remaining = max(0, self.duration - elapsed)
        if remaining > 0:
            req_id_offset = self._orchestrator.get_total_requests_generated()
            trace, _, _ = generate_trace(
                self.trace_type, workload, remaining,
                {task_name: task_spec}, self.seed, req_id_offset)
            for r in trace:
                r.req_time += elapsed
            self._runner.add_requests(trace)
            self._orchestrator.increment_requests_generated(len(trace))
            print(f"[LocalExperiment] Added {len(trace)} requests for {task_name}.")

    def add_workload(self, task_name: str, delta: float, elapsed: float,
                     task_type: str = "classification"):
        print(f"\n[LocalExperiment] add_workload: {task_name} delta={delta:+.1f} (t={elapsed}s)")

        remaining = max(0, self.duration - elapsed)
        if remaining <= 0 or delta <= 0:
            print(f"  skipped (remaining={remaining}s, delta={delta})")
            return

        _, tasks = load_config(self.exp_type)
        known = tasks.get(task_name, {})
        task_spec = {
            'type': task_type,
            'peak_workload': delta,
            'latency': known.get('latency', 1000),
            'metric': known.get('metric', 'accuracy' if task_type == 'classification' else 'mae'),
            'value': known.get('value', 0.9 if task_type == 'classification' else 0.5),
        }

        scheduler_mode = self.scheduler.lower() in ("fmaas_share", "clipper-ha", "m4-ha")
        ok, msg, diffs, _ = self._orchestrator.handle_add_task(task_name, task_spec, scheduler_mode)
        if not ok:
            raise RuntimeError(f"handle_add_task failed: {msg}")

        for diff in diffs:
            if diff.full_deployment is not None:
                diff.full_deployment.setdefault("scheduler_policy", _fmaas_to_device_scheduler(self.scheduler))
                diff.full_deployment.setdefault("max_batch_size", self.max_batch_size)
                diff.full_deployment.setdefault("max_batch_wait_ms", self.max_batch_wait_ms)
                diff.full_deployment.setdefault("isolation_mode", self.isolation_mode)
            self._site_manager.apply_diff(diff)

        self._site_manager.live_plan.clear()
        self._site_manager.live_plan.update(self._orchestrator.plan)
        self._runner.invalidate_plan_cache()

        req_id_offset = self._orchestrator.get_total_requests_generated()
        trace, _, _ = generate_trace(
            self.trace_type, delta, remaining,
            {task_name: task_spec}, self.seed, req_id_offset)
        for r in trace:
            r.req_time += elapsed
        self._runner.add_requests(trace)
        self._orchestrator.increment_requests_generated(len(trace))
        print(f"[LocalExperiment] Added {len(trace)} spike requests for {task_name}.")

    def wait(self, timeout: float = 1200):
        print(f"\n{'='*60}\nWAITING FOR COMPLETION\n{'='*60}")
        if self._runner_thread:
            self._runner_thread.join(timeout=timeout)
        self._runner.save_results()
        print(f"[LocalExperiment] Done. Results in {self.output_dir}")

    def cleanup(self):
        print(f"\n{'='*60}\nCLEANUP\n{'='*60}")
        self._site_manager.cleanup()


# ============================================================================
# API Endpoints (MQTT mode)
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "FMaaS Orchestrator",
        "status": "running",
        "deployed": state.deployed,
        "inference_running": state.running,
    }


@app.get("/status")
async def get_status() -> StatusResponse:
    if not state.orchestrator:
        return StatusResponse(deployed=False, running=False,
                              total_requests_generated=0, tasks=[], plan=None)
    return StatusResponse(
        deployed=state.deployed,
        running=state.running,
        total_requests_generated=state.orchestrator.get_total_requests_generated(),
        tasks=list(state.orchestrator.tasks.keys()),
        plan=state.orchestrator.plan,
    )


@app.post("/deploy")
async def deploy(req: DeployRequest):
    print(f"\n{'='*80}\nDEPLOYMENT\n{'='*80}\n")
    try:
        devices, tasks = load_config(req.exp_type)

        trace, avg_workload, _ = generate_trace(
            req.trace, req.req_rate, req.duration, tasks, req.seed)
        for task_name in tasks:
            if task_name in avg_workload:
                tasks[task_name]['peak_workload'] = avg_workload[task_name]
        print(f"[Server] Generated {len(trace)} requests")

        output_dir = os.path.abspath(os.path.join(req.exp_dir, req.scheduler, str(req.req_rate)))
        os.makedirs(output_dir, exist_ok=True)

        state.orchestrator = Orchestrator(devices, tasks, req.scheduler, output_dir)

        from site_manager.mqtt import MQTTSiteManager
        state.site_manager = MQTTSiteManager()

        plan = state.orchestrator.run_deployment_plan(
            devices, tasks, scheduler_name=req.scheduler, output_dir=output_dir)

        routed_trace = route_trace(trace, plan, req.seed)
        state.orchestrator.increment_requests_generated(len(routed_trace))

        state.site_manager.set_plan(plan)
        state.site_manager.deploy(plan, routed_trace, output_dir=output_dir)

        state.config = {
            'exp_type': req.exp_type, 'trace': req.trace,
            'req_rate': req.req_rate, 'duration': req.duration,
            'seed': req.seed, 'scheduler': req.scheduler, 'exp_dir': req.exp_dir,
        }
        state.deployed = True
        print(f"\n[Server] Deployment complete! Results in {output_dir}")

        return {"status": "deployed", "total_requests": len(trace),
                "output_dir": output_dir, "plan": plan}

    except Exception as e:
        print(f"[Server] Deployment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run")
async def run_inference(background_tasks: BackgroundTasks):
    print(f"\n{'='*80}\nRUNTIME\n{'='*80}\n")
    if not state.deployed:
        raise HTTPException(status_code=400, detail="Must deploy before running inference")
    if state.running:
        raise HTTPException(status_code=400, detail="Inference already running")

    async def run_in_background():
        try:
            state.running = True
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                await loop.run_in_executor(pool, state.site_manager.run_inference)
            print(f"\n[Server] Inference triggered (continuous mode - still running on site managers)")
        except Exception as e:
            state.running = False
            print(f"[Server] Inference failed: {e}")

    background_tasks.add_task(run_in_background)
    return {"status": "started", "message": "Inference running in background"}


@app.post("/wait")
async def wait_for_completion():
    if not state.running:
        return {"status": "not_running", "message": "Inference not currently running"}
    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, state.site_manager.wait_for_completion)
        state.running = False
        return {"status": "completed", "message": "Inference completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-task")
async def add_task(req: AddTaskRequest):
    if not state.deployed:
        raise HTTPException(status_code=400, detail="Must deploy before adding tasks")
    if not state.config:
        raise HTTPException(status_code=500, detail="Missing deployment config")

    try:
        print(f"\n[Server] Adding task: {req.task_name} (workload={req.task_workload}, elapsed={req.elapsed_time}s)")

        devices, tasks = load_config(state.config['exp_type'])
        known = tasks.get(req.task_name, {})
        task_spec = {
            'type': req.task_type,
            'peak_workload': req.task_workload,
            'latency': known.get('latency', 1000),
            'metric': known.get('metric', 'accuracy' if req.task_type == 'classification' else 'mae'),
            'value': known.get('value', 0.9 if req.task_type == 'classification' else 0.5),
        }

        scheduler_name = state.config['scheduler'].lower()
        scheduler_mode = scheduler_name in ('fmaas_share', 'clipper-ha', 'm4-ha')

        success, message, diffs, incremental_plan = state.orchestrator.handle_add_task(
            req.task_name, task_spec, scheduler_mode)
        if not success:
            raise HTTPException(status_code=400, detail=message)

        print(f"  ✓ {message}")

        remaining_duration = max(0, state.config['duration'] - req.elapsed_time)
        n_new = 0
        if remaining_duration > 0:
            req_id_offset = state.orchestrator.get_total_requests_generated()
            trace, _, _ = generate_trace(
                state.config['trace'], task_spec['peak_workload'], remaining_duration,
                {req.task_name: task_spec}, state.config['seed'], req_id_offset)

            if req.elapsed_time > 0:
                for r in trace:
                    r.req_time += req.elapsed_time

            state.site_manager.set_plan(state.orchestrator.plan)

            if incremental_plan:
                _send_incremental_batches(
                    state.orchestrator, state.site_manager, trace,
                    incremental_plan, diffs, state.config['seed'], req.elapsed_time)
            else:
                routed_trace = route_trace(trace, state.orchestrator.plan, state.config['seed'])
                state.site_manager.send_requests(routed_trace)
                state.orchestrator.increment_requests_generated(len(routed_trace))
            n_new = len(trace)

        return {
            "status": "task_added", "task_name": req.task_name,
            "new_requests_generated": n_new,
            "total_requests": state.orchestrator.get_total_requests_generated(),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Server] Add task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-workload")
async def add_workload(req: AddWorkloadRequest):
    if not state.deployed:
        raise HTTPException(status_code=400, detail="Must deploy before changing workload")
    if not state.config:
        raise HTTPException(status_code=500, detail="Missing deployment config")

    try:
        print(f"\n[Server] Workload change: {req.task_name} delta={req.task_workload:+.1f} req/s "
              f"(elapsed={req.elapsed_time}s)")

        remaining_duration = max(0, state.config['duration'] - req.elapsed_time)
        if remaining_duration <= 0:
            return {"status": "skipped", "reason": "no remaining duration"}

        if req.task_workload > 0:
            _, tasks = load_config(state.config['exp_type'])
            known = tasks.get(req.task_name, {})
            task_spec = {
                'type': req.task_type,
                'peak_workload': req.task_workload,
                'latency': known.get('latency', 1000),
                'metric': known.get('metric', 'accuracy' if req.task_type == 'classification' else 'mae'),
                'value': known.get('value', 0.9 if req.task_type == 'classification' else 0.5),
            }

            req_id_offset = state.orchestrator.get_total_requests_generated()
            trace, _, _ = generate_trace(
                state.config['trace'], req.task_workload, remaining_duration,
                {req.task_name: task_spec}, state.config['seed'], req_id_offset)
            for r in trace:
                r.req_time += req.elapsed_time

            scheduler_mode = state.config['scheduler'].lower() in ('fmaas_share', 'clipper-ha', 'm4-ha')
            _, _, diffs, incremental_plan = state.orchestrator.handle_add_task(
                req.task_name, task_spec, scheduler_mode)

            state.site_manager.set_plan(state.orchestrator.plan)

            if incremental_plan:
                _send_incremental_batches(
                    state.orchestrator, state.site_manager, trace,
                    incremental_plan, diffs, state.config['seed'], req.elapsed_time)
            else:
                routed_trace = route_trace(trace, state.orchestrator.plan, state.config['seed'])
                state.site_manager.send_requests(routed_trace)
                state.orchestrator.increment_requests_generated(len(routed_trace))
            n_new = len(trace)
            print(f"  ✓ Workload spike sent!")
        else:
            print(f"  ✓ Workload drop recorded (no new requests sent).")
            n_new = 0

        return {
            "status": "workload_updated", "task_name": req.task_name,
            "delta_req_per_sec": req.task_workload, "new_requests_generated": n_new,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Server] add-workload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cleanup")
async def cleanup():
    print(f"\n{'='*80}\nCLEANUP\n{'='*80}\n")
    if not state.deployed:
        raise HTTPException(status_code=400, detail="Nothing to cleanup")
    try:
        state.site_manager.cleanup(state.orchestrator.plan)
        state.reset()
        print(f"\n[Server] Cleanup complete!")
        return {"status": "cleaned"}
    except Exception as e:
        print(f"[Server] Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main — mode selection
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FMaaS Orchestrator")
    parser.add_argument("--mode", choices=["local", "mqtt"], default="local",
                        help="local: single-process no MQTT; mqtt: geo-distributed via broker")

    # MQTT mode args
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")

    # Local mode args
    parser.add_argument("--exp-type", default="SystemInAction")
    parser.add_argument("--scheduler", default="fmaas_share")
    parser.add_argument("--req-rate", type=int, default=10)
    parser.add_argument("--duration", type=int, default=480)
    parser.add_argument("--trace", default="deterministic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-dir", default="./experiments/SystemInAction/results")
    parser.add_argument("--max-batch-size", type=int, default=5)
    parser.add_argument("--max-batch-wait-ms", type=float, default=0.0)
    parser.add_argument("--isolation-mode", default="shared",
                        choices=["shared", "process", "none"])
    parser.add_argument("--warmup-gap", type=float, default=2.0)
    parser.add_argument("--pretrace-warmup-secs", type=float, default=20.0)
    args = parser.parse_args()

    if args.mode == "local":
        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║               FMaaS Orchestrator (Local mode)                 ║
╚═══════════════════════════════════════════════════════════════╝
""")
        exp = LocalExperiment(
            exp_type=args.exp_type,
            scheduler=args.scheduler,
            req_rate=args.req_rate,
            duration=args.duration,
            trace_type=args.trace,
            seed=args.seed,
            exp_dir=args.exp_dir,
            max_batch_size=args.max_batch_size,
            max_batch_wait_ms=args.max_batch_wait_ms,
            isolation_mode=args.isolation_mode,
            warmup_gap=args.warmup_gap,
            pretrace_warmup_secs=args.pretrace_warmup_secs,
        )

        exp.deploy()
        exp.run()

        # ── Runtime event timeline ──────────────────────────────────────
        # Adjust these to match your experiment.
        start = time.time()

        def elapsed():
            return time.time() - start

        # Example: add gestureclass at t=60s
        # time.sleep(max(0, 60 - elapsed()))
        # exp.add_task("gestureclass", workload=8.0, elapsed=60.0)

        # Example: add sysbp at t=300s
        # time.sleep(max(0, 300 - elapsed()))
        # exp.add_task("sysbp", workload=8.0, elapsed=300.0)
        # ────────────────────────────────────────────────────────────────

        exp.wait()
        exp.cleanup()

    else:
        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║               FMaaS Orchestrator Server (MQTT mode)           ║
╚═══════════════════════════════════════════════════════════════╝

Starting server on http://{args.host}:{args.port}

Available endpoints:
  GET  /              - Health check
  GET  /status        - Get orchestrator status
  POST /deploy        - Deploy initial tasks
  POST /run           - Run inference
  POST /wait          - Wait for inference completion
  POST /add-task      - Add task at runtime
  POST /add-workload  - Change workload at runtime
  POST /cleanup       - Cleanup devices

Press Ctrl+C to stop
""")
        uvicorn.run("orchestrator.server:app", host=args.host, port=args.port,
                    reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
