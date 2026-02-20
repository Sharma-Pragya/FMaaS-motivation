#!/usr/bin/env python3
"""FastAPI server for stateful orchestrator control.

This server maintains a single long-running Orchestrator instance,
eliminating the need for file-based state persistence.

Usage:
    python -m orchestrator.server --port 8080 --exp-type runtime

    # Then control via HTTP:
    curl -X POST http://localhost:8080/deploy
    curl -X POST http://localhost:8080/run
    curl -X POST http://localhost:8080/add-task -d '{"task_name": "gestureclass", ...}'
    curl -X POST http://localhost:8080/add-workload -d '{"task_name": "heartrate", "task_workload": 20}'
    curl -X POST http://localhost:8080/cleanup
"""

import argparse
import os
import sys
import time
import uvicorn
import asyncio
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

from orchestrator.main import Orchestrator
from orchestrator.router import route_trace


# ============================================================================
# Pydantic Models (Request/Response schemas)
# ============================================================================

class DeployRequest(BaseModel):
    """Request to deploy initial tasks."""
    exp_type: str = "runtime"
    trace: str = "lmsyschat"
    req_rate: int = 10
    duration: int = 60
    seed: int = 42
    scheduler: str = "fmaas_share"
    exp_dir: str = "./experiments/runtime/results"


class RunRequest(BaseModel):
    """Request to start inference."""
    pass  # No parameters needed - uses state from deploy


class AddTaskRequest(BaseModel):
    """Request to add a new task at runtime."""
    task_name: str
    task_type: str = "classification"
    task_workload: float = 8.0
    elapsed_time: float = 0.0  # How much time has elapsed since experiment start


class AddWorkloadRequest(BaseModel):
    """Request to change workload for an existing task at runtime.

    task_workload is a *delta* (positive = spike, negative = drop).
    The orchestrator accumulates this on top of the current rate and
    generates additional request traffic for the remaining duration.
    """
    task_name: str
    task_type: str = "classification"
    task_workload: float = 10.0   # delta req/s (can be negative)
    elapsed_time: float = 0.0     # seconds since experiment start


class CleanupRequest(BaseModel):
    """Request to cleanup devices."""
    pass  # No parameters needed


class StatusResponse(BaseModel):
    """Current orchestrator status."""
    deployed: bool
    running: bool
    total_requests_generated: int
    tasks: list
    plan: Optional[Dict[str, Any]]


# ============================================================================
# Global State
# ============================================================================

class OrchestratorState:
    """Singleton state for the orchestrator."""
    def __init__(self):
        self.orchestrator: Optional[Orchestrator] = None
        self.deployed = False
        self.running = False
        self.config = None  # Store deploy config for add-task

    def reset(self):
        """Reset state (used for cleanup)."""
        self.deployed = False
        self.running = False


# Global state instance
state = OrchestratorState()
app = FastAPI(title="FMaaS Orchestrator", version="1.0.0")


# ============================================================================
# Helper Functions
# ============================================================================

def load_config(exp_type: str):
    """Load task and device configuration for the experiment."""
    if exp_type == "baselines":
        from experiments.baselines.user_config import devices, tasks
    elif exp_type == "batching":
        from experiments.batching.user_config import devices, tasks
    elif exp_type == "runtime":
        from experiments.runtime.user_config import devices, tasks
    else:
        raise ValueError(f"Unknown exp_type: {exp_type}")
    return devices, tasks


def generate_trace(trace_type: str, req_rate: int, duration: int, tasks: dict, seed: int, req_id_offset: int = 0):
    """Generate workload trace and compute per-task workloads."""
    all_task_names = sorted(tasks.keys())
    routed_tasks = [(t, None, None, None) for t in all_task_names]

    if trace_type == 'lmsyschat':
        from traces.lmsyschat import generate_requests
        trace, avg_workload, peak_workload = generate_requests(
            req_rate, duration, routed_tasks, seed, req_id_offset
        )
    elif trace_type == 'gamma':
        from traces.gamma import generate_requests
        num_tasks, alpha, cv = len(all_task_names), 1, 1
        trace, avg_workload, peak_workload = generate_requests(
            num_tasks, alpha, req_rate, cv, duration, routed_tasks, seed, req_id_offset
        )
    elif trace_type == 'chatbotarena':
        from traces.chatbotarena import generate_requests
        trace, avg_workload, peak_workload = generate_requests(
            req_rate, duration, routed_tasks, seed, req_id_offset
        )
    else:
        raise ValueError(f"Unknown trace: {trace_type}")

    return trace, avg_workload, peak_workload


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "FMaaS Orchestrator",
        "status": "running",
        "deployed": state.deployed,
        "inference_running": state.running
    }


@app.get("/status")
async def get_status() -> StatusResponse:
    """Get current orchestrator status."""
    if not state.orchestrator:
        return StatusResponse(
            deployed=False,
            running=False,
            total_requests_generated=0,
            tasks=[],
            plan=None
        )

    return StatusResponse(
        deployed=state.deployed,
        running=state.running,
        total_requests_generated=state.orchestrator._total_requests_generated,
        tasks=list(state.orchestrator.tasks.keys()),
        plan=state.orchestrator.plan
    )


@app.post("/deploy")
async def deploy(req: DeployRequest):
    """Deploy initial set of tasks."""
    print(f"\n{'='*80}")
    print("DEPLOYMENT")
    print(f"{'='*80}\n")

    try:
        # Load config
        print(f"[Server] Loading config: {req.exp_type}")
        devices, tasks = load_config(req.exp_type)

        # Generate trace
        print(f"[Server] Generating trace: {req.trace} (rate={req.req_rate}, duration={req.duration}s)")
        trace, avg_workload, peak_workload = generate_trace(
            req.trace, req.req_rate, req.duration, tasks, req.seed
        )

        # Update task workloads from trace
        for task_name in tasks:
            if task_name in avg_workload:
                tasks[task_name]['peak_workload'] = avg_workload[task_name]

        print(f"[Server] Generated {len(trace)} requests")

        # Create output directory
        output_dir = os.path.abspath(os.path.join(req.exp_dir, req.scheduler, str(req.req_rate)))
        os.makedirs(output_dir, exist_ok=True)

        # Create orchestrator
        state.orchestrator = Orchestrator(devices, tasks, req.scheduler, output_dir)

        # Run deployment plan
        print(f"[Server] Planning deployment...")
        plan = state.orchestrator.run_deployment_plan(devices, tasks, scheduler_name=req.scheduler, output_dir=output_dir)

        # Route trace
        print(f"[Server] Routing trace...")
        routed_trace = route_trace(trace, plan, req.seed)

        # Deploy
        state.orchestrator.initial_deployment(plan, routed_trace, output_dir=output_dir)

        # Save config for add-task
        state.config = {
            'exp_type': req.exp_type,
            'trace': req.trace,
            'req_rate': req.req_rate,
            'duration': req.duration,
            'seed': req.seed,
            'scheduler': req.scheduler,
            'exp_dir': req.exp_dir,
        }

        state.deployed = True
        print(f"\n[Server] Deployment complete! Results in {output_dir}")

        return {
            "status": "deployed",
            "total_requests": len(trace),
            "output_dir": output_dir,
            "plan": plan
        }

    except Exception as e:
        print(f"[Server] Deployment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run")
async def run_inference(background_tasks: BackgroundTasks):
    """Start inference in background (non-blocking)."""
    print(f"\n{'='*80}")
    print("RUNTIME")
    print(f"{'='*80}\n")

    if not state.deployed:
        raise HTTPException(status_code=400, detail="Must deploy before running inference")

    if state.running:
        raise HTTPException(status_code=400, detail="Inference already running")

    # Run inference in a separate thread to avoid blocking FastAPI event loop
    async def run_in_background():
        try:
            state.running = True
            print(f"[Server] Starting inference in background thread...")

            # Run blocking orchestrator call in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                await loop.run_in_executor(pool, state.orchestrator.run_inference_requests)

            # Don't set state.running = False here in continuous mode
            # The inference keeps running on site managers until cleanup is called
            print(f"\n[Server] Inference triggered (continuous mode - still running on site managers)")

        except Exception as e:
            state.running = False
            print(f"[Server] Inference failed: {e}")

    # Schedule inference to run in background
    background_tasks.add_task(run_in_background)

    return {
        "status": "started",
        "message": "Inference running in background"
    }


@app.post("/wait")
async def wait_for_completion():
    """Wait for inference to complete (blocking call)."""
    if not state.running:
        return {"status": "not_running", "message": "Inference not currently running"}

    try:
        # Run the wait call in thread pool to avoid blocking FastAPI event loop
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, state.orchestrator.wait_for_inference_completion)

        state.running = False
        return {"status": "completed", "message": "Inference completed successfully"}

    except Exception as e:
        print(f"[Server] Wait failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-task")
async def add_task(req: AddTaskRequest):
    """Add a new task at runtime."""
    if not state.deployed:
        raise HTTPException(status_code=400, detail="Must deploy before adding tasks")

    if not state.config:
        raise HTTPException(status_code=500, detail="Missing deployment config")

    try:
        print(f"\n[Server] Adding task: {req.task_name}")
        print(f"         Type: {req.task_type}, Workload: {req.task_workload}")
        print(f"         Elapsed time: {req.elapsed_time}s")

        # Build task spec
        task_spec = {
            'type': req.task_type,
            'peak_workload': req.task_workload,
            'latency': 1000,
            'metric': 'accuracy' if req.task_type == 'classification' else 'mae',
            'value': 0.9 if req.task_type == 'classification' else 0.5,
        }

        # Determine scheduler mode based on scheduler name
        scheduler_name = state.config['scheduler'].lower()
        if scheduler_name == 'fmaas_share':
            scheduler_mode = True  # share_mode=True
        elif scheduler_name == 'fmaas':
            scheduler_mode = False  # share_mode=False
        elif scheduler_name == 'clipper-ha':
            scheduler_mode = True  # accuracy_mode=True
        elif scheduler_name == 'clipper-ht':
            scheduler_mode = False  # accuracy_mode=False
        elif scheduler_name == 'm4-ha':
            scheduler_mode = True  # accuracy_mode=True
        elif scheduler_name == 'm4-ht':
            scheduler_mode = False  # accuracy_mode=False
        else:
            scheduler_mode = False  # default

        # Add task to deployment
        success, message, actions = state.orchestrator.handle_add_task(
            req.task_name, task_spec, scheduler_mode
        )

        if not success:
            raise HTTPException(status_code=400, detail=message)

        print(f"  ✓ {message}")
        if actions:
            for action in actions:
                print(f"    {action['action']}: {action['server_name']} @ {action['site_manager']}")

        # Generate and send workload
        remaining_duration = max(0, state.config['duration'] - req.elapsed_time)

        if remaining_duration > 0:
            print(f"\n[Server] Generating workload for new task (duration={remaining_duration}s, offset={req.elapsed_time}s)...")

            # Get current request count to avoid req_id collision
            req_id_offset = state.orchestrator.get_total_requests_generated()
            print(f"[Server] Using req_id_offset={req_id_offset}")

            # Generate trace
            req_rate = int(task_spec.get('peak_workload', 10))
            trace, _, _ = generate_trace(
                state.config['trace'],
                req_rate,
                remaining_duration,
                {req.task_name: task_spec},
                state.config['seed'],
                req_id_offset
            )

            # Apply time offset
            if req.elapsed_time > 0:
                for r in trace:
                    r.req_time += req.elapsed_time
                print(f"[Server] Applied time_offset={req.elapsed_time}s to {len(trace)} requests")

            print(f"[Server] Generated {len(trace)} requests (req_ids {req_id_offset} to {req_id_offset + len(trace) - 1})")
            print(f"[Server] Time range: {req.elapsed_time:.1f}s to {req.elapsed_time + remaining_duration:.1f}s")

            # Route and send
            routed_trace = route_trace(trace, state.orchestrator.plan, state.config['seed'])

            print(f"[Server] Sending new workload to site managers...")
            state.orchestrator.send_new_requests(routed_trace)
            print(f"  ✓ Workload sent!")

        return {
            "status": "task_added",
            "task_name": req.task_name,
            "actions": actions,
            "new_requests_generated": len(trace) if remaining_duration > 0 else 0,
            "total_requests": state.orchestrator._total_requests_generated
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Server] Add task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-workload")
async def add_workload(req: AddWorkloadRequest):
    """Change workload for an existing task at runtime (spike or drop).

    Generates additional request traffic proportional to the delta for the
    remaining experiment duration.  No new infrastructure is deployed — the
    planner just accumulates the rate change on the existing task.
    """
    if not state.deployed:
        raise HTTPException(status_code=400, detail="Must deploy before changing workload")
    if not state.config:
        raise HTTPException(status_code=500, detail="Missing deployment config")

    try:
        print(f"\n[Server] Workload change: {req.task_name} delta={req.task_workload:+.1f} req/s")
        print(f"         Elapsed time: {req.elapsed_time}s")

        remaining_duration = max(0, state.config['duration'] - req.elapsed_time)
        if remaining_duration <= 0:
            return {"status": "skipped", "reason": "no remaining duration"}

        # Only generate additional traffic if delta is positive (spike).
        # For a drop we still record the workload change but don't send
        # negative requests — the existing queued requests simply thin out.
        if req.task_workload > 0:
            task_spec = {
                'type': req.task_type,
                'peak_workload': req.task_workload,  # delta only
                'latency': 1000,
                'metric': 'accuracy' if req.task_type == 'classification' else 'mae',
                'value': 0.9 if req.task_type == 'classification' else 0.5,
            }

            req_id_offset = state.orchestrator.get_total_requests_generated()
            trace, _, _ = generate_trace(
                state.config['trace'],
                int(req.task_workload),
                remaining_duration,
                {req.task_name: task_spec},
                state.config['seed'],
                req_id_offset
            )
            for r in trace:
                r.req_time += req.elapsed_time

            print(f"[Server] Generated {len(trace)} additional requests for {req.task_name} "
                  f"(t={req.elapsed_time}s → {req.elapsed_time + remaining_duration:.0f}s)")

            routed_trace = route_trace(trace, state.orchestrator.plan, state.config['seed'])
            state.orchestrator.send_new_requests(routed_trace)
            print(f"  ✓ Workload spike sent!")
            n_new = len(trace)
        else:
            # Drop: just log it; no new requests generated
            print(f"  ✓ Workload drop recorded (no new requests sent).")
            n_new = 0

        # Update internal workload accounting via handle_add_task (accumulates rate)
        task_spec_for_state = {
            'type': req.task_type,
            'peak_workload': req.task_workload,
            'latency': 1000,
            'metric': 'accuracy' if req.task_type == 'classification' else 'mae',
            'value': 0.9 if req.task_type == 'classification' else 0.5,
        }
        scheduler_name = state.config['scheduler'].lower()
        scheduler_mode = scheduler_name == 'fmaas_share'
        state.orchestrator.handle_add_task(req.task_name, task_spec_for_state, scheduler_mode)

        return {
            "status": "workload_updated",
            "task_name": req.task_name,
            "delta_req_per_sec": req.task_workload,
            "new_requests_generated": n_new,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Server] add-workload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cleanup")
async def cleanup():
    """Cleanup all devices."""
    print(f"\n{'='*80}")
    print("CLEANUP")
    print(f"{'='*80}\n")

    if not state.deployed:
        raise HTTPException(status_code=400, detail="Nothing to cleanup")

    try:
        state.orchestrator.cleanup_only(state.orchestrator.plan)
        state.reset()

        print(f"\n[Server] Cleanup complete!")

        return {"status": "cleaned"}

    except Exception as e:
        print(f"[Server] Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FMaaS Orchestrator Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║               FMaaS Orchestrator Server                       ║
╚═══════════════════════════════════════════════════════════════╝

Starting server on http://{args.host}:{args.port}

Available endpoints:
  GET  /              - Health check
  GET  /status        - Get orchestrator status
  POST /deploy        - Deploy initial tasks
  POST /run           - Run inference
  POST /add-task      - Add task at runtime
  POST /cleanup       - Cleanup devices

Press Ctrl+C to stop
""")

    uvicorn.run(
        "orchestrator.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
