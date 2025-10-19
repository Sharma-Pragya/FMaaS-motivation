# orchestrator/runtime_executor.py
import asyncio, json, time, numpy as np, base64
from typing import List, Tuple
import aiohttp
from traces.gamma import generate_requests
from orchestrator.deployment_planner import deploy_to_sites
from orchestrator.router import get_route
from orchestrator.config import RESULTS_DIR, DEPLOYMENT_PLAN_PATH

REQUEST_LATENCY = []

def encode_raw(arr) -> dict:
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
    }

async def send_request(i, site_manager, payload):
    st = time.time()
    try:
        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(site_manager, json=payload) as resp:
                await resp.json()
                et = time.time()
    except Exception as e:
        et = time.time()
        print(f"Request {i} failed: {e}")
    REQUEST_LATENCY.append((i, et - st))
    return (i, et - st)

async def benchmark(input_requests: List[Tuple[str, str, int, int]]):
    start = time.time()
    tasks = []
    for req in input_requests:
        await asyncio.sleep(start + req.req_time - time.time())
        payload = {
            "req_id": req.req_id,
            "task": req.task,
            "device": req.device,
        }
        t = asyncio.create_task(send_request(req.req_id, req.site_manager, payload))
        tasks.append(t)
    return await asyncio.gather(*tasks)

def get_stats(per_req_latency, benchmark_time, config):
    throughput = len(per_req_latency) / benchmark_time
    avg_latency = np.mean([lat for _, lat in per_req_latency])
    result = {
        "total_time": benchmark_time,
        "throughput": throughput,
        "avg_latency": avg_latency,
    }
    return {"config": config, "result": result}

def to_config_dict(config_tuple):
    num_tasks, alpha, req_rate, cv, duration = config_tuple
    return {"num_tasks": num_tasks, "alpha": alpha, "req_rate": req_rate, "cv": cv, "duration": duration}

async def run_experiment(config, seed=42):
    """Main orchestration entrypoint for deployment + runtime."""
    num_tasks, alpha, req_rate, cv, duration = config

    # print("[Orchestrator] Starting deployment phase...")
    # await deploy_to_sites(DEPLOYMENT_PLAN_PATH)

    # print("[Orchestrator] Deployment complete. Starting runtime benchmark...")
    tasks = []
    for task_name in ["vqa"]:
        site_manager, device = get_route(task_name)
        tasks.append((task_name, site_manager, device))

    trace_requests = generate_requests(num_tasks, alpha, req_rate, cv, duration, tasks, seed)
    benchmark_start = time.time()
    per_req_latency = await benchmark(trace_requests)
    benchmark_end = time.time()

    stats = get_stats(per_req_latency, benchmark_end - benchmark_start, to_config_dict(config))
    with open(f"{RESULTS_DIR}/results.jsonl", "a") as f:
        f.write(json.dumps(stats) + "\n")

    print(f"[Orchestrator] Benchmark complete. Avg latency: {stats['result']['avg_latency']:.3f}s")
