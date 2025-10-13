# orchestrator/main.py
import asyncio, uvicorn, argparse
from fastapi import FastAPI
from orchestrator.manager_registry import site_registry

from orchestrator.config import ORCHESTRATOR_PORT

app = FastAPI()

@app.post("/heartbeat")
async def heartbeat(data: dict):
    site_registry.update_heartbeat(data.get("site_id"))
    return {"ok": True}

@app.get("/sites")
def get_sites():
    return site_registry.get_sites()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy-only", action="store_true", help="Only run deployment phase")
    parser.add_argument("--run-exp", action="store_true", help="runtime benchmark")
    args = parser.parse_args()

    if args.deploy_only:
        from orchestrator.deployment_planner import deploy_to_sites
        asyncio.run(deploy_to_sites())
    elif args.run_exp:
        from orchestrator.runtime_executor import run_experiment
        config = (1, 1, 6, 1, 120)
        asyncio.run(run_experiment(config))
    else:
        uvicorn.run("orchestrator.main:app", host="0.0.0.0", port=ORCHESTRATOR_PORT)
