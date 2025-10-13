# orchestrator/deployment_planner.py
import json, aiohttp, asyncio
from orchestrator.manager_registry import site_registry
from orchestrator.config import DEPLOYMENT_PLAN_PATH

async def deploy_to_sites(plan_path="deployment_plan.json"):
    with open(plan_path, "r") as f:
        plan = json.load(f)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for site in plan["sites"]:
            url = f"{site['site_manager_url']}/deploy"
            tasks.append(session.post(url, json=site["deployments"]))
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    print("[Orchestrator] Deployment complete.")

